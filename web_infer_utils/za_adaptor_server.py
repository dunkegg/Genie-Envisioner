import asyncio
import http
import logging
import os
import time
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import websockets.asyncio.server as _server
import websockets.frames

from isaac_test_tdmpc2_external_z import (
    ExternalZTDMPCTestController,
    PlannerMapHistory,
    action_for_env,
    extract_velocity_model_norm,
    load_tdmpc2_agent,
    load_tdmpc_cfg,
)
from web_infer_utils.ZaAdaptorActor import ZaAdaptorActor
from web_infer_utils.openpi_client import msgpack_numpy


logger = logging.getLogger(__name__)


class OnlineTDMPCReplay:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.storage)

    def add(self, z, action, next_z, reward=None, done=False):
        self.storage.append(
            {
                "z": np.asarray(z, dtype=np.float32).reshape(-1).copy(),
                "action": np.asarray(action, dtype=np.float32).reshape(-1).copy(),
                "next_z": np.asarray(next_z, dtype=np.float32).reshape(-1).copy(),
                "reward": None if reward is None else float(np.asarray(reward, dtype=np.float32).reshape(-1)[0]),
                "done": bool(done),
            }
        )

    def sample(self, batch_size):
        if len(self.storage) < batch_size:
            raise ValueError(f"Need at least {batch_size} transitions, got {len(self.storage)}.")
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        items = [self.storage[int(i)] for i in indices]
        z = np.stack([item["z"] for item in items], axis=0)
        action = np.stack([item["action"] for item in items], axis=0)
        next_z = np.stack([item["next_z"] for item in items], axis=0)
        rewards = [item["reward"] for item in items]
        reward_mask = np.asarray([reward is not None for reward in rewards], dtype=np.float32)[:, None]
        reward = np.asarray([0.0 if reward is None else reward for reward in rewards], dtype=np.float32)[:, None]
        done = np.asarray([item["done"] for item in items], dtype=np.float32)[:, None]
        return z, action, next_z, reward, reward_mask, done


class OnlineTDMPCTrainer:
    def __init__(self, agent, cfg, extra_modules=None):
        self.agent = agent
        self.model = agent.model
        self.cfg = agent.cfg
        self.train_cfg = cfg or {}
        self.extra_modules = extra_modules or {}
        self.device = next(self.model.parameters()).device
        self.replay = OnlineTDMPCReplay(self.train_cfg.get("buffer_size", 20000))
        self.batch_size = int(self.train_cfg.get("batch_size", 64))
        self.warmup_steps = int(self.train_cfg.get("warmup_steps", self.batch_size))
        self.updates_per_step = int(self.train_cfg.get("updates_per_step", 1))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 10.0))
        self.dynamics_weight = float(self.train_cfg.get("dynamics_weight", 1.0))
        self.bc_weight = float(self.train_cfg.get("bc_weight", 0.1))
        self.reward_weight = float(self.train_cfg.get("reward_weight", 0.0))
        self.q_weight = float(self.train_cfg.get("q_weight", 0.0))
        self.end_to_end_weight = float(self.train_cfg.get("end_to_end_weight", 1.0))
        self.end_to_end_dynamics_weight = float(self.train_cfg.get("end_to_end_dynamics_weight", 1.0))
        self.end_to_end_bc_weight = float(self.train_cfg.get("end_to_end_bc_weight", 0.1))
        self.end_to_end_reward_weight = float(self.train_cfg.get("end_to_end_reward_weight", 0.0))
        self.end_to_end_q_weight = float(self.train_cfg.get("end_to_end_q_weight", 0.0))
        self.save_every_updates = int(self.train_cfg.get("save_every_updates", 0))
        self.save_path = self.train_cfg.get("save_path", None)
        lr = float(self.train_cfg.get("lr", getattr(self.cfg, "lr", 1e-4)))
        adaptor_lr = float(self.train_cfg.get("adaptor_lr", lr))
        weight_decay = float(self.train_cfg.get("weight_decay", 0.0))

        self.model.requires_grad_(True)
        if hasattr(self.model, "_target_Qs"):
            self.model._target_Qs.requires_grad_(False)
        self.model.eval()
        tdmpc_params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and not name.startswith("_target_Qs.")
        ]
        param_groups = [{"params": tdmpc_params, "lr": lr}]
        extra_params = []
        for module in self.extra_modules.values():
            if module is None:
                continue
            module.requires_grad_(True)
            module.eval()
            extra_params.extend([p for p in module.parameters() if p.requires_grad])
        if extra_params:
            param_groups.append({"params": extra_params, "lr": adaptor_lr})
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        self.num_updates = 0

    def add_transition(self, z, action, next_z, reward=None, done=False):
        action_dim = int(getattr(self.cfg, "action_dim", np.asarray(action).reshape(-1).shape[0]))
        self.replay.add(z, np.asarray(action, dtype=np.float32).reshape(-1)[:action_dim], next_z, reward, done)

    def _to_tensor_batch(self):
        z, action, next_z, reward, reward_mask, done = self.replay.sample(self.batch_size)
        return (
            torch.as_tensor(z, dtype=torch.float32, device=self.device),
            torch.as_tensor(action, dtype=torch.float32, device=self.device),
            torch.as_tensor(next_z, dtype=torch.float32, device=self.device),
            torch.as_tensor(reward, dtype=torch.float32, device=self.device),
            torch.as_tensor(reward_mask, dtype=torch.float32, device=self.device),
            torch.as_tensor(done, dtype=torch.float32, device=self.device),
        )

    def train_ready(self):
        return len(self.replay) >= max(self.batch_size, self.warmup_steps)

    def _task_for_batch(self, batch_size):
        if bool(getattr(self.cfg, "multitask", False)):
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return None

    def _reward_q_losses(self, z, action, reward, reward_mask, done, task, reward_weight, q_weight):
        reward_loss = torch.zeros((), device=self.device)
        if reward_weight > 0 and reward_mask is not None and reward_mask.sum() > 0:
            reward_pred = self.model.reward(z, action, task).float()
            if reward_pred.shape[-1] == 1:
                reward_mse = (reward_pred - reward).pow(2) * reward_mask
                reward_loss = reward_mse.sum() / reward_mask.sum().clamp_min(1.0)

        q_loss = torch.zeros((), device=self.device)
        if q_weight > 0 and reward_mask is not None and reward_mask.sum() > 0:
            with torch.no_grad():
                next_pi_output = self.model.pi(z.detach(), task)
                next_action = next_pi_output[0] if isinstance(next_pi_output, tuple) else next_pi_output
                discount = getattr(self.agent, "discount", getattr(self.cfg, "discount", 0.99))
                if torch.is_tensor(discount):
                    discount = discount.to(device=self.device, dtype=torch.float32)
                else:
                    discount = torch.tensor(float(discount), device=self.device, dtype=torch.float32)
                target_q = self.model.Q(z.detach(), next_action, task, return_type="min", target=True).float()
                td_target = reward + discount * (1.0 - done) * target_q

            current_q1, current_q2 = self.model.Q(z, action, task, return_type="sep")
            if current_q1.shape == td_target.shape and current_q2.shape == td_target.shape:
                q_err = ((current_q1.float() - td_target).pow(2) + (current_q2.float() - td_target).pow(2))
                q_loss = (q_err * reward_mask).sum() / reward_mask.sum().clamp_min(1.0)
        return reward_loss, q_loss

    def update(self, current_z=None, current_action=None, previous_z=None, previous_action=None, reward=None, done=False):
        if not self.train_ready() and current_z is None:
            return None

        logs = []
        self.model.train()
        for module in self.extra_modules.values():
            if module is not None:
                module.train()
        for _ in range(self.updates_per_step):
            loss = torch.zeros((), device=self.device)
            dynamics_loss = torch.zeros((), device=self.device)
            bc_loss = torch.zeros((), device=self.device)
            reward_loss = torch.zeros((), device=self.device)
            q_loss = torch.zeros((), device=self.device)

            if self.train_ready():
                z, action, next_z, reward_batch, reward_mask, done_batch = self._to_tensor_batch()
                task = self._task_for_batch(z.shape[0])

                pred_next_z = self.model.next(z, action, task)
                dynamics_loss = F.mse_loss(pred_next_z.float(), next_z.float())

                pi_output = self.model.pi(z, task)
                pi_action = pi_output[0] if isinstance(pi_output, tuple) else pi_output
                bc_loss = F.mse_loss(pi_action.float(), action.float())
                reward_loss, q_loss = self._reward_q_losses(
                    z,
                    action,
                    reward_batch,
                    reward_mask,
                    done_batch,
                    task,
                    self.reward_weight,
                    self.q_weight,
                )
                loss = loss + (
                    self.dynamics_weight * dynamics_loss
                    + self.bc_weight * bc_loss
                    + self.reward_weight * reward_loss
                    + self.q_weight * q_loss
                )

            e2e_dynamics_loss = torch.zeros((), device=self.device)
            e2e_bc_loss = torch.zeros((), device=self.device)
            e2e_reward_loss = torch.zeros((), device=self.device)
            e2e_q_loss = torch.zeros((), device=self.device)
            if current_z is not None and current_action is not None and self.end_to_end_weight > 0:
                current_z = current_z.to(self.device, dtype=torch.float32)
                if current_z.ndim == 1:
                    current_z = current_z.unsqueeze(0)
                current_action = torch.as_tensor(current_action, dtype=torch.float32, device=self.device)
                if current_action.ndim == 1:
                    current_action = current_action.unsqueeze(0)
                action_dim = int(getattr(self.cfg, "action_dim", current_action.shape[-1]))
                current_action = current_action[..., :action_dim]
                task = self._task_for_batch(current_z.shape[0])

                pi_output = self.model.pi(current_z, task)
                pi_action = pi_output[0] if isinstance(pi_output, tuple) else pi_output
                e2e_bc_loss = F.mse_loss(pi_action.float(), current_action.float())

                if previous_z is not None and previous_action is not None:
                    prev_z = torch.as_tensor(previous_z, dtype=torch.float32, device=self.device)
                    prev_action = torch.as_tensor(previous_action, dtype=torch.float32, device=self.device)
                    if prev_z.ndim == 1:
                        prev_z = prev_z.unsqueeze(0)
                    if prev_action.ndim == 1:
                        prev_action = prev_action.unsqueeze(0)
                    prev_action = prev_action[..., :action_dim]
                    pred_current_z = self.model.next(prev_z, prev_action, task)
                    e2e_dynamics_loss = F.mse_loss(pred_current_z.float(), current_z.float())

                reward_mask = None
                reward_tensor = None
                done_tensor = None
                if reward is not None:
                    reward_tensor = torch.as_tensor(
                        np.asarray(reward, dtype=np.float32).reshape(1, 1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    reward_mask = torch.ones_like(reward_tensor)
                    done_tensor = torch.as_tensor([[float(done)]], dtype=torch.float32, device=self.device)
                if reward_tensor is not None:
                    e2e_reward_loss, e2e_q_loss = self._reward_q_losses(
                        current_z,
                        current_action,
                        reward_tensor,
                        reward_mask,
                        done_tensor,
                        task,
                        self.end_to_end_reward_weight,
                        self.end_to_end_q_weight,
                    )

                loss = loss + self.end_to_end_weight * (
                    self.end_to_end_dynamics_weight * e2e_dynamics_loss
                    + self.end_to_end_bc_weight * e2e_bc_loss
                    + self.end_to_end_reward_weight * e2e_reward_loss
                    + self.end_to_end_q_weight * e2e_q_loss
                )

            if not loss.requires_grad:
                return None
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_params = []
            for group in self.optimizer.param_groups:
                grad_params.extend(group["params"])
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_params, self.max_grad_norm)
            self.optimizer.step()
            if (self.q_weight > 0 or self.end_to_end_q_weight > 0) and hasattr(self.model, "soft_update_target_Q"):
                self.model.soft_update_target_Q()
            self.num_updates += 1

            logs.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "dynamics_loss": float(dynamics_loss.detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "reward_loss": float(reward_loss.detach().cpu()),
                    "q_loss": float(q_loss.detach().cpu()),
                    "e2e_dynamics_loss": float(e2e_dynamics_loss.detach().cpu()),
                    "e2e_bc_loss": float(e2e_bc_loss.detach().cpu()),
                    "e2e_reward_loss": float(e2e_reward_loss.detach().cpu()),
                    "e2e_q_loss": float(e2e_q_loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                    "buffer_size": len(self.replay),
                    "num_updates": self.num_updates,
                }
            )

        if self.save_every_updates > 0 and self.save_path and self.num_updates % self.save_every_updates == 0:
            self.save(self.save_path)

        self.model.eval()
        for module in self.extra_modules.values():
            if module is not None:
                module.eval()
        return logs[-1] if logs else None

    def save(self, save_path):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        payload = {"model": self.model.state_dict(), "num_updates": self.num_updates}
        if self.extra_modules.get("adaptor") is not None:
            payload["adaptor_state_dict"] = self.extra_modules["adaptor"].state_dict()
        if self.extra_modules.get("beta_fusion") is not None:
            payload["beta_fusion_state_dict"] = self.extra_modules["beta_fusion"].state_dict()
        torch.save(payload, save_path)


class _PrecomputedZEncoder:
    """Adapter expected by ExternalZTDMPCTestController: return the z already computed by GE."""

    def __init__(self, device):
        self.device = torch.device(device)
        self._z = None

    def set_z(self, z):
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        elif z_t.ndim > 2:
            z_t = z_t.reshape(z_t.shape[0], -1)
        self._z = z_t.contiguous()

    def encode(self, obs, agent_id=0):
        if self._z is None:
            raise RuntimeError("No precomputed z is available for this request.")
        return self._z


class ZaAdaptorServer(ZaAdaptorActor):
    def __init__(self, host, port, metadata=None, tdmpc_overrides=None, **kwargs):
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self.online_trainer = None
        self._online_pending = {}
        self._cached_zb = {}
        self.reuse_visual_zb_for_buffered_actions = bool(
            self.server_cfg.get("reuse_visual_zb_for_buffered_actions", False)
        )
        if tdmpc_overrides:
            tdmpc_cfg = getattr(self.args, "tdmpc", {}) or {}
            tdmpc_cfg.update({k: v for k, v in tdmpc_overrides.items() if v is not None})
            self.args.tdmpc = tdmpc_cfg
        self._prepare_tdmpc_controller()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _prepare_tdmpc_controller(self):
        self.tdmpc_cfg = getattr(self.args, "tdmpc", {})
        if not self.tdmpc_cfg.get("enabled", True):
            self.controller = None
            return

        project_root = self.tdmpc_cfg.get("project_root", "")
        model_path = self.tdmpc_cfg.get("model_path", "")
        if not project_root or not model_path:
            raise ValueError(
                "Set tdmpc.project_root and tdmpc.model_path in the YAML. "
                "The Za adaptor server now returns TD-MPC2 actions, so it must load the TD-MPC2 bridge and checkpoint."
            )

        gpu = self.tdmpc_cfg.get("gpu", None)
        tdm_cfg_args = self.tdmpc_cfg.get("tdmpc_cfg_args", "")
        cfg_overrides = self.tdmpc_cfg.get("cfg_overrides", {}) or {}
        self.tdmpc_runtime_cfg = load_tdmpc_cfg(
            project_root=project_root,
            tdm_cfg_args=tdm_cfg_args,
            cfg_overrides=cfg_overrides,
            gpu=gpu,
        )
        if self.tdmpc_cfg.get("deterministic", True):
            setattr(self.tdmpc_runtime_cfg, "eval_mode", True)
        setattr(self.tdmpc_runtime_cfg, "tb_log_figures", False)

        self.tdmpc_agent = load_tdmpc2_agent(self.tdmpc_runtime_cfg, model_path, gpu=gpu)
        self.z_encoder_for_tdmpc = _PrecomputedZEncoder(device="cuda")
        planner_map_history = PlannerMapHistory(
            cfg=self.tdmpc_runtime_cfg,
            map_key=self.tdmpc_cfg.get("map_key", "laser_map"),
            normalize=self.tdmpc_cfg.get("map_normalize", "auto"),
            invert=bool(self.tdmpc_cfg.get("map_invert", False)),
        )
        self.controller = ExternalZTDMPCTestController(
            tdmpc_agent=self.tdmpc_agent,
            z_encoder=self.z_encoder_for_tdmpc,
            cfg=self.tdmpc_runtime_cfg,
            velocity_key=self.tdmpc_cfg.get("velocity_key", "velocity"),
            velocity_format=self.tdmpc_cfg.get("velocity_format", "real"),
            planner_map_history=planner_map_history,
            disable_env_dyn_weight=bool(self.tdmpc_cfg.get("disable_env_dyn_weight", False)),
            deterministic=bool(self.tdmpc_cfg.get("deterministic", True)),
        )
        self._last_action_norm = {}
        online_cfg = self.tdmpc_cfg.get("online_training", {}) or {}
        if online_cfg.get("enabled", False):
            self.online_train_adaptor = bool(online_cfg.get("train_adaptor", False))
            self.online_trainer = OnlineTDMPCTrainer(
                self.tdmpc_agent,
                online_cfg,
                extra_modules={
                    "adaptor": self.adaptor if self.online_train_adaptor else None,
                    "beta_fusion": self.beta_fusion if self.online_train_adaptor else None,
                },
            )
            logger.info("Enabled online TD-MPC training: %s", online_cfg)
        else:
            self.online_train_adaptor = False

    def reset(self):
        super().reset()
        if hasattr(self, "controller") and self.controller is not None:
            self.controller.reset()
        if hasattr(self, "_last_action_norm"):
            self._last_action_norm.clear()
        if hasattr(self, "_online_pending"):
            self._online_pending.clear()
        if hasattr(self, "_cached_zb"):
            self._cached_zb.clear()

    def _has_buffered_action(self, agent_id):
        buffers = getattr(self.tdmpc_agent, "action_buffer", None)
        return buffers is not None and agent_id < len(buffers) and len(buffers[agent_id]) > 0

    def _build_tdmpc_obs(self, request, z_beta):
        obs_for_planner = {}

        for key in ("laser_map", "lidar_map", "costmap", "obs_map", "map"):
            if key in request:
                obs_for_planner[key] = request[key]

        if "velocity" in request:
            obs_for_planner["velocity"] = request["velocity"]
        else:
            beta = np.asarray(z_beta, dtype=np.float32).reshape(-1)
            if beta.shape[0] >= 5:
                # 3D state beta: [vx, vy, vyaw, goal_r, goal_theta].
                # 8D state beta: [torso_r, torso_p, torso_y, hb, pyaw,
                #                 vx, vy, vyaw, goal_r, goal_theta].
                velocity_indices = (5, 7) if beta.shape[0] >= 10 else (0, 2)
                obs_for_planner["velocity"] = np.asarray(
                    [beta[velocity_indices[0]], beta[velocity_indices[1]]], dtype=np.float32
                )

        if "goal_rel" in request:
            obs_for_planner["goal_rel"] = request["goal_rel"]
        else:
            beta = np.asarray(z_beta, dtype=np.float32).reshape(-1)
            if beta.shape[0] >= 5:
                obs_for_planner["goal_rel"] = beta[-2:].astype(np.float32)

        return obs_for_planner

    def _extract_beta_numpy(self, request):
        fusion_cfg = getattr(self.args, "beta_fusion", {})
        expected_beta_dim = int(fusion_cfg.get("beta_dim", 5))
        if self.beta_key in request:
            beta = np.asarray(request[self.beta_key], dtype=np.float32).reshape(-1)
            if beta.shape[0] != expected_beta_dim:
                raise ValueError(
                    f"Expected beta with {expected_beta_dim} values from beta_fusion.beta_dim, got {beta.shape[0]}."
                )
            return beta
        state_key = self.state_key if self.state_key in request else "state"
        state = np.asarray(request[state_key], dtype=np.float32).reshape(-1)
        goal_key = self.goal_key if self.goal_key in request else "target"
        goal = np.asarray(request[goal_key], dtype=np.float32).reshape(-1)
        state_dim = expected_beta_dim - 2
        if state_dim == 3:
            beta_state = state[:3]
        elif state.shape[0] >= state_dim:
            beta_state = state[:state_dim]
        else:
            raise ValueError(
                f"beta_fusion.beta_dim={expected_beta_dim} requires robot_state with at least {state_dim} "
                f"values plus goal_rel[2], but got robot_state shape {state.shape}."
            )
        return np.concatenate([beta_state, goal[:2]], axis=0).astype(np.float32)

    def _print_velocity_debug(self, beta, tdmpc_obs, result=None, action=None, action_format=None):
        beta_np = np.asarray(beta, dtype=np.float32).reshape(-1)
        velocity = np.asarray(tdmpc_obs.get("velocity", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        goal_rel = np.asarray(tdmpc_obs.get("goal_rel", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        v_model, w_model = extract_velocity_model_norm(
            tdmpc_obs,
            self.tdmpc_runtime_cfg,
            velocity_key=self.tdmpc_cfg.get("velocity_key", "velocity"),
            velocity_format=self.tdmpc_cfg.get("velocity_format", "real"),
            fallback_action_norm=None,
        )

        print(
            "[za_adaptor_server][velocity_parse] "
            f"beta={beta_np.tolist()} "
            f"tdmpc_velocity_real=[linear_v={float(velocity[0]):.6f}, angular_w={float(velocity[1]):.6f}] "
            f"tdmpc_velocity_model_norm=[v_model={v_model:.6f}, w_model={w_model:.6f}] "
            f"goal_rel={goal_rel.tolist()} "
            f"max_linear_vel={float(getattr(self.tdmpc_runtime_cfg, 'max_linear_vel', 1.0)):.6f} "
            f"max_angular_vel={float(getattr(self.tdmpc_runtime_cfg, 'max_angular_vel', 1.0)):.6f}",
            flush=True,
        )

        if result is not None:
            action_np = np.asarray(action, dtype=np.float32).reshape(-1)
            action_norm = np.asarray(result.action_norm, dtype=np.float32).reshape(-1)
            action_real = np.asarray(result.action_real, dtype=np.float32).reshape(-1)
            print(
                "[za_adaptor_server][action_output] "
                f"action_format={action_format} "
                f"returned_action={action_np.tolist()} "
                f"action_norm=[linear_v_norm={float(action_norm[0]):.6f}, angular_w_norm={float(action_norm[1]):.6f}] "
                f"action_real=[linear_v={float(action_real[0]):.6f}, angular_w={float(action_real[1]):.6f}] "
                f"new_traj={bool(result.new_traj)}",
                flush=True,
            )

    def _extract_online_feedback(self, request):
        reward = None
        for key in ("reward", "prev_reward", "last_reward"):
            if key in request:
                reward = request[key]
                break
        done = False
        for key in ("done", "terminal", "is_done"):
            if key in request:
                done = bool(request[key])
                break
        return reward, done

    def _online_train_from_request(self, agent_id, z, z_tensor, action_norm, request, reset_requested):
        if self.online_trainer is None:
            return None

        reward, done = self._extract_online_feedback(request)
        train_log = None
        pending = self._online_pending.get(agent_id)
        if pending is not None and not reset_requested:
            self.online_trainer.add_transition(
                z=pending["z"],
                action=pending["action_norm"],
                next_z=z,
                reward=reward,
                done=done,
            )

        if not reset_requested:
            train_log = self.online_trainer.update(
                current_z=z_tensor if self.online_train_adaptor else None,
                current_action=action_norm,
                previous_z=pending["z"] if pending is not None else None,
                previous_action=pending["action_norm"] if pending is not None else None,
                reward=reward,
                done=done,
            )

        if done or reset_requested:
            self._online_pending.pop(agent_id, None)

        return {
            "buffer_size": len(self.online_trainer.replay),
            "num_updates": self.online_trainer.num_updates,
            "last_update": train_log,
        }

    def infer_action(self, **request):
        reset_requested = bool(request.get("reset", False))
        prompt = request.get("prompt", self.prompt_default)
        if reset_requested or ("<reset>" in prompt):
            self.reset()
        agent_id = int(request.get("agent_id", 0))
        _, done_requested = self._extract_online_feedback(request)
        t0 = bool(request.get("t0", False) or reset_requested or done_requested or ("<reset>" in prompt))
        clean_prompt = prompt.replace("<reset>", "")

        # Always advance the camera history. The default recomputes visual Zb on
        # every request; trajectory-level reuse is available only via the
        # explicit server config switch below. BetaFusion + ZaAdaptor are still
        # evaluated every request so current state/goal changes are kept.
        obs_tensor, clean_prompt, raw_size, n_view = self.update_observation_history(
            request["obs"],
            prompt=clean_prompt,
            execution_step=request.get("execution_step", 1),
        )
        can_reuse_zb = (
            self.reuse_visual_zb_for_buffered_actions
            and not t0
            and self._has_buffered_action(agent_id)
            and agent_id in self._cached_zb
        )
        if can_reuse_zb:
            zb = self._cached_zb[agent_id]
        else:
            zb = self.encode_zb_from_history(obs_tensor, clean_prompt, raw_size, n_view)
            self._cached_zb[agent_id] = zb.detach()

        z_tensor = self.adapt_zb_tensor(
            zb,
            trainable_adaptor=self.online_train_adaptor,
            **request,
        )
        z = z_tensor.detach().cpu().numpy()[0]
        if not self.online_train_adaptor:
            z_tensor = None

        beta = self._extract_beta_numpy(request)
        tdmpc_obs = self._build_tdmpc_obs(request, beta)
        self._print_velocity_debug(beta, tdmpc_obs)

        self.z_encoder_for_tdmpc.set_z(z)
        result = self.controller.act(
            obs=tdmpc_obs,
            agent_id=agent_id,
            t0=t0,
            fallback_action_norm=self._last_action_norm.get(agent_id),
            task=request.get("task", None),
        )
        self._last_action_norm[agent_id] = result.action_norm.copy()

        action_format = self.tdmpc_cfg.get("action_format", "real")
        if action_format == "real":
            action = result.action_real
        elif action_format == "norm":
            action = result.action_norm
        else:
            action = action_for_env(result.action_norm, self.tdmpc_runtime_cfg, action_format)
        online_train_status = self._online_train_from_request(
            agent_id,
            z,
            z_tensor,
            result.action_norm,
            request,
            reset_requested,
        )
        if self.online_trainer is not None and not (done_requested or reset_requested):
            self._online_pending[agent_id] = {
                "z": np.asarray(z, dtype=np.float32).reshape(-1).copy(),
                "action_norm": result.action_norm.astype(np.float32).reshape(-1).copy(),
            }
        self._print_velocity_debug(beta, tdmpc_obs, result=result, action=action, action_format=action_format)

        response = {
            "actions": np.asarray(action, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "action_norm": result.action_norm.astype(np.float32),
            "action_real": result.action_real.astype(np.float32),
            "new_traj": bool(result.new_traj),
            "visual_cache_hit": bool(can_reuse_zb),
        }
        if online_train_status is not None:
            response["online_training"] = online_train_status
        if self.tdmpc_cfg.get("return_z", False):
            response["z"] = z
            response["za"] = z
        return response

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                start_time = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())
                response = self.infer_action(**request)
                elapsed_ms = 1000.0 * (time.monotonic() - start_time)
                response["server_timing"] = {"infer_action_ms": elapsed_ms}
                await websocket.send(packer.pack(response))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break

            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
