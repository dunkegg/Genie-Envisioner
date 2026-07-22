#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isaac test runner for TD-MPC2 with an external latent-z encoder.

核心逻辑：
    Isaac obs_t --(大 encoder + 小 adapter)--> z_t
    z_t --( TD-MPC2 dynamics/reward/Q/pi + MPPI planner)--> action_t
    action_t --> Isaac env.step(action_t)

本脚本会加载 TD-MPC2 checkpoint，但不会调用原来的 TD-MPC2 encoder / multi_encode。
外部 z encoder 由下面任一接口提供：

1) 函数式接口，推荐：

    # my_encoder_pkg/z_adapter.py
    def encode_z(obs, agent_id=None, device=None, cfg=None, **kwargs):
        # obs 是 Isaac env 返回的单个 agent 的 observation dict
        # 返回 torch.Tensor 或 np.ndarray，shape [latent_dim] 或 [1, latent_dim]
        return z

运行参数：
    --z-encoder-module my_encoder_pkg.z_adapter
    --z-encoder-function encode_z

2) 类接口：

    class IsaacZEncoder:
        def __init__(self, checkpoint_path=None, device="cuda", cfg=None, **kwargs): ...
        def __call__(self, obs, agent_id=None, device=None, cfg=None):
            return z

运行参数：
    --z-encoder-module my_encoder_pkg.z_adapter
    --z-encoder-class IsaacZEncoder
    --z-encoder-checkpoint /path/to/adapter.pt

Isaac env 仍建议返回：
    obs_i = {
        "laser_map": np.ndarray,      # [128, 256] 或 [T, 128, 256]，外部 encoder 可自行使用
        "velocity": [v, w],           # 当前真实速度，m/s 和 rad/s，供 TD-MPC2 planner 做速度/加速度约束
        "goal_rel": [distance, theta],# 外部 encoder 可自行使用
        ...
    }

注意：
    - z 必须与原 TD-MPC2 checkpoint 的 latent 维度一致。
    - 输出动作 action_norm 是 TD-MPC2 归一化动作：v,w 均在 [-1,1]。
    - 如果 --action-format real，则传给 Isaac env 的动作会转换为 [m/s, rad/s]。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import random
import shlex
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# -----------------------------
# Generic helpers
# -----------------------------


def _json_loads_object(text: str, arg_name: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Got: {text}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object/dict. Got {type(obj)}")
    return obj


def as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def first_existing(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return default


def bool_from_any(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        if x.size == 1:
            return bool(x.reshape(-1)[0])
        return bool(np.any(x))
    if isinstance(x, (list, tuple)):
        return any(bool_from_any(v) for v in x)
    return bool(x)


def value_for_agent(value: Any, agent_id: int, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] > agent_id:
            return value[agent_id]
        return default
    if isinstance(value, (list, tuple)):
        if len(value) > agent_id:
            return value[agent_id]
        return default
    if isinstance(value, Mapping):
        # A dict is usually global info. Do not index by agent unless it explicitly stores per-agent values.
        return value.get(agent_id, default)
    return value


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Project / cfg / model loading
# -----------------------------


def add_to_syspath(path: Union[str, Path], must_exist: bool = True) -> Path:
    p = Path(path).expanduser().resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def add_project_root_to_syspath(project_root: Union[str, Path]) -> Path:
    root = add_to_syspath(project_root)
    missing = []
    for name in ("exp_config.py", "tdmpc2.py"):
        if not (root / name).exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"--project-root must contain exp_config.py and tdmpc2.py. Missing {missing} under {root}"
        )
    return root


def load_tdmpc_cfg(
    project_root: Union[str, Path],
    tdm_cfg_args: str = "",
    cfg_overrides: Optional[Dict[str, Any]] = None,
    gpu: Optional[int] = None,
) -> Any:
    """Load cfg from the copied exp_config.py without letting our script args confuse argparse."""
    add_project_root_to_syspath(project_root)

    argv_backup = sys.argv[:]
    try:
        # exp_config.get_config() uses argparse.parse_args(), so hide this script's custom args.
        sys.argv = [argv_backup[0]] + shlex.split(tdm_cfg_args)
        exp_config = importlib.import_module("exp_config")
        cfg = exp_config.get_config()
    finally:
        sys.argv = argv_backup

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    if gpu is not None:
        setattr(cfg, "gpu", int(gpu))

    return cfg


def load_tdmpc2_agent(cfg: Any, model_path: Union[str, Path], gpu: Optional[int] = None) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The uploaded TD-MPC2 code uses torch.device('cuda') and .cuda() internally. "
            "Please run this script in a CUDA-enabled Isaac environment."
        )
    if gpu is not None:
        torch.cuda.set_device(int(gpu))

    # WorldVLN also exposes a top-level package named ``tools``.  When its
    # backbone is initialized first, Python caches that package in sys.modules
    # and TD-MPC's absolute ``from tools...`` imports resolve against WorldVLN
    # even though the TD-MPC project root is now first on sys.path.  Temporarily
    # isolate the cached namespace for TD-MPC's initial import, then restore the
    # WorldVLN modules used by video inference.  Imported TD-MPC modules retain
    # direct references to their own utilities.
    worldvln_tools = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "tools" or name.startswith("tools.")
    }
    for name in worldvln_tools:
        sys.modules.pop(name, None)
    try:
        tdmpc2_mod = importlib.import_module("tdmpc2")
        TDMPC2 = getattr(tdmpc2_mod, "TDMPC2")
        agent = TDMPC2(cfg)
        agent.load(str(Path(model_path).expanduser().resolve()))
    finally:
        for name in list(sys.modules):
            if name == "tools" or name.startswith("tools."):
                sys.modules.pop(name, None)
        sys.modules.update(worldvln_tools)

    agent.model.eval()
    for p in agent.model.parameters():
        p.requires_grad_(False)
    return agent


# -----------------------------
# External z encoder wrapper
# -----------------------------


def _call_with_filtered_kwargs(fn: Callable[..., Any], obs: Any, all_kwargs: Dict[str, Any]) -> Any:
    """Call fn(obs, **supported_kwargs). Falls back to a few common signatures."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        kwargs = all_kwargs if has_var_kw else {k: v for k, v in all_kwargs.items() if k in params}
        return fn(obs, **kwargs)
    except (TypeError, ValueError):
        # Some torch modules / C++ functions have signatures that inspect cannot parse.
        pass

    # Conservative fallbacks. We deliberately avoid swallowing non-TypeError exceptions from user code.
    try:
        return fn(obs, **all_kwargs)
    except TypeError:
        try:
            return fn(obs)
        except TypeError:
            return fn(obs, all_kwargs.get("agent_id", None))


class ExternalZEncoder:
    """
    Loads the colleague's big frozen encoder + small adapter through a flexible function/class interface.

    The wrapped callable should accept one agent's Isaac obs and return z.
    The return can be:
        - torch.Tensor / np.ndarray / list with shape [Z] or [1,Z]
        - dict containing key "z"
        - tuple/list whose first element is z
    """

    def __init__(
        self,
        module_name: str,
        function_name: Optional[str],
        class_name: Optional[str],
        checkpoint_path: Optional[str],
        kwargs: Dict[str, Any],
        device: torch.device,
        cfg: Any,
    ) -> None:
        self.module_name = module_name
        self.function_name = function_name
        self.class_name = class_name
        self.checkpoint_path = checkpoint_path
        self.kwargs = dict(kwargs)
        self.device = device
        self.cfg = cfg

        mod = importlib.import_module(module_name)
        if class_name:
            cls = getattr(mod, class_name)
            init_kwargs = dict(self.kwargs)
            init_kwargs.setdefault("device", str(device))
            init_kwargs.setdefault("cfg", cfg)
            if checkpoint_path is not None:
                # Common names; unsupported names are filtered by constructor signature when possible.
                init_kwargs.setdefault("checkpoint_path", checkpoint_path)
                init_kwargs.setdefault("ckpt_path", checkpoint_path)
                init_kwargs.setdefault("model_path", checkpoint_path)
            obj = self._instantiate_class(cls, init_kwargs)
            self.obj = obj
            self.fn = obj
            self._try_post_load(obj, checkpoint_path)
            if hasattr(obj, "to"):
                try:
                    obj.to(device)
                except Exception:
                    pass
            if hasattr(obj, "eval"):
                try:
                    obj.eval()
                except Exception:
                    pass
        else:
            if not function_name:
                function_name = "encode_z"
            self.obj = None
            self.fn = getattr(mod, function_name)

    def _instantiate_class(self, cls: type, kwargs: Dict[str, Any]) -> Any:
        try:
            sig = inspect.signature(cls)
            params = sig.parameters
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            use_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
            return cls(**use_kwargs)
        except (TypeError, ValueError):
            return cls(**kwargs)

    def _try_post_load(self, obj: Any, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            return
        # If the checkpoint was already consumed by __init__, this is harmless only if load exists.
        if hasattr(obj, "load"):
            try:
                obj.load(checkpoint_path)
                return
            except TypeError:
                pass
            except Exception as exc:
                print(f"[WARN] external encoder .load({checkpoint_path}) failed: {exc}")
                return
        if hasattr(obj, "load_state_dict"):
            try:
                sd = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(sd, Mapping):
                    for key in ("model", "state_dict", "encoder", "adapter"):
                        if key in sd and isinstance(sd[key], Mapping):
                            sd = sd[key]
                            break
                obj.load_state_dict(sd)
            except Exception as exc:
                print(f"[WARN] external encoder load_state_dict failed: {exc}")

    @torch.no_grad()
    def encode(self, obs: Any, agent_id: int) -> torch.Tensor:
        kwargs = {
            "agent_id": agent_id,
            "device": self.device,
            "cfg": self.cfg,
            "tdmpc_cfg": self.cfg,
        }
        out = _call_with_filtered_kwargs(self.fn, obs, kwargs)
        z = self._extract_z(out)
        z_t = self._to_tensor(z)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        elif z_t.ndim > 2:
            z_t = z_t.reshape(z_t.shape[0], -1)
        if z_t.shape[0] != 1:
            # Per-agent testing expects one latent per call.
            if z_t.shape[0] > 1:
                z_t = z_t[:1]
            else:
                raise ValueError(f"External z encoder returned invalid shape: {tuple(z_t.shape)}")
        expected = int(getattr(self.cfg, "latent_dim", z_t.shape[-1])) * int(getattr(self.cfg, "enc_num", 1))
        if z_t.shape[-1] != expected:
            print(
                f"[WARN] z dim mismatch for agent {agent_id}: got {z_t.shape[-1]}, "
                f"expected cfg.latent_dim*cfg.enc_num={expected}. Continue anyway."
            )
        return z_t.contiguous()

    def _extract_z(self, out: Any) -> Any:
        if isinstance(out, Mapping):
            if "z" not in out:
                raise KeyError("External encoder returned a dict but no key 'z' was found.")
            return out["z"]
        if isinstance(out, tuple):
            if len(out) == 0:
                raise ValueError("External encoder returned an empty tuple.")
            return out[0]
        return out

    def _to_tensor(self, z: Any) -> torch.Tensor:
        if isinstance(z, torch.Tensor):
            return z.to(self.device, dtype=torch.float32, non_blocking=True)
        return torch.as_tensor(z, dtype=torch.float32, device=self.device)


# -----------------------------
# Isaac env adapter loading / return parsing
# -----------------------------


def import_isaac_env(module_name: str, class_name: str, kwargs: Dict[str, Any]) -> Any:
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def call_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    if seed is not None:
        try:
            ret = env.reset(seed=seed)
        except TypeError:
            ret = env.reset()
    else:
        ret = env.reset()

    if isinstance(ret, tuple) and len(ret) == 2:
        obs, info = ret
        return obs, info if isinstance(info, dict) else {"info": info}
    if isinstance(ret, Mapping) and "obs" in ret:
        return ret["obs"], ret.get("info", {})
    if isinstance(ret, Mapping) and "observation" in ret:
        return ret["observation"], ret.get("info", {})
    return ret, {}


def unpack_step(ret: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    if isinstance(ret, Mapping):
        obs = ret.get("obs", ret.get("observation", ret.get("next_obs", None)))
        reward = ret.get("reward", ret.get("rewards", None))
        done = ret.get("done", ret.get("dones", None))
        if done is None:
            terminated = ret.get("terminated", ret.get("terminations", False))
            truncated = ret.get("truncated", ret.get("truncations", False))
            done = np.asarray(terminated) | np.asarray(truncated)
        info = ret.get("info", {})
        return obs, reward, done, info if isinstance(info, dict) else {"info": info}

    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = np.asarray(terminated) | np.asarray(truncated)
            return obs, reward, done, info if isinstance(info, dict) else {"info": info}
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, reward, done, info if isinstance(info, dict) else {"info": info}
        if len(ret) == 2:
            obs, info = ret
            return obs, None, False, info if isinstance(info, dict) else {"info": info}
    raise ValueError(
        "env.step(action) must return (obs,reward,done,info), "
        "(obs,reward,terminated,truncated,info), or a dict with obs/reward/done/info."
    )


def split_batched_dict(obs: Mapping[str, Any], num_agents: int) -> Optional[List[Dict[str, Any]]]:
    if num_agents <= 1:
        return None
    # Only split if at least one array-like field has first dimension == num_agents.
    split_keys = []
    for k, v in obs.items():
        if isinstance(v, Mapping):
            continue
        try:
            arr = as_numpy(v)
        except Exception:
            continue
        if arr.ndim >= 1 and arr.shape[0] == num_agents:
            split_keys.append(k)
    if not split_keys:
        return None

    out = [dict() for _ in range(num_agents)]
    for k, v in obs.items():
        if isinstance(v, Mapping):
            # Keep nested objects shared unless they also explicitly contain per-agent values.
            for i in range(num_agents):
                out[i][k] = v
            continue
        try:
            arr = as_numpy(v)
        except Exception:
            for i in range(num_agents):
                out[i][k] = v
            continue
        if arr.ndim >= 1 and arr.shape[0] == num_agents:
            for i in range(num_agents):
                out[i][k] = arr[i]
        else:
            for i in range(num_agents):
                out[i][k] = v
    return out


def to_agent_obs_list(obs: Any, num_agents: Optional[int] = None) -> List[Any]:
    if isinstance(obs, Mapping):
        # Common wrapper keys.
        for key in ("obs", "observation", "agents", "agent_obs"):
            if key in obs and key not in ("laser_map", "map"):
                nested = obs[key]
                # Avoid unwrapping a single-agent obs that has an actual "obs" field for something else.
                if isinstance(nested, (list, tuple)) or (isinstance(nested, Mapping) and nested is not obs):
                    try:
                        return to_agent_obs_list(nested, num_agents=num_agents)
                    except Exception:
                        pass
        if num_agents is not None:
            split = split_batched_dict(obs, int(num_agents))
            if split is not None:
                return split
        return [obs]

    if isinstance(obs, (list, tuple)):
        return list(obs)

    arr = as_numpy(obs)
    if arr.ndim >= 2 and num_agents is not None and arr.shape[0] == int(num_agents):
        return [arr[i] for i in range(int(num_agents))]
    return [obs]


# -----------------------------
# Velocity, action, map history helpers
# -----------------------------


def extract_velocity_model_norm(
    obs: Any,
    cfg: Any,
    velocity_key: str = "velocity",
    velocity_format: str = "real",
    fallback_action_norm: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Return velocity in the same normalized form used by the original TD-MPC2 observation:
        v_model = v_real / max_linear_vel           in [0,1]
        w_model = (w_real / max_angular_vel + 1)/2  in [0,1]
    """
    vel = None
    if isinstance(obs, Mapping):
        vel = first_existing(
            obs,
            [velocity_key, "velocity", "vel", "cmd_vw", "current_velocity", "robot_velocity", "vw"],
            default=None,
        )

    if vel is None:
        if fallback_action_norm is not None:
            a = np.asarray(fallback_action_norm, dtype=np.float32).reshape(-1)
            return float(np.clip((a[0] + 1.0) * 0.5, 0.0, 1.0)), float(np.clip((a[1] + 1.0) * 0.5, 0.0, 1.0))
        return 0.0, 0.5

    v = np.asarray(vel, dtype=np.float32).reshape(-1)
    if v.size < 2:
        raise ValueError(f"Velocity field must contain at least [v,w]. Got: {vel}")

    max_v = float(getattr(cfg, "max_linear_vel", 1.0))
    max_w = float(getattr(cfg, "max_angular_vel", 1.0))

    if velocity_format == "real":
        v_model = float(np.clip(v[0] / max_v, 0.0, 1.0))
        w_model = float(np.clip((v[1] / max_w + 1.0) * 0.5, 0.0, 1.0))
    elif velocity_format == "model_norm":
        v_model = float(np.clip(v[0], 0.0, 1.0))
        w_model = float(np.clip(v[1], 0.0, 1.0))
    elif velocity_format == "action_norm":
        v_model = float(np.clip((v[0] + 1.0) * 0.5, 0.0, 1.0))
        w_model = float(np.clip((v[1] + 1.0) * 0.5, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown velocity_format: {velocity_format}")
    return v_model, w_model


def action_norm_to_real(action_norm: np.ndarray, cfg: Any) -> np.ndarray:
    a = np.asarray(action_norm, dtype=np.float32)
    out = np.zeros_like(a, dtype=np.float32)
    out[..., 0] = ((a[..., 0] + 1.0) * 0.5) * float(getattr(cfg, "max_linear_vel", 1.0))
    out[..., 1] = a[..., 1] * float(getattr(cfg, "max_angular_vel", 1.0))
    return out


def action_for_env(action_norm: np.ndarray, cfg: Any, action_format: str) -> np.ndarray:
    if action_format == "norm":
        return np.asarray(action_norm, dtype=np.float32)
    if action_format == "real":
        return action_norm_to_real(action_norm, cfg)
    raise ValueError(f"Unknown --action-format: {action_format}")


def normalize_map_values(arr: np.ndarray, mode: str = "auto", invert: bool = False) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if mode == "auto":
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            x = np.zeros_like(x, dtype=np.float32)
        else:
            mn = float(finite.min())
            mx = float(finite.max())
            if mx > 2.0 or mn < -1.5:
                x = x / 255.0
            elif mn < -0.05 and mx <= 1.05:
                # [-1,1] -> [0,1]
                x = (x + 1.0) * 0.5
    elif mode == "zero_one":
        x = x
    elif mode == "minus_one_one":
        x = (x + 1.0) * 0.5
    elif mode == "uint8":
        x = x / 255.0
    elif mode == "none":
        x = x
    else:
        raise ValueError(f"Unknown map normalize mode: {mode}")
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    if invert:
        x = 1.0 - x
    return x.astype(np.float32, copy=False)


def extract_map_from_obs(obs: Any, map_key: str) -> Optional[np.ndarray]:
    if not isinstance(obs, Mapping):
        return None
    v = first_existing(
        obs,
        [map_key, "laser_map", "lidar_map", "costmap", "obs_map", "map", "surrounding", "image"],
        default=None,
    )
    if v is None:
        return None
    if isinstance(v, Mapping):
        v = first_existing(v, ["temporal", "history", "combine", "map", "image"], default=None)
        if v is None:
            return None
    arr = np.asarray(v)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        # [C,H,W] -> use first channel for planner map history.
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        # [H,W,C] -> grayscale-ish. For RGB use mean.
        arr = arr.mean(axis=-1)
    if arr.ndim not in (2, 3):
        return None
    return arr


class PlannerMapHistory:
    """
    Only used by the TD-MPC2 risk-aware planner when cfg.use_env_dyn_weight=True.

    It is NOT the encoder. It just gives the existing planner a simple temporal map [T,H,W]
    so its dynamic-risk heuristic can run without the old vector observation.
    """

    def __init__(self, cfg: Any, map_key: str, normalize: str, invert: bool) -> None:
        self.cfg = cfg
        self.map_key = map_key
        self.normalize = normalize
        self.invert = invert
        self.T = int(getattr(cfg, "sample_length", 8))
        self.H = int(getattr(cfg, "img_height", 128))
        self.W = int(getattr(cfg, "img_width", 256))
        self.hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.T))

    def reset(self) -> None:
        self.hist.clear()

    def update_and_get(self, agent_id: int, obs: Any) -> Optional[np.ndarray]:
        arr = extract_map_from_obs(obs, self.map_key)
        if arr is None:
            return None
        arr = normalize_map_values(arr, self.normalize, self.invert)
        if arr.ndim == 3:
            # Already [T,H,W] style history.
            frames = [self._ensure_hw(arr[i]) for i in range(arr.shape[0])]
            for frame in frames[-self.T:]:
                self.hist[agent_id].append(frame)
        else:
            frame = self._ensure_hw(arr)
            if len(self.hist[agent_id]) == 0:
                for _ in range(self.T):
                    self.hist[agent_id].append(frame.copy())
            else:
                self.hist[agent_id].append(frame)
        return np.stack(list(self.hist[agent_id]), axis=0).astype(np.float32)

    def zeros_history(self) -> np.ndarray:
        return np.zeros((self.T, self.H, self.W), dtype=np.float32)

    def _ensure_hw(self, frame: np.ndarray) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32)
        if x.shape == (self.H, self.W):
            return x
        if x.shape == (self.W, self.H):
            return x.T.copy()
        # Avoid adding cv2 dependency here. Use simple nearest-neighbor resize with numpy indexing.
        src_h, src_w = x.shape[:2]
        rows = np.linspace(0, src_h - 1, self.H).round().astype(np.int64)
        cols = np.linspace(0, src_w - 1, self.W).round().astype(np.int64)
        return x[rows][:, cols].astype(np.float32)


def patch_tdmpc_dynamic_map_from_history(agent: Any) -> None:
    """Patch TDMPC2._build_dyn_map_from_observation to accept [T,H,W] map history."""
    original = getattr(agent, "_build_dyn_map_from_observation", None)

    def _build_dyn_map_from_map_history(self: Any, observation_np: Any) -> np.ndarray:
        arr = np.asarray(observation_np)
        Himg = int(getattr(self.cfg, "img_height", 128))
        Wimg = int(getattr(self.cfg, "img_width", 256))
        if arr.ndim == 2:
            # Single static map: no temporal difference available.
            return np.zeros((Himg, Wimg), dtype=np.float32)
        if arr.ndim == 3:
            maps = arr.astype(np.float32)
            if maps.shape[1:] != (Himg, Wimg):
                # Simple nearest-neighbor resize frame by frame.
                out = []
                for frame in maps:
                    src_h, src_w = frame.shape[:2]
                    rows = np.linspace(0, src_h - 1, Himg).round().astype(np.int64)
                    cols = np.linspace(0, src_w - 1, Wimg).round().astype(np.int64)
                    out.append(frame[rows][:, cols])
                maps = np.stack(out, axis=0)
            # Same convention as original: free=1, obstacle=0 -> occupancy=1-map.
            occ = 1.0 - np.clip(maps, 0.0, 1.0)
            if occ.shape[0] <= 1:
                dyn = np.zeros((Himg, Wimg), dtype=np.float32)
            else:
                dyn = np.abs(occ[1:] - occ[:-1]).mean(axis=0)
            p_lo = float(np.percentile(dyn, getattr(self.cfg, "dynmap_p_lo", 5)))
            p_hi = float(np.percentile(dyn, getattr(self.cfg, "dynmap_p_hi", 95)))
            dyn_norm = (dyn - p_lo) / (p_hi - p_lo + 1e-6)
            return np.clip(dyn_norm, 0.0, 1.0).astype(np.float32)
        if original is not None:
            return original(observation_np)
        return np.zeros((Himg, Wimg), dtype=np.float32)

    agent._build_dyn_map_from_observation = MethodType(_build_dyn_map_from_map_history, agent)


# -----------------------------
# TD-MPC2 latent-z controller
# -----------------------------


@dataclass
class ActionResult:
    action_norm: np.ndarray
    action_real: np.ndarray
    new_traj: bool
    z: torch.Tensor
    value: Optional[np.ndarray] = None


class ExternalZTDMPCTestController:
    """A thin replacement for TDMPC2.act(): it skips model.multi_encode and starts from external z."""

    def __init__(
        self,
        tdmpc_agent: Any,
        z_encoder: ExternalZEncoder,
        cfg: Any,
        velocity_key: str,
        velocity_format: str,
        planner_map_history: PlannerMapHistory,
        disable_env_dyn_weight: bool = False,
        deterministic: bool = True,
    ) -> None:
        self.agent = tdmpc_agent
        self.z_encoder = z_encoder
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.velocity_key = velocity_key
        self.velocity_format = velocity_format
        self.planner_map_history = planner_map_history
        self.deterministic = deterministic
        if disable_env_dyn_weight:
            setattr(self.cfg, "use_env_dyn_weight", False)
        patch_tdmpc_dynamic_map_from_history(self.agent)

    def reset(self) -> None:
        self.agent.reset_action_buffer()
        self.planner_map_history.reset()

    @torch.no_grad()
    def act(
        self,
        obs: Any,
        agent_id: int,
        t0: bool,
        fallback_action_norm: Optional[np.ndarray] = None,
        task: Optional[Any] = None,
    ) -> ActionResult:
        # PYL: check the dim of obs and z
        z = self.z_encoder.encode(obs, agent_id=agent_id)
        v_model, w_model = extract_velocity_model_norm(
            obs,
            self.cfg,
            velocity_key=self.velocity_key,
            velocity_format=self.velocity_format,
            fallback_action_norm=fallback_action_norm,
        )
        v_t = torch.tensor(v_model, dtype=torch.float32, device=self.device)
        w_t = torch.tensor(w_model, dtype=torch.float32, device=self.device)

        planner_obs = self.planner_map_history.update_and_get(agent_id, obs)
        if planner_obs is None:
            planner_obs = self.planner_map_history.zeros_history()

        action_tensor, new_traj, mpc_as, pi_a, act_a, value = self._act_from_z(
            z=z,
            agent_id=agent_id,
            t0=t0,
            eval_mode=self.deterministic,
            task=task,
            v_model=v_t,
            w_model=w_t,
            planner_observation=planner_obs,
        )
        action_norm = np.asarray(action_tensor.detach().cpu(), dtype=np.float32).reshape(-1, int(getattr(self.cfg, "action_dim", 2)))
        action_norm = np.clip(action_norm[0], -1.0, 1.0)
        action_real = action_norm_to_real(action_norm, self.cfg)
        value_np = None if value is None else as_numpy(value)
        return ActionResult(action_norm=action_norm, action_real=action_real, new_traj=new_traj, z=z, value=value_np)

    @torch.no_grad()
    def _act_from_z(
        self,
        z: torch.Tensor,
        agent_id: int,
        t0: bool,
        eval_mode: bool,
        task: Optional[Any],
        v_model: torch.Tensor,
        w_model: torch.Tensor,
        planner_observation: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, bool, Any, Any, Any, Any]:
        if task is not None and not isinstance(task, torch.Tensor):
            task = torch.tensor([task], device=self.device)

        # No-MPC mode: directly use policy head pi(z).
        if not bool(getattr(self.cfg, "mpc", True)):
            a = self.agent.model.pi(z, task)[int(not eval_mode)][0].unsqueeze(0)
            return a.cpu(), True, None, None, None, None

        # use_one mode in the original code returns one action immediately from plan.
        if bool(getattr(self.cfg, "use_one", False)):
            a, mpc_as, pi_a, act_a, value = self.agent.plan(
                z,
                agent_id,
                t0=t0,
                eval_mode=eval_mode,
                task=task,
                v_real=v_model,
                w_real=w_model,
                writer=None,
                observation=planner_observation,
            )
            return a.cpu(), True, _safe_cpu(mpc_as), _safe_cpu(pi_a), _safe_cpu(act_a), _safe_cpu(value)

        # Buffered trajectory execution, matching the original TDMPC2.act behavior.
        if len(self.agent.action_buffer[agent_id]) == 0:
            new_traj = True
            a_seq, self.agent.mpc_as_buffer[agent_id], self.agent.pi_a_buffer[agent_id], self.agent.act_a_buffer[agent_id], self.agent.value_buffer[agent_id] = self.agent.plan(
                z,
                agent_id,
                t0=t0,
                eval_mode=eval_mode,
                task=task,
                v_real=v_model,
                w_real=w_model,
                writer=None,
                observation=planner_observation,
            )
            if bool(getattr(self.cfg, "random_act_len", False)):
                act_len = random.randint(1, int(getattr(self.cfg, "rand_horizon", getattr(self.cfg, "horizon", 1))))
            else:
                act_len = int(getattr(self.cfg, "use_horizon", getattr(self.cfg, "horizon", 1)))
            for i in range(act_len):
                self.agent.action_buffer[agent_id].append(a_seq[i])
        else:
            new_traj = False

        action_now = self.agent.action_buffer[agent_id].pop(0).unsqueeze(0)
        return (
            action_now.cpu(),
            new_traj,
            _safe_cpu(self.agent.mpc_as_buffer[agent_id]),
            _safe_cpu(self.agent.pi_a_buffer[agent_id]),
            _safe_cpu(self.agent.act_a_buffer[agent_id]),
            _safe_cpu(self.agent.value_buffer[agent_id]),
        )


def _safe_cpu(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, list):
        return [_safe_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_safe_cpu(v) for v in x)
    return x


# -----------------------------
# Metrics / logging helpers
# -----------------------------


def extract_event(agent_obs: Any, info: Mapping[str, Any], keys: Sequence[str], agent_id: int) -> bool:
    # Prefer per-agent observation flags.
    if isinstance(agent_obs, Mapping):
        for k in keys:
            if k in agent_obs:
                return bool_from_any(agent_obs[k])
    # Then info dict.
    for k in keys:
        if k in info:
            return bool_from_any(value_for_agent(info[k], agent_id, default=False))
    return False


def reward_for_agent(reward: Any, agent_id: int) -> float:
    r = value_for_agent(reward, agent_id, default=0.0)
    try:
        return float(np.asarray(r).reshape(-1)[0])
    except Exception:
        return 0.0


def done_for_agent(done: Any, agent_id: int) -> bool:
    return bool_from_any(value_for_agent(done, agent_id, default=done))


def write_episode_csv_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "steps",
                "num_agents",
                "reward_sum",
                "success_count",
                "collision_count",
                "done_count",
                "seconds",
            ],
        )
        writer.writeheader()


def append_episode_csv(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# -----------------------------
# Main test loop
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test TD-MPC2 in Isaac using an external latent-z encoder instead of the original encoder."
    )

    # TD-MPC2 project/checkpoint
    parser.add_argument("--project-root", required=True, help="Directory containing exp_config.py, tdmpc2.py, tdmpc2_common/, tools/")
    parser.add_argument("--model-path", required=True, help="Path to your trained TD-MPC2 checkpoint")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA GPU id; also overrides cfg.gpu")
    parser.add_argument("--tdmpc-cfg-args", default="", help="Extra args passed to exp_config.get_config(), e.g. '--world_idx 20'")
    parser.add_argument("--cfg-overrides-json", default="{}", help="JSON dict applied to cfg after exp_config.get_config()")

    # Isaac env
    parser.add_argument("--isaac-project-root", default=None, help="Optional path added to sys.path so Python can import Isaac modules")
    parser.add_argument("--isaac-env-module", required=True, help="Python module containing the Isaac env adapter")
    parser.add_argument("--isaac-env-class", required=True, help="Class name of the Isaac env adapter")
    parser.add_argument("--isaac-env-kwargs-json", default="{}", help="JSON kwargs passed to Isaac env constructor")

    # External z encoder
    parser.add_argument("--z-encoder-module", required=True, help="Python module containing external z encoder function/class")
    parser.add_argument("--z-encoder-function", default="encode_z", help="Function name. Ignored if --z-encoder-class is set")
    parser.add_argument("--z-encoder-class", default=None, help="Optional class name for external z encoder")
    parser.add_argument("--z-encoder-checkpoint", default=None, help="Optional checkpoint for the colleague's encoder/adapter")
    parser.add_argument("--z-encoder-kwargs-json", default="{}", help="JSON kwargs for external z encoder class/function")

    # Evaluation loop
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=None, help="Defaults to cfg.episode_length")
    parser.add_argument("--num-agents", type=int, default=None, help="Overrides cfg.num_agent for splitting Isaac observations")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True, help="Use deterministic policy/planning noise settings")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Allow stochastic action noise in TD-MPC2 plan")
    parser.add_argument("--print-every", type=int, default=1, help="Print every N environment steps")

    # Env action / velocity conventions
    parser.add_argument("--action-format", choices=["real", "norm"], default="real", help="Action format passed to Isaac env.step")
    parser.add_argument("--squeeze-single-action", action="store_true", help="If num_agents=1, pass shape [2] instead of [1,2]")
    parser.add_argument("--velocity-key", default="velocity")
    parser.add_argument("--velocity-format", choices=["real", "model_norm", "action_norm"], default="real")

    # Optional planner dynamic-map support. This is separate from z encoding.
    parser.add_argument("--map-key", default="laser_map", help="Isaac obs key used only for risk-aware dynamic map history if needed")
    parser.add_argument("--map-normalize", choices=["auto", "zero_one", "minus_one_one", "uint8", "none"], default="auto")
    parser.add_argument("--map-invert", action="store_true")
    parser.add_argument("--disable-env-dyn-weight", action="store_true", help="Set cfg.use_env_dyn_weight=False during testing")

    # Output
    parser.add_argument("--output", default=None, help="Directory for evaluation metrics and optional rollout file")
    parser.add_argument("--save-rollout", action="store_true", help="Save per-step z/action/reward/debug tensors to rollout.pt")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.isaac_project_root:
        add_to_syspath(args.isaac_project_root)

    set_global_seed(args.seed)

    cfg_overrides = _json_loads_object(args.cfg_overrides_json, "--cfg-overrides-json")
    cfg = load_tdmpc_cfg(
        project_root=args.project_root,
        tdm_cfg_args=args.tdmpc_cfg_args,
        cfg_overrides=cfg_overrides,
        gpu=args.gpu,
    )
    if args.num_agents is not None:
        setattr(cfg, "num_agent", int(args.num_agents))
    num_agents = int(getattr(cfg, "num_agent", args.num_agents or 1))
    episode_steps = int(args.episode_steps or getattr(cfg, "episode_length", 100))

    # Keep module creation compatible with checkpoint; then optionally make planner deterministic after load.
    tdmpc_agent = load_tdmpc2_agent(cfg, args.model_path, gpu=args.gpu)
    if args.deterministic:
        # plan() uses cfg.eval_mode in several places for final sampling noise.
        setattr(cfg, "eval_mode", True)
    setattr(cfg, "tb_log_figures", False)

    # PYL: frozen encoder + adapter function
    z_encoder_kwargs = _json_loads_object(args.z_encoder_kwargs_json, "--z-encoder-kwargs-json")
    z_encoder = ExternalZEncoder(
        module_name=args.z_encoder_module,
        function_name=args.z_encoder_function,
        class_name=args.z_encoder_class,
        checkpoint_path=args.z_encoder_checkpoint,
        kwargs=z_encoder_kwargs,
        device=torch.device("cuda"),
        cfg=cfg,
    )

    env_kwargs = _json_loads_object(args.isaac_env_kwargs_json, "--isaac-env-kwargs-json")
    env = import_isaac_env(args.isaac_env_module, args.isaac_env_class, env_kwargs)

    planner_map_history = PlannerMapHistory(
        cfg=cfg,
        map_key=args.map_key,
        normalize=args.map_normalize,
        invert=args.map_invert,
    )
    controller = ExternalZTDMPCTestController(
        tdmpc_agent=tdmpc_agent,
        z_encoder=z_encoder,
        cfg=cfg,
        velocity_key=args.velocity_key,
        velocity_format=args.velocity_format,
        planner_map_history=planner_map_history,
        disable_env_dyn_weight=args.disable_env_dyn_weight,
        deterministic=args.deterministic,
    )

    output_dir = None
    csv_path = None
    rollout_records: List[Dict[str, Any]] = []
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "episode_metrics.csv"
        write_episode_csv_header(csv_path)
        meta = {
            "project_root": str(Path(args.project_root).expanduser().resolve()),
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "isaac_env_module": args.isaac_env_module,
            "isaac_env_class": args.isaac_env_class,
            "z_encoder_module": args.z_encoder_module,
            "z_encoder_function": args.z_encoder_function,
            "z_encoder_class": args.z_encoder_class,
            "z_encoder_checkpoint": args.z_encoder_checkpoint,
            "num_agents": num_agents,
            "episode_steps": episode_steps,
            "action_format": args.action_format,
            "velocity_format": args.velocity_format,
            "deterministic": args.deterministic,
            "cfg_overrides": cfg_overrides,
        }
        with (output_dir / "test_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    total_success = 0
    total_collision = 0
    total_reward = 0.0
    total_steps = 0
    t_all0 = time.time()

    try:
        for ep in range(args.num_episodes):
            controller.reset()
            ep_t0 = time.time()
            obs, reset_info = call_reset(env, seed=None if args.seed is None else args.seed + ep)
            agent_obs_list = to_agent_obs_list(obs, num_agents=num_agents)
            if len(agent_obs_list) != num_agents:
                print(f"[WARN] reset obs produced {len(agent_obs_list)} agent obs, expected {num_agents}; using produced length.")
                num_agents_ep = len(agent_obs_list)
            else:
                num_agents_ep = num_agents

            done_flags = np.zeros(num_agents_ep, dtype=bool)
            success_flags = np.zeros(num_agents_ep, dtype=bool)
            collision_flags = np.zeros(num_agents_ep, dtype=bool)
            ep_reward_sum = 0.0
            last_action_norm: Dict[int, np.ndarray] = {}

            for step in range(episode_steps):
                actions_norm = np.zeros((num_agents_ep, int(getattr(cfg, "action_dim", 2))), dtype=np.float32)
                z_cpu_for_log: List[torch.Tensor] = []
                new_traj_flags: List[bool] = []
                record_indices: Dict[int, int] = {}

                for agent_id in range(num_agents_ep):
                    if done_flags[agent_id]:
                        # Normalized stop: v_norm=-1 -> v_real=0, w_norm=0 -> w_real=0.
                        actions_norm[agent_id] = np.array([-1.0, 0.0], dtype=np.float32)
                        continue

                    result = controller.act(
                        obs=agent_obs_list[agent_id],
                        agent_id=agent_id,
                        t0=(step == 0),
                        fallback_action_norm=last_action_norm.get(agent_id),
                        task=None,
                    )
                    actions_norm[agent_id] = result.action_norm
                    last_action_norm[agent_id] = result.action_norm.copy()
                    z_cpu_for_log.append(result.z.detach().cpu())
                    new_traj_flags.append(result.new_traj)

                    if args.save_rollout:
                        record_indices[agent_id] = len(rollout_records)
                        rollout_records.append(
                            {
                                "episode": ep,
                                "step": step,
                                "agent": agent_id,
                                "z": result.z.detach().cpu(),
                                "action_norm": torch.from_numpy(result.action_norm.copy()),
                                "action_real": torch.from_numpy(result.action_real.copy()),
                                "new_traj": bool(result.new_traj),
                            }
                        )

                env_action = action_for_env(actions_norm, cfg, args.action_format)
                if args.squeeze_single_action and num_agents_ep == 1:
                    env_action = env_action[0]

                ret = env.step(env_action)
                next_obs, reward, done, info = unpack_step(ret)
                next_agent_obs_list = to_agent_obs_list(next_obs, num_agents=num_agents_ep)
                if len(next_agent_obs_list) != num_agents_ep:
                    raise RuntimeError(
                        f"env.step returned {len(next_agent_obs_list)} agent obs, expected {num_agents_ep}."
                    )

                for agent_id in range(num_agents_ep):
                    r = reward_for_agent(reward, agent_id)
                    ep_reward_sum += r
                    success = extract_event(next_agent_obs_list[agent_id], info, ["success", "arrive", "arrival", "reached_goal", "is_success"], agent_id)
                    collision = extract_event(next_agent_obs_list[agent_id], info, ["collision", "collide", "crash", "is_collision"], agent_id)
                    d = done_for_agent(done, agent_id) or success or collision
                    done_flags[agent_id] = done_flags[agent_id] or d
                    success_flags[agent_id] = success_flags[agent_id] or success
                    collision_flags[agent_id] = collision_flags[agent_id] or collision

                    if args.save_rollout and agent_id in record_indices:
                        rec = rollout_records[record_indices[agent_id]]
                        rec["reward"] = float(r)
                        rec["done"] = bool(d)
                        rec["success"] = bool(success)
                        rec["collision"] = bool(collision)

                total_steps += 1
                if args.print_every > 0 and (step % args.print_every == 0):
                    print(
                        f"[ep {ep:04d} step {step:04d}] "
                        f"reward_sum={ep_reward_sum:.3f} "
                        f"done={done_flags.sum()}/{num_agents_ep} "
                        f"success={success_flags.sum()} collision={collision_flags.sum()} "
                        f"a_norm[0]={actions_norm[0].round(3).tolist()}"
                    )

                agent_obs_list = next_agent_obs_list
                if bool(done_flags.all()):
                    break

            ep_seconds = time.time() - ep_t0
            ep_success = int(success_flags.sum())
            ep_collision = int(collision_flags.sum())
            ep_done = int(done_flags.sum())
            total_success += ep_success
            total_collision += ep_collision
            total_reward += ep_reward_sum
            row = {
                "episode": ep,
                "steps": step + 1,
                "num_agents": num_agents_ep,
                "reward_sum": round(float(ep_reward_sum), 6),
                "success_count": ep_success,
                "collision_count": ep_collision,
                "done_count": ep_done,
                "seconds": round(float(ep_seconds), 3),
            }
            if csv_path is not None:
                append_episode_csv(csv_path, row)
            print(
                f"[EP DONE] ep={ep} steps={row['steps']} reward={row['reward_sum']} "
                f"success={ep_success}/{num_agents_ep} collision={ep_collision}/{num_agents_ep} "
                f"time={ep_seconds:.2f}s"
            )

    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    elapsed = time.time() - t_all0
    denom = max(1, args.num_episodes * max(1, num_agents))
    summary = {
        "episodes": args.num_episodes,
        "num_agents": num_agents,
        "total_env_steps": total_steps,
        "success_total": int(total_success),
        "collision_total": int(total_collision),
        "success_rate_per_agent_episode": float(total_success / denom),
        "collision_rate_per_agent_episode": float(total_collision / denom),
        "reward_sum_total": float(total_reward),
        "elapsed_seconds": float(elapsed),
    }
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False, indent=2))

    if output_dir is not None:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if args.save_rollout:
            torch.save({"records": rollout_records, "summary": summary}, output_dir / "rollout.pt")
            print(f"Saved rollout to: {output_dir / 'rollout.pt'}")
        print(f"Saved metrics to: {output_dir}")


if __name__ == "__main__":
    main()
