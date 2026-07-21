import numpy as np
import torch
import torch.nn.functional as F

from tdmpc2_common import math
from tdmpc2_common.scale import RunningScale
from tdmpc2_common.world_model import WorldModel

import random
import math as math_common
import time
import copy
# import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from torchviz import make_dot
import io
import cv2
from tools.utils import CostMap
import os


def gaussian_nll_diag(target, mu, logvar, eps=1e-8):
    """
    Diagonal Gaussian negative log-likelihood.
    target/mu/logvar: (..., Z)
    return: (...)  mean over Z  (important for stability)
    """
    # avoid exp under/overflow and divide-by-0
    var = torch.exp(logvar).clamp_min(eps)
    nll = 0.5 * (((target - mu) ** 2) / var + logvar + math_common.log(2.0 * math_common.pi))
    return nll.mean(dim=-1)   # <-- mean, not sum


def _dbg_stats(name, x: torch.Tensor):
    """Print min/max/mean/std and nan/inf flags for a tensor."""
    if x is None:
        print(f"[DBG] {name}: None")
        return
    x_ = x.detach()
    print(f"[DBG] {name}: shape={tuple(x_.shape)} "
          f"min={x_.min().item():.6g} max={x_.max().item():.6g} "
          f"mean={x_.mean().item():.6g} std={x_.std().item():.6g} "
          f"nan={torch.isnan(x_).any().item()} inf={torch.isinf(x_).any().item()}")



def _fig_to_rgb_numpy(fig):
    """matplotlib figure -> HWC uint8"""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(h, w, 3)
    return img

def tb_add_figure_compat(writer, tag, fig, global_step):
    """tensorboardX 有的版本没 add_figure，用 add_image 兜底"""
    if hasattr(writer, "add_figure"):
        writer.add_figure(tag, fig, global_step=global_step)
        plt.close(fig)
    else:
        img = _fig_to_rgb_numpy(fig)  # HWC
        writer.add_image(tag, img, global_step=global_step, dataformats="HWC")
        plt.close(fig)


class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.train_step = 0
		self.device = torch.device('cuda')
		self.model = WorldModel(cfg).to(self.device)
		# 当前 combine 分支是否处于冻结状态
		self.combine_branch_frozen = False
		if self.cfg.update_all and not self.cfg.update_nomodel:
			if self.cfg.use_multi_mlp_enc:
				if self.cfg.use_multi_dyn:
					self.optim = torch.optim.Adam([
						{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
						{'params': self.model._encoder_state.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
						{'params': self.model._dynamics.parameters(), 'lr': self.cfg.lr * self.cfg.dyn_lr_scale},
						{'params': self.model._dynamics_goal.parameters(), 'lr': self.cfg.lr * self.cfg.dyn_lr_scale},
						{'params': self.model._reward.parameters()},
						{'params': self.model._Qs.parameters()},
						{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
						{'params': self.model.action_embedding.parameters() if self.cfg.use_action_embedding else []},
						{
							'params': self.model.action_embedding_q.parameters() if self.cfg.use_action_embedding and self.cfg.diff_action_embedding else []},
						{
							'params': self.model.action_embedding_r.parameters() if self.cfg.use_action_embedding and self.cfg.diff_action_embedding else []},
					], lr=self.cfg.lr)
				else:
					self.optim = torch.optim.Adam([
						{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
						{'params': self.model._encoder_state.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
						{'params': self.model._dynamics.parameters(), 'lr': self.cfg.lr * self.cfg.dyn_lr_scale},
						{'params': self.model._reward.parameters()},
						{'params': self.model._Qs.parameters()},
						{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
						{'params': self.model.action_embedding.parameters() if self.cfg.use_action_embedding else []},
						{
							'params': self.model.action_embedding_q.parameters() if self.cfg.use_action_embedding and self.cfg.diff_action_embedding else []},
						{
							'params': self.model.action_embedding_r.parameters() if self.cfg.use_action_embedding and self.cfg.diff_action_embedding else []},
					], lr=self.cfg.lr)
			else:
				if self.cfg.use_mdn:
					self.optim = torch.optim.Adam([
						{'params': self.model.fe.parameters() if not self.cfg.mlp_obs else self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
						{'params': self.model._dynamics_core.parameters(), 'lr': self.cfg.lr*self.cfg.dyn_lr_scale},
						{'params': self.model._dynamics_fallback.parameters(), 'lr': self.cfg.lr*self.cfg.dyn_lr_scale},
						{'params': self.model._decoder.parameters() if self.cfg.use_decoder else [], 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
						{'params': self.model._reward.parameters()},
						{'params': self.model._Qs.parameters()},
						{'params': self.model.mdn_pi.parameters()},
						{'params': self.model.mdn_mu.parameters()},
						{'params': self.model.mdn_sigma.parameters()},
						{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
					], lr=self.cfg.lr)
				else:
					self.optim = torch.optim.Adam([
						{'params': self.model.fe.parameters() if not self.cfg.mlp_obs else self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
						{'params': self.model._dynamics.parameters(), 'lr': self.cfg.lr*self.cfg.dyn_lr_scale},
						{'params': self.model._decoder.parameters() if self.cfg.use_decoder else [], 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
						{'params': self.model._reward.parameters()},
						{'params': self.model._Qs.parameters()},
						{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
					], lr=self.cfg.lr)

		
		elif self.cfg.update_nomodel:
				self.optim = torch.optim.Adam([
					{'params': self.model.fe.parameters() if not self.cfg.mlp_obs else self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
					{'params': self.model._decoder.parameters() if self.cfg.use_decoder else [], 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
					{'params': self.model._dynamics.parameters(), 'lr': self.cfg.lr * self.cfg.dyn_lr_scale},
					{'params': self.model._reward.parameters()},
					{'params': self.model._Qs.parameters()},
					{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
					], lr=self.cfg.lr)
		else:
			self.enc_dyn_optim = torch.optim.Adam([
				{'params': self.model.fe.parameters() if not self.cfg.mlp_obs else self.model._encoder.parameters()},
				{'params': self.model._dynamics.parameters()},
			], lr=self.cfg.lr*self.cfg.enc_lr_scale)
			self.r_q_optim = torch.optim.Adam([
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
			], lr=self.cfg.lr)

		aux_params = []

		if getattr(self.cfg, 'use_temporal_future_aux', False) and hasattr(self.model, 'temporal_future_predictor'):
			aux_params += list(self.model.temporal_future_predictor.parameters())

		if getattr(self.cfg, 'use_temporal_interp_aux', False) and hasattr(self.model, 'temporal_interp_predictor'):
			aux_params += list(self.model.temporal_interp_predictor.parameters())

		if getattr(self.cfg, 'use_risk_proxy_aux', False) and hasattr(self.model, 'risk_proxy_predictor'):
			aux_params += list(self.model.risk_proxy_predictor.parameters())

		if len(aux_params) > 0 and hasattr(self, 'optim'):
			self.optim.add_param_group({
				'params': aux_params,
				'lr': self.cfg.lr
			})
			print(f"[AUX] auxiliary predictors joined optim | #params={sum(p.numel() for p in aux_params)}")
		
		# Phase B: 把 survival 头加入优化器（之前漏了 → 头一直冻在 zero-init）
		if getattr(self.cfg, 'use_psafe_head', False) and hasattr(self.model, '_collision') and hasattr(self, 'optim'):
			self.optim.add_param_group({'params': self.model._collision.parameters()})
			print(f"[PSAFE] _collision joined optim | #params={sum(p.numel() for p in self.model._collision.parameters())}")
		_opt_ids = {id(p) for g in self.optim.param_groups for p in g['params']}
		
		for _n in ['fe','_dynamics','_reward','_Qs','temporal_future_predictor',
		           'temporal_interp_predictor','risk_proxy_predictor','_collision']:
			_m = getattr(self.model, _n, None)
			if _m is None:
				print(f"[OPTCHK] {_n}: MISSING"); continue
			_ps = list(_m.parameters())
			_in = sum(id(p) in _opt_ids for p in _ps)
			print(f"[OPTCHK] {_n}: {_in}/{len(_ps)} params in optim")
		
		# for name, p in self.model.named_parameters():
		# 	if 'temporal_future_predictor' in name or 'temporal_interp_predictor' in name or 'risk_proxy_predictor' in name or '_collision' in name:
		# 		print(name, p.requires_grad)
		
		# if getattr(self.cfg, 'use_training_only_aux', False):
		# aux_params = self.model.get_aux_training_params()
		# aux_num = sum(p.numel() for p in aux_params)
		# print('[AUX CHECK] aux param num =', aux_num)

		# if hasattr(self, 'optim'):
		# 	total_groups = len(self.optim.param_groups)
		# 	print('[AUX CHECK] optimizer param groups =', total_groups)
		# 	last_group_num = sum(p.numel() for p in self.optim.param_groups[-1]['params'])
		# 	print('[AUX CHECK] last group param num =', last_group_num)
		
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr*self.cfg.pi_lr_scale, eps=1e-5)
		if cfg.check_traj:
			self.check_optim = torch.optim.Adam([
				{'params': self.model._check_enc.parameters()},
				{'params': self.model._check_dynamics.parameters()},
			], lr=self.cfg.lr * self.cfg.enc_lr_scale)

		if cfg.update_sac_pi:
			self.log_alpha = torch.full((), np.log(1.0), requires_grad=True, dtype=torch.float32, device=self.device)
			self.alpha = self.log_alpha.exp().detach()
			self.target_entropy = -np.prod((self.cfg.action_dim,)).item()
			self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
		
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self.action_buffer = [[] for _ in range(self.cfg.num_agent)]
		self.mu_buffer = [[] for _ in range(self.cfg.num_agent)]
		self.std_buffer = [[] for _ in range(self.cfg.num_agent)]
		self.mpc_as_buffer = [None for _ in range(self.cfg.num_agent)]
		self.pi_a_buffer = [None for _ in range(self.cfg.num_agent)]
		self.act_a_buffer = [None for _ in range(self.cfg.num_agent)]
		self.value_buffer = [None for _ in range(self.cfg.num_agent)]
		self._prev_mean = torch.zeros(self.cfg.num_agent, self.cfg.mppi_horizon, self.cfg.action_dim, device=self.device)
		# if self.cfg.no_laser:
		# 	self.obs_fix_laser_dim = 0
		# elif self.cfg.state_mlp:
		# 	self.obs_fix_laser_dim = self.cfg.state_num*2
		# else:
		# 	self.obs_fix_laser_dim = self.cfg.laser_dim
		self.obs_fix_laser_dim = self.cfg.obs_dim
		self.act_len = 1
		self._last_risk_dbg = {}
		self.plan_counter = 0
		if self.cfg.laser_goal_cat:
			self.obs_fix_goal_dim = self.cfg.goal_dim
		else:
			self.obs_fix_goal_dim = 0

	def _apply_combine_freeze_schedule(self):
		"""
		实验 2：
		前若干步冻结 combine 分支，只训练 temporal 分支 + gate。
		"""
		warmup_steps = int(getattr(self.cfg, 'hybrid_freeze_combine_warmup_steps', 0))

		# 不启用 warmup
		if warmup_steps <= 0:
			if self.combine_branch_frozen:
				self.model.set_combine_branch_trainable(True)
				self.combine_branch_frozen = False
			return

		# warmup 期间：冻结 combine 分支
		if self.train_step <= warmup_steps:
			if not self.combine_branch_frozen:
				self.model.set_combine_branch_trainable(False)
				self.combine_branch_frozen = True
		else:
			# warmup 结束：解冻 combine 分支
			if self.combine_branch_frozen:
				self.model.set_combine_branch_trainable(True)
				self.combine_branch_frozen = False

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		try:
			self.model.load_state_dict(state_dict["model"])
		except RuntimeError as exc:
			msg = str(exc)
			known_aux_prefixes = (
				"temporal_interp_predictor.",
				"risk_proxy_predictor.",
			)
			unexpected_keys = [
				k for k in state_dict["model"].keys()
				if any(k.startswith(prefix) for prefix in known_aux_prefixes)
			]
			if "Unexpected key(s) in state_dict" not in msg or not unexpected_keys:
				raise
			print(
				"[WARN] Ignoring training-only auxiliary checkpoint keys while loading TD-MPC2: "
				f"{unexpected_keys}"
			)
			self.model.load_state_dict(state_dict["model"], strict=False)

	@torch.no_grad()
	def reset_action_buffer(self):
		self.action_buffer = [[] for _ in range(self.cfg.num_agent)]
		self.mu_buffer = [[] for _ in range(self.cfg.num_agent)]
		self.std_buffer = [[] for _ in range(self.cfg.num_agent)]
		self.mpc_as_buffer = [None for _ in range(self.cfg.num_agent)]
		self.pi_a_buffer = [None for _ in range(self.cfg.num_agent)]
		self.act_a_buffer = [None for _ in range(self.cfg.num_agent)]
		self.value_buffer = [None for _ in range(self.cfg.num_agent)]

	@torch.no_grad()
	def act(self, obs, agent_id, t0=False, eval_mode=False, task=None, writer=None):
		"""
		Select an action by planning in the latent space of the world model.
		
		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs_tensor = torch.tensor(obs)
		obs_tensor = obs_tensor.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		# z = self.model.encode(obs_tensor, task)
		# print("obs shape: \n", obs.shape)
		if self.cfg.obs_fix:
			print('act obs fix:', obs.shape)
			if self.cfg.tdmpc_rand_xyyaw:
				g_x = obs[-12] * math_common.cos(obs[-11])
				g_y = obs[-12] * math_common.sin(obs[-11])
				gx_trans = obs[-4] + (g_x*obs[-1] - g_y*obs[-2])
				gy_trans = obs[-3] + (g_x*obs[-2] + g_y*obs[-1])
				g_dis_trans = math_common.hypot(gx_trans, gy_trans)
				g_theta_trans = math_common.atan2(gy_trans, gx_trans)
				obs[-12] = g_dis_trans
				obs[-11] = g_theta_trans
				obs[-8] = obs[-4]
				obs[-7] = obs[-3]
				obs[-6] = obs[-2]
				obs[-5] = obs[-1]
				obs = obs[:-4]
			else:
				if self.cfg.normalization_state:
					if self.cfg.use_cos_sin:
						obs[-4] = 0.0
						obs[-3] = 0.0
						obs[-2] = 0.0
						obs[-1] = 1.0
						if self.cfg.use_dxdydtheta:
							# v -7 w -6 0 -5
							dx_0 = obs[-7] / 10.0
							dtheta_0 = obs[-6] / 10.0

							dx_0 = np.clip((dx_0 / self.cfg.max_traj_obs_fix), -1.0, 1.0)
							dtheta_0 = (dtheta_0 + math_common.pi) / (math_common.pi * 2)

							obs[-7] = dx_0
							obs[-6] = 0.0
							obs[-5] = dtheta_0
					else:
						obs[-3] = 0.5
						obs[-2] = 0.5
						obs[-1] = 0.5
				else:
					if self.cfg.use_cos_sin:
						obs[-4] = 0.0
						obs[-3] = 0.0
						obs[-2] = 0.0
						obs[-1] = 1.0
						if self.cfg.use_dxdydtheta:
							# v -7 w -6 0 -5
							dx_0 = obs[-7] / 10.0
							dtheta_0 = obs[-6] / 10.0

							obs[-7] = dx_0
							obs[-6] = 0.0
							obs[-5] = dtheta_0
					else:
						obs[-3] = 0.0
						obs[-2] = 0.0
						obs[-1] = 0.0
		if self.cfg.abandon_yaw:
			obs = obs[:-2]
		# print("tdmpc act obs", obs)
		if self.cfg.use_multi_mlp_enc:
			# need trans to tensor
			if self.cfg.use_obs_multi_enc and self.obs_fix_laser_dim>0:
				laser_z = self.model.encode(torch.from_numpy(obs[:self.obs_fix_laser_dim]).float().cuda(), task)
				goal_z = self.model.encode_goal(torch.from_numpy(obs[self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim]).float().cuda(), task)
				laser_goal = torch.cat([laser_z, goal_z], dim=-1)
			else:
				laser_goal = self.model.encode(torch.from_numpy(obs[:self.obs_fix_laser_dim+self.obs_fix_goal_dim]).float().cuda(), task)
			vw_state = self.model.encode_state(torch.from_numpy(obs[self.obs_fix_laser_dim+self.obs_fix_goal_dim:]).float().cuda(), task)
			z = torch.cat([laser_goal, vw_state], dim=-1)
		else:
			z = self.model.multi_encode(obs, agent_id, False)
		# z = self.model.multi_encode(obs, agent_id, False)
		# print("z shape: \n", z.shape)
		if not self.cfg.mpc:
			# a = self.model.pi(z, task)[int(not eval_mode)][0].unsqueeze(0)
			a = self.model.pi(z, task)[int(not eval_mode)][0].unsqueeze(0)
			print('no mpc', a)
			return a.cpu(), True, None, None, None, None
		
		if self.cfg.use_one:
			print('use one org')
			a, mpc_as, pi_a, act_a, value = self.plan(z, agent_id, t0=t0, eval_mode=eval_mode, task=task, v_real=torch.tensor(obs[-4], dtype=torch.float32, device=self.device), w_real=torch.tensor(obs[-3], dtype=torch.float32, device=self.device), writer=writer, observation=obs)
			
			# print('mpc a', a)
			# print('mpc as', mpc_as)
			
			return a.cpu(), True, mpc_as.cpu(), pi_a.cpu(), act_a.cpu(), value.cpu()
			
		# print("action buffer", self.action_buffer[agent_id])
		if len(self.action_buffer[agent_id]) == 0:
			# print("here")
			new_traj = True
			if self.cfg.mpc:
				a, self.mpc_as_buffer[agent_id], self.pi_a_buffer[agent_id], self.act_a_buffer[agent_id], self.value_buffer[agent_id] = self.plan(z, agent_id, t0=t0, eval_mode=eval_mode, task=task, v_real=torch.tensor(obs[-4], dtype=torch.float32, device=self.device), w_real=torch.tensor(obs[-3], dtype=torch.float32, device=self.device), writer=writer, observation=obs)
			else:	
				a = self.model.pi(z, task)[int(not eval_mode)][0]

			if self.cfg.use_one:
				for i in range(1):
					self.action_buffer[agent_id].append(a[i])
					# self.mu_buffer[agent_id].append(mus[i])
					# self.std_buffer[agent_id].append(stds[i])
				print("now agent ", agent_id, " use_one ", self.cfg.use_one, " horizon ", self.cfg.horizon, " lstm ", self.cfg.lstm_dyn)
			elif self.cfg.random_act_len:
				self.act_len = random.randint(1, self.cfg.rand_horizon)
				for i in range(self.act_len):
					self.action_buffer[agent_id].append(a[i])
					# self.mu_buffer[agent_id].append(mus[i])
					# self.std_buffer[agent_id].append(stds[i])
				print("now agent ", agent_id, " use_random_act ", self.act_len, " horizon ", self.cfg.horizon)
			else:
				for i in range(self.cfg.use_horizon):
					self.action_buffer[agent_id].append(a[i])
					# self.mu_buffer[agent_id].append(mus[i])
					# self.std_buffer[agent_id].append(stds[i])
				print("now agent ", agent_id, " use_one ", self.cfg.use_one, " horizon ", self.cfg.horizon, " use_horizon ", self.cfg.use_horizon)
		else:
			new_traj = False

		action_now = self.action_buffer[agent_id].pop(0).unsqueeze(0)
		# mu_now = self.mu_buffer[agent_id].pop(0)
		# std_now = self.std_buffer[agent_id].pop(0)
		# print(action_now)
		# print(self.mpc_as_buffer[agent_id])
		return action_now.cpu(), new_traj, self.mpc_as_buffer[agent_id].cpu(), self.pi_a_buffer[agent_id].cpu(), self.act_a_buffer[agent_id].cpu(), self.value_buffer[agent_id].cpu()
	
		# return a.cpu()
	@torch.no_grad()
	def reward_s(self, s):
		reward_goal = -s[:, self.obs_fix_laser_dim] * self.cfg.norm_threshold_goal_distance
		done_indices = s[:, self.obs_fix_laser_dim] < (self.cfg.threshold_arrival / self.cfg.norm_threshold_goal_distance) 
		
		even_indices = torch.arange(0, self.obs_fix_laser_dim, 2)
		selected_elements = s[:, even_indices]
		min_collision = selected_elements.min(dim=1).values * self.cfg.state_range
		mask = min_collision < self.cfg.norm_threshold_collision_distance
		reward_collision = - self.cfg.reward_collision * \
			(1 - (min_collision - self.cfg.threshold_collision) / (self.cfg.norm_threshold_collision_distance - self.cfg.threshold_collision))
		reward_collision = reward_collision * mask.float()
		collision_indices = min_collision < self.cfg.threshold_collision

		reward_s = reward_goal + reward_collision
		reward_s[done_indices] = self.cfg.reward_done
		reward_s[collision_indices] = -self.cfg.reward_done

		return reward_s
	
	
	def apply_action_clip(
		self, a_target, v_prev, w_prev,
		a_max, w_max,
		max_linear_vel, max_angular_vel, t0
	):
		"""
		Args:
			a_target: Tensor [..., 2] ∈ [-1, 1], normalized (v, w)
			v_prev: Tensor [...], normalized v ∈ [0, 1]
			w_prev: Tensor [...], normalized w ∈ [0, 1]
			a_max: float, max linear acceleration in m/s²
			w_max: float, max angular acceleration in rad/s²
			dt: float, step duration in s
			max_linear_vel: float, max linear velocity in m/s
			max_angular_vel: float, max angular velocity in rad/s

		Returns:
			a_exec: Tensor [..., 2], normalized velocity after clip ∈ [-1, 1]
		"""

		# === 解码为真实速度单位 ===
		# print('a_traget v_prev w_prev: ', a_target[0], v_prev[0], w_prev[0])
		
		
		v_target_real = ((a_target[:, 0] + 1.) / 2.) * max_linear_vel
		w_target_real = a_target[:, 1] * max_angular_vel  # 反归一化到 [-w_max, w_max]

		if t0 < 1:
			v_prev_real = v_prev * max_linear_vel
			w_prev_real = (w_prev * 2.0 - 1.0) * max_angular_vel
		else:
			v_prev_real = ((v_prev + 1.) / 2.) * max_linear_vel
			w_prev_real = w_prev * max_angular_vel

		# print('v_target_real w_target_real: ', v_target_real[0], w_target_real[0])
		# print('v_prev_real w_prev_real: ', v_prev_real[0], w_prev_real[0])
		# === 限制加速度变化 ===
		v_delta = torch.clamp(v_target_real - v_prev_real, -a_max, a_max)
		w_delta = torch.clamp(w_target_real - w_prev_real, -w_max, w_max)

		v_exec_real = v_prev_real + v_delta
		w_exec_real = w_prev_real + w_delta

		# print('v_exec_real w_exec_real: ', v_exec_real[0], w_exec_real[0])

		# === 重新归一化为 [-1, 1] ===
		v_exec = torch.clamp(v_exec_real * 2. / max_linear_vel - 1., -1.0, 1.0)
		w_exec = torch.clamp(w_exec_real / max_angular_vel, -1.0, 1.0)  # 映射回 [-1, 1]
		
		# print('v_exec w_exec: ', v_exec[0], w_exec[0])

		return torch.stack([v_exec, w_exec], dim=-1)

	def _make_member_cover_idx(self, K: int, M: int, S: int, device):
		"""
		返回:
		member_idx: (S*K,)  每个 (s,k) 对应的 member id
		S_eff: 真实使用的粒子数（可能向上补齐到 M 的倍数）
		member_idx_SK: (S_eff, K) 便于 debug/可视化：每列 k 覆盖 member
		约定：同一行 s 在所有 k 上 member 相同（行语义=member），便于读 U(S,K) heatmap。
		"""
		# 让 S 变成 M 的倍数，避免某些 k 覆盖不全
		if S < M:
			S_eff = M
		else:
			S_eff = ((S + M - 1) // M) * M  # 向上补齐
		# 行索引 -> member（循环覆盖）
		row_member = (torch.arange(S_eff, device=device) % M).view(S_eff, 1)  # (S_eff,1)
		member_idx_SK = row_member.repeat(1, K)                                # (S_eff,K)
		member_idx = member_idx_SK.reshape(-1)                                 # (S_eff*K,)
		return member_idx, S_eff, member_idx_SK


	@torch.no_grad()
	def _estimate_value_risk(self, z0, actions, task, step, v_prev=None, w_prev=None, collect_vis: bool = False, observation_np=None):
		"""
		Risk-aware estimate of value for K action sequences via S stochastic rollouts.

		Returns:
			score: (K,1) higher is better
		"""
		dbg = getattr(self.cfg, "risk_debug", True)
		H, K, A = actions.shape

		S_cfg = getattr(self.cfg, "num_particles", 8)

		# ====== rollout control ======
		rollout_mode = getattr(self.cfg, "dyn_plan_mode", "ts_mean")
		dyn_beta = getattr(self.cfg, "dyn_noise_scale", 0.0)

		# ====== (A) return aggregation: keep it stable/clean ======
		ret_reduce = getattr(self.cfg, "ret_reduce", "mean")  # 建议先固定 "mean"

		# ====== (B) uncertainty risk (the core of B1/B2/B3) ======
		use_unc_risk = getattr(self.cfg, "use_unc_risk", True)
		unc_beta = getattr(self.cfg, "unc_beta", getattr(self.cfg, "unc_penalty", 0.1))  # β
		unc_mode = getattr(self.cfg, "unc_mode", "epistemic")
		unc_reduce = getattr(self.cfg, "unc_reduce", "mean")
		unc_clip = getattr(self.cfg, "unc_clip_max", None)

		unc_risk_type = getattr(self.cfg, "unc_risk_type", "cvar")   # {"mean","mean_var","cvar"}
		unc_cvar_alpha = getattr(self.cfg, "unc_cvar_alpha", 0.9)
		unc_lam = getattr(self.cfg, "unc_lam", 1.0)
		unc_per_step = getattr(self.cfg, "unc_per_step", True)
		unc_sample_by_member = getattr(self.cfg, "unc_sample_by_member", False)

		# ====== Legacy switch: do NOT subtract unc from reward if you want B ======
		# 原来 use_unc_penalty 是 r -= unc_w*u (价值端 shaping)。B方案里要关掉它。
		# 你也可以保留这个开关用于对比实验。
		use_unc_penalty_in_reward = getattr(self.cfg, "use_unc_penalty_in_reward", False)
		unc_w = getattr(self.cfg, "unc_penalty", 0.1)

		# >>> ADD: 可视化/日志开关：即使不用unc_risk，也能算U用于画图
		log_unc_vis = getattr(self.cfg, "log_unc_vis", True)
		need_unc_stats = (use_unc_risk or use_unc_penalty_in_reward or collect_vis or log_unc_vis)

		# 路线1开关
		cover_all_members = bool(getattr(self.cfg, "unc_cover_all_members", True))
		rollout_by_member = bool(getattr(self.cfg, "risk_rollout_by_member", True))  # 强烈建议 True
		rollout_sample_noise = bool(getattr(self.cfg, "risk_rollout_sample_noise", False))  # 可选：是否从 var_sel 采样

		M = getattr(self.cfg, "ensemble_num", None)
		# ============ 构造 member_idx，确定 S ============
		if cover_all_members:
			member_idx, S, member_idx_SK = self._make_member_cover_idx(K=K, M=M, S=S_cfg, device=self.device)
		else:
			S = S_cfg
			member_idx = torch.randint(low=0, high=M, size=(S * K,), device=self.device)


		# ====== helper: risk aggregation over S ======
		def _risk_agg_over_S(x_S_K: torch.Tensor, risk_type: str):
			# x_S_K: (S,K)  bigger = worse (for uncertainty cost)
			if risk_type == "mean":
				return x_S_K.mean(dim=0)  # (K,)
			if risk_type == "mean_var":
				return x_S_K.mean(dim=0) + unc_lam * x_S_K.std(dim=0, unbiased=False)  # (K,)  cost + λ*std
			if risk_type == "cvar":
				# CVaR on cost: take worst alpha% (largest costs)
				n_worst = max(1, int(math_common.ceil(unc_cvar_alpha * S)))
				x_sorted, _ = torch.sort(x_S_K, dim=0, descending=True)  # worst first
				return x_sorted[:n_worst].mean(dim=0)  # (K,)
			raise ValueError(f"unknown unc_risk_type={risk_type}")

		# ====== print header ======
		if dbg and step == 0:
			print("RISK AWARE MODE: S rollout_mode dyn_noise_scale:", S, rollout_mode, dyn_beta)
			print("B-mode: use_unc_risk unc_beta unc_risk_type unc_cvar_alpha unc_lam ret_reduce:",
				use_unc_risk, unc_beta, unc_risk_type, unc_cvar_alpha, unc_lam, ret_reduce)
			print("Legacy shaping: use_unc_penalty_in_reward:", use_unc_penalty_in_reward, "unc_w:", unc_w)

		# normalize z0
		if z0.dim() == 1:
			z0 = z0.unsqueeze(0)
		z0 = z0.to(self.device)
		
		# Expand to (S*K, Z)
		z = z0.repeat(S, 1)  # (S*K, Z)
		print("estimate value risk z shape", z.shape)

		# ---- build fixed ensemble member indices for TS (you already have this) ----
		# member_idx = None
		# if rollout_mode in {"ts", "ts_mean"}:
			
		# 	if M is None:
		# 		mu_all, _ = self.model.next_dist(z[:1], actions[0][:1].repeat(1, 1), task, return_type="all")
		# 		M = mu_all.shape[0]
		# 	idx_s = torch.randint(0, M, (S,), device=self.device)
		# 	member_idx = idx_s[:, None].expand(S, K).reshape(S * K)
		# 	if dbg and step == 0:
		# 		mi = member_idx.view(S, K)
		# 		same_per_particle = (mi == mi[:, :1]).all().item()
		# 		print(f"[DBG] TS member_idx shared_per_particle={same_per_particle} "
		# 			f"unique_members={mi.unique().numel()} / S={S}")
		# elif rollout_mode == "mean":  # Special handling for "mean" mode
		# 	# Create a fixed set of member indices for "mean" mode
		# 	# You can use a simple fixed strategy, e.g., repeating indices or using a deterministic pattern
		# 	member_idx = torch.randint(0, M, (S * K,), device=self.device)  # Generate random indices for members
			
		# 	if dbg and step == 0:
		# 		print(f"[DBG] In 'mean' mode, fixed member_idx generated for S={S}, K={K}")


		# init accumulators
		G = torch.zeros(S * K, 1, device=self.device)
		discount = torch.ones(S * K, 1, device=self.device)
		disc = self.discount[torch.tensor(task, device=self.device)] if self.cfg.multitask else self.discount

		# ====== NEW: uncertainty cost accumulator (S*K,1) ======
		U = torch.zeros(S * K, 1, device=self.device)
		U_epi = torch.zeros(S * K, 1, device=self.device)
		U_ale = torch.zeros(S * K, 1, device=self.device)

		for t in range(H):
			a_t = actions[t].to(self.device)    # (K,A)
			a_t = a_t.repeat(S, 1)              # (S*K,A)

			# reward
			if self.cfg.use_mse_r:
				r = self.model.reward(z, a_t, task)  # (S*K,1)
			else:
				r = math.two_hot_inv(self.model.reward(z, a_t, task), self.cfg)

			# ====== uncertainty measurement ======
			if need_unc_stats:
				# if use_unc_risk or use_unc_penalty_in_reward:
				if unc_sample_by_member and (member_idx is not None):
					# --- B4 stable: sample uncertainty by TS member, rollout can still be mean ---
					if t==0 and step == 0:
						print("=================B4 stable: sample uncertainty by TS member==========")
					# mu_all, _ = self.model.next_dist(z, a_t, task, return_type="all")  # (M, B, Z)
					# M, B, Zdim = mu_all.shape
					# mu_mean = mu_all.mean(dim=0)  # (B,Z)

					# idx = member_idx  # (B,) where B = S*K
					# arange = torch.arange(B, device=mu_all.device)
					# mu_sel = mu_all[idx, arange]  # (B,Z)

					# u = (mu_sel - mu_mean).pow(2).mean(dim=-1, keepdim=True)  # (B,1)
					# if unc_clip is not None:
					# 	u = u.clamp(max=unc_clip)

					# mu_all, logvar_all = self.model.next_dist(z, a_t, task, return_type="all")  # (M,B,Z)
					# M, B, Zdim = mu_all.shape

					# # ---- 1) ensemble mean (for rollout mean / reference) ----
					# mu_mean = mu_all.mean(dim=0)  # (B,Z)

					# # ---- 2) epistemic: Var_m(mu) ----
					# # (mu - mu_mean)^2 averaged over members -> (B,Z)
					# var_epi = (mu_all - mu_mean.unsqueeze(0)).pow(2).mean(dim=0)  # (B,Z)

					# # ---- 3) aleatoric: E_m[var] ----
					# var_all = torch.exp(logvar_all)  # (M,B,Z)
					# var_ale = var_all.mean(dim=0)    # (B,Z)

					# # ---- 4) total variance (diagonal) ----
					# var_total = var_epi + var_ale    # (B,Z)

					# # ---- 5) scalar uncertainty for each sample (B,1) ----
					# # 用 mean 更不依赖 latent_dim 的尺度；也可以用 sum/ sqrt 看你后面 β 怎么调
					# u_epi = var_epi.mean(dim=-1, keepdim=True)     # (B,1)
					# u_ale = var_ale.mean(dim=-1, keepdim=True)     # (B,1)
					# u = var_total.mean(dim=-1, keepdim=True)       # (B,1)


					# # ---- 7) clip (可选) ----
					# if unc_clip is not None:
					# 	u = u.clamp(max=unc_clip)
					# 	u_epi = u_epi.clamp(max=unc_clip)
					# 	u_ale = u_ale.clamp(max=unc_clip)

					# ----- dynamics ensemble stats -----
					mu_all, logvar_all = self.model.next_dist(z, a_t, task, return_type="all")  # (M,B,Z)
					B = mu_all.shape[1]
					arange = torch.arange(B, device=self.device)
					var_all = torch.exp(logvar_all)

					# ensemble mean（用于 epi 计算）
					mu_mean = mu_all.mean(dim=0)  # (B,Z)

					# 路线1：按 member 取 mu_sel/var_sel（每个粒子固定一个 member）
					idx = member_idx  # (B,)
					mu_sel  = mu_all[idx, arange]   # (B,Z)
					var_sel = var_all[idx, arange]  # (B,Z)

					# epi：member mean 的偏离（采样式，与原写法一致但不随机）
					u_epi = (mu_sel - mu_mean).pow(2).mean(dim=-1, keepdim=True)  # (B,1)
					# ale：该 member 预测噪声
					u_ale = var_sel.mean(dim=-1, keepdim=True)                    # (B,1)
					u_all = u_epi + u_ale

					if unc_clip is not None:
						u_all = u_all.clamp(max=unc_clip)
						u_epi = u_epi.clamp(max=unc_clip)
						u_ale = u_ale.clamp(max=unc_clip)

					

				else:
					# fallback to your original uncertainty function (deterministic)
					u_all = self.model.dynamics_uncertainty(
						z, a_t, task,
						mode=unc_mode,
						reduce=unc_reduce,
						clip_max=unc_clip
					) # (S*K,1)

				# 累积 U
				# B-mode: accumulate uncertainty as cost (do NOT touch reward)
				U = U + u_all
				U_epi = U_epi + u_epi
				U_ale = U_ale + u_ale
				
				# legacy shaping: optional for ablation only
				if use_unc_penalty_in_reward:
					r = r - unc_w * u

			# accumulate return
			G = G + discount * r
			# rollout dynamics
			# ----- update z -----
			if rollout_by_member:
				# 每个粒子用自己的 member 动力学走（强烈推荐）
				if rollout_sample_noise:
					eps = torch.randn_like(mu_sel)
					z = mu_sel + eps * torch.sqrt(var_sel.clamp_min(1e-8))
				else:
					z = mu_sel
			else:
				# 退化版：仍然走 mean（不推荐，risk 会弱）
				z = mu_mean
			# z = self.model.sample_next(z, a_t, task, mode=rollout_mode, member_idx=member_idx)
			discount = discount * disc

		# terminal value
		pi_a = self.model.pi(z, task)[1]
		V = self.model.Q(z, pi_a, task, return_type="avg")  # (S*K,1) or (S*K,)

		if V.dim() == 1:
			V = V.unsqueeze(1)

		ret = G + discount * V     # (S*K,1)
		ret = ret.view(S, K)       # (S,K)

		# U -> (S,K)
		U_use = None
		unc_risk_K = None
		U_epi_use = None
		unc_epi_risk_K = None
		U_ale_use = None
		unc_ale_risk_K = None
		U_vis = None
		if need_unc_stats:
			U_use = (U / float(H)).view(S, K) if unc_per_step else U.view(S, K)
			# >>> ADD: 计算每条候选的 unc_risk(K)，即使不用于score也算（用于画图）
			print("mean std over S:", U_use.std(dim=0).mean().item())
			
			unc_risk_K = _risk_agg_over_S(U_use, unc_risk_type)  # (K,)

			U_epi_use = (U_epi / float(H)).view(S, K) if unc_per_step else U_epi.view(S, K)
			unc_epi_risk_K = _risk_agg_over_S(U_epi_use, unc_risk_type)  # (K,)

			U_ale_use = (U_ale / float(H)).view(S, K) if unc_per_step else U_ale.view(S, K)
			unc_ale_risk_K = _risk_agg_over_S(U_ale_use, unc_risk_type)  # (K,)
			
			U_vis = U_use - U_use.mean(dim=1, keepdim=True)


		# ====== return aggregation (keep stable) ======
		if ret_reduce == "mean":
			ret_agg = ret.mean(dim=0)  # (K,)
		elif ret_reduce == "mean_var":
			lam_ret = getattr(self.cfg, "ret_lam", 1.0)
			ret_agg = ret.mean(dim=0) - lam_ret * ret.std(dim=0, unbiased=False)  # 越大越好
		elif ret_reduce == "cvar":
			# 如果你非要 return 也做 CVaR，建议只在 mean rollout 情况下用
			alpha_ret = getattr(self.cfg, "ret_cvar_alpha", 0.9)
			n_worst = max(1, int(math_common.ceil(alpha_ret * S)))
			cost = -ret
			cost_sorted, _ = torch.sort(cost, dim=0, descending=True)
			ret_agg = -cost_sorted[:n_worst].mean(dim=0)
		else:
			raise ValueError(f"unknown ret_reduce={ret_reduce}")

		# dyn_map + w_ale(K)
		dyn_map = None
		if getattr(self.cfg, "use_env_dyn_weight", True):
			dyn_map = self._build_dyn_map_from_observation(observation_np)      # (Himg,Wimg)
			w_ale_K = self._traj_dynamic_weight_K_v2(
							dyn_map_np=dyn_map,          # 你已有的 dyn_map (H,W)
							actions_HK2=actions,         # (H,K,2)
							dyn_thr=getattr(self.cfg, "dyn_thr", 0.15),
							sigma_pix=getattr(self.cfg, "pixel_per_meter", 10.0),
							lookahead_pix=getattr(self.cfg, "dyn_lookahead_pix", 10.0),
							gamma_t=getattr(self.cfg, "dyn_gamma_t", 0.98),
							tau_traj=getattr(self.cfg, "dyn_tau_traj", 0.15),
							tau_global=getattr(self.cfg, "dyn_tau_global", 0.02),
							alpha=getattr(self.cfg, "dyn_alpha", 8.0),
							beta=getattr(self.cfg, "dyn_beta", 6.0),
							w_min=getattr(self.cfg, "w_ale_min", 0.0),
							w_max=getattr(self.cfg, "w_ale_max", 1.0),
						)

			# w_ale_K = self._traj_dynamic_weight_K(dyn_map, actions)             # (K,)
		else:
			w_ale_K = torch.full((K,), 0.5, device=self.device)

		w_epi_K = 1.0 - w_ale_K

		# 融合（线性融合，最稳、最好解释）
		unc_mix_K = w_epi_K * unc_epi_risk_K + w_ale_K * unc_ale_risk_K   # (K,)

		
		# ====== B1/B2/B3 core: risk on uncertainty ======
		if use_unc_risk:
			# unc_risk = _risk_agg_over_S(U_use, unc_risk_type)  # (K,) bigger=worse
			if getattr(self.cfg, "use_all_unc", True):
				if getattr(self.cfg, "use_env_dyn_weight", True):
					score = ret_agg - unc_beta * unc_mix_K
				else:
					score = ret_agg - unc_beta * unc_risk_K              # (K,)
			else:
				score = ret_agg - unc_beta * unc_epi_risk_K              # (K,)
		else:
			score = ret_agg

		# ====== debug print (step==0 only) ======
		if dbg and step == 0:
			_dbg_stats("[DBG] ret(S,K)", ret)
			if use_unc_risk:
				_dbg_stats("[DBG] U(S,K)", U_use)
				print("[DBG] unc_risk stats:",
					"min", unc_risk_K.min().item(),
					"max", unc_risk_K.max().item(),
					"mean", unc_risk_K.mean().item(),
					"std", unc_risk_K.std(unbiased=False).item())
			_dbg_stats("[DBG] score(K,)", score)

		# cache for tensorboard
		self._last_risk_dbg = {
			"ret_mean_mean": ret.mean(dim=0).mean().item(),
			"ret_std_mean": ret.std(dim=0, unbiased=False).mean().item(),
			"ret_min": ret.min().item(),
			"ret_max": ret.max().item(),
		}

		if need_unc_stats and (U_use is not None):
			self._last_risk_dbg.update({
				"unc_mean": U_use.mean().item(),
				"unc_p90": U_use.flatten().quantile(0.9).item(),
				"unc_max": U_use.max().item(),
				"unc_risk_mean": float(unc_risk_K.mean().item()) if unc_risk_K is not None else 0.0,
			})
		
		# --- cache arrays for plotting (overwrite each call) ---
		# >>> ADD: collect_vis 时缓存向量/矩阵给 plan 画图
		
		
		self._last_risk_dbg.update({
			"U_S_K": U_use.detach().float().cpu() if U_use is not None else None,         # (S,K)
			"U_epi_S_K": U_epi_use.detach().float().cpu() if U_epi_use is not None else None,         # (S,K)
			"U_ale_S_K": U_ale_use.detach().float().cpu() if U_ale_use is not None else None,         # (S,K)
			"U_vis": U_vis.detach().float().cpu() if U_vis is not None else None,         # (S,K)
			"unc_risk_K": unc_risk_K.detach().float().cpu() if unc_risk_K is not None else None,  # (K,)
			"ret_agg_K": ret_agg.detach().float().cpu(),                                 # (K,)
			"score_K": score.detach().float().cpu(),                                     # (K,)
			"unc_metric_K": U_use.mean(dim=0).detach().cpu(),# 给 ret_vs_unc/hist 用（通用）
			"unc_epi_risk_K": unc_epi_risk_K.detach().float().cpu() if unc_epi_risk_K is not None else None,  # (K,)
			"unc_ale_risk_K": unc_ale_risk_K.detach().float().cpu() if unc_ale_risk_K is not None else None,  # (K,)
			"unc_epi_metric_K": U_epi_use.mean(dim=0).detach().cpu(),# 给 ret_vs_unc/hist 用（通用）
			"unc_ale_metric_K": U_ale_use.mean(dim=0).detach().cpu(),# 给 ret_vs_unc/hist 用（通用）
			"w_ale_K": w_ale_K.detach(),
			"unc_mix_K": unc_mix_K.detach(),
		})


		return score.unsqueeze(1)

	@torch.no_grad()
	def _compute_unc_dbg(self, z0, actions, task, v_prev=None, w_prev=None):
		"""
		Compute uncertainty debug tensors for a fixed batch of candidate action sequences.

		Args:
			z0: (1,D) or (K,D) latent (we'll treat (1,D) as shared start)
			actions: (H,K,act_dim)
		Returns: dict with CPU tensors:
			U_S_K: (S,K) per-particle unc cost
			unc_mean_K: (K,)
			unc_cvar_K: (K,)
			ret_base_K: (K,)  # OPTIONAL: if you already have it here; otherwise fill in plan()
		"""
		# ---------- IMPORTANT ----------
		# 1) 这里把你 estimate_value_risk 里原本计算:
		#    - ret_S_K (S,K) 或者至少 U_S_K (S,K)
		#    - 然后聚合得到 unc_mean_K / unc_cvar_K
		#    的代码挪进来
		# --------------------------------
		H, K, A = actions.shape

		S = getattr(self.cfg, "num_particles", 8)

		# ====== rollout control ======
		rollout_mode = getattr(self.cfg, "dyn_plan_mode", "ts_mean")
		dyn_beta = getattr(self.cfg, "dyn_noise_scale", 0.0)

		# ====== (A) return aggregation: keep it stable/clean ======
		ret_reduce = getattr(self.cfg, "ret_reduce", "mean")  # 建议先固定 "mean"

		# ====== (B) uncertainty risk (the core of B1/B2/B3) ======
		use_unc_risk = getattr(self.cfg, "use_unc_risk", True)
		unc_beta = getattr(self.cfg, "unc_beta", getattr(self.cfg, "unc_penalty", 0.1))  # β
		unc_mode = getattr(self.cfg, "unc_mode", "epistemic")
		unc_reduce = getattr(self.cfg, "unc_reduce", "mean")
		unc_clip = getattr(self.cfg, "unc_clip_max", None)

		unc_risk_type = getattr(self.cfg, "unc_risk_type", "cvar")   # {"mean","mean_var","cvar"}
		unc_cvar_alpha = getattr(self.cfg, "unc_cvar_alpha", 0.9)
		unc_lam = getattr(self.cfg, "unc_lam", 1.0)
		unc_per_step = getattr(self.cfg, "unc_per_step", True)
		unc_sample_by_member = getattr(self.cfg, "unc_sample_by_member", False)

		# ====== Legacy switch: do NOT subtract unc from reward if you want B ======
		# 原来 use_unc_penalty 是 r -= unc_w*u (价值端 shaping)。B方案里要关掉它。
		# 你也可以保留这个开关用于对比实验。
		use_unc_penalty_in_reward = getattr(self.cfg, "use_unc_penalty_in_reward", False)
		unc_w = getattr(self.cfg, "unc_penalty", 0.1)

		# >>> ADD: 可视化/日志开关：即使不用unc_risk，也能算U用于画图
		log_unc_vis = getattr(self.cfg, "log_unc_vis", True)
		need_unc_stats = (use_unc_risk or use_unc_penalty_in_reward or log_unc_vis)

		# ====== helper: risk aggregation over S ======
		def _risk_agg_over_S(x_S_K: torch.Tensor, risk_type: str):
			# x_S_K: (S,K)  bigger = worse (for uncertainty cost)
			if risk_type == "mean":
				return x_S_K.mean(dim=0)  # (K,)
			if risk_type == "mean_var":
				return x_S_K.mean(dim=0) + unc_lam * x_S_K.std(dim=0, unbiased=False)  # (K,)  cost + λ*std
			if risk_type == "cvar":
				# CVaR on cost: take worst alpha% (largest costs)
				n_worst = max(1, int(math_common.ceil(unc_cvar_alpha * S)))
				x_sorted, _ = torch.sort(x_S_K, dim=0, descending=True)  # worst first
				return x_sorted[:n_worst].mean(dim=0)  # (K,)
			raise ValueError(f"unknown unc_risk_type={risk_type}")

		# normalize z0
		if z0.dim() == 1:
			z0 = z0.unsqueeze(0)
		z0 = z0.to(self.device)
		
		# Expand to (S*K, Z)
		z = z0.repeat(S, 1)  # (S*K, Z)
		print("compute unc dbg z shape", z.shape)

		# ---- build fixed ensemble member indices for TS (you already have this) ----
		member_idx = None
		if rollout_mode in {"ts", "ts_mean"}:
			M = getattr(self.cfg, "ensemble_num", None)
			if M is None:
				mu_all, _ = self.model.next_dist(z[:1], actions[0][:1].repeat(1, 1), task, return_type="all")
				M = mu_all.shape[0]
			idx_s = torch.randint(0, M, (S,), device=self.device)
			member_idx = idx_s[:, None].expand(S, K).reshape(S * K)
			
		elif rollout_mode == "mean":  # Special handling for "mean" mode
			# Create a fixed set of member indices for "mean" mode
			# You can use a simple fixed strategy, e.g., repeating indices or using a deterministic pattern
			M = getattr(self.cfg, "ensemble_num", None)
			member_idx = torch.randint(0, M, (S * K,), device=self.device)  # Generate random indices for members


		# init accumulators
		G = torch.zeros(S * K, 1, device=self.device)
		G_B3 = torch.zeros(S * K, 1, device=self.device)
		discount = torch.ones(S * K, 1, device=self.device)
		disc = self.discount[torch.tensor(task, device=self.device)] if self.cfg.multitask else self.discount

		# ====== NEW: uncertainty cost accumulator (S*K,1) ======
		U = torch.zeros(S * K, 1, device=self.device)

		for t in range(H):
			a_t = actions[t].to(self.device)    # (K,A)
			a_t = a_t.repeat(S, 1)              # (S*K,A)

			# reward
			if self.cfg.use_mse_r:
				r = self.model.reward(z, a_t, task)  # (S*K,1)
				r_B3 = self.model.reward(z, a_t, task)  # (S*K,1)
			else:
				r = math.two_hot_inv(self.model.reward(z, a_t, task), self.cfg)
				r_B3 = math.two_hot_inv(self.model.reward(z, a_t, task), self.cfg)

			# ====== uncertainty measurement ======
			if need_unc_stats:
				# if use_unc_risk or use_unc_penalty_in_reward:
				if unc_sample_by_member and (member_idx is not None):
					# --- B4 stable: sample uncertainty by TS member, rollout can still be mean ---
					
					mu_all, _ = self.model.next_dist(z, a_t, task, return_type="all")  # (M, B, Z)
					M, B, Zdim = mu_all.shape
					mu_mean = mu_all.mean(dim=0)  # (B,Z)

					idx = member_idx  # (B,) where B = S*K
					arange = torch.arange(B, device=mu_all.device)
					mu_sel = mu_all[idx, arange]  # (B,Z)

					u = (mu_sel - mu_mean).pow(2).mean(dim=-1, keepdim=True)  # (B,1)
					if unc_clip is not None:
						u = u.clamp(max=unc_clip)
				else:
					# fallback to your original uncertainty function (deterministic)
					u = self.model.dynamics_uncertainty(
						z, a_t, task,
						mode=unc_mode,
						reduce=unc_reduce,
						clip_max=unc_clip
					) # (S*K,1)


				# B-mode: accumulate uncertainty as cost (do NOT touch reward)
				
				U = U + u

				# legacy shaping: optional for ablation only
				# if use_unc_penalty_in_reward:
				r_B3 = r_B3 - unc_w * u

			# accumulate return
			G = G + discount * r
			G_B3 = G_B3 + discount * r_B3
			# rollout dynamics
			z = self.model.sample_next(z, a_t, task, mode=rollout_mode, member_idx=member_idx)
			discount = discount * disc

		# terminal value
		pi_a = self.model.pi(z, task)[1]
		V = self.model.Q(z, pi_a, task, return_type="avg")  # (S*K,1) or (S*K,)

		if V.dim() == 1:
			V = V.unsqueeze(1)

		ret = G + discount * V     # (S*K,1)
		ret = ret.view(S, K)       # (S,K)

		ret_B3 = G_B3 + discount * V
		ret_B3 = ret_B3.view(S, K)

		# U -> (S,K)
		U_use = None
		unc_risk_K = None
		if need_unc_stats:
			U_use = (U / float(H)).view(S, K) if unc_per_step else U.view(S, K)
			# >>> ADD: 计算每条候选的 unc_risk(K)，即使不用于score也算（用于画图）
			unc_risk_K = _risk_agg_over_S(U_use, unc_risk_type)  # (K,)
		

		# ====== return aggregation (keep stable) ======
		if ret_reduce == "mean":
			ret_agg = ret.mean(dim=0)  # (K,)
			ret_agg_B3 = ret_B3.mean(dim=0)  # (K,)
		elif ret_reduce == "mean_var":
			lam_ret = getattr(self.cfg, "ret_lam", 1.0)
			ret_agg = ret.mean(dim=0) - lam_ret * ret.std(dim=0, unbiased=False)  # 越大越好
			ret_agg_B3 = ret_B3.mean(dim=0) - lam_ret * ret_B3.std(dim=0, unbiased=False)  # 越大越好
		elif ret_reduce == "cvar":
			# 如果你非要 return 也做 CVaR，建议只在 mean rollout 情况下用
			alpha_ret = getattr(self.cfg, "ret_cvar_alpha", 0.9)
			n_worst = max(1, int(math_common.ceil(alpha_ret * S)))
			cost = -ret
			cost_sorted, _ = torch.sort(cost, dim=0, descending=True)
			ret_agg = -cost_sorted[:n_worst].mean(dim=0)

			cost_B3 = -ret_B3
			cost_sorted_B3, _ = torch.sort(cost_B3, dim=0, descending=True)
			ret_agg_B3 = -cost_sorted_B3[:n_worst].mean(dim=0)
		else:
			raise ValueError(f"unknown ret_reduce={ret_reduce}")

		# ====== B1/B2/B3 core: risk on uncertainty ======
		score_B2 = ret_agg
		score_B3 = ret_agg_B3
		score_B4 = ret_agg - unc_beta * unc_risk_K
		


		# === 示例：你最终应该产出 U_S_K (S,K) ===
		# U_S_K = ... (torch.Tensor, device same as model, shape (S,K), >=0)

		# 下面三行是“聚合方式”，可直接用
		U_use = U_use.detach()

		unc_mean_K = U_use.mean(dim=0)  # (K,)

		# CVaR over particles: take top tail (worst alpha)
		# alpha=0.9 -> worst 10%
		tail = max(1, int((1.0 - unc_cvar_alpha) * S + 1e-6))
		unc_sorted, _ = torch.sort(U_use, dim=0, descending=True)  # (S,K)
		unc_cvar_K = unc_sorted[:tail].mean(dim=0)  # (K,)

		unc_dbg = {
			"U_S_K": U_use.detach().cpu(),
			"unc_mean_K": unc_mean_K.detach().cpu(),
			"unc_cvar_K": unc_cvar_K.detach().cpu(),
		}
		return unc_dbg

	def _aligned_eval_scores(self, ret_base_K, unc_mean_K, unc_cvar_K):
		"""
		All inputs are torch CPU or GPU tensors of shape (K,).
		Return dict for exp2/3/4: scores, elites, k_star, plus some metrics.
		"""
		K = ret_base_K.numel()
		E = int(getattr(self.cfg, "num_elites", 64))
		E = min(E, K)

		beta = float(getattr(self.cfg, "unc_beta", 0.1))
		alpha = float(getattr(self.cfg, "unc_cvar_alpha", 0.9))

		# exp2: ensemble-only (no risk)
		score2 = ret_base_K

		# exp3: unc shaping (value端扣 u 的等价对齐版)
		# 你如果 exp3 的定义是 “reward里扣u”，严格版需要在 rollout 里扣；
		# 但为了“对齐对比选择偏好”，score3=ret_base - beta*unc_mean 是最直接、可解释的对齐指标
		score3 = ret_base_K - beta * unc_mean_K

		# exp4: B-mode (CVaR on unc)
		score4 = ret_base_K - beta * unc_cvar_K

		def _topk(score):
			elite = torch.topk(score, k=E, dim=0).indices
			k_star = torch.argmax(score).item()
			return elite, k_star

		elite2, k2 = _topk(score2)
		elite3, k3 = _topk(score3)
		elite4, k4 = _topk(score4)

		def _metrics(elite, k_star, unc):
			k_unc = float(unc[k_star].item())
			k_ret = float(ret_base_K[k_star].item())
			elite_unc_mean = float(unc[elite].mean().item())
			# percentile: fraction of candidates with unc <= k_unc
			unc_pct = float((unc <= unc[k_star]).float().mean().item())
			return dict(k_unc=k_unc, k_ret=k_ret, elite_unc_mean=elite_unc_mean, k_unc_pct=unc_pct)

		out = {
			"exp2": dict(score=score2, elite=elite2, k_star=k2, **_metrics(elite2, k2, unc_mean_K)),
			"exp3": dict(score=score3, elite=elite3, k_star=k3, **_metrics(elite3, k3, unc_mean_K)),
			"exp4": dict(score=score4, elite=elite4, k_star=k4, **_metrics(elite4, k4, unc_cvar_K)),
		}
		return out

	def _tb_log_aligned_eval(self, writer, rid, step_x, ret_base_K, unc_mean_K, unc_cvar_K, aligned):
		# -------- scalars --------
		for name in ["exp2", "exp3", "exp4"]:
			d = aligned[name]
			writer.add_scalar(f"aligned/r{rid}/{name}/k_ret", d["k_ret"], step_x)
			writer.add_scalar(f"aligned/r{rid}/{name}/k_unc", d["k_unc"], step_x)
			writer.add_scalar(f"aligned/r{rid}/{name}/k_unc_pct", d["k_unc_pct"], step_x)
			writer.add_scalar(f"aligned/r{rid}/{name}/elite_unc_mean", d["elite_unc_mean"], step_x)

		# -------- one aligned figure --------
		try:

			x_all = unc_mean_K.detach().cpu().numpy()
			y_all = ret_base_K.detach().cpu().numpy()

			fig = plt.figure(figsize=(5, 5))
			ax = fig.add_subplot(1,1,1)
			ax.scatter(x_all, y_all, s=6, alpha=0.25, label="all(K)")

			# exp2/3 use unc_mean; exp4 use unc_cvar
			k2 = aligned["exp2"]["k_star"]
			k3 = aligned["exp3"]["k_star"]
			k4 = aligned["exp4"]["k_star"]

			ax.scatter([unc_mean_K[k2].item()], [ret_base_K[k2].item()], marker="*", s=160, label="k* exp2")
			ax.scatter([unc_mean_K[k3].item()], [ret_base_K[k3].item()], marker="X", s=120, label="k* exp3")
			ax.scatter([unc_cvar_K[k4].item()], [ret_base_K[k4].item()], marker="P", s=120, label="k* exp4 (cvar)")

			ax.set_xlabel("unc (mean or cvar)  bigger=worse")
			ax.set_ylabel("ret_base(K)  bigger=better")
			ax.set_title("aligned k* comparison (same candidates)")
			ax.legend(loc="best", fontsize=8)

			tb_add_figure_compat(writer, f"fig/planner/r{rid}/ret_vs_unc_aligned_234", fig, step_x)
		except Exception:
			pass

	def _rollout_unicycle_xy(self, actions_HK2: torch.Tensor):
		"""
		actions_HK2: (H,K,2) normalized in [-1,1] with [v_norm, w_norm]
		returns: x_HK, y_HK in meters, both (H,K)
		"""
		a = actions_HK2.detach()
		H, K, _ = a.shape
		
		# denorm
		v = (a[..., 0] + 1.0) * 0.5 * float(self.cfg.max_linear_vel)    # (H,K)
		w = a[..., 1] * float(self.cfg.max_angular_vel)   # (H,K)
		dt = float(getattr(self.cfg, "T", 0.1))

		# theta(t) = sum w*dt
		theta = torch.cumsum(w * dt, dim=0)  # (H,K)

		# integrate x,y (local frame)
		dx = v * dt * torch.cos(theta)
		dy = v * dt * torch.sin(theta)
		x = torch.cumsum(dx, dim=0)
		y = torch.cumsum(dy, dim=0)
		return x, y  # meters

	def _xy_to_pix(self, x_m, y_m):
		"""
		Map: x in [0,R], y in [-R,R]
		Robot frame: x up, y left
		Image: row down, col right
		Returns:
			u (col, x-axis), v (row, y-axis) as numpy arrays with shape (H,K)
		"""
		R = float(self.cfg.laser_range)      # meters
		Himg = int(self.cfg.img_height)       # height (rows)
		Wimg = int(self.cfg.img_width)       # width  (cols), should correspond to 2R
		pix_per_m = float(self.cfg.pixel_per_meter)

		# meters per pixel
		# mpp_x = R / max(Himg - 1, 1)
		# mpp_y = (2 * R) / max(Wimg - 1, 1)

		# robot pixel anchor: bottom center
		row0 = (Himg - 1)
		col0 = (Wimg - 1) * 0.5

		# robot(x up) -> image(row down): subtract
		row = row0 - (x_m * pix_per_m)

		# robot(y left) -> image(col right): subtract

		col = col0 - (y_m * pix_per_m) if self.plan_counter != 40 else col0 + (y_m * pix_per_m)
		# col = col0 - (y_m * pix_per_m)
		# print("row0 col0 pix_per_meter", )

		u = col.detach().cpu().numpy()
		v = row.detach().cpu().numpy()
		return u, v

	def _build_dyn_map_from_observation(self, observation_np):
		"""
		return: dyn_norm (Himg,Wimg) in [0,1], float32
		"""
		Himg = int(getattr(self.cfg, "img_height", 128))
		Wimg = int(getattr(self.cfg, "img_width", 256))

		# T帧 costmap: (T,H,W)
		maps = CostMap(
			observation_list=observation_np[: self.cfg.observation_dim * self.cfg.sample_length],
			laser_dim=int(self.cfg.laser_dim),
			laser_range=float(self.cfg.laser_range),
			img_h=Himg, img_w=Wimg,
			obs_dim=int(self.cfg.observation_dim),
			highlight_iterations=int(getattr(self.cfg, "highlight_iterations", 1)),
			highlight=True, combine=False, batch=False, black_obs=True
		)
		
		# maps shape: (T,H,W) or (1,T,H,W)
		if maps.ndim == 4:
			maps = maps[0]

		# black_obs=True: obstaclkne=0, free=1  -> occupancy=1-map
		occ = 1.0 - maps.astype(np.float32)  # (T,H,W)

		# 动态变化强度：帧间差分
		diff = np.abs(occ[1:] - occ[:-1])    # (T-1,H,W)
		dyn = diff.mean(axis=0)              # (H,W)

		# percentile 拉伸到 [0,1]
		p_lo = float(np.percentile(dyn, getattr(self.cfg, "dynmap_p_lo", 5)))
		p_hi = float(np.percentile(dyn, getattr(self.cfg, "dynmap_p_hi", 95)))
		dyn_norm = (dyn - p_lo) / (p_hi - p_lo + 1e-6)
		dyn_norm = np.clip(dyn_norm, 0.0, 1.0).astype(np.float32)
		return dyn_norm

	def _traj_dynamic_weight_K(self, dyn_map_np, actions_HK2):
		"""
		dyn_map_np: (Himg,Wimg) in [0,1]
		actions_HK2: torch.Tensor or np, (H,K,2)
		return: w_ale_K torch (K,) on self.device
		"""
		if isinstance(actions_HK2, torch.Tensor):
			a = actions_HK2
		else:
			a = torch.from_numpy(actions_HK2).to(self.device)

		# rollout -> pixels
		x, y = self._rollout_unicycle_xy(a)    # (H,K) meters
		px, py = self._xy_to_pix(x, y)         # (H,K) pixels (float)

		# 转换为 torch 张量
		px = torch.from_numpy(px).to(self.device)
		py = torch.from_numpy(py).to(self.device)

		Himg, Wimg = dyn_map_np.shape
		px_i = px.round().long().clamp(0, Wimg - 1)
		py_i = py.round().long().clamp(0, Himg - 1)

		dyn_t = torch.from_numpy(dyn_map_np).to(self.device)  # (Himg,Wimg)
		# 采样 dyn 值：注意 dyn_t[py, px]
		vals = dyn_t[py_i, px_i]   # (H,K)
		exposure_K = vals.mean(dim=0)  # (K,)

		# exposure -> weight
		thr = float(getattr(self.cfg, "dyn_weight_thr", 0.30))
		temp = float(getattr(self.cfg, "dyn_weight_temp", 0.08))
		w_ale_K = torch.sigmoid((exposure_K - thr) / max(temp, 1e-6))  # (K,)
		return w_ale_K

	def _sigmoid_np(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	def _to_numpy(self, x):
		if isinstance(x, torch.Tensor):
			return x.detach().cpu().numpy()
		return np.asarray(x)

	def _traj_dynamic_weight_K_v2(
		self,
		dyn_map_np,                 # (Himg,Wimg) float, 越大越动态（来自多帧差异）
		actions_HK2,                # (H,K,2) torch
		k_star=None,
		dyn_thr=0.15,               # dyn_map 阈值（按你 dyn_map 的尺度调）
		sigma_pix=10.0,             # 距离核宽度：约等于“1m 对应多少像素”（后面给你换算建议）
		lookahead_pix=10.0,         # 朝向采样前瞻距离（像素）
		gamma_t=0.98,               # 时间折扣，后面步影响略小（可设 1.0）
		# gating 参数（默认很保守，你可调更敏感）
		tau_traj=0.15,
		tau_global=0.02,
		alpha=8.0,                  # traj_dyn 影响强度
		beta=6.0,                   # global_dyn 影响强度
		w_min=0.0,
		w_max=1.0,
	):
		"""
		输出:
		w_ale_K: torch (K,)  in [0,1]
		思路:
		- dyn_mask = dyn_map > dyn_thr
		- dist_to_dyn = distanceTransform(1-dyn_mask)  -> 每个像素到最近动态像素距离
		- proximity = exp(-(dist/sigma)^2)             -> 离动态越近越大
		- score_map = max(dyn_map_norm, proximity)     -> 重叠或接近都能触发
		- traj_score = 沿轨迹采样 score_map 并按时间聚合
		- look_score = 沿轨迹朝向前瞻一点采样 score_map
		- traj_dyn = max(traj_score, look_score)
		- global_dyn = dyn_mask.mean()
		- w_ale = sigmoid(alpha*(traj_dyn-tau_traj)+beta*(global_dyn-tau_global))
		"""
		dyn_map = np.asarray(dyn_map_np, dtype=np.float32)
		Himg, Wimg = dyn_map.shape

		# 1) 动态 mask
		dyn_mask = (dyn_map > float(dyn_thr)).astype(np.uint8)  # 1=dynamic

		# 2) 全局动态密度（场景级）
		global_dyn = float(dyn_mask.mean())  # [0,1]，动态越多越大

		# 3) 距离变换：每个像素到最近动态像素的距离（像素）
		# OpenCV distanceTransform: 对非零像素计算到最近零像素距离
		# 想“到动态像素距离”：把动态像素设为0，其它设为1
		inv = (dyn_mask == 0).astype(np.uint8)  # 非动态=1，动态=0
		dist_pix = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)

		# 4) proximity：距离越近越高（不重叠也能有值）
		proximity = np.exp(- (dist_pix / float(sigma_pix)) ** 2)  # (H,W) in (0,1]

		# 5) 把 dyn_map 也做一个 0~1 归一（避免尺度影响 max）
		dm = dyn_map
		dm_min, dm_max = float(np.nanmin(dm)), float(np.nanmax(dm))
		if not np.isfinite(dm_min): dm_min = 0.0
		if not np.isfinite(dm_max): dm_max = dm_min + 1e-6
		dm_norm = (dm - dm_min) / (dm_max - dm_min + 1e-6)
		dm_norm = np.clip(dm_norm, 0.0, 1.0)

		# 6) score_map：重叠(动态强) 或 接近(距离近) 都能触发
		score_map = np.maximum(dm_norm, proximity)  # (H,W)

		# 7) rollout -> px,py
		# 你的 _rollout_unicycle_xy / _xy_to_pix 支持 torch 输入，返回 numpy/torch 都行
		x, y = self._rollout_unicycle_xy(actions_HK2)  # (H,K)
		px, py = self._xy_to_pix(x, y)                 # (H,K)

		px_np = self._to_numpy(px).astype(np.int32)
		py_np = self._to_numpy(py).astype(np.int32)
		px_np = np.clip(px_np, 0, Wimg - 1)
		py_np = np.clip(py_np, 0, Himg - 1)

		H, K = px_np.shape

		# 8) 沿轨迹采样 score_map
		traj_samp = score_map[py_np, px_np]  # (H,K)

		# 时间权重（可选）
		if gamma_t is None or gamma_t >= 0.999:
			w_t = np.ones((H, 1), dtype=np.float32)
		else:
			w_t = (gamma_t ** np.arange(H, dtype=np.float32)).reshape(H, 1)

		traj_score = (traj_samp * w_t).sum(axis=0) / (w_t.sum(axis=0) + 1e-6)  # (K,)

		# 9) look-ahead：轨迹朝向前方一点（用像素差分估计 heading）
		# 用 (t) 到 (t+1) 的位移方向作为 heading，然后往前 lookahead_pix 采样
		dx = np.diff(px_np, axis=0, prepend=px_np[:1, :]).astype(np.float32)
		dy = np.diff(py_np, axis=0, prepend=py_np[:1, :]).astype(np.float32)
		norm = np.sqrt(dx*dx + dy*dy) + 1e-6
		ux = dx / norm
		uy = dy / norm

		px_look = (px_np + ux * float(lookahead_pix)).round().astype(np.int32)
		py_look = (py_np + uy * float(lookahead_pix)).round().astype(np.int32)
		px_look = np.clip(px_look, 0, Wimg - 1)
		py_look = np.clip(py_look, 0, Himg - 1)

		look_samp = score_map[py_look, px_look]  # (H,K)
		look_score = (look_samp * w_t).sum(axis=0) / (w_t.sum(axis=0) + 1e-6)  # (K,)

		# 10) 轨迹动态接近度：取 max，让“贴近”或“朝向”任一触发都算
		traj_dyn = np.maximum(traj_score, look_score)  # (K,)

		# 11) gating：同时考虑 traj_dyn + global_dyn
		z = float(alpha) * (traj_dyn - float(tau_traj)) + float(beta) * (global_dyn - float(tau_global))
		w_ale = self._sigmoid_np(z)
		w_ale = np.clip(w_ale, float(w_min), float(w_max)).astype(np.float32)  # (K,)

		# 返回 torch
		return torch.from_numpy(w_ale).to(device=actions_HK2.device)

	def _make_costmap_bg(self, observation_np):
		"""
		返回 (Himg,Wimg) 的灰度背景，值域建议 [0,1]，1=free, 0=obstacle(黑)
		"""
		Himg = int(self.cfg.img_height)
		Wimg = int(self.cfg.img_width)
		print("make costmap")
		# 1) try your CostMap
		# try:
		print("try to make now")
		bg = CostMap(
			observation_list=observation_np[:self.cfg.observation_dim*self.cfg.sample_length],
			laser_dim=int(self.cfg.laser_dim),              # 180
			laser_range=float(self.cfg.laser_range),
			img_h=Himg, img_w=Wimg,
			obs_dim=int(self.cfg.observation_dim),          # 你要保证 >= laser_dim+3
			highlight_iterations=int(getattr(self.cfg, "highlight_iterations", 1)),
			highlight=True, combine=True, batch=False, black_obs=True
		)
		print("use costmap !!!!!")
		# combine=True -> (1,H,W) or (H,W)
		lidar_ego_image = cv2.cvtColor(bg[0].astype(np.float32), cv2.COLOR_GRAY2RGB)
		cv2.putText(lidar_ego_image, str(time.time()), (64,64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
		cv2.imshow('lidar_ego_net_dbg', lidar_ego_image)
		cv2.waitKey(10)
		if bg.ndim == 3:
			bg = bg[0]
		return bg.astype(np.float32)
		# except Exception:
		# 	pass
		
		print("use last map!!!")
		# 2) fallback: use last-frame lidar only
		laser_dim = int(self.cfg.laser_dim)
		hist_len = int(getattr(self.cfg, "laser_hist", 8))
		lidar = observation_np[:laser_dim * hist_len].reshape(hist_len, laser_dim)
		r = lidar[-1]  # last frame ranges (normalized or meters? 你需要确保这里是 meters)
		# 如果 r 是归一化，先反归一化
		if getattr(self.cfg, "lidar_is_normalized", True):
			r = r * float(self.cfg.laser_range)

		# simple rasterization
		bg = np.ones((Himg, Wimg), dtype=np.float32)
		R = float(self.cfg.laser_range)
		angles = np.linspace(-np.pi/2, np.pi/2, laser_dim)  # 这里按你的雷达FOV改
		xs = r * np.cos(angles)
		ys = r * np.sin(angles)

		# map to pixels
		mpp = (2*R) / max(Wimg-1, 1)
		cx, cy = (Wimg-1)/2, (Himg-1)/2
		px = (cx + xs/mpp).astype(np.int32)
		py = (cy - ys/mpp).astype(np.int32)
		ok = (px>=0)&(px<Wimg)&(py>=0)&(py<Himg)
		bg[py[ok], px[ok]] = 0.0
		return bg

	def _fig_to_bgr(self, fig):
		"""
		将 matplotlib figure 渲染成 OpenCV 可用的 BGR 图像 (uint8)
		"""
		# 确保有 canvas（某些后端下 fig.canvas 可能不完整）
		try:
			fig.canvas.draw()
			w, h = fig.canvas.get_width_height()
			# 新版 matplotlib 推荐 buffer_rgba
			buf = np.asarray(fig.canvas.buffer_rgba())  # (h,w,4) RGBA
			img_rgb = buf[..., :3]  # RGB
		except Exception:
			# 兜底：强制使用 Agg canvas
			from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
			canvas = FigureCanvas(fig)
			canvas.draw()
			w, h = canvas.get_width_height()
			img_rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

		img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
		return img_bgr


	def _imshow_realtime(self, win_name, img_bgr, wait_ms=1, put_ts=True, scale=1.0):
		"""
		OpenCV 实时显示，支持缩放和时间戳
		"""
		if img_bgr is None:
			return
		show = img_bgr
		if scale is not None and abs(scale - 1.0) > 1e-6:
			show = cv2.resize(show, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

		if put_ts:
			cv2.putText(show, f"{time.time():.3f}", (16, 28),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.imshow(win_name, show)
		cv2.waitKey(int(wait_ms))

	def _tb_fig_traj_cluster(self,
                         observation_np,
                         actions_HK2,
                         metric_K,
                         metric_name,
                         elite_idxs=None,
                         k_star=None,
                         maxK=256,
                         cmap_name="plasma",          # turbo:red green plasma:red purple inferno
                         line_alpha=0.5,            # 叠加后像热力图的关键：透明度别太高
                         line_width=1.5,
                         bg_alpha=0.55,
                         kstar_color="green",
                         kstar_width=2.0,
						 iter_num=0,
						 # ====== NEW: realtime display options ======
                         realtime_show=False,
                         rt_win_name=None,
                         rt_wait_ms=1,
                         rt_put_ts=True,
                         rt_scale=1.0):
		"""
		画“轨迹簇热力图”：
		- 每条 candidate 轨迹整条线按 metric_K[k] 着色
		- k* 用固定颜色高亮覆盖画一遍
		inputs:
		actions_HK2: np/torch, shape (H, K, 2)
		metric_K: torch.Tensor, shape (K,)
		"""

		# --------- basic shapes ----------
		H, K, _ = actions_HK2.shape

		# --------- subsample K but ALWAYS keep k_star ----------
		K_plot = min(K, int(maxK))

		if K_plot < K:
			# random choose + force include k_star
			idx = np.random.choice(K, size=K_plot, replace=False)
			if k_star is not None and (k_star >= 0) and (k_star < K):
				if (k_star not in idx):
					idx[0] = int(k_star)  # force include
			idx = np.unique(idx)
			# 如果 unique 后数量变少，补齐
			if idx.size < K_plot:
				extra = np.setdiff1d(np.arange(K), idx)
				need = K_plot - idx.size
				if extra.size > 0 and need > 0:
					add = np.random.choice(extra, size=min(need, extra.size), replace=False)
					idx = np.concatenate([idx, add])
			idx = np.sort(idx)

			a = actions_HK2[:, idx]                  # (H, Kplot, 2)
			m = metric_K[idx]                        # (Kplot,)
			k_star_in = None
			if k_star is not None:
				hit = np.where(idx == int(k_star))[0]
				k_star_in = int(hit[0]) if hit.size > 0 else None
		else:
			a = actions_HK2
			m = metric_K
			k_star_in = int(k_star) if (k_star is not None) else None

		# --------- background costmap ----------
		print("here1")
		bg = self._make_costmap_bg(observation_np)   # (Himg, Wimg) gray map
		print("here2")
		# --------- rollout to pixels ----------
		# x,y: (H, Kplot) in meters
		x, y = self._rollout_unicycle_xy(a)
		# px,py: (H, Kplot) in pixels
		px, py = self._xy_to_pix(x, y)

		# --------- metric -> color mapping (NO per-frame fake colorbar) ----------
		m_np = m.detach().float().cpu().numpy()
		# 避免全相同导致除0
		m_min = float(np.nanmin(m_np))
		m_max = float(np.nanmax(m_np))
		if not np.isfinite(m_min): m_min = 0.0
		if not np.isfinite(m_max): m_max = 1.0
		if m_max <= m_min + 1e-12:
			m_max = m_min + 1e-12

		# print("m_np before norm", m_np)
		m_np_norm = (m_np - m_min) / (m_max - m_min + 1e-6)
		# print("m_np after norm", m_np_norm)
		norm = Normalize(vmin=m_min, vmax=m_max)
		cmap = cm.get_cmap(cmap_name)
		# print("m np", norm(m_np))
		
		# 每条轨迹一个颜色（整条线同色）
		# m_enhanced = np.log1p(norm(m_np) * 9) / np.log1p(9) 
		colors = cmap(norm(m_np))  # (Kplot, 4)
		# colors = cmap(m_np_norm)  # (Kplot, 4)

		# --------- build line segments for LineCollection ----------
		# segments: list of (H,2)
		segments = []
		for k in range(px.shape[1]):
			seg = np.stack([px[:, k], py[:, k]], axis=1)  # (H,2)
			segments.append(seg)

		fig = plt.figure(figsize=(5.8, 4.8))
		ax = fig.add_subplot(1, 1, 1)

		ax.imshow(bg, cmap="gray", origin="upper", alpha=bg_alpha)
		ax.set_title(f"traj cluster colored by {metric_name} in it{iter_num} (heat-style)")
		ax.set_xlabel("x(pixel)")
		ax.set_ylabel("y(pixel)")

		# 主体：彩色半透明轨迹簇（形成“热力扇形”）
		lc = LineCollection(segments,
							colors=colors,
							linewidths=line_width,
							alpha=line_alpha)
		ax.add_collection(lc)

		# 高亮最终选中的轨迹 k*
		if k_star_in is not None and 0 <= k_star_in < px.shape[1]:
			ax.plot(px[:, k_star_in], py[:, k_star_in],
					color=kstar_color, linewidth=kstar_width, alpha=0.95)

		# 让视野自动贴合背景大小（避免 add_collection 后坐标不更新）
		ax.set_xlim(0, bg.shape[1] - 1)
		ax.set_ylim(bg.shape[0] - 1, 0)

		# 正确的 colorbar（对应 metric 的真实数值范围）
		sm = cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])  # for matplotlib compatibility
		cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label(metric_name)

		# ====== NEW: realtime show via OpenCV ======
		if realtime_show:
			try:
				win = rt_win_name or f"traj_dbg/{metric_name}"
				img_bgr = self._fig_to_bgr(fig)
				self._imshow_realtime(win, img_bgr, wait_ms=rt_wait_ms,
									put_ts=rt_put_ts, scale=rt_scale)
			except Exception:
				# 没有 GUI / 远程环境等情况下不让它影响主流程
				pass

		return fig



	def _tb_fig_traj_heatmap(self, observation_np, actions_HK2, weight_K, title, maxK=256):
		"""
		weight_K: torch (K,) 例如 score_K / ret_agg_K / -unc / mppi权重扩展到K 等
		把每条轨迹每个点按 weight 累加到栅格 -> 热力图
		"""
		Himg = int(self.cfg.img_height)
		Wimg = int(self.cfg.img_width)

		H, K, _ = actions_HK2.shape
		K_plot = min(K, maxK)
		if K_plot < K:
			idx = np.random.choice(K, size=K_plot, replace=False)
			a = actions_HK2[:, idx]
			w = weight_K[idx]
		else:
			a = actions_HK2
			w = weight_K

		bg = self._make_costmap_bg(observation_np)
		x, y = self._rollout_unicycle_xy(a)
		px, py = self._xy_to_pix(x, y)

		w_np = w.detach().cpu().numpy()
		# normalize weights to [0,1] for visualization
		w_min, w_max = float(np.nanmin(w_np)), float(np.nanmax(w_np))
		denom = (w_max - w_min) if (w_max > w_min) else 1.0
		w01 = (w_np - w_min) / denom

		heat = np.zeros((Himg, Wimg), dtype=np.float32)
		for k in range(px.shape[1]):
			xs = np.clip(px[:, k].astype(np.int32), 0, Wimg-1)
			ys = np.clip(py[:, k].astype(np.int32), 0, Himg-1)
			heat[ys, xs] += float(w01[k])

		fig = plt.figure(figsize=(5.5, 5.0))
		ax = fig.add_subplot(1,1,1)
		ax.imshow(bg, cmap="gray", origin="upper")
		im = ax.imshow(heat, alpha=0.65, origin="upper")
		ax.set_title(title)
		ax.set_xlabel("x(pixel)"); ax.set_ylabel("y(pixel)")
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		return fig

	def _tb_fig_w_ale_hist(self, w_ale_K, title="w_ale_K hist", bins=30):
		w = w_ale_K.detach().float().cpu().numpy().reshape(-1)

		fig = plt.figure(figsize=(5.0, 3.0))
		ax = fig.add_subplot(1, 1, 1)
		ax.hist(w, bins=bins)
		ax.set_title(title)
		ax.set_xlabel("w_ale")
		ax.set_ylabel("count")

		# 额外：打印/显示一些统计更直观
		w_min, w_max, w_mean, w_std = np.min(w), np.max(w), np.mean(w), np.std(w)
		ax.text(0.02, 0.98,
				f"min={w_min:.3g}\nmax={w_max:.3g}\nmean={w_mean:.3g}\nstd={w_std:.3g}",
				transform=ax.transAxes, va="top", ha="left", fontsize=9)

		fig.tight_layout()
		return fig

	def _tb_fig_unc_epi_vs_ale_scatter(
		self,
		unc_epi_K,
		unc_ale_K,
		w_ale_K,
		elite_idxs=None,
		k_star=None,
		title="unc_epi_K vs unc_ale_K (colored by w_ale_K)",
		maxK=256,
		cmap_name="viridis"
	):
		# to numpy
		epi = unc_epi_K.detach().float().cpu().numpy().reshape(-1)
		ale = unc_ale_K.detach().float().cpu().numpy().reshape(-1)
		w   = w_ale_K.detach().float().cpu().numpy().reshape(-1)

		K = epi.shape[0]
		K_plot = min(K, int(maxK))

		# subsample but keep elites / k*
		if K_plot < K:
			idx = np.random.choice(K, size=K_plot, replace=False)

			if elite_idxs is not None:
				elite_list = elite_idxs.detach().long().cpu().numpy().tolist()
				idx = np.unique(np.concatenate([idx, np.array(elite_list, dtype=np.int64)]))

			if k_star is not None and 0 <= int(k_star) < K:
				idx = np.unique(np.concatenate([idx, np.array([int(k_star)], dtype=np.int64)]))

			# 如果超了，裁掉（优先保留 elites/k*）
			if idx.size > K_plot:
				# 简单做法：随机裁掉，但保证 k* 和 elites 在
				must = set()
				if elite_idxs is not None:
					must |= set(elite_idxs.detach().long().cpu().numpy().tolist())
				if k_star is not None:
					must.add(int(k_star))

				idx_must = np.array(sorted(list(must)), dtype=np.int64)
				idx_rest = np.setdiff1d(idx, idx_must)
				need = max(0, K_plot - idx_must.size)
				if need > 0 and idx_rest.size > 0:
					idx_rest = np.random.choice(idx_rest, size=min(need, idx_rest.size), replace=False)
				idx = np.sort(np.unique(np.concatenate([idx_must, idx_rest])))
			else:
				idx = np.sort(idx)
		else:
			idx = np.arange(K)

		epi_p = epi[idx]
		ale_p = ale[idx]
		w_p   = w[idx]

		fig = plt.figure(figsize=(5.0, 4.2))
		ax = fig.add_subplot(1, 1, 1)

		norm = Normalize(vmin=float(np.min(w_p)), vmax=float(np.max(w_p)) if np.max(w_p) > np.min(w_p) else float(np.min(w_p)+1e-6))
		cmap = cm.get_cmap(cmap_name)

		sc = ax.scatter(epi_p, ale_p, c=w_p, cmap=cmap, norm=norm, s=14, alpha=0.7)

		# elites overlay
		if elite_idxs is not None:
			elite_set = set(elite_idxs.detach().long().cpu().numpy().tolist())
			mask_elite = np.array([int(k) in elite_set for k in idx], dtype=bool)
			if mask_elite.any():
				ax.scatter(epi_p[mask_elite], ale_p[mask_elite],
						s=40, facecolors="none", edgecolors="orange", linewidths=1.3, alpha=0.95)

		# k* overlay
		if k_star is not None:
			k_star = int(k_star)
			hit = np.where(idx == k_star)[0]
			if hit.size > 0:
				j = int(hit[0])
				ax.scatter([epi_p[j]], [ale_p[j]], s=120, marker="*", edgecolors="k", facecolors="red", linewidths=1.0, alpha=0.95)

		# correlation hint
		if epi_p.size >= 3:
			corr = np.corrcoef(epi_p, ale_p)[0, 1]
			ax.text(0.02, 0.98, f"corr={corr:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=10)

		ax.set_title(title)
		ax.set_xlabel("unc_epi_K")
		ax.set_ylabel("unc_ale_K")
		cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label("w_ale_K")

		fig.tight_layout()
		return fig



	@torch.no_grad()
	def _estimate_value(self, z, actions, task, step, v_prev=None, w_prev=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		# actions_embed = self.model.action_encode(actions, 'd')
		# actions_embed_r = self.model.action_encode(actions,'r')
		# v_prev, w_prev = actions[0][:, 0], actions[0][:, 1]
		smooth_penalty = torch.zeros(actions.shape[1], device=actions.device)  # [N]
		a_prev = actions[0]
		use_psafe = getattr(self.cfg, 'use_psafe', False) and getattr(self.model, 'use_psafe_head', False)
		if use_psafe:
			K_ = actions.shape[1]
			log_surv = torch.zeros(K_, 1, device=actions.device)
			sum_p    = torch.zeros(K_, 1, device=actions.device)
			max_p    = torch.zeros(K_, 1, device=actions.device)	

		dbg = getattr(self.cfg, "risk_debug", False)  # risk/base 共用一个开关

		for t in range(self.cfg.mppi_horizon):
			if self.cfg.use_dec_reward:
				s_dec = self.model.decode(z)
				reward = self.reward_s(s_dec)
			else:
				if self.cfg.use_mse_r:
					if self.cfg.estimate_clip:
						a_exec = self.apply_action_clip(actions[t], v_prev, w_prev, self.cfg.max_linear_acc, self.cfg.max_angular_acc, self.cfg.max_linear_vel, self.cfg.max_angular_vel, t0=t)
						reward = self.model.reward(z, a_exec, task)
					else:
						reward = self.model.reward(z, actions[t], task)
				else:
					reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
				if self.cfg.reward_model_reduce:
					reward[(reward > 0.9) | (reward < -0.9)] *= self.cfg.reward_done
					
			# ====== reward 已经算完，这里开始加 debug（放在 dynamics 更新前）======
			if dbg and t == 0 and step == 0:
				print("[DBG][base] task=", task, "mppi_horizon=", self.cfg.mppi_horizon)
				print(_dbg_stats("[DBG][base] z(t=0) before dyn", z))
				# actions[t] 可能是 (K,A)
				print(_dbg_stats("[DBG][base] a(t=0)", actions[t]))
				print(_dbg_stats("[DBG][base] reward(t=0)", reward))
			# =============================================================
			if use_psafe:
				# —— 诊断：collision 输入 z 在候选间是否分叉 ——
				if not getattr(self, '_zin_dbg', False):
					self._zin_dbg = True
					zc = z.reshape(z.shape[0], -1)
					print(f"[PSAFE-IN] t={t} K={z.shape[0]} | z(before-next) per-cand std={zc.std(0).mean().item():.6f} "
						f"| estimate_clip={self.cfg.estimate_clip}")
				if not getattr(self, '_adbg', False):
					self._adbg = True
					a_t = actions[t].reshape(actions[t].shape[0], -1)   # [K, action_dim]
					print(f"[ACT] t={t} | action per-cand std={a_t.std(0).mean().item():.6f} "
						f"| a_min={a_t.min().item():.3f} a_max={a_t.max().item():.3f}")
					# 顺便看 z 和 a 的维度比
					print(f"[DIM] z_dim={z.shape[-1]} action_dim={actions[t].shape[-1]}")

			if self.cfg.lstm_dyn:
				z, h = self.model.lstm_next(z, actions[t]) if t==0 else self.model.lstm_next(z, actions[t], h)
			elif self.cfg.use_multi_mlp_enc and not self.cfg.obs_state_cat:
			
				vw_state_z = self.model.next(z[:, self.cfg.latent_dim:], actions[t], task)
				z[:, self.cfg.latent_dim:] = vw_state_z

				if self.cfg.use_multi_sep_dyn:
					laser_goal_z = self.model.next_obs_sep(z[:, :self.cfg.latent_dim], actions[t], task)
					z[:, :self.cfg.latent_dim] = laser_goal_z

				if self.cfg.use_multi_dyn:
					laser_goal_z = self.model.next_goal(z)
					z[:, :self.cfg.latent_dim] = laser_goal_z
			else:
				if self.cfg.estimate_clip:
					z = self.model.next(z, a_exec, task, mode='mean')	
					v_prev, w_prev = a_exec[:, 0], a_exec[:, 1]
				else:
					z = self.model.next(z, actions[t], task, mode='mean')
				# —— 诊断：推进后 z 是否还分叉 ——
				if not getattr(self, '_zout_dbg', False):
					self._zout_dbg = True
					zc = z.reshape(z.shape[0], -1)
					print(f"[PSAFE-OUT] t={t} | z(after-next mean) per-cand std={zc.std(0).mean().item():.6f}")

			if use_psafe:
				_pcap = getattr(self.cfg, 'psafe_p_cap', 0.9)					
				p_unsafe = self.model.collision_prob(z, actions[t], task).clamp(1e-4, _pcap)
				log_surv = log_surv + torch.log1p(-p_unsafe)
				sum_p    = sum_p + p_unsafe
				max_p    = torch.maximum(max_p, p_unsafe)

			# ====== dynamics 更新完 z 后的 debug ======
			if dbg and t == 0 and step == 0:
				print(_dbg_stats("[DBG][base] z(t=0) after dyn", z))
			# ==========================================

			G += discount * reward
			discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			if t > 0 and self.cfg.use_smooth_penalty:
				smooth_penalty += ((actions[t] - a_prev) ** 2).sum(dim=-1)
			a_prev = actions[t]
		
		# ====== terminal debug（循环结束后）======
		if dbg and step == 0:
			pi_a = self.model.pi(z, task)[1]
			Qterm = self.model.Q(z, pi_a, task, return_type='avg')
			print(_dbg_stats("[DBG][base] terminal z", z))
			print(_dbg_stats("[DBG][base] terminal pi_a", pi_a))
			print(_dbg_stats("[DBG][base] terminal Q", Qterm))
			print(_dbg_stats("[DBG][base] G", G))
			print("[DBG][base] discount_end=", float(discount if not torch.is_tensor(discount) else discount.mean().item()))
		# =========================================
		
		if use_psafe and not getattr(self, '_sens_dbg', False):
			self._sens_dbg = True
			z0 = z[:1]                                  # 取一个候选的 z
			a_lo = torch.tensor([[-1., -1.]], device=z.device)
			a_hi = torch.tensor([[ 1.,  1.]], device=z.device)
			p_lo = self.model.collision_prob(z0, a_lo, task)
			p_hi = self.model.collision_prob(z0, a_hi, task)
			print(f"[SENS-HEAD] same z, a=(-1,-1) p={p_lo.item():.5f} | a=(1,1) p={p_hi.item():.5f} "
				f"| diff={abs(p_hi-p_lo).item():.5f}")
			# 再测 dynamics 对 action 的敏感度
			zc0 = z0.clone()
			znext_lo = self.model.next(zc0, a_lo, task)
			znext_hi = self.model.next(zc0, a_hi, task)
			print(f"[SENS-DYN] same z, next z diff (a_lo vs a_hi) std={(znext_hi-znext_lo).std().item():.6f}")
			
		if use_psafe:
			if getattr(self.cfg, 'psafe_additive_ablation', False):
				psafe_raw = -sum_p                       # 加性
			else:
				psafe_raw = log_surv                     # 乘性 log-survival
			psafe_raw = psafe_raw - getattr(self.cfg, 'psafe_tail_beta', 0.0) * max_p
			if getattr(self.cfg, 'psafe_zscore', True):
				_mu = psafe_raw.mean()
				_sd = psafe_raw.std().clamp_min(1e-6)
				psafe_raw = (psafe_raw - _mu) / _sd       # 跨 K 候选标准化，量纲无关、压住爆炸
			# if use_psafe and not getattr(self, '_psafe_dbg_done', False):
			# 	self._psafe_dbg_done = True
			print(f"[PSAFE-DBG] BLOCK ENTERED | "
					f"log_surv std={float(log_surv.std()):.5f} mean={float(log_surv.mean()):.5f} | "
					f"sum_p std={float(sum_p.std()):.5f} | "
					f"psafe_raw std(beforeZ)={float(psafe_raw.std()):.5f}")
			G = G + self.cfg.psafe_lambda * psafe_raw

		if self.cfg.estimate_clip:
			pi_action = self.model.pi(z, task)[1]
			a_exec = self.apply_action_clip(pi_action, v_prev, w_prev, self.cfg.max_linear_acc, self.cfg.max_angular_acc, self.cfg.max_linear_vel, self.cfg.max_angular_vel, t0=2)
			return G + discount * self.cfg.q_discount_scale * self.model.Q(z, a_exec, task, return_type='avg')
		elif self.cfg.use_smooth_penalty:
			return G + discount * self.cfg.q_discount_scale * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg'), smooth_penalty
		else:
			return G + discount * self.cfg.q_discount_scale * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	def _tb_fig_ret_vs_unc(self, x_unc_K: torch.Tensor, y_ret_K: torch.Tensor,
						elite_idxs: torch.Tensor = None,
						k_star: int = None,
						title: str = "ret vs unc (candidate-wise)",
						maxK: int = 256):
		"""
		x_unc_K: (K,) bigger=worse
		y_ret_K: (K,) bigger=better
		elite_idxs: (E,) indices in [0, K)
		k_star: int, selected candidate index in [0, K)
		"""
		# to cpu numpy
		x = x_unc_K.detach().float().flatten().cpu().numpy()
		y = y_ret_K.detach().float().flatten().cpu().numpy()
		K = x.shape[0]

		# background sampling (but ALWAYS keep elites and k*)
		if (maxK is not None) and (K > maxK):
			bg_idx = np.random.choice(K, size=maxK, replace=False)
		else:
			bg_idx = np.arange(K)

		elite_np = None
		if elite_idxs is not None:
			elite_np = elite_idxs.detach().long().flatten().cpu().numpy()
			# merge bg with elites to ensure elites visible
			bg_idx = np.unique(np.concatenate([bg_idx, elite_np], axis=0))

		if (k_star is not None) and (0 <= int(k_star) < K):
			bg_idx = np.unique(np.concatenate([bg_idx, np.array([int(k_star)])], axis=0))

		fig = plt.figure(figsize=(4.6, 4.2))
		ax = fig.add_subplot(1, 1, 1)

		# all/background
		ax.scatter(x[bg_idx], y[bg_idx], s=6, alpha=0.35, label="all(candidates)")

		# elites
		if elite_np is not None and elite_np.size > 0:
			ax.scatter(x[elite_np], y[elite_np], s=14, alpha=0.85, label="elites")

		# k*
		if (k_star is not None) and (0 <= int(k_star) < K):
			ax.scatter([x[int(k_star)]], [y[int(k_star)]], s=90, marker="*", label="k*")

		ax.set_xlabel("unc (bigger=worse)")
		ax.set_ylabel("ret/value (bigger=better)")
		ax.set_title(title)
		ax.legend(loc="best", frameon=False)
		ax.grid(True, alpha=0.2)
		fig.tight_layout()
		return fig


	def _tb_fig_unc_hist_elite_vs_all(self, x_unc_K: torch.Tensor,
									elite_idxs: torch.Tensor,
									title: str = "unc_risk(elites) vs unc_risk(all)",
									bins: int = 40):
		"""
		x_unc_K: (K,)
		elite_idxs: (E,)
		"""
		x = x_unc_K.detach().float().flatten().cpu().numpy()
		elite_np = elite_idxs.detach().long().flatten().cpu().numpy()

		x_elite = x[elite_np] if elite_np.size > 0 else x[:0]

		fig = plt.figure(figsize=(5.0, 3.2))
		ax = fig.add_subplot(1, 1, 1)

		ax.hist(x, bins=bins, alpha=0.45, label="all(K)")
		ax.hist(x_elite, bins=bins, alpha=0.65, label="elites(E)")
		ax.set_title(title)
		ax.set_xlabel("unc (bigger=worse)")
		ax.set_ylabel("count")
		ax.legend(loc="best", frameon=False)
		ax.grid(True, alpha=0.2)
		fig.tight_layout()
		return fig


	@torch.no_grad()
	def plan(self, z, a_idx, t0=False, eval_mode=False, task=None, v_real=None, w_real=None, writer=None, observation=None):
		"""
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""		
		# observation to numpy
		if isinstance(observation, torch.Tensor):
			obs_np = observation.detach().cpu().numpy()
		else:
			obs_np = np.asarray(observation)

		self.plan_counter += 1
		# print("[PSAFE] collision head w-norm:", float(self.model._collision[-1].weight.abs().sum()))
		if self.plan_counter == 1:
			print(f"[PSAFE-DBG] cfg.use_psafe={getattr(self.cfg,'use_psafe',None)} | "
			      f"model.use_psafe_head={getattr(self.model,'use_psafe_head','MISSING')} | "
			      f"use_risk_aware={getattr(self.cfg,'use_risk_aware',None)} | "
			      f"estimate_clip={getattr(self.cfg,'estimate_clip',None)} | "
			      f"use_smooth_penalty={getattr(self.cfg,'use_smooth_penalty',None)} | "
			      f"psafe_lambda={getattr(self.cfg,'psafe_lambda',None)} | "
			      f"additive={getattr(self.cfg,'psafe_additive_ablation',None)}")
		
		# Sample policy trajectories num_pi_trajs = 6 num_samples = 512 iterations = 6 num_elites = 64
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.mppi_horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)

			v_prev = v_real.expand(self.cfg.num_pi_trajs).to(self.device)
			w_prev = w_real.expand(self.cfg.num_pi_trajs).to(self.device)

			if self.cfg.clip_vel_rollout:
				for t in range(self.cfg.mppi_horizon - 1):
					a_target = self.model.pi(_z, task)[1]
					a_exec = self.apply_action_clip(a_target, v_prev, w_prev, self.cfg.max_linear_acc, self.cfg.max_angular_acc, self.cfg.max_linear_vel, self.cfg.max_angular_vel, t0=t)
					pi_actions[t] = a_exec
					_z = self.model.next(_z, a_exec, task)
					v_prev, w_prev = a_exec[:, 0], a_exec[:, 1]
				
				a_target = self.model.pi(_z, task)[1]
				a_exec = self.apply_action_clip(a_target, v_prev, w_prev, self.cfg.max_linear_acc, self.cfg.max_angular_acc, self.cfg.max_linear_vel, self.cfg.max_angular_vel, t0=2)
				pi_actions[-1] = a_exec
			else:
				for t in range(self.cfg.mppi_horizon-1):
					pi_actions[t] = self.model.pi(_z, task)[1]
					if self.cfg.lstm_dyn:
						_z, h = self.model.lstm_next(_z, pi_actions[t]) if t==0 else self.model.lstm_next(_z, pi_actions[t], h)
					elif self.cfg.use_multi_mlp_enc and not self.cfg.obs_state_cat:
						vw_state_z = self.model.next(_z[:, self.cfg.latent_dim:], pi_actions[t], task)
						_z[:, self.cfg.latent_dim:] = vw_state_z

						if self.cfg.use_multi_sep_dyn:
							laser_goal_z = self.model.next_obs_sep(_z[:, :self.cfg.latent_dim], pi_actions[t], task)
							_z[:, :self.cfg.latent_dim] = laser_goal_z
							
						if self.cfg.use_multi_dyn:
							laser_goal_z = self.model.next_goal(_z)
							_z[:, :self.cfg.latent_dim] = laser_goal_z
					elif self.cfg.use_ensemble_dyn:
						_z = self.model.sample_next(_z, pi_actions[t], task, mode="mean")
					else:
						_z = self.model.next(_z, pi_actions[t], task, mode='mean')
				pi_actions[-1] = self.model.pi(_z, task)[1]
		# print(" pi actions shape: \n", pi_actions.shape) # [3, 6, 2]
		# print(" z shape: \n", z.shape) # [1, 512]
		# Initialize state and parameters
		
		z = z.repeat(self.cfg.num_samples, 1)
		v_prev_all = v_real.expand(self.cfg.num_samples).to(self.device)
		w_prev_all = w_real.expand(self.cfg.num_samples).to(self.device)
		
		# s = s.repeat(self.cfg.num_samples, 1)
		# print(" repeat z shape: \n", z.shape) # [512, 512]
		
		mean = torch.zeros(self.cfg.mppi_horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.mppi_horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[a_idx, 1:]
		actions = torch.empty(self.cfg.mppi_horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0 and not self.cfg.warm_start:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for i in range(self.cfg.iterations):
			if self.cfg.clip_vel_sample:
				# （a）生成随机动作样本
				raw_sampled_actions = (
					mean.unsqueeze(1) + std.unsqueeze(1) *
					torch.randn(self.cfg.mppi_horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
				).clamp(-1, 1)

				# 初始化速度
				v_prev_rdactions = v_real.expand(self.cfg.num_samples - self.cfg.num_pi_trajs).to(self.device)
				w_prev_rdactions = w_real.expand(self.cfg.num_samples - self.cfg.num_pi_trajs).to(self.device)

				# （b）对所有样本动作进行 clip
				for t in range(self.cfg.mppi_horizon):
					a_t = raw_sampled_actions[t]
					a_exec_t = self.apply_action_clip(
						a_t, v_prev_rdactions, w_prev_rdactions,
						self.cfg.max_linear_acc, self.cfg.max_angular_acc, 
						self.cfg.max_linear_vel, self.cfg.max_angular_vel, t0=t)
					
					actions[t, self.cfg.num_pi_trajs:] = a_exec_t
					v_prev_rdactions, w_prev_rdactions = a_exec_t[:, 0], a_exec_t[:, 1]
			else:
				# Sample actions
				if self.cfg.warm_start:
					actions = (mean.unsqueeze(1) + std.unsqueeze(1) * \
					torch.randn(self.cfg.mppi_horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device)) \
					.clamp(-1, 1)
				else:
					actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
						torch.randn(self.cfg.mppi_horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
						.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			if i == self.cfg.iterations - 1 and eval_mode:
				vis_mpc_as = actions
			# Compute elite actions
			if self.cfg.use_smooth_penalty:
				# Step 1: 获取原始评估值
				value, smooth_penalty = self._estimate_value(z, actions, task, i, v_prev=v_prev_all, w_prev=w_prev_all)
				value = value.nan_to_num_(0)
				smooth_penalty = smooth_penalty.unsqueeze(1).nan_to_num_(0)
				# print(" value shape: \n", value.shape)
				# print(" smooth penalty shape: \n", smooth_penalty.shape) # [512, 1]
				# print(" smooth penalty norm shape: \n", smooth_penalty_norm.
				# Step 2: 归一化
				smooth_penalty_norm = (smooth_penalty - smooth_penalty.min()) / (smooth_penalty.max() - smooth_penalty.min() + 1e-6)

				# Step 3: 加权组合
				combined_score = value - self.cfg.smooth_weight * smooth_penalty_norm  # [N]
				elite_idxs = torch.topk(combined_score.squeeze(1), self.cfg.num_elites, dim=0).indices
				elite_value, elite_actions = combined_score[elite_idxs], actions[:, elite_idxs]
				
			else:
				if self.cfg.use_risk_aware:
					# risk-aware score over K candidates
					value = self._estimate_value_risk(z.squeeze(0), actions, task, i, v_prev=v_prev_all, w_prev=w_prev_all, observation_np=obs_np).nan_to_num_(0)
					if i == 0:  # 只在每次plan的第0次迭代打一次
						v = value.squeeze(1)
						print("risk value nan/inf:", torch.isnan(value).any().item(), torch.isinf(value).any().item())
						# print("v0 nan/inf:", torch.isnan(v0).any().item(), torch.isinf(v0).any().item())

						print("risk value stats:", v.min().item(), v.max().item(), v.mean().item(), v.std().item())
					v0 = self._estimate_value(z, actions, task, i, v_prev=v_prev_all, w_prev=w_prev_all).nan_to_num_(0)
					corr = torch.corrcoef(torch.stack([v0.squeeze(1), value.squeeze(1)]))[0,1]
					print("corr:", corr.item())

					v = value.squeeze(1)
					u = v0.squeeze(1)

					print("scale ratio std(risk)/std(base):", (v.std() / (u.std() + 1e-8)).item())
					print("mean diff:", (v.mean() - u.mean()).item())

					if getattr(self.cfg, "risk_debug", False) and i == 0 and getattr(self.cfg, "use_unc_risk", False):
						dbg_data = getattr(self, "_last_risk_dbg", None)
						if dbg_data:
							print("[DBG][risk] unc_mean/p90/max:",
								dbg_data["unc_mean"], dbg_data["unc_p90"], dbg_data["unc_max"],
								"ret_std_mean:", dbg_data["ret_std_mean"])



				else:
					value = self._estimate_value(z, actions, task, i, v_prev=v_prev_all, w_prev=w_prev_all).nan_to_num_(0)
				# print(" value shape: \n", value.shape) # [512, 1]
				# ---- after value is computed (shape: (K,1) or (K,)) ----
				# ensure shape (K,)
				value_raw = value
				if value.dim() == 2 and value.size(1) == 1:
					v = value.squeeze(1)
				else:
					v = value

				# (optional) print once per plan call, first MPPI iteration
				if i == 0 and getattr(self.cfg, "risk_debug", False):
					print("[DBG][plan] value_raw stats:",
						"min", v.min().item(), "max", v.max().item(),
						"mean", v.mean().item(), "std", v.std(unbiased=False).item(),
						"nan", torch.isnan(v).any().item(), "inf", torch.isinf(v).any().item())

				# ---- z-score normalization across K candidates ----
				# This is crucial when risk-aware score has a wildly different scale than base value.
				if getattr(self.cfg, "value_zscore", True):
					v_mean = v.mean()
					v_std = v.std(unbiased=False).clamp_min(getattr(self.cfg, "value_zscore_eps", 1e-6))
					v = (v - v_mean) / v_std

					# optional clamp to avoid extreme tails making exp() collapse
					z_clip = getattr(self.cfg, "value_zscore_clip", 5.0)
					if z_clip is not None and z_clip > 0:
						v = v.clamp(-z_clip, z_clip)

					if i == 0 and getattr(self.cfg, "risk_debug", False):
						print("[DBG][plan] value_z stats:",
							"min", v.min().item(), "max", v.max().item(),
							"mean", v.mean().item(), "std", v.std(unbiased=False).item())

				# write back standardized value to keep downstream logic unchanged
				value = v.unsqueeze(1)  # shape (K,1)

				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
				# print(" elite idxs shape: \n", elite_idxs.shape) # [64]
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
				# print(" elite actions shape: \n", elite_actions.shape) # [3, 64, 2]
				# print(" elite value shape: \n", elite_value.shape) # [64, 1]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			# print("score elite action", score.shape, elite_actions.shape)

			# ---- DEBUG (Experiment-1) ----
			dbg = getattr(self.cfg, "risk_debug", True)
			if dbg and i == 0:  # 只在每次plan的第0次迭代打印，避免刷屏
				p = score.squeeze(1).detach()  # (E,)  E=num_elites
				# 熵：越大越均匀，越小越one-hot
				entropy = -(p * (p + 1e-12).log()).sum()
				# ESS：有效样本数，越接近E越均匀，越接近1越one-hot
				ess = 1.0 / (p.pow(2).sum() + 1e-12)
				top1_ratio = (p.max() / (p.sum() + 1e-12)).item()

				print(f"[DBG] MPPI weights: E={p.numel()} "
					f"min={p.min().item():.6g} max={p.max().item():.6g} mean={p.mean().item():.6g} "
					f"entropy={entropy.item():.6g} (max~{math_common.log(p.numel()+1e-12):.6g}) "
					f"ESS={ess.item():.3f}/{p.numel()} top1_ratio={top1_ratio:.3f}")

				# elite_value 的尺度也很关键（决定softmax是否塌缩）
				ev = elite_value.squeeze(1).detach()
				print(f"[DBG] elite_value: min={ev.min().item():.6g} max={ev.max().item():.6g} "
					f"mean={ev.mean().item():.6g} std={ev.std().item():.6g} "
					f"nan={torch.isnan(ev).any().item()} inf={torch.isinf(ev).any().item()}")
			# ---- DEBUG end ----

			# # 在算完score后
			# p = score.squeeze(1)
			# print("score stats:", p.min().item(), p.max().item(), p.mean().item())
			
			# ess = 1.0 / (p.pow(2).sum() + 1e-12)
			# print("score ESS:", ess.item(), " / elites:", p.numel())

			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			
			
			std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]



			if writer is not None and getattr(self.cfg, "tb_log_figures", False):
				rid = 0 if a_idx is None else int(a_idx)

				fig_mode = getattr(self.cfg, "tb_fig_mode", "interval")  # {"episode","interval"}
				fig_interval = getattr(self.cfg, "tb_fig_interval", 10)

				save_dir = os.path.join("traj_fig/pedestrain_unc_mix_0")   # 或者 self.cfg.output_dir 下
				print("traj dir", save_dir)
				os.makedirs(save_dir, exist_ok=True)

				# 统一判断：need_fig
				need_fig = bool(t0) if fig_mode == "episode" else ((self.plan_counter % fig_interval) == 0)
				print("=====================>need fig in this step ", need_fig, "<=======================")
				if need_fig:
					step_x = self.plan_counter
					maxK_scatter = getattr(self.cfg, "tb_fig_maxK", 256)

					# --------- compute k* (use highest MPPI weight elite as "selected") ----------
					# score here is MPPI weights over elites: shape (E,1)
					try:
						p_np = score.squeeze(1).detach().cpu().numpy()  # (E,)
						elite_idxs_cpu = elite_idxs.detach().long().cpu()  # (E,)
						k_star = int(elite_idxs_cpu[int(p_np.argmax())].item())  # global candidate index in [0,K)
					except Exception:
						k_star = None
						elite_idxs_cpu = None

					# ---------- (A) Always-available figure ----------
					try:
						p = score.squeeze(1).detach().cpu().numpy()        # (E,)
						ev = elite_value.squeeze(1).detach().cpu().numpy() # (E,)

						fig = plt.figure(figsize=(8, 3))
						ax1 = fig.add_subplot(1,2,1)
						ax1.hist(p, bins=30)
						ax1.set_title("elite weight hist")
						ax1.set_xlabel("w"); ax1.set_ylabel("count")

						ax2 = fig.add_subplot(1,2,2)
						ax2.hist(ev, bins=30)
						ax2.set_title("elite_value hist")
						ax2.set_xlabel("value"); ax2.set_ylabel("count")

						tb_add_figure_compat(writer, f"fig/planner/r{rid}/mppi_weight_value_hist", fig, step_x)
						plt.close(fig)
					except Exception:
						pass

					# ---------- (B) Risk/uncertainty-dependent figures ----------
					dbg_data = getattr(self, "_last_risk_dbg", None)
					if isinstance(dbg_data, dict):

						def _is_tensor(x):
							return isinstance(x, torch.Tensor)

						U = dbg_data.get("U_S_K", None)                 # (S,K)  optional
						U_epi = dbg_data.get("U_epi_S_K", None)                 # (S,K)  optional
						U_ale = dbg_data.get("U_ale_S_K", None)                 # (S,K)  optional
						U_vis = dbg_data.get("U_vis", None)                 # (S,K)  optional

						ret_agg_K = dbg_data.get("ret_agg_K", None)     # (K,)   optional
						score_K = dbg_data.get("score_K", None)         # (K,)   optional
						unc_risk_K = dbg_data.get("unc_risk_K", None)   # (K,)   optional
						unc_epi_risk_K = dbg_data.get("unc_epi_risk_K", None)   # (K,)   optional
						unc_ale_risk_K = dbg_data.get("unc_ale_risk_K", None)   # (K,)   optional

						# ---- NEW: validate adaptive weighting / component CVaR ----
						w_ale_K   = dbg_data.get("w_ale_K", None)
						unc_mix_K = dbg_data.get("unc_mix_K", None)

						# ---- pick an "x axis uncertainty metric" that works for ALL modes ----
						# priority: unc_risk_K (B-mode) > unc_metric_K (shaping/any) > mean(U) (if only heatmap exists)
						x_unc_K, x_unc_epi_K, x_unc_ale_K, x_unc_mix_K = None, None, None, None
						if _is_tensor(unc_risk_K):
							x_unc_K = unc_risk_K
						else:
							unc_metric_K = dbg_data.get("unc_metric_K", None)
							if _is_tensor(unc_metric_K):
								x_unc_K = unc_metric_K
							elif _is_tensor(U):
								# fallback: mean over particles -> (K,)
								x_unc_K = U.mean(dim=0)
						
						if _is_tensor(unc_epi_risk_K):
							x_unc_epi_K = unc_epi_risk_K
						else:
							unc_epi_metric_K = dbg_data.get("unc_epi_metric_K", None)
							if _is_tensor(unc_epi_metric_K):
								x_unc_epi_K = unc_epi_metric_K

						if _is_tensor(unc_ale_risk_K):
							x_unc_ale_K = unc_ale_risk_K
						else:
							unc_ale_metric_K = dbg_data.get("unc_ale_metric_K", None)
							if _is_tensor(unc_ale_metric_K):
								x_unc_ale_K = unc_ale_metric_K
						
						if _is_tensor(unc_mix_K):
							x_unc_mix_K = unc_mix_K


						# ---- pick y axis metric: ret_agg_K preferred, else score_K, else (fallback) none ----
						y_ret_K = None
						if _is_tensor(ret_agg_K):
							y_ret_K = ret_agg_K
						elif _is_tensor(score_K):
							y_ret_K = score_K

						# 1) U heatmap (if available)
						if _is_tensor(U):
							try:
								U_np = U.detach().cpu().numpy() if getattr(self.cfg, "use_all_unc", True) else U_epi.detach().cpu().numpy()
								K_plot = min(U_np.shape[1], getattr(self.cfg, "tb_fig_maxK_heatmap", 128))
								U_np = U_np[:, :K_plot]

								fig = plt.figure(figsize=(10, 3))
								ax = fig.add_subplot(1,1,1)
								im = ax.imshow(U_np, aspect="auto")
								ax.set_title("U_all(S,K) heatmap (uncertainty cost)")
								ax.set_xlabel("candidate k"); ax.set_ylabel("particle s")
								fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/U_all_heatmap", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_U_all_heatmap.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass
							
							try:
								U_np = U_vis.detach().cpu().numpy()
								K_plot = min(U_np.shape[1], getattr(self.cfg, "tb_fig_maxK_heatmap", 128))
								U_np = U_np[:, :K_plot]

								fig = plt.figure(figsize=(10, 3))
								ax = fig.add_subplot(1,1,1)
								im = ax.imshow(U_np, aspect="auto")
								ax.set_title("U_vis(S,K) heatmap (uncertainty cost)")
								ax.set_xlabel("candidate k"); ax.set_ylabel("particle s")
								fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/U_vis_heatmap", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_U_vis_heatmap.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass

							try:
								U_np = U_epi.detach().cpu().numpy()
								K_plot = min(U_np.shape[1], getattr(self.cfg, "tb_fig_maxK_heatmap", 128))
								U_np = U_np[:, :K_plot]

								fig = plt.figure(figsize=(10, 3))
								ax = fig.add_subplot(1,1,1)
								im = ax.imshow(U_np, aspect="auto")
								ax.set_title("U_epi(S,K) heatmap (uncertainty cost)")
								ax.set_xlabel("candidate k"); ax.set_ylabel("particle s")
								fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/U_epi_heatmap", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_U_epi_heatmap.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass
							
							try:
								U_np = U_ale.detach().cpu().numpy()
								K_plot = min(U_np.shape[1], getattr(self.cfg, "tb_fig_maxK_heatmap", 128))
								U_np = U_np[:, :K_plot]

								fig = plt.figure(figsize=(10, 3))
								ax = fig.add_subplot(1,1,1)
								im = ax.imshow(U_np, aspect="auto")
								ax.set_title("U_ale(S,K) heatmap (uncertainty cost)")
								ax.set_xlabel("candidate k"); ax.set_ylabel("particle s")
								fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/U_ale_heatmap", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_U_ale_heatmap.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass

						# 2) ret vs unc scatter (marked: elites + k*)
						if _is_tensor(x_unc_K) and _is_tensor(y_ret_K) and (elite_idxs_cpu is not None):
							try:
								fig = self._tb_fig_ret_vs_unc(
									x_unc_K=x_unc_K if getattr(self.cfg, "use_all_unc", True) else x_unc_epi_K,
									y_ret_K=y_ret_K,
									elite_idxs=elite_idxs_cpu,
									k_star=k_star,
									title="ret vs unc (candidate-wise)  [elites + k*]",
									maxK=maxK_scatter
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/ret_vs_unc", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_ret_vs_unc.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass

							# 2.5) NEW: unc histogram elites vs all
							try:
								fig = self._tb_fig_unc_hist_elite_vs_all(
									x_unc_K=x_unc_K if getattr(self.cfg, "use_all_unc", True) else x_unc_epi_K,
									elite_idxs=elite_idxs_cpu,
									title="unc(elites) vs unc(all)"
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/unc_hist_elite_vs_all", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_unc_heist_elite_vs_all.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass

						# 3) score(K) hist (if available)
						if _is_tensor(score_K):
							try:
								s = score_K.detach().cpu().numpy()
								fig = plt.figure(figsize=(4, 3))
								ax = fig.add_subplot(1,1,1)
								ax.hist(s, bins=40)
								ax.set_title("score(K) hist")
								ax.set_xlabel("score"); ax.set_ylabel("count")
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/score_hist", fig, step_x)
								plt.close(fig)
							except Exception:
								pass

						# ---- NEW: validate adaptive weighting / component CVaR ----
						if _is_tensor(w_ale_K):
							try:
								fig = self._tb_fig_w_ale_hist(
									w_ale_K=w_ale_K,
									title=f"w_ale_K hist (it{i})"
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/w_ale_hist/iteration{i}", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_w_ale_hist.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass

						if _is_tensor(unc_epi_risk_K) and _is_tensor(unc_ale_risk_K) and _is_tensor(w_ale_K):
							try:
								fig = self._tb_fig_unc_epi_vs_ale_scatter(
									unc_epi_K=unc_epi_risk_K,
									unc_ale_K=unc_ale_risk_K,
									w_ale_K=w_ale_K,
									elite_idxs=elite_idxs_cpu,
									k_star=k_star,
									title=f"unc_epi vs unc_ale (colored by w_ale)  it{i}",
									maxK=maxK_scatter,
									cmap_name="viridis"
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/unc_epi_vs_ale/iteration{i}", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_unc_epi_vs_ale.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass

						# ret vs unc scatter using mixed uncertainty
						if _is_tensor(unc_mix_K) and _is_tensor(y_ret_K) and (elite_idxs_cpu is not None):
							try:
								fig = self._tb_fig_ret_vs_unc(
									x_unc_K=unc_mix_K,
									y_ret_K=y_ret_K,
									elite_idxs=elite_idxs_cpu,
									k_star=k_star,
									title=f"ret vs unc_mix (candidate-wise) [elites + k*]  it{i}",
									maxK=maxK_scatter
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/ret_vs_unc_mix/iteration{i}", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_ret_vs_unc_mix.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)
							except Exception:
								pass


					# --- NEW: trajectory overlay + heatmap ---
					print("new traj map !!!!!!!!!!!!!!!!")
					try:
						if observation is not None and _is_tensor(x_unc_K) and _is_tensor(x_unc_epi_K) and _is_tensor(x_unc_ale_K) and (elite_idxs_cpu is not None):
							print("in new traj map!!!!!!!!!!!!!!!")
							# observation to numpy
							# if isinstance(observation, torch.Tensor):
							# 	obs_np = observation.detach().cpu().numpy()
							# else:
							# 	obs_np = np.asarray(observation)

							# (1) 轨迹簇：按 ret 着色
							if _is_tensor(y_ret_K):
								fig = self._tb_fig_traj_cluster(
									observation_np=obs_np,
									actions_HK2=actions.detach(),
									metric_K=y_ret_K.detach(),
									metric_name="ret",
									elite_idxs=elite_idxs_cpu,
									k_star=k_star,
									maxK=maxK_scatter,
									iter_num=i,
									realtime_show=getattr(self.cfg, "show_traj_ret_realtime", False)
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_ret/iteration{i}", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_traj_ret.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)

							# (2) 轨迹簇：按 unc 着色（更直观看“风险在哪”）
							fig = self._tb_fig_traj_cluster(
								observation_np=obs_np,
								actions_HK2=actions.detach(),
								metric_K=x_unc_K.detach(),
								metric_name="unc_all",
								elite_idxs=elite_idxs_cpu,
								k_star=k_star,
								maxK=maxK_scatter,
								cmap_name="plasma",
								iter_num=i
							)
							tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_unc_all/iteration{i}", fig, step_x)
							try:
								fname = f"r{rid}_step{step_x}_it{i}_traj_unc_all.png"
								fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
							except Exception:
								pass
							plt.close(fig)

							fig = self._tb_fig_traj_cluster(
								observation_np=obs_np,
								actions_HK2=actions.detach(),
								metric_K=x_unc_epi_K.detach(),
								metric_name="unc_epi",
								elite_idxs=elite_idxs_cpu,
								k_star=k_star,
								maxK=maxK_scatter,
								cmap_name="plasma",
								iter_num=i
							)
							tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_unc_epi/iteration{i}", fig, step_x)
							try:
								fname = f"r{rid}_step{step_x}_it{i}_traj_unc_epi.png"
								fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
							except Exception:
								pass
							plt.close(fig)

							fig = self._tb_fig_traj_cluster(
								observation_np=obs_np,
								actions_HK2=actions.detach(),
								metric_K=x_unc_ale_K.detach(),
								metric_name="unc_ale",
								elite_idxs=elite_idxs_cpu,
								k_star=k_star,
								maxK=maxK_scatter,
								cmap_name="plasma",
								iter_num=i
							)
							tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_unc_ale/iteration{i}", fig, step_x)
							try:
								fname = f"r{rid}_step{step_x}_it{i}_traj_unc_ale.png"
								fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
							except Exception:
								pass
							plt.close(fig)

							fig = self._tb_fig_traj_cluster(
								observation_np=obs_np,
								actions_HK2=actions.detach(),
								metric_K=x_unc_mix_K.detach(),
								metric_name="unc_mix",
								elite_idxs=elite_idxs_cpu,
								k_star=k_star,
								maxK=maxK_scatter,
								cmap_name="plasma",
								iter_num=i
							)
							tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_unc_mix/iteration{i}", fig, step_x)
							try:
								fname = f"r{rid}_step{step_x}_it{i}_traj_unc_mix.png"
								fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
							except Exception:
								pass
							plt.close(fig)

							# (3) 轨迹热力图：用 score_K 或 ret_agg_K 做权重
							wK = None
							if _is_tensor(score_K):
								wK = score_K
								title = "traj heatmap weighted by score"
							elif _is_tensor(y_ret_K):
								wK = y_ret_K
								title = "traj heatmap weighted by ret"
							else:
								wK = x_unc_K.neg()   # risk high => weight low
								title = "traj heatmap weighted by (-unc)"

							fig = self._tb_fig_traj_heatmap(
								observation_np=obs_np,
								actions_HK2=actions.detach(),
								weight_K=wK.detach(),
								title=title,
								maxK=maxK_scatter
							)
							tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_heatmap/iteration{i}", fig, step_x)
							plt.close(fig)

							# (4) 轨迹簇：按 w_ale 着色（验证自适应权重是否“打中动态区域”）
							if _is_tensor(w_ale_K):
								fig = self._tb_fig_traj_cluster(
									observation_np=obs_np,
									actions_HK2=actions.detach(),
									metric_K=w_ale_K.detach(),          # (K,)
									metric_name="w_ale",
									elite_idxs=elite_idxs_cpu,
									k_star=k_star,
									maxK=maxK_scatter,
									cmap_name="plasma",               # w_ale 用 viridis 直观（暗=小，亮=大）
									iter_num=i
								)
								tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_wale/iteration{i}", fig, step_x)
								try:
									fname = f"r{rid}_step{step_x}_it{i}_traj_w_ale.png"
									fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
								except Exception:
									pass
								plt.close(fig)

					except Exception:
						pass

		
		# ======================> MPPI DONE <=====================
		# ======================> TB RECORD <=====================
    	# --- after you computed dbg values like ESS/top1_ratio/unc_mean ---
		# print(writer)
		# print(self.cfg.use_unc_risk)
		
		if writer is not None and getattr(self.cfg, "use_unc_risk", False):
			# 1) 用 plan_counter 节流（更合理）
			log_interval = getattr(self.cfg, "tb_log_interval", 100)
			if (self.plan_counter % log_interval) == 0:
				rid = 0 if a_idx is None else int(a_idx)

				# 用 env_step 做横轴也行，但同一个 env_step 会写 4 次
				# 所以强烈建议 tag 带 rid
				step_x = self.plan_counter  # 或 env_step（你二选一）

				# 你在 _estimate_value_risk 里存的 dbg
				dbg_data = getattr(self, "_last_risk_dbg", {}) or {}
				writer.add_scalar(f"planner/r{rid}/unc_mean", dbg_data.get("unc_mean", 0.0), step_x)
				writer.add_scalar(f"planner/r{rid}/unc_p90",  dbg_data.get("unc_p90",  0.0), step_x)
				writer.add_scalar(f"planner/r{rid}/unc_max",  dbg_data.get("unc_max",  0.0), step_x)

				# 你在 plan() 里算的 ESS/top1/entropy（建议你也存一下）
				writer.add_scalar(f"planner/r{rid}/ess", float(ess), step_x)
				writer.add_scalar(f"planner/r{rid}/top1_ratio", float(top1_ratio), step_x)
				writer.add_scalar(f"planner/r{rid}/entropy", float(entropy), step_x)

				# 可选：标记一下“这是一次 plan”
				writer.add_scalar(f"planner/r{rid}/plan_called", 1.0, step_x)
				
		# ====== TB figures logging (inside plan) ======
		# ====== TB figures logging (SAFE) ======
		# if writer is not None and getattr(self.cfg, "tb_log_figures", True):
		# 	rid = 0 if a_idx is None else int(a_idx)

		# 	fig_mode = getattr(self.cfg, "tb_fig_mode", "interval")  # {"episode","interval"}
		# 	fig_interval = getattr(self.cfg, "tb_fig_interval", 10)

		# 	# 统一判断：need_fig
		# 	need_fig = bool(t0) if fig_mode == "episode" else ((self.plan_counter % fig_interval) == 0)
		# 	print("=====================>need fig in this step ", need_fig, "<=======================")
		# 	if need_fig:
		# 		step_x = self.plan_counter
		# 		maxK_scatter = getattr(self.cfg, "tb_fig_maxK", 256)

		# 		# --------- compute k* (use highest MPPI weight elite as "selected") ----------
		# 		# score here is MPPI weights over elites: shape (E,1)
		# 		try:
		# 			p_np = score.squeeze(1).detach().cpu().numpy()  # (E,)
		# 			elite_idxs_cpu = elite_idxs.detach().long().cpu()  # (E,)
		# 			k_star = int(elite_idxs_cpu[int(p_np.argmax())].item())  # global candidate index in [0,K)
		# 		except Exception:
		# 			k_star = None
		# 			elite_idxs_cpu = None

		# 		# ---------- (A) Always-available figure ----------
		# 		try:
		# 			p = score.squeeze(1).detach().cpu().numpy()        # (E,)
		# 			ev = elite_value.squeeze(1).detach().cpu().numpy() # (E,)

		# 			fig = plt.figure(figsize=(8, 3))
		# 			ax1 = fig.add_subplot(1,2,1)
		# 			ax1.hist(p, bins=30)
		# 			ax1.set_title("elite weight hist")
		# 			ax1.set_xlabel("w"); ax1.set_ylabel("count")

		# 			ax2 = fig.add_subplot(1,2,2)
		# 			ax2.hist(ev, bins=30)
		# 			ax2.set_title("elite_value hist")
		# 			ax2.set_xlabel("value"); ax2.set_ylabel("count")

		# 			tb_add_figure_compat(writer, f"fig/planner/r{rid}/mppi_weight_value_hist", fig, step_x)
		# 			plt.close(fig)
		# 		except Exception:
		# 			pass

		# 		# ---------- (B) Risk/uncertainty-dependent figures ----------
		# 		dbg_data = getattr(self, "_last_risk_dbg", None)
		# 		if isinstance(dbg_data, dict):

		# 			def _is_tensor(x):
		# 				return isinstance(x, torch.Tensor)

		# 			U = dbg_data.get("U_S_K", None)                 # (S,K)  optional
		# 			ret_agg_K = dbg_data.get("ret_agg_K", None)     # (K,)   optional
		# 			score_K = dbg_data.get("score_K", None)         # (K,)   optional
		# 			unc_risk_K = dbg_data.get("unc_risk_K", None)   # (K,)   optional

		# 			# ---- pick an "x axis uncertainty metric" that works for ALL modes ----
		# 			# priority: unc_risk_K (B-mode) > unc_metric_K (shaping/any) > mean(U) (if only heatmap exists)
		# 			x_unc_K = None
		# 			if _is_tensor(unc_risk_K):
		# 				x_unc_K = unc_risk_K
		# 			else:
		# 				unc_metric_K = dbg_data.get("unc_metric_K", None)
		# 				if _is_tensor(unc_metric_K):
		# 					x_unc_K = unc_metric_K
		# 				elif _is_tensor(U):
		# 					# fallback: mean over particles -> (K,)
		# 					x_unc_K = U.mean(dim=0)

		# 			# ---- pick y axis metric: ret_agg_K preferred, else score_K, else (fallback) none ----
		# 			y_ret_K = None
		# 			if _is_tensor(ret_agg_K):
		# 				y_ret_K = ret_agg_K
		# 			elif _is_tensor(score_K):
		# 				y_ret_K = score_K

		# 			# 1) U heatmap (if available)
		# 			if _is_tensor(U):
		# 				try:
		# 					U_np = U.detach().cpu().numpy()
		# 					K_plot = min(U_np.shape[1], getattr(self.cfg, "tb_fig_maxK_heatmap", 128))
		# 					U_np = U_np[:, :K_plot]

		# 					fig = plt.figure(figsize=(10, 3))
		# 					ax = fig.add_subplot(1,1,1)
		# 					im = ax.imshow(U_np, aspect="auto")
		# 					ax.set_title("U(S,K) heatmap (uncertainty cost)")
		# 					ax.set_xlabel("candidate k"); ax.set_ylabel("particle s")
		# 					fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/U_heatmap", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		# 			# 2) ret vs unc scatter (marked: elites + k*)
		# 			if _is_tensor(x_unc_K) and _is_tensor(y_ret_K) and (elite_idxs_cpu is not None):
		# 				try:
		# 					fig = self._tb_fig_ret_vs_unc(
		# 						x_unc_K=x_unc_K,
		# 						y_ret_K=y_ret_K,
		# 						elite_idxs=elite_idxs_cpu,
		# 						k_star=k_star,
		# 						title="ret vs unc (candidate-wise)  [elites + k*]",
		# 						maxK=maxK_scatter
		# 					)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/ret_vs_unc", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		# 				# 2.5) NEW: unc histogram elites vs all
		# 				try:
		# 					fig = self._tb_fig_unc_hist_elite_vs_all(
		# 						x_unc_K=x_unc_K,
		# 						elite_idxs=elite_idxs_cpu,
		# 						title="unc(elites) vs unc(all)"
		# 					)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/unc_hist_elite_vs_all", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		# 			# 3) score(K) hist (if available)
		# 			if _is_tensor(score_K):
		# 				try:
		# 					s = score_K.detach().cpu().numpy()
		# 					fig = plt.figure(figsize=(4, 3))
		# 					ax = fig.add_subplot(1,1,1)
		# 					ax.hist(s, bins=40)
		# 					ax.set_title("score(K) hist")
		# 					ax.set_xlabel("score"); ax.set_ylabel("count")
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/score_hist", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		# 		# --- NEW: trajectory overlay + heatmap ---
		# 		print("new traj map !!!!!!!!!!!!!!!!")
		# 		try:
		# 			if observation is not None and _is_tensor(x_unc_K) and (elite_idxs_cpu is not None):
		# 				print("in new traj map!!!!!!!!!!!!!!!")
		# 				# observation to numpy
		# 				if isinstance(observation, torch.Tensor):
		# 					obs_np = observation.detach().cpu().numpy()
		# 				else:
		# 					obs_np = np.asarray(observation)

		# 				# (1) 轨迹簇：按 ret 着色
		# 				if _is_tensor(y_ret_K):
		# 					fig = self._tb_fig_traj_cluster(
		# 						observation_np=obs_np,
		# 						actions_HK2=actions.detach(),
		# 						metric_K=y_ret_K.detach(),
		# 						metric_name="ret",
		# 						elite_idxs=elite_idxs_cpu,
		# 						k_star=k_star,
		# 						maxK=maxK_scatter
		# 					)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_ret", fig, step_x)
		# 					plt.close(fig)

		# 				# (2) 轨迹簇：按 unc 着色（更直观看“风险在哪”）
		# 				fig = self._tb_fig_traj_cluster(
		# 					observation_np=obs_np,
		# 					actions_HK2=actions.detach(),
		# 					metric_K=x_unc_K.detach(),
		# 					metric_name="unc",
		# 					elite_idxs=elite_idxs_cpu,
		# 					k_star=k_star,
		# 					maxK=maxK_scatter
		# 				)
		# 				tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_cluster_by_unc", fig, step_x)
		# 				plt.close(fig)

		# 				# (3) 轨迹热力图：用 score_K 或 ret_agg_K 做权重
		# 				wK = None
		# 				if _is_tensor(score_K):
		# 					wK = score_K
		# 					title = "traj heatmap weighted by score"
		# 				elif _is_tensor(y_ret_K):
		# 					wK = y_ret_K
		# 					title = "traj heatmap weighted by ret"
		# 				else:
		# 					wK = x_unc_K.neg()   # risk high => weight low
		# 					title = "traj heatmap weighted by (-unc)"

		# 				fig = self._tb_fig_traj_heatmap(
		# 					observation_np=obs_np,
		# 					actions_HK2=actions.detach(),
		# 					weight_K=wK.detach(),
		# 					title=title,
		# 					maxK=maxK_scatter
		# 				)
		# 				tb_add_figure_compat(writer, f"fig/planner/r{rid}/traj_heatmap", fig, step_x)
		# 				plt.close(fig)
		# 		except Exception:
		# 			pass


		# if writer is not None and getattr(self.cfg, "tb_log_figures", True):
		# 	rid = 0 if a_idx is None else int(a_idx)

		# 	fig_mode = getattr(self.cfg, "tb_fig_mode", "episode")  # {"episode","interval"}
		# 	fig_interval = getattr(self.cfg, "tb_fig_interval", 50)

		# 	# 统一判断：need_fig
		# 	need_fig = bool(t0) if fig_mode == "episode" else ((self.plan_counter % fig_interval) == 0)

		# 	if need_fig:
		# 		step_x = self.plan_counter
		# 		maxK_scatter = getattr(self.cfg, "tb_fig_maxK", 256)
		# 		#    确保你在 need_fig 之前保留了 v0（shape (K,1)）
		# 		# ret_base_K = v0.squeeze(1).detach()

		# 		# # 1) 拿 unc/U：优先复用 _last_risk_dbg；没有就算一次
		# 		# # dbg = getattr(self, "_last_risk_dbg", None)
		# 		# # if not (isinstance(dbg, dict) and ("unc_mean_K" in dbg) and ("unc_cvar_K" in dbg)):
		# 		# # 这里用“同一批 actions”算unc（对齐关键）
		# 		# # z0 用原始(1,D)的 latent，不要用 repeat 后的 K 份
		# 		# z0 = z.detach()
		# 		# unc_dbg = self._compute_unc_dbg(z0, actions, task, v_prev=v_prev_all, w_prev=w_prev_all)
					

		# 		# unc_mean_K = unc_dbg["unc_mean_K"].to(ret_base_K.device)
		# 		# unc_cvar_K = unc_dbg["unc_cvar_K"].to(ret_base_K.device)

		# 		# # 2) 对齐评估：同一 ret_base_K + 同一 unc
		# 		# if getattr(self.cfg, "tb_aligned_eval", True):
		# 		# 	aligned = self._aligned_eval_scores(ret_base_K, unc_mean_K, unc_cvar_K)
		# 		# 	self._tb_log_aligned_eval(writer, rid, step_x, ret_base_K, unc_mean_K, unc_cvar_K, aligned)

		# 		# --------- compute k* (use highest MPPI weight elite as "selected") ----------
		# 		# score here is MPPI weights over elites: shape (E,1)
		# 		try:
		# 			p_np = score.squeeze(1).detach().cpu().numpy()  # (E,)
		# 			elite_idxs_cpu = elite_idxs.detach().long().cpu()  # (E,)
		# 			k_star = int(elite_idxs_cpu[int(p_np.argmax())].item())  # global candidate index in [0,K)
		# 		except Exception:
		# 			k_star = None
		# 			elite_idxs_cpu = None

		# 		# ---------- (A) Always-available figure ----------
		# 		try:
		# 			p = score.squeeze(1).detach().cpu().numpy()        # (E,)
		# 			ev = elite_value.squeeze(1).detach().cpu().numpy() # (E,)

		# 			fig = plt.figure(figsize=(8, 3))
		# 			ax1 = fig.add_subplot(1,2,1)
		# 			ax1.hist(p, bins=30)
		# 			ax1.set_title("elite weight hist")
		# 			ax1.set_xlabel("w"); ax1.set_ylabel("count")

		# 			ax2 = fig.add_subplot(1,2,2)
		# 			ax2.hist(ev, bins=30)
		# 			ax2.set_title("elite_value hist")
		# 			ax2.set_xlabel("value"); ax2.set_ylabel("count")

		# 			tb_add_figure_compat(writer, f"fig/planner/r{rid}/mppi_weight_value_hist", fig, step_x)
		# 			plt.close(fig)
		# 		except Exception:
		# 			pass

		# 		# ---------- (B) Risk/uncertainty-dependent figures ----------
		# 		dbg_data = getattr(self, "_last_risk_dbg", None)
		# 		if isinstance(dbg_data, dict):

		# 			def _is_tensor(x):
		# 				return isinstance(x, torch.Tensor)

		# 			U = dbg_data.get("U_S_K", None)                 # (S,K)  optional
		# 			ret_agg_K = dbg_data.get("ret_agg_K", None)     # (K,)   optional
		# 			score_K = dbg_data.get("score_K", None)         # (K,)   optional
		# 			unc_risk_K = dbg_data.get("unc_risk_K", None)   # (K,)   optional

		# 			# ---- pick an "x axis uncertainty metric" that works for ALL modes ----
		# 			# priority: unc_risk_K (B-mode) > unc_metric_K (shaping/any) > mean(U) (if only heatmap exists)
		# 			x_unc_K = None
		# 			if _is_tensor(unc_risk_K):
		# 				x_unc_K = unc_risk_K
		# 			else:
		# 				unc_metric_K = dbg_data.get("unc_metric_K", None)
		# 				if _is_tensor(unc_metric_K):
		# 					x_unc_K = unc_metric_K
		# 				elif _is_tensor(U):
		# 					# fallback: mean over particles -> (K,)
		# 					x_unc_K = U.mean(dim=0)

		# 			# ---- pick y axis metric: ret_agg_K preferred, else score_K, else (fallback) none ----
		# 			y_ret_K = None
		# 			if _is_tensor(ret_agg_K):
		# 				y_ret_K = ret_agg_K
		# 			elif _is_tensor(score_K):
		# 				y_ret_K = score_K

					

		# 			# 1) U heatmap (if available)
		# 			if _is_tensor(U):
		# 				try:
		# 					U_np = U.detach().cpu().numpy()
		# 					K_plot = min(U_np.shape[1], getattr(self.cfg, "tb_fig_maxK_heatmap", 128))
		# 					U_np = U_np[:, :K_plot]

		# 					fig = plt.figure(figsize=(10, 3))
		# 					ax = fig.add_subplot(1,1,1)
		# 					im = ax.imshow(U_np, aspect="auto")
		# 					ax.set_title("U(S,K) heatmap (uncertainty cost)")
		# 					ax.set_xlabel("candidate k"); ax.set_ylabel("particle s")
		# 					fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/U_heatmap", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		# 			# 2) ret vs unc scatter (marked: elites + k*)
		# 			if _is_tensor(x_unc_K) and _is_tensor(y_ret_K) and (elite_idxs_cpu is not None):						
		# 				try:
		# 					fig = self._tb_fig_ret_vs_unc(
		# 						x_unc_K=x_unc_K,
		# 						y_ret_K=y_ret_K,
		# 						elite_idxs=elite_idxs_cpu,
		# 						k_star=k_star,
		# 						title="ret vs unc (candidate-wise)  [elites + k*]",
		# 						maxK=maxK_scatter
		# 					)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/ret_vs_unc", fig, step_x)
		# 					plt.close(fig)
							
		# 				except Exception:
		# 					pass

		# 				# 2.5) NEW: unc histogram elites vs all
		# 				try:
		# 					fig = self._tb_fig_unc_hist_elite_vs_all(
		# 						x_unc_K=x_unc_K,
		# 						elite_idxs=elite_idxs_cpu,
		# 						title="unc(elites) vs unc(all)"
		# 					)
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/unc_hist_elite_vs_all", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		# 			# 3) score(K) hist (if available)
		# 			if _is_tensor(score_K):
		# 				try:
		# 					s = score_K.detach().cpu().numpy()
		# 					fig = plt.figure(figsize=(4, 3))
		# 					ax = fig.add_subplot(1,1,1)
		# 					ax.hist(s, bins=40)
		# 					ax.set_title("score(K) hist")
		# 					ax.set_xlabel("score"); ax.set_ylabel("count")
		# 					tb_add_figure_compat(writer, f"fig/planner/r{rid}/score_hist", fig, step_x)
		# 					plt.close(fig)
		# 				except Exception:
		# 					pass

		

		# Select action
		score = score.squeeze(1).cpu().numpy()
		# vis_mpc_as = actions + std.unsqueeze(1).repeat(1, self.cfg.num_samples, 1) * torch.randn(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device) if not self.cfg.eval_mode else actions
		if not eval_mode:
			vis_mpc_as = actions
		vis_pi_a = pi_actions[:, 0, :] + std * torch.randn(self.cfg.horizon, self.cfg.action_dim, device=std.device) if not self.cfg.eval_mode else pi_actions[:, 0, :]

		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		vis_act_a = actions + std * torch.randn(self.cfg.horizon, self.cfg.action_dim, device=std.device) if not self.cfg.eval_mode else actions
		self._prev_mean[a_idx] = mean
		# print("std", std.shape) 3,2                          
		# print(std.unsqueeze(1).repeat(1, self.cfg.num_elites, 1).shape)

		# vis more trajs
		if self.cfg.vis_extra_trajs:
			### ======= 生成额外可视化轨迹簇（大扇形）用于热力图展示 ========
			# num_extra_samples = 64
			# v_range = (0.1, 3.0)      # 线速度范围
			# w_range = (-1.0, 1.0)     # 角速度范围

			# # 均匀采样速度对
			# v_samples = torch.linspace(v_range[0], v_range[1], int(num_extra_samples**0.5), device=self.device)
			# w_samples = torch.linspace(w_range[0], w_range[1], int(num_extra_samples**0.5), device=self.device)
			# v_grid, w_grid = torch.meshgrid(v_samples, w_samples, indexing='ij')  # shape [N, N]
			# vw_pairs = torch.stack([v_grid.flatten(), w_grid.flatten()], dim=-1)  # shape [num_extra_samples, 2]

			# extra_actions = torch.zeros(self.cfg.horizon, vw_pairs.shape[0], self.cfg.action_dim, device=self.device)
			# for t in range(self.cfg.horizon):
			# 	extra_actions[t, :, :] = vw_pairs  # 每个时间步相同速度组合
			num_extra_samples = 64  # 轨迹簇数量
			theta_range = (-1.0, 1.0)  # 角速度范围（归一化动作）
			v_fixed = 3.0  # 固定线速度（归一化动作）

			extra_actions = torch.zeros(self.cfg.horizon, num_extra_samples, self.cfg.action_dim, device=self.device)
			extra_actions[:, :, 0] = v_fixed  # 固定线速度
			extra_actions[:, :, 1] = torch.linspace(theta_range[0], theta_range[1], num_extra_samples, device=self.device).repeat(self.cfg.horizon, 1)

			z_extra = z[:num_extra_samples]  # 用相同的初始隐状态
			v_extra = v_real.expand(num_extra_samples).to(self.device)
			w_extra = w_real.expand(num_extra_samples).to(self.device)
			extra_values, extra_smooth = self._estimate_value(z_extra, extra_actions, task, v_prev=v_extra, w_prev=w_extra)
			extra_values = extra_values.nan_to_num_(0)
			### ===========================================================



		if self.cfg.use_one:
			a, std = actions[0], std[0]
			if not eval_mode:
				a += std * torch.randn(self.cfg.action_dim, device=std.device)
			print("use one action", a)
			return a.clamp_(-1, 1).unsqueeze(0), vis_mpc_as.clamp_(-1, 1), vis_pi_a.clamp_(-1, 1), vis_act_a.clamp_(-1, 1), value.squeeze()
		# print(" actions std shape: \n", actions.shape, std.shape)
		# return all actions for trajectory 
		a_all = []
		for i in range (self.cfg.mppi_horizon):
			if not self.cfg.eval_mode:
				a_single = actions[i] + std[i] * torch.randn(self.cfg.action_dim, device=std[i].device)
			else:
				a_single = actions[i]
			a_all.append(a_single.clamp_(-1, 1))
			vis_act_a[i] = a_single.clamp_(-1, 1)
		
		# TDMPC square
		# mu, std = actions[0], std[0]
		# print("plan", a_all, vis_mpc_as)
		if self.cfg.vis_extra_trajs:
			return a_all, extra_actions.clamp_(-1, 1), vis_pi_a.clamp_(-1, 1), vis_act_a.clamp_(-1, 1), extra_values.squeeze() #, actions, std
		else:
			return a_all, vis_mpc_as.clamp_(-1, 1), vis_pi_a.clamp_(-1, 1), vis_act_a.clamp_(-1, 1), value.squeeze() #, actions, std
		# return a.clamp_(-1, 1)
	
	def update_sac_pi(self, zs, task):
		"""
		Update policy using SAC-inspired approach.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# 计算策略 π 输出的动作及 log π(a|s)
		_, pis, log_pis, _ = self.model.pi(zs, task)

		# 计算 Q 值
		qs = self.model.Q(zs, pis, task, return_type='avg')

		# Q 值归一化（可选）
		if self.cfg.norm_q:
			self.scale.update(qs[0])
			qs = self.scale(qs)

		# 自动调整 α（借鉴 SAC）
		alpha_loss = self.log_alpha.exp() * (-log_pis.mean() - self.target_entropy).detach()
		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		# 计算动态 α
		self.alpha = self.log_alpha.exp().detach()

		# 计算 Actor 损失
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((-qs + self.alpha * log_pis).mean(dim=(1,2)) * rho).mean()

		# 反向传播优化
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()

		self.model.track_q_grad(True)

		return pi_loss.item()

	def update_pi(self, zs, action, task):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		# print("update pi zs grad_fn:", zs.grad_fn)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		# pis_embed = self.model.action_encode(pis, 'q')
		qs = self.model.Q(zs, pis, task, return_type='avg')
		# print("update pi qs: ", qs.shape)
		if self.cfg.norm_q:
			self.scale.update(qs[0])
			qs = self.scale(qs)
		# print("update pi scale qs: ", qs.shape)
		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		if self.cfg.actor_mode=="sac":
			# TD-MPC2 baseline setting.
			pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
			print("========>TD-MPC2 baseline setting<===========")

		# elif self.cfg.actor_mode=="residual":
		# 	# Loss for TD-M(PC)^2
		# 	action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
		# 	std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
		# 	eps = (pis - mu) / std
		# 	log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
		# 	#log_pis_prior = torch.clamp(log_pis_prior, -50000, 0.0)

		# 	log_pis_prior = self.scale(log_pis_prior) if self.scale.value > self.cfg.scale_threshold else torch.zeros_like(log_pis_prior)

		# 	q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
		# 	prior_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
		# 	pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss

		elif self.cfg.actor_mode=="bc_sac": 
			# Vanilla BC-SAC loss for policy learning
			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = (((pis - action) ** 2).sum(dim=-1).mean(dim=1) * rho).mean()
			pi_loss = q_loss + self.cfg.prior_coef * prior_loss
			print("========>Vanilla BC-SAC loss for policy learning<=======")

		# print(" pi loss:", pi_loss)
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task, not_done=None):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		# pi_embed = self.model.action_encode(pi, 'q')
		# _, next_action, next_logprob, _ = self.model.pi(next_z, task)
		# target_q = self.model.Q(next_z, next_action, task, return_type='min', target=True) - self.alpha * next_logprob
			
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		if self.cfg.use_done and not_done != None:
			print("----------td target use not done !-----------")
			return reward + discount * not_done * self.model.Q(next_z, pi, task, return_type='min', target=True)
		else:
			return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)
	
	def plot_traj(self, true_states, predicted_states, epoch, writer):
		"""
		可视化并比较真实轨迹和预测轨迹
		:param true_states: 真值状态 [batch_size, horizon_len, 5]
		:param predicted_states: 预测状态 [batch_size, horizon_len, 5]
		:param epoch: 当前epoch
		:param writer: TensorBoard Writer
		"""
		# 取第一个batch的轨迹
		true_x, true_y = true_states[:, -4].cpu().detach().numpy(), true_states[:, -3].cpu().detach().numpy()
		pred_x, pred_y = predicted_states[:, -4].cpu().detach().numpy(), predicted_states[:, -3].cpu().detach().numpy()
		
		# 绘制轨迹
		plt.figure(figsize=(10, 6))
		plt.plot(true_x, true_y, label='True Trajectory', color='blue', linestyle='-', marker='o')
		plt.plot(pred_x, pred_y, label='Predicted Trajectory', color='red', linestyle='--', marker='x')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title(f'Epoch {epoch} - Trajectory Comparison')
		plt.legend()
		
		# 将图像保存到TensorBoard
		writer.add_figure('Trajectory Comparison', plt.gcf(), epoch)
		plt.close()

		if self.cfg.check_goal:
			true_gx, true_gy = true_states[:, self.obs_fix_laser_dim].cpu().detach().numpy() * np.cos(true_states[:, self.obs_fix_laser_dim+1].cpu().detach().numpy()), true_states[:, self.obs_fix_laser_dim].cpu().detach().numpy() * np.sin(true_states[:, self.obs_fix_laser_dim+1].cpu().detach().numpy())
			pred_gx, pred_gy = predicted_states[:, self.obs_fix_laser_dim].cpu().detach().numpy() * np.cos(predicted_states[:, self.obs_fix_laser_dim+1].cpu().detach().numpy()), predicted_states[:, self.obs_fix_laser_dim].cpu().detach().numpy() * np.sin(predicted_states[:, self.obs_fix_laser_dim+1].cpu().detach().numpy())
			plt.figure(figsize=(10, 6))
			plt.plot(true_gx, true_gy, label='True Goal', color='blue', linestyle='-', marker='o')
			plt.plot(pred_gx, pred_gy, label='Predicted Goal', color='red', linestyle='--', marker='x')
			plt.xlabel('X')
			plt.ylabel('Y')
			plt.title(f'Epoch {epoch} - Goal Comparison')
			plt.legend()
			
			# 将图像保存到TensorBoard
			writer.add_figure('Goal Comparison', plt.gcf(), epoch)
			plt.close()

	# def ideal_physics_model(self, obs, action):
	def visualize_predictions(self, batch_idx, mode, actions_batch, pred_obs, true_obs, pred_reward, true_reward, writer, epoch):
		"""
		可视化预测和真实状态
		Args:
		- batch_idx: 当前可视化的批次索引
		- obstacle_polar: (B, num_obstacles, 2), 障碍物的极坐标
		- goal_polar: (B, 2), 目标点的极坐标
		- pred_states: (B, horizon, 2), 动态模型预测的障碍物和目标点位置
		- true_states: (B, horizon, 2), 理想物理推算的障碍物和目标点位置
		- writer: TensorBoard SummaryWriter
		- epoch: 当前训练的 epoch
		"""
		fig, ax = plt.subplots(figsize=(8, 8))

		# 当前批次的障碍物和目标点
		
		pred_obs = pred_obs[:, batch_idx].cpu().detach().numpy()
		true_obs = true_obs[:, batch_idx].cpu().detach().numpy()
		actions = actions_batch[:, batch_idx].cpu().detach().numpy()

		done_indices = ((true_reward > self.cfg.reward_done-1) | (true_reward < -self.cfg.reward_done+1)).nonzero()
		reward_idx = done_indices[0][1] if done_indices.size(0) > 0 else batch_idx

		true_reward = true_reward[:, reward_idx].cpu().detach().numpy()
		pred_reward = pred_reward[:, reward_idx]

		# print("\n\n")
		# print("true_obs", true_obs)
		# print("action", actions)
		# print("\n\n")
		
		scale = self.cfg.state_range if self.cfg.normalization_state else 1
		# 绘制障碍物
		if mode == 'real_motion':
			color = ['red', 'green', 'blue']			
			for i in range(self.cfg.horizon):
				for j in range(int(self.obs_fix_laser_dim / 2)):
					ax.scatter(pred_obs[i, j*2]*scale*math_common.cos((pred_obs[i, j*2+1]-1)*math_common.pi), pred_obs[i, j*2]*scale*math_common.sin((pred_obs[i, j*2+1]-1)*math_common.pi), c=color[i], marker='s', label='Pred Obstacle in t'+str(i))
					ax.scatter(true_obs[i, j*2]*scale*math_common.cos((true_obs[i, j*2+1]-1)*math_common.pi), true_obs[i, j*2]*scale*math_common.sin((true_obs[i, j*2+1]-1)*math_common.pi), c=color[i], marker='o', label='True Obstacle in t'+str(i))
									
				ax.scatter(pred_obs[i][self.obs_fix_laser_dim]*scale*math_common.cos((pred_obs[i][self.obs_fix_laser_dim+1]-1)*math_common.pi), 
			   				pred_obs[i][self.obs_fix_laser_dim]*scale*math_common.sin((pred_obs[i][self.obs_fix_laser_dim+1]-1)*math_common.pi), c=color[i], marker='s', label='Pred Goal in t'+str(i))
				ax.scatter(true_obs[i][self.obs_fix_laser_dim]*scale*math_common.cos((true_obs[i][self.obs_fix_laser_dim+1]-1)*math_common.pi), 
			   				true_obs[i][self.obs_fix_laser_dim]*scale*math_common.sin((true_obs[i][self.obs_fix_laser_dim+1]-1)*math_common.pi), c=color[i], marker='o', label='True Goal in t'+str(i))

		else:
			for i in range(self.cfg.horizon):
				for j in range(int(self.obs_fix_laser_dim / 2)):
					ax.scatter(pred_obs[i, j*2]*scale+sum(actions[:i+1, 0])*self.cfg.ideal_dis, pred_obs[i, j*2+1]*scale+sum(actions[:i+1, 1])*self.cfg.ideal_dis, c='red', label='Pred Obstacle')
					ax.scatter(true_obs[i, j*2]*scale+sum(actions[:i+1, 0])*self.cfg.ideal_dis, true_obs[i, j*2+1]*scale+sum(actions[:i+1, 1])*self.cfg.ideal_dis, c='yellow', label='True Obstacle')
				
				ax.scatter(pred_obs[i][self.obs_fix_laser_dim]*scale+sum(actions[:i+1, 0])*self.cfg.ideal_dis, pred_obs[i][self.obs_fix_laser_dim+1]*scale+sum(actions[:i+1, 1])*self.cfg.ideal_dis, c='blue', label='Pred Goal')
				ax.scatter(true_obs[i][self.obs_fix_laser_dim]*scale+sum(actions[:i+1, 0])*self.cfg.ideal_dis, true_obs[i][self.obs_fix_laser_dim+1]*scale+sum(actions[:i+1, 1])*self.cfg.ideal_dis, c='black', label='True Goal')
			

		ax.legend()
		ax.set_title(f"Trajectory Visualization (Epoch {epoch})")
		ax.set_xlabel("X")
		ax.set_ylabel("Y")

		# 将图片保存到 TensorBoard
		writer.add_figure(f"Trajectory/Batch_{batch_idx}", fig, epoch)
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(8, 8))
		for i in range(self.cfg.horizon):
			ax.scatter(i+1, true_reward[i], c='black', label='True Reward')
			
			pred_reward_plot = pred_reward[i].cpu().detach().numpy() if self.cfg.use_mse_r else math.two_hot_inv(pred_reward[i], self.cfg).cpu().detach().numpy()
			
			if self.cfg.reward_model_reduce:
				if pred_reward_plot > 0.9 or pred_reward_plot < -0.9:
					pred_reward_plot *= self.cfg.reward_done

			ax.scatter(i+1, pred_reward_plot, c='blue', label='Pred Reward')
			# ax.scatter(i+1, math.two_hot_inv(pred_reward[i], self.cfg).cpu().detach().numpy(), c='blue', label='Pred Reward')
			# else:
			# 	ax.scatter(i+1, pred_reward[i].cpu().detach().numpy(), c='blue', label='Pred Reward')
			
		ax.legend()
		ax.set_title(f"Reward Visualization (Epoch {epoch})")
		ax.set_xlabel("X")
		ax.set_ylabel("Y")

		# 将图片保存到 TensorBoard
		writer.add_figure(f"Reward/Batch_{batch_idx}", fig, epoch)
		plt.close(fig)


	def update(self, buffer, writer, total_step):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		self.train_step += 1
		
		# 实验 2：按 train_step 执行 combine 分支 warmup freeze schedule
		self._apply_combine_freeze_schedule()

		if total_step <= self.cfg.random_exploration_length:
			return self.train_step
		
		t1 = time.time()
		# check buffer sample is continuous and in the same episode or not ???
		if self.cfg.use_done:
			obs, action, reward, not_done, task = buffer.sample()
		else:
			obs, action, reward, task = buffer.sample()

		print('buffer shape obs action reward', obs.shape, action.shape, reward.shape) # [h+1, batch_size, obs_dim] [h, batch_size, action_dim] [h, batch_size, 1]
		action_org = copy.deepcopy(action)
		# action_embed = self.model.action_encode(action, 'd') # [horizon, batch, action_embedding_dim]
		# action_embed_q = self.model.action_encode(action, 'q')
		# action_embed_r = self.model.action_encode(action, 'r')
		# print("action_embed", action_embed.shape)
		# print("buffer sample: \n", obs)
		# print("obs: ", obs.shape) # (horizon+1, 64, 1468)

		# print('update obs ', obs[:, 0, :])

		if self.cfg.obs_fix:
			# with torch.no_grad():
			print("update obs fix and same frame")
			# trans x y theta at t1 t2 .. th and keep laser and goal information same 
			print('org obs goal vw x y yaw', obs[:, 0, self.obs_fix_laser_dim:])
			print('org laser', obs[:, 0, :self.obs_fix_laser_dim])
			print('org action', action[:, 0, :])
			for i in range(self.cfg.horizon):
				if self.cfg.tdmpc_rand_xyyaw:
					x_trans_0 = obs[0, :, -5] * (obs[i+1, :, -8]-obs[0, :, -8]) + obs[0, :, -6] * (obs[i+1, :, -7]-obs[0, :, -7])
					y_trans_0 = -obs[0, :, -6] * (obs[i+1, :, -8]-obs[0, :, -8]) + obs[0, :, -5] * (obs[i+1, :, -7]-obs[0, :, -7])
					
					# org sin_trans = obs[i+1, :, -2]*obs[0, :, -1] - obs[i+1, :, -1]*obs[0, :, -2] (+sin)
					sin_trans_0 = obs[i+1, :, -6]*obs[0, :, -5] - obs[i+1, :, -5]*obs[0, :, -6]
					cos_trans_0 = obs[i+1, :, -5]*obs[0, :, -5] + obs[i+1, :, -6]*obs[0, :, -6]

					x_trans = obs[0, :, -4] + (x_trans_0*obs[0, :, -1] - y_trans_0*obs[0, :, -2])
					y_trans = obs[0, :, -3] + (x_trans_0*obs[0, :, -2] + y_trans_0*obs[0, :, -1])

					sin_trans = sin_trans_0*obs[0, :, -1] + cos_trans_0*obs[0, :, -2]
					cos_trans = cos_trans_0*obs[0, :, -1] - sin_trans_0*obs[0, :, -2]
					
					if i < self.cfg.horizon-1 and self.cfg.use_fix_traj_a:
						traj_x_t = (action[i+1, :, 0]*self.cfg.fix_max_traj-obs[i+1, :, -4])*obs[i+1, :, -1] + (action[i+1, :, 1]*self.cfg.fix_max_traj-obs[i+1, :, -3])*obs[i+1, :, -2]
						traj_y_t = -(action[i+1, :, 0]*self.cfg.fix_max_traj-obs[i+1, :, -4])*obs[i+1, :, -2] + (action[i+1, :, 1]*self.cfg.fix_max_traj-obs[i+1, :, -3])*obs[i+1, :, -1]
						# print('traj_xy_t', traj_x_t[0], traj_y_t[0])
						traj_x_trans = x_trans/self.cfg.fix_max_traj + (traj_x_t*cos_trans - traj_y_t*sin_trans)/self.cfg.fix_max_traj
						traj_y_trans = y_trans/self.cfg.fix_max_traj + (traj_x_t*sin_trans + traj_y_t*cos_trans)/self.cfg.fix_max_traj
						# print('traj_xy_trans', traj_x_trans[0], traj_y_trans[0])

					# test goal information: +-robot_yaw sin cos trans 
					if self.cfg.same_goal:
						goal_x = obs[i+1, :, -12] * torch.cos(obs[i+1, :, -11])
						goal_y = obs[i+1, :, -12] * torch.sin(obs[i+1, :, -11])
						goal_x_trans = x_trans + (goal_x*cos_trans - goal_y*sin_trans)
						goal_y_trans = y_trans + (goal_x*sin_trans + goal_y*cos_trans)
						goal_dis_trans = torch.hypot(goal_x_trans, goal_y_trans)
						goal_theta_trans = torch.atan2(goal_y_trans, goal_x_trans)
						obs[i+1, :, -12] = goal_dis_trans
						obs[i+1, :, -11] = goal_theta_trans
					# if self.cfg.same_obs_hand:
					# 	print("same obs hand", self.obs_fix_laser_dim)
					# 	for j in range(int(self.obs_fix_laser_dim/2)):
							
					# 		laser_x = obs[i+1, :, 2*j] * torch.cos(obs[i+1, :, 2*j+1])
					# 		laser_y = obs[i+1, :, 2*j] * torch.sin(obs[i+1, :, 2*j+1])
					# 		laser_x_trans = x_trans + (laser_x*cos_trans - laser_y*sin_trans)
					# 		laser_y_trans = y_trans + (laser_x*sin_trans + laser_y*cos_trans)
					# 		laser_dis_trans = torch.hypot(laser_x_trans, laser_y_trans)
					# 		laser_theta_trans = torch.atan2(laser_y_trans, laser_x_trans)
					# 		obs[i+1, :, 2*j] = laser_dis_trans
					# 		obs[i+1, :, 2*j+1] = laser_theta_trans
					
						obs[i+1, :, -8] = x_trans
						obs[i+1, :, -7] = y_trans
						obs[i+1, :, -6] = sin_trans
						obs[i+1, :, -5] = cos_trans 

						if i < self.cfg.horizon-1 and self.cfg.use_fix_traj_a:
							action[i+1, :, 0] = traj_x_trans
							action[i+1, :, 1] = traj_y_trans
				
				else:
					if self.cfg.use_cos_sin:
						# x -4 y -3 sin -2 cos -1
						# + yaw
						x_trans = obs[0, :, -1] * (obs[i+1, :, -4]-obs[0, :, -4]) + obs[0, :, -2] * (obs[i+1, :, -3]-obs[0, :, -3])
						y_trans = -obs[0, :, -2] * (obs[i+1, :, -4]-obs[0, :, -4]) + obs[0, :, -1] * (obs[i+1, :, -3]-obs[0, :, -3])
						# -yaw
						# x_trans = obs[0, :, -1] * (obs[i+1, :, -4]-obs[0, :, -4]) - obs[0, :, -2] * (obs[i+1, :, -3]-obs[0, :, -3])
						# y_trans = obs[0, :, -2] * (obs[i+1, :, -4]-obs[0, :, -4]) + obs[0, :, -1] * (obs[i+1, :, -3]-obs[0, :, -3])
						
						# change for test
						# org sin_trans = obs[i+1, :, -2]*obs[0, :, -1] - obs[i+1, :, -1]*obs[0, :, -2] (+sin)
						sin_trans = obs[i+1, :, -2]*obs[0, :, -1] - obs[i+1, :, -1]*obs[0, :, -2]
						cos_trans = obs[i+1, :, -1]*obs[0, :, -1] + obs[i+1, :, -2]*obs[0, :, -2]
						
						if self.cfg.use_dxdydtheta:
							# v -7 w -6 0 -5
							dx_trans = obs[i+1, :, -7] * cos_trans / 10.0
							dy_trans = obs[i+1, :, -7] * sin_trans / 10.0
							dtheta_trans = obs[i+1, :, -6] / 10.0

							if self.cfg.normalization_state:
								dx_trans = torch.clip((dx_trans / self.cfg.max_traj_obs_fix), -1.0, 1.0)
								dy_trans = torch.clip((dy_trans / self.cfg.max_traj_obs_fix), -1.0, 1.0)
								dtheta_trans = (dtheta_trans + torch.pi) / (torch.pi * 2)

							obs[i+1, :, -7] = dx_trans
							obs[i+1, :, -6] = dy_trans
							obs[i+1, :, -5] = dtheta_trans


						if i < self.cfg.horizon-1 and self.cfg.use_fix_traj_a:
							# traj_x_trans = torch.clip(x_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*cos_trans - action[i+1, :, 1]*sin_trans), -1.0, 1.0)
							# traj_y_trans = torch.clip(y_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*sin_trans + action[i+1, :, 1]*cos_trans), -1.0, 1.0)
							traj_x_trans = x_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*cos_trans - action[i+1, :, 1]*sin_trans)
							traj_y_trans = y_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*sin_trans + action[i+1, :, 1]*cos_trans)
							# print("i:", i)
							# print("traj_y_trans", traj_y_trans[0])
						# test goal information: +-robot_yaw sin cos trans 
						if self.cfg.same_goal_hand:
							print("same goal hand")
							goal_x = obs[i+1, :, self.obs_fix_laser_dim] * torch.cos(obs[i+1, :, self.obs_fix_laser_dim+1])
							goal_y = obs[i+1, :, self.obs_fix_laser_dim] * torch.sin(obs[i+1, :, self.obs_fix_laser_dim+1])
							goal_x_trans = x_trans + (goal_x*cos_trans - goal_y*sin_trans)
							goal_y_trans = y_trans + (goal_x*sin_trans + goal_y*cos_trans)
							goal_dis_trans = torch.hypot(goal_x_trans, goal_y_trans)
							goal_theta_trans = torch.atan2(goal_y_trans, goal_x_trans)
							obs[i+1, :, self.obs_fix_laser_dim] = goal_dis_trans
							obs[i+1, :, self.obs_fix_laser_dim+1] = goal_theta_trans
						if self.cfg.same_obs_hand:
							print("same obs hand", self.obs_fix_laser_dim)
							for j in range(int(self.obs_fix_laser_dim/2)):
								
								laser_x = obs[i+1, :, 2*j] * torch.cos(obs[i+1, :, 2*j+1])
								laser_y = obs[i+1, :, 2*j] * torch.sin(obs[i+1, :, 2*j+1])
								laser_x_trans = x_trans + (laser_x*cos_trans - laser_y*sin_trans)
								laser_y_trans = y_trans + (laser_x*sin_trans + laser_y*cos_trans)
								laser_dis_trans = torch.hypot(laser_x_trans, laser_y_trans)
								laser_theta_trans = torch.atan2(laser_y_trans, laser_x_trans)
								obs[i+1, :, 2*j] = laser_dis_trans
								obs[i+1, :, 2*j+1] = laser_theta_trans
					else:
						# x -3 y -2 theta -1
						x_trans = torch.cos(-obs[0, :, -1]) * (obs[i+1, :, -3]-obs[0, :, -3]) - torch.sin(-obs[0, :, -1]) * (obs[i+1, :, -2]-obs[0, :, -2])
						y_trans = torch.sin(-obs[0, :, -1]) * (obs[i+1, :, -3]-obs[0, :, -3]) + torch.cos(-obs[0, :, -1]) * (obs[i+1, :, -2]-obs[0, :, -2])
						theta_trans = obs[i+1, :, -1] - obs[0, :, -1]
						theta_trans = (theta_trans + torch.pi) % (2*torch.pi) - torch.pi

						if i < self.cfg.horizon-1 and self.cfg.use_fix_traj_a:
							traj_x_trans = torch.clip(x_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*torch.cos(theta_trans) - action[i+1, :, 1]*torch.sin(theta_trans)), -1.0, 1.0)
							traj_y_trans = torch.clip(y_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*torch.sin(theta_trans) + action[i+1, :, 1]*torch.cos(theta_trans)), -1.0, 1.0)
						# print('t ', i, 'x_trans ', x_trans[:3])
						# print('t ', i, 'y_trans ', y_trans[:3])
						# print('t ', i,'theta trans ', theta_trans[:3])

					if self.cfg.normalization_state:
						x_trans_norm = torch.clip((x_trans / self.cfg.max_traj_obs_fix), -1.0, 1.0)
						y_trans_norm = torch.clip((y_trans / self.cfg.max_traj_obs_fix), -1.0, 1.0)
						

						if self.cfg.use_cos_sin:
							obs[i+1, :, -4] = x_trans_norm
							obs[i+1, :, -3] = y_trans_norm
							obs[i+1, :, -2] = sin_trans
							obs[i+1, :, -1] = cos_trans 
							
						else:
							theta_trans_norm = (theta_trans + torch.pi) / (torch.pi * 2)
							obs[i+1, :, -3] = x_trans_norm
							obs[i+1, :, -2] = y_trans_norm
							obs[i+1, :, -1] = theta_trans_norm
					else:
						if self.cfg.use_cos_sin:
							obs[i+1, :, -4] = x_trans
							obs[i+1, :, -3] = y_trans
							obs[i+1, :, -2] = sin_trans
							obs[i+1, :, -1] = cos_trans 
							
						else:
							obs[i+1, :, -3] = x_trans 
							obs[i+1, :, -2] = y_trans 
							obs[i+1, :, -1] = theta_trans 

					if i < self.cfg.horizon-1 and self.cfg.use_fix_traj_a:
						action[i+1, :, 0] = traj_x_trans
						action[i+1, :, 1] = traj_y_trans

			# change x y theta at t0 to 0.5 0.5 0.0
			if self.cfg.tdmpc_rand_xyyaw:
				# t0 goal 
				goal_x_t0 = obs[0, :, -12] * torch.cos(obs[0, :, -11])
				goal_y_t0 = obs[0, :, -12] * torch.sin(obs[0, :, -11])
				goal_x_trans_t0 = obs[0, :, -4] + (goal_x_t0*obs[0, :, -1] - goal_y_t0*obs[0, :, -2])
				goal_y_trans_t0 = obs[0, :, -3] + (goal_x_t0*obs[0, :, -2] + goal_y_t0*obs[0, :, -1])
				goal_dis_trans_t0 = torch.hypot(goal_x_trans_t0, goal_y_trans_t0)
				goal_theta_trans_t0 = torch.atan2(goal_y_trans_t0, goal_x_trans_t0)
				
				obs[0, :, -12] = goal_dis_trans_t0
				obs[0, :, -11] = goal_theta_trans_t0
				obs[0, :, -8] = obs[0, :, -4]
				obs[0, :, -7] = obs[0, :, -3]
				obs[0, :, -6] = obs[0, :, -2]
				obs[0, :, -5] = obs[0, :, -1]

				obs = obs[:, :, :-4]

			else:
				if self.cfg.normalization_state:
					if self.cfg.use_cos_sin:
						obs[0, :, -4] = 0.0
						obs[0, :, -3] = 0.0
						obs[0, :, -2] = 0.0
						obs[0, :, -1] = 1.0
						if self.cfg.use_dxdydtheta:
							# v -7 w -6 0 -5
							dx_trans_0 = obs[0, :, -7] / 10.0
							dtheta_trans_0 = obs[0, :, -6] / 10.0

							dx_trans_0 = torch.clip((dx_trans_0 / self.cfg.max_traj_obs_fix), -1.0, 1.0)
							dtheta_trans_0 = (dtheta_trans_0 + torch.pi) / (torch.pi * 2)

							obs[0, :, -7] = dx_trans_0
							obs[0, :, -6] = 0.0
							obs[0, :, -5] = dtheta_trans_0
					else:
						obs[0, :, -3] = 0.5
						obs[0, :, -2] = 0.5
						obs[0, :, -1] = 0.5
				else:
					if self.cfg.use_cos_sin:
						obs[0, :, -4] = 0.0
						obs[0, :, -3] = 0.0
						obs[0, :, -2] = 0.0
						obs[0, :, -1] = 1.0
						if self.cfg.use_dxdydtheta:
							# v -7 w -6 0 -5
							dx_trans_0 = obs[0, :, -7] / 10.0
							dtheta_trans_0 = obs[0, :, -6] / 10.0

							obs[0, :, -7] = dx_trans_0
							obs[0, :, -6] = 0.0
							obs[0, :, -5] = dtheta_trans_0
					else:
						obs[0, :, -3] = 0.0
						obs[0, :, -2] = 0.0
						obs[0, :, -1] = 0.0
			# print("obs t0", obs[0, :, -3:])
			# print('update obs ', obs[:, 0, :])

		if self.cfg.same_obs and self.obs_fix_laser_dim > 0 and not self.cfg.same_obs_hand:
			print("same obs 1-h together")
			# if self.cfg.same_goal:
			# 	obs[1:, :, :self.obs_fix_laser_dim+2] = obs[0, :, :self.obs_fix_laser_dim+2]
			# else:
				# print('obs_fix_laser_dim', self.obs_fix_laser_dim)
			obs[1:, :, :self.obs_fix_laser_dim] = obs[0, :, :self.obs_fix_laser_dim]
	
		if self.cfg.same_goal and not self.cfg.same_goal_hand:
			obs[1:, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim] = obs[0, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim]


			print('trans obs goal vw x y yaw', obs[:, 0, self.obs_fix_laser_dim:])
			print('trans laser', obs[:, 0, :self.obs_fix_laser_dim])
			print('trans action', action[:, 0, :])

		if self.cfg.random_laser_goal:
			random_addition = (torch.rand(obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim].size()) - 0.5) * 2 * self.cfg.rand_obs_range  # 均匀分布 [-a, a]
			obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim] = obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim] + random_addition
			print('rand obs goal vw x y yaw', obs[:, 0, self.obs_fix_laser_dim:])
			print('rand laser', obs[:, 0, :self.obs_fix_laser_dim])
			
		if self.cfg.abandon_yaw:
			obs = obs[:, :, :-2]
		obs_np = obs.cpu().numpy()
		# print("obs_np", type(obs_np))
		# print("obs ", obs[i+1, 0, :])
		if self.cfg.use_done:
			obs, action, reward, not_done, task = buffer._to_device(obs, action, reward, not_done, task)
		else:
			obs, action, reward, task = buffer._to_device(obs, action, reward, task)
		# print('cuda obs', obs[0, 0, :])
		
		# print("obs grad_fn:", obs.grad_fn)
		# print("action grad_fn:", action.grad_fn)
		# print("reward grad_fn:", reward.grad_fn)
		
		# print('update obs np ', obs_np[:, 0, :])

		if self.cfg.lstm_dyn and self.cfg.lstm_seq_len > 1:
			obs = obs.view(-1, -1, self.cfg.lstm_seq_len, self.cfg.obs_shape['state'][0]+self.cfg.action_dim)
			obs_seq = obs[:, :, :, :self.cfg.obs_shape['state'][0]] # [horizon+1, batch, lstm_seq_len, obs_dim]
			a_seq = obs[:, :, :, self.cfg.obs_shape['state'][0]:self.cfg.obs_shape['state'][0]+self.cfg.action_dim]

		# Compute targets
		with torch.no_grad():
			if self.cfg.mlp_obs:
				if self.cfg.lstm_dyn and self.cfg.lstm_seq_len > 1:
					# lstm encode
					next_z = self.model.encode(obs_seq[1:, :, -1, :], task)
				elif self.cfg.use_multi_mlp_enc:
					# two mlp encode
					if self.cfg.use_obs_multi_enc and self.obs_fix_laser_dim>0:
						print("use 3 enc for laser goal and state")
						next_laser_single = self.model.encode(obs[1:, :, :self.obs_fix_laser_dim], task)
						next_goal = self.model.encode_goal(obs[1:, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
						next_laser = torch.cat([next_laser_single, next_goal], dim=-1)
					else:
						next_laser = self.model.encode(obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
					next_z = self.model.encode_state(obs[1:, :, self.obs_fix_laser_dim+self.obs_fix_goal_dim:], task)   # [horizon, batch_size, 6] -> [horizon, bathc_size, state_latent_dim]
					if self.cfg.obs_state_cat or self.cfg.use_multi_dyn:
						next_z = torch.cat([next_laser, next_z], dim=-1)
				else:
					# mlp encode
					next_z = self.model.encode(obs[1:], task)
			else:
				# cnn encode
				t1_cnn = time.time()
				
				next_z = self.model.multi_encode(obs_np[1]).unsqueeze(0)
				for i in range(1, self.cfg.horizon):
					next_z_single = self.model.multi_encode(obs_np[i+1]).unsqueeze(0)
					next_z = torch.cat((next_z, next_z_single), dim=0)
				t2_cnn = time.time()
				print("Total CNN enc Time : {} ms".format(round(1000*(t2_cnn-t1_cnn),2)))
	
			# next_z = self.model.encode(obs[1:], task)
			# print("next_z: ", next_z.shape) # (horizon, 64, 512)
			if self.cfg.use_multi_mlp_enc and not self.cfg.obs_state_cat and not self.cfg.use_multi_dyn:
				if self.cfg.use_done:
					td_targets = self._td_target(torch.cat([next_laser, next_z], dim=-1), reward, task, not_done=not_done)
				else:
					td_targets = self._td_target(torch.cat([next_laser, next_z], dim=-1), reward, task)	
			else:
				if self.cfg.use_done:
					td_targets = self._td_target(next_z, reward, task, not_done=not_done)
				else:
					td_targets = self._td_target(next_z, reward, task)

	
		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim * self.cfg.enc_num, device=self.device)
		decode_obs_s = torch.empty(self.cfg.horizon, self.cfg.batch_size, self.cfg.obs_shape['state'][0], device=self.device)
		# z = self.model.encode(obs[0], task)
		if self.cfg.lstm_dyn and self.cfg.lstm_seq_len > 1:
			lstm_z = self.model.encode(obs_seq, task) # [horizon+1, batch, lstm_seq_len, latent_dim]
			lstm_a = a_seq[:, :, 1:, :] # [horizon+1, batch, lstm_seq_len-1, action_dim]
			z = lstm_z[0, :, -1, :] # [batch, latent_dim]
		# two mlp encode
		elif self.cfg.use_multi_mlp_enc:
			if self.cfg.use_obs_multi_enc and self.obs_fix_laser_dim>0:
				laser_0 = self.model.encode(obs[0, :, :self.obs_fix_laser_dim], task)
				goal_0 = self.model.encode_goal(obs[0, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
				laser_goal_t = torch.cat([laser_0, goal_0], dim=-1)
				laser_h = self.model.encode(obs[1:, :, :self.obs_fix_laser_dim], task)
				goal_h = self.model.encode_goal(obs[1:, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
				laser_goal_h = torch.cat([laser_h, goal_h], dim=-1)
			else:
				laser_goal_t = self.model.encode(obs[0, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
				laser_goal_h = self.model.encode(obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
			# print("laser goal 0", laser_goal_0[0])
			# print("laser goal h", laser_goal_h[:,0,:])
			z = self.model.encode_state(obs[0, :, self.obs_fix_laser_dim+self.obs_fix_goal_dim:], task)
			# print("laser_goal_t grad_fn:", laser_goal_t.grad_fn)
			# print("laser_goal_h grad_fn:", laser_goal_h.grad_fn)
			# print("z0 grad_fn:", z.grad_fn)
			if self.cfg.obs_state_cat:
				# print("obs state cat !!!!!!!!!!!!!!")
				z = torch.cat([laser_goal_t, z], dim=-1)

		elif self.cfg.mlp_obs:
			z = self.model.encode(obs[0], task)
		else:
			z = self.model.multi_encode(obs_np[0])
	

		if self.cfg.use_multi_mlp_enc and not self.cfg.obs_state_cat:
			# cat laser and state z for q reward and policy model training 
			# print("laser goal", laser_goal_0.shape)
			# print("z", z.shape)
			if self.cfg.detach_state_z:
				zs[0] = torch.cat([laser_goal_t, z.detach()], dim=-1)
			else:	
				zs[0] = torch.cat([laser_goal_t, z], dim=-1)
			#TODO : detach z 
		else:
			zs[0] = z

		consistency_loss = 0
		consistency_s_loss = 0
		consistency_laser_goal_loss = 0
		v_prev = action[0][:, 0]
		w_prev = action[0][:, 1]
		a_execs = torch.empty_like(action)
		if self.cfg.lstm_dyn and self.cfg.lstm_seq_len > 1:
			for t in range(self.cfg.horizon):
				action_dyn = torch.cat(lstm_a[t], action[t].unsqueeze(1), dim=1) # [batch, lstm_seq_len, action_dim]
				# add 7 real z and 1 pre z now 
				# other considerations: t = 3: 6 real z and 2 pre z
				z_dyn = torch.cat(lstm_z[t, :, :-1, :], z.unsqueeze(1), dim=1) # [batch, lstm_seq_len, latent_dim]
				z, h = self.model.lstm_next(z_dyn, action_dyn) if t==0 else self.model.lstm_next(z_dyn, action_dyn, h)
				consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
				zs[t+1] = z

		elif self.cfg.use_ensemble_dyn:
			print("use ensemble dyn loss")
			for t in range(self.cfg.horizon):
				# Teacher forcing: use ground-truth z_t from encoder (zs[t]) to predict distribution of z_{t+1}
				z_t = zs[t]             # (B,Z)  encoder rollout state
				target = next_z[t]      # (B,Z)  encoded next latent
				a_t = action[t]         # (B,A)

				mu_all, logvar_all = self.model.next_dist(z_t, a_t, task, return_type="all")  # (M,B,Z)
				# print("mu log shape", mu_all.shape, logvar_all.shape)
				if t == 0:
					print("mu", mu_all.mean().item(), mu_all.std().item())
					print("logvar", logvar_all.mean().item(), logvar_all.min().item(), logvar_all.max().item())
					var_all = torch.exp(logvar_all)
					print("var", var_all.mean().item(), var_all.min().item(), var_all.max().item())

				# NLL per member -> average over members & batch
				# gaussian_nll_diag returns (M,B); mean over B then over M
				nll_mb = gaussian_nll_diag(target.unsqueeze(0), mu_all, logvar_all)  # (M,B)
				# print("nll mb shape", nll_mb.shape)
				
				dyn_loss_t = nll_mb.mean(dim=1).mean(dim=0)  # scalar

				consistency_loss += dyn_loss_t * (self.cfg.rho ** t)

				# For zs rollout (used by update_pi), keep it stable: use deterministic mean next
				z = self.model.sample_next(z_t, a_t, task, mode="mean")
				zs[t+1] = z

		else:
			for t in range(self.cfg.horizon):
				t1_dynamics = time.time()
				# z = self.model.next(z, action_embed[t], task)
				if self.cfg.lstm_dyn:
					z, h = self.model.lstm_next(z, action[t]) if t==0 else self.model.lstm_next(z, action[t], h)
				
				else:
					# MDN
					if self.cfg.use_mdn:
						pi, mu, sigma = self.model.mdn_dynamic(z, action[t], task)
						z_sample = self.model.sample_from_mdn(pi, mu, sigma)
						
						z = z_sample
					else:
						if self.cfg.use_org_action:
							z = self.model.next(z, action_org[t], task)
						else:
							# if self.cfg.clip_vel_rollout:
							# 	a_execs[t] = self.apply_action_clip(action[t], v_prev, w_prev, self.cfg.max_linear_acc, self.cfg.max_angular_acc, self.cfg.max_linear_velocity, self.cfg.max_angular_velocity)
							# 	z = self.model.next(z, a_execs[t], task)
							# 	v_prev, w_prev = a_execs[t][:, 0], a_execs[t][:, 1]
							# else:
							z = self.model.next(z, action[t], task)
						# print("dyn z grad_fn", z.grad_fn)

				if self.cfg.use_multi_dyn:
					laser_goal_t = self.model.next_goal(torch.cat([laser_goal_t, z], dim=-1))
					consistency_loss += F.mse_loss(torch.cat([laser_goal_t, z], dim=-1), next_z[t]) * self.cfg.rho**t
				
				else:
					if self.cfg.use_mdn and self.cfg.use_mdn_loss:
						print("using mdn loss")
						consistency_loss += self.model.mdn_loss(pi, mu, sigma, next_z[t]) * self.cfg.rho**t
					else:
						consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t

				if self.cfg.teacher_forcing:
					p_teacher = self.cfg.p_end + (self.cfg.p_start - self.cfg.p_end) * math_common.exp(-total_step/self.cfg.p_decay_rate)
					use_teacher = random.random() < p_teacher
					if use_teacher:
						z = next_z[t]
				
				if self.cfg.use_multi_mlp_enc and not self.cfg.obs_state_cat:
					if (self.cfg.same_obs and not self.cfg.same_obs_hand) or self.cfg.use_multi_dyn:
						if self.cfg.use_obs_multi_enc and self.obs_fix_laser_dim>0:
							if self.cfg.use_multi_sep_dyn:
								laser_goal_t = self.model.next_obs_sep(laser_goal_t, action[t], task)
								consistency_laser_goal_loss += F.mse_loss(laser_goal_t, next_laser[t]) * self.cfg.rho**t
							else:
								laser_t = self.model.encode(obs[t+1, :, :self.obs_fix_laser_dim], task)
								goal_t = self.model.encode_goal(obs[t+1, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
								laser_goal_t = torch.cat([laser_t, goal_t], dim=-1)
						else:
							if self.cfg.use_multi_sep_dyn:
								print("use dyn to predict next laser and goal")
								laser_goal_t = self.model.next_obs_sep(laser_goal_t, action[t], task)
								consistency_laser_goal_loss += F.mse_loss(laser_goal_t, next_laser[t]) * self.cfg.rho**t
							else:
								laser_goal_t = self.model.encode(obs[t+1, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
						#TODO : detach z 
						if self.cfg.detach_state_z:
							zs[t + 1] = torch.cat([laser_goal_t, z.detach()], dim=-1)
						else:	
							print("dynamic encode laser goal 0 ")
							zs[t + 1] = torch.cat([laser_goal_t, z], dim=-1)
					else:
						if self.cfg.use_obs_multi_enc and self.obs_fix_laser_dim>0:
							if self.cfg.use_multi_sep_dyn:
								laser_goal_t = self.model.next_obs_sep(laser_goal_t, action[t], task)
								consistency_laser_goal_loss += F.mse_loss(laser_goal_t, next_laser[t]) * self.cfg.rho**t
							else:
								laser_t = self.model.encode(obs[t+1, :, :self.obs_fix_laser_dim], task)
								goal_t = self.model.encode_goal(obs[t+1, :, self.obs_fix_laser_dim:self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
								laser_goal_t = torch.cat([laser_t, goal_t], dim=-1)
						else:
							if self.cfg.use_multi_sep_dyn:
								laser_goal_t = self.model.next_obs_sep(laser_goal_t, action[t], task)
								consistency_laser_goal_loss += F.mse_loss(laser_goal_t, next_laser[t]) * self.cfg.rho**t
							else:
								laser_goal_t = self.model.encode(obs[t+1, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
						zs[t + 1] = torch.cat([laser_goal_t, z], dim=-1)
				elif not self.cfg.use_real_z:
					zs[t + 1] = z

				if self.cfg.use_decoder:
					print("z", z.shape)
					if self.cfg.dec_detach:
						decode_obs = self.model.decode(z.detach())
					else:
						decode_obs = self.model.decode(z)
					consistency_s_loss += F.mse_loss(decode_obs, obs[t+1].to(torch.float32)) * self.cfg.rho**t
					decode_obs_s[t] = decode_obs
				t2_dynamics = time.time()

		if self.cfg.use_real_z:
			print("use real z")
			zs[1:] = self.model.encode(obs[1:], task)
		# Predictions
		_zs = zs[:-1]
		# print("zs", zs[:, :2])
		# print("dyn zs grad_fn", _zs.grad_fn)
		# qs = self.model.Q(_zs, action_embed_q, task, return_type='all')
		# reward_preds = self.model.reward(_zs, action_embed_r, task)
		# qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task) # if not self.cfg.clip_vel_rollout else self.model.reward(_zs, a_execs, task)
		# print("qs grad_fn", qs.grad_fn)
		# print("reward preds grad_fn", reward_preds.grad_fn)
		# make_dot(reward_preds, params=dict(self.model.named_parameters())).render("/home/wheel-arm/catkin_ws/src/navigation_research-master/src/navigation_research/graph", format="png")
		
		# Compute losses
		reward_loss, value_loss, path_loss = 0, 0, 0

		mse_loss_fn = torch.nn.MSELoss()
		# action[:1] action_q
		'''value loss'''
		if self.cfg.soft_ce_q:
			print("soft ce q loss")
			qs = self.model.Q(_zs, action, task, return_type='all') # if not self.cfg.clip_vel_rollout else self.model.Q(_zs, a_execs, task, return_type='all')
			for t in range(self.cfg.horizon):
				for q in range(self.cfg.num_q):
					value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		else:
			current_q1, current_q2 = self.model.Q(_zs, action, task, return_type='sep') # if not self.cfg.clip_vel_rollout else self.model.Q(_zs, a_execs, task, return_type='sep')
			print("mse q loss")
			# print("q1 q2 tq", current_q1.shape, current_q2.shape, td_targets.shape)
			for t in range(self.cfg.horizon):
				value_loss += (mse_loss_fn(current_q1[t], td_targets[t]) + mse_loss_fn(current_q2[t], td_targets[t])) * self.cfg.rho**t
		
		'''reward loss'''
		for t in range(self.cfg.horizon):
			if self.cfg.use_mse_r:
				if t < 1:
					print('use mse loss for reward org')
				reward_loss += mse_loss_fn(reward_preds[t], reward[t]) * self.cfg.rho**t
			else:	
				print('use soft ce for reward org')
				reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
		

		# for t in range(self.cfg.horizon):
		# 	reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
		# 	# print('reward loss ', reward_loss)
		# 	for q in range(self.cfg.num_q):
		# 		value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		# 		# print('value loss ', value_loss)
		consistency_loss *= (1/self.cfg.horizon)
		consistency_s_loss *= (1/self.cfg.horizon)
		consistency_laser_goal_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		if self.cfg.soft_ce_q:
			value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		else:
			value_loss *= (1/(self.cfg.horizon))
		print("consistency reward value loss: ", consistency_loss, consistency_s_loss, reward_loss, value_loss)

		# ==================================================
        # training-only auxiliaries
        # ==================================================
		temporal_future_aux_loss = torch.zeros((), device=self.device)
		temporal_interp_aux_loss = torch.zeros((), device=self.device)
		risk_proxy_aux_loss = torch.zeros((), device=self.device)

        # ---------- Aux1: 一步未来特征预测 ----------
		if getattr(self.cfg, 'use_temporal_future_aux', False):
			aux_h = min(int(getattr(self.cfg, 'temporal_future_aux_horizon', 1)), self.cfg.horizon)
			aux_accum = torch.zeros((), device=self.device)
			aux_count = 0

			for t_aux in range(aux_h):
				aux_t = self.model.temporal_future_aux_loss(
					obs_np[t_aux],
					action[t_aux],
					obs_np[t_aux + 1]
				)
				aux_accum = aux_accum + (self.cfg.rho ** t_aux) * aux_t
				aux_count += 1

			if aux_count > 0:
				temporal_future_aux_loss = aux_accum / aux_count

		# ---------- Aux2: 中间时刻插值 / 掩码重建 ----------
		if getattr(self.cfg, 'use_temporal_interp_aux', False):
			# 需要 t, t+1, t+2 三个时刻，因此最大只能到 horizon-1
			aux_h = min(int(getattr(self.cfg, 'temporal_interp_aux_horizon', 1)), max(self.cfg.horizon - 1, 0))
			aux_accum = torch.zeros((), device=self.device)
			aux_count = 0

			for t_aux in range(aux_h):
				aux_t = self.model.temporal_interp_aux_loss(
					obs_np[t_aux],
					obs_np[t_aux + 1],
					obs_np[t_aux + 2]
				)
				aux_accum = aux_accum + (self.cfg.rho ** t_aux) * aux_t
				aux_count += 1

			if aux_count > 0:
				temporal_interp_aux_loss = aux_accum / aux_count

		# ---------- Aux3: 风险代理预测 ----------
		if getattr(self.cfg, 'use_risk_proxy_aux', False):
			aux_h = min(int(getattr(self.cfg, 'risk_proxy_aux_horizon', 1)), self.cfg.horizon)
			aux_accum = torch.zeros((), device=self.device)
			aux_count = 0

			for t_aux in range(aux_h):
				aux_t = self.model.risk_proxy_aux_loss(
					obs_np[t_aux],
					action[t_aux],
					obs_np[t_aux + 1]
				)
				aux_accum = aux_accum + (self.cfg.rho ** t_aux) * aux_t
				aux_count += 1

			if aux_count > 0:
				risk_proxy_aux_loss = aux_accum / aux_count

		# ---------- Aux4: P_safe survival head (BCE) ----------
		collision_aux_loss = torch.zeros((), device=self.device)
		if getattr(self.cfg, 'use_psafe_head', False):
			aux_h = min(int(getattr(self.cfg, 'risk_proxy_aux_horizon', 1)), self.cfg.horizon)
			aux_accum = torch.zeros((), device=self.device)
			aux_count = 0
			for t_aux in range(aux_h):
				aux_t = self.model.collision_aux_loss(obs_np[t_aux], action[t_aux], obs_np[t_aux + 1])
				aux_accum = aux_accum + (self.cfg.rho ** t_aux) * aux_t
				aux_count += 1
			if aux_count > 0:
				collision_aux_loss = aux_accum / aux_count

		aux_total = (
			float(getattr(self.cfg, 'temporal_future_aux_weight', 0.05)) * temporal_future_aux_loss +
			float(getattr(self.cfg, 'temporal_interp_aux_weight', 0.03)) * temporal_interp_aux_loss +
			float(getattr(self.cfg, 'risk_proxy_aux_weight', 0.03)) * risk_proxy_aux_loss +
			float(getattr(self.cfg, 'psafe_head_weight', 0.05)) * collision_aux_loss
		)

		if self.cfg.use_multi_sep_dyn:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.consistency_coef * consistency_laser_goal_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * value_loss +
				aux_total
			)
		elif self.cfg.use_decoder:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.consistency_coef * consistency_s_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * value_loss +
				aux_total
			)
		else:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * value_loss +
				aux_total
			)

		
		

		# Update model
		t1_backward = time.time()
		total_loss.backward()
		t2_backward = time.time()
		# for name, param in self.model.named_parameters():
		# 	if param.grad is not None:
		# 		print(f"Gradient for {name}: {param.grad}")
		# 	else:
		# 		print(f"No gradient for {name}")
		print("Total Loss Backward Time : {} ms".format(round(1000*(t2_backward-t1_backward),2)))
	
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# check_goal = self.model.encode(torch.tensor([3, 0.5]).to(self.device), task)
		# print("check_goal", check_goal)
		# two parts of optim
		# self.enc_dyn_optim.zeor_grad(set_to_none=True)
		# consistency_loss.backward()
		# grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		# self.enc_dyn_optim.step()
		
		
		# self.r_q_optim.zeor_grad(set_to_none=True)
		# r_q_loss = self.cfg.reward_coef * reward_loss +
		# 	self.cfg.value_coef * value_loss
		# r_q_loss.backward()
		# grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		# grad_norm = 0.0
		# self.r_q_optim.step()

		# Update policy
		if self.cfg.update_sac_pi:
			pi_loss = self.update_sac_pi(zs.detach(), task)
		else:
			if self.cfg.actor_mode=="sac":
				pi_loss = self.update_pi(zs.detach(), action.detach(), task)
			else:
				pi_loss = self.update_pi(_zs.detach(), action.detach(), task)
		# Update check traj
		if self.cfg.check_traj:
			check_cons_loss = 0
			check_obs = obs[0, :, self.obs_fix_laser_dim+self.obs_fix_goal_dim:]
			print("check_obs", check_obs.grad_fn)
			print("action", action.grad_fn)
			pred_traj = [check_obs[0]]
			true_traj = obs[:, 0, self.obs_fix_laser_dim+self.obs_fix_goal_dim:]
			for t in range(self.cfg.horizon):
				check_obs = self.model.check_next(check_obs, action[t])  # 使用当前状态和控制动作
				check_cons_loss += F.mse_loss(check_obs, obs[t+1, :, self.obs_fix_laser_dim+self.obs_fix_goal_dim:])* self.cfg.rho**t
				pred_traj.append(check_obs[0])
			pred_traj = torch.stack(pred_traj, dim=0)
			print('pred traj', pred_traj)
			print('true_traj', true_traj)
			# 反向传播
			self.check_optim.zero_grad()
			check_cons_loss.backward()
			self.check_optim.step()

			if self.train_step % 200 == 0:
				self.plot_traj(true_traj, pred_traj, self.train_step, writer)
			writer.add_scalar('Check_consistency_loss', float(check_cons_loss.mean().item()), self.train_step)

		if self.cfg.visualize_dyn_obs and self.cfg.use_decoder:
			if self.train_step % 200 == 0:
				self.visualize_predictions(0, 'real_motion', action, decode_obs_s, obs[1:], reward_preds, reward, writer, self.train_step)
		
		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()

		writer.add_scalar('Consistency_loss', float(consistency_loss.mean().item()), self.train_step)
		if self.cfg.use_multi_sep_dyn:
			writer.add_scalar('Consistency_laser_goal_loss', float(consistency_laser_goal_loss.mean().item()), self.train_step)
		if self.cfg.use_decoder:
			writer.add_scalar('Consistency_s_loss', float(consistency_s_loss.mean().item()), self.train_step)

		if getattr(self.cfg, 'use_temporal_future_aux', False):
			writer.add_scalar('Aux/temporal_future_loss', float(temporal_future_aux_loss.detach().cpu().item()), self.train_step)

		if getattr(self.cfg, 'use_temporal_interp_aux', False):
			writer.add_scalar('Aux/temporal_interp_loss', float(temporal_interp_aux_loss.detach().cpu().item()), self.train_step)

		if getattr(self.cfg, 'use_risk_proxy_aux', False):
			writer.add_scalar('Aux/risk_proxy_loss', float(risk_proxy_aux_loss.detach().cpu().item()), self.train_step)
		
		if writer is not None:
			writer.add_scalar('Aux/collision_bce', float(collision_aux_loss.detach().cpu().item()), self.train_step)

		writer.add_scalar('Reward_loss', float(reward_loss.mean().item()), self.train_step)
		writer.add_scalar('Value_loss', float(value_loss.mean().item()), self.train_step)
		writer.add_scalar('Pi_loss', pi_loss, self.train_step)
		writer.add_scalar('Total_loss', float(total_loss.mean().item()), self.train_step)
		writer.add_scalar('Grad_norm', float(grad_norm), self.train_step)
		writer.add_scalar('Pi_scale', float(self.scale.value), self.train_step)
		if self.cfg.update_sac_pi:
			writer.add_scalar('Alpha', self.alpha.detach().item(), self.train_step)
		# --------------------------------------------------
		# gate / dual-branch 调试日志
		# --------------------------------------------------
		if getattr(self.cfg, 'hybrid_log_gate_stats', True):
			gate_stats = self.model.get_encoder_debug_stats()

			if len(gate_stats) > 0:
				if 'alpha' in gate_stats:
					writer.add_scalar('Gate/alpha', float(gate_stats['alpha']), self.train_step)
				if 'gate_step' in gate_stats:
					writer.add_scalar('Gate/gate_step', float(gate_stats['gate_step']), self.train_step)
				if 'gate_mean' in gate_stats:
					writer.add_scalar('Gate/gate_mean', float(gate_stats['gate_mean']), self.train_step)
				if 'gate_std' in gate_stats:
					writer.add_scalar('Gate/gate_std', float(gate_stats['gate_std']), self.train_step)
				if 'gate_min' in gate_stats:
					writer.add_scalar('Gate/gate_min', float(gate_stats['gate_min']), self.train_step)
				if 'gate_max' in gate_stats:
					writer.add_scalar('Gate/gate_max', float(gate_stats['gate_max']), self.train_step)
				if 'env_complexity_mean' in gate_stats:
					writer.add_scalar('Gate/env_complexity_mean', float(gate_stats['env_complexity_mean']), self.train_step)
				if 'delta_norm' in gate_stats:
					writer.add_scalar('Gate/delta_norm', float(gate_stats['delta_norm']), self.train_step)
				if 'combine_norm' in gate_stats:
					writer.add_scalar('Gate/combine_norm', float(gate_stats['combine_norm']), self.train_step)
				if 'delta_over_combine' in gate_stats:
					writer.add_scalar('Gate/delta_over_combine', float(gate_stats['delta_over_combine']), self.train_step)
		
		# --------------------------------------------------
		# temporal attention debug stats
		# --------------------------------------------------
		if getattr(self.cfg, 'hybrid_log_attn_stats', True):
			attn_stats = self.model.get_temporal_attn_debug_stats()
			if len(attn_stats) > 0:
				if 'attn_diag_mean' in attn_stats:
					writer.add_scalar('Attn/diag_mean', float(attn_stats['attn_diag_mean']), self.train_step)
				if 'attn_last_to_last' in attn_stats:
					writer.add_scalar('Attn/last_to_last', float(attn_stats['attn_last_to_last']), self.train_step)
				if 'attn_last_to_first' in attn_stats:
					writer.add_scalar('Attn/last_to_first', float(attn_stats['attn_last_to_first']), self.train_step)
				if 'attn_entropy' in attn_stats:
					writer.add_scalar('Attn/entropy', float(attn_stats['attn_entropy']), self.train_step)
				if 'temporal_attn_disabled' in attn_stats:
					writer.add_scalar('Attn/disabled', float(attn_stats['temporal_attn_disabled']), self.train_step)

		# --------------------------------------------------
		# temporal T=1 debug stats
		# --------------------------------------------------
		if getattr(self.cfg, 'hybrid_log_temporal_t1', True):
			t1_stats = self.model.get_temporal_t1_debug_stats()
			if len(t1_stats) > 0:
				if 'temporal_frames_used' in t1_stats:
					writer.add_scalar('TemporalT1/frames_used', float(t1_stats['temporal_frames_used']), self.train_step)
				if 'temporal_t1_enabled' in t1_stats:
					writer.add_scalar('TemporalT1/enabled', float(t1_stats['temporal_t1_enabled']), self.train_step)
				if 'temporal_t1_index' in t1_stats:
					writer.add_scalar('TemporalT1/index', float(t1_stats['temporal_t1_index']), self.train_step)

		# 额外记录 combine 分支当前是否冻结
		writer.add_scalar('Gate/combine_branch_frozen', 1.0 if self.combine_branch_frozen else 0.0, self.train_step)
		t2 = time.time()
		print("Train Step : {}  Time : {}  ms".format(self.train_step, round(1000*(t2-t1),2)))

		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}
	
	def update_org(self, buffer, writer, total_step):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		self.train_step += 1

		if total_step <= self.cfg.random_exploration_length:
			return self.train_step
		t1 = time.time()
		obs, action, reward, task = buffer.sample()
	
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		
		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()

		writer.add_scalar('Consistency_loss', float(consistency_loss.mean().item()), self.train_step)
		writer.add_scalar('Reward_loss', float(reward_loss.mean().item()), self.train_step)
		writer.add_scalar('Value_loss', float(value_loss.mean().item()), self.train_step)
		writer.add_scalar('Pi_loss', pi_loss, self.train_step)
		writer.add_scalar('Grad_norm', float(grad_norm), self.train_step)
		writer.add_scalar('Pi_scale', float(self.scale.value), self.train_step)
		t2 = time.time()
		print("Train Step : {}  Time : {}  ms".format(self.train_step, round(1000*(t2-t1),2)))

		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}
	
	def update_nomodel(self, buffer, writer, total_step):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		self.train_step += 1

		if total_step <= self.cfg.random_exploration_length:
			return self.train_step
		
		t1 = time.time()
		# check buffer sample is continuous and in the same episode or not ???
		obs, action, reward, not_done, task = buffer.sample()
		print('buffer shape obs action reward', obs.shape, action.shape, reward.shape) # [h+1, batch_size, obs_dim] [h, batch_size, action_dim] [h, batch_size, 1]
		if self.cfg.reward_model_reduce:
			reward_target = reward.clone()
			reward_target[(reward_target > self.cfg.reward_done - 1) | (reward_target < - self.cfg.reward_done + 1)] /= self.cfg.reward_done
			# print('reward', reward)
			# print("reward - reward_target", reward - reward_target)
			obs, action, reward, not_done, task, reward_target = buffer._to_device(obs, action, reward, not_done, task, reward_target)
		# obs, action, reward, not_done = obs[2:4], action[2:3], reward[2:3], not_done[2:3]
		else:
			obs, action, reward, not_done, task = buffer._to_device(obs, action, reward, not_done, task)
		
		# obs_q, action_q, reward_q, not_done_q = obs[:2], action[:1], reward[:1], not_done[:1]

		print("obs grad_fn:", obs.grad_fn)
		print("action grad_fn:", action.grad_fn)
		print("reward grad_fn:", reward.grad_fn)

		#==========================>TDMPC ORG<==========================
		# with torch.no_grad():
		# 	next_z = self.model.encode(obs[1:], task)
		# 	td_targets = self._td_target(next_z, reward, task)

		# # Prepare for update
		# self.optim.zero_grad(set_to_none=True)
		# self.model.train()

		# # Latent rollout
		# zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		# z = self.model.encode(obs[0], task)
		# zs[0] = z
		# consistency_loss = 0
		# for t in range(self.cfg.horizon):
		# 	z = self.model.next(z, action[t], task)
		# 	consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
		# 	zs[t+1] = z

		# # Predictions
		# _zs = zs[:-1]
		# qs = self.model.Q(_zs, action, task, return_type='all')
		# reward_preds = self.model.reward(_zs, action, task)
		
		# # Compute losses
		# reward_loss, value_loss = 0, 0
		# for t in range(self.cfg.horizon):
		# 	reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
		# 	for q in range(self.cfg.num_q):
		# 		value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		#==========================>TDMPC ORG<==========================

		current_z = self.model.encode(obs[0], task)
		next_z = self.model.encode(obs[1:], task)
		# next_z_q = self.model.encode(obs_q[1:], task)
		# Compute targets
		# with torch.no_grad():
		# 	_, next_action, next_logprob, _ = self.model.pi(next_z_q, task)
		# 	target_q = self.model.Q(next_z_q, next_action, task, return_type='min', target=True) - self.alpha * next_logprob
		# 	print("target_q r done", target_q.shape, reward.shape, not_done.shape)
		# 	target_q = reward_q + self.discount * not_done_q * target_q if self.cfg.use_done else reward + self.discount * target_q
		with torch.no_grad():
			_, next_action, next_logprob, _ = self.model.pi(next_z, task)
			target_q = self.model.Q(next_z, next_action, task, return_type='min', target=True) - self.alpha * next_logprob
			print("target_q r done", target_q.shape, reward.shape, not_done.shape)
			target_q = reward + self.discount * not_done * target_q if self.cfg.use_done else reward + self.discount * target_q
			
		
		# Prepare for update q
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		decode_obs_s = torch.empty(self.cfg.horizon, self.cfg.batch_size, self.cfg.obs_shape['state'][0], device=self.device)
		zs[0] = current_z

		'''try to use real z instead of dyn z when h>1'''
		# zs_sample = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		# zs_sample[0] = current_z
		if self.cfg.use_real_z:
			zs[1:] = next_z

		'''dyn'''
		consistency_loss = 0
		consistency_s_loss = 0
		for t in range(self.cfg.horizon):
			current_z = self.model.next(current_z, action[t], task)
			consistency_loss += F.mse_loss(current_z, next_z[t]) * self.cfg.rho**t
			if not self.cfg.use_real_z:
				zs[t+1] = current_z

			if self.cfg.use_decoder:
				if self.cfg.dec_detach:
					decode_obs = self.model.decode(current_z.detach())
				else:
					decode_obs = self.model.decode(current_z)
				consistency_s_loss += F.mse_loss(decode_obs, obs[t+1].to(torch.float32)) * self.cfg.rho**t
				decode_obs_s[t] = decode_obs
		_current_z = zs[:-1]
		# _current_z = zs[:1]

		'''q'''
		# check no dyn mode h=3 only update q and pi
		q_loss = 0
		mse_loss_fn = torch.nn.MSELoss()
		# action[:1] action_q
		if self.cfg.soft_ce_q:
			print("soft ce q loss")
			qs = self.model.Q(_current_z, action, task, return_type='all')
			for t in range(self.cfg.horizon):
				for q in range(self.cfg.num_q):
					q_loss += math.soft_ce(qs[q][t], target_q[t], self.cfg).mean() * self.cfg.rho**t
		else:
			current_q1, current_q2 = self.model.Q(_current_z, action, task, return_type='sep')
			print("mse q loss")
			print("q1 q2 tq", current_q1.shape, current_q2.shape, target_q.shape)
			for t in range(self.cfg.horizon):
				q_loss += (mse_loss_fn(current_q1[t], target_q[t]) + mse_loss_fn(current_q2[t], target_q[t])) * self.cfg.rho**t
		
		'''reward'''
		reward_preds = self.model.reward(_current_z, action, task)
		reward_loss = 0
		weight = 1.0  # 对普通奖励的权重
		weight_special = 0.1  # 对成功/失败奖励的权重
		for t in range(self.cfg.horizon):
			if self.cfg.weight_reward_loss:
				# 对每个时间步，检查reward[t]是否是特殊奖励（100或-100）
				is_special = (reward[t] == -self.cfg.reward_done) | (reward[t] == self.cfg.reward_done)  # 返回 batch_size 大小的布尔向量
				print("is special", is_special)
				# 将布尔值转换为0和1的浮点数，并根据其应用不同的权重
				weights = is_special.float() * weight_special + (1 - is_special.float()) * weight
				
				# 计算损失并加权
				reward_loss += (math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * weights).mean() * self.cfg.rho**t
			elif self.cfg.reward_model_reduce:
				if self.cfg.use_mse_r:
					reward_loss += mse_loss_fn(reward_preds[t], reward_target[t]) * self.cfg.rho**t
				else:	
					reward_loss += math.soft_ce(reward_preds[t], reward_target[t], self.cfg).mean() * self.cfg.rho**t
			else:
				if self.cfg.use_mse_r:
					print('use mse loss for reward org')
					reward_loss += mse_loss_fn(reward_preds[t], reward[t]) * self.cfg.rho**t
				else:	
					print('use soft ce for reward org')
					reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
		

		

		'''dyn r q backward'''
		consistency_loss *= (1/self.cfg.horizon)
		consistency_s_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		if self.cfg.soft_ce_q:
			q_loss *= (1/(self.cfg.horizon*self.cfg.num_q))
		else:
			q_loss *= (1/(self.cfg.horizon))
		
		if self.cfg.use_decoder:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.consistency_coef * consistency_s_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * q_loss
			)
		else:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * q_loss
			)
		total_loss.backward()
	
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		'''pi'''
		# Prepare for update pi
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		_, current_action, current_logprob, _ = self.model.pi(_current_z.detach(), task)
		qs = self.model.Q(_current_z.detach(), current_action, task, return_type='avg')
		if self.cfg.norm_q:
			self.scale.update(qs[0])
			qs = self.scale(qs)
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((-qs + self.alpha * current_logprob).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		'''alpha'''
		# Prepare for update alpha
		# 自动调整 α（借鉴 SAC）
		alpha_loss = self.log_alpha.exp() * (-current_logprob.mean() - self.target_entropy).detach()
		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()
		# 计算动态 α
		self.alpha = self.log_alpha.exp().detach()

		if self.cfg.visualize_dyn_obs and not self.cfg.dyn_state:
			if self.train_step % 200 == 0:
				self.visualize_predictions(0, 'real_motion', action, decode_obs_s, obs[1:], reward_preds, reward, writer, self.train_step)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()

		

		if self.cfg.use_decoder:
			writer.add_scalar('Consistency_s_loss', float(consistency_s_loss.mean().item()), self.train_step)
		writer.add_scalar('Consistency_loss', float(consistency_loss.mean().item()), self.train_step)
		writer.add_scalar('Reward_loss', float(reward_loss.mean().item()), self.train_step)
		writer.add_scalar('Value_loss', float(q_loss.mean().item()), self.train_step)
		writer.add_scalar('Pi_loss', pi_loss, self.train_step)
		writer.add_scalar('Grad_norm', float(grad_norm), self.train_step)
		writer.add_scalar('Pi_scale', float(self.scale.value), self.train_step)
		if self.cfg.update_sac_pi:
			writer.add_scalar('Alpha', self.alpha.detach().item(), self.train_step)
		t2 = time.time()
		print("Train Step : {}  Time : {}  ms".format(self.train_step, round(1000*(t2-t1),2)))

		return {
			"value_loss": float(q_loss.mean().item()),
			"pi_loss": pi_loss,
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}

	def update_sep(self, buffer, writer, total_step):
		"""
		Main update function. Corresponds to one iteration of model learning.
		consistency reward and q upadte separately.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		self.train_step += 1

		if total_step <= self.cfg.random_exploration_length:
			return self.train_step
		
		t1 = time.time()
		# check buffer sample is continuous and in the same episode or not ???
		obs, action, reward, task = buffer.sample()
		print('now update sep!')
		print('buffer shape obs action reward', obs.shape, action.shape, reward.shape) # [h+1, batch_size, obs_dim] [h, batch_size, action_dim] [h, batch_size, 1]

		obs_np = obs.cpu().numpy()

		# Compute targets
		with torch.no_grad():
			if self.cfg.mlp_obs:
				next_z = self.model.encode(obs[1:], task)
			else:
				next_z = self.model.multi_encode(obs_np[1]).unsqueeze(0)
				
				for i in range(1, self.cfg.horizon):
					next_z_single = self.model.multi_encode(obs_np[i+1]).unsqueeze(0)
					next_z = torch.cat((next_z, next_z_single), dim=0)
				
			td_targets = self._td_target(next_z, reward, task)
	
		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)

		# z = self.model.encode(obs[0], task)
		z = self.model.multi_encode(obs_np[0])
	
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			t1_dynamics = time.time()
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t+1] = z
			t2_dynamics = time.time()
			# print("Dynamics Single Step Time : {}  ms".format(round(1000*(t2_dynamics-t1_dynamics),2)))

		# Update dyn enc model
		consistency_loss *= (self.cfg.consistency_coef/self.cfg.horizon)
		self.enc_dyn_optim.zero_grad(set_to_none=True)
		consistency_loss.backward()
		grad_norm_cons = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.enc_dyn_optim.step()

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs.detach(), action, task, return_type='all')
		reward_preds = self.model.reward(_zs.detach(), action, task)
		
		# Compute losses
		reward_loss, value_loss, path_loss = 0, 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			# print('reward loss ', reward_loss)
			for q in range(self.cfg.num_q):

				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
				# print('value loss ', value_loss)
		
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		print("consistency reward value loss: ", consistency_loss, reward_loss, value_loss)

		# Update r q model
		r_q_loss = self.cfg.reward_coef * reward_loss +self.cfg.value_coef * value_loss
		self.r_q_optim.zero_grad(set_to_none=True)
		r_q_loss.backward()
		grad_norm_rq = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.r_q_optim.step()

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()

		writer.add_scalar('Consistency_loss', float(consistency_loss.mean().item()), self.train_step)
		writer.add_scalar('Reward_loss', float(reward_loss.mean().item()), self.train_step)
		writer.add_scalar('Value_loss', float(value_loss.mean().item()), self.train_step)
		writer.add_scalar('Pi_loss', pi_loss, self.train_step)
		writer.add_scalar('Grad_norm_cons', float(grad_norm_cons), self.train_step)
		writer.add_scalar('Grad_norm_rq', float(grad_norm_rq), self.train_step)
		writer.add_scalar('Pi_scale', float(self.scale.value), self.train_step)

		t2 = time.time()
		print("Train Step : {}  Time : {}  ms".format(self.train_step, round(1000*(t2-t1),2)))

		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"pi_scale": float(self.scale.value),
		}
	
	def update_2enc(self, buffer, writer, total_step):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		self.train_step += 1

		if total_step <= self.cfg.random_exploration_length:
			return self.train_step

		t1 = time.time()
		# check buffer sample is continuous and in the same episode or not ???
		obs, action, reward, task = buffer.sample()
		print('buffer shape obs action reward', obs.shape, action.shape, reward.shape) # [h+1, batch_size, obs_dim] [h, batch_size, action_dim] [h, batch_size, 1]
		print('org obs', obs[:, 0, :])
		for i in range(self.cfg.horizon):
			# x -4 y -3 sin -2 cos -1
			# + yaw
			x_trans = obs[0, :, -1] * (obs[i+1, :, -4]-obs[0, :, -4]) + obs[0, :, -2] * (obs[i+1, :, -3]-obs[0, :, -3])
			y_trans = -obs[0, :, -2] * (obs[i+1, :, -4]-obs[0, :, -4]) + obs[0, :, -1] * (obs[i+1, :, -3]-obs[0, :, -3])
			
			# change for test
			# org sin_trans = obs[i+1, :, -2]*obs[0, :, -1] - obs[i+1, :, -1]*obs[0, :, -2] (+sin)
			sin_trans = obs[i+1, :, -2]*obs[0, :, -1] - obs[i+1, :, -1]*obs[0, :, -2]
			cos_trans = obs[i+1, :, -1]*obs[0, :, -1] + obs[i+1, :, -2]*obs[0, :, -2]
			
			if i < self.cfg.horizon-1:
				traj_x_trans = x_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*cos_trans - action[i+1, :, 1]*sin_trans)
				traj_y_trans = y_trans/self.cfg.fix_max_traj + (action[i+1, :, 0]*sin_trans + action[i+1, :, 1]*cos_trans)
				
			# test goal information: +-robot_yaw sin cos trans 
			# goal_x = obs[i+1, :, -8] * torch.cos(obs[i+1, :, -7])
			# goal_y = obs[i+1, :, -8] * torch.sin(obs[i+1, :, -7])
			# goal_x_trans = x_trans + (goal_x*cos_trans - goal_y*sin_trans)
			# goal_y_trans = y_trans + (goal_x*sin_trans + goal_y*cos_trans)
			# goal_dis_trans = torch.hypot(goal_x_trans, goal_y_trans)
			# goal_theta_trans = torch.atan2(goal_y_trans, goal_x_trans)
			# obs[i+1, :, -8] = goal_dis_trans
			# obs[i+1, :, -7] = goal_theta_trans
			
			obs[i+1, :, -4] = x_trans
			obs[i+1, :, -3] = y_trans
			obs[i+1, :, -2] = sin_trans
			obs[i+1, :, -1] = cos_trans 

			if i < self.cfg.horizon-1:
				action[i+1, :, 0] = traj_x_trans
				action[i+1, :, 1] = traj_y_trans

		obs[0, :, -4] = 0.0
		obs[0, :, -3] = 0.0
		obs[0, :, -2] = 0.0
		obs[0, :, -1] = 1.0
		if self.cfg.same_obs and self.cfg.same_goal:
			obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim] = obs[0, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim]
		print("trans obs", obs[:, 0, :])
		# grad_fn: None
		obs, action, reward, task = buffer._to_device(obs, action, reward, task)

		# Compute targets
		with torch.no_grad():
			next_laser = self.model.encode(obs[1:, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
			next_z = self.model.encode_state(obs[1:, :, self.obs_fix_laser_dim+self.obs_fix_goal_dim:], task)   # [horizon, batch_size, 6] -> [horizon, bathc_size, state_latent_dim]
			td_targets = self._td_target(torch.cat([next_laser, next_z], dim=-1), reward, task)	


		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim * self.cfg.enc_num, device=self.device)

		laser_goal_t = self.model.encode(obs[0, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
		
		z = self.model.encode_state(obs[0, :, self.obs_fix_laser_dim+self.obs_fix_goal_dim:], task)

		zs[0] = torch.cat([laser_goal_t, z], dim=-1)

		consistency_loss = 0
		consistency_laser_goal_loss = 0

		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t

			if self.cfg.use_multi_sep_dyn:
				laser_goal_t = self.model.next_obs_sep(laser_goal_t, action[t], task)
				consistency_laser_goal_loss += F.mse_loss(laser_goal_t, next_laser[t]) * self.cfg.rho**t
			
			else:
				laser_goal_t = self.model.encode(obs[t+1, :, :self.obs_fix_laser_dim+self.obs_fix_goal_dim], task)
			
			zs[t + 1] = torch.cat([laser_goal_t, z], dim=-1)
					


		# Predictions
		_zs = zs[:-1]

		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss, path_loss = 0, 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t

			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t

		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		consistency_laser_goal_loss *= (1/self.cfg.horizon)

		if self.cfg.use_multi_sep_dyn:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.consistency_coef * consistency_laser_goal_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * value_loss
			)
		else:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * value_loss
			)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()

		writer.add_scalar('Consistency_loss', float(consistency_loss.mean().item()), self.train_step)
		writer.add_scalar('Consistency_laser_goal_loss', float(consistency_laser_goal_loss.mean().item()), self.train_step)
		writer.add_scalar('Reward_loss', float(reward_loss.mean().item()), self.train_step)
		writer.add_scalar('Value_loss', float(value_loss.mean().item()), self.train_step)
		writer.add_scalar('Pi_loss', pi_loss, self.train_step)
		writer.add_scalar('Total_loss', float(total_loss.mean().item()), self.train_step)
		writer.add_scalar('Grad_norm', float(grad_norm), self.train_step)
		writer.add_scalar('Pi_scale', float(self.scale.value), self.train_step)

		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}
