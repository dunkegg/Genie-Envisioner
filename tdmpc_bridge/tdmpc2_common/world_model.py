from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tdmpc2_common import layers, math, init
from tools.utils import weights_init, CostMap, CostMapDual
from tools.utils_network import FeatureMapper, soft_update, save_models, FeatureExtractor
import cv2
import time


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		# if cfg.use_base_mlp:

		self._encoder = layers.enc(cfg)
		
		if cfg.use_multi_mlp_enc:
			self._encoder_state = layers.enc_state(cfg)
			if cfg.use_obs_multi_enc:
				self._encoder_goal = layers.enc_goal(cfg)
		
		if cfg.use_multi_dyn:
			self._dynamics_goal = layers.mlp(cfg.latent_dim * 2, self.cfg.num_dyn_layers*[cfg.mlp_dim], cfg.latent_dim, use_lnrelu=cfg.use_lnrelu) 
		if cfg.use_multi_sep_dyn:
			self._dynamics_obs_sep = layers.mlp(cfg.latent_dim + cfg.action_embedding_dim, self.cfg.num_dyn_layers*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg), use_lnrelu=cfg.use_lnrelu)
		if cfg.lstm_dyn:
			self._dynamics = layers.DynamicModelLSTM(cfg.latent_dim + cfg.action_embedding_dim, self.cfg.lstm_hidden_dim, cfg.latent_dim, cfg.lstm_hidden_num)
		else:
			if cfg.use_act_dyn:
				if cfg.use_mdn:
					# Shared dynamics encoder
					self._dynamics_core = layers.mlp(cfg.latent_dim + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_dyn_layers*[cfg.mlp_dim], 256, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu)

					# Optional up-projection for deterministic fallback
					self._dynamics_fallback = nn.Linear(256, cfg.latent_dim)

					# MDN dynamic model heads
					self.mdn_pi = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, cfg.num_mixtures)
					)
					self.mdn_mu = nn.Sequential(
						nn.Linear(256, 512),
						nn.ReLU(),
						nn.Linear(512, cfg.num_mixtures * cfg.latent_dim)
					)
					self.mdn_sigma = nn.Sequential(
						nn.Linear(256, 512),
						nn.ReLU(),
						nn.Linear(512, cfg.num_mixtures * cfg.latent_dim)
					)
				elif cfg.use_ensemble_dyn:
					print("use ensemble dyn model (act dyn)")
					self._dynamics = layers.Ensemble([
						layers.mlp(cfg.latent_dim * cfg.dyn_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_dyn_layers * [cfg.mlp_dim], 2 * cfg.latent_dim, act=layers.SimNorm(cfg), use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu)
						for _ in range(self.cfg.ensemble_num)
					])
				else:
					self._dynamics = layers.mlp(cfg.latent_dim * cfg.dyn_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_dyn_layers*[cfg.mlp_dim], cfg.latent_dim*cfg.dyn_num, act=layers.SimNorm(cfg), use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu)
			else:
				if cfg.use_ensemble_dyn:
					print("use ensemble dyn model (act dyn)")
					self._dynamics = layers.Ensemble([
						layers.mlp(cfg.latent_dim * cfg.dyn_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_dyn_layers * [cfg.mlp_dim], 2 * cfg.latent_dim, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu)
						for _ in range(self.cfg.ensemble_num)
					])
				else:
					self._dynamics = layers.mlp(cfg.latent_dim * cfg.dyn_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_dyn_layers*[cfg.mlp_dim], cfg.latent_dim*cfg.dyn_num, use_mish=cfg.use_mish, use_lnrelu=cfg.use_lnrelu)
		
		if cfg.use_decoder:
			self._decoder = layers.mlp(cfg.latent_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.obs_shape['state'][0], use_lnrelu=cfg.use_lnrelu)
		
		if self.cfg.use_mse_r:
			self._reward = layers.mlp(cfg.latent_dim * cfg.enc_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_reward_layers*[cfg.mlp_dim], 1, use_lnrelu=cfg.use_lnrelu)
		else:
			self._reward = layers.mlp(cfg.latent_dim * cfg.enc_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_reward_layers*[cfg.mlp_dim], max(cfg.num_bins, 1), use_lnrelu=cfg.use_lnrelu)
		if self.cfg.pi_normln:
			self._pi = layers.mlp(cfg.latent_dim * cfg.enc_num + cfg.task_dim, self.cfg.num_pi_layers*[cfg.mlp_dim], 2*cfg.action_dim, use_lnrelu=False)
		else:
			self._pi = layers.mlp(cfg.latent_dim * cfg.enc_num + cfg.task_dim, self.cfg.num_pi_layers*[cfg.mlp_dim], 2*cfg.action_dim, use_lnrelu=cfg.use_lnrelu)
		
		if not cfg.use_hot_q:
			self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim * cfg.enc_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_q_layers*[cfg.mlp_dim], 1, dropout=cfg.dropout, use_lnrelu=cfg.use_lnrelu) for _ in range(cfg.num_q)])
		else:
			self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim * cfg.enc_num + cfg.action_embedding_dim + cfg.task_dim, self.cfg.num_q_layers*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout, use_lnrelu=cfg.use_lnrelu) for _ in range(cfg.num_q)])
		
		if not cfg.mlp_obs:
			self.fe = FeatureExtractor(cfg)

		# ==================================================
        # training-only: temporal future auxiliary
        # 用当前 temporal correction feature + 当前动作，预测下一时刻 temporal feature
        # ==================================================
		self.use_temporal_future_aux = bool(getattr(cfg, 'use_temporal_future_aux', False))
		self.temporal_future_aux_normalize = bool(getattr(cfg, 'temporal_future_aux_normalize', True))

		action_dim = int(getattr(cfg, 'action_dim', 2))

		# if self.cfg.eval_preset != 'baseline' or not self.cfg.eval_mode:
		self.temporal_future_predictor = nn.Sequential(
			nn.Linear(256 + action_dim, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 256)
		)

		# ==================================================
		# training-only: temporal interpolation auxiliary
		# 用 f_t 和 f_{t+2} 预测中间时刻 f_{t+1}
		# ==================================================
		self.use_temporal_interp_aux = bool(getattr(cfg, 'use_temporal_interp_aux', False))
		self.temporal_interp_aux_normalize = bool(getattr(cfg, 'temporal_interp_aux_normalize', True))

		if (self.cfg.eval_preset != 'baseline' and self.cfg.eval_preset != 'd3' and self.cfg.eval_preset != 'd3_scalar') or not self.cfg.eval_mode:
			self.temporal_interp_predictor = nn.Sequential(
				nn.Linear(256 + 256, 256),
				nn.LeakyReLU(),
				nn.Linear(256, 256)
			)

		# ==================================================
		# training-only: risk proxy auxiliary
		# 用当前 temporal feature + 当前动作，预测下一时刻风险代理
		# 若 multi_roi，则目标维度为 5；否则为 1
		# ==================================================
		self.use_risk_proxy_aux = bool(getattr(cfg, 'use_risk_proxy_aux', False))
		self.risk_proxy_aux_horizon = int(getattr(cfg, 'risk_proxy_aux_horizon', 1))

		env_mode = getattr(cfg, 'hybrid_env_feature_mode', 'scalar')
		self.risk_proxy_dim = 5 if env_mode == 'multi_roi' else 1

		if (self.cfg.eval_preset != 'baseline' and self.cfg.eval_preset != 'd3' and self.cfg.eval_preset != 'd3_scalar') or not self.cfg.eval_mode:
			self.risk_proxy_predictor = nn.Sequential(
				nn.Linear(256 + action_dim, 128),
				nn.LeakyReLU(),
				nn.Linear(128, self.risk_proxy_dim)
			)

		# ==================================================
		# Phase B: deterministic survival head  p_unsafe(z, a) in (0,1)
		# 输入约定与 _reward 完全一致，保证维度兼容
		# ==================================================
		self.use_psafe_head = bool(getattr(cfg, 'use_psafe_head', False))
		self.psafe_label_soft = bool(getattr(cfg, 'psafe_label_soft', False))
		self.psafe_label_thr = float(getattr(cfg, 'psafe_label_thr', 0.5))
		self.psafe_head_detach = bool(getattr(cfg, 'psafe_head_detach', False))
		if self.use_psafe_head:
			self._collision = layers.mlp(
				cfg.latent_dim * cfg.enc_num + cfg.action_embedding_dim + cfg.task_dim,
				self.cfg.num_reward_layers * [cfg.mlp_dim], 1,
				use_lnrelu=cfg.use_lnrelu)

		# self.action_embedding = nn.Linear(cfg.action_dim, cfg.action_embedding_dim)
		# self._path_predict = layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], cfg.path_dim)
		if cfg.check_traj:
			if cfg.use_dxdydtheta:
				self.check_dim = 9 if cfg.check_goal else 7
			else:
				self.check_dim = 8 if cfg.check_goal else 6
			self._check_enc = layers.mlp(self.check_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim)
			self._check_dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim, self.cfg.num_dyn_layers*[cfg.mlp_dim], self.check_dim)

		self.apply(init.weight_init)
		
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		# if self.use_psafe_head:
		# 	init.zero_([self._collision[-1].weight])

		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min
		
		self.surrouding_length = self.cfg.state_dim - self.cfg.vel_dim - self.cfg.goal_dim - self.cfg.path_dim - self.cfg.progress_dim
		self.laser_dim = self.cfg.laser_dim
		self.laser_range = self.cfg.laser_range
		self.true_laser_range = self.cfg.true_laser_range
		self.img_width = self.cfg.img_width
		self.img_height = self.cfg.img_height
		self.observation_dim = self.cfg.observation_dim
		self.highlight_iterations = self.cfg.highlight_iterations
		self.merge_vis = self.cfg.merge_vis
		self.deviation_mode = self.cfg.deviation_mode
		self.no_goal = self.cfg.no_goal
		self.progress_ratio_mode = self.cfg.progress_ratio_mode
		# 新增开关：是否启用 hybrid temporal-attention encoder
		self.use_hybrid_attn_encoder = bool(getattr(self.cfg, 'use_hybrid_attn_encoder', False))
		# 新增开关：是否启用双分支 encoder
		self.use_dual_branch_encoder = bool(getattr(self.cfg, 'use_dual_branch_encoder', False))
		# 新增：图像 backbone 路线选择
		self.image_encoder_backbone = getattr(self.cfg, 'image_encoder_backbone', 'cnn')
		self.vit_input_mode = getattr(self.cfg, 'vit_input_mode', 'temporal')
		# there are some questions here
		# self.fe = FeatureExtractor(cfg)
		# self.action_embedding = nn.Linear(cfg.action_dim, cfg.action_embedding_dim)
		if self.cfg.diff_action_embedding:
			# if use diff embedding, need to add another two layers
			self.action_embedding_q = nn.Linear(cfg.action_dim, cfg.action_embedding_dim)
			self.action_embedding_r = nn.Linear(cfg.action_dim, cfg.action_embedding_dim)
		

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def set_combine_branch_trainable(self, trainable=True):
		"""
		供 tdmpc2.py 调用：
		在 dual-branch 模式下，冻结 / 解冻 combine 分支。
		"""
		if hasattr(self, 'fe') and hasattr(self.fe, 'surrounding_embedding'):
			enc = self.fe.surrounding_embedding
			if hasattr(enc, 'set_combine_branch_requires_grad'):
				enc.set_combine_branch_requires_grad(trainable)

	def get_encoder_debug_stats(self):
		"""
		供 tdmpc2.py 调用：
		读取最近一次 forward 缓存下来的 gate 调试统计。
		"""
		if hasattr(self, 'fe') and hasattr(self.fe, 'surrounding_embedding'):
			enc = self.fe.surrounding_embedding
			if hasattr(enc, 'get_latest_gate_stats'):
				return enc.get_latest_gate_stats()
		return {}

	def get_temporal_attn_debug_stats(self):
		"""
		从 surrounding encoder 里取 temporal attention 的 debug 统计。
		"""
		if hasattr(self, 'fe') and hasattr(self.fe, 'surrounding_embedding'):
			enc = self.fe.surrounding_embedding
			if hasattr(enc, 'get_latest_temporal_attn_stats'):
				return enc.get_latest_temporal_attn_stats()
		return {}

	def get_temporal_t1_debug_stats(self):
		"""
		从 surrounding encoder 里取当前 temporal branch 使用的帧数 / 帧索引。
		"""
		if hasattr(self, 'fe') and hasattr(self.fe, 'surrounding_embedding'):
			enc = self.fe.surrounding_embedding
			if hasattr(enc, 'get_latest_temporal_t1_stats'):
				return enc.get_latest_temporal_t1_stats()
		return {}

	# def get_latest_branch_feats(self):
	# 	"""
    #     从 surrounding encoder 里取最近一次 forward 的 branch features
    #     """
	# 	if hasattr(self, 'fe') and hasattr(self.fe, 'surrounding_embedding'):
	# 		enc = self.fe.surrounding_embedding
	# 		if hasattr(enc, 'get_latest_branch_feats'):
	# 			return enc.get_latest_branch_feats()
	# 	return {}

	def get_latest_branch_feats(self):
		"""
		从 surrounding encoder 里取最近一次 forward 的分支特征和环境代理目标
		"""
		if hasattr(self, 'fe') and hasattr(self.fe, 'surrounding_embedding'):
			enc = self.fe.surrounding_embedding
			if hasattr(enc, 'get_latest_branch_feats'):
				return enc.get_latest_branch_feats()
		return {}
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""


		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)
		if self.cfg.multitask:
			for p in self._task_emb.parameters():
				p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def multi_encode(self, state, id=0, if_batch=True):
		"""
		当前 encoder 三种模式：
		1) 原始模式:
			use_hybrid_attn_encoder=False, use_dual_branch_encoder=False
			-> CostMap(combine=True) 单张灰度融合图

		2) 单分支 hybrid:
			use_hybrid_attn_encoder=True, use_dual_branch_encoder=False
			-> CostMap(combine=False) 保留 8 帧时间维

		3) 双分支:
			use_hybrid_attn_encoder=True, use_dual_branch_encoder=True
			-> 同时构造 combine 图 和 temporal 图，传给 dual-branch Image_Encoder
		"""
		if self.cfg.mlp_obs:
			return self.encode(torch.from_numpy(state).float().cuda(), None)

		if if_batch:
			ego = torch.from_numpy(state[:, (self.surrouding_length):(self.surrouding_length+2)]).float().cuda()

			if self.no_goal:
				path = torch.from_numpy(state[:, (self.surrouding_length+2):(self.surrouding_length+4)]).float().cuda()
				if self.progress_ratio_mode:
					progress = torch.from_numpy(state[:, (self.surrouding_length+4):(self.surrouding_length+7)]).float().cuda()
			else:
				goal = torch.from_numpy(state[:, (self.surrouding_length+2):(self.surrouding_length+4)]).float().cuda()
				if self.deviation_mode:
					path = torch.from_numpy(state[:, (self.surrouding_length+4):(self.surrouding_length+6)]).float().cuda()
			
			# --------------------------------------------------
			# ViT route
			# --------------------------------------------------
			if self.image_encoder_backbone == 'vit':
				if self.vit_input_mode == 'combine':
					surrounding_np = CostMap(
						state[:, :self.surrouding_length],
						self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
						self.observation_dim, self.highlight_iterations,
						combine=True,
						batch=True,
						hybrid_encoder_mode=False,
					)
					# [B,1,H,W] -> [B,1,1,H,W]
					surrounding = torch.from_numpy(surrounding_np).unsqueeze(1).float().cuda()
				else:
					# temporal mode
					surrounding_np = CostMap(
						state[:, :self.surrouding_length],
						self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
						self.observation_dim, self.highlight_iterations,
						combine=False,
						batch=True,
						hybrid_encoder_mode=False,
					)
					# [B,T,H,W] -> [B,1,T,H,W]
					surrounding = torch.from_numpy(surrounding_np).unsqueeze(1).float().cuda()

			# --------------------------------------------------
			# 原有 CNN 路线
			# --------------------------------------------------
			elif self.use_dual_branch_encoder:
				surrounding_combine_np, surrounding_temporal_np = CostMapDual(
					state[:, :self.surrouding_length],
					self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
					self.observation_dim, self.highlight_iterations,
					batch=True,
					black_obs=True
				)
				surrounding = {
					"combine": torch.from_numpy(surrounding_combine_np).unsqueeze(1).float().cuda(),
					"temporal": torch.from_numpy(surrounding_temporal_np).unsqueeze(1).float().cuda(),
				}

			elif self.use_hybrid_attn_encoder:
				surrounding_np = CostMap(
					state[:, :self.surrouding_length],
					self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
					self.observation_dim, self.highlight_iterations,
					combine=False,
					batch=True,
					hybrid_encoder_mode=False,
				)
				surrounding = torch.from_numpy(surrounding_np).unsqueeze(1).float().cuda()

			else:
				surrounding_np = CostMap(
					state[:, :self.surrouding_length],
					self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
					self.observation_dim, self.highlight_iterations,
					combine=self.merge_vis,
					batch=True,
					hybrid_encoder_mode=False,
				)
				surrounding = torch.from_numpy(surrounding_np).unsqueeze(1).float().cuda()
			
		else:
			ego = torch.from_numpy(state[(self.surrouding_length):(self.surrouding_length+2)]).unsqueeze(0).float().cuda()

			if self.no_goal:
				path = torch.from_numpy(state[(self.surrouding_length+2):(self.surrouding_length+4)]).unsqueeze(0).float().cuda()
				if self.progress_ratio_mode:
					progress = torch.from_numpy(state[(self.surrouding_length+4):(self.surrouding_length+7)]).unsqueeze(0).float().cuda()
			else:
				goal = torch.from_numpy(state[(self.surrouding_length+2):(self.surrouding_length+4)]).unsqueeze(0).float().cuda()
				if self.deviation_mode:
					path = torch.from_numpy(state[(self.surrouding_length+4):(self.surrouding_length+6)]).unsqueeze(0).float().cuda()

			# --------------------------------------------------
			# ViT route
			# --------------------------------------------------
			if self.image_encoder_backbone == 'vit':
				if self.vit_input_mode == 'combine':
					surrounding_np = CostMap(
						state[:self.surrouding_length],
						self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
						self.observation_dim, self.highlight_iterations,
						combine=True,
						hybrid_encoder_mode=False,
					)
					# [1,H,W] -> [1,1,1,H,W]
					surrounding = torch.from_numpy(surrounding_np).unsqueeze(0).unsqueeze(1).float().cuda()
				else:
					surrounding_np = CostMap(
						state[:self.surrouding_length],
						self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
						self.observation_dim, self.highlight_iterations,
						combine=False,
						hybrid_encoder_mode=False,
					)
					# [T,H,W] -> [1,1,T,H,W]
					surrounding = torch.from_numpy(surrounding_np).unsqueeze(0).unsqueeze(1).float().cuda()

			# --------------------------------------------------
			# 原有 CNN 路线
			# --------------------------------------------------
			elif self.use_dual_branch_encoder:
				surrounding_combine_np, surrounding_temporal_np = CostMapDual(
					state[:self.surrouding_length],
					self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
					self.observation_dim, self.highlight_iterations,
					batch=False,
					black_obs=True
				)
				surrounding = {
					"combine": torch.from_numpy(surrounding_combine_np).unsqueeze(0).unsqueeze(1).float().cuda(),
					"temporal": torch.from_numpy(surrounding_temporal_np).unsqueeze(0).unsqueeze(1).float().cuda(),
				}

			elif self.use_hybrid_attn_encoder:
				surrounding_np = CostMap(
					state[:self.surrouding_length],
					self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
					self.observation_dim, self.highlight_iterations,
					combine=False,
					hybrid_encoder_mode=False,
				)
				surrounding = torch.from_numpy(surrounding_np).unsqueeze(0).unsqueeze(1).float().cuda()

			else:
				surrounding_np = CostMap(
					state[:self.surrouding_length],
					self.laser_dim, self.true_laser_range, self.img_height, self.img_width,
					self.observation_dim, self.highlight_iterations,
					combine=self.merge_vis,
					hybrid_encoder_mode=False,
				)
				surrounding = torch.from_numpy(surrounding_np).unsqueeze(0).unsqueeze(1).float().cuda()
			
		# ------------------------------------------------------
		# 送入 FeatureExtractor
		# ------------------------------------------------------
		if self.no_goal:
			if self.progress_ratio_mode:
				return self.fe([ego, path, surrounding, progress])
			else:
				return self.fe([ego, path, surrounding])
		else:
			return self.fe([ego, goal, surrounding]) if not self.deviation_mode else self.fe([ego, goal, surrounding, path])
	# def multi_encode(self, state, id=0, if_batch=True):

	# 	if self.cfg.mlp_obs:
	# 		return self.encode(torch.from_numpy(state).float().cuda(), None)
	# 	if if_batch:   
	# 		ego = torch.from_numpy(state[:, (self.surrouding_length):(self.surrouding_length+2)]).float().cuda() # B,2
	# 		if self.no_goal:
	# 			path = torch.from_numpy(state[:, (self.surrouding_length+2):(self.surrouding_length+4)]).float().cuda() # B,2
	# 			if self.progress_ratio_mode:
	# 				progress = torch.from_numpy(state[:, (self.surrouding_length+4):(self.surrouding_length+7)]).float().cuda() # B,2
	# 		else:
	# 			goal = torch.from_numpy(state[:, (self.surrouding_length+2):(self.surrouding_length+4)]).float().cuda() # B,2
	# 			if self.deviation_mode:
	# 				path = torch.from_numpy(state[:, (self.surrouding_length+4):(self.surrouding_length+6)]).float().cuda() # B,2
			
	# 		surrouding = torch.from_numpy(CostMap(state[:, :self.surrouding_length], 
	# 													self.laser_dim, self.true_laser_range, self.img_height, self.img_width, self.observation_dim, self.highlight_iterations,
	# 													combine=self.merge_vis, batch=True)).unsqueeze(1).float().cuda() # B,C,T,H,W
        
	# 	else:
	# 		ego = torch.from_numpy(state[(self.surrouding_length):(self.surrouding_length+2)]).unsqueeze(0).float().cuda() # B,2
	# 		if self.no_goal:
	# 			path = torch.from_numpy(state[(self.surrouding_length+2):(self.surrouding_length+4)]).unsqueeze(0).float().cuda() # B,2
	# 			if self.progress_ratio_mode:
	# 				progress = torch.from_numpy(state[(self.surrouding_length+4):(self.surrouding_length+7)]).unsqueeze(0).float().cuda() # B,2
	# 		else:
	# 			goal = torch.from_numpy(state[(self.surrouding_length+2):(self.surrouding_length+4)]).unsqueeze(0).float().cuda() # B,2
	# 			if self.deviation_mode:
	# 				path = torch.from_numpy(state[(self.surrouding_length+4):(self.surrouding_length+6)]).unsqueeze(0).float().cuda() # B,2
			
	# 		# if id == 0:
	# 		# 	print('encode id', id, state.shape)
	# 		# 	print('net state', state)
	# 		# 	print('net ego goal', ego, goal)
	# 		image = CostMap(state[:self.surrouding_length], 
	# 													self.laser_dim, self.true_laser_range, self.img_height, self.img_width, self.observation_dim, self.highlight_iterations,
	# 													combine=self.merge_vis)
	# 		surrouding = torch.from_numpy(CostMap(state[:self.surrouding_length], 
	# 													self.laser_dim, self.true_laser_range, self.img_height, self.img_width, self.observation_dim, self.highlight_iterations,
	# 													combine=self.merge_vis)).unsqueeze(0).unsqueeze(1).float().cuda() # B,C,T,H,W
        
	# 		# lidar_ego_image = cv2.cvtColor(image[0].astype(np.float32), cv2.COLOR_GRAY2RGB)
	# 		# cv2.putText(lidar_ego_image, str(time.time()), (64,64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
	# 		# cv2.imshow('lidar_ego_net_'+str(id), lidar_ego_image)
	# 		# cv2.waitKey(10)
			
	# 	# print('surrouding: ', surrouding.shape) [64, 1, 1, 128, 256]
	# 	if self.no_goal:
	# 		if self.progress_ratio_mode:
	# 			return self.fe([ego, path, surrouding, progress])
	# 		else:
	# 			return self.fe([ego, path, surrouding])
	# 	else:
	# 		return self.fe([ego, goal, surrouding]) if not self.deviation_mode else self.fe([ego, goal, surrouding, path])

	def temporal_future_aux_loss(self, obs_t_np, action_t, obs_tp1_np):
		"""
		training-only auxiliary:
		用当前 temporal correction feature + 当前动作，预测下一时刻 temporal feature

		Args:
			obs_t_np:   numpy array, [B, obs_dim]
			action_t:   torch tensor, [B, action_dim]
			obs_tp1_np: numpy array, [B, obs_dim]
		"""
		if not self.use_temporal_future_aux:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# 目前这套 auxiliary 只对非 mlp_obs 路线（也就是有 multi_encode / image encoder 的路线）开放
		if self.cfg.mlp_obs:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# 1) 当前时刻 temporal feature
		_ = self.multi_encode(obs_t_np)
		feats_now = self.get_latest_branch_feats()
		feat_now = feats_now.get('temporal', None)

		if feat_now is None:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# 2) 下一时刻 temporal feature（stop-grad target）
		with torch.no_grad():
			_ = self.multi_encode(obs_tp1_np)
			feats_next = self.get_latest_branch_feats()
			feat_next = feats_next.get('temporal', None)

			if feat_next is None:
				device = next(self.parameters()).device
				return torch.zeros((), device=device)

			feat_next = feat_next.detach()

		# 3) predictor
		pred_next = self.temporal_future_predictor(torch.cat([feat_now, action_t], dim=-1))

		if self.temporal_future_aux_normalize:
			pred_next = F.normalize(pred_next, dim=-1)
			feat_next = F.normalize(feat_next, dim=-1)
			loss = 1.0 - (pred_next * feat_next).sum(dim=-1).mean()
		else:
			loss = F.smooth_l1_loss(pred_next, feat_next)

		return loss

	def temporal_interp_aux_loss(self, obs_t_np, obs_tp1_np, obs_tp2_np):
		"""
		training-only auxiliary:
		用当前和后两步的 temporal feature，预测中间时刻 temporal feature
		"""
		if not self.use_temporal_interp_aux:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		if self.cfg.mlp_obs:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# f_t
		_ = self.multi_encode(obs_t_np)
		feats_t = self.get_latest_branch_feats()
		feat_t = feats_t.get('temporal', None)

		if feat_t is None:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# f_{t+1} 作为 target
		with torch.no_grad():
			_ = self.multi_encode(obs_tp1_np)
			feats_mid = self.get_latest_branch_feats()
			feat_mid = feats_mid.get('temporal', None)

			if feat_mid is None:
				device = next(self.parameters()).device
				return torch.zeros((), device=device)

			feat_mid = feat_mid.detach()

		# f_{t+2}
		_ = self.multi_encode(obs_tp2_np)
		feats_t2 = self.get_latest_branch_feats()
		feat_t2 = feats_t2.get('temporal', None)

		if feat_t2 is None:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		pred_mid = self.temporal_interp_predictor(torch.cat([feat_t, feat_t2], dim=-1))

		if self.temporal_interp_aux_normalize:
			pred_mid = F.normalize(pred_mid, dim=-1)
			feat_mid = F.normalize(feat_mid, dim=-1)
			loss = 1.0 - (pred_mid * feat_mid).sum(dim=-1).mean()
		else:
			loss = F.smooth_l1_loss(pred_mid, feat_mid)

		return loss

	def risk_proxy_aux_loss(self, obs_t_np, action_t, obs_tp1_np):
		"""
		training-only auxiliary:
		用当前 temporal feature + 当前动作，预测下一时刻环境风险代理
		"""
		if not self.use_risk_proxy_aux:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		if self.cfg.mlp_obs:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# 当前 temporal feature
		_ = self.multi_encode(obs_t_np)
		feats_now = self.get_latest_branch_feats()
		feat_now = feats_now.get('temporal', None)

		if feat_now is None:
			device = next(self.parameters()).device
			return torch.zeros((), device=device)

		# 下一时刻环境代理 target
		with torch.no_grad():
			_ = self.multi_encode(obs_tp1_np)
			feats_next = self.get_latest_branch_feats()
			env_target = feats_next.get('env_features', None)

			if env_target is None:
				device = next(self.parameters()).device
				return torch.zeros((), device=device)

			env_target = env_target.detach()

		pred_env = self.risk_proxy_predictor(torch.cat([feat_now, action_t], dim=-1))
		pred_env = torch.sigmoid(pred_env)

		loss = F.smooth_l1_loss(pred_env, env_target)
		return loss

	def collision_logit(self, z, a, task):
		"""与 reward() 完全相同的输入约定，输出 1 维 logit。"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._collision(z)

	def collision_prob(self, z, a, task):
		"""p_unsafe(z,a) in (0,1)，供 _estimate_value 在线打分使用。"""
		return torch.sigmoid(self.collision_logit(z, a, task))

	def collision_aux_loss(self, obs_t_np, action_t, obs_tp1_np):
		"""
		training-only：用 z_t + a_t 预测“执行 a_t 后是否进入近碰撞”，BCE。
		标签来自下一步 env_features 的近场占据（front/left/right near 取 max）。
		"""
		if not self.use_psafe_head:
			return torch.zeros((), device=next(self.parameters()).device)
		if self.cfg.mlp_obs:
			return torch.zeros((), device=next(self.parameters()).device)

		# 1) 标签：下一步近场占据（stop-grad）
		with torch.no_grad():
			_ = self.multi_encode(obs_tp1_np)
			env_next = self.get_latest_branch_feats().get('env_features', None)
			if env_next is None:
				return torch.zeros((), device=next(self.parameters()).device)
			# env_features 顺序: [front_near, front_mid, left_near, right_near, global_occ]
			near = torch.stack([env_next[:, 0], env_next[:, 2], env_next[:, 3]], dim=1)  # [B,3]
			near_max = near.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)              # [B,1]
			label = near_max if self.psafe_label_soft else (near_max > self.psafe_label_thr).float()
			label = label.detach()

		# 2) z_t（带梯度，让 BCE 同时塑造 encoder —— 这就是“grounding 让 latent 可问责”）
		z_t = self.multi_encode(obs_t_np)               # [B, latent_dim*enc_num]
		if getattr(self, 'psafe_head_detach', False):
			z_t = z_t.detach()                          # 头只读出，不反塑 encoder
		logit = self.collision_logit(z_t, action_t, None)  # 单任务传 None
		loss = F.binary_cross_entropy_with_logits(logit, label)
		return loss

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		
		obs = obs.to(torch.float32)
		return self._encoder[self.cfg.obs](obs)
	
	def encode_state(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder_state[self.cfg.obs](o) for o in obs])
		
		obs = obs.to(torch.float32)
		return self._encoder_state[self.cfg.obs](obs)
	
	def encode_goal(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder_goal[self.cfg.obs](o) for o in obs])
		
		obs = obs.to(torch.float32)
		return self._encoder_goal[self.cfg.obs](obs)
	
	def decode(self, z):
		return self._decoder(z)
	
	def check_next(self, state, action):
		encoded_state = self._check_enc(state)
		next_state = self._check_dynamics(torch.cat([encoded_state, action], dim=-1))
		return next_state


	# def mdn_dynamic(self, z, a, task=None):
	# 	if self.cfg.multitask:
	# 		z = self.task_emb(z, task)
	# 	x = torch.cat([z, a], dim=-1)
	# 	h = self._dynamics_core(x)
	# 	pi = F.softmax(self.mdn_pi(h), dim=-1)
	# 	mu = self.mdn_mu(h).view(-1, self.cfg.num_mixtures, self.cfg.latent_dim)
	# 	sigma = torch.exp(self.mdn_sigma(h).view(-1, self.cfg.num_mixtures, self.cfg.latent_dim))
	# 	return pi, mu, sigma

	def mdn_dynamic(self, z, a, task=None):
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		x = torch.cat([z, a], dim=-1)
		h = self._dynamics_core(x)

		# 原始 logits 输出
		pi_logits = self.mdn_pi(h)

		# 数值稳定的 softmax
		pi = F.softmax(pi_logits, dim=-1)
		pi = torch.clamp(pi, 1e-6, 1.0)                # 限制最小值为1e-6，防止为0
		pi = pi / pi.sum(dim=-1, keepdim=True)         # 再次归一化（确保和为1）

		# mu/sigma 输出
		mu = self.mdn_mu(h).view(-1, self.cfg.num_mixtures, self.cfg.latent_dim)

		# 保证 sigma 是正值且稳定
		log_sigma = self.mdn_sigma(h).view(-1, self.cfg.num_mixtures, self.cfg.latent_dim)
		log_sigma = log_sigma.clamp(min=-5.0, max=2.0)  # 限制log_sigma范围
		sigma = torch.exp(log_sigma)

		return pi, mu, sigma
	
	def sample_from_mdn(self, pi, mu, sigma):
		idx = torch.multinomial(pi, num_samples=1).squeeze(-1)
		mu_selected = mu[torch.arange(mu.size(0)), idx]
		sigma_selected = sigma[torch.arange(sigma.size(0)), idx]
		eps = torch.randn_like(sigma_selected)
		return mu_selected + eps * sigma_selected

	def next_dist(self, z, a, task, return_type: str = "all"):
		"""
		Return dynamics distribution parameters.
		Args:
			z: (B, Z)
			a: (B, A)
			task: task index or tensor (only for multitask)
			return_type:
				'all'  -> return (mu, logvar) with shape (M, B, Z)
				'mean' -> return ensemble aggregated (mu, logvar) with shape (B, Z)
		"""
		assert return_type in {"all", "mean"}
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		x = torch.cat([z, a], dim=-1)  # (B, Z+A+task)
		out = self._dynamics(x)        # (M, B, 2Z)

		Z = self.cfg.latent_dim
		mu = out[..., :Z]
		logvar = out[..., Z:]
		logvar = logvar.clamp(self.cfg.dyn_logvar_min, self.cfg.dyn_logvar_max)

		if return_type == "all":
			return mu, logvar  # (M,B,Z)
		# Ensemble aggregation: mixture approx -> mean/var of mixture (diagonal)
		mu_mean = mu.mean(dim=0)  # (B,Z)
		# Var_total = E[var] + Var[E]
		var = torch.exp(logvar)
		var_mean = var.mean(dim=0) + (mu - mu_mean.unsqueeze(0)).pow(2).mean(dim=0)#two
		logvar_mean = torch.log(var_mean.clamp_min(1e-8))
		return mu_mean, logvar_mean

	def sample_next(self, z, a, task, mode: str = "ts", member_idx=None):
		"""
		Sample next latent state from probabilistic ensemble dynamics.

		Args:
			z: (B, Z)
			a: (B, A)
			member_idx: Optional[Tensor], shape (B,) with int64 in [0, M-1].
						If provided, use this fixed ensemble member per sample (no resampling).
		"""
		assert mode in {"ts", "ts_mean", "mean", "sample_mean", "ts_mean_fixed"}

		if mode == "mean":
			mu_mean, _ = self.next_dist(z, a, task, return_type="mean")
			return mu_mean

		if mode == "sample_mean":
			mu_mean, logvar_mean = self.next_dist(z, a, task, return_type="mean")
			eps = torch.randn_like(mu_mean)
			return mu_mean + torch.exp(0.5 * logvar_mean) * eps

		if mode == "ts_mean_fixed":
			mu_all, logvar_all = self.next_dist(z, a, task, return_type="all")  # (M,B,Z)
			mu = mu_all[0]  # 永远用 member 0
			return mu

		# --- TS variants ---
		mu_all, logvar_all = self.next_dist(z, a, task, return_type="all")  # (M,B,Z)
		M, B, Z = mu_all.shape

		if member_idx is None:
			# backward-compatible: resample member per call & per sample
			member_idx = torch.randint(0, M, (B,), device=mu_all.device)

		arange = torch.arange(B, device=mu_all.device)
		mu = mu_all[member_idx, arange]       # (B,Z)
		logvar = logvar_all[member_idx, arange]

		if mode == "ts_mean":
			return mu

		eps = torch.randn_like(mu)
		beta = getattr(self.cfg, "dyn_noise_scale", 0.0)  # 先 0.0
		return mu + beta * torch.exp(0.5 * logvar) * eps
		

	@torch.no_grad()
	def dynamics_uncertainty(self, z, a, task,
								mode: str = "epistemic",
								reduce: str = "mean",
								clip_max: float | None = None):
		"""
		Returns uncertainty scalar per sample: (B,1)
		mode:
			- "epistemic": Var_mu across ensemble members
			- "aleatoric": E[var] from predicted logvar
			- "total": epistemic + aleatoric
		reduce: "mean" or "sum" over latent dims
		"""
		mu_all, logvar_all = self.next_dist(z, a, task, return_type="all")  # (M,B,Z)
		# epistemic: Var over members of mu
		mu_mean = mu_all.mean(dim=0)  # (B,Z)
		var_epi = (mu_all - mu_mean.unsqueeze(0)).pow(2).mean(dim=0)  # (B,Z)

		# aleatoric: mean predicted variance over members
		var_ale = torch.exp(logvar_all).mean(dim=0)  # (B,Z)

		if mode == "epistemic":
			var = var_epi
		elif mode == "aleatoric":
			var = var_ale
		elif mode == "total":
			var = var_epi + var_ale
		else:
			raise ValueError(f"unknown unc mode: {mode}")

		if reduce == "mean":
			u = var.mean(dim=-1, keepdim=True)  # (B,1)
		elif reduce == "sum":
			u = var.sum(dim=-1, keepdim=True)   # (B,1)
		else:
			raise ValueError(f"unknown reduce: {reduce}")

		if clip_max is not None:
			u = u.clamp_max(clip_max)

		return u

	@torch.no_grad()
	def dynamics_uncertainty_all(self, z, a, task, reduce="mean"):
		"""Optional: return (u_total,u_epi,u_ale) each (B,1) for logging."""
		mu_all, logvar_all = self.next_dist(z, a, task, return_type="all")  # (M,B,Z)
		mu_mean = mu_all.mean(dim=0)
		var_epi = (mu_all - mu_mean.unsqueeze(0)).pow(2).mean(dim=0)
		var_ale = torch.exp(logvar_all).mean(dim=0)
		var_total = var_epi + var_ale

		def red(x):
			return x.mean(dim=-1, keepdim=True) if reduce == "mean" else x.sum(dim=-1, keepdim=True)

		return red(var_total), red(var_epi), red(var_ale)


	# def sample_next(self, z, a, task, mode: str = "ts"):
	# 	"""
	# 	Sample next latent state from probabilistic ensemble dynamics.
	# 	Args:
	# 		mode:
	# 			'ts'   : PETS-style trajectory sampling (per-step random member)
	# 			'mean' : deterministic rollout using ensemble aggregated mean
	# 			'sample_mean' : sample from aggregated Gaussian (mu_mean, logvar_mean)
	# 			'ts_mean': only member sampling, no Gaussion sampling
	# 	Returns:
	# 		z_next: (B, Z)
	# 	"""
	# 	assert mode in {"ts", "mean", "sample_mean", "ts_mean"}

	# 	if mode == "mean":
	# 		mu_mean, _ = self.next_dist(z, a, task, return_type="mean")
	# 		return mu_mean

	# 	if mode == "sample_mean":
	# 		mu_mean, logvar_mean = self.next_dist(z, a, task, return_type="mean")
	# 		eps = torch.randn_like(mu_mean)
	# 		return mu_mean + torch.exp(0.5 * logvar_mean) * eps

	# 	# --- TS variants ---
	# 	mu_all, logvar_all = self.next_dist(z, a, task, return_type="all")  # (M,B,Z)
	# 	M, B, Z = mu_all.shape
	# 	idx = torch.randint(0, M, (B,), device=mu_all.device)
	# 	arange = torch.arange(B, device=mu_all.device)
	# 	mu = mu_all[idx, arange]         # (B,Z)
	# 	logvar = logvar_all[idx, arange] # (B,Z)

	# 	if mode == "ts_mean":
	# 		# 只做成员采样，不加高斯噪声
	# 		return mu
	# 	# mode == "ts": 成员采样 + 高斯噪声 (高风险)
	# 	eps = torch.randn_like(mu)
	# 	beta = getattr(self.cfg, "dyn_noise_scale", 0.0)  # 先 0.0
	# 	return mu + beta * torch.exp(0.5 * logvar) * eps



	def next(self, z, a, task=None, mode='sample'):
		if self.cfg.use_mdn:
			pi, mu, sigma = self.mdn_dynamic(z, a, task)
			if mode == 'sample':
				return self.sample_from_mdn(pi, mu, sigma)
			elif mode == 'mean':
				return torch.sum(pi.unsqueeze(-1) * mu, dim=1)
			elif mode == 'most_likely':
				max_idx = pi.argmax(dim=1)
				return mu[torch.arange(mu.size(0)), max_idx]
		elif self.cfg.use_ensemble_dyn:
			return self.sample_next(z, a, task, mode=getattr(self.cfg, "dyn_rollout_mode", "mean"))
		else:
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			z = torch.cat([z, a], dim=-1)
			return self._dynamics(z)
		
	# def next(self, z, a, task):
	# 	"""
	# 	Predicts the next latent state given the current latent state and action.
	# 	"""
	# 	if self.cfg.multitask:
	# 		z = self.task_emb(z, task)
		
	# 	z = torch.cat([z, a], dim=-1)
	# 	return self._dynamics(z)
	
	def next_goal(self, z):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		
		return self._dynamics_goal(z)
	
	def next_obs_sep(self, z, a, task):
		z=torch.cat([z, a], dim=-1)
		return self._dynamics_obs_sep(z)
	
	def lstm_next(self, z, a, h=None):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		
		z = torch.cat([z, a], dim=-1)
		if self.cfg.lstm_seq_len <= 1:
			z = z.unsqueeze(1)
		return self._dynamics(z, h)
	
	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def action_encode(self, a, mode):
		if self.cfg.use_action_embedding and self.cfg.diff_action_embedding:	
			if mode == 'd':
				return F.leaky_relu(self.action_embedding(a))
			elif mode == 'q':
				return F.leaky_relu(self.action_embedding_q(a))
			elif mode == 'r':
				return F.leaky_relu(self.action_embedding_r(a))
		elif self.cfg.use_action_embedding and not self.cfg.diff_action_embedding:
			return F.leaky_relu(self.action_embedding(a))
		else:
			return a
		# return F.leaky_relu(self.action_embedding(a)) if self.cfg.use_action_embedding else a
	
	
	# def path_predict(self, z, a):
	# 	z = torch.cat([z, a], dim=-1)
	# 	return self._path_predict(z)

	def sac_pi(self, z, task):
		"""
		改进版 pi，借鉴 SAC Actor 结构，优化策略稳定性
		"""
		# 计算均值 mu 和 log 标准差 log_std
		mu, log_std = self._pi(z).chunk(2, dim=-1)

		# 采用 Tanh 限制 log_std，并进行线性缩放
		log_std = torch.tanh(log_std)  # 确保 log_std 变化稳定
		log_std = 0.5 * ((self.log_std_dif) * log_std + (self.log_std_min + self.log_std_dif))  # 线性缩放到 [log_std_min, log_std_max]

		# 生成高斯分布
		std = log_std.exp()  # 计算标准差
		cov = torch.diag_embed(std)  # 生成对角协方差矩阵
		dist = MultivariateNormal(mu, cov)  # 创建多元高斯分布

		# 进行重参数化采样
		eps = dist.rsample()  # 采样带梯度信息
		pi = torch.tanh(eps)  # 使用 Tanh 归一化到 [-1, 1]

		# 计算 log_pi，修正 tanh 变换导致的概率密度变化
		log_pi = dist.log_prob(eps).unsqueeze(1) - torch.log(1 - pi.pow(2) + 1e-6).sum(dim=1, keepdim=True)

		return mu, pi, log_pi, log_std

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		# print('mu, log_std', mu.shape, log_std.shape)

		if self.cfg.sac_pi:
			# 采用 Tanh 限制 log_std，并进行线性缩放
			log_std = torch.tanh(log_std)  # 确保 log_std 变化稳定
			log_std = 0.5 * ((self.log_std_dif) * log_std + (self.log_std_min + self.log_std_dif))  # 线性缩放到 [log_std_min, log_std_max]

			# 生成高斯分布
			std = log_std.exp()  # 计算标准差
			cov = torch.diag_embed(std)  # 生成对角协方差矩阵
			dist = MultivariateNormal(mu, cov)  # 创建多元高斯分布

			# 进行重参数化采样
			eps = dist.rsample()

			# **根据 batch 维度判断是训练还是推理模式**
			is_inference = (eps.ndim == 1)  # 1D tensor → 推理模式

			if is_inference:
				eps = eps.unsqueeze(0)  # 变成 [1, action_dim]
			# print("eps", eps.shape)
			# 计算最终动作
			pi = torch.tanh(eps)  # 使用 Tanh 归一化到 [-1, 1]
			# print("pi", pi.shape)
			# **修正 pi 形状，使其与原版匹配**
			if is_inference:
				pi = pi.squeeze(0)  # 变成 [action_dim]

			# 计算 log_pi
			log_pi = dist.log_prob(eps)
			if not is_inference:
				log_pi = log_pi.unsqueeze(-1)
			# print("log_pi pi", log_pi.shape, pi.shape)
			# **修正 log_pi，补偿 tanh 变换的影响**
			log_pi -= torch.log(1 - pi.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
			# llll
		
			# **确保 log_pi 形状符合原版 pi**
			# if log_pi.shape == (1, self.cfg.action_dim):  # 只在推理模式下调整
			# 	log_pi = log_pi.squeeze(-1)  # 变成 [1]
		else:
			log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
			eps = torch.randn_like(mu)

			if self.cfg.multitask: # Mask out unused action dimensions
				mu = mu * self._action_masks[task]
				log_std = log_std * self._action_masks[task]
				eps = eps * self._action_masks[task]
				action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
			else: # No masking
				action_dims = None

			log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
			pi = mu + eps * log_std.exp()
			mu, pi, log_pi = math.squash(mu, pi, log_pi)
		# print('mu', mu)
		# print('pi', pi)
		# print('mu pi logpi logstd', mu.shape, pi.shape, log_pi.shape, log_std.shape)
		
		return mu, pi, log_pi, log_std

	def Q(self, z, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all', 'sep'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)
			
		z = torch.cat([z, a], dim=-1)
		out = (self._target_Qs if target else self._Qs)(z)

		if return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		# print("	q1q2", Q1.shape, Q2.shape)
		if self.cfg.use_hot_q:
			Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		if return_type == 'sep':
			return Q1, Q2
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
	
	def mdn_loss(self, pi, mu, sigma, target):
		m = torch.distributions.Normal(mu, sigma)
		log_probs = m.log_prob(target.unsqueeze(1))  # [B, K, D]
		log_probs = log_probs.sum(dim=2)             # [B, K]
		log_probs = log_probs + torch.log(pi + 1e-8) # [B, K]
		log_sum_exp = torch.logsumexp(log_probs, dim=1)  # [B]
		nll = -torch.mean(log_sum_exp)
		return nll
