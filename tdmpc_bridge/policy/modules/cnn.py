import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelfAttentionBlock(nn.Module):
    """
    Temporal self-attention block.
    对每一个空间位置 (h, w)，在时间维 T 上做 self-attention。
    输入:  [B, T, C, H, W]
    输出:  [B, T, C, H, W]

    新增：
    - 缓存最近一次 attention weights
    - 计算可用于 debug 的统计量
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # 新增：缓存最近一次 attention 信息
        self.latest_attn_weights = None
        self.latest_attn_stats = {}

    def get_latest_attn_stats(self):
        return dict(self.latest_attn_stats)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        # 每个空间位置单独在 T 维做 attention
        seq = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)  # [B*H*W, T, C]

        attn_in = self.norm1(seq)

        # 注意：打开 average_attn_weights=False，拿到每个 head 的注意力
        attn_out, attn_weights = self.attn(
            attn_in, attn_in, attn_in,
            need_weights=True,
            average_attn_weights=False
        )
        # attn_weights: [B*H*W, num_heads, T, T]
        self.latest_attn_weights = attn_weights.detach()

        # 统计一些简单但很有用的指标
        with torch.no_grad():
            w_mean = attn_weights.mean(dim=(0, 1))  # [T, T]

            # 1) 对角均值：越高说明越偏向“自己看自己”
            diag_mean = torch.diagonal(w_mean, 0).mean()

            # 2) 最后一帧 query 对最后一帧 key 的关注
            last_to_last = w_mean[-1, -1]

            # 3) 最后一帧 query 对第一帧 key 的关注
            last_to_first = w_mean[-1, 0]

            # 4) 行熵（归一化），越接近 1 越均匀，越接近 0 越有选择性
            p = w_mean.clamp_min(1e-8)
            
            if t > 1:
                entropy = -(p * p.log()).sum(dim=-1).mean() / math.log(t)
            else:
                entropy = torch.zeros((), device=p.device)

            self.latest_attn_stats = {
                'attn_diag_mean': float(diag_mean.cpu().item()),
                'attn_last_to_last': float(last_to_last.cpu().item()),
                'attn_last_to_first': float(last_to_first.cpu().item()),
                'attn_entropy': float(entropy.cpu().item()),
            }

        seq = seq + attn_out
        seq = seq + self.ffn(self.norm2(seq))

        out = seq.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
        return out
# class TemporalSelfAttentionBlock(nn.Module):
#     """
#     Temporal self-attention block.
#     对每一个空间位置 (h, w)，在时间维 T 上做 self-attention。
#     输入:  [B, T, C, H, W]
#     输出:  [B, T, C, H, W]
#     """
#     def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         hidden_dim = int(dim * mlp_ratio)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )

#     def forward(self, x):
#         # x: [B, T, C, H, W]
#         b, t, c, h, w = x.shape

#         # 每个空间位置单独在 T 维做 attention
#         seq = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)  # [B*H*W, T, C]

#         attn_in = self.norm1(seq)
#         attn_out, _ = self.attn(attn_in, attn_in, attn_in)

#         seq = seq + attn_out
#         seq = seq + self.ffn(self.norm2(seq))

#         out = seq.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
#         return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block.
    Query 来自地图 token，Key / Value 来自条件 token（vel / goal）。
    输入:
        query_tokens: [B, N, C]
        cond_tokens : [B, M, C]
    输出:
        [B, N, C]
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, query_tokens, cond_tokens):
        if cond_tokens is None:
            return query_tokens

        q = self.norm_q(query_tokens)
        kv = self.norm_kv(cond_tokens)

        attn_out, _ = self.attn(q, kv, kv)
        x = query_tokens + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class HybridTemporalConditionEncoder(nn.Module):
    """
    CNN stem + temporal self-attention + optional vel/goal cross-attention

    新增功能：
    1. 显式 temporal position embedding
    2. 可通过参数关闭 cross-attention，用来做 ablation
    """
    def __init__(self, args):
        super().__init__()
        self.img_width = args.img_width
        self.img_height = args.img_height

        self.map_dim = int(getattr(args, 'hybrid_map_dim', max(getattr(args, 'img_channel', 8) * 2, 16)))
        self.num_heads = int(getattr(args, 'hybrid_num_heads', 4))
        self.num_temporal_layers = int(getattr(args, 'hybrid_temporal_layers', 1))
        self.num_cross_layers = int(getattr(args, 'hybrid_cross_layers', 1))
        self.cond_dim = int(getattr(args, 'hybrid_cond_dim', self.map_dim))

        # -------- 新增：显式时间位置编码 --------
        self.use_temporal_pos_emb = bool(getattr(args, 'hybrid_temporal_pos_emb', True))
        self.temporal_pos_type = getattr(args, 'hybrid_temporal_pos_type', 'sin')
        self.max_T = int(getattr(args, 'sample_length', 8))

        # -------- 新增：是否关闭 cross-attention --------
        self.disable_cross_attn = bool(getattr(args, 'hybrid_disable_cross_attn', False))

        # -------- 新增：用于排查 temporal attention 是否本身有问题 --------
        self.disable_temporal_attn = bool(getattr(args, 'hybrid_disable_temporal_attn', False))
        self.temporal_pool = getattr(args, 'hybrid_temporal_pool', 'mean')   # 'mean' or 'last'
        self.log_attn_stats = bool(getattr(args, 'hybrid_log_attn_stats', True))

        # -------- 新增：实验 E / F，用于把 temporal 分支强制裁成 T=1 --------
        # 注意：这是“时间长度变成 1”，不是“temporal_layers=1”
        self.force_temporal_t1 = bool(getattr(args, 'hybrid_force_temporal_t1', False))
        self.temporal_t1_source = getattr(args, 'hybrid_temporal_t1_source', 'last')  # 'last' / 'first' / 'middle'
        self.log_temporal_t1 = bool(getattr(args, 'hybrid_log_temporal_t1', True))

        # -------- 新增：实验 H，用于切换 temporal branch backbone --------
        # 2d_stem_attn : 当前默认实现（2D stem + temporal self-attn）
        # 3d_stem_only : 实验 H1（3D CNN stem，不接 temporal attn）
        # 3d_stem_attn : 实验 H2（3D CNN stem + temporal self-attn）
        self.temporal_encoder_mode = getattr(args, 'hybrid_temporal_encoder_mode', '2d_stem_attn')
        assert self.temporal_encoder_mode in [
            '2d_stem_attn',
            '3d_stem_only',
            '3d_stem_attn',
            '3d_stem_flatten_mlp',   # H3
            'orig_3dcnn',            # H0（注意：这个模式不会真正走到 HybridTemporalConditionEncoder.forward）
        ]
        
        # 3D stem 在时间维上的 stride，默认 1，表示保留完整时间长度
        self.temporal_stride_3d = int(getattr(args, 'hybrid_3d_temporal_stride', 1))

        # 缓存最近一次 forward 的 T=1 调试信息
        self._latest_temporal_t1_stats = {}

        # 缓存最近一次 temporal attention 的统计量
        self._latest_temporal_attn_stats = {}

        # 1) CNN stem：逐帧提局部几何
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.map_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.map_dim // 2, self.map_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.map_dim, self.map_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        # 新增：3D CNN stem
        # 目的：先在局部时空立方体 (t, x, y) 上提取 motion / occupancy pattern
        # 与当前“逐帧 2D stem + 同位置 temporal attention”形成对照
        self.stem3d = nn.Sequential(
            nn.Conv3d(1, self.map_dim // 2, kernel_size=(3, 3, 3),
                      stride=(self.temporal_stride_3d, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(self.map_dim // 2, self.map_dim, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(self.map_dim, self.map_dim, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(),
        )

        # --------------------------------------------------
        # H3: 3D stem + flatten MLP
        # 这里不是 aggregate_time + token mean，而是尽量模仿原始 3D CNN 的“保留整个时空体到最后”思路
        # --------------------------------------------------
        temporal_t_dim_cfg = 1 if self.force_temporal_t1 else self.max_T
        stem3d_out_t = ((temporal_t_dim_cfg - 1) // self.temporal_stride_3d) + 1
        stem3d_flat_dim = self.map_dim * stem3d_out_t * (self.img_height // 8) * (self.img_width // 8)

        self.stem3d_flatten_mlp = nn.Sequential(
            nn.Linear(stem3d_flat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
        )

        # 2) temporal self-attention
        self.temporal_blocks = nn.ModuleList([
            TemporalSelfAttentionBlock(self.map_dim, num_heads=self.num_heads)
            for _ in range(self.num_temporal_layers)
        ])

        # learnable temporal position embedding（可选）
        if self.use_temporal_pos_emb and self.temporal_pos_type == 'learnable':
            self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_T, self.map_dim, 1, 1))
            nn.init.normal_(self.temporal_pos, mean=0.0, std=0.02)
        else:
            self.temporal_pos = None

        # 3) condition token：vel / goal
        self.vel_embed = nn.Sequential(
            nn.Linear(2, self.cond_dim),
            nn.LeakyReLU(),
            nn.Linear(self.cond_dim, self.map_dim),
        )
        self.goal_embed = nn.Sequential(
            nn.Linear(2, self.cond_dim),
            nn.LeakyReLU(),
            nn.Linear(self.cond_dim, self.map_dim),
        )

        # 4) cross-attention（可关闭）
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(self.map_dim, num_heads=self.num_heads)
            for _ in range(self.num_cross_layers)
        ])

        # 5) 输出 MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(self.map_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
        )

    def get_latest_temporal_attn_stats(self):
        return dict(self._latest_temporal_attn_stats)

    def get_latest_temporal_t1_stats(self):
        """
        供外部读取当前 temporal branch 实际用了几帧、取的是哪一帧。
        """
        return dict(self._latest_temporal_t1_stats)
    
    def forward_with_mode(self, image, ego=None, goal=None, mode_override=None):
        """
        允许外部临时覆盖 temporal_encoder_mode，用于 dual-branch 下切换 correction branch。
        """
        if mode_override is None:
            return self.forward(image, ego=ego, goal=goal)

        old_mode = self.temporal_encoder_mode
        self.temporal_encoder_mode = mode_override
        try:
            out = self.forward(image, ego=ego, goal=goal)
        finally:
            self.temporal_encoder_mode = old_mode
        return out

    def _select_temporal_t1(self, image):
        """
        image: [B, T, H, W]
        return:
            image_t1: [B, 1, H, W]
            used_idx: int
        """
        b, t, h, w = image.shape
        if t == 1:
            return image, 0

        source = self.temporal_t1_source
        if source == 'last':
            idx = t - 1
        elif source == 'first':
            idx = 0
        elif source == 'middle':
            idx = t // 2
        else:
            raise NotImplementedError(f"Unknown hybrid_temporal_t1_source: {source}")

        return image[:, idx:idx+1, :, :], idx

    def _aggregate_time(self, feat):
        """
        feat: [B, T, C, H, W]
        return:
            feat_now: [B, C, H, W]
        """
        if self.temporal_pool == 'mean':
            return feat.mean(dim=1)
        elif self.temporal_pool == 'last':
            return feat[:, -1]
        else:
            raise NotImplementedError(f"Unknown hybrid_temporal_pool: {self.temporal_pool}")

    def _build_condition_tokens(self, ego, goal):
        vel_token = self.vel_embed(ego).unsqueeze(1)  # [B,1,C]
        if goal is None:
            goal = torch.zeros_like(ego)
        goal_token = self.goal_embed(goal).unsqueeze(1)  # [B,1,C]
        return torch.cat([vel_token, goal_token], dim=1)  # [B,2,C]

    def _build_sinusoidal_temporal_pos(self, t, c, device):
        """
        生成 [1, T, C, 1, 1] 的 sin-cos temporal position embedding
        """
        pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div = torch.exp(torch.arange(0, c, 2, device=device, dtype=torch.float32) *
                        (-math.log(10000.0) / c))
        pe = torch.zeros(t, c, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.view(1, t, c, 1, 1)

    def forward(self, image, ego=None, goal=None):
        """
        支持：
        - E/F: force_temporal_t1
        - H1 : 3d_stem_only
        - H2 : 3d_stem_attn

        image:
            [B,1,T,H,W] 或 [B,T,H,W]
        """
        # --------------------------------------------------
        # 0) 统一输入形状到 [B,T,H,W]
        # --------------------------------------------------
        if image.dim() == 5:
            if image.size(1) == 1:
                image = image.squeeze(1)  # -> [B,T,H,W]
            else:
                raise ValueError('HybridTemporalConditionEncoder expects image channel = 1.')
        elif image.dim() != 4:
            raise ValueError(f'Unexpected image shape: {image.shape}')

        b, t, h, w = image.shape

        # --------------------------------------------------
        # 1) 实验 E / F：如果 force_temporal_t1=True，则把时间维强制裁成 1
        # --------------------------------------------------
        used_idx = -1
        if self.force_temporal_t1:
            image, used_idx = self._select_temporal_t1(image)   # [B,1,H,W]
            b, t, h, w = image.shape

        if self.log_temporal_t1:
            self._latest_temporal_t1_stats = {
                'temporal_frames_used': float(t),
                'temporal_t1_enabled': 1.0 if self.force_temporal_t1 else 0.0,
                'temporal_t1_index': float(used_idx if used_idx >= 0 else (t - 1)),
            }

        # attention debug 默认清空
        self._latest_temporal_attn_stats = {}

        # H0 不应该走到这里，而应该在 Image_Encoder 里直接走原始 group + temporal_linear
        if self.temporal_encoder_mode == 'orig_3dcnn':
            raise RuntimeError(
                "hybrid_temporal_encoder_mode='orig_3dcnn' should be routed by Image_Encoder._forward_original_temporal_branch(), "
                "not HybridTemporalConditionEncoder.forward()."
            )

        # ==================================================
        # 模式 A：2D stem + temporal attention（当前默认）
        # ==================================================
        if self.temporal_encoder_mode == '2d_stem_attn':
            # 1) 逐帧 2D stem
            x = image.reshape(b * t, 1, h, w)
            feat = self.stem(x)  # [B*T, C, H', W']
            _, c, h2, w2 = feat.shape
            feat = feat.view(b, t, c, h2, w2)  # [B,T,C,H',W']

            # 2) temporal position embedding
            if self.use_temporal_pos_emb:
                if self.temporal_pos_type == 'sin':
                    pos = self._build_sinusoidal_temporal_pos(t, c, feat.device)   # [1,T,C,1,1]
                elif self.temporal_pos_type == 'learnable':
                    pos = self.temporal_pos[:, :t]                                 # [1,T,C,1,1]
                else:
                    pos = None

                if pos is not None:
                    feat = feat + pos

            # 3) temporal self-attention
            if not self.disable_temporal_attn:
                block_stats = []
                for blk in self.temporal_blocks:
                    feat = blk(feat)
                    if self.log_attn_stats and hasattr(blk, 'get_latest_attn_stats'):
                        block_stats.append(blk.get_latest_attn_stats())

                if self.log_attn_stats and len(block_stats) > 0:
                    keys = block_stats[0].keys()
                    self._latest_temporal_attn_stats = {
                        k: float(sum(s[k] for s in block_stats) / len(block_stats))
                        for k in keys
                    }
                    self._latest_temporal_attn_stats['temporal_attn_disabled'] = 0.0
            else:
                self._latest_temporal_attn_stats = {
                    'attn_diag_mean': 0.0,
                    'attn_last_to_last': 0.0,
                    'attn_last_to_first': 0.0,
                    'attn_entropy': 0.0,
                    'temporal_attn_disabled': 1.0,
                }

            # 4) 时间聚合
            feat_now = self._aggregate_time(feat)   # [B,C,H',W']

        # ==================================================
        # 模式 B：H1 -> 3D stem only
        # ==================================================
        elif self.temporal_encoder_mode == '3d_stem_only':
            # image: [B,T,H,W] -> [B,1,T,H,W]
            x3d = image.unsqueeze(1)
            feat3d = self.stem3d(x3d)               # [B,C,T',H',W']
            _, c, t2, h2, w2 = feat3d.shape

            # 统一转成 [B,T',C,H',W']，复用后面的聚合逻辑
            feat = feat3d.permute(0, 2, 1, 3, 4).contiguous()

            # 这里没有 temporal attention，但为了日志格式统一，填 0
            self._latest_temporal_attn_stats = {
                'attn_diag_mean': 0.0,
                'attn_last_to_last': 0.0,
                'attn_last_to_first': 0.0,
                'attn_entropy': 0.0,
                'temporal_attn_disabled': 1.0,
            }

            feat_now = self._aggregate_time(feat)   # [B,C,H',W']

        # ==================================================
        # 模式 C：H2 -> 3D stem + temporal attention
        # ==================================================
        elif self.temporal_encoder_mode == '3d_stem_attn':
            x3d = image.unsqueeze(1)                # [B,1,T,H,W]
            feat3d = self.stem3d(x3d)               # [B,C,T',H',W']
            _, c, t2, h2, w2 = feat3d.shape

            feat = feat3d.permute(0, 2, 1, 3, 4).contiguous()   # [B,T',C,H',W']

            # 3D stem 输出之后再加 temporal pos
            if self.use_temporal_pos_emb:
                if self.temporal_pos_type == 'sin':
                    pos = self._build_sinusoidal_temporal_pos(t2, c, feat.device)
                elif self.temporal_pos_type == 'learnable':
                    pos = self.temporal_pos[:, :t2]
                else:
                    pos = None

                if pos is not None:
                    feat = feat + pos

            # temporal attention
            if not self.disable_temporal_attn:
                block_stats = []
                for blk in self.temporal_blocks:
                    feat = blk(feat)
                    if self.log_attn_stats and hasattr(blk, 'get_latest_attn_stats'):
                        block_stats.append(blk.get_latest_attn_stats())

                if self.log_attn_stats and len(block_stats) > 0:
                    keys = block_stats[0].keys()
                    self._latest_temporal_attn_stats = {
                        k: float(sum(s[k] for s in block_stats) / len(block_stats))
                        for k in keys
                    }
                    self._latest_temporal_attn_stats['temporal_attn_disabled'] = 0.0
            else:
                self._latest_temporal_attn_stats = {
                    'attn_diag_mean': 0.0,
                    'attn_last_to_last': 0.0,
                    'attn_last_to_first': 0.0,
                    'attn_entropy': 0.0,
                    'temporal_attn_disabled': 1.0,
                }

            feat_now = self._aggregate_time(feat)   # [B,C,H',W']

        # ==================================================
        # 模式 D：H3 -> 3D stem + flatten MLP
        # ==================================================
        elif self.temporal_encoder_mode == '3d_stem_flatten_mlp':
            x3d = image.unsqueeze(1).contiguous()            # [B,1,T,H,W]
            feat3d = self.stem3d(x3d)                        # [B,C,T',H',W']

            # H3 不走 temporal attention，不走 aggregate_time，不走 token mean
            # 直接保留整个时空体 flatten 后做 MLP
            self._latest_temporal_attn_stats = {
                'attn_diag_mean': 0.0,
                'attn_last_to_last': 0.0,
                'attn_last_to_first': 0.0,
                'attn_entropy': 0.0,
                'temporal_attn_disabled': 1.0,
            }

            feat_flat = feat3d.reshape(feat3d.size(0), -1)  # [B, C*T'*H'*W']
            return self.stem3d_flatten_mlp(feat_flat)       # [B,256]
        else:
            raise NotImplementedError(f"Unknown hybrid_temporal_encoder_mode: {self.temporal_encoder_mode}")

        # --------------------------------------------------
        # 2) flatten 成地图 tokens
        # --------------------------------------------------
        map_tokens = feat_now.flatten(2).transpose(1, 2)  # [B, H'*W', C]

        # --------------------------------------------------
        # 3) 可选的 cross-attention
        # --------------------------------------------------
        if (not self.disable_cross_attn) and (ego is not None):
            cond_tokens = self._build_condition_tokens(ego, goal)
            for blk in self.cross_blocks:
                map_tokens = blk(map_tokens, cond_tokens)

        # --------------------------------------------------
        # 4) 全局池化 -> 输出 latent
        # --------------------------------------------------
        pooled = map_tokens.mean(dim=1)   # [B,C]
        return self.out_mlp(pooled)       # [B,256]

class PatchEmbed2D(nn.Module):
    """
    纯 ViT 风格 patch embedding，不用 CNN stem。
    使用 nn.Unfold + Linear，把单通道 occupancy map 切成 patch tokens。
    """
    def __init__(self, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(patch_size * patch_size, embed_dim)

    def forward(self, x):
        """
        x: [B, 1, H, W]
        return:
            tokens: [B, N, D]
            gh, gw: patch grid size
        """
        b, c, h, w = x.shape
        assert c == 1, "PatchEmbed2D expects single-channel input."

        # 自动 pad 到 patch_size 的倍数，避免尺寸不能整除
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        h_pad = h + pad_h
        w_pad = w + pad_w
        gh = h_pad // self.patch_size
        gw = w_pad // self.patch_size

        patches = self.unfold(x).transpose(1, 2)   # [B, N, patch_size*patch_size]
        tokens = self.proj(patches)                # [B, N, D]
        return tokens, gh, gw


class ViTBlock(nn.Module):
    """
    标准 Transformer encoder block（batch_first）。
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )

    def forward(self, x):
        return self.block(x)


class VisionTransformerTemporalEncoder(nn.Module):
    """
    ViT 图像编码分支：
    - 不走 CNN stem
    - 对每一帧 occupancy map patchify
    - 加 spatial + temporal 位置编码
    - vel / goal 作为 condition tokens 拼进 token sequence
    - 经过 Transformer encoder blocks
    - 最后输出 256 维 image latent

    支持两种输入：
        1) temporal: [B,1,T,H,W] / [B,T,H,W]
        2) combine : [B,1,1,H,W] / [B,1,H,W]
    """
    def __init__(self, args):
        super().__init__()
        self.img_width = args.img_width
        self.img_height = args.img_height

        self.input_mode = getattr(args, 'vit_input_mode', 'temporal')
        self.patch_size = int(getattr(args, 'vit_patch_size', 8))
        self.embed_dim = int(getattr(args, 'vit_embed_dim', 128))
        self.depth = int(getattr(args, 'vit_depth', 4))
        self.num_heads = int(getattr(args, 'vit_num_heads', 4))
        self.mlp_ratio = float(getattr(args, 'vit_mlp_ratio', 4.0))
        self.dropout = float(getattr(args, 'vit_dropout', 0.0))
        self.use_cls_token = bool(getattr(args, 'vit_use_cls_token', True))
        self.pos_emb_type = getattr(args, 'vit_pos_emb_type', 'learnable')
        self.use_condition_tokens = bool(getattr(args, 'vit_use_condition_tokens', True))

        self.max_T = int(getattr(args, 'sample_length', 8))

        # patch embedding
        self.patch_embed = PatchEmbed2D(self.patch_size, self.embed_dim)

        # 预先根据最大图像尺寸计算 patch 数
        self.grid_h = math.ceil(self.img_height / self.patch_size)
        self.grid_w = math.ceil(self.img_width / self.patch_size)
        self.num_patches = self.grid_h * self.grid_w

        # spatial / temporal pos embedding
        if self.pos_emb_type == 'learnable':
            self.spatial_pos = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
            self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_T, self.embed_dim))
            nn.init.normal_(self.spatial_pos, std=0.02)
            nn.init.normal_(self.temporal_pos, std=0.02)
        else:
            self.spatial_pos = None
            self.temporal_pos = None

        # cls token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # vel / goal token
        if self.use_condition_tokens:
            self.vel_embed = nn.Sequential(
                nn.Linear(2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
            )
            self.goal_embed = nn.Sequential(
                nn.Linear(2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
            )

        self.drop = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout)
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.out_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

    def _build_sinusoidal_pos(self, length, dim, device):
        """
        标准 sin-cos 位置编码: [1, length, dim]
        """
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) *
                        (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def _get_spatial_pos(self, n, device):
        if self.pos_emb_type == 'learnable':
            return self.spatial_pos[:, :n]
        return self._build_sinusoidal_pos(n, self.embed_dim, device)

    def _get_temporal_pos(self, t, device):
        if self.pos_emb_type == 'learnable':
            return self.temporal_pos[:, :t]
        return self._build_sinusoidal_pos(t, self.embed_dim, device)

    def _normalize_input_shape(self, image):
        """
        统一 image shape 到 [B, T, H, W]
        """
        if image.dim() == 5:
            # [B,1,T,H,W] or [B,C,T,H,W]
            if image.size(1) != 1:
                raise ValueError(f"ViT encoder expects single channel image, got {image.shape}")
            image = image.squeeze(1)  # -> [B,T,H,W]
        elif image.dim() == 4:
            # [B,T,H,W]
            pass
        elif image.dim() == 3:
            # [B,H,W] -> 当成 T=1
            image = image.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected image shape for ViT encoder: {image.shape}")
        return image

    def forward(self, image, ego=None, goal=None):
        image = self._normalize_input_shape(image)  # [B,T,H,W]
        b, t, h, w = image.shape

        # 如果是 combine 模式，强制只取一帧
        if self.input_mode == 'combine' and t > 1:
            image = image[:, -1:, :, :]
            t = 1

        # patchify each frame
        x = image.reshape(b * t, 1, h, w)
        tokens, gh, gw = self.patch_embed(x)  # [B*T, N, D]
        n = tokens.size(1)
        tokens = tokens.view(b, t, n, self.embed_dim)  # [B,T,N,D]

        # spatial + temporal pos
        spatial_pos = self._get_spatial_pos(n, tokens.device).view(1, 1, n, self.embed_dim)  # [1,1,N,D]
        temporal_pos = self._get_temporal_pos(t, tokens.device).view(1, t, 1, self.embed_dim)  # [1,T,1,D]
        tokens = tokens + spatial_pos + temporal_pos

        # flatten all frame tokens
        map_tokens = tokens.view(b, t * n, self.embed_dim)  # [B, T*N, D]

        seq = []

        # cls token
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(b, -1, -1)  # [B,1,D]
            seq.append(cls_tok)

        # condition tokens
        if self.use_condition_tokens and ego is not None:
            vel_tok = self.vel_embed(ego).unsqueeze(1)  # [B,1,D]
            seq.append(vel_tok)

            if goal is None:
                goal = torch.zeros_like(ego)
            goal_tok = self.goal_embed(goal).unsqueeze(1)  # [B,1,D]
            seq.append(goal_tok)

        seq.append(map_tokens)

        x = torch.cat(seq, dim=1)  # [B, L, D]
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # pooling
        if self.use_cls_token:
            pooled = x[:, 0]  # [B,D]
        else:
            # 如果没有 cls token，则跳过可能的 condition tokens，对 map tokens 做 mean pooling
            offset = 0
            if self.use_condition_tokens and ego is not None:
                offset += 2
            pooled = x[:, offset:].mean(dim=1)

        return self.out_mlp(pooled)  # [B,256]

class Image_Encoder(nn.Module):
    """
    支持 5 种 surrounding 分支模式：

    1) 原始模式:
        use_hybrid_attn_encoder=False, use_dual_branch_encoder=False
        -> 原始 3D CNN

    2) 单分支 hybrid:
        use_hybrid_attn_encoder=True, use_dual_branch_encoder=False
        -> 只走 temporal/hybrid 分支

    3) dual-branch + combine_only:
        用来做 sanity check，验证 dual-branch plumbing 是否破坏 baseline

    4) dual-branch + residual:
        推荐。保守融合，保证 temporal 分支只是 baseline 的增量修正

    5) dual-branch + gated_residual:
        更强一点的版本。让 combine 分支决定 temporal 分支该加多少
    """
    def __init__(self, args):
        super(Image_Encoder, self).__init__()

        # 新增：图像分支 backbone 路线选择
        # 'cnn' -> 走你当前已有的 CNN / hybrid / dual-branch 逻辑
        # 'vit' -> 走 ViT 图像编码逻辑
        self.merge_vis = bool(getattr(args, 'merge_vis', True))
        self.image_encoder_backbone = getattr(args, 'image_encoder_backbone', 'cnn')

        self.use_dual_branch_encoder = bool(getattr(args, 'use_dual_branch_encoder', False))
        self.use_hybrid_attn_encoder = bool(getattr(args, 'use_hybrid_attn_encoder', False)) or self.use_dual_branch_encoder

        # dual-branch 下 temporal correction 分支的来源
        self.dual_temporal_branch_mode = getattr(args, 'dual_temporal_branch_mode', 'hybrid')
        assert self.dual_temporal_branch_mode in ['hybrid', 'orig_3dcnn', 'h3', 'vit']

        self.temporal_encoder_mode = getattr(args, 'hybrid_temporal_encoder_mode', '2d_stem_attn')
        self.force_temporal_t1 = bool(getattr(args, 'hybrid_force_temporal_t1', False))
        self.max_T = int(getattr(args, 'sample_length', 8))

        self.hybrid_fusion_mode = getattr(args, 'hybrid_fusion_mode', 'residual')
        self.hybrid_residual_alpha = float(getattr(args, 'hybrid_residual_alpha', 0.05))
        self.hybrid_learnable_alpha = bool(getattr(args, 'hybrid_learnable_alpha', True))

        self.img_width = args.img_width
        self.img_height = args.img_height
        self.img_channel = args.img_channel

        # --------------------------------------------------
        # gate logging 开关
        # --------------------------------------------------
        self.hybrid_log_gate_stats = bool(getattr(args, 'hybrid_log_gate_stats', True))

        # --------------------------------------------------
        # 显式环境复杂度引导 gate
        # --------------------------------------------------
        

        self.use_env_complexity_gate = bool(getattr(args, 'hybrid_use_env_complexity_gate', False))
        self.env_gate_scale = float(getattr(args, 'hybrid_env_gate_scale', 1.0))
        self.env_front_ratio = float(getattr(args, 'hybrid_env_front_ratio', 0.5))
        self.env_near_ratio = float(getattr(args, 'hybrid_env_near_ratio', 0.35))

        # scalar / multi_roi
        self.env_feature_mode = getattr(args, 'hybrid_env_feature_mode', 'scalar')
        assert self.env_feature_mode in ['scalar', 'multi_roi']

        self.env_gate_hidden_dim = int(getattr(args, 'hybrid_env_gate_hidden_dim', 64))

        if self.use_env_complexity_gate:
            if self.env_feature_mode == 'scalar':
                # 兼容你当前已有实现
                self.env_gate_proj = nn.Linear(1, 256)
                nn.init.zeros_(self.env_gate_proj.weight)
                nn.init.zeros_(self.env_gate_proj.bias)
            else:
                # multi_roi:
                # [front_near, front_mid, left_near, right_near, global_occ] -> 256
                self.env_gate_proj = nn.Sequential(
                    nn.Linear(5, self.env_gate_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.env_gate_hidden_dim, 256)
                )
                # 初始化成接近 0 bias，避免一开始 complexity gate 把系统冲掉
                nn.init.zeros_(self.env_gate_proj[0].weight)
                nn.init.zeros_(self.env_gate_proj[0].bias)
                nn.init.zeros_(self.env_gate_proj[2].weight)
                nn.init.zeros_(self.env_gate_proj[2].bias)

        # --------------------------------------------------
        # 缓存最近一次 forward 的 gate 调试统计
        # tdmpc2.update() 里会从这里取数据写 TensorBoard
        # --------------------------------------------------
        self._latest_gate_stats = {}

        # --------------------------------------------------
        # training-only auxiliary: 缓存最近一次分支特征
        # temporal / combine feature 都是 [B,256]
        # --------------------------------------------------
        self._latest_temporal_feat = None
        self._latest_combine_feat = None
        # 供 risk proxy auxiliary 使用
        self._latest_env_features = None

        # --------------------------------------------------
        # 原始 combine 分支：始终按 T=1 构建
        # 注意：dual-branch 时，combine 图一定是 [B,1,1,H,W]
        # --------------------------------------------------
        self.combine_T_dim = 1 if self.merge_vis else args.sample_length
        T_kernel_size = 1

        self.group = nn.Sequential(
            nn.Conv3d(1, int(self.img_channel/2), kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(T_kernel_size, 2, 2), stride=(T_kernel_size, 2, 2)),

            nn.Conv3d(int(self.img_channel/2), int(self.img_channel), kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(T_kernel_size, 2, 2), stride=(T_kernel_size, 2, 2)),

            nn.Conv3d(int(self.img_channel), int(self.img_channel*2), kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(T_kernel_size, 2, 2), stride=(T_kernel_size, 2, 2)),

            nn.Conv3d(int(self.img_channel*2), int(self.img_channel), kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(1, 1, 1))
        )

        self.group_linear = nn.Sequential(
            nn.Linear(self.img_channel * self.combine_T_dim * int(self.img_height / 8) * int(self.img_width / 8), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU()
        )

        # --------------------------------------------------
        # H0: 原始 3D CNN + temporal 输入的严格对照分支
        # 注意：不能直接复用 group_linear，因为 group_linear 的输入维度跟 combine_T_dim 绑定
        # --------------------------------------------------
        temporal_T_dim = 1 if self.force_temporal_t1 else self.max_T
        temporal_flat_dim = self.img_channel * temporal_T_dim * int(self.img_height / 8) * int(self.img_width / 8)

        self.group_linear_temporal = nn.Sequential(
            nn.Linear(temporal_flat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU()
        )

        # --------------------------------------------------
        # temporal / hybrid 分支
        # --------------------------------------------------
        self.hybrid_encoder = HybridTemporalConditionEncoder(args)

        # 新增：ViT 图像编码分支
        # ViT 不仅 single-branch 会用，dual-branch 的 temporal correction 也可能会用
        if self.image_encoder_backbone == 'vit' or self.dual_temporal_branch_mode == 'vit':
            self.vit_encoder = VisionTransformerTemporalEncoder(args)

        # --------------------------------------------------
        # dual-branch 融合模块
        # --------------------------------------------------
        # 旧版 concat + MLP 仍然保留，方便对比
        self.dual_fuse_concat = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )

        # residual / gated residual 需要的 temporal 投影
        self.temporal_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )

        # gated residual 的 gate 网络
        self.gate_mlp = nn.Sequential(
            nn.Linear(256, 256)
        )
        # 初始化 gate 偏向小值，避免一开始 temporal 分支把 baseline 冲掉
        nn.init.constant_(self.gate_mlp[0].bias, -2.0)

        # 可学习 or 固定的全局 alpha
        alpha0 = min(max(self.hybrid_residual_alpha, 1e-4), 0.99)
        alpha_logit0 = math.log(alpha0 / (1.0 - alpha0))
        if self.hybrid_learnable_alpha:
            self.temporal_alpha_logit = nn.Parameter(torch.tensor(alpha_logit0, dtype=torch.float32))
        else:
            self.register_buffer('temporal_alpha_logit', torch.tensor(alpha_logit0, dtype=torch.float32))

    def _alpha(self):
        return torch.sigmoid(self.temporal_alpha_logit)
    
    def get_latest_gate_stats(self):
        """
        返回最近一次 forward 的 gate 统计，用于 TensorBoard 日志。
        """
        return dict(self._latest_gate_stats)

    def get_latest_temporal_attn_stats(self):
        """
        供外部（例如 world_model / tdmpc2）读取 temporal attention 的 debug 统计。
        """
        if hasattr(self, 'hybrid_encoder') and hasattr(self.hybrid_encoder, 'get_latest_temporal_attn_stats'):
            return self.hybrid_encoder.get_latest_temporal_attn_stats()
        return {}

    def get_latest_temporal_t1_stats(self):
        """
        供外部读取 temporal branch 的 T=1 实验状态。
        """
        if hasattr(self, 'hybrid_encoder') and hasattr(self.hybrid_encoder, 'get_latest_temporal_t1_stats'):
            return self.hybrid_encoder.get_latest_temporal_t1_stats()
        return {}

    def get_latest_branch_feats(self):
        """
        返回最近一次 forward 的分支特征:
            {
                'temporal': [B,256] or None,
                'combine' : [B,256] or None
            }
        """
        return {
            'temporal': self._latest_temporal_feat,
            'combine': self._latest_combine_feat,
            'env_features': self._latest_env_features,
        }
        

    def _forward_original_temporal_branch(self, image):
        """
        H0:
        直接复现原始 3D CNN temporal path（相当于 merge_vis=False 的单独分支）

        输入:
            image: [B,1,T,H,W] 或 [B,T,H,W]
        输出:
            [B,256]
        """
        if image.dim() == 4:
            image = image.unsqueeze(1)   # [B,1,T,H,W]
        elif image.dim() != 5:
            raise ValueError(f"Unexpected temporal image shape: {image.shape}")

        state = self.group(image)
        state = state.view(state.size(0), -1)
        state = self.group_linear_temporal(state)
        return state

    def set_combine_branch_requires_grad(self, flag: bool):
        """
        用于实验 2：
        前若干步冻结 combine 分支，只训练 temporal 分支 + gate。
        """
        for p in self.group.parameters():
            p.requires_grad_(flag)
        for p in self.group_linear.parameters():
            p.requires_grad_(flag)

    def _compute_env_complexity(self, combine_img):
        """
        从 combine 图中提取一个显式环境复杂度 scalar。

        combine_img:
            [B,1,1,H,W] 或 [B,1,H,W]

        返回:
            complexity: [B,1]
        """
        if combine_img.dim() == 5:
            # [B,1,1,H,W] -> [B,H,W]
            x = combine_img[:, 0, 0]
        elif combine_img.dim() == 4:
            # [B,1,H,W] -> [B,H,W]
            x = combine_img[:, 0]
        else:
            raise ValueError(f"Unexpected combine_img shape: {combine_img.shape}")

        b, h, w = x.shape

        # 当前 black_obs=True 时，自由空间更接近 1，障碍更接近 0
        occ = (1.0 - x).clamp(0.0, 1.0)  # [B,H,W]

        # ROI：底部 near_ratio 的区域 + 中间 front_ratio 的宽度
        row_start = int((1.0 - self.env_near_ratio) * h)
        row_end = h

        col_margin = int((1.0 - self.env_front_ratio) * 0.5 * w)
        col_start = col_margin
        col_end = w - col_margin

        roi = occ[:, row_start:row_end, col_start:col_end]  # [B,h_roi,w_roi]

        # 关键：这里不要 keepdim=True
        complexity = roi.mean(dim=(1, 2)).unsqueeze(1)      # [B,1]
        return complexity

    def _compute_env_features(self, combine_img):
        """
        multi_roi 版本的显式环境复杂度特征
        返回:
            [B,5]
            依次为:
            0) front_near
            1) front_mid
            2) left_near
            3) right_near
            4) global_occ
        """
        if combine_img.dim() == 5:
            x = combine_img[:, 0, 0]   # [B,H,W]
        elif combine_img.dim() == 4:
            x = combine_img[:, 0]
        else:
            raise ValueError(f"Unexpected combine_img shape: {combine_img.shape}")

        b, h, w = x.shape

        # black_obs=True 时，自由空间接近 1，障碍接近 0
        occ = (1.0 - x).clamp(0.0, 1.0)   # [B,H,W]

        # 一些简单 ROI
        near_h = max(1, int(self.env_near_ratio * h))
        mid_h = max(1, int(0.20 * h))
        half_w = w // 2
        front_margin = int((1.0 - self.env_front_ratio) * 0.5 * w)

        # robot 默认在图底部中心附近，所以“前方”用底部区域近似
        row_near_start = h - near_h
        row_mid_start = max(0, h - near_h - mid_h)

        col_front_start = front_margin
        col_front_end = w - front_margin

        # feature 1: 前方近场占据密度
        front_near = occ[:, row_near_start:h, col_front_start:col_front_end].mean(dim=(1, 2))

        # feature 2: 前方中场占据密度
        front_mid = occ[:, row_mid_start:row_near_start, col_front_start:col_front_end].mean(dim=(1, 2))

        # feature 3: 左近场
        left_near = occ[:, row_near_start:h, :half_w].mean(dim=(1, 2))

        # feature 4: 右近场
        right_near = occ[:, row_near_start:h, half_w:].mean(dim=(1, 2))

        # feature 5: 全局占据密度
        global_occ = occ.mean(dim=(1, 2))

        feats = torch.stack([front_near, front_mid, left_near, right_near, global_occ], dim=1)  # [B,5]
        return feats
        
    def _forward_original_branch(self, image):
        """
        原始 combine 图分支:
            image: [B,1,1,H,W]
        """
        state = self.group(image)
        # print("state 0", state.shape)
        state = state.view(-1, self.img_channel * self.combine_T_dim * int(self.img_height / 8) * int(self.img_width / 8))
        # print("state 1", state.shape)
        state = self.group_linear(state)
        # print("state 2", state.shape)
        return state

    def _fuse_dual(self, feat_combine, feat_temporal, combine_img=None):
        """
        dual-branch 融合逻辑

        参数:
            feat_combine: [B,256]
            feat_temporal:[B,256]
            combine_img  : [B,1,1,H,W]，用于计算显式环境复杂度
        """
        mode = self.hybrid_fusion_mode

        # 默认清空，避免日志拿到旧值
        self._latest_gate_stats = {}

        # 1) combine-only recovery test
        if mode == 'combine_only':
            if self.hybrid_log_gate_stats:
                self._latest_gate_stats = {
                    'alpha': 0.0,
                    'gate_mean': 0.0,
                    'gate_std': 0.0,
                    'gate_min': 0.0,
                    'gate_max': 0.0,
                    'env_complexity_mean': 0.0,
                    'delta_norm': 0.0,
                    'combine_norm': float(feat_combine.norm(dim=-1).mean().detach().cpu().item()),
                    'delta_over_combine': 0.0,
                }
            return feat_combine

        # 2) temporal-only debug
        if mode == 'temporal_only':
            if self.hybrid_log_gate_stats:
                self._latest_gate_stats = {
                    'alpha': 1.0,
                    'gate_mean': 1.0,
                    'gate_std': 0.0,
                    'gate_min': 1.0,
                    'gate_max': 1.0,
                    'env_complexity_mean': 0.0,
                    'delta_norm': float(feat_temporal.norm(dim=-1).mean().detach().cpu().item()),
                    'combine_norm': 0.0,
                    'delta_over_combine': 0.0,
                }
            return feat_temporal

        # 3) 旧版 concat + MLP（保留做对照）
        if mode == 'concat_mlp':
            fused = torch.cat([feat_combine, feat_temporal], dim=-1)
            out = self.dual_fuse_concat(fused)
            if self.hybrid_log_gate_stats:
                self._latest_gate_stats = {
                    'alpha': 1.0,
                    'gate_mean': 1.0,
                    'gate_std': 0.0,
                    'gate_min': 1.0,
                    'gate_max': 1.0,
                    'env_complexity_mean': 0.0,
                    'delta_norm': float(feat_temporal.norm(dim=-1).mean().detach().cpu().item()),
                    'combine_norm': float(feat_combine.norm(dim=-1).mean().detach().cpu().item()),
                    'delta_over_combine': float(
                        (feat_temporal.norm(dim=-1).mean() / (feat_combine.norm(dim=-1).mean() + 1e-8)).detach().cpu().item()
                    ),
                }
            return out

        # 4) residual fusion
        if mode == 'residual':
            delta = self.temporal_proj(feat_temporal)  # [B,256]
            alpha = self._alpha()                      # scalar
            out = feat_combine + alpha * delta

            if self.hybrid_log_gate_stats:
                combine_norm = feat_combine.norm(dim=-1).mean()
                delta_norm = delta.norm(dim=-1).mean()
                self._latest_gate_stats = {
                    'alpha': float(alpha.detach().cpu().item()),
                    'gate_mean': 1.0,
                    'gate_std': 0.0,
                    'gate_min': 1.0,
                    'gate_max': 1.0,
                    'env_complexity_mean': 0.0,
                    'delta_norm': float(delta_norm.detach().cpu().item()),
                    'combine_norm': float(combine_norm.detach().cpu().item()),
                    'delta_over_combine': float((delta_norm / (combine_norm + 1e-8)).detach().cpu().item()),
                }
            return out

        # 5) gated residual fusion（推荐）
        if mode == 'gated_residual':
            delta = self.temporal_proj(feat_temporal)   # [B,256]
            alpha = self._alpha()                       # scalar

            gate_logits = self.gate_mlp(feat_combine)   # [B,256]
            env_complexity_mean = 0.0

            # 缓存当前环境代理特征，供 risk proxy auxiliary 使用
            if combine_img is not None:
                if getattr(self, 'env_feature_mode', 'scalar') == 'multi_roi':
                    self._latest_env_features = self._compute_env_features(combine_img)
                else:
                    self._latest_env_features = self._compute_env_complexity(combine_img)
            else:
                self._latest_env_features = None

            # 新增：显式环境复杂度引导
            if self.use_env_complexity_gate and (combine_img is not None):
                if self.env_feature_mode == 'scalar':
                    env_complexity = self._compute_env_complexity(combine_img)      # [B,1]
                    env_complexity_mean = float(env_complexity.mean().detach().cpu().item())
                    env_bias = self.env_gate_proj(env_complexity)                   # [B,256]
                else:
                    env_features = self._compute_env_features(combine_img)          # [B,5]
                    env_complexity_mean = float(env_features.mean().detach().cpu().item())
                    env_bias = self.env_gate_proj(env_features)                     # [B,256]

                gate_logits = gate_logits + self.env_gate_scale * env_bias
            
            gate = torch.sigmoid(gate_logits)                                   # [B,256]
            out = feat_combine + alpha * gate * delta

            if self.hybrid_log_gate_stats:
                combine_norm = feat_combine.norm(dim=-1).mean()
                delta_norm = delta.norm(dim=-1).mean()
                self._latest_gate_stats = {
                    'alpha': float(alpha.detach().cpu().item()),
                    'gate_mean': float(gate.mean().detach().cpu().item()),
                    'gate_std': float(gate.std().detach().cpu().item()),
                    'gate_min': float(gate.min().detach().cpu().item()),
                    'gate_max': float(gate.max().detach().cpu().item()),
                    'env_complexity_mean': env_complexity_mean,
                    'env_feature_mode': 0.0 if self.env_feature_mode == 'scalar' else 1.0,
                    'delta_norm': float(delta_norm.detach().cpu().item()),
                    'combine_norm': float(combine_norm.detach().cpu().item()),
                    'delta_over_combine': float((delta_norm / (combine_norm + 1e-8)).detach().cpu().item()),
                }
            return out

        raise NotImplementedError(f'Unknown hybrid_fusion_mode: {mode}')
    
    def forward(self, image, ego=None, goal=None):
        """
        输入:
            原始单分支 / hybrid单分支:
                image: Tensor

            双分支:
                image: {
                    "combine":  [B,1,1,H,W],
                    "temporal": [B,1,T,H,W]
                }
        """
        # --------------------------------------------------
        # 新增：ViT 路线
        # --------------------------------------------------
        # --------------------------------------------------
        # ViT single-branch 路线
        # 注意：dual-branch 时不要在这里提前 return
        # --------------------------------------------------
        if self.image_encoder_backbone == 'vit' and (not self.use_dual_branch_encoder):
            feat = self.vit_encoder(image, ego=ego, goal=goal)
            self._latest_temporal_feat = feat
            self._latest_combine_feat = None
            return feat
                

        # --------------------------------------------------
        # 双分支模式
        # --------------------------------------------------
        if self.use_dual_branch_encoder:
            assert isinstance(image, dict), "dual-branch mode expects image as dict"
            combine_img = image["combine"]
            temporal_img = image["temporal"]

            # combine-only 模式：完全退化回 baseline，用来检查 dual-branch plumbing
            if self.hybrid_fusion_mode == 'combine_only':
                return self._forward_original_branch(combine_img)

            feat_combine = self._forward_original_branch(combine_img)

            # dual-branch temporal correction 分支来源：
            # 1) hybrid     -> 当前默认 hybrid branch
            # 2) orig_3dcnn -> H4a
            # 3) h3         -> H4b
            # 4) vit        -> dual-branch + ViT temporal correction
            if self.dual_temporal_branch_mode == 'orig_3dcnn':
                feat_temporal = self._forward_original_temporal_branch(temporal_img)

            elif self.dual_temporal_branch_mode == 'h3':
                feat_temporal = self.hybrid_encoder.forward_with_mode(
                    temporal_img, ego=ego, goal=goal, mode_override='3d_stem_flatten_mlp'
                )

            elif self.dual_temporal_branch_mode == 'vit':
                feat_temporal = self.vit_encoder(temporal_img, ego=ego, goal=goal)

            else:  # 'hybrid'
                feat_temporal = self.hybrid_encoder(temporal_img, ego=ego, goal=goal)

            # 缓存最近一次 dual-branch 分支特征，供 training-only auxiliary 使用
            self._latest_combine_feat = feat_combine
            self._latest_temporal_feat = feat_temporal

            return self._fuse_dual(feat_combine, feat_temporal, combine_img=combine_img)
            
        # --------------------------------------------------
        # 单分支 hybrid
        # --------------------------------------------------
        if self.use_hybrid_attn_encoder:
            # H0：单分支时，直接用原始 3D CNN + temporal 输入
            if self.temporal_encoder_mode == 'orig_3dcnn':
                feat = self._forward_original_temporal_branch(image)
                self._latest_temporal_feat = feat
                self._latest_combine_feat = None
                self._latest_env_features = None
                return feat

            # H1 / H2 / H3 / 默认 hybrid
            feat = self.hybrid_encoder(image, ego=ego, goal=goal)
            self._latest_temporal_feat = feat
            self._latest_combine_feat = None
            self._latest_env_features = None
            return feat
        

        # --------------------------------------------------
        # 原始单分支
        # --------------------------------------------------
        feat = self._forward_original_branch(image)
        self._latest_temporal_feat = None
        self._latest_combine_feat = feat
        return feat
        

class NavigationCNN(nn.Module):
    def __init__(self, output_dim):
        super(NavigationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 8 * 64, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x