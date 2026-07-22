import math

import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timestep.device) / max(half - 1, 1))
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))


class ZaBEVDiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, frames: int, spatial_tokens: int):
        super().__init__()
        self.frames = frames
        self.spatial_tokens = spatial_tokens
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, condition: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(temb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        # Factorized spatial/temporal attention keeps the 18-frame 128x256 BEV affordable.
        h_spatial = rearrange(h, "b (t s) d -> (b t) s d", t=self.frames, s=self.spatial_tokens)
        spatial = self.self_attn(h_spatial, h_spatial, h_spatial, need_weights=False)[0]
        spatial = rearrange(spatial, "(b t) s d -> b (t s) d", b=x.shape[0], t=self.frames)
        x = x + gate1[:, None] * spatial
        h_temporal = rearrange(self.norm_temporal(x), "b (t s) d -> (b s) t d", t=self.frames, s=self.spatial_tokens)
        temporal = self.temporal_attn(h_temporal, h_temporal, h_temporal, need_weights=False)[0]
        x = x + rearrange(temporal, "(b s) t d -> b (t s) d", b=x.shape[0], s=self.spatial_tokens)
        x = x + self.cross_attn(self.norm2(x), condition, condition, need_weights=False)[0]
        h = self.norm3(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.ff(h)


class BEVPixelRefinementBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ZaBEVDiT(nn.Module):
    """Flow-matching DiT that predicts a BEV sequence conditioned on Za-prime."""
    def __init__(self, za_dim=512, channels=1, height=128, width=256, frames=18, patch_size=16,
                 dim=512, depth=8, num_heads=8, mlp_ratio=4.0, dropout=0.0,
                 refine_channels=0, refine_depth=0):
        super().__init__()
        if height % patch_size or width % patch_size:
            raise ValueError("BEV height and width must be divisible by patch_size")
        self.channels, self.height, self.width, self.frames = channels, height, width, frames
        self.patch_size = patch_size
        self.grid_h, self.grid_w = height // patch_size, width // patch_size
        self.patch_in = nn.Conv2d(channels, dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, frames * self.grid_h * self.grid_w, dim) * 0.02)
        self.za_proj = nn.Sequential(nn.LayerNorm(za_dim), nn.Linear(za_dim, dim))
        self.time_embed = nn.Sequential(SinusoidalTimestepEmbedding(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        spatial_tokens = self.grid_h * self.grid_w
        self.blocks = nn.ModuleList([
            ZaBEVDiTBlock(dim, num_heads, mlp_ratio, dropout, frames, spatial_tokens) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim)
        self.patch_out = nn.Linear(dim, channels * patch_size * patch_size)
        self.refine = None
        if refine_channels > 0 and refine_depth > 0:
            self.refine = nn.Sequential(
                nn.Conv2d(2 * channels, refine_channels, 3, padding=1),
                nn.SiLU(),
                *[BEVPixelRefinementBlock(refine_channels) for _ in range(refine_depth)],
                nn.Conv2d(refine_channels, channels, 3, padding=1),
            )
            # Start as an exact residual identity so old coarse behavior is preserved.
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def forward(self, noisy_bev: torch.Tensor, timestep: torch.Tensor, za_prime: torch.Tensor) -> torch.Tensor:
        if noisy_bev.ndim != 5:
            raise ValueError(f"Expected noisy_bev [B,T,C,H,W], got {tuple(noisy_bev.shape)}")
        b, t, c, h, w = noisy_bev.shape
        if (t, c, h, w) != (self.frames, self.channels, self.height, self.width):
            raise ValueError(f"Expected BEV [B,{self.frames},{self.channels},{self.height},{self.width}], got {tuple(noisy_bev.shape)}")
        x = self.patch_in(rearrange(noisy_bev, "b t c h w -> (b t) c h w"))
        x = rearrange(x, "(b t) d h w -> b (t h w) d", b=b, t=t) + self.pos_embed
        condition = self.za_proj(za_prime).unsqueeze(1)
        temb = self.time_embed(timestep)
        for block in self.blocks:
            x = block(x, condition, temb)
        x = self.patch_out(self.norm_out(x))
        coarse = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", t=t, h=self.grid_h,
                           w=self.grid_w, c=c, p=self.patch_size, q=self.patch_size)
        if self.refine is None:
            return coarse
        refine_input = torch.cat([coarse, noisy_bev], dim=2)
        residual = self.refine(rearrange(refine_input, "b t c h w -> (b t) c h w"))
        return coarse + rearrange(residual, "(b t) c h w -> b t c h w", b=b, t=t)

    @staticmethod
    def to_dataset_uint8(bev: torch.Tensor) -> torch.Tensor:
        return ((bev.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
