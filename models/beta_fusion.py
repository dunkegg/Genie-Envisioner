from typing import Optional

import torch
import torch.nn as nn


BETA_FUSION_SIZES = {
    "small": {
        "fusion_dim": 256,
        "beta_hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 1,
        "beta_tokens": 2,
        "mlp_ratio": 2.0,
    },
    "base": {
        "fusion_dim": 512,
        "beta_hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 1,
        "beta_tokens": 4,
        "mlp_ratio": 2.0,
    },
    "large": {
        "fusion_dim": 1024,
        "beta_hidden_dim": 512,
        "num_heads": 16,
        "num_layers": 2,
        "beta_tokens": 8,
        "mlp_ratio": 2.0,
    },
}


class BetaZBFusionBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2048,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, zb: torch.Tensor, beta_tokens: torch.Tensor) -> torch.Tensor:
        self_in = self.norm_self(zb)
        zb = zb + self.self_attn(self_in, self_in, self_in, need_weights=False)[0]

        cross_in = self.norm_cross(zb)
        zb = zb + self.cross_attn(cross_in, beta_tokens, beta_tokens, need_weights=False)[0]

        zb = zb + self.ff(self.norm_ff(zb))
        return zb


class BetaZBFusion(nn.Module):
    def __init__(
        self,
        zb_dim: int = 2048,
        beta_dim: int = 5,
        size: str = "base",
        fusion_dim: Optional[int] = None,
        beta_hidden_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        beta_tokens: Optional[int] = None,
        mlp_ratio: Optional[float] = None,
        dropout: float = 0.0,
        beta_pool: str = "mean",
    ):
        super().__init__()
        if size not in BETA_FUSION_SIZES:
            raise KeyError(f"Unknown beta fusion size '{size}'. Choose from {sorted(BETA_FUSION_SIZES)}.")
        size_cfg = BETA_FUSION_SIZES[size]
        fusion_dim = int(fusion_dim if fusion_dim is not None else size_cfg["fusion_dim"])
        beta_hidden_dim = int(beta_hidden_dim if beta_hidden_dim is not None else size_cfg["beta_hidden_dim"])
        num_heads = int(num_heads if num_heads is not None else size_cfg["num_heads"])
        num_layers = int(num_layers if num_layers is not None else size_cfg["num_layers"])
        beta_tokens = int(beta_tokens if beta_tokens is not None else size_cfg["beta_tokens"])
        mlp_ratio = float(mlp_ratio if mlp_ratio is not None else size_cfg["mlp_ratio"])

        self.zb_dim = zb_dim
        self.fusion_dim = fusion_dim
        self.beta_dim = beta_dim
        self.beta_tokens = beta_tokens
        self.beta_pool = beta_pool

        self.zb_norm = nn.LayerNorm(zb_dim)
        self.zb_in = nn.Linear(zb_dim, fusion_dim)
        self.beta_proj = nn.Sequential(
            nn.LayerNorm(beta_dim),
            nn.Linear(beta_dim, beta_hidden_dim),
            nn.SiLU(),
            nn.Linear(beta_hidden_dim, fusion_dim * beta_tokens),
        )
        self.blocks = nn.ModuleList(
            [
                BetaZBFusionBlock(
                    dim=fusion_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(fusion_dim)
        self.zb_out = nn.Linear(fusion_dim, zb_dim)

    def pool_beta(self, beta: torch.Tensor) -> torch.Tensor:
        if beta.ndim == 2:
            return beta
        if beta.ndim != 3:
            raise ValueError(f"Expected beta with shape [B, D] or [B, T, D], got {tuple(beta.shape)}.")

        if self.beta_pool == "mean":
            return beta.mean(dim=1)
        if self.beta_pool == "first":
            return beta[:, 0]
        if self.beta_pool == "last":
            return beta[:, -1]
        raise NotImplementedError(f"unsupported beta_pool: {self.beta_pool}")

    def forward(self, zb: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        beta = self.pool_beta(beta).to(device=zb.device, dtype=zb.dtype)
        if beta.shape[-1] != self.beta_dim:
            raise ValueError(
                f"BetaZBFusion expected beta_dim={self.beta_dim}, got beta shape {tuple(beta.shape)}. "
                "Set beta_fusion.beta_dim in the YAML to match the dataset beta column."
            )
        beta_tokens = self.beta_proj(beta).view(beta.shape[0], self.beta_tokens, self.fusion_dim)
        residual = zb
        zb = self.zb_in(self.zb_norm(zb))
        for block in self.blocks:
            zb = block(zb, beta_tokens)
        return residual + self.zb_out(self.out_norm(zb))
