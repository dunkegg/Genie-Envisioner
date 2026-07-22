from typing import Iterable, Optional

import torch
import torch.nn as nn


ADAPTOR_SIZES = {
    "small": (512,),
    "base": (1024, 1024),
    "large": (2048, 2048, 1024),
}


class ZaAdaptor(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 512,
        hidden_dims: Optional[Iterable[int]] = None,
        size: str = "base",
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()
        if hidden_dims is None:
            if size not in ADAPTOR_SIZES:
                raise KeyError(f"Unknown adaptor size '{size}'. Choose from {sorted(ADAPTOR_SIZES)}.")
            hidden_dims = ADAPTOR_SIZES[size]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool = pool

        layers = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, int(hidden_dim)),
                    nn.SiLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def pool_zb(self, zb: torch.Tensor) -> torch.Tensor:
        if zb.ndim == 2:
            return zb
        if zb.ndim != 3:
            raise ValueError(f"Expected Zb with shape [B, L, C] or [B, C], got {tuple(zb.shape)}.")

        if self.pool == "mean":
            return zb.mean(dim=1)
        if self.pool == "first":
            return zb[:, 0]
        if self.pool == "last":
            return zb[:, -1]
        raise NotImplementedError(f"unsupported Zb pool mode: {self.pool}")

    def forward(self, zb: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool_zb(zb))


def pool_za_target(za: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if za.ndim == 2:
        return za
    if za.ndim != 3:
        raise ValueError(f"Expected Za with shape [B, T, C] or [B, C], got {tuple(za.shape)}.")

    if mode == "mean":
        return za.mean(dim=1)
    if mode == "first":
        return za[:, 0]
    if mode == "last":
        return za[:, -1]
    raise NotImplementedError(f"unsupported Za target pool mode: {mode}")
