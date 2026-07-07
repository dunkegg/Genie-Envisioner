import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


class BEVRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)
        grid = grid / self.base_seq_length
        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


@maybe_allow_in_graph
class BEVTransformerBlock(nn.Module):
    def __init__(
        self,
        attention_class,
        attention_args,
        dim: int = 512,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = attention_class(**attention_args[0])

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = attention_class(**attention_args[1])

        self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        return hidden_states


def add_bev_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu",
    norm_eps: float = 1e-6,
    bev_in_channels: int = 1,
    bev_out_channels: int = None,
    bev_patch_size: int = 4,
    bev_num_attention_heads: int = 16,
    bev_attention_head_dim: int = 32,
    bev_rope_dim: int = None,
    bev_base_seq_length: int = 1024,
    bev_final_embeddings: bool = True,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    attention_class=None,
    attention_processor=None,
    **kwargs,
):
    if bev_out_channels is None:
        bev_out_channels = bev_in_channels

    self.bev_in_channels = bev_in_channels
    self.bev_out_channels = bev_out_channels
    self.bev_patch_size = bev_patch_size
    self.bev_inner_dim = bev_num_attention_heads * bev_attention_head_dim

    self.bev_proj_in = nn.Conv2d(
        bev_in_channels,
        self.bev_inner_dim,
        kernel_size=bev_patch_size,
        stride=bev_patch_size,
    )
    self.bev_scale_shift_table = nn.Parameter(torch.randn(2, self.bev_inner_dim) / self.bev_inner_dim**0.5)
    self.bev_time_embed = AdaLayerNormSingle(self.bev_inner_dim, use_additional_conditions=False)

    if bev_rope_dim is None:
        bev_rope_dim = self.bev_inner_dim
    self.bev_rope = BEVRotaryPosEmbed(
        dim=bev_rope_dim,
        base_seq_length=bev_base_seq_length,
        theta=10000.0,
    )

    attention_args = [
        dict(
            query_dim=self.bev_inner_dim,
            heads=bev_num_attention_heads,
            kv_heads=bev_num_attention_heads,
            dim_head=bev_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=attention_processor,
        ),
        dict(
            query_dim=self.bev_inner_dim,
            heads=bev_num_attention_heads,
            kv_heads=bev_num_attention_heads,
            dim_head=bev_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=inner_dim,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=attention_processor,
        ),
    ]

    self.bev_blocks = nn.ModuleList(
        [
            BEVTransformerBlock(
                attention_class=attention_class,
                attention_args=attention_args,
                dim=self.bev_inner_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
            )
            for _ in range(num_layers)
        ]
    )

    self.bev_proj_out = nn.Linear(self.bev_inner_dim, bev_out_channels * bev_patch_size * bev_patch_size)
    self.bev_final_embeddings = bev_final_embeddings
    if not self.bev_final_embeddings:
        self.bev_proj_extra = nn.Linear(self.bev_inner_dim, self.bev_inner_dim)
    self.bev_norm_out = nn.LayerNorm(self.bev_inner_dim, eps=1e-6, elementwise_affine=False)


def preprocessing_bev_states(
    self,
    bev_states: torch.Tensor = None,
    bev_timestep: torch.LongTensor = None,
):
    assert self.bev_expert is True
    assert bev_states is not None and bev_timestep is not None

    if bev_states.ndim == 3:
        bev_states = bev_states.unsqueeze(1)
    assert bev_states.ndim == 4, "bev_states should have shape [B, C, H, W] or [B, H, W]"

    batch_size, _, bev_height, bev_width = bev_states.shape
    patch_size = self.bev_patch_size
    assert bev_height % patch_size == 0 and bev_width % patch_size == 0

    bev_tokens = self.bev_proj_in(bev_states)
    bev_hidden_states = rearrange(bev_tokens, "b c h w -> b (h w) c")
    bev_seq_length = bev_hidden_states.shape[1]
    bev_rotary_emb = self.bev_rope(bev_hidden_states, bev_seq_length)

    if bev_timestep.ndim == 0:
        bev_timestep = bev_timestep[None].repeat(batch_size)
    elif bev_timestep.ndim == 2 and bev_timestep.shape[1] == bev_seq_length:
        bev_timestep = bev_timestep[:, :1]

    bev_temb, bev_embedded_timestep = self.bev_time_embed(
        bev_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=bev_hidden_states.dtype,
    )
    bev_temb = bev_temb.view(batch_size, -1, bev_temb.size(-1))
    bev_embedded_timestep = bev_embedded_timestep.view(batch_size, -1, bev_embedded_timestep.size(-1))

    return bev_temb, bev_embedded_timestep, bev_rotary_emb, bev_hidden_states, bev_height, bev_width


def unpatchify_bev_output(
    self,
    bev_hidden_states: torch.Tensor,
    bev_height: int,
    bev_width: int,
) -> torch.Tensor:
    patch_size = self.bev_patch_size
    patch_height = bev_height // patch_size
    patch_width = bev_width // patch_size
    bev_output = self.bev_proj_out(bev_hidden_states)
    bev_output = rearrange(
        bev_output,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=patch_height,
        w=patch_width,
        c=self.bev_out_channels,
        p1=patch_size,
        p2=patch_size,
    )
    return bev_output

