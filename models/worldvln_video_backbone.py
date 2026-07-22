"""Frozen WorldVLN/InfinityStar video predictor adapter for Genie-Envisioner."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldVLNVideoBackbone(nn.Module):
    """Expose either InfinityStar visual hidden states or predicted latents as Zb."""

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        genie_root = Path(__file__).resolve().parents[1]
        configured_root = os.environ.get("WORLDVLN_ROOT", str(config.get("worldvln_root", "../../worldWAM/WorldVLN.code")))
        worldvln_root = Path(configured_root).expanduser()
        if not worldvln_root.is_absolute():
            worldvln_root = genie_root / worldvln_root
        worldvln_root = worldvln_root.resolve()

        def resolve_asset(key: str) -> Path:
            value = Path(str(config[key])).expanduser()
            return value.resolve() if value.is_absolute() else (worldvln_root / value).resolve()

        runtime_root = resolve_asset("runtime_root")
        if not (runtime_root / "infinity").is_dir():
            raise FileNotFoundError(
                f"WorldVLN runtime root is invalid: {runtime_root}. "
                f"Resolved WORLDVLN_ROOT={worldvln_root}. Set WORLDVLN_ROOT to the WorldVLN.code directory if needed."
            )
        sys.path.insert(0, str(runtime_root))

        from tools.closed_loop_streaming_infer_480p_81f import _make_args
        from tools.infinity_streaming_session import InfinityStreamingSession
        from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer

        checkpoint_path = resolve_asset("checkpoint_path")
        vae_path = resolve_asset("vae_path")
        text_encoder_path = resolve_asset("text_encoder_path")
        for label, path in (
            ("InfinityStar checkpoint", checkpoint_path),
            ("VideoVAE checkpoint", vae_path),
            ("text encoder", text_encoder_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"WorldVLN {label} not found: {path}")
        checkpoint = str(checkpoint_path)
        args = _make_args(
            ckpt=checkpoint,
            pn=str(config.get("pn", "0.40M")),
            fps=int(config.get("fps", 16)),
            num_frames=int(config.get("prediction_frames", 49)),
            seed=int(config.get("seed", 0)),
            dynamic_scale_schedule=str(config.get("dynamic_scale_schedule", "infinity_elegant_clip4frames_v2_allpt")),
            mask_type=str(config.get("mask_type", "infinity_elegant_clip4frames_v2_allpt")),
            cfg=float(config.get("cfg", 34.0)),
            tau_image=float(config.get("tau_image", 1.0)),
            tau_video=float(config.get("tau_video", 0.4)),
        )
        args.vae_path = str(vae_path)
        args.text_encoder_ckpt = str(text_encoder_path)
        args.frames_inner_clip = int(config.get("frames_inner_clip", 4))

        self.args = args
        self.feature_mode = str(config.get("feature_mode", "intermediate")).lower()
        if self.feature_mode not in {"intermediate", "predicted_latent"}:
            raise ValueError(
                f"Unsupported WorldVLN feature_mode={self.feature_mode!r}; "
                "expected 'intermediate' or 'predicted_latent'."
            )
        self.cfg_scale = float(config.get("cfg", 34.0))
        self.tau_image = float(config.get("tau_image", 1.0))
        self.tau_video = float(config.get("tau_video", 0.4))
        self.top_k = int(config.get("top_k", 900))
        self.top_p = float(config.get("top_p", 0.97))
        self.prediction_frames = int(config.get("prediction_frames", 49))
        self.seed = int(config.get("seed", 0))
        self.context_frames = int(config.get("context_frames", 8))
        self.output_tokens = int(config.get("output_tokens", 8))

        vae = load_visual_tokenizer(args).float().to(device)
        infinity = load_transformer(vae, args).to(device)
        infinity.eval().requires_grad_(False)
        vae.eval().requires_grad_(False)
        if self.feature_mode == "predicted_latent":
            tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
            text_encoder.eval().requires_grad_(False)
        else:
            tokenizer, text_encoder = None, None
        self.infinity = infinity
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.session_class = InfinityStreamingSession
        self.h_div_w_template = float(config.get("h_div_w_template", 1.0))
        self.feature_dim = int(infinity.C if self.feature_mode == "intermediate" else vae.codebook_dim)
        configured_dim = int(config.get("latent_channels", self.feature_dim))
        if configured_dim != self.feature_dim:
            raise ValueError(
                f"WorldVLN feature_mode={self.feature_mode!r} produces dimension {self.feature_dim}, "
                f"but latent_channels={configured_dim}. Update the config to match the selected feature mode."
            )

    @torch.no_grad()
    def _extract_intermediate_one(self, video: torch.Tensor) -> torch.Tensor:
        session = self.session_class(
            args=self.args,
            infinity_model=self.infinity,
            vae=self.vae,
            text_tokenizer=None,
            text_encoder=None,
            h_div_w_template=self.h_div_w_template,
        )
        schedule = session.build_schedule_for_num_frames(self.prediction_frames)
        context = video[:, : self.context_frames]
        context = F.interpolate(
            context.permute(1, 0, 2, 3),
            size=(schedule.tgt_h, schedule.tgt_w),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        hidden_states = session.compute_kv_cache_gt(context, write_cache=False)
        # [1,L,C] -> a fixed [1,output_tokens,C] sequence for Za/projector.
        return F.adaptive_avg_pool1d(
            hidden_states.float().transpose(1, 2), self.output_tokens
        ).transpose(1, 2)

    @torch.no_grad()
    def _predict_one(self, video: torch.Tensor, prompt: str, sample_seed: int) -> torch.Tensor:
        session = self.session_class(
            args=self.args,
            infinity_model=self.infinity,
            vae=self.vae,
            text_tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            h_div_w_template=self.h_div_w_template,
        )
        schedule = session.build_schedule_for_num_frames(self.prediction_frames)
        context = video[:, : self.context_frames]
        context = F.interpolate(
            context.permute(1, 0, 2, 3),
            size=(schedule.tgt_h, schedule.tgt_w),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        session.reset(prompt, cfg_scale=self.cfg_scale)
        session.compute_kv_cache_gt(context)

        text_cond, negative_cond = session._text_cond_tuple
        tau = [self.tau_image] * schedule.tower_split_index + [self.tau_video] * (
            len(schedule.scale_schedule) - schedule.tower_split_index
        )
        cfg = [self.cfg_scale] * len(schedule.scale_schedule)
        model_dtype = next(iter(self.infinity.parameters())).dtype
        with torch.amp.autocast("cuda", dtype=model_dtype):
            summed_codes = self.infinity.autoregressive_infer(
                vae=self.vae,
                scale_schedule=schedule.scale_schedule,
                label_B_or_BLT=text_cond,
                negative_label_B_or_BLT=negative_cond,
                B=1,
                g_seed=sample_seed,
                cfg_list=cfg,
                tau_list=tau,
                top_k=self.top_k,
                top_p=self.top_p,
                gt_leak=-1,
                gt_ls_Bl=None,
                low_vram_mode=True,
                args=self.args,
                get_visual_rope_embeds=session.get_visual_rope_embeds,
                context_info=schedule.context_info,
                kv_cache_reset=False,
                skip_text_forward=True,
                cache_text_as_gt=False,
                extra_ref_text_scale_inds=[session.gt_obs_cache_key],
                return_summed_code_only=True,
            )
        return summed_codes

    @torch.no_grad()
    def forward(self, video: torch.Tensor, prompts: list[str], global_step: int = 0) -> torch.Tensor:
        if video.ndim != 6:
            raise ValueError(f"Expected video [B,C,V,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] != 1:
            raise ValueError("WorldVLN adapter currently supports exactly one camera view.")
        outputs = []
        for index in range(video.shape[0]):
            if self.feature_mode == "intermediate":
                tokens = self._extract_intermediate_one(video[index, :, 0])
            else:
                codes = self._predict_one(video[index, :, 0], str(prompts[index]), self.seed + global_step + index)
                # [1,C,T,H,W] -> temporal tokens [1,T,C]. Spatial pooling preserves prediction time.
                tokens = codes.float().mean(dim=(-1, -2)).transpose(1, 2)
                tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.output_tokens).transpose(1, 2)
            outputs.append(tokens[0])
        return torch.stack(outputs, dim=0).to(video.device)


class WorldVLNFeatureProjector(nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.SiLU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"WorldVLN latent channel mismatch: configured {self.input_dim}, got {tokens.shape[-1]}. "
                "Check the loaded InfinityStar VAE/use_feat_proj checkpoint contract."
            )
        return self.net(tokens)
