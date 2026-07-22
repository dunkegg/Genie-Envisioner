import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from yaml import Loader, load

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
repo_dir = os.path.dirname(current_dir)
sys.path.insert(0, repo_dir)

from models.beta_fusion import BetaZBFusion
from models.latent_adaptor import ZaAdaptor
from models.worldvln_video_backbone import WorldVLNFeatureProjector, WorldVLNVideoBackbone
from utils import import_custom_class
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent
from utils.model_utils import forward_pass, load_condition_models, load_diffusion_model, load_latent_models, load_vae_models


class ZaAdaptorActor:
    def __init__(
        self,
        config_file,
        adaptor_checkpoint=None,
        device="cuda",
        dtype=torch.bfloat16,
        n_prev=None,
        threshold=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.dtype = dtype

        cd = load(open(config_file, "r"), Loader=Loader)
        self.args = argparse.Namespace(**cd)
        self.server_cfg = getattr(self.args, "server", {})

        adaptor_cfg = getattr(self.args, "za_adaptor", {})
        if adaptor_checkpoint is None:
            adaptor_checkpoint = adaptor_cfg.get("checkpoint_path", None)
        if adaptor_checkpoint is None:
            raise ValueError("Set --adaptor_checkpoint or za_adaptor.checkpoint_path in the YAML config.")
        self.adaptor_checkpoint = adaptor_checkpoint

        self.n_prev = int(n_prev if n_prev is not None else self.server_cfg.get("n_prev", self.args.data["train"]["n_previous"]))
        self.threshold = int(threshold if threshold is not None else self.server_cfg.get("threshold", 1))
        self.resize = tuple(self.args.data["train"]["sample_size"])
        self.beta_key = self.server_cfg.get("beta_key", "beta")
        self.state_key = self.server_cfg.get("state_key", "robot_state")
        self.goal_key = self.server_cfg.get("goal_key", "goal_rel")
        self.prompt_default = self.server_cfg.get("prompt", "")

        self.prepare_models()
        self.reset()

    def prepare_models(self):
        payload = torch.load(self.adaptor_checkpoint, map_location="cpu")
        checkpoint_worldvln_cfg = payload.get("worldvln_video_backbone", {}) if isinstance(payload, dict) else {}
        yaml_worldvln_cfg = getattr(self.args, "worldvln_video_backbone", {})
        worldvln_cfg = {**checkpoint_worldvln_cfg, **yaml_worldvln_cfg}
        self.use_worldvln_backbone = bool(worldvln_cfg.get("enabled", False))

        if self.use_worldvln_backbone:
            self.worldvln_backbone = WorldVLNVideoBackbone(worldvln_cfg, device=self.device)
            self.worldvln_feature_projector = WorldVLNFeatureProjector(
                input_dim=self.worldvln_backbone.feature_dim,
                output_dim=int(worldvln_cfg.get("output_dim", 2048)),
            ).to(self.device, dtype=torch.float32).eval()
            if not isinstance(payload, dict) or "worldvln_feature_projector_state_dict" not in payload:
                raise KeyError(
                    f"{self.adaptor_checkpoint} enables WorldVLN but does not contain "
                    "worldvln_feature_projector_state_dict."
                )
            self.worldvln_feature_projector.load_state_dict(payload["worldvln_feature_projector_state_dict"])
            self.worldvln_backbone.eval().requires_grad_(False)
            self.worldvln_feature_projector.eval().requires_grad_(False)
            self.tokenizer = None
            self.text_encoder = None
            self.vae = None
            self.diffusion_model = None
            self.spatial_down_ratio = None
            self.temporal_down_ratio = None
        else:
            self.worldvln_backbone = None
            self.worldvln_feature_projector = None
            self._prepare_ltx_backbone()

        adaptor_cfg = getattr(self.args, "za_adaptor", {})
        checkpoint_adaptor_cfg = payload.get("za_adaptor", {}) if isinstance(payload, dict) else {}
        merged_adaptor_cfg = {**checkpoint_adaptor_cfg, **adaptor_cfg}
        self.adaptor = ZaAdaptor(
            input_dim=int(merged_adaptor_cfg.get("input_dim", 2048)),
            output_dim=int(merged_adaptor_cfg.get("output_dim", 512)),
            hidden_dims=merged_adaptor_cfg.get("hidden_dims", None),
            size=merged_adaptor_cfg.get("size", "base"),
            dropout=float(merged_adaptor_cfg.get("dropout", 0.0)),
            pool=merged_adaptor_cfg.get("pool", "mean"),
        ).to(self.device, dtype=torch.float32).eval()

        fusion_cfg = getattr(self.args, "beta_fusion", {})
        checkpoint_fusion_cfg = payload.get("beta_fusion", {}) if isinstance(payload, dict) else {}
        merged_fusion_cfg = {**checkpoint_fusion_cfg, **fusion_cfg}
        self.beta_fusion = None
        if merged_fusion_cfg.get("enabled", False):
            self.beta_fusion = BetaZBFusion(
                zb_dim=int(merged_fusion_cfg.get("zb_dim", merged_adaptor_cfg.get("input_dim", 2048))),
                beta_dim=int(merged_fusion_cfg.get("beta_dim", 5)),
                fusion_dim=int(merged_fusion_cfg.get("fusion_dim", 512)),
                beta_hidden_dim=int(merged_fusion_cfg.get("beta_hidden_dim", 256)),
                num_heads=int(merged_fusion_cfg.get("num_heads", 8)),
                num_layers=int(merged_fusion_cfg.get("num_layers", 1)),
                beta_tokens=int(merged_fusion_cfg.get("beta_tokens", 4)),
                mlp_ratio=float(merged_fusion_cfg.get("mlp_ratio", 2.0)),
                dropout=float(merged_fusion_cfg.get("dropout", 0.0)),
                beta_pool=merged_fusion_cfg.get("beta_pool", "mean"),
            ).to(self.device, dtype=torch.float32).eval()

        if isinstance(payload, dict) and "adaptor_state_dict" in payload:
            self.adaptor.load_state_dict(payload["adaptor_state_dict"])
        elif isinstance(payload, dict) and "state_dict" in payload:
            self.adaptor.load_state_dict(payload["state_dict"])
        else:
            self.adaptor.load_state_dict(payload)

        if self.beta_fusion is not None:
            if "beta_fusion_state_dict" not in payload:
                raise KeyError(f"{self.adaptor_checkpoint} does not contain beta_fusion_state_dict.")
            self.beta_fusion.load_state_dict(payload["beta_fusion_state_dict"])

    def _prepare_ltx_backbone(self):
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class,
            self.args.pretrained_model_name_or_path
            if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path")
            else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True,
        )
        self.tokenizer = cond_models["tokenizer"]
        self.text_encoder = cond_models["text_encoder"].to(self.device, dtype=self.dtype).eval()

        vae_class = import_custom_class(self.args.vae_class, getattr(self.args, "vae_class_path", "transformers"))
        if getattr(self.args, "vae_path", False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(self.device, dtype=self.dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(
                self.device, dtype=self.dtype
            ).eval()
        if isinstance(self.vae.latents_mean, list):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, list):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if getattr(self.args, "enable_slicing", False):
            self.vae.enable_slicing()
        if getattr(self.args, "enable_tiling", False):
            self.vae.enable_tiling()
        self.spatial_down_ratio = self.vae.spatial_compression_ratio
        self.temporal_down_ratio = self.vae.temporal_compression_ratio

        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model["model_path"],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model["config"],
        ).to(self.device, dtype=self.dtype).eval()

    def reset(self):
        self.obs = []
        self.buffer = []
        self.count = 0

    def _prepare_obs(self, obs):
        obs = np.asarray(obs)
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 127.5 - 1.0
            obs = np.transpose(obs, (0, 3, 1, 2))
        elif obs.ndim == 4 and obs.shape[-1] == 3:
            obs = np.transpose(obs, (0, 3, 1, 2))
        if obs.ndim != 4:
            raise ValueError(f"Expected obs shape [V,H,W,3] or [V,3,H,W], got {obs.shape}.")
        return torch.as_tensor(obs, dtype=self.dtype, device=self.device)

    def _build_beta(self, request):
        fusion_cfg = getattr(self.args, "beta_fusion", {})
        expected_beta_dim = int(fusion_cfg.get("beta_dim", 5))
        if self.beta_key in request:
            beta = np.asarray(request[self.beta_key], dtype=np.float32).reshape(-1)
        else:
            if self.state_key in request:
                state = np.asarray(request[self.state_key], dtype=np.float32).reshape(-1)
            elif "state" in request:
                state = np.asarray(request["state"], dtype=np.float32).reshape(-1)
            else:
                raise KeyError(f"Request must contain '{self.beta_key}' or '{self.state_key}'/'state'.")

            if state.shape[0] < 3:
                raise ValueError(f"robot state must contain at least [vx, vy, vyaw], got shape {state.shape}.")

            if self.goal_key in request:
                goal_rel = np.asarray(request[self.goal_key], dtype=np.float32).reshape(-1)
            elif "target" in request:
                goal_rel = np.asarray(request["target"], dtype=np.float32).reshape(-1)
            else:
                raise KeyError(f"Request must contain '{self.goal_key}' when '{self.beta_key}' is absent.")

            if goal_rel.shape[0] < 2:
                raise ValueError(f"goal_rel must contain [distance, theta], got shape {goal_rel.shape}.")
            state_dim = expected_beta_dim - 2
            if state_dim == 3:
                beta_state = state[:3]
            elif state.shape[0] >= state_dim:
                beta_state = state[:state_dim]
            else:
                raise ValueError(
                    f"beta_fusion.beta_dim={expected_beta_dim} requires robot_state with at least {state_dim} "
                    f"values plus goal_rel[2], but got robot_state shape {state.shape}. "
                    f"Send '{self.beta_key}' directly with {expected_beta_dim} values, or send a matching "
                    f"'{self.state_key}' and '{self.goal_key}'."
                )
            beta = np.concatenate([beta_state, goal_rel[:2]], axis=0).astype(np.float32)

        if beta.shape[0] != expected_beta_dim:
            raise ValueError(
                f"Expected beta with {expected_beta_dim} values from beta_fusion.beta_dim, got {beta.shape[0]}. "
                f"Send '{self.beta_key}' directly with {expected_beta_dim} values, or adjust the server YAML."
            )

        return torch.as_tensor(beta, dtype=torch.float32, device=self.device).unsqueeze(0)

    def update_observation_history(self, obs, prompt=None, execution_step=1):
        """Advance the rolling RGB window without running either video backbone."""
        if prompt is None:
            prompt = self.prompt_default
        if "<reset>" in prompt:
            self.reset()
            prompt = prompt.replace("<reset>", "")

        obs = self._prepare_obs(obs)
        if not self.obs:
            self.obs = [obs] * self.n_prev
            self.count = self.threshold - 1
            self.buffer = [self.obs[-1]]
        else:
            self.count += int(execution_step)
            if self.count >= self.threshold:
                self.count = 0
                self.obs.pop(0)
                self.obs[-1] = self.buffer[0]
                self.obs.append(obs)
            else:
                self.obs[-1] = obs
            self.buffer = [self.obs[-1]]

        n_view, _, raw_height, raw_width = obs.shape
        obs_tensor = torch.stack(self.obs, dim=1)
        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w").unsqueeze(0)
        return obs_tensor, prompt, (raw_height, raw_width), n_view

    def encode_zb_from_history(self, obs_tensor, prompt, raw_size, n_view):
        """Run the expensive frozen video backbone for the current history window."""
        if self.use_worldvln_backbone:
            with torch.no_grad():
                world_tokens = self.worldvln_backbone(
                    obs_tensor.float(), [prompt], global_step=0
                )
                zb = self.worldvln_feature_projector(world_tokens).float()
        else:
            raw_height, raw_width = raw_size
            video = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")
            mem_size = self.n_prev
            mem = video[:, :, :mem_size]
            future_video = video[:, :, mem_size - 1 : mem_size]
            raw_frames = future_video.shape[2]
            latent_frames = raw_frames // self.temporal_down_ratio + 1 + mem_size
            latent_height = raw_height // self.spatial_down_ratio
            latent_width = raw_width // self.spatial_down_ratio

            with torch.no_grad():
                mem_latents, future_video_latents = get_latents(
                    self.vae,
                    mem,
                    future_video,
                    device=self.device,
                    dtype=self.dtype,
                )
                mem_latents = rearrange(
                    mem_latents,
                    "(b v m) (h w) c -> (b v) c m h w",
                    b=1,
                    m=mem_size,
                    h=latent_height,
                )
                future_video_latents = rearrange(
                    future_video_latents,
                    "(b v) (f h w) c -> (b v) c f h w",
                    b=1,
                    h=latent_height,
                    w=latent_width,
                )
                latents = torch.cat((mem_latents, future_video_latents), dim=2)
                latents = rearrange(latents, "bv c f h w -> bv (f h w) c")

                _, conditioning_mask, _ = gen_noise_from_condition_frame_latent(
                    mem_latents,
                    latent_frames,
                    latent_height,
                    latent_width,
                    noise_to_condition_frames=getattr(self.args, "noise_to_first_frame", 0.1),
                )
                timesteps = torch.zeros(latents.shape[:2], device=self.device, dtype=torch.long)

                text_conds = get_text_conditions(self.tokenizer, self.text_encoder, prompt)
                pred_all = forward_pass(
                    model=self.diffusion_model,
                    timesteps=timesteps,
                    noisy_latents=latents,
                    prompt_embeds=text_conds["prompt_embeds"],
                    prompt_attention_mask=text_conds["prompt_attention_mask"],
                    num_frames=latent_frames,
                    height=latent_height,
                    width=latent_width,
                    n_view=n_view,
                    return_video=False,
                    return_zb=True,
                    condition_mask=conditioning_mask,
                )["latents"]
                zb = pred_all["zb"].detach().float()
        return zb.detach()

    def adapt_zb_tensor(self, zb, trainable_adaptor=False, **request):
        """Apply current beta conditioning and ZaAdaptor to a cached visual Zb."""
        beta = self._build_beta(request)
        self.adaptor.train(trainable_adaptor)
        if self.beta_fusion is not None:
            self.beta_fusion.train(trainable_adaptor)
        if self.beta_fusion is not None:
            zb = self.beta_fusion(zb, beta)
        z = self.adaptor(zb)
        if not trainable_adaptor:
            z = z.detach()
        return z

    def encode_z_tensor(self, obs, prompt=None, execution_step=1, trainable_adaptor=False, **request):
        obs_tensor, prompt, raw_size, n_view = self.update_observation_history(
            obs, prompt=prompt, execution_step=execution_step
        )
        zb = self.encode_zb_from_history(obs_tensor, prompt, raw_size, n_view)
        return self.adapt_zb_tensor(zb, trainable_adaptor=trainable_adaptor, **request)

    @torch.no_grad()
    def encode_z(self, obs, prompt=None, execution_step=1, **request):
        z = self.encode_z_tensor(
            obs=obs,
            prompt=prompt,
            execution_step=execution_step,
            trainable_adaptor=False,
            **request,
        )
        return z.detach().cpu().numpy()[0]
