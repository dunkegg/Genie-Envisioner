import json
import math
import os
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import DistributedType
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from einops import rearrange
from tqdm import tqdm
from PIL import Image

from models.beta_fusion import BetaZBFusion
from models.latent_adaptor import ZaAdaptor, pool_za_target
from models.za_bev_dit import ZaBEVDiT
from models.worldvln_video_backbone import WorldVLNFeatureProjector, WorldVLNVideoBackbone
from runner.ge_trainer import Trainer, logger
from utils.data_utils import (
    apply_color_jitter_to_video,
    gen_noise_from_condition_frame_latent,
    get_latents,
    get_text_conditions,
)
from utils.memory_utils import get_memory_statistics
from utils.model_utils import forward_pass, unwrap_model
from utils.optimizer_utils import get_optimizer


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _flatten_za_for_distribution(value):
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value.reshape(value.shape[0], -1).float()


def _za_distribution_stats(value, eps=1e-6):
    value = _flatten_za_for_distribution(value)
    mean = value.mean(dim=0)
    var = value.var(dim=0, unbiased=False).clamp_min(eps)
    return mean, var


def _diagonal_gaussian_kl(mean_p, var_p, mean_q, var_q, eps=1e-6):
    var_p = var_p.clamp_min(eps)
    var_q = var_q.clamp_min(eps)
    kl = 0.5 * (torch.log(var_q) - torch.log(var_p) + (var_p + (mean_p - mean_q).pow(2)) / var_q - 1.0)
    return kl.mean()


def _diagonal_gaussian_wasserstein2(mean_p, var_p, mean_q, var_q):
    mean_term = (mean_p - mean_q).pow(2)
    std_term = (var_p.sqrt() - var_q.sqrt()).pow(2)
    return (mean_term + std_term).mean()


class ZaAdaptorTrainer(Trainer):
    def _resolve_resume_checkpoint(self):
        resume = getattr(self.args, "resume_from_checkpoint", None)
        if resume in (None, False, "", "none", "None"):
            return None
        if str(resume).lower() != "latest":
            path = Path(str(resume)).expanduser()
            if path.is_dir():
                path = path / "za_adaptor.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {path}")
            return path.resolve()

        output_root = Path(self.args.output_dir).expanduser()
        candidates = [path for path in output_root.glob("*/step_*/za_adaptor.pt") if path.is_file()]
        if not candidates:
            logger.info(f"resume_from_checkpoint=latest, but no za_adaptor.pt was found under {output_root}")
            return None
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.as_posix())).resolve()

    def _is_joint_tdmpc_mode(self):
        return getattr(self.args, "train_mode", None) == "za_adaptor_tdmpc_joint"

    def _train_tdmpc_backbone(self):
        joint_cfg = _as_dict(getattr(self.args, "joint_training", {}))
        return bool(joint_cfg.get("train_tdmpc_backbone", True))

    def _za_loss_config(self):
        cfg = _as_dict(getattr(self.args, "za_loss", {}))
        return {
            "mse_weight": float(cfg.get("mse_weight", 1.0)),
            "kl_weight": float(cfg.get("kl_weight", 0.0)),
            "wasserstein_weight": float(cfg.get("wasserstein_weight", 0.0)),
            "kl_symmetric": bool(cfg.get("kl_symmetric", True)),
            "eps": float(cfg.get("eps", 1e-6)),
        }

    def _compute_za_losses(self, za_pred, za_target):
        loss_cfg = self._za_loss_config()
        mse_loss = F.mse_loss(za_pred, za_target)
        kl_loss = za_pred.new_zeros(())
        wasserstein_loss = za_pred.new_zeros(())

        if loss_cfg["kl_weight"] > 0.0 or loss_cfg["wasserstein_weight"] > 0.0:
            pred_mean, pred_var = _za_distribution_stats(za_pred, eps=loss_cfg["eps"])
            target_mean, target_var = _za_distribution_stats(za_target, eps=loss_cfg["eps"])
            if loss_cfg["kl_weight"] > 0.0:
                kl_target_pred = _diagonal_gaussian_kl(
                    target_mean,
                    target_var,
                    pred_mean,
                    pred_var,
                    eps=loss_cfg["eps"],
                )
                if loss_cfg["kl_symmetric"]:
                    kl_pred_target = _diagonal_gaussian_kl(
                        pred_mean,
                        pred_var,
                        target_mean,
                        target_var,
                        eps=loss_cfg["eps"],
                    )
                    kl_loss = 0.5 * (kl_target_pred + kl_pred_target)
                else:
                    kl_loss = kl_target_pred
            if loss_cfg["wasserstein_weight"] > 0.0:
                wasserstein_loss = _diagonal_gaussian_wasserstein2(pred_mean, pred_var, target_mean, target_var)

        total = (
            loss_cfg["mse_weight"] * mse_loss
            + loss_cfg["kl_weight"] * kl_loss
            + loss_cfg["wasserstein_weight"] * wasserstein_loss
        )
        return total, {
            "za_mse": mse_loss,
            "za_kl": kl_loss,
            "za_wasserstein": wasserstein_loss,
        }

    def prepare_models(self):
        worldvln_cfg = _as_dict(getattr(self.args, "worldvln_video_backbone", {}))
        self.use_worldvln_backbone = bool(worldvln_cfg.get("enabled", False))
        if self.use_worldvln_backbone:
            logger.info("Replacing the LTX video backbone with frozen WorldVLN InfinityStar")
            device = self.state.accelerator.device
            self.worldvln_backbone = WorldVLNVideoBackbone(worldvln_cfg, device=device)
            logger.info(
                "WorldVLN feature mode=%s, feature_dim=%d, output_tokens=%d",
                self.worldvln_backbone.feature_mode,
                self.worldvln_backbone.feature_dim,
                self.worldvln_backbone.output_tokens,
            )
            self.worldvln_feature_projector = WorldVLNFeatureProjector(
                input_dim=self.worldvln_backbone.feature_dim,
                output_dim=int(worldvln_cfg.get("output_dim", 2048)),
            ).to(device=device, dtype=torch.float32)
            self.diffusion_model = None
            self.vae = None
            self.tokenizer = None
            self.text_encoder = None
            diffusion_scheduler_class = __import__(
                self.args.diffusion_scheduler_class_path,
                fromlist=[self.args.diffusion_scheduler_class],
            ).__dict__[self.args.diffusion_scheduler_class]
            self.scheduler = diffusion_scheduler_class(**getattr(self.args, "diffusion_scheduler_args", {}))
        else:
            self.worldvln_backbone = None
            self.worldvln_feature_projector = None
            super().prepare_models()

        device = self.state.accelerator.device
        adaptor_cfg = getattr(self.args, "za_adaptor", {})
        fusion_cfg = getattr(self.args, "beta_fusion", {})
        self.adaptor = ZaAdaptor(
            input_dim=int(adaptor_cfg.get("input_dim", 2048)),
            output_dim=int(adaptor_cfg.get("output_dim", 512)),
            hidden_dims=adaptor_cfg.get("hidden_dims", None),
            size=adaptor_cfg.get("size", "base"),
            dropout=float(adaptor_cfg.get("dropout", 0.0)),
            pool=adaptor_cfg.get("pool", "mean"),
        ).to(device=device, dtype=torch.float32)

        self.beta_fusion = None
        if fusion_cfg.get("enabled", False):
            fusion_kwargs = {
                "zb_dim": int(fusion_cfg.get("zb_dim", adaptor_cfg.get("input_dim", 2048))),
                "beta_dim": int(fusion_cfg.get("beta_dim", 5)),
                "size": fusion_cfg.get("size", "base"),
                "dropout": float(fusion_cfg.get("dropout", 0.0)),
                "beta_pool": fusion_cfg.get("beta_pool", "mean"),
            }
            for key, caster in (
                ("fusion_dim", int),
                ("beta_hidden_dim", int),
                ("num_heads", int),
                ("num_layers", int),
                ("beta_tokens", int),
                ("mlp_ratio", float),
            ):
                if key in fusion_cfg:
                    fusion_kwargs[key] = caster(fusion_cfg[key])
            self.beta_fusion = BetaZBFusion(
                **fusion_kwargs,
            ).to(device=device, dtype=torch.float32)

        self.resume_checkpoint_path = self._resolve_resume_checkpoint()
        adaptor_checkpoint_path = self.resume_checkpoint_path or adaptor_cfg.get("checkpoint_path", None)
        state_dict = None
        if adaptor_checkpoint_path:
            state_dict = torch.load(adaptor_checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "adaptor_state_dict" in state_dict:
                self.adaptor.load_state_dict(state_dict["adaptor_state_dict"])
                if self.beta_fusion is not None and "beta_fusion_state_dict" in state_dict:
                    self.beta_fusion.load_state_dict(state_dict["beta_fusion_state_dict"])
                if self.worldvln_feature_projector is not None and "worldvln_feature_projector_state_dict" in state_dict:
                    projector_state = state_dict["worldvln_feature_projector_state_dict"]
                    current_state = self.worldvln_feature_projector.state_dict()
                    incompatible = {
                        key: (tuple(value.shape), tuple(current_state[key].shape))
                        for key, value in projector_state.items()
                        if key in current_state and value.shape != current_state[key].shape
                    }
                    if incompatible:
                        logger.warning(
                            "Skipping WorldVLN projector restore because feature dimensions changed: "
                            f"{incompatible}. The projector will be trained from scratch."
                        )
                    else:
                        self.worldvln_feature_projector.load_state_dict(projector_state)
            else:
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.adaptor.load_state_dict(state_dict)
            logger.info(f"Loaded Za adaptor checkpoint from {adaptor_checkpoint_path}")

        logger.info(f"Za adaptor: {self.adaptor}")
        if self.beta_fusion is not None:
            logger.info(f"Beta-Zb fusion: {self.beta_fusion}")

        bev_cfg = _as_dict(getattr(self.args, "za_bev_dit", {}))
        self.za_bev_dit = None
        if bev_cfg.get("enabled", False):
            self.za_bev_dit = ZaBEVDiT(
                za_dim=int(bev_cfg.get("za_dim", adaptor_cfg.get("output_dim", 512))),
                channels=int(bev_cfg.get("channels", 1)),
                height=int(bev_cfg.get("height", 128)),
                width=int(bev_cfg.get("width", 256)),
                frames=int(bev_cfg.get("frames", self.args.data["train"]["chunk"])),
                patch_size=int(bev_cfg.get("patch_size", 16)),
                dim=int(bev_cfg.get("dim", 512)),
                depth=int(bev_cfg.get("depth", 8)),
                num_heads=int(bev_cfg.get("num_heads", 8)),
                mlp_ratio=float(bev_cfg.get("mlp_ratio", 4.0)),
                dropout=float(bev_cfg.get("dropout", 0.0)),
                refine_channels=int(bev_cfg.get("refine_channels", 0)),
                refine_depth=int(bev_cfg.get("refine_depth", 0)),
            ).to(device=device, dtype=torch.float32)

            def load_compatible_bev_state(bev_state, source):
                current = self.za_bev_dit.state_dict()
                compatible = {
                    key: value
                    for key, value in bev_state.items()
                    if key in current and value.shape == current[key].shape
                }
                skipped = {
                    key: (tuple(value.shape), tuple(current[key].shape) if key in current else None)
                    for key, value in bev_state.items()
                    if key not in compatible
                }
                missing, unexpected = self.za_bev_dit.load_state_dict(compatible, strict=False)
                logger.info(
                    "Loaded compatible BEV weights from %s: loaded=%d, skipped=%d, missing=%d, unexpected=%d",
                    source, len(compatible), len(skipped), len(missing), len(unexpected),
                )
                if skipped:
                    logger.warning(
                        "Reinitialized incompatible BEV pixel-resolution layers from %s: %s",
                        source, sorted(skipped),
                    )

            if isinstance(state_dict, dict) and "za_bev_dit_state_dict" in state_dict:
                load_compatible_bev_state(state_dict["za_bev_dit_state_dict"], "resume checkpoint")
            bev_checkpoint = bev_cfg.get("checkpoint_path", None)
            if bev_checkpoint:
                bev_state = torch.load(bev_checkpoint, map_location="cpu")
                if isinstance(bev_state, dict):
                    bev_state = bev_state.get("za_bev_dit_state_dict", bev_state.get("state_dict", bev_state))
                load_compatible_bev_state(bev_state, str(bev_checkpoint))
            logger.info(f"Za-prime BEV DiT: {self.za_bev_dit}")

        self.tdmpc_agent = None
        self.tdmpc_model = None
        if self._is_joint_tdmpc_mode():
            self._prepare_tdmpc_model(device)
            if isinstance(state_dict, dict) and "tdmpc_model_state_dict" in state_dict:
                self.tdmpc_model.load_state_dict(state_dict["tdmpc_model_state_dict"])
                logger.info("Restored TD-MPC model state from resume checkpoint")

    def _prepare_tdmpc_model(self, device):
        tdmpc_cfg = _as_dict(getattr(self.args, "tdmpc", {}))
        project_root = tdmpc_cfg.get("project_root", "tdmpc_bridge")
        model_path = tdmpc_cfg.get("model_path", None)
        if model_path is None:
            raise ValueError("Joint TD-MPC mode requires tdmpc.model_path in the config.")

        from isaac_test_tdmpc2_external_z import load_tdmpc2_agent, load_tdmpc_cfg

        gpu = tdmpc_cfg.get("gpu", None)
        if gpu is None and device.type == "cuda":
            gpu = device.index
        runtime_cfg = load_tdmpc_cfg(
            project_root=project_root,
            tdm_cfg_args=tdmpc_cfg.get("tdmpc_cfg_args", ""),
            cfg_overrides=tdmpc_cfg.get("cfg_overrides", {}) or {},
            gpu=gpu,
        )
        self.tdmpc_agent = load_tdmpc2_agent(runtime_cfg, model_path, gpu=gpu)
        self.tdmpc_model = self.tdmpc_agent.model
        self.tdmpc_model.to(device)
        logger.info(f"Loaded TD-MPC model from {Path(model_path).expanduser().resolve()}")

    def prepare_trainable_parameters(self):
        logger.info("Freezing VAE and text encoder")
        for component in [self.vae, self.text_encoder]:
            if component is not None:
                component.requires_grad_(False)
                component.eval()

        if self.diffusion_model is not None:
            self.diffusion_model.requires_grad_(False)
            self.diffusion_model.eval()
        if self.worldvln_backbone is not None:
            self.worldvln_backbone.requires_grad_(False)
            self.worldvln_backbone.eval()
        if self.worldvln_feature_projector is not None:
            self.worldvln_feature_projector.requires_grad_(True)
            self.worldvln_feature_projector.train()

        self.adaptor.requires_grad_(True)
        self.adaptor.train()
        if self.beta_fusion is not None:
            self.beta_fusion.requires_grad_(True)
            self.beta_fusion.train()
        if self.za_bev_dit is not None:
            self.za_bev_dit.requires_grad_(True)
            self.za_bev_dit.train()
        if self.tdmpc_model is not None:
            train_tdmpc_backbone = self._train_tdmpc_backbone()
            self.tdmpc_model.requires_grad_(train_tdmpc_backbone)
            if train_tdmpc_backbone:
                self.tdmpc_model.train()
            else:
                self.tdmpc_model.eval()

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 or fp32."
            )

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        trainable_modules = [self.adaptor]
        if self.beta_fusion is not None:
            trainable_modules.append(self.beta_fusion)
        if self.worldvln_feature_projector is not None:
            trainable_modules.append(self.worldvln_feature_projector)
        if self.za_bev_dit is not None:
            trainable_modules.append(self.za_bev_dit)
        if self._is_joint_tdmpc_mode() and self._train_tdmpc_backbone():
            if self.tdmpc_model is not None:
                trainable_modules.append(self.tdmpc_model)

        joint_cfg = _as_dict(getattr(self.args, "joint_training", {}))
        trainable_params = [p for module in trainable_modules for p in module.parameters() if p.requires_grad]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable parameters: {self.state.num_trainable_parameters}")

        params_to_optimize = []
        adaptor_params = [p for p in self.adaptor.parameters() if p.requires_grad]
        if self.beta_fusion is not None:
            adaptor_params.extend([p for p in self.beta_fusion.parameters() if p.requires_grad])
        if adaptor_params:
            params_to_optimize.append({"params": adaptor_params, "lr": self.state.learning_rate})
        if self.za_bev_dit is not None:
            bev_lr = float(_as_dict(getattr(self.args, "za_bev_dit", {})).get("lr", self.state.learning_rate))
            params_to_optimize.append({"params": list(self.za_bev_dit.parameters()), "lr": bev_lr})
        if self.worldvln_feature_projector is not None:
            worldvln_lr = float(
                _as_dict(getattr(self.args, "worldvln_video_backbone", {})).get("projector_lr", self.state.learning_rate)
            )
            params_to_optimize.append({"params": list(self.worldvln_feature_projector.parameters()), "lr": worldvln_lr})
        if self._is_joint_tdmpc_mode() and self._train_tdmpc_backbone():
            if self.tdmpc_model is not None:
                tdmpc_params = [p for p in self.tdmpc_model.parameters() if p.requires_grad]
                if tdmpc_params:
                    params_to_optimize.append(
                        {"params": tdmpc_params, "lr": float(joint_cfg.get("tdmpc_lr", self.state.learning_rate))}
                    )
        self.optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.state.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit=self.args.optimizer_8bit,
            use_torchao=self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

    def prepare_for_training(self):
        if self._is_joint_tdmpc_mode():
            modules = [self.adaptor]
            if self.beta_fusion is not None:
                modules.append(self.beta_fusion)
            if self.za_bev_dit is not None:
                modules.append(self.za_bev_dit)
            if self.worldvln_feature_projector is not None:
                modules.append(self.worldvln_feature_projector)
            if self.tdmpc_model is not None and self._train_tdmpc_backbone():
                modules.append(self.tdmpc_model)
            prepared = self.state.accelerator.prepare(
                *modules,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
            prepared_modules = prepared[: len(modules)]
            idx = 0
            self.adaptor = prepared_modules[idx]
            idx += 1
            if self.beta_fusion is not None:
                self.beta_fusion = prepared_modules[idx]
                idx += 1
            if self.za_bev_dit is not None:
                self.za_bev_dit = prepared_modules[idx]
                idx += 1
            if self.worldvln_feature_projector is not None:
                self.worldvln_feature_projector = prepared_modules[idx]
                idx += 1
            if self.tdmpc_model is not None and self._train_tdmpc_backbone():
                self.tdmpc_model = prepared_modules[idx]
                self.tdmpc_agent.model = self.tdmpc_model
            self.optimizer, self.train_dataloader, self.lr_scheduler = prepared[-3:]

        elif self.beta_fusion is None and self.za_bev_dit is None and self.worldvln_feature_projector is None:
            self.adaptor, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
                self.adaptor, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            modules = [self.adaptor]
            if self.beta_fusion is not None:
                modules.append(self.beta_fusion)
            if self.za_bev_dit is not None:
                modules.append(self.za_bev_dit)
            if self.worldvln_feature_projector is not None:
                modules.append(self.worldvln_feature_projector)
            prepared = self.state.accelerator.prepare(*modules, self.optimizer, self.train_dataloader, self.lr_scheduler)
            idx = 0
            self.adaptor = prepared[idx]; idx += 1
            if self.beta_fusion is not None:
                self.beta_fusion = prepared[idx]; idx += 1
            if self.za_bev_dit is not None:
                self.za_bev_dit = prepared[idx]; idx += 1
            if self.worldvln_feature_projector is not None:
                self.worldvln_feature_projector = prepared[idx]; idx += 1
            self.optimizer, self.train_dataloader, self.lr_scheduler = prepared[-3:]

        self.resume_global_step = 0
        if self.resume_checkpoint_path is not None:
            checkpoint = torch.load(self.resume_checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                logger.warning("Resume checkpoint has no training-state dictionary; optimizer starts fresh.")
                return
            self.resume_global_step = int(checkpoint.get("global_step", 0))
            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logger.info("Restored optimizer state")
                except (ValueError, RuntimeError) as error:
                    logger.warning(
                        "Optimizer state is incompatible with the upgraded BEV head; "
                        f"optimizer starts fresh: {error}"
                    )
            else:
                logger.warning("Old checkpoint has no optimizer_state_dict; optimizer starts fresh.")
            if "lr_scheduler_state_dict" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                logger.info("Restored LR scheduler state")
            else:
                logger.warning("Old checkpoint has no lr_scheduler_state_dict; LR scheduler starts fresh.")
            logger.info(f"Resuming training from global_step={self.resume_global_step}: {self.resume_checkpoint_path}")

    def _trainable_for_accumulate(self):
        modules = [self.adaptor]
        if self.beta_fusion is not None:
            modules.append(self.beta_fusion)
        if self.za_bev_dit is not None:
            modules.append(self.za_bev_dit)
        if self.worldvln_feature_projector is not None:
            modules.append(self.worldvln_feature_projector)
        if self._is_joint_tdmpc_mode() and self._train_tdmpc_backbone():
            if self.tdmpc_model is not None:
                modules.append(self.tdmpc_model)
        return modules

    def _build_latent_inputs(self, batch, accelerator, weight_dtype):
        video = batch["video"].to(accelerator.device, dtype=weight_dtype).contiguous()
        batch_size, _, n_view, _, _, _ = video.shape
        video = rearrange(video, "b c v t h w -> (b v) c t h w")

        if self.args.use_color_jitter:
            video = apply_color_jitter_to_video(video)

        mem_size = self.args.data["train"]["n_previous"]
        mem = video[:, :, :mem_size]
        future_video = video[:, :, mem_size:]

        _, _, raw_frames, raw_height, raw_width = future_video.shape
        latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
        latent_height = raw_height // self.SPATIAL_DOWN_RATIO
        latent_width = raw_width // self.SPATIAL_DOWN_RATIO

        mem_latents, future_video_latents = get_latents(self.vae, mem, future_video)
        mem_latents = rearrange(
            mem_latents,
            "(b v m) (h w) c -> (b v) c m h w",
            b=batch_size,
            m=mem_size,
            h=latent_height,
        )
        future_video_latents = rearrange(
            future_video_latents,
            "(b v) (f h w) c -> (b v) c f h w",
            b=batch_size,
            h=latent_height,
            w=latent_width,
        )
        latents = torch.cat((mem_latents, future_video_latents), dim=2)
        latents = rearrange(latents, "bv c f h w -> bv (f h w) c")

        return latents, mem_latents, batch_size, n_view, latent_frames, latent_height, latent_width

    def _build_noisy_latents(self, latents, mem_latents, batch_size, n_view, latent_frames, latent_height, latent_width):
        adaptor_cfg = getattr(self.args, "za_adaptor", {})
        timestep_mode = adaptor_cfg.get("timestep_mode", "clean")
        device = latents.device
        dtype = latents.dtype

        if timestep_mode == "clean":
            noisy_latents = latents
            _, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                mem_latents,
                latent_frames,
                latent_height,
                latent_width,
                noise_to_condition_frames=self.args.noise_to_first_frame,
            )
            if self.args.pixel_wise_timestep:
                timesteps = torch.zeros(latents.shape[:2], device=device, dtype=torch.long)
            else:
                timesteps = torch.zeros((latents.shape[0], latent_frames), device=device, dtype=torch.long)
            return noisy_latents, timesteps, conditioning_mask

        if timestep_mode != "random":
            raise NotImplementedError(f"unsupported za_adaptor.timestep_mode: {timestep_mode}")

        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=device, dtype=dtype)
        weights = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.args.flow_logit_mean,
            logit_std=self.args.flow_logit_std,
            mode_scale=self.args.flow_mode_scale,
        )
        weights = rearrange(weights.unsqueeze(1).repeat(1, n_view), "b v -> (b v)")
        indices = (weights * self.scheduler.config.num_train_timesteps).long()
        sigmas = scheduler_sigmas[indices]
        timesteps = (sigmas * 1000.0).long()

        noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
            mem_latents,
            latent_frames,
            latent_height,
            latent_width,
            noise_to_condition_frames=self.args.noise_to_first_frame,
        )
        if self.args.pixel_wise_timestep:
            timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
        else:
            timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

        ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
        noisy_latents = (1.0 - ss) * latents + ss * noise
        return noisy_latents, timesteps, conditioning_mask

    def _save_adaptor(self, accelerator, global_step):
        model_to_save = unwrap_model(accelerator, self.adaptor)
        fusion_to_save = unwrap_model(accelerator, self.beta_fusion) if self.beta_fusion is not None else None
        bev_to_save = unwrap_model(accelerator, self.za_bev_dit) if self.za_bev_dit is not None else None
        worldvln_projector_to_save = (
            unwrap_model(accelerator, self.worldvln_feature_projector)
            if self.worldvln_feature_projector is not None
            else None
        )
        model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, "za_adaptor.pt")
        payload = {
            "state_dict": model_to_save.state_dict(),
            "adaptor_state_dict": model_to_save.state_dict(),
            "za_adaptor": getattr(self.args, "za_adaptor", {}),
            "beta_fusion": getattr(self.args, "beta_fusion", {}),
            "global_step": global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }
        if fusion_to_save is not None:
            payload["beta_fusion_state_dict"] = fusion_to_save.state_dict()
        if bev_to_save is not None:
            payload["za_bev_dit_state_dict"] = bev_to_save.state_dict()
            payload["za_bev_dit"] = getattr(self.args, "za_bev_dit", {})
        if worldvln_projector_to_save is not None:
            payload["worldvln_feature_projector_state_dict"] = worldvln_projector_to_save.state_dict()
            payload["worldvln_video_backbone"] = getattr(self.args, "worldvln_video_backbone", {})
        if self._is_joint_tdmpc_mode() and self._train_tdmpc_backbone():
            if self.tdmpc_model is not None:
                payload["tdmpc_model_state_dict"] = unwrap_model(accelerator, self.tdmpc_model).state_dict()
        # Publish checkpoints atomically so "latest" never selects a half-written file.
        temporary_save_path = save_path + ".tmp"
        torch.save(payload, temporary_save_path)
        os.replace(temporary_save_path, save_path)
        logger.info(f"Saved Za adaptor checkpoint to {save_path}")

    def _compute_bev_loss(self, za_prime, batch, accelerator):
        if self.za_bev_dit is None:
            return None, {}
        cfg = _as_dict(getattr(self.args, "za_bev_dit", {}))
        bev_key = cfg.get("bev_key", self.args.data["train"].get("bev_map_key", "bev_map"))
        if bev_key not in batch:
            raise KeyError(f"Batch does not contain '{bev_key}'. Run h5_tdmpc_2_lerobot.py with BEV extraction enabled.")
        bev = batch[bev_key].to(accelerator.device, dtype=torch.float32).contiguous()
        batch_size = bev.shape[0]
        scheduler_sigmas = self.scheduler.sigmas.to(device=bev.device, dtype=bev.dtype)
        density = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.args.flow_logit_mean,
            logit_std=self.args.flow_logit_std,
            mode_scale=self.args.flow_mode_scale,
        )
        indices = (density * self.scheduler.config.num_train_timesteps).long().clamp_max(len(scheduler_sigmas) - 1)
        sigmas = scheduler_sigmas[indices]
        noise = torch.randn_like(bev)
        sigma_view = sigmas.view(batch_size, 1, 1, 1, 1)
        noisy_bev = (1.0 - sigma_view) * bev + sigma_view * noise
        pred = self.za_bev_dit(noisy_bev, (sigmas * 1000.0).long(), za_prime)
        target = noise - bev
        weights = compute_loss_weighting_for_sd3(
            weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
        ).view(batch_size, 1, 1, 1, 1)
        pred_f = pred.float()
        target_f = target.float()
        flow_mse = (weights.float() * (pred_f - target_f).pow(2)).mean()

        # Rectified flow: x_t=(1-sigma)*x0+sigma*noise, v=noise-x0,
        # hence x0_hat=x_t-sigma*v_hat. Directly supervise reconstructed
        # pixels and spatial derivatives instead of relying on velocity MSE only.
        pred_x0 = noisy_bev.float() - sigma_view.float() * pred_f
        target_x0 = bev.float()
        pixel_mse = (pred_x0 - target_x0).pow(2).mean()
        pixel_l1 = (pred_x0 - target_x0).abs().mean()

        pred_dx = pred_x0[..., :, 1:] - pred_x0[..., :, :-1]
        target_dx = target_x0[..., :, 1:] - target_x0[..., :, :-1]
        pred_dy = pred_x0[..., 1:, :] - pred_x0[..., :-1, :]
        target_dy = target_x0[..., 1:, :] - target_x0[..., :-1, :]
        edge_l1 = 0.5 * (
            (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()
        )

        total = (
            float(cfg.get("flow_mse_weight", 1.0)) * flow_mse
            + float(cfg.get("pixel_mse_weight", 0.0)) * pixel_mse
            + float(cfg.get("pixel_l1_weight", 0.0)) * pixel_l1
            + float(cfg.get("edge_l1_weight", 0.0)) * edge_l1
        ) * float(cfg.get("loss_weight", 1.0))
        return total, {
            "bev_total": total,
            "bev_flow_mse": flow_mse,
            "bev_pixel_mse": pixel_mse,
            "bev_pixel_l1": pixel_l1,
            "bev_edge_l1": edge_l1,
        }

    @torch.no_grad()
    def _infer_and_save_bev(self, accelerator, global_step):
        """Run flow-matching BEV sampling on one validation batch and save visual comparisons."""
        if self.za_bev_dit is None or not hasattr(self, "val_dataloader"):
            return
        cfg = _as_dict(getattr(self.args, "za_bev_dit", {}))
        num_samples = int(cfg.get("validation_samples", 1))
        num_steps = int(cfg.get("validation_inference_steps", 30))
        bev_key = cfg.get("bev_key", self.args.data["val"].get("bev_map_key", "bev_map"))
        batch = next(iter(self.val_dataloader))
        small_batch = {}
        for key, value in batch.items():
            small_batch[key] = value[:num_samples] if hasattr(value, "__getitem__") else value
        if bev_key not in small_batch:
            raise KeyError(f"Validation batch does not contain '{bev_key}'.")

        if self.use_worldvln_backbone:
            video = small_batch["video"].to(accelerator.device, dtype=torch.float32).contiguous()
            world_tokens = self.worldvln_backbone(video, list(small_batch["caption"]), global_step=global_step)
            zb = self.worldvln_feature_projector(world_tokens)
        else:
            weight_dtype = self.state.weight_dtype
            latents, mem_latents, batch_size, n_view, latent_frames, latent_height, latent_width = (
                self._build_latent_inputs(small_batch, accelerator, weight_dtype)
            )
            noisy_latents, timesteps, conditioning_mask = self._build_noisy_latents(
                latents, mem_latents, batch_size, n_view, latent_frames, latent_height, latent_width
            )
            text_conds = get_text_conditions(self.tokenizer, self.text_encoder, small_batch["caption"])
            pred_all = forward_pass(
                model=self.diffusion_model,
                timesteps=timesteps,
                noisy_latents=noisy_latents,
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
        if self.beta_fusion is not None:
            beta_key = _as_dict(getattr(self.args, "beta_fusion", {})).get("beta_key", "beta")
            zb = self.beta_fusion(zb, small_batch[beta_key].to(accelerator.device, dtype=torch.float32))
        za_prime = self.adaptor(zb)

        gt = small_batch[bev_key].to(accelerator.device, dtype=torch.float32)
        generator = torch.Generator(device=accelerator.device).manual_seed(int(self.args.seed) + global_step)
        sample = torch.randn(gt.shape, generator=generator, device=gt.device, dtype=gt.dtype)
        inference_scheduler = copy.deepcopy(self.scheduler)
        inference_scheduler.set_timesteps(num_steps, device=accelerator.device)
        bev_model = unwrap_model(accelerator, self.za_bev_dit)
        bev_model.eval()
        for timestep in inference_scheduler.timesteps:
            timestep_batch = timestep.expand(sample.shape[0])
            velocity = bev_model(sample, timestep_batch, za_prime)
            sample = inference_scheduler.step(velocity, timestep, sample, return_dict=False)[0]

        pred_u8 = ZaBEVDiT.to_dataset_uint8(sample).cpu().numpy()
        gt_u8 = ZaBEVDiT.to_dataset_uint8(gt).cpu().numpy()
        output_dir = Path(self.save_folder) / f"Validation_step_{global_step}" / "bev"
        output_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(pred_u8.shape[0]):
            for frame_idx in range(pred_u8.shape[1]):
                pred_frame = pred_u8[sample_idx, frame_idx, 0]
                gt_frame = gt_u8[sample_idx, frame_idx, 0]
                Image.fromarray(pred_frame, mode="L").save(output_dir / f"sample_{sample_idx:02d}_frame_{frame_idx:02d}_pred.png")
                Image.fromarray(gt_frame, mode="L").save(output_dir / f"sample_{sample_idx:02d}_frame_{frame_idx:02d}_gt.png")
                comparison = np.concatenate([gt_frame, pred_frame], axis=1)
                Image.fromarray(comparison, mode="L").save(
                    output_dir / f"sample_{sample_idx:02d}_frame_{frame_idx:02d}_gt_pred.png"
                )
            gt_strip = np.concatenate([gt_u8[sample_idx, frame, 0] for frame in range(gt_u8.shape[1])], axis=1)
            pred_strip = np.concatenate([pred_u8[sample_idx, frame, 0] for frame in range(pred_u8.shape[1])], axis=1)
            Image.fromarray(np.concatenate([gt_strip, pred_strip], axis=0), mode="L").save(
                output_dir / f"sample_{sample_idx:02d}_sequence_gt_top_pred_bottom.png"
            )
        logger.info(f"Saved BEV validation inference to {output_dir}")
        self.za_bev_dit.train()

    def _compute_tdmpc_bc_loss(self, za_pred, batch, accelerator):
        joint_cfg = _as_dict(getattr(self.args, "joint_training", {}))
        weight = float(joint_cfg.get("tdmpc_bc_weight", 0.0))
        if weight <= 0.0 or self.tdmpc_model is None:
            return None
        if "actions" not in batch:
            raise KeyError("TD-MPC BC loss requires 'actions' in the batch.")

        action_target = batch["actions"].to(accelerator.device, dtype=torch.float32).contiguous()
        action_index = int(joint_cfg.get("tdmpc_action_index", -1))
        if action_target.ndim == 3:
            action_target = action_target[:, action_index]
        action_dim = int(getattr(self.tdmpc_agent.cfg, "action_dim", action_target.shape[-1]))
        action_target = action_target[..., :action_dim]

        task = None
        if bool(getattr(self.tdmpc_agent.cfg, "multitask", False)):
            task = torch.zeros(action_target.shape[0], dtype=torch.long, device=accelerator.device)

        tdmpc_model = unwrap_model(accelerator, self.tdmpc_model)
        pi_output = tdmpc_model.pi(za_pred, task)
        if isinstance(pi_output, tuple):
            output_name = joint_cfg.get("tdmpc_bc_output", "mu")
            output_index = 1 if output_name == "sample" else 0
            pi_action = pi_output[output_index]
        else:
            pi_action = pi_output
        if pi_action.shape != action_target.shape:
            raise ValueError(
                f"TD-MPC BC shape mismatch: policy predicts {tuple(pi_action.shape)}, "
                f"target is {tuple(action_target.shape)}. Check cfg.action_dim and dataset action_key."
            )
        return F.mse_loss(pi_action.float(), action_target.float()) * weight

    def _compute_tdmpc_consistency_loss(self, za_sequence, batch, accelerator):
        joint_cfg = _as_dict(getattr(self.args, "joint_training", {}))
        weight = float(joint_cfg.get("tdmpc_consistency_weight", 0.0))
        if weight <= 0.0 or self.tdmpc_model is None:
            return None
        if za_sequence.ndim != 3 or za_sequence.shape[1] < 2:
            return None
        if "actions" not in batch:
            raise KeyError("TD-MPC consistency loss requires 'actions' in the batch.")

        actions = batch["actions"].to(accelerator.device, dtype=torch.float32).contiguous()
        if actions.ndim != 3:
            return None
        action_dim = int(getattr(self.tdmpc_agent.cfg, "action_dim", actions.shape[-1]))
        actions = actions[..., :action_dim]

        tdmpc_model = unwrap_model(accelerator, self.tdmpc_model)
        horizon = min(
            int(joint_cfg.get("tdmpc_consistency_horizon", actions.shape[1] - 1)),
            za_sequence.shape[1] - 1,
            actions.shape[1] - 1,
        )
        if horizon <= 0:
            return None

        task = None
        if bool(getattr(self.tdmpc_agent.cfg, "multitask", False)):
            task = torch.zeros(za_sequence.shape[0], dtype=torch.long, device=accelerator.device)

        z = za_sequence[:, 0].float()
        targets = za_sequence[:, 1 : horizon + 1].float()
        consistency_loss = torch.zeros((), device=accelerator.device)
        rho = float(getattr(self.tdmpc_agent.cfg, "rho", 1.0))
        teacher_forcing = bool(joint_cfg.get("tdmpc_consistency_teacher_forcing", False))
        for t in range(horizon):
            z = tdmpc_model.next(z, actions[:, t].float(), task)
            consistency_loss = consistency_loss + F.mse_loss(z.float(), targets[:, t]) * (rho**t)
            if teacher_forcing:
                z = targets[:, t]
        return consistency_loss * (weight / horizon)

    def train(self):
        logger.info("Starting Za adaptor training")
        logger.info(f"Memory before training start: {json.dumps(get_memory_statistics(), indent=4)}")

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        adaptor_cfg = getattr(self.args, "za_adaptor", {})
        fusion_cfg = getattr(self.args, "beta_fusion", {})
        target_pool = adaptor_cfg.get("target_pool", "mean")
        beta_key = fusion_cfg.get("beta_key", "beta")
        joint_tdmpc_mode = self._is_joint_tdmpc_mode()

        self.state.train_batch_size = (
            self.args.batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
            "adaptor config": adaptor_cfg,
        }
        logger.info(f"Za adaptor training configuration: {json.dumps(info, indent=4)}")

        global_step = int(getattr(self, "resume_global_step", 0))
        progress_bar = tqdm(
            range(global_step, self.state.train_steps),
            desc="Za adaptor training steps",
            disable=not accelerator.is_local_main_process,
        )

        steps_per_epoch = max(1, math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps))
        start_epoch = global_step // steps_per_epoch
        if global_step >= self.state.train_steps:
            logger.info(
                f"Checkpoint global_step={global_step} already reached configured train_steps={self.state.train_steps}; "
                "no additional optimization steps are required."
            )
            accelerator.wait_for_everyone()
            accelerator.end_training()
            return
        for epoch in range(start_epoch, self.state.train_epochs):
            self.adaptor.train()
            if self.beta_fusion is not None:
                self.beta_fusion.train()
            if self.za_bev_dit is not None:
                self.za_bev_dit.train()
            if self.worldvln_feature_projector is not None:
                self.worldvln_feature_projector.train()
            if self.worldvln_backbone is not None:
                self.worldvln_backbone.eval()
            if joint_tdmpc_mode:
                if self.diffusion_model is not None:
                    self.diffusion_model.eval()
                if self.tdmpc_model is not None:
                    if self._train_tdmpc_backbone():
                        self.tdmpc_model.train()
                    else:
                        self.tdmpc_model.eval()
            else:
                if self.diffusion_model is not None:
                    self.diffusion_model.eval()
            if self.vae is not None:
                self.vae.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()

            for batch in self.train_dataloader:
                if "za_latents" not in batch:
                    raise KeyError("Batch does not contain 'za_latents'. Set data.train.za_latent_key in the config.")
                if self.beta_fusion is not None and beta_key not in batch:
                    raise KeyError(f"Batch does not contain '{beta_key}'. Set data.train.beta_key in the config.")

                with accelerator.accumulate(self._trainable_for_accumulate()):
                    if self.use_worldvln_backbone:
                        video = batch["video"].to(accelerator.device, dtype=torch.float32).contiguous()
                        world_tokens = self.worldvln_backbone(video, list(batch["caption"]), global_step=global_step)
                        zb = self.worldvln_feature_projector(world_tokens)
                    else:
                        with torch.no_grad():
                            latents, mem_latents, batch_size, n_view, latent_frames, latent_height, latent_width = (
                                self._build_latent_inputs(batch, accelerator, weight_dtype)
                            )
                            noisy_latents, timesteps, conditioning_mask = self._build_noisy_latents(
                                latents,
                                mem_latents,
                                batch_size,
                                n_view,
                                latent_frames,
                                latent_height,
                                latent_width,
                            )
                            text_conds = get_text_conditions(self.tokenizer, self.text_encoder, batch["caption"])

                        pred_all = forward_pass(
                            model=self.diffusion_model,
                            timesteps=timesteps,
                            noisy_latents=noisy_latents,
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

                    za_sequence = batch["za_latents"].to(accelerator.device, dtype=torch.float32).contiguous()
                    za_target = pool_za_target(za_sequence, mode=target_pool)
                    if self.beta_fusion is not None:
                        beta = batch[beta_key].to(accelerator.device, dtype=torch.float32).contiguous()
                        zb = self.beta_fusion(zb, beta)
                    za_pred = self.adaptor(zb)
                    if za_pred.shape != za_target.shape:
                        raise ValueError(
                            f"Za shape mismatch: adaptor predicts {tuple(za_pred.shape)}, "
                            f"target is {tuple(za_target.shape)}. Check za_adaptor.output_dim and target pooling."
                        )

                    za_loss, za_loss_items = self._compute_za_losses(za_pred, za_target)
                    tdmpc_bc_loss = self._compute_tdmpc_bc_loss(za_pred, batch, accelerator)
                    tdmpc_consistency_loss = self._compute_tdmpc_consistency_loss(za_sequence, batch, accelerator)
                    bev_loss, bev_loss_items = self._compute_bev_loss(za_pred, batch, accelerator)
                    loss = za_loss
                    if tdmpc_bc_loss is not None:
                        loss = loss + tdmpc_bc_loss
                    if tdmpc_consistency_loss is not None:
                        loss = loss + tdmpc_consistency_loss
                    if bev_loss is not None:
                        loss = loss + bev_loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(
                            nn.ModuleList(self._trainable_for_accumulate()).parameters(),
                            self.args.max_grad_norm,
                        )
                    else:
                        grad_norm = None
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                loss_reduced = accelerator.reduce(loss.detach(), reduction="mean")
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss_reduced.item(),
                    "za_loss": accelerator.reduce(za_loss.detach(), reduction="mean").item(),
                    "za_mse": accelerator.reduce(za_loss_items["za_mse"].detach(), reduction="mean").item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if self._za_loss_config()["kl_weight"] > 0.0:
                    logs["za_kl"] = accelerator.reduce(
                        za_loss_items["za_kl"].detach(), reduction="mean"
                    ).item()
                if self._za_loss_config()["wasserstein_weight"] > 0.0:
                    logs["za_wasserstein"] = accelerator.reduce(
                        za_loss_items["za_wasserstein"].detach(), reduction="mean"
                    ).item()
                if tdmpc_bc_loss is not None:
                    logs["tdmpc_bc"] = accelerator.reduce(tdmpc_bc_loss.detach(), reduction="mean").item()
                if tdmpc_consistency_loss is not None:
                    logs["tdmpc_consistency"] = accelerator.reduce(
                        tdmpc_consistency_loss.detach(), reduction="mean"
                    ).item()
                if bev_loss is not None:
                    for key, value in bev_loss_items.items():
                        logs[key] = accelerator.reduce(value.detach(), reduction="mean").item()
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("Za adaptor total", logs["za_loss"], global_step)
                    self.writer.add_scalar("Za adaptor MSE", logs["za_mse"], global_step)
                    if "za_kl" in logs:
                        self.writer.add_scalar("Za adaptor KL", logs["za_kl"], global_step)
                    if "za_wasserstein" in logs:
                        self.writer.add_scalar("Za adaptor Wasserstein", logs["za_wasserstein"], global_step)
                    if tdmpc_bc_loss is not None:
                        self.writer.add_scalar("TD-MPC BC MSE", logs["tdmpc_bc"], global_step)
                    if tdmpc_consistency_loss is not None:
                        self.writer.add_scalar("TD-MPC consistency MSE", logs["tdmpc_consistency"], global_step)
                    if bev_loss is not None:
                        self.writer.add_scalar("Za-prime BEV total", logs["bev_total"], global_step)
                        self.writer.add_scalar("Za-prime BEV flow MSE", logs["bev_flow_mse"], global_step)
                        self.writer.add_scalar("Za-prime BEV pixel MSE", logs["bev_pixel_mse"], global_step)
                        self.writer.add_scalar("Za-prime BEV pixel L1", logs["bev_pixel_l1"], global_step)
                        self.writer.add_scalar("Za-prime BEV edge L1", logs["bev_edge_l1"], global_step)

                if (
                    accelerator.sync_gradients
                    and global_step > 0
                    and global_step % self.args.steps_to_val == 0
                    and self.za_bev_dit is not None
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self._infer_and_save_bev(accelerator, global_step)
                    accelerator.wait_for_everyone()

                if global_step > 0 and global_step % self.args.steps_to_save == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self._save_adaptor(accelerator, global_step)

                if global_step >= self.state.train_steps:
                    break

            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(get_memory_statistics(), indent=4)}")
            if global_step >= self.state.train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self._save_adaptor(accelerator, global_step)
        accelerator.end_training()
