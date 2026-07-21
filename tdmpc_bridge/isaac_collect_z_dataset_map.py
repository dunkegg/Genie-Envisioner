#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect a z-latent dataset on Isaac when Isaac already provides the TD-MPC2
laser-map image input, e.g. H x W = 128 x 256.

This script intentionally DOES NOT call env/state_interpreter.py and DOES NOT
convert a 180-dim laser scan into CostMap.  It bypasses WorldModel.multi_encode's
CostMap/CostMapDual construction and directly feeds the provided map tensor into
WorldModel.fe with the same tensor shapes that multi_encode would have produced:

  combine CNN / ViT-combine:      [B, 1, 1, H, W]
  temporal CNN / ViT-temporal:    [B, 1, T, H, W]
  dual branch CNN:                {
                                      "combine":  [B, 1, 1, H, W],
                                      "temporal": [B, 1, T, H, W],
                                   }

PYL: check isaac output                                 
Expected per-agent Isaac observation, minimum:

  {
      "laser_map": np.ndarray,     # [128,256], [1,128,256], [T,128,256],
                                    # or dict {"combine":..., "temporal":...}
      "velocity": [v, w],          # real m/s, rad/s by default
      "goal_rel": [distance, theta]# real polar in robot frame by default
  }

Alternative goal fields supported:
  - "goal_xy" or "goal_rel_xy": local x/y goal in robot frame
  - "pose" + "goal": global robot pose [x,y,yaw] and global goal [gx,gy]

The environment adapter is intentionally generic.  The imported Isaac class must
provide reset() and step(action).  step() may follow gym/gymnasium or return a
custom dict with keys like obs/reward/done/info.

How to run this python file:
export ISAAC_PROJECT=/media/nav/18a974f3-ee2a-474a-8c7b-8ee5b345bb1d10/wheel-arm/isaac_ws/isaac_work-master/scripts
export TDMPC_BRIDGE=$ISAAC_PROJECT/tdmpc_bridge
export PYTHONPATH=$ISAAC_PROJECT:$TDMPC_BRIDGE:$PYTHONPATH

python $TDMPC_BRIDGE/isaac_collect_z_dataset_map.py \
  --project-root $TDMPC_BRIDGE \
  --model-path $TDMPC_BRIDGE/checkpoints/801 \
  --output $ISAAC_PROJECT/datasets/isaac_z_dataset_v1 \
  --isaac-env-module isaac_env_adapter.random_nav_env \
  --isaac-env-class IsaacRandomNavEnv \
  --isaac-env-kwargs-json '{"num_envs":1,"num_agents":4,"headless":true}' \
  --num-agents 4 \
  --num-episodes 200 \
  --episode-steps 150 \
  --gpu 0 \
  --map-key laser_map \
  --map-input-mode combine \
  --map-normalize auto \
  --velocity-key velocity \
  --velocity-format real \
  --goal-rel-key goal_rel \
  --goal-rel-format polar \
  --action-format real \
  --chunk-size 50000 \
  --print-every 1000

<你的Isaac工程>/
├── tdmpc_bridge/
│   ├── isaac_collect_z_dataset_map.py
│   ├── exp_config.py
│   ├── tdmpc2.py
│   ├── tdmpc2_common/
│   │   ├── __init__.py
│   │   ├── world_model.py
│   │   ├── layers.py
│   │   ├── math.py
│   │   ├── init.py
│   │   ├── scale.py
│   │   └── ...
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── utils_network.py
│   │   └── ...
│   └── checkpoints/
│       └── tdmpc2_irsim_trained.pt
│
├── isaac_env_adapter/
│   ├── __init__.py
│   ├── random_nav_env.py
│
└── datasets/
    └── isaac_z_dataset/
"""


from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 may be absent on headless Isaac nodes
    cv2 = None


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[Sequence[float]]]


# -----------------------------------------------------------------------------
# Small math utilities kept local so we do not need to import State_Interpreter.
# -----------------------------------------------------------------------------


def pi2pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi



def clip_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))



def as_numpy(x: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)



def first_existing(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return default



def ensure_1d(x: Any, name: str, min_len: int = 1) -> np.ndarray:
    arr = as_numpy(x).reshape(-1)
    if arr.size < min_len:
        raise ValueError(f"{name} must contain at least {min_len} values, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Config / checkpoint loading
# -----------------------------------------------------------------------------


def add_project_root_to_syspath(project_root: Union[str, Path]) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--project-root does not exist: {root}")

    candidates = [root]
    # Common layout: navigation_research/scripts contains exp_config.py.
    candidates.append(root / "src" / "navigation_research-master" / "src" / "navigation_research" / "scripts")
    candidates.append(root / "navigation_research" / "scripts")

    chosen: Optional[Path] = None
    for c in candidates:
        if (c / "exp_config.py").exists() and (c / "tdmpc2.py").exists():
            chosen = c
            break
    if chosen is None:
        raise FileNotFoundError(
            "Could not find exp_config.py and tdmpc2.py under --project-root. "
            "Point --project-root to src/navigation_research-master/src/navigation_research/scripts."
        )

    chosen_str = str(chosen)
    if chosen_str not in sys.path:
        sys.path.insert(0, chosen_str)
    return chosen



def load_tdmpc_cfg(project_root: Union[str, Path], forwarded_argv: Sequence[str], gpu: Optional[int]) -> Tuple[Any, Path]:
    scripts_dir = add_project_root_to_syspath(project_root)
    old_argv = sys.argv[:]
    try:
        # exp_config.get_config() parses sys.argv directly.
        sys.argv = [old_argv[0]] + list(forwarded_argv)
        exp_config = importlib.import_module("exp_config")
        cfg = exp_config.get_config()
    finally:
        sys.argv = old_argv

    if gpu is not None:
        setattr(cfg, "gpu", int(gpu))
    if not hasattr(cfg, "path_dim"):
        setattr(cfg, "path_dim", 2 if bool(getattr(cfg, "deviation_mode", False)) else 0)
    if not hasattr(cfg, "goal_dim"):
        setattr(cfg, "goal_dim", 0 if bool(getattr(cfg, "no_goal", False)) else 2)
    if not hasattr(cfg, "progress_dim"):
        setattr(cfg, "progress_dim", 2 if bool(getattr(cfg, "progress_ratio_mode", False)) else 0)
    if not hasattr(cfg, "sample_length"):
        hist = int(getattr(cfg, "history_length", 8))
        interval = int(getattr(cfg, "sample_interval", 1))
        setattr(cfg, "sample_length", int(math.ceil(hist / max(interval, 1))))
    return cfg, scripts_dir



def load_tdmpc2_agent(cfg: Any, model_path: Union[str, Path]) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Your uploaded TDMPC2/WorldModel code hard-codes CUDA in several places. "
            "Run this script on a CUDA machine, or first patch .cuda()/torch.device('cuda') to use a configurable device."
        )
    torch.cuda.set_device(int(getattr(cfg, "gpu", 0)))
    TDMPC2 = importlib.import_module("tdmpc2").TDMPC2
    agent = TDMPC2(cfg)
    agent.load(str(Path(model_path).expanduser()))
    agent.model.eval()
    return agent


# -----------------------------------------------------------------------------
# Map preprocessing
# -----------------------------------------------------------------------------


@dataclass
class MapBundle:
    combine: np.ndarray      # [H, W]
    temporal: np.ndarray     # [T, H, W]
    input_kind: str


class IsaacMapPreprocessor:
    def __init__(
        self,
        cfg: Any,
        map_key: str,
        temporal_map_key: str,
        combine_map_key: str,
        map_normalize: str,
        map_invert: bool,
        map_input_mode: str,
        combine_from_temporal: str,
    ) -> None:
        self.cfg = cfg
        self.map_key = map_key
        self.temporal_map_key = temporal_map_key
        self.combine_map_key = combine_map_key
        self.map_normalize = map_normalize
        self.map_invert = bool(map_invert)
        self.map_input_mode = map_input_mode
        self.combine_from_temporal = combine_from_temporal
        self.H = int(getattr(cfg, "img_height", 128))
        self.W = int(getattr(cfg, "img_width", 256))
        self.T = int(getattr(cfg, "sample_length", 8))
        self._hist: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.T))

    def reset_history(self) -> None:
        self._hist.clear()

    def _normalize_map_values(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        mode = self.map_normalize

        if mode == "none":
            out = arr
        elif mode == "zero_one":
            # Interpret input as uint8-like if needed, otherwise min-max only when outside [0,1].
            if arr.size > 0 and (arr.max() > 1.0 or arr.min() < 0.0):
                if arr.max() > 2.0:
                    out = arr / 255.0
                else:
                    out = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            else:
                out = arr
        elif mode == "minus_one_one":
            out = (arr + 1.0) * 0.5
        elif mode == "auto":
            if arr.size == 0:
                out = arr
            elif arr.min() >= -1.05 and arr.max() <= 1.05 and arr.min() < 0.0:
                out = (arr + 1.0) * 0.5
            elif arr.max() > 2.0:
                out = arr / 255.0
            else:
                out = arr
        else:
            raise ValueError(f"unknown --map-normalize={mode}")

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        if self.map_invert:
            out = 1.0 - out
        return out

    def _resize_hw(self, img: np.ndarray) -> np.ndarray:
        """Return [H,W].  If img is [W,H], transpose; otherwise resize."""
        img = np.asarray(img, dtype=np.float32)
        if img.shape == (self.H, self.W):
            return img
        if img.shape == (self.W, self.H):
            return img.T.copy()
        if cv2 is not None:
            return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Simple nearest-neighbor fallback without cv2.
        y_idx = np.linspace(0, img.shape[0] - 1, self.H).round().astype(np.int64)
        x_idx = np.linspace(0, img.shape[1] - 1, self.W).round().astype(np.int64)
        return img[np.ix_(y_idx, x_idx)].astype(np.float32)

    def _to_hw(self, img: Any) -> np.ndarray:
        arr = self._normalize_map_values(as_numpy(img))
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"single map must be 2-D after squeeze, got shape={arr.shape}")
        return self._resize_hw(arr)

    def _to_thw(self, maps: Any) -> np.ndarray:
        arr = self._normalize_map_values(as_numpy(maps))
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr.reshape(1, *arr.shape)
        if arr.ndim != 3:
            raise ValueError(f"temporal map must be [T,H,W] or [H,W,T], got shape={arr.shape}")

        # Convert [H,W,T] to [T,H,W] when obvious.
        if arr.shape[0] == self.H and arr.shape[1] == self.W:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.shape[1] == self.H and arr.shape[2] == self.W:
            pass
        elif arr.shape[0] == self.W and arr.shape[1] == self.H:
            arr = np.transpose(arr, (2, 1, 0))
        # Otherwise assume [T,H,W] and resize each frame.

        frames = [self._resize_hw(frame) for frame in arr]
        return np.stack(frames, axis=0).astype(np.float32)

    def _temporal_to_combine(self, temporal: np.ndarray) -> np.ndarray:
        if self.combine_from_temporal == "last":
            return temporal[-1]
        if self.combine_from_temporal == "first":
            return temporal[0]
        if self.combine_from_temporal == "max":
            return temporal.max(axis=0)
        if self.combine_from_temporal == "mean":
            return temporal.mean(axis=0)
        raise ValueError(f"unknown --combine-from-temporal={self.combine_from_temporal}")

    def _fit_temporal_length(self, temporal: np.ndarray) -> np.ndarray:
        if temporal.shape[0] == self.T:
            return temporal.astype(np.float32, copy=False)
        if temporal.shape[0] > self.T:
            return temporal[-self.T:].astype(np.float32, copy=False)
        pad_count = self.T - temporal.shape[0]
        pad = np.repeat(temporal[:1], pad_count, axis=0)
        return np.concatenate([pad, temporal], axis=0).astype(np.float32, copy=False)

    def _extract_raw_maps(self, agent_obs: Mapping[str, Any]) -> Tuple[Optional[Any], Optional[Any], str]:
        combine_raw = first_existing(agent_obs, [self.combine_map_key, "combine", "combine_map", "map_combine"])
        temporal_raw = first_existing(agent_obs, [self.temporal_map_key, "temporal", "temporal_map", "map_temporal", "laser_map_temporal"])
        main_raw = first_existing(
            agent_obs,
            [self.map_key, "laser_map", "lidar_map", "costmap", "obs_map", "map", "surrounding", "image"],
        )

        if isinstance(main_raw, Mapping):
            combine_raw = combine_raw if combine_raw is not None else first_existing(main_raw, ["combine", "combine_map"])
            temporal_raw = temporal_raw if temporal_raw is not None else first_existing(main_raw, ["temporal", "temporal_map"])
            main_raw = None

        if combine_raw is not None or temporal_raw is not None:
            return combine_raw, temporal_raw, "explicit"
        if main_raw is None:
            raise KeyError(
                f"Cannot find map in agent observation. Tried keys: {self.map_key}, laser_map, lidar_map, costmap, obs_map, map, surrounding."
            )
        return main_raw, None, "main"

    def build(self, agent_id: int, agent_obs: Mapping[str, Any]) -> MapBundle:
        raw_a, raw_b, kind = self._extract_raw_maps(agent_obs)

        if kind == "explicit":
            combine: Optional[np.ndarray] = self._to_hw(raw_a) if raw_a is not None else None
            temporal: Optional[np.ndarray] = self._to_thw(raw_b) if raw_b is not None else None
            if temporal is None and combine is not None:
                temporal = self._history_from_combine(agent_id, combine)
            if combine is None and temporal is not None:
                temporal = self._fit_temporal_length(temporal)
                combine = self._temporal_to_combine(temporal)
            assert combine is not None and temporal is not None
            return MapBundle(combine=combine, temporal=self._fit_temporal_length(temporal), input_kind="explicit")

        # Main map auto interpretation.
        main_arr = as_numpy(raw_a)
        squeezed = np.squeeze(main_arr)
        mode = self.map_input_mode
        if mode == "auto":
            if squeezed.ndim == 2:
                mode = "combine"
            elif squeezed.ndim == 3:
                # [1,H,W] is a single combine map; [T,H,W] or [H,W,T] with T>1 is temporal.
                if 1 in squeezed.shape and max(squeezed.shape) in (self.H, self.W):
                    if squeezed.shape[0] == 1 or squeezed.shape[-1] == 1:
                        mode = "combine"
                    else:
                        mode = "temporal"
                else:
                    mode = "temporal"
            else:
                raise ValueError(f"Cannot infer map input mode from shape={squeezed.shape}")

        if mode == "combine":
            combine = self._to_hw(raw_a)
            temporal = self._history_from_combine(agent_id, combine)
            return MapBundle(combine=combine, temporal=temporal, input_kind="combine")
        if mode == "temporal":
            temporal = self._fit_temporal_length(self._to_thw(raw_a))
            combine = self._temporal_to_combine(temporal)
            return MapBundle(combine=combine, temporal=temporal, input_kind="temporal")
        if mode == "dual":
            # A numeric dual tensor is ambiguous.  Prefer explicit dict keys for dual.
            raise ValueError("--map-input-mode dual requires dict keys combine/temporal or combine_map/temporal_map")
        raise ValueError(f"unknown --map-input-mode={self.map_input_mode}")

    def _history_from_combine(self, agent_id: int, combine: np.ndarray) -> np.ndarray:
        h = self._hist[int(agent_id)]
        if len(h) == 0:
            for _ in range(self.T):
                h.append(combine.copy())
        else:
            h.append(combine.copy())
        if len(h) < self.T:
            while len(h) < self.T:
                h.appendleft(h[0].copy())
        return np.stack(list(h), axis=0).astype(np.float32)

    def to_surrounding_for_model(self, bundle: MapBundle) -> Any:
        device = torch.device("cuda")
        combine = torch.from_numpy(bundle.combine).to(device=device, dtype=torch.float32).view(1, 1, 1, self.H, self.W)
        temporal = torch.from_numpy(bundle.temporal).to(device=device, dtype=torch.float32).view(1, 1, self.T, self.H, self.W)

        backbone = getattr(self.cfg, "image_encoder_backbone", "cnn")
        if backbone == "vit":
            if getattr(self.cfg, "vit_input_mode", "temporal") == "combine":
                return combine
            return temporal
        if bool(getattr(self.cfg, "use_dual_branch_encoder", False)):
            return {"combine": combine, "temporal": temporal}
        if bool(getattr(self.cfg, "use_hybrid_attn_encoder", False)):
            return temporal
        # Original CNN branch follows CostMap(... combine=self.merge_vis).
        return combine if bool(getattr(self.cfg, "merge_vis", True)) else temporal


# -----------------------------------------------------------------------------
# Observation feature building: velocity / goal / optional path / progress.
# -----------------------------------------------------------------------------


@dataclass
class BuiltObservation:
    ego: np.ndarray
    goal: Optional[np.ndarray]
    path: Optional[np.ndarray]
    progress: Optional[np.ndarray]
    maps: MapBundle
    extra: Dict[str, Any]


class ObservationBuilder:
    def __init__(self, cfg: Any, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.map_builder = IsaacMapPreprocessor(
            cfg=cfg,
            map_key=args.map_key,
            temporal_map_key=args.temporal_map_key,
            combine_map_key=args.combine_map_key,
            map_normalize=args.map_normalize,
            map_invert=args.map_invert,
            map_input_mode=args.map_input_mode,
            combine_from_temporal=args.combine_from_temporal,
        )

    def reset(self) -> None:
        self.map_builder.reset_history()

    def build(self, agent_id: int, agent_obs: Mapping[str, Any], last_action_real: Optional[np.ndarray] = None) -> BuiltObservation:
        maps = self.map_builder.build(agent_id, agent_obs)
        ego, real_vel = self._build_ego(agent_obs, last_action_real)
        goal = None if bool(getattr(self.cfg, "no_goal", False)) else self._build_goal(agent_obs)
        path = self._build_path(agent_obs)
        progress = self._build_progress(agent_obs)
        extra = {
            "real_velocity": real_vel.astype(np.float32).tolist(),
            "map_kind": maps.input_kind,
            "map_min": float(np.min(maps.combine)),
            "map_max": float(np.max(maps.combine)),
        }
        for k in ["pose", "goal", "goal_rel", "goal_xy", "collision", "arrive", "reward"]:
            if k in agent_obs:
                try:
                    extra[k] = as_numpy(agent_obs[k]).tolist()
                except Exception:
                    extra[k] = agent_obs[k]
        return BuiltObservation(ego=ego, goal=goal, path=path, progress=progress, maps=maps, extra=extra)

    def _normalize_velocity_model_input(self, v: float, w: float) -> np.ndarray:
        if bool(getattr(self.cfg, "normalization_state", True)):
            return np.asarray([
                clip_float(v / float(getattr(self.cfg, "max_linear_vel", 1.0)), 0.0, 1.0),
                clip_float((w / float(getattr(self.cfg, "max_angular_vel", 1.0)) + 1.0) * 0.5, 0.0, 1.0),
            ], dtype=np.float32)
        return np.asarray([v, w], dtype=np.float32)

    def _build_ego(self, agent_obs: Mapping[str, Any], last_action_real: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        vel_raw = first_existing(agent_obs, [self.args.velocity_key, "velocity", "vel", "vw", "robot_velocity", "cmd_vw"])
        if vel_raw is None:
            if self.args.velocity_fallback == "last_action" and last_action_real is not None:
                real = np.asarray(last_action_real[:2], dtype=np.float32)
                return self._normalize_velocity_model_input(float(real[0]), float(real[1])), real
            if self.args.velocity_fallback == "zeros":
                real = np.zeros(2, dtype=np.float32)
                return self._normalize_velocity_model_input(0.0, 0.0), real
            raise KeyError("velocity is missing from Isaac observation")

        vel = ensure_1d(vel_raw, "velocity", min_len=2)[:2]
        fmt = self.args.velocity_format
        if fmt == "model_norm":
            ego = vel.astype(np.float32)
            # Approximate real velocity for metadata/action fallback only.
            real = np.asarray([
                ego[0] * float(getattr(self.cfg, "max_linear_vel", 1.0)),
                (ego[1] * 2.0 - 1.0) * float(getattr(self.cfg, "max_angular_vel", 1.0)),
            ], dtype=np.float32)
        elif fmt == "action_norm":
            real = np.asarray([
                ((float(vel[0]) + 1.0) * 0.5) * float(getattr(self.cfg, "max_linear_vel", 1.0)),
                float(vel[1]) * float(getattr(self.cfg, "max_angular_vel", 1.0)),
            ], dtype=np.float32)
            ego = self._normalize_velocity_model_input(float(real[0]), float(real[1]))
        elif fmt == "real":
            real = vel.astype(np.float32)
            ego = self._normalize_velocity_model_input(float(real[0]), float(real[1]))
        else:
            raise ValueError(f"unknown --velocity-format={fmt}")
        return ego, real

    def _goal_from_pose_goal(self, agent_obs: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        pose = first_existing(agent_obs, [self.args.pose_key, "pose", "robot_pose", "state"])
        goal = first_existing(agent_obs, [self.args.goal_key, "goal", "target", "target_position"])
        if pose is None or goal is None:
            return None
        p = ensure_1d(pose, "pose", min_len=3)
        g = ensure_1d(goal, "goal", min_len=2)
        dx = float(g[0] - p[0])
        dy = float(g[1] - p[1])
        yaw = float(p[2])
        # Robot local frame convention aligned with State_Interpreter.goal: x forward, y left.
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        dist = math.hypot(local_x, local_y)
        theta = pi2pi(math.atan2(local_y, local_x))
        return dist, theta, local_x, local_y

    def _normalize_goal_from_polar_or_xy(self, dist: float, theta: float, gx: float, gy: float) -> np.ndarray:
        if bool(getattr(self.cfg, "normalization_state", True)):
            if bool(getattr(self.cfg, "use_xy_goal", False)):
                state_range = float(getattr(self.cfg, "state_range", 8.0))
                return np.asarray([
                    clip_float(gx / state_range, -1.0, 1.0),
                    clip_float(gy / state_range, -1.0, 1.0),
                ], dtype=np.float32)
            norm_thr = float(getattr(self.cfg, "norm_threshold_goal_distance", 5.0))
            return np.asarray([
                1.0 if dist > norm_thr else clip_float(dist / norm_thr, 0.0, 1.0),
                clip_float((theta / math.pi + 1.0) * 0.5, 0.0, 1.0),
            ], dtype=np.float32)
        if bool(getattr(self.cfg, "use_xy_goal", False)):
            return np.asarray([gx, gy], dtype=np.float32)
        return np.asarray([dist, theta], dtype=np.float32)

    def _build_goal(self, agent_obs: Mapping[str, Any]) -> np.ndarray:
        fmt = self.args.goal_rel_format
        model_norm = first_existing(agent_obs, ["goal_model", "goal_norm", "goal_input"])
        if fmt == "model_norm" and model_norm is None:
            model_norm = first_existing(agent_obs, [self.args.goal_rel_key, "goal_rel", "relative_goal"])
        if model_norm is not None and fmt == "model_norm":
            return ensure_1d(model_norm, "goal model input", min_len=2)[:2].astype(np.float32)

        goal_xy_raw = first_existing(agent_obs, ["goal_xy", "goal_rel_xy", "local_goal", "relative_goal_xy"])
        goal_rel_raw = first_existing(agent_obs, [self.args.goal_rel_key, "goal_rel", "relative_goal", "goal_polar"])

        if goal_xy_raw is not None and fmt in {"auto", "xy"}:
            xy = ensure_1d(goal_xy_raw, "goal_xy", min_len=2)
            gx, gy = float(xy[0]), float(xy[1])
            dist, theta = math.hypot(gx, gy), pi2pi(math.atan2(gy, gx))
            return self._normalize_goal_from_polar_or_xy(dist, theta, gx, gy)

        if goal_rel_raw is not None and fmt in {"auto", "polar"}:
            gr = ensure_1d(goal_rel_raw, "goal_rel", min_len=2)
            dist, theta = float(gr[0]), pi2pi(float(gr[1]))
            gx, gy = dist * math.cos(theta), dist * math.sin(theta)
            return self._normalize_goal_from_polar_or_xy(dist, theta, gx, gy)

        derived = self._goal_from_pose_goal(agent_obs)
        if derived is not None:
            dist, theta, gx, gy = derived
            return self._normalize_goal_from_polar_or_xy(dist, theta, gx, gy)

        if self.args.missing_goal == "zeros":
            return np.zeros(2, dtype=np.float32)
        raise KeyError("goal relation is missing. Provide goal_rel, goal_xy, or pose+goal.")

    def _build_path(self, agent_obs: Mapping[str, Any]) -> Optional[np.ndarray]:
        if not bool(getattr(self.cfg, "deviation_mode", False)):
            return None
        raw = first_existing(agent_obs, [self.args.path_key, "path", "deviation", "path_deviation"])
        dim = int(getattr(self.cfg, "path_dim", 2) or 2)
        if raw is None:
            if self.args.missing_path == "zeros":
                return np.zeros(dim, dtype=np.float32)
            raise KeyError("cfg.deviation_mode=True but path/deviation feature is missing")
        arr = ensure_1d(raw, "path", min_len=dim)[:dim].astype(np.float32)
        return arr

    def _build_progress(self, agent_obs: Mapping[str, Any]) -> Optional[np.ndarray]:
        # The uploaded WorldModel only consumes progress in the no_goal branch.
        if not (bool(getattr(self.cfg, "no_goal", False)) and bool(getattr(self.cfg, "progress_ratio_mode", False))):
            return None
        raw = first_existing(agent_obs, [self.args.progress_key, "progress", "progress_feature"])
        dim = int(getattr(self.cfg, "progress_dim", 2) or 2)
        # Some older WorldModel slices progress with 3 values in the no_goal branch.
        if dim <= 0:
            dim = 2
        if raw is None:
            if self.args.missing_progress == "zeros":
                return np.zeros(dim, dtype=np.float32)
            raise KeyError("cfg.progress_ratio_mode=True but progress feature is missing")
        return ensure_1d(raw, "progress", min_len=dim)[:dim].astype(np.float32)


# -----------------------------------------------------------------------------
# Direct map encoder wrapper
# -----------------------------------------------------------------------------


class DirectMapZEncoder:
    def __init__(self, agent: Any, obs_builder: ObservationBuilder) -> None:
        self.agent = agent
        self.cfg = agent.cfg
        self.model = agent.model
        self.obs_builder = obs_builder
        self.device = torch.device("cuda")
        if bool(getattr(self.cfg, "mlp_obs", False)):
            raise RuntimeError(
                "cfg.mlp_obs=True means the trained encoder expects a vector, not a 256x128 map image. "
                "Use a CNN/dual-branch model, or provide a separate vector-mode collector."
            )

    @torch.no_grad()
    def encode(self, built: BuiltObservation) -> torch.Tensor:
        ego = torch.from_numpy(built.ego).to(self.device, dtype=torch.float32).view(1, -1)
        surrounding = self.obs_builder.map_builder.to_surrounding_for_model(built.maps)

        if bool(getattr(self.cfg, "no_goal", False)):
            if built.path is None:
                raise RuntimeError("no_goal=True path branch requires path feature; check cfg and --missing-path")
            path = torch.from_numpy(built.path).to(self.device, dtype=torch.float32).view(1, -1)
            if built.progress is not None:
                progress = torch.from_numpy(built.progress).to(self.device, dtype=torch.float32).view(1, -1)
                z = self.model.fe([ego, path, surrounding, progress])
            else:
                z = self.model.fe([ego, path, surrounding])
        else:
            if built.goal is None:
                raise RuntimeError("no_goal=False but goal feature is missing")
            goal = torch.from_numpy(built.goal).to(self.device, dtype=torch.float32).view(1, -1)
            if bool(getattr(self.cfg, "deviation_mode", False)):
                if built.path is None:
                    raise RuntimeError("deviation_mode=True but path feature is missing")
                path = torch.from_numpy(built.path).to(self.device, dtype=torch.float32).view(1, -1)
                z = self.model.fe([ego, goal, surrounding, path])
            else:
                z = self.model.fe([ego, goal, surrounding])
        return z.squeeze(0).detach().cpu()


# -----------------------------------------------------------------------------
# Isaac environment adapter helpers
# -----------------------------------------------------------------------------


def find_isaac_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "nav_isaaclab").is_dir() and (parent / "utils" / "cli_args.py").exists():
            return parent
    raise RuntimeError(f"Cannot find Isaac project root from: {current}")


def ensure_isaac_project_paths() -> Path:
    project_root = find_isaac_project_root()
    path_entries = [str(project_root), str(project_root / "scripts")]
    for entry in reversed(path_entries):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    ensure_repo_cli_args_module(project_root)
    return project_root


def ensure_repo_cli_args_module(project_root: Path) -> None:
    utils_dir = (project_root / "utils").resolve()

    loaded_utils = sys.modules.get("utils")
    loaded_paths = []
    if loaded_utils is not None:
        loaded_paths.extend(Path(p).resolve() for p in getattr(loaded_utils, "__path__", []))
        loaded_file = getattr(loaded_utils, "__file__", None)
        if loaded_file:
            loaded_paths.append(Path(loaded_file).resolve().parent)
    if loaded_utils is not None and utils_dir not in loaded_paths:
        for module_name in list(sys.modules):
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]

    importlib.import_module("utils.cli_args")


def parse_isaac_env_kwargs(kwargs_json: str) -> Dict[str, Any]:
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if not isinstance(kwargs, dict):
        raise ValueError("--isaac-env-kwargs-json must decode to a JSON object")
    return kwargs


def launch_isaac_app(args: argparse.Namespace, env_kwargs: Mapping[str, Any]) -> Any:
    if "headless" in env_kwargs and hasattr(args, "headless"):
        args.headless = bool(env_kwargs["headless"])
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    return app_launcher.app


def import_isaac_env(module_name: str, class_name: str, kwargs: Mapping[str, Any]) -> Any:
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**dict(kwargs))



def split_batched_dict(obs: Mapping[str, Any], num_agents: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """Split dict of batched arrays {key: [N,...]} into list of dicts."""
    if "agents" in obs and isinstance(obs["agents"], (list, tuple)):
        return [dict(x) if isinstance(x, Mapping) else {"obs": x} for x in obs["agents"]]

    # Do not split a normal single-agent dict whose map is [H,W].
    candidate_keys = [k for k, v in obs.items() if isinstance(v, (np.ndarray, torch.Tensor, list, tuple))]
    lengths: List[int] = []
    for k in candidate_keys:
        v = as_numpy(obs[k]) if not isinstance(obs[k], (list, tuple)) else np.asarray(obs[k], dtype=object)
        if v.ndim >= 1 and k not in {"laser_map", "lidar_map", "map", "costmap", "obs_map", "surrounding", "image"}:
            lengths.append(int(v.shape[0]))
    if num_agents is None and lengths:
        # Choose a common small-ish leading dimension if present.
        vals, counts = np.unique(lengths, return_counts=True)
        num_agents = int(vals[np.argmax(counts)])
    if num_agents is None or num_agents <= 1:
        return None

    out: List[Dict[str, Any]] = [dict() for _ in range(num_agents)]
    any_split = False
    for k, v in obs.items():
        arr = v
        try:
            narr = as_numpy(v)
            if narr.ndim >= 1 and narr.shape[0] == num_agents:
                # Avoid splitting a single map [H,W] when H happens to equal num_agents; unlikely but explicit.
                if k in {"laser_map", "lidar_map", "map", "costmap", "obs_map", "surrounding", "image"} and narr.ndim == 2:
                    for i in range(num_agents):
                        out[i][k] = v
                else:
                    for i in range(num_agents):
                        out[i][k] = narr[i]
                    any_split = True
            else:
                for i in range(num_agents):
                    out[i][k] = arr
        except Exception:
            for i in range(num_agents):
                out[i][k] = arr
    return out if any_split else None



def to_agent_obs_list(obs: Any, num_agents: Optional[int] = None) -> List[Dict[str, Any]]:
    if isinstance(obs, Mapping):
        # Common custom return: {"obs": [...]} or {"observations": [...]}.
        inner = first_existing(obs, ["obs", "observation", "observations", "state", "states"])
        if inner is not None and inner is not obs:
            try:
                return to_agent_obs_list(inner, num_agents=num_agents)
            except Exception:
                pass
        split = split_batched_dict(obs, num_agents=num_agents)
        if split is not None:
            return split
        return [dict(obs)]
    if isinstance(obs, (list, tuple)):
        return [dict(x) if isinstance(x, Mapping) else {"obs": x} for x in obs]
    arr = as_numpy(obs)
    if arr.ndim >= 2 and (num_agents is None or arr.shape[0] == num_agents):
        return [{"obs": arr[i]} for i in range(arr.shape[0])]
    return [{"obs": arr}]



def unpack_reset(ret: Any) -> Any:
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0]
    return ret



def unpack_step(ret: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    if isinstance(ret, Mapping):
        obs = first_existing(ret, ["obs", "observation", "observations", "state", "states", "next_obs"], ret)
        reward = first_existing(ret, ["reward", "rewards"], None)
        done = first_existing(ret, ["done", "dones", "terminated", "terminations"], None)
        truncated = first_existing(ret, ["truncated", "truncations"], None)
        if done is None and truncated is not None:
            done = truncated
        elif done is not None and truncated is not None:
            done = np.logical_or(as_numpy(done), as_numpy(truncated))
        info = first_existing(ret, ["info", "infos"], {})
        return obs, reward, done, dict(info) if isinstance(info, Mapping) else {"info": info}
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, reward, np.logical_or(as_numpy(terminated), as_numpy(truncated)), info if isinstance(info, dict) else {"info": info}
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, reward, done, info if isinstance(info, dict) else {"info": info}
        if len(ret) >= 1:
            return ret[0], None, None, {}
    return ret, None, None, {}



def done_for_agent(done: Any, agent_id: int, agent_obs: Mapping[str, Any]) -> bool:
    if "done" in agent_obs:
        return bool(np.asarray(agent_obs["done"]).item())
    if done is None:
        return False
    arr = np.asarray(done)
    if arr.ndim == 0:
        return bool(arr.item())
    if agent_id < arr.shape[0]:
        return bool(arr[agent_id])
    return bool(arr.any())


# -----------------------------------------------------------------------------
# Dataset accumulator
# -----------------------------------------------------------------------------


class ChunkedDatasetWriter:
    def __init__(self, output: Union[str, Path], chunk_size: int, store_map: bool) -> None:
        self.output = Path(output).expanduser()
        self.chunk_size = int(chunk_size)
        self.store_map = bool(store_map)
        self.output.mkdir(parents=True, exist_ok=True)
        self.chunks: List[str] = []
        self.buf: Dict[str, List[Any]] = defaultdict(list)
        self.total = 0
        self.chunk_idx = 0

    def add(self, sample: Dict[str, Any]) -> None:
        if not self.store_map:
            sample = {k: v for k, v in sample.items() if k not in {"map_combine", "map_temporal"}}
        for k, v in sample.items():
            self.buf[k].append(v)
        self.total += 1
        if self.chunk_size > 0 and len(next(iter(self.buf.values()))) >= self.chunk_size:
            self.flush()

    def _pack_value(self, values: List[Any]) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(values, dim=0)
        if isinstance(first, np.ndarray):
            try:
                return torch.from_numpy(np.stack(values, axis=0))
            except Exception:
                return values
        if isinstance(first, (float, int, bool, np.bool_)):
            return torch.as_tensor(values)
        return values

    def flush(self) -> Optional[Path]:
        if not self.buf:
            return None
        packed = {k: self._pack_value(v) for k, v in self.buf.items()}
        path = self.output / f"chunk_{self.chunk_idx:06d}.pt"
        torch.save(packed, path)
        self.chunks.append(path.name)
        self.buf.clear()
        self.chunk_idx += 1
        return path

    def write_manifest(self, meta: Dict[str, Any]) -> Path:
        manifest = {
            "total_samples": self.total,
            "num_chunks": len(self.chunks),
            "chunks": self.chunks,
            "meta": meta,
        }
        path = self.output / "manifest.pt"
        torch.save(manifest, path)
        return path

    def finish(self, meta: Dict[str, Any]) -> Path:
        self.flush()
        return self.write_manifest(meta)


def get_episode_hdf5_path(output_dir: Union[str, Path], ep_id: int) -> Path:
    return Path(output_dir).expanduser() / f"episode_{int(ep_id)}.h5"


def find_next_available_episode_id(output_dir: Union[str, Path], start_id: int = 0) -> int:
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    ep_id = max(0, int(start_id))
    skipped_ids: List[int] = []
    while get_episode_hdf5_path(output, ep_id).exists():
        skipped_ids.append(ep_id)
        ep_id += 1
    if skipped_ids:
        if len(skipped_ids) <= 5:
            skipped_text = ", ".join(str(idx) for idx in skipped_ids)
        else:
            skipped_text = f"{skipped_ids[0]}..{skipped_ids[-1]}"
        print(
            f"[Collection] Existing episode h5 found; skipped episode id(s): {skipped_text}. "
            f"Next episode id: {ep_id}",
            flush=True,
        )
    return ep_id


def value_to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (float, int, bool, np.bool_)):
        arr = np.asarray(value)
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def build_find_object_episode_frame(agent_obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    frame: Dict[str, np.ndarray] = {}
    image = first_existing(agent_obs, ["rgb", "image", "camera_rgb", "camera"])
    if image is not None:
        frame["rgb"] = value_to_numpy(image, dtype=np.uint8)

    pose = first_existing(agent_obs, ["camera_pos", "camera_pose", "pose", "robot_pose"])
    if pose is not None:
        frame["camera_pos"] = ensure_1d(pose, "camera_pos", min_len=1).astype(np.float32)

    velocity = first_existing(agent_obs, ["robot_velocity", "velocity"])
    if velocity is not None:
        frame["robot_velocity"] = ensure_1d(velocity, "robot_velocity", min_len=1).astype(np.float32)

    target_polar = first_existing(agent_obs, ["target_polar", "goal_rel", "relative_goal", "goal_polar"])
    if target_polar is not None:
        frame["target_polar"] = ensure_1d(target_polar, "target_polar", min_len=1).astype(np.float32)

    bev = first_existing(agent_obs, ["bev", "laser_map", "lidar_map", "map"])
    if bev is not None:
        bev_arr = value_to_numpy(bev)
        if np.issubdtype(bev_arr.dtype, np.floating):
            bev_arr = np.clip(bev_arr, 0.0, 1.0) * 255.0
        frame["bev"] = bev_arr.astype(np.uint8, copy=False)

    return frame


def save_episode_hdf5(
    frames: Sequence[Mapping[str, np.ndarray]],
    output_dir: Union[str, Path],
    ep_id: int,
    episode_result: str,
    instruction: str = "Random factory navigation target",
    metadata: Optional[Mapping[str, Any]] = None,
) -> int:
    if not frames:
        return ep_id

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    ep_id = find_next_available_episode_id(output, ep_id)
    hdf5_path = get_episode_hdf5_path(output, ep_id)

    with h5py.File(hdf5_path, "w") as f:
        group = f.create_group("traj_0")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, dtype in (
            ("rgb", np.uint8),
            ("camera_pos", np.float32),
            ("robot_velocity", np.float32),
            ("target_polar", np.float32),
            ("bev", np.uint8),
        ):
            values = [frame[key] for frame in frames if key in frame]
            if not values:
                continue
            if len(values) != len(frames):
                print(
                    f"[Collection] skip h5 dataset {key}: present frames={len(values)}, total={len(frames)}",
                    flush=True,
                )
                continue
            arr = np.stack([value_to_numpy(value, dtype=dtype) for value in values], axis=0)
            group.create_dataset(key, data=arr, compression="lzf", dtype=dtype)

        group.create_dataset("instruction", data=str(instruction), dtype=string_dtype)
        group.create_dataset("episode_result", data=str(episode_result), dtype=string_dtype)
        group.attrs["num_frames"] = int(len(frames))
        if metadata is not None:
            group.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False, default=str)

    print(
        f"[Saved] traj_0 -> {hdf5_path}, Frames: {len(frames)}, "
        f"episode_result: {episode_result}",
        flush=True,
    )
    return ep_id


def reward_for_agent(reward: Any, agent_id: int) -> float:
    if reward is None:
        return 0.0
    arr = np.asarray(value_to_numpy(reward), dtype=np.float32)
    if arr.ndim == 0:
        return float(arr.item())
    if agent_id < arr.shape[0]:
        return float(np.asarray(arr[agent_id]).reshape(-1)[0])
    return float(np.asarray(arr).reshape(-1)[0])


def episode_result_from_info(done_flags: Sequence[bool], info: Mapping[str, Any]) -> str:
    if bool(info.get("collision", False)):
        return "collide"
    if bool(info.get("arrive", False)):
        return "reach_goal"
    if done_flags and all(done_flags):
        return "done"
    return "max_steps"


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------



def sample_action_norm(action_dim: int, rng: random.Random) -> np.ndarray:
    if action_dim != 2:
        return np.asarray([rng.uniform(-1.0, 1.0) for _ in range(action_dim)], dtype=np.float32)
    return np.asarray([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)], dtype=np.float32)



def norm_action_to_real(action_norm: np.ndarray, cfg: Any, apply_min_linear_vel: bool) -> np.ndarray:
    if action_norm.size < 2:
        raise ValueError("action_dim must be at least 2 for [v,w]")
    v = ((float(action_norm[0]) + 1.0) * 0.5) * float(getattr(cfg, "max_linear_vel", 1.0))
    if apply_min_linear_vel:
        v = max(v, float(getattr(cfg, "min_linear_vel", 0.0)))
    w = float(action_norm[1]) * float(getattr(cfg, "max_angular_vel", 1.0))
    return np.asarray([v, w], dtype=np.float32)


def real_action_to_norm(action_real: np.ndarray, cfg: Any) -> np.ndarray:
    if action_real.size < 2:
        raise ValueError("action_dim must be at least 2 for [v,w]")
    max_v = max(float(getattr(cfg, "max_linear_vel", 1.0)), 1e-6)
    max_w = max(float(getattr(cfg, "max_angular_vel", 1.0)), 1e-6)
    v_norm = clip_float((float(action_real[0]) / max_v) * 2.0 - 1.0, -1.0, 1.0)
    w_norm = clip_float(float(action_real[1]) / max_w, -1.0, 1.0)
    return np.asarray([v_norm, w_norm], dtype=np.float32)



def action_for_env(action_norm_all: np.ndarray, action_real_all: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "norm":
        return action_norm_all
    if fmt == "real":
        return action_real_all
    raise ValueError(f"unknown --action-format={fmt}")


# -----------------------------------------------------------------------------
# Main collection loop
# -----------------------------------------------------------------------------



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect z latent dataset on Isaac using provided 256x128 map input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", required=True, help="Directory containing exp_config.py and tdmpc2.py, usually navigation_research/scripts")
    parser.add_argument("--model-path", required=True, help="Path to saved TDMPC2 checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for .pt chunks and manifest.pt")
    parser.add_argument("--gpu", type=int, default=None, help="Override cfg.gpu")

    parser.add_argument("--isaac-env-module", required=True, help="Python module containing the Isaac env adapter")
    parser.add_argument("--isaac-env-class", required=True, help="Class name of the Isaac env adapter")
    parser.add_argument("--isaac-env-kwargs-json", default="{}", help="JSON kwargs passed to Isaac env constructor")

    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Deprecated unless --stop-on-max-steps is set; defaults to cfg.episode_length.",
    )
    parser.add_argument(
        "--stop-on-max-steps",
        action="store_true",
        help="End an episode at --episode-steps. By default episodes end only on reach_goal or collide.",
    )
    parser.add_argument(
        "--reset-each-episode",
        action="store_true",
        help="Reset Isaac at every logical episode. By default Isaac is reset once and collection continues.",
    )
    parser.add_argument("--num-agents", type=int, default=None, help="Override cfg.num_agent for splitting Isaac observations")
    parser.add_argument("--seed", type=int, default=None)

    # Map options.
    parser.add_argument("--map-key", default="laser_map")
    parser.add_argument("--combine-map-key", default="combine_map")
    parser.add_argument("--temporal-map-key", default="temporal_map")
    parser.add_argument("--map-input-mode", choices=["auto", "combine", "temporal", "dual"], default="auto")
    parser.add_argument("--combine-from-temporal", choices=["last", "first", "max", "mean"], default="last")
    parser.add_argument("--map-normalize", choices=["auto", "none", "zero_one", "minus_one_one"], default="auto")
    parser.add_argument("--map-invert", action="store_true", help="Invert map values after normalization")

    # Component keys / formats.
    parser.add_argument("--velocity-key", default="velocity")
    parser.add_argument("--velocity-format", choices=["real", "model_norm", "action_norm"], default="real")
    parser.add_argument("--velocity-fallback", choices=["error", "zeros", "last_action"], default="last_action")
    parser.add_argument("--goal-rel-key", default="goal_rel")
    parser.add_argument("--goal-rel-format", choices=["auto", "polar", "xy", "model_norm"], default="auto")
    parser.add_argument("--pose-key", default="pose")
    parser.add_argument("--goal-key", default="goal")
    parser.add_argument("--missing-goal", choices=["error", "zeros"], default="error")
    parser.add_argument("--path-key", default="path")
    parser.add_argument("--missing-path", choices=["error", "zeros"], default="zeros")
    parser.add_argument("--progress-key", default="progress")
    parser.add_argument("--missing-progress", choices=["error", "zeros"], default="zeros")

    # Action / output.
    parser.add_argument(
        "--action-source",
        choices=["route", "random"],
        default="route",
        help="Use route-following actions by default, matching find_object.py-style navigation.",
    )
    parser.add_argument("--action-format", choices=["real", "norm"], default="real", help="Action format sent into Isaac env.step")
    parser.add_argument("--apply-min-linear-vel", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50000, help="0 means keep one final chunk only")
    parser.add_argument("--no-store-map", action="store_true")
    parser.add_argument("--store-temporal-map", action="store_true", help="Also store [T,H,W] temporal maps; can be large")
    parser.add_argument("--print-every", type=int, default=1000)

    AppLauncher.add_app_launcher_args(parser)
    args, forwarded_tdmpc_args = parser.parse_known_args()
    isaac_env_kwargs = parse_isaac_env_kwargs(args.isaac_env_kwargs_json)
    isaac_project_root = ensure_isaac_project_paths()
    simulation_app = launch_isaac_app(args, isaac_env_kwargs)
    isaac_env_kwargs.setdefault("launch_app", False)
    isaac_env_kwargs.setdefault("auto_reset", False)

    cfg, scripts_dir = load_tdmpc_cfg(args.project_root, forwarded_tdmpc_args, args.gpu)
    ensure_isaac_project_paths()
    if args.seed is None:
        args.seed = int(getattr(cfg, "seed", 0))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    num_agents = int(args.num_agents if args.num_agents is not None else getattr(cfg, "num_agent", 1))
    episode_steps = int(args.episode_steps if args.episode_steps is not None else getattr(cfg, "episode_length", 100))

    agent = load_tdmpc2_agent(cfg, args.model_path)
    obs_builder = ObservationBuilder(cfg, args)
    encoder = DirectMapZEncoder(agent, obs_builder)
    # PYL: import isaac env
    env = import_isaac_env(args.isaac_env_module, args.isaac_env_class, isaac_env_kwargs)
    rng = random.Random(args.seed)
    writer = ChunkedDatasetWriter(args.output, args.chunk_size, store_map=not args.no_store_map)
    collection_episode_id = find_next_available_episode_id(args.output, 0)

    print("[collector] scripts_dir:", scripts_dir)
    print("[collector] isaac_project_root:", isaac_project_root)
    print("[collector] model_path:", args.model_path)
    print("[collector] output:", Path(args.output).expanduser().resolve())
    print("[collector] img_height,img_width:", int(getattr(cfg, "img_height", 128)), int(getattr(cfg, "img_width", 256)))
    print("[collector] sample_length:", int(getattr(cfg, "sample_length", 8)))
    print("[collector] reset_each_episode:", bool(args.reset_each_episode))
    print("[collector] encoder mode:", {
        "mlp_obs": bool(getattr(cfg, "mlp_obs", False)),
        "use_hybrid_attn_encoder": bool(getattr(cfg, "use_hybrid_attn_encoder", False)),
        "use_dual_branch_encoder": bool(getattr(cfg, "use_dual_branch_encoder", False)),
        "image_encoder_backbone": getattr(cfg, "image_encoder_backbone", "cnn"),
        "vit_input_mode": getattr(cfg, "vit_input_mode", "temporal"),
        "merge_vis": bool(getattr(cfg, "merge_vis", True)),
    })

    total_start = time.time()
    if hasattr(env, "observe"):
        obs0 = env.observe()
    elif hasattr(env, "_get_obs_all"):
        obs0 = env._get_obs_all()
    else:
        raise AttributeError("Isaac env adapter must provide observe() for non-reset startup collection")
    agent_obs_list = to_agent_obs_list(obs0, num_agents=num_agents)
    if len(agent_obs_list) != num_agents:
        num_agents = len(agent_obs_list)
    last_action_real: Dict[int, np.ndarray] = {
        i: np.zeros(2, dtype=np.float32) for i in range(num_agents)
    }

    def make_manifest_meta() -> Dict[str, Any]:
        return {
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(Path(args.project_root).expanduser().resolve()),
            "scripts_dir": str(scripts_dir),
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "num_episodes": args.num_episodes,
            "episode_steps": episode_steps,
            "stop_on_max_steps": bool(args.stop_on_max_steps),
            "num_agents": num_agents,
            "reset_each_episode": bool(args.reset_each_episode),
            "action_source": args.action_source,
            "episode_hdf5": {
                "enabled": True,
                "filename_pattern": "episode_{id}.h5",
                "next_episode_id": int(collection_episode_id),
            },
            "cfg_subset": {
                "img_height": int(getattr(cfg, "img_height", 128)),
                "img_width": int(getattr(cfg, "img_width", 256)),
                "sample_length": int(getattr(cfg, "sample_length", 8)),
                "action_dim": int(getattr(cfg, "action_dim", 2)),
                "latent_dim": int(getattr(cfg, "latent_dim", -1)),
                "enc_num": int(getattr(cfg, "enc_num", 1)),
                "normalization_state": bool(getattr(cfg, "normalization_state", True)),
                "use_xy_goal": bool(getattr(cfg, "use_xy_goal", False)),
                "mlp_obs": bool(getattr(cfg, "mlp_obs", False)),
                "use_hybrid_attn_encoder": bool(getattr(cfg, "use_hybrid_attn_encoder", False)),
                "use_dual_branch_encoder": bool(getattr(cfg, "use_dual_branch_encoder", False)),
                "image_encoder_backbone": getattr(cfg, "image_encoder_backbone", "cnn"),
                "vit_input_mode": getattr(cfg, "vit_input_mode", "temporal"),
            },
            "cli_args": vars(args),
            "forwarded_tdmpc_args": list(forwarded_tdmpc_args),
        }

    for ep in range(args.num_episodes):
        obs_builder.reset()
        episode_frames: List[Dict[str, np.ndarray]] = []
        episode_result = "max_steps"

        # PYL: reset Isaac only when explicitly requested.  The default keeps
        # collection continuous, matching find_object.py's route-switching flow.
        if args.reset_each_episode and ep > 0:
            try:
                obs0 = unpack_reset(env.reset(seed=args.seed + ep))
            except TypeError:
                obs0 = unpack_reset(env.reset())
            agent_obs_list = to_agent_obs_list(obs0, num_agents=num_agents)
            if len(agent_obs_list) != num_agents:
                num_agents = len(agent_obs_list)
        last_action_real = {
            i: np.zeros(2, dtype=np.float32) for i in range(num_agents)
        }

        step = 0
        while True:
            # Encode the current observation for every agent before applying random action.
            action_norm_all = []
            action_real_all = []
            done_flags: List[bool] = []
            pending_samples: List[Dict[str, Any]] = []

            for agent_id, agent_obs in enumerate(agent_obs_list):
                built = obs_builder.build(agent_id, agent_obs, last_action_real.get(agent_id))
                if agent_id == 0:
                    episode_frame = build_find_object_episode_frame(agent_obs)
                    if episode_frame:
                        episode_frames.append(episode_frame)
                z = encoder.encode(built)
                if args.action_source == "route" and hasattr(env, "compute_route_action"):
                    a_real = np.asarray(env.compute_route_action(), dtype=np.float32)
                    a_norm = real_action_to_norm(a_real, cfg)
                else:
                    a_norm = sample_action_norm(int(getattr(cfg, "action_dim", 2)), rng)
                    a_real = norm_action_to_real(a_norm, cfg, args.apply_min_linear_vel)
                action_norm_all.append(a_norm)
                action_real_all.append(a_real)

                sample: Dict[str, Any] = {
                    "z": z.to(torch.float32),
                    "ego": torch.from_numpy(built.ego.astype(np.float32)),
                    "action_norm": torch.from_numpy(a_norm.astype(np.float32)),
                    "action_real": torch.from_numpy(a_real.astype(np.float32)),
                    "episode": int(ep),
                    "step": int(step),
                    "agent": int(agent_id),
                }
                if built.goal is not None:
                    sample["goal"] = torch.from_numpy(built.goal.astype(np.float32))
                if built.path is not None:
                    sample["path"] = torch.from_numpy(built.path.astype(np.float32))
                if built.progress is not None:
                    sample["progress"] = torch.from_numpy(built.progress.astype(np.float32))
                if not args.no_store_map:
                    sample["map_combine"] = torch.from_numpy(built.maps.combine.astype(np.float32))
                if args.store_temporal_map:
                    sample["map_temporal"] = torch.from_numpy(built.maps.temporal.astype(np.float32))
                pending_samples.append(sample)

            action_norm_np = np.stack(action_norm_all, axis=0).astype(np.float32)
            action_real_np = np.stack(action_real_all, axis=0).astype(np.float32)
            env_action = action_for_env(action_norm_np, action_real_np, args.action_format)

            # PYL: step isaac env
            ret = env.step(env_action)
            next_obs, reward, done, info = unpack_step(ret)
            agent_obs_list = to_agent_obs_list(next_obs, num_agents=num_agents)
            if len(agent_obs_list) != num_agents:
                num_agents = len(agent_obs_list)
            for i in range(min(num_agents, len(action_real_all))):
                last_action_real[i] = action_real_all[i]

            done_flags = [done_for_agent(done, i, agent_obs_list[i]) for i in range(len(agent_obs_list))]
            for agent_id, sample in enumerate(pending_samples):
                sample["reward"] = reward_for_agent(reward, agent_id)
                sample["done"] = bool(done_flags[agent_id]) if agent_id < len(done_flags) else False
                if isinstance(info, Mapping):
                    sample["collision"] = bool(info.get("collision", False))
                    sample["arrive"] = bool(info.get("arrive", False))
                    sample["target_index"] = int(info.get("target_index", -1))
                    sample["layout_seed"] = int(info.get("layout_seed", -1))
                writer.add(sample)
            if writer.total > 0 and args.print_every > 0 and writer.total % args.print_every == 0:
                elapsed = time.time() - total_start
                print(f"[collector] samples={writer.total} ep={ep} step={step} elapsed={elapsed:.1f}s")
            if done_flags and all(done_flags):
                episode_result = episode_result_from_info(done_flags, info if isinstance(info, Mapping) else {})
                break
            step += 1
            if args.stop_on_max_steps and episode_steps > 0 and step >= episode_steps:
                episode_result = "max_steps"
                break

        if episode_frames:
            saved_episode_id = save_episode_hdf5(
                frames=episode_frames,
                output_dir=args.output,
                ep_id=collection_episode_id,
                episode_result=episode_result,
                instruction="Random factory navigation target",
                metadata={
                    "logical_episode": int(ep),
                    "episode_steps_limit": int(episode_steps),
                    "stop_on_max_steps": bool(args.stop_on_max_steps),
                    "reset_each_episode": bool(args.reset_each_episode),
                },
            )
            collection_episode_id = find_next_available_episode_id(args.output, saved_episode_id + 1)
            writer.flush()
            writer.write_manifest(make_manifest_meta())

    manifest = writer.finish(make_manifest_meta())
    print(f"[collector] finished: total_samples={writer.total}, manifest={manifest}")
    if hasattr(env, "close"):
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
