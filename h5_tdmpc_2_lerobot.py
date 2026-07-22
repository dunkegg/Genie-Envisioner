#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


STATE_NAMES_8D = [
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
    "body_height",
    "yaw_position",
    "linear_velocity_x",
    "linear_velocity_y",
    "angular_velocity_yaw",
]

STATE_NAMES_3D = [
    "linear_velocity_x",
    "linear_velocity_y",
    "angular_velocity_yaw",
]


@dataclass
class ConvertConfig:
    input_dir: str = "dataset_tdmpc"
    output_dir: str = "lerobot_data/tdmpc_za"
    domain_name: str = "tdmpc_za"
    fps: int = 30
    chunk_size: int = 1000
    state_mode: str = "8d"
    use_velocity_as_action: bool = True
    robot_type: str = "humanoid_nav"
    latent_key: str = "z"
    latent_column: str = "latent"
    beta_column: str = "beta"
    bev_column: str = "bev_map"
    bev_key: str = "bev"
    goal_key: str = "goal"
    verify_latents: bool = True
    latent_dim: int = 512
    strict_latent_length: bool = True
    skip_missing_pt: bool = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def natural_episode_id(path: str | Path) -> int:
    match = re.search(r"episode_(\d+)\.h5$", str(path))
    if match is None:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def summarize_array(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]

    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def summarize_for_normalizer(values: np.ndarray) -> dict[str, Any]:
    stats = summarize_array(values)
    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "q01": stats["q01"],
        "q99": stats["q99"],
    }


def read_h5_episodes(path: str | Path, bev_key: str = "bev") -> list[dict[str, Any]]:
    episodes = []
    with h5py.File(path, "r") as f:
        groups = [f[key] for key in sorted(f.keys()) if isinstance(f[key], h5py.Group)]

        if not groups and "camera_pos" in f:
            groups = [f]

        for group in groups:
            camera_pos = group["camera_pos"][:]
            if camera_pos.ndim != 2:
                raise ValueError(f"Expected camera_pos with shape [T, C], got {camera_pos.shape} in {path}")

            if camera_pos.shape[1] >= 8:
                raw = {
                    "torso_r": camera_pos[:, 0],
                    "torso_p": camera_pos[:, 1],
                    "torso_y": camera_pos[:, 2],
                    "hb": camera_pos[:, 3],
                    "vx": camera_pos[:, 4],
                    "vy": camera_pos[:, 5],
                    "vyaw": camera_pos[:, 6],
                    "pyaw": camera_pos[:, 7],
                }
            elif camera_pos.shape[1] >= 3:
                raw = {
                    "torso_r": np.zeros(camera_pos.shape[0], dtype=np.float32),
                    "torso_p": np.zeros(camera_pos.shape[0], dtype=np.float32),
                    "torso_y": camera_pos[:, 2],
                    "hb": np.zeros(camera_pos.shape[0], dtype=np.float32),
                    "vx": camera_pos[:, 0],
                    "vy": camera_pos[:, 1],
                    "vyaw": camera_pos[:, 2],
                    "pyaw": camera_pos[:, 2],
                }
            else:
                raise ValueError(
                    f"Unsupported camera_pos shape {camera_pos.shape} in {path}. "
                    "Expected at least 3 columns for --state_mode 3d."
                )

            if "instruction" in group:
                instruction = group["instruction"][()]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                raw["instruction"] = str(instruction)

            if "rgb" in group:
                raw["images"] = group["rgb"][:]
            elif "images" in group:
                raw["images"] = group["images"][:]
            if bev_key in group:
                raw[bev_key] = group[bev_key][:]

            episodes.append(raw)

    return episodes


def load_tdmpc_object(path: str | Path) -> Any:
    obj = torch.load(path, map_location="cpu")
    return obj


def find_latent_value(obj: Any, latent_key: str, path: str | Path) -> Any:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in (latent_key, "z", "latent", "latents", "za"):
            if key in obj:
                return find_latent_value(obj[key], latent_key, path)
        tensor_items = [(key, value) for key, value in obj.items() if torch.is_tensor(value)]
        if len(tensor_items) == 1:
            return tensor_items[0][1]
        raise KeyError(f"No latent key found in {path}. Available keys: {list(obj.keys())}")
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return find_latent_value(obj[0], latent_key, path)
    if hasattr(obj, latent_key):
        return find_latent_value(getattr(obj, latent_key), latent_key, path)
    return obj


def load_tdmpc_z(obj: Any, path: str | Path, latent_key: str) -> torch.Tensor:
    z = find_latent_value(obj, latent_key, path)
    z = torch.as_tensor(z, dtype=torch.float32)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.numel() == 0:
        raise ValueError(f"Empty latent tensor in {path}")
    return z


def get_tdmpc_tensor(obj: Any, key: str) -> torch.Tensor | None:
    if isinstance(obj, dict):
        value = obj.get(key, None)
    else:
        value = getattr(obj, key, None)
    if value is None:
        return None
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value


def align_latents(z: torch.Tensor, length: int, source: str | Path) -> torch.Tensor:
    if z.shape[0] == length:
        return z
    if z.shape[0] == 1:
        return z.repeat(length, *([1] * (z.ndim - 1)))
    if z.shape[0] > length:
        print(f"[WARN] {source}: z length {z.shape[0]} > episode length {length}; clipping.")
        return z[:length]

    print(f"[WARN] {source}: z length {z.shape[0]} < episode length {length}; padding last latent.")
    pad = z[-1:].repeat(length - z.shape[0], *([1] * (z.ndim - 1)))
    return torch.cat([z, pad], dim=0)


def _squeeze_non_time_singletons(z: torch.Tensor) -> torch.Tensor:
    squeeze_dims = [dim for dim in range(1, z.ndim) if z.shape[dim] == 1]
    for dim in reversed(squeeze_dims):
        z = z.squeeze(dim)
    return z


def normalize_latents_for_length(z: torch.Tensor, length: int, source: str | Path, cfg: ConvertConfig) -> torch.Tensor:
    if z.ndim == 0:
        raise ValueError(f"{source}: latent tensor is scalar, expected time-major z.")

    matching_dims = [dim for dim, size in enumerate(z.shape) if size == length]
    if matching_dims:
        time_dim = 0 if 0 in matching_dims else matching_dims[0]
        z = torch.movedim(z, time_dim, 0)
        z = _squeeze_non_time_singletons(z)
    else:
        z = _squeeze_non_time_singletons(z)
        if cfg.strict_latent_length:
            raise ValueError(
                f"{source}: latent z shape {tuple(z.shape)} does not contain h5 frame length {length}. "
                "Expected z to be [num_frames, latent_dim], e.g. [N, 512]."
            )
        z = align_latents(z, length, source)

    if z.shape[0] != length:
        if cfg.strict_latent_length:
            raise ValueError(f"{source}: normalized z length {z.shape[0]} != h5 frame length {length}.")
        z = align_latents(z, length, source)
    if z.ndim != 2:
        raise ValueError(
            f"{source}: normalized z shape {tuple(z.shape)} is not [num_frames, latent_dim]. "
            "For this TD-MPC dataset, chunk_*.pt['z'] should be [N, 512]."
        )
    if cfg.latent_dim and z.shape[1] != cfg.latent_dim:
        raise ValueError(
            f"{source}: normalized z dim {z.shape[1]} != expected latent_dim {cfg.latent_dim}. "
            f"Got shape {tuple(z.shape)}."
        )
    if z.numel() == 0:
        raise ValueError(f"{source}: normalized latent z is empty.")
    return z.contiguous()


def split_tdmpc_z_by_episodes(
    z_all: torch.Tensor,
    raw_episodes: list[dict[str, Any]],
    pt_path: str | Path,
    cfg: ConvertConfig,
) -> tuple[list[torch.Tensor], list[int]]:
    lengths = []
    for raw in raw_episodes:
        state, _, _ = build_state_action(raw, cfg)
        lengths.append(len(state))

    if len(raw_episodes) == 1:
        return [normalize_latents_for_length(z_all, lengths[0], pt_path, cfg)], lengths

    if z_all.ndim >= 2 and z_all.shape[0] == len(raw_episodes):
        return [
            normalize_latents_for_length(z_all[i], lengths[i], pt_path, cfg)
            for i in range(len(raw_episodes))
        ], lengths

    total_length = sum(lengths)
    matching_dims = [dim for dim, size in enumerate(z_all.shape) if size == total_length]
    if matching_dims:
        time_dim = 0 if 0 in matching_dims else matching_dims[0]
        z_time = torch.movedim(z_all, time_dim, 0)
        z_time = _squeeze_non_time_singletons(z_time)
        z_slices = list(torch.split(z_time, lengths, dim=0))
        return [normalize_latents_for_length(z, lengths[i], pt_path, cfg) for i, z in enumerate(z_slices)], lengths

    raise ValueError(
        f"Cannot map {pt_path} z shape {tuple(z_all.shape)} to {len(raw_episodes)} "
        f"h5 groups with frame lengths {lengths}. Expected time dim {total_length} "
        f"or episode dim {len(raw_episodes)}."
    )


def build_state_action(raw: dict[str, Any], cfg: ConvertConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if cfg.state_mode == "8d":
        state = np.stack(
            [
                raw["torso_r"],
                raw["torso_p"],
                raw["torso_y"],
                raw["hb"],
                raw["pyaw"],
                raw["vx"],
                raw["vy"],
                raw["vyaw"],
            ],
            axis=1,
        ).astype(np.float32)
        names = STATE_NAMES_8D
    elif cfg.state_mode == "3d":
        state = np.stack(
            [
                raw["vx"],
                raw["vy"],
                raw["vyaw"],
            ],
            axis=1,
        ).astype(np.float32)
        names = STATE_NAMES_3D
    else:
        raise NotImplementedError(f"unsupported state_mode: {cfg.state_mode}")

    if cfg.use_velocity_as_action:
        actions = state.copy()
    else:
        actions = state[1:] - state[:-1]
        state = state[:-1]

    return state.astype(np.float32), actions.astype(np.float32), names


def align_array(values: np.ndarray, length: int, source: str | Path, name: str) -> np.ndarray:
    if values.shape[0] == length:
        return values
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0)
    if values.shape[0] > length:
        print(f"[WARN] {source}: {name} length {values.shape[0]} > episode length {length}; clipping.")
        return values[:length]

    print(f"[WARN] {source}: {name} length {values.shape[0]} < episode length {length}; padding last value.")
    pad = np.repeat(values[-1:], length - values.shape[0], axis=0)
    return np.concatenate([values, pad], axis=0)


def goal_to_polar(goal: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal, dtype=np.float32)
    if goal.ndim == 1:
        goal = goal[:, None]
    if goal.shape[1] >= 2:
        gx = goal[:, 0]
        gy = goal[:, 1]
        radius = np.sqrt(gx * gx + gy * gy)
        theta = np.arctan2(gy, gx)
        return np.stack([radius, theta], axis=1).astype(np.float32)
    if goal.shape[1] == 1:
        theta = np.zeros_like(goal[:, 0])
        return np.stack([goal[:, 0], theta], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported goal shape for beta polar conversion: {goal.shape}")


def build_beta(state: np.ndarray, tdmpc_obj: Any, cfg: ConvertConfig, source: str | Path) -> np.ndarray:
    velocity = state.astype(np.float32)
    goal_tensor = get_tdmpc_tensor(tdmpc_obj, cfg.goal_key)
    if goal_tensor is None:
        print(f"[WARN] {source}: no '{cfg.goal_key}' in pt; beta will contain velocity only.")
        goal_polar = np.zeros((len(state), 2), dtype=np.float32)
        return np.concatenate([velocity, goal_polar], axis=1).astype(np.float32)

    goal = align_array(goal_tensor.detach().cpu().numpy().astype(np.float32), len(state), source, cfg.goal_key)
    goal_polar = goal_to_polar(goal)
    return np.concatenate([velocity, goal_polar], axis=1).astype(np.float32)


def write_video(frames: np.ndarray, path: str | Path, fps: int) -> tuple[int, int]:
    import cv2

    if len(frames) == 0:
        raise ValueError("Cannot write empty video")

    ensure_dir(Path(path).parent)
    first = frames[0]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame in frames:
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")
        writer.write(frame)

    writer.release()
    return int(height), int(width)


def write_latents(z: torch.Tensor, ep_idx: int, cfg: ConvertConfig) -> list[str]:
    chunk_id = ep_idx // cfg.chunk_size
    latent_dir = Path(cfg.output_dir) / "latents" / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}"
    ensure_dir(latent_dir)
    for stale_path in latent_dir.glob("frame_*.pt"):
        stale_path.unlink()

    rel_paths = []
    for frame_idx in range(z.shape[0]):
        rel_path = Path("latents") / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}" / f"frame_{frame_idx:06d}.pt"
        expected = z[frame_idx].detach().cpu().float()
        if expected.numel() == 0:
            raise ValueError(f"Empty latent frame for episode {ep_idx}, frame {frame_idx}")
        if cfg.latent_dim and tuple(expected.shape) != (cfg.latent_dim,):
            raise ValueError(
                f"Unexpected latent frame shape for episode {ep_idx}, frame {frame_idx}: "
                f"{tuple(expected.shape)}; expected ({cfg.latent_dim},)"
            )
        save_path = Path(cfg.output_dir) / rel_path
        torch.save({"z": expected}, save_path)
        if cfg.verify_latents:
            saved = torch.load(save_path, map_location="cpu")
            if isinstance(saved, dict):
                saved = saved.get("z")
            saved = torch.as_tensor(saved, dtype=torch.float32)
            if tuple(saved.shape) != tuple(expected.shape) or not torch.equal(saved, expected):
                max_diff = None
                if tuple(saved.shape) == tuple(expected.shape):
                    max_diff = float((saved - expected).abs().max().item())
                raise RuntimeError(
                    f"Latent verification failed for {save_path}: "
                    f"saved shape={tuple(saved.shape)}, expected shape={tuple(expected.shape)}, max_diff={max_diff}"
                )
        rel_paths.append(rel_path.as_posix())
    return rel_paths


def write_bev_maps(bev: np.ndarray, ep_idx: int, cfg: ConvertConfig) -> list[str]:
    """Save each BEV frame losslessly and return paths relative to the dataset root."""
    if bev.ndim != 3:
        raise ValueError(f"Expected BEV [T,H,W], got {bev.shape} for episode {ep_idx}")
    chunk_id = ep_idx // cfg.chunk_size
    bev_dir = Path(cfg.output_dir) / "bev" / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}"
    ensure_dir(bev_dir)
    rel_paths = []
    for frame_idx, frame in enumerate(bev):
        rel_path = Path("bev") / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}" / f"frame_{frame_idx:06d}.npy"
        np.save(Path(cfg.output_dir) / rel_path, frame, allow_pickle=False)
        rel_paths.append(rel_path.as_posix())
    return rel_paths


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_dataset(cfg: ConvertConfig) -> None:
    input_dir = Path(cfg.input_dir)
    h5_files = sorted(glob.glob(str(input_dir / "episode_*.h5")), key=natural_episode_id)
    if not h5_files:
        raise FileNotFoundError(f"No episode_*.h5 found in {input_dir}")

    ensure_dir(cfg.output_dir)
    ensure_dir(Path(cfg.output_dir) / "meta")

    task_to_idx: dict[str, int] = {}
    tasks: list[dict[str, Any]] = []
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    video_hw = None
    state_names = STATE_NAMES_8D if cfg.state_mode == "8d" else STATE_NAMES_3D

    global_ep_idx = 0
    for h5_path in tqdm(h5_files, desc="Converting h5+tdmpc"):
        h5_path = Path(h5_path)
        source_ep_id = natural_episode_id(h5_path)
        pt_path = input_dir / f"chunk_{source_ep_id:06d}.pt"
        if not pt_path.exists():
            if cfg.skip_missing_pt:
                print(f"[WARN] Skip {h5_path}: missing matching TDMPC latent file {pt_path}")
                continue
            raise FileNotFoundError(f"Missing matching TDMPC pt for {h5_path}: {pt_path}")

        raw_episodes = read_h5_episodes(h5_path, bev_key=cfg.bev_key)
        tdmpc_obj = load_tdmpc_object(pt_path)
        z_all = load_tdmpc_z(tdmpc_obj, pt_path, cfg.latent_key)
        print(
            f"[latent] source_h5={h5_path.name} source_pt={pt_path.name} "
            f"z_shape={tuple(z_all.shape)} raw_episodes={len(raw_episodes)}"
        )

        z_slices, lengths = split_tdmpc_z_by_episodes(z_all, raw_episodes, pt_path, cfg)
        print(
            f"[latent] source_pt={pt_path.name} frame_lengths={lengths} "
            f"slice_shapes={[tuple(z.shape) for z in z_slices]}"
        )

        for raw, z in zip(raw_episodes, z_slices):
            ep_idx = global_ep_idx
            global_ep_idx += 1

            state, actions, state_names = build_state_action(raw, cfg)
            length = len(state)
            z = align_latents(z, length, pt_path)
            print(
                f"[latent] output_episode={ep_idx:06d} source_pt={pt_path.name} "
                f"frames={length} aligned_z_shape={tuple(z.shape)} frame0_shape={tuple(z[0].shape)}"
            )
            latent_paths = write_latents(z, ep_idx, cfg)
            beta = build_beta(state, tdmpc_obj, cfg, pt_path)
            if cfg.bev_key not in raw:
                raise KeyError(f"{h5_path} does not contain '{cfg.bev_key}'; BEV supervision is required.")
            bev = align_array(np.asarray(raw[cfg.bev_key]), length, h5_path, cfg.bev_key)
            bev_paths = write_bev_maps(bev, ep_idx, cfg)

            instruction = raw.get("instruction", "unknown task")
            if instruction not in task_to_idx:
                task_index = len(tasks)
                task_to_idx[instruction] = task_index
                tasks.append({"task_index": task_index, "task": instruction})
            task_index = task_to_idx[instruction]

            chunk_id = ep_idx // cfg.chunk_size
            parquet_dir = Path(cfg.output_dir) / "data" / f"chunk-{chunk_id:03d}"
            ensure_dir(parquet_dir)
            parquet_path = parquet_dir / f"episode_{ep_idx:06d}.parquet"
            df = pd.DataFrame(
                {
                    "observation.state": list(state),
                    "actions": list(actions),
                    cfg.latent_column: latent_paths,
                    cfg.beta_column: list(beta),
                    cfg.bev_column: bev_paths,
                    "timestamp": np.arange(length, dtype=np.float32) / cfg.fps,
                    "frame_index": np.arange(length, dtype=np.int64),
                    "episode_index": np.full(length, ep_idx, dtype=np.int64),
                    "source_episode_id": np.full(length, source_ep_id, dtype=np.int64),
                    "source_pt": np.full(length, pt_path.as_posix(), dtype=object),
                    "index": np.arange(length, dtype=np.int64),
                    "task_index": np.full(length, task_index, dtype=np.int64),
                }
            )
            df.to_parquet(parquet_path)

            if "images" not in raw:
                raise KeyError(f"{h5_path} does not contain rgb/images; video is required by CustomLeRobotDataset.")
            video_path = (
                Path(cfg.output_dir)
                / "videos"
                / f"chunk-{chunk_id:03d}"
                / "observation.images.chest"
                / f"episode_{ep_idx:06d}.mp4"
            )
            video_hw = write_video(raw["images"][:length], video_path, cfg.fps)

            all_states.append(state)
            all_actions.append(actions)
            episodes_meta.append(
                {
                    "episode_index": ep_idx,
                    "length": int(length),
                    "tasks": [instruction],
                    "task_index": task_index,
                }
            )
            episodes_stats.append(
                {
                    "episode_index": ep_idx,
                    "stats": {
                        "timestamp": summarize_array(np.arange(length, dtype=np.float32) / cfg.fps),
                        "frame_index": summarize_array(np.arange(length, dtype=np.int64)),
                        "episode_index": summarize_array(np.full(length, ep_idx, dtype=np.int64)),
                        "index": summarize_array(np.arange(length, dtype=np.int64)),
                        "task_index": summarize_array(np.full(length, task_index, dtype=np.int64)),
                        "observation.state": {
                            **summarize_array(state),
                            "dtype": "float32",
                            "shape": [state.shape[1]],
                            "names": state_names,
                        },
                        "actions": {
                            **summarize_array(actions),
                            "dtype": "float32",
                            "shape": [actions.shape[1]],
                            "names": state_names,
                        },
                    },
                }
            )

    if not episodes_meta:
        raise RuntimeError("No episodes were converted")

    all_states_arr = np.concatenate(all_states, axis=0)
    all_actions_arr = np.concatenate(all_actions, axis=0)
    total_frames = int(sum(ep["length"] for ep in episodes_meta))
    height, width = video_hw if video_hw is not None else (640, 720)

    meta_stats = {
        "timestamp": {
            **summarize_array(np.concatenate([np.arange(ep["length"], dtype=np.float32) / cfg.fps for ep in episodes_meta])),
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },
        "frame_index": {
            **summarize_array(np.concatenate([np.arange(ep["length"], dtype=np.int64) for ep in episodes_meta])),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "episode_index": {
            **summarize_array(np.concatenate([np.full(ep["length"], ep["episode_index"], dtype=np.int64) for ep in episodes_meta])),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "index": {
            **summarize_array(np.arange(total_frames, dtype=np.int64)),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "task_index": {
            **summarize_array(np.concatenate([np.full(ep["length"], ep["task_index"], dtype=np.int64) for ep in episodes_meta])),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "observation.state": {
            **summarize_array(all_states_arr),
            "dtype": "float32",
            "shape": [all_states_arr.shape[1]],
            "names": state_names,
        },
        "actions": {
            **summarize_array(all_actions_arr),
            "dtype": "float32",
            "shape": [all_actions_arr.shape[1]],
            "names": state_names,
        },
    }

    info = {
        "codebase_version": "v2.0",
        "robot_type": cfg.robot_type,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": cfg.chunk_size,
        "fps": cfg.fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.chest": {
                "dtype": "video",
                "shape": [3, height, width],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": cfg.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [all_states_arr.shape[1]],
                "names": state_names,
            },
            "actions": {
                "dtype": "float32",
                "shape": [all_actions_arr.shape[1]],
                "names": state_names,
            },
            cfg.latent_column: {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
            cfg.beta_column: {
                "dtype": "float32",
                "shape": [int(beta.shape[1])],
                "names": None,
            },
            cfg.bev_column: {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
        },
        "total_chunks": int(np.ceil(len(episodes_meta) / cfg.chunk_size)),
        "total_videos": len(episodes_meta),
    }

    delta_actions = np.diff(all_actions_arr, axis=0)
    if len(delta_actions) == 0:
        delta_actions = all_actions_arr
    normalizer_stats = {
        f"{cfg.domain_name}_joint": summarize_for_normalizer(all_actions_arr),
        f"{cfg.domain_name}_delta_joint": summarize_for_normalizer(delta_actions),
        f"{cfg.domain_name}_state_joint": summarize_for_normalizer(all_states_arr),
    }

    meta_dir = Path(cfg.output_dir) / "meta"
    write_json(meta_dir / "info.json", info)
    write_json(meta_dir / "stats.json", meta_stats)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats)
    write_jsonl(meta_dir / "tasks.jsonl", tasks)
    write_json(Path(cfg.output_dir) / "statistics.json", normalizer_stats)

    print("Done!")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Domain: {cfg.domain_name}")
    print(f"Episodes: {len(episodes_meta)}")
    print(f"Frames: {total_frames}")
    print(f"Latent column: {cfg.latent_column}")
    print(f"Beta column: {cfg.beta_column}")
    print(f"BEV column: {cfg.bev_column}")


def parse_args() -> ConvertConfig:
    parser = argparse.ArgumentParser(description="Convert h5 + TDMPC z pt files to Genie-Envisioner LeRobot format.")
    parser.add_argument("--input_dir", default="isaac_z_dataset_7_10")
    parser.add_argument("--output_dir", default="lerobot_data/tdmpc_za_7_10")
    parser.add_argument("--domain_name", default="tdmpc_za")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--state_mode", choices=["8d", "3d"], default="8d")
    parser.add_argument("--use_delta_action", action="store_true")
    parser.add_argument("--robot_type", default="humanoid_nav")
    parser.add_argument("--latent_key", default="z")
    parser.add_argument("--latent_column", default="latent")
    parser.add_argument("--beta_column", default="beta")
    parser.add_argument("--bev_column", default="bev_map")
    parser.add_argument("--bev_key", default="bev")
    parser.add_argument("--goal_key", default="goal")
    parser.add_argument("--no_verify_latents", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--allow_latent_length_mismatch", action="store_true")
    parser.add_argument(
        "--skip_missing_pt",
        action="store_true",
        help="Skip episode_*.h5 files without a matching chunk_XXXXXX.pt latent file.",
    )
    args = parser.parse_args()
    return ConvertConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        domain_name=args.domain_name,
        fps=args.fps,
        chunk_size=args.chunk_size,
        state_mode=args.state_mode,
        use_velocity_as_action=not args.use_delta_action,
        robot_type=args.robot_type,
        latent_key=args.latent_key,
        latent_column=args.latent_column,
        beta_column=args.beta_column,
        bev_column=args.bev_column,
        bev_key=args.bev_key,
        goal_key=args.goal_key,
        verify_latents=not args.no_verify_latents,
        latent_dim=args.latent_dim,
        strict_latent_length=not args.allow_latent_length_mismatch,
        skip_missing_pt=args.skip_missing_pt,
    )


def main() -> None:
    convert_dataset(parse_args())


if __name__ == "__main__":
    main()
