#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================
# Config
# ==============================

@dataclass
class ConvertConfig:
    input_glob: str = "dataset/*.h5"
    output_dir: str = "lerobot_data/isaac_lerobot/"
    fps: int = 30
    chunk_size: int = 1000
    use_velocity_as_action: bool = True
    robot_type: str = "humanoid_nav"

STATE_NAMES = [
    "linear_velocity_x",
    "linear_velocity_y",
    # "angular_velocity_yaw",
]

ACTION_NAMES = [
    "linear_velocity_x",
    "linear_velocity_y",
    # "angular_velocity_yaw",
]

# ==============================
# Data structures
# ==============================

@dataclass
class EpisodeData:
    episode_index: int
    task_index: int
    instruction: str
    state: np.ndarray
    actions: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.state))

    def chunk_id(self, chunk_size: int) -> int:
        return self.episode_index // chunk_size


@dataclass
class TaskRegistry:
    task_to_idx: dict[str, int] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def get_or_create(self, instruction: str) -> int:
        if instruction not in self.task_to_idx:
            task_index = len(self.tasks)
            self.task_to_idx[instruction] = task_index
            self.tasks.append({
                "task_index": task_index,
                "task": instruction,
            })
        return self.task_to_idx[instruction]


# ==============================
# Utilities
# ==============================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def print_hdf5_keys(h5_path: str | Path) -> None:
    def _print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")

    with h5py.File(h5_path, "r") as f:
        print(f"\n===== HDF5 Structure: {h5_path} =====")
        f.visititems(_print_item)


# ==============================
# HDF5 loading
# ==============================

def resolve_vx_vy_from_vw(
    robot_velocity: np.ndarray,
    yaw: np.ndarray,
    source: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot_velocity = np.asarray(robot_velocity, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32)

    if robot_velocity.ndim != 2 or robot_velocity.shape[1] < 2:
        raise ValueError(
            f"Expected robot_velocity with shape [T, >=2] for [v, w], "
            f"got {robot_velocity.shape} in {source}"
        )
    if robot_velocity.shape[0] != yaw.shape[0]:
        raise ValueError(
            f"robot_velocity length {robot_velocity.shape[0]} does not match "
            f"camera_pos length {yaw.shape[0]} in {source}"
        )

    v = robot_velocity[:, 0]
    w = robot_velocity[:, 1]
    vx = v * np.cos(yaw)
    vy = v * np.sin(yaw)
    return vx.astype(np.float32), vy.astype(np.float32), w.astype(np.float32)


def load_hdf5(path: str | Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []

    with h5py.File(path, "r") as f:
        for traj_name in sorted(f.keys()):
            group = f[traj_name]

            camera_pos = group["camera_pos"][:]
            robot_velocity = group["robot_velocity"][:]
            pyaw = camera_pos[:, 7]
            vx, vy, vyaw = resolve_vx_vy_from_vw(
                robot_velocity=robot_velocity,
                yaw=pyaw,
                source=f"{path}:{traj_name}/robot_velocity",
            )

            data: dict[str, Any] = {
                "torso_r": camera_pos[:, 0],
                "torso_p": camera_pos[:, 1],
                "torso_y": camera_pos[:, 2],
                "hb": camera_pos[:, 3],
                "vx": vx,
                "vy": vy,
                "vyaw": vyaw,
                "pyaw": pyaw,
            }

            if "instruction" in group:
                instruction = group["instruction"][()]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                data["instruction"] = instruction

            if "rgb" in group:
                data["images"] = group["rgb"][:]

            episodes.append(data)

    return episodes


# ==============================
# State / action building
# ==============================

def build_state_action(
    raw: dict[str, Any],
    use_velocity_as_action: bool,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.stack([
        raw["vx"],
        raw["vy"],
        # raw["vyaw"],
    ], axis=1).astype(np.float32)

    if use_velocity_as_action:
        actions = state.copy()
    else:
        actions = state[1:] - state[:-1]
        state = state[:-1]

    return state.astype(np.float32), actions.astype(np.float32)


def build_episode(
    raw: dict[str, Any],
    episode_index: int,
    task_registry: TaskRegistry,
    cfg: ConvertConfig,
) -> EpisodeData:
    instruction = raw.get("instruction", "unknown task")
    task_index = task_registry.get_or_create(instruction)

    state, actions = build_state_action(
        raw,
        use_velocity_as_action=cfg.use_velocity_as_action,
    )

    return EpisodeData(
        episode_index=episode_index,
        task_index=task_index,
        instruction=instruction,
        state=state,
        actions=actions,
    )


# ==============================
# Video writing
# ==============================

def write_video(frames, path, fps):
    import cv2
    from pathlib import Path

    if len(frames) == 0:
        raise ValueError("Cannot write video: empty frames")

    path = Path(path)
    ensure_dir(path.parent)

    first = frames[0]

    if first.shape[-1] == 4:
        height, width = first.shape[:2]
    elif first.shape[-1] == 3:
        height, width = first.shape[:2]
    else:
        raise ValueError(f"Unsupported frame shape: {first.shape}")

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


def write_episode_video(
    ep: EpisodeData,
    raw: dict[str, Any],
    cfg: ConvertConfig,
) -> None:
    if "images" not in raw:
        return

    chunk_id = ep.chunk_id(cfg.chunk_size)

    video_path = (
        Path(cfg.output_dir)
        / "videos"
        / f"chunk-{chunk_id:03d}"
        / "observation.images.chest"
        / f"episode_{ep.episode_index:06d}.mp4"
    )

    write_video(raw["images"], video_path, cfg.fps)


# ==============================
# Parquet writing
# ==============================

def write_episode_parquet(ep: EpisodeData, cfg: ConvertConfig) -> None:
    chunk_id = ep.chunk_id(cfg.chunk_size)

    parquet_dir = (
        Path(cfg.output_dir)
        / "data"
        / f"chunk-{chunk_id:03d}"
    )
    ensure_dir(parquet_dir)

    length = ep.length

    df = pd.DataFrame(
        {
            "observation.state": list(ep.state),
            "actions": list(ep.actions),
            "timestamp": np.arange(length, dtype=np.float32) / cfg.fps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep.episode_index, dtype=np.int64),
            "index": np.arange(length, dtype=np.int64),
            "task_index": np.full(length, ep.task_index, dtype=np.int64),
        }
    )

    parquet_path = parquet_dir / f"episode_{ep.episode_index:06d}.parquet"
    df.to_parquet(parquet_path)


# ==============================
# Stats
# ==============================

def compute_episode_stats(ep: EpisodeData, cfg: ConvertConfig) -> dict[str, Any]:
    length = ep.length

    timestamp = np.arange(length, dtype=np.float32) / cfg.fps
    frame_index = np.arange(length, dtype=np.int64)
    episode_index_arr = np.full(length, ep.episode_index, dtype=np.int64)
    index_arr = np.arange(length, dtype=np.int64)
    task_index_arr = np.full(length, ep.task_index, dtype=np.int64)

    return {
        "episode_index": ep.episode_index,
        "stats": {
            "timestamp": summarize_array(timestamp),
            "frame_index": summarize_array(frame_index),
            "episode_index": summarize_array(episode_index_arr),
            "index": summarize_array(index_arr),
            "task_index": summarize_array(task_index_arr),
            "observation.state": {
                **summarize_array(ep.state),
                "dtype": "float32",
                "shape": [ep.state.shape[1]],
                "names": STATE_NAMES,
            },
            "actions": {
                **summarize_array(ep.actions),
                "dtype": "float32",
                "shape": [ep.actions.shape[1]],
                "names": ACTION_NAMES,
            },
        },
    }


def build_global_stats(
    episodes: list[EpisodeData],
    cfg: ConvertConfig,
) -> dict[str, Any]:
    all_states = np.concatenate([ep.state for ep in episodes], axis=0)
    all_actions = np.concatenate([ep.actions for ep in episodes], axis=0)

    total_frames = all_states.shape[0]

    timestamp_arr = np.concatenate([
        np.arange(ep.length, dtype=np.float32) / cfg.fps
        for ep in episodes
    ])

    frame_index_arr = np.concatenate([
        np.arange(ep.length, dtype=np.int64)
        for ep in episodes
    ])

    episode_index_arr = np.concatenate([
        np.full(ep.length, ep.episode_index, dtype=np.int64)
        for ep in episodes
    ])

    index_arr = np.arange(total_frames, dtype=np.int64)

    task_index_arr = np.concatenate([
        np.full(ep.length, ep.task_index, dtype=np.int64)
        for ep in episodes
    ])

    return {
        "timestamp": {
            **summarize_array(timestamp_arr),
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },
        "frame_index": {
            **summarize_array(frame_index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "episode_index": {
            **summarize_array(episode_index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "index": {
            **summarize_array(index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "task_index": {
            **summarize_array(task_index_arr),
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "observation.state": {
            **summarize_array(all_states),
            "dtype": "float32",
            "shape": [all_states.shape[1]],
            "names": STATE_NAMES,
        },
        "actions": {
            **summarize_array(all_actions),
            "dtype": "float32",
            "shape": [all_actions.shape[1]],
            "names": ACTION_NAMES,
        },
    }


# ==============================
# Meta
# ==============================

def build_info_json(
    episodes: list[EpisodeData],
    task_registry: TaskRegistry,
    cfg: ConvertConfig,
) -> dict[str, Any]:
    total_frames = sum(ep.length for ep in episodes)
    total_chunks = int(np.ceil(len(episodes) / cfg.chunk_size))

    state_dim = episodes[0].state.shape[1]
    action_dim = episodes[0].actions.shape[1]

    return {
        "codebase_version": "v2.0",
        "robot_type": cfg.robot_type,
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(task_registry.tasks),
        "chunks_size": cfg.chunk_size,
        "fps": cfg.fps,
        "splits": {
            "train": f"0:{len(episodes)}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.chest": {
                "dtype": "video",
                "shape": [3, 640, 720],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": 640,
                    "video.width": 720,
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
                "shape": [state_dim],
                "names": STATE_NAMES,
            },
            "actions": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": ACTION_NAMES,
            },
        },
        "total_chunks": total_chunks,
        "total_videos": len(episodes),
    }


def build_episodes_jsonl(episodes: list[EpisodeData]) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": ep.episode_index,
            "length": ep.length,
            "tasks": [ep.instruction],
            "task_index": ep.task_index,
        }
        for ep in episodes
    ]


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta_files(
    episodes: list[EpisodeData],
    episode_stats: list[dict[str, Any]],
    task_registry: TaskRegistry,
    cfg: ConvertConfig,
) -> None:
    meta_dir = Path(cfg.output_dir) / "meta"
    ensure_dir(meta_dir)

    info = build_info_json(episodes, task_registry, cfg)
    global_stats = build_global_stats(episodes, cfg)
    episodes_meta = build_episodes_jsonl(episodes)

    write_json(meta_dir / "info.json", info)
    write_json(meta_dir / "stats.json", global_stats)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)
    write_jsonl(meta_dir / "tasks.jsonl", task_registry.tasks)


# ==============================
# Main conversion
# ==============================

def init_output_dir(cfg: ConvertConfig) -> None:
    ensure_dir(cfg.output_dir)
    ensure_dir(Path(cfg.output_dir) / "meta")


def convert_dataset(cfg: ConvertConfig) -> None:
    files = sorted(glob.glob(cfg.input_glob))
    print(f"Found {len(files)} HDF5 files")

    if not files:
        raise FileNotFoundError(f"No h5 files matched: {cfg.input_glob}")

    init_output_dir(cfg)

    task_registry = TaskRegistry()

    episodes: list[EpisodeData] = []
    episode_stats: list[dict[str, Any]] = []

    global_episode_index = 0

    for h5_file in files:
        raw_episodes = load_hdf5(h5_file)

        for raw in tqdm(raw_episodes, desc=f"Converting {Path(h5_file).name}"):
            ep = build_episode(
                raw=raw,
                episode_index=global_episode_index,
                task_registry=task_registry,
                cfg=cfg,
            )

            write_episode_parquet(ep, cfg)
            write_episode_video(ep, raw, cfg)

            episodes.append(ep)
            episode_stats.append(compute_episode_stats(ep, cfg))

            global_episode_index += 1
        

    write_meta_files(
        episodes=episodes,
        episode_stats=episode_stats,
        task_registry=task_registry,
        cfg=cfg,
    )

    print("Done!")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Tasks: {len(task_registry.tasks)}")
    print(f"Frames: {sum(ep.length for ep in episodes)}")


def main() -> None:
    cfg = ConvertConfig()
    convert_dataset(cfg)


if __name__ == "__main__":
    main()
