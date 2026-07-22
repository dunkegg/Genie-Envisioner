#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import torch


def find_z(obj: Any, key: str) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        if key in obj:
            return find_z(obj[key], key)
        if "z" in obj:
            return find_z(obj["z"], key)
        raise KeyError(f"Cannot find latent key '{key}' or 'z'. Available keys: {list(obj.keys())}")
    if hasattr(obj, key):
        return find_z(getattr(obj, key), key)
    if hasattr(obj, "z"):
        return find_z(getattr(obj, "z"), key)
    raise TypeError(f"Cannot extract z tensor from object type {type(obj).__name__}")


def h5_frame_count(path: Path) -> int:
    with h5py.File(path, "r") as f:
        groups = [f[key] for key in sorted(f.keys()) if isinstance(f[key], h5py.Group)]
        if not groups:
            if "camera_pos" not in f:
                raise KeyError(f"{path} has no camera_pos dataset")
            return int(f["camera_pos"].shape[0])

        total = 0
        for group in groups:
            if "camera_pos" not in group:
                raise KeyError(f"{path}:{group.name} has no camera_pos dataset")
            total += int(group["camera_pos"].shape[0])
        return total


def normalize_z(z: torch.Tensor, frames: int, latent_dim: int, source: Path) -> torch.Tensor:
    z = torch.as_tensor(z, dtype=torch.float32).detach().cpu()
    while z.ndim > 2 and 1 in z.shape:
        z = z.squeeze(next(i for i, size in enumerate(z.shape) if size == 1))
    if z.ndim != 2:
        raise ValueError(f"{source}: expected z shape [N, {latent_dim}], got {tuple(z.shape)}")
    if z.shape[0] != frames:
        raise ValueError(f"{source}: z frame count {z.shape[0]} != h5 frame count {frames}")
    if z.shape[1] != latent_dim:
        raise ValueError(f"{source}: z dim {z.shape[1]} != expected {latent_dim}")
    if z.numel() == 0:
        raise ValueError(f"{source}: z is empty")
    return z.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite one converted episode's per-frame TD-MPC z latents.")
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--out_episode_dir", required=True, type=Path)
    parser.add_argument("--latent_key", default="z")
    parser.add_argument("--latent_dim", type=int, default=512)
    args = parser.parse_args()

    frames = h5_frame_count(args.h5)
    obj = torch.load(args.pt, map_location="cpu")
    z = normalize_z(find_z(obj, args.latent_key), frames, args.latent_dim, args.pt)

    args.out_episode_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(frames):
        frame = z[frame_idx].clone()
        path = args.out_episode_dir / f"frame_{frame_idx:06d}.pt"
        torch.save({"z": frame}, path)
        saved = torch.load(path, map_location="cpu")
        if not isinstance(saved, dict) or "z" not in saved:
            raise RuntimeError(f"Verification failed for {path}: expected dict with key 'z'")
        saved = saved["z"]
        saved = torch.as_tensor(saved, dtype=torch.float32)
        if tuple(saved.shape) != (args.latent_dim,) or not torch.equal(saved, frame):
            raise RuntimeError(f"Verification failed for {path}: saved shape {tuple(saved.shape)}")

    print(
        f"rewrote {frames} frames from {args.pt.name} to {args.out_episode_dir} "
        f"with per-frame latent shape ({args.latent_dim},)"
    )


if __name__ == "__main__":
    main()
