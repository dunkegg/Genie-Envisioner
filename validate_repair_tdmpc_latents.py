#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from h5_tdmpc_2_lerobot import (
    ConvertConfig,
    align_latents,
    build_state_action,
    load_tdmpc_object,
    load_tdmpc_z,
    natural_episode_id,
    read_h5_episodes,
    split_tdmpc_z_by_episodes,
)


def sorted_h5_files(input_dir: Path):
    return sorted(input_dir.glob("episode_*.h5"), key=natural_episode_id)


def load_saved_tensor(path: Path):
    value = torch.load(path, map_location="cpu")
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    if isinstance(value, dict):
        for key in ("z", "latent", "za"):
            if key in value and torch.is_tensor(value[key]):
                return value[key].detach().cpu().float()
        if len(value) == 1:
            only_value = next(iter(value.values()))
            if torch.is_tensor(only_value):
                return only_value.detach().cpu().float()
    raise TypeError(f"{path}: expected tensor-like .pt, got {type(value).__name__}")


def compare_episode(expected_z, output_dir: Path, ep_idx: int, cfg, repair: bool):
    chunk_id = ep_idx // cfg.chunk_size
    episode_dir = output_dir / "latents" / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}"
    result = {
        "episode_index": ep_idx,
        "episode_dir": str(episode_dir),
        "expected_frames": int(expected_z.shape[0]),
        "existing_frames": 0,
        "missing_frames": 0,
        "bad_frames": 0,
        "shape_mismatch": 0,
        "max_abs_diff": 0.0,
        "mse_sum": 0.0,
        "checked_frames": 0,
        "repaired_frames": 0,
    }

    if episode_dir.exists():
        result["existing_frames"] = len(list(episode_dir.glob("frame_*.pt")))
    elif repair:
        episode_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(expected_z.shape[0]):
        expected = expected_z[frame_idx].detach().cpu().float()
        frame_path = episode_dir / f"frame_{frame_idx:06d}.pt"
        needs_repair = False

        if not frame_path.exists():
            result["missing_frames"] += 1
            needs_repair = True
        else:
            try:
                saved = load_saved_tensor(frame_path)
                result["checked_frames"] += 1
                if tuple(saved.shape) != tuple(expected.shape):
                    result["shape_mismatch"] += 1
                    needs_repair = True
                else:
                    diff = saved - expected
                    result["max_abs_diff"] = max(result["max_abs_diff"], float(diff.abs().max().item()))
                    result["mse_sum"] += float(diff.pow(2).mean().item())
                    if not torch.equal(saved, expected):
                        needs_repair = True
            except Exception:
                result["bad_frames"] += 1
                needs_repair = True

        if needs_repair and repair:
            torch.save(expected, frame_path)
            result["repaired_frames"] += 1

    if result["checked_frames"] > 0:
        result["mse_mean"] = result["mse_sum"] / result["checked_frames"]
    else:
        result["mse_mean"] = None
    result["ok"] = (
        result["missing_frames"] == 0
        and result["bad_frames"] == 0
        and result["shape_mismatch"] == 0
        and result["max_abs_diff"] == 0.0
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate or repair converted per-frame TD-MPC latent .pt files.")
    parser.add_argument("--input_dir", default="isaac_z_dataset_7_10")
    parser.add_argument("--output_dir", default="lerobot_data/tdmpc_za_7_10")
    parser.add_argument("--latent_key", default="z")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--state_mode", choices=["8d", "3d"], default="8d")
    parser.add_argument("--use_delta_action", action="store_true")
    parser.add_argument("--episode", type=int, default=None, help="Only validate one converted output episode index.")
    parser.add_argument("--max_episodes", type=int, default=0, help="Limit converted output episodes; 0 means all.")
    parser.add_argument("--repair", action="store_true", help="Overwrite missing/bad/mismatched frame_*.pt from source chunk_*.pt.")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--allow_latent_length_mismatch", action="store_true")
    args = parser.parse_args()

    cfg = ConvertConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        state_mode=args.state_mode,
        use_velocity_as_action=not args.use_delta_action,
        latent_key=args.latent_key,
        latent_dim=args.latent_dim,
        strict_latent_length=not args.allow_latent_length_mismatch,
    )
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    reports = []
    global_ep_idx = 0
    stop = False
    for h5_path in sorted_h5_files(input_dir):
        source_ep_id = natural_episode_id(h5_path)
        pt_path = input_dir / f"chunk_{source_ep_id:06d}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing matching TDMPC pt for {h5_path}: {pt_path}")

        raw_episodes = read_h5_episodes(h5_path)
        tdmpc_obj = load_tdmpc_object(pt_path)
        z_all = load_tdmpc_z(tdmpc_obj, pt_path, args.latent_key)
        z_slices, _ = split_tdmpc_z_by_episodes(z_all, raw_episodes, pt_path, cfg)

        for raw, z in zip(raw_episodes, z_slices):
            ep_idx = global_ep_idx
            global_ep_idx += 1

            state, _, _ = build_state_action(raw, cfg)
            expected_z = align_latents(z, len(state), pt_path)

            if args.episode is not None and ep_idx != args.episode:
                continue
            report = compare_episode(expected_z, output_dir, ep_idx, cfg, args.repair)
            report["source_h5"] = str(h5_path)
            report["source_pt"] = str(pt_path)
            report["source_episode_id"] = source_ep_id
            report["source_z_shape"] = list(z.shape)
            report["aligned_z_shape"] = list(expected_z.shape)
            reports.append(report)

            if args.episode is not None:
                stop = True
                break
            if args.max_episodes and len(reports) >= args.max_episodes:
                stop = True
                break
        if stop:
            break

    summary = {
        "reports": reports,
        "num_reports": len(reports),
        "num_not_ok": sum(1 for item in reports if not item["ok"]),
        "num_repaired_frames": sum(item["repaired_frames"] for item in reports),
        "repair": args.repair,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
