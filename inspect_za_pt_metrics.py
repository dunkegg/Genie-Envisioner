#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise SystemExit("Missing torch. Run this in the genie_envisioner training env.") from exc


DEFAULT_PATH = "lerobot_data/tdmpc_za_7_10/latents"
LATENT_KEYS = ("za", "z", "latent", "latents", "state_dict")


def frame_index(path: Path) -> int:
    match = re.search(r"frame_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def episode_key(path: Path) -> str:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("episode_"):
            prefix = "/".join(parts[max(0, idx - 1) : idx + 1])
            return prefix
    return str(path.parent)


def unwrap_tensor(obj, path: Path) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in LATENT_KEYS:
            value = obj.get(key)
            if torch.is_tensor(value):
                return value
        tensor_items = [(key, value) for key, value in obj.items() if torch.is_tensor(value)]
        if len(tensor_items) == 1:
            return tensor_items[0][1]
        keys = ", ".join(str(k) for k in obj.keys())
        raise TypeError(f"{path}: dict does not contain a unique tensor latent. Keys: {keys}")
    raise TypeError(f"{path}: expected Tensor or dict, got {type(obj).__name__}")


def list_pt_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.pt"))


def update_delta_stats(stats, prev_tensor, curr_tensor):
    if prev_tensor.shape != curr_tensor.shape:
        stats["shape_mismatch"] += 1
        return
    delta = curr_tensor - prev_tensor
    stats["count"] += 1
    stats["mse_sum"] += float(delta.pow(2).mean().item())
    stats["rmse_sum"] += float(delta.pow(2).mean().sqrt().item())
    stats["l2_sum"] += float(delta.norm().item())
    stats["abs_mean_sum"] += float(delta.abs().mean().item())
    stats["max_abs"] = max(stats["max_abs"], float(delta.abs().max().item()))


def finalize_delta_stats(stats):
    count = stats["count"]
    if count == 0:
        return {"count": 0, "shape_mismatch": stats["shape_mismatch"]}
    return {
        "count": count,
        "shape_mismatch": stats["shape_mismatch"],
        "mse_mean": stats["mse_sum"] / count,
        "rmse_mean": stats["rmse_sum"] / count,
        "l2_mean": stats["l2_sum"] / count,
        "abs_mean": stats["abs_mean_sum"] / count,
        "max_abs": stats["max_abs"],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Za latent .pt metrics.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="A .pt file or directory containing latent .pt files")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files; 0 means all")
    parser.add_argument("--top-dims", type=int, default=12, help="How many dimensions to show for per-dim std extremes")
    parser.add_argument("--no-deltas", action="store_true", help="Skip consecutive-frame delta metrics")
    parser.add_argument("--device", default="cpu", help="torch.load map_location")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise FileNotFoundError(root)

    files = list_pt_files(root)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .pt files found under {root}")

    shape_counter = Counter()
    dtype_counter = Counter()
    num_files = 0
    num_values = 0
    dim_sum = None
    dim_sumsq = None
    dim_min = None
    dim_max = None
    scalar_sum = 0.0
    scalar_sumsq = 0.0
    scalar_abs_sum = 0.0
    scalar_min = None
    scalar_max = None
    norm_l2_sum = 0.0
    norm_l2_min = None
    norm_l2_max = None
    sample_files = []
    bad_files = []

    by_episode = defaultdict(list)
    delta_stats = {
        "count": 0,
        "shape_mismatch": 0,
        "mse_sum": 0.0,
        "rmse_sum": 0.0,
        "l2_sum": 0.0,
        "abs_mean_sum": 0.0,
        "max_abs": 0.0,
    }

    loaded = []
    for path in files:
        try:
            obj = torch.load(path, map_location=args.device)
            tensor = unwrap_tensor(obj, path).detach().float().cpu()
        except Exception as exc:
            bad_files.append({"path": str(path), "error": repr(exc)})
            continue

        flat = tensor.reshape(-1)
        if flat.numel() == 0:
            bad_files.append({"path": str(path), "error": "empty tensor"})
            continue

        num_files += 1
        num_values += int(flat.numel())
        shape_counter[str(tuple(tensor.shape))] += 1
        dtype_counter[str(tensor.dtype)] += 1
        if len(sample_files) < 5:
            sample_files.append(
                {
                    "path": str(path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "preview": flat[:8].tolist(),
                }
            )

        vec = flat.double()
        if dim_sum is None:
            dim_sum = torch.zeros_like(vec)
            dim_sumsq = torch.zeros_like(vec)
            dim_min = vec.clone()
            dim_max = vec.clone()
        elif vec.shape != dim_sum.shape:
            bad_files.append(
                {
                    "path": str(path),
                    "error": f"shape {tuple(tensor.shape)} flattens to {vec.numel()}, expected {dim_sum.numel()}",
                }
            )
            continue

        dim_sum += vec
        dim_sumsq += vec * vec
        dim_min = torch.minimum(dim_min, vec)
        dim_max = torch.maximum(dim_max, vec)

        scalar_sum += float(vec.sum().item())
        scalar_sumsq += float((vec * vec).sum().item())
        scalar_abs_sum += float(vec.abs().sum().item())
        cur_min = float(vec.min().item())
        cur_max = float(vec.max().item())
        scalar_min = cur_min if scalar_min is None else min(scalar_min, cur_min)
        scalar_max = cur_max if scalar_max is None else max(scalar_max, cur_max)

        l2 = float(vec.norm().item())
        norm_l2_sum += l2
        norm_l2_min = l2 if norm_l2_min is None else min(norm_l2_min, l2)
        norm_l2_max = l2 if norm_l2_max is None else max(norm_l2_max, l2)

        if not args.no_deltas:
            by_episode[episode_key(path)].append((frame_index(path), path, vec.float()))

    if num_files == 0 or dim_sum is None:
        raise RuntimeError(f"No readable latent tensors found. Bad files: {bad_files[:5]}")

    dim_mean = dim_sum / num_files
    dim_var = (dim_sumsq / num_files) - dim_mean * dim_mean
    dim_std = dim_var.clamp_min(0.0).sqrt()
    scalar_mean = scalar_sum / num_values
    scalar_var = scalar_sumsq / num_values - scalar_mean * scalar_mean
    scalar_std = max(scalar_var, 0.0) ** 0.5

    if not args.no_deltas:
        for items in by_episode.values():
            items.sort(key=lambda x: x[0])
            prev = None
            for _, _, tensor in items:
                if prev is not None:
                    update_delta_stats(delta_stats, prev, tensor)
                prev = tensor

    top_k = min(args.top_dims, dim_std.numel())
    high_std = torch.topk(dim_std, k=top_k, largest=True).indices.tolist()
    low_std = torch.topk(dim_std, k=top_k, largest=False).indices.tolist()

    result = {
        "path": str(root),
        "files_seen": len(files),
        "files_read": num_files,
        "bad_files_count": len(bad_files),
        "shape_counts": dict(shape_counter),
        "dtype_counts": dict(dtype_counter),
        "num_values_per_file": int(dim_sum.numel()),
        "global": {
            "mean": scalar_mean,
            "std": scalar_std,
            "abs_mean": scalar_abs_sum / num_values,
            "min": scalar_min,
            "max": scalar_max,
        },
        "per_file_l2_norm": {
            "mean": norm_l2_sum / num_files,
            "min": norm_l2_min,
            "max": norm_l2_max,
        },
        "per_dim": {
            "mean_abs_mean": float(dim_mean.abs().mean().item()),
            "std_mean": float(dim_std.mean().item()),
            "std_min": float(dim_std.min().item()),
            "std_max": float(dim_std.max().item()),
            "highest_std_dims": [
                {
                    "dim": int(i),
                    "mean": float(dim_mean[i].item()),
                    "std": float(dim_std[i].item()),
                    "min": float(dim_min[i].item()),
                    "max": float(dim_max[i].item()),
                }
                for i in high_std
            ],
            "lowest_std_dims": [
                {
                    "dim": int(i),
                    "mean": float(dim_mean[i].item()),
                    "std": float(dim_std[i].item()),
                    "min": float(dim_min[i].item()),
                    "max": float(dim_max[i].item()),
                }
                for i in low_std
            ],
        },
        "consecutive_frame_delta": finalize_delta_stats(delta_stats) if not args.no_deltas else None,
        "sample_files": sample_files,
        "bad_files_preview": bad_files[:10],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
