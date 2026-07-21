#!/usr/bin/env python3
"""Calculate streaming statistics for per-frame latent tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_LATENTS_DIR = Path(
    "lerobot_data/tdmpc_za_7_10/latents"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate global and per-dimension statistics for all latent .pt files. "
            "Statistics are accumulated in float64 without loading the whole dataset."
        )
    )
    parser.add_argument("--latents-dir", type=Path, default=DEFAULT_LATENTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <latents-dir>/latent_statistics.json).",
    )
    parser.add_argument(
        "--key",
        default="z",
        help="Tensor key when a .pt file contains a dict (default: z).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N files; use 0 to disable (default: 500).",
    )
    return parser.parse_args()


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not support weights_only.
        return torch.load(path, map_location="cpu")


def extract_tensor(obj: Any, key: str, path: Path) -> torch.Tensor:
    if torch.is_tensor(obj):
        value = obj
    elif isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"{path}: key {key!r} not found; available keys: {list(obj)}")
        value = obj[key]
    else:
        raise TypeError(f"{path}: expected a tensor or dict, got {type(obj).__name__}")

    value = torch.as_tensor(value).detach().cpu()
    if value.numel() == 0:
        raise ValueError(f"{path}: latent tensor is empty")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D latent, got shape {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all():
        bad = int((~torch.isfinite(value)).sum().item())
        raise ValueError(f"{path}: latent contains {bad} NaN/Inf value(s)")
    return value.to(torch.float64)


def merge_batch(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Merge rows in batch into Welford state along dimension 0."""
    batch_count = batch.shape[0]
    batch_mean = batch.mean(dim=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    delta = batch_mean - mean
    new_count = count + batch_count
    mean = mean + delta * (batch_count / new_count)
    m2 = m2 + batch_m2 + delta.square() * count * batch_count / new_count
    return new_count, mean, m2


def main() -> None:
    args = parse_args()
    latents_dir = args.latents_dir.expanduser().resolve()
    output = (args.output or latents_dir / "latent_statistics.json").expanduser().resolve()
    files = sorted(latents_dir.rglob("*.pt"))
    if not latents_dir.is_dir():
        raise NotADirectoryError(f"Latents directory does not exist: {latents_dir}")
    if not files:
        raise FileNotFoundError(f"No .pt files found under: {latents_dir}")

    count = 0
    feature_shape: tuple[int, ...] | None = None
    mean = m2 = minimum = maximum = None
    source_dtypes: set[str] = set()

    for index, path in enumerate(files, start=1):
        raw = safe_torch_load(path)
        raw_value = raw.get(args.key) if isinstance(raw, dict) and args.key in raw else raw
        if torch.is_tensor(raw_value):
            source_dtypes.add(str(raw_value.dtype).removeprefix("torch."))
        value = extract_tensor(raw, args.key, path)

        if feature_shape is None:
            feature_shape = tuple(value.shape)
            mean = torch.zeros_like(value)
            m2 = torch.zeros_like(value)
            minimum = value.clone()
            maximum = value.clone()
        elif tuple(value.shape) != feature_shape:
            raise ValueError(
                f"{path}: shape {tuple(value.shape)} differs from expected {feature_shape}"
            )

        count, mean, m2 = merge_batch(count, mean, m2, value.unsqueeze(0))
        minimum = torch.minimum(minimum, value)
        maximum = torch.maximum(maximum, value)
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(files)):
            print(f"Processed {index}/{len(files)} files")

    population_std = torch.sqrt(m2 / count)
    sample_std = torch.sqrt(m2 / (count - 1)) if count > 1 else torch.full_like(mean, float("nan"))
    global_count = count * mean.numel()
    global_mean = mean.mean()
    # Within-feature variation plus variation among feature means.
    global_m2 = m2.sum() + count * ((mean - global_mean) ** 2).sum()

    result = {
        "latents_dir": str(latents_dir),
        "file_pattern": "**/*.pt",
        "key": args.key,
        "num_files": len(files),
        "num_latent_vectors": count,
        "latent_shape": list(feature_shape),
        "source_dtypes": sorted(source_dtypes),
        "global": {
            "count": global_count,
            "mean": global_mean.item(),
            "std_population": torch.sqrt(global_m2 / global_count).item(),
            "std_sample": (
                torch.sqrt(global_m2 / (global_count - 1)).item()
                if global_count > 1
                else None
            ),
            "min": minimum.min().item(),
            "max": maximum.max().item(),
        },
        "per_dimension": {
            "mean": mean.tolist(),
            "std_population": population_std.tolist(),
            "std_sample": sample_std.tolist() if count > 1 else None,
            "min": minimum.tolist(),
            "max": maximum.tolist(),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Statistics written to: {output}")
    print(json.dumps(result["global"], indent=2))


if __name__ == "__main__":
    main()
