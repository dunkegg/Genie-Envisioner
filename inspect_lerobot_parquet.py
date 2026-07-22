#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing numpy. Run this in the training env with numpy + pandas + pyarrow.") from exc


DEFAULT_PARQUET = "lerobot_data/tdmpc_za_7_10/data/chunk-000/episode_000001.parquet"
KEY_COLUMNS = ("actions", "observation.state", "beta", "latent")


def to_numpy(value):
    arr = np.asarray(value)
    if arr.dtype == object and arr.ndim == 0:
        arr = np.asarray(arr.item())
    return arr


def summarize_value(value, max_items=8):
    if isinstance(value, str):
        return {"type": "str", "value": value}
    arr = to_numpy(value)
    flat = arr.reshape(-1) if arr.ndim > 0 else arr.reshape(1)
    preview = flat[:max_items].tolist()
    summary = {
        "type": type(value).__name__,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "preview": preview,
    }
    if np.issubdtype(arr.dtype, np.number) and flat.size:
        summary.update(
            {
                "min": float(np.nanmin(flat)),
                "max": float(np.nanmax(flat)),
                "mean": float(np.nanmean(flat)),
            }
        )
    return summary


def resolve_latent_path(parquet_path, latent_value):
    if not isinstance(latent_value, str):
        return None
    candidate = Path(latent_value)
    if candidate.is_absolute():
        return candidate

    parquet_dir = parquet_path.parent
    data_dir = parquet_dir.parent.parent
    domain_root = data_dir.parent
    candidates = [
        parquet_dir / candidate,
        data_dir / candidate,
        domain_root / candidate,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def print_json(title, obj):
    print(f"\n## {title}")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inspect a LeRobot parquet episode.")
    parser.add_argument("parquet", nargs="?", default=DEFAULT_PARQUET, help="Path to episode_*.parquet")
    parser.add_argument("--rows", type=int, default=3, help="How many rows to preview")
    parser.add_argument("--load-latent", action="store_true", help="Load preview latent .pt files and print tensor shapes")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas/parquet support. Run this in the training env with pandas + pyarrow.") from exc

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    print(f"parquet: {parquet_path}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    column_info = {}
    for col in df.columns:
        first_valid = None
        for value in df[col].head(max(args.rows, 1)):
            if value is not None:
                first_valid = value
                break
        column_info[col] = summarize_value(first_valid) if first_valid is not None else {"type": "None"}
    print_json("column first-value summary", column_info)

    for col in KEY_COLUMNS:
        if col not in df.columns:
            continue
        print(f"\n## {col}")
        for idx in range(min(args.rows, len(df))):
            print(f"row {idx}: {json.dumps(summarize_value(df[col].iloc[idx]), ensure_ascii=False)}")

    if "latent" in df.columns:
        latent_paths = []
        for idx in range(min(args.rows, len(df))):
            path = resolve_latent_path(parquet_path, df["latent"].iloc[idx])
            latent_paths.append(None if path is None else str(path))
        print_json("resolved latent paths", latent_paths)

        if args.load_latent:
            try:
                import torch
            except ImportError as exc:
                raise SystemExit("Missing torch; cannot load latent .pt files.") from exc
            latent_info = []
            for path_str in latent_paths:
                if path_str is None:
                    latent_info.append({"path": None})
                    continue
                path = Path(path_str)
                if not path.exists():
                    latent_info.append({"path": path_str, "exists": False})
                    continue
                tensor = torch.load(path, map_location="cpu")
                latent_info.append(
                    {
                        "path": path_str,
                        "exists": True,
                        "type": type(tensor).__name__,
                        "shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
                        "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
                    }
                )
            print_json("latent tensor summary", latent_info)


if __name__ == "__main__":
    main()
