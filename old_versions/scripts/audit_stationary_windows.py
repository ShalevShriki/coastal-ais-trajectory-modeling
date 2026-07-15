#!/usr/bin/env python3
"""Report how many windows are confined/stationary before training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proj.project.window_data import (
    StationaryFilterConfig,
    compute_window_motion_metrics,
    filter_stationary_windows,
    load_windows,
    stationary_window_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit stationary/confined AIS windows.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined/train.parquet"),
    )
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--max-confined-radius-km", type=float, default=0.5)
    parser.add_argument("--min-future-displacement-km", type=float, default=1.0)
    parser.add_argument("--min-mean-sog-kn", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = StationaryFilterConfig(
        enabled=True,
        max_confined_radius_km=args.max_confined_radius_km,
        min_future_displacement_km=args.min_future_displacement_km,
        min_mean_sog_kn=args.min_mean_sog_kn,
    )

    print(f"Loading up to {args.sample:,} windows from {args.input} ...")
    df = load_windows(args.input, sample_size=args.sample)
    metrics = compute_window_motion_metrics(df)
    remove = stationary_window_mask(metrics, config)

    report = {
        "input": str(args.input),
        "sample_size": len(df),
        "filter": {
            "max_confined_radius_km": config.max_confined_radius_km,
            "min_future_displacement_km": config.min_future_displacement_km,
            "min_mean_sog_kn": config.min_mean_sog_kn,
        },
        "stationary_windows": int(remove.sum()),
        "moving_windows": int((~remove).sum()),
        "stationary_fraction": float(remove.mean()),
        "metrics_removed": {
            "median_max_radius_km": float(metrics.loc[remove, "max_radius_km"].median()),
            "median_future_displacement_km": float(
                metrics.loc[remove, "future_displacement_km"].median()
            ),
            "median_path_length_km": float(metrics.loc[remove, "path_length_km"].median()),
            "median_mean_sog_kn": float(metrics.loc[remove, "mean_sog_kn"].median()),
        },
        "metrics_kept": {
            "median_max_radius_km": float(metrics.loc[~remove, "max_radius_km"].median()),
            "median_future_displacement_km": float(
                metrics.loc[~remove, "future_displacement_km"].median()
            ),
            "median_path_length_km": float(metrics.loc[~remove, "path_length_km"].median()),
            "median_mean_sog_kn": float(metrics.loc[~remove, "mean_sog_kn"].median()),
        },
    }

    print("\n=== Stationary window audit ===")
    print(f"  Total sampled:     {len(df):,}")
    print(f"  Would remove:      {report['stationary_windows']:,} ({report['stationary_fraction']*100:.1f}%)")
    print(f"  Would keep:        {report['moving_windows']:,}")
    print("\nRemoved (confined) — medians:")
    for k, v in report["metrics_removed"].items():
        print(f"    {k}: {v:.3f}")
    print("\nKept (moving) — medians:")
    for k, v in report["metrics_kept"].items():
        print(f"    {k}: {v:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
