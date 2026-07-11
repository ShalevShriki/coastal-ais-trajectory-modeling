#!/usr/bin/env python3
"""
Apply training-time data filters to combined window parquets.

Currently applies:
  1. Stationary / confined-window removal (radius + future displacement + SOG)

Writes filtered parquets row-group-by-row-group (memory safe) and a JSON report.
Maneuver oversampling is a training-time subsampling strategy (not row removal);
this script reports maneuver-score stats on the filtered set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proj.project.window_data import (
    SmartMotionFilterConfig,
    StationaryFilterConfig,
    classify_window_motion,
    compute_history_motion_metrics,
    compute_maneuver_scores_df,
    compute_smart_motion_metrics,
    compute_window_motion_metrics,
    filter_stationary_windows,
    infer_window_shape,
    smart_motion_window_mask,
    stationary_window_mask,
)


def _lat_lon_columns(history_steps: int, future_steps: int) -> list[str]:
    cols: list[str] = []
    for t in range(history_steps):
        cols.extend([f"x_t{t:03d}_lat", f"x_t{t:03d}_lon"])
    for t in range(future_steps):
        cols.extend([f"y_t{t:03d}_lat", f"y_t{t:03d}_lon"])
    return cols


def _infer_steps_from_schema(names: list[str]) -> tuple[int, int]:
    hist = 0
    fut = 0
    for c in names:
        m = re.match(r"^x_t(\d+)_", c)
        if m:
            hist = max(hist, int(m.group(1)) + 1)
        m = re.match(r"^y_t(\d+)_", c)
        if m:
            fut = max(fut, int(m.group(1)) + 1)
    return hist, fut


def filter_parquet_incremental(
    src: Path,
    dst: Path,
    config: StationaryFilterConfig,
    smart_config: SmartMotionFilterConfig | None = None,
    *,
    batch_size: int = 50_000,
) -> dict:
    pf = pq.ParquetFile(src)
    schema_names = pf.schema.names
    history_steps, future_steps = _infer_steps_from_schema(schema_names)
    geo_cols = _lat_lon_columns(history_steps, future_steps)
    sog_cols = [f"x_t{t:03d}_sog" for t in range(history_steps)] + [
        f"y_t{t:03d}_sog" for t in range(future_steps)
    ]
    metric_cols = [c for c in geo_cols + sog_cols if c in schema_names]

    writer: pq.ParquetWriter | None = None
    total_in = 0
    total_out = 0
    maneuver_scores: list[np.ndarray] = []
    radius_kept: list[float] = []

    t0 = time.perf_counter()
    batch_idx = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        full_df = batch.to_pandas()
        total_in += len(full_df)

        remove = np.zeros(len(full_df), dtype=bool)
        if config.enabled:
            remove = stationary_window_mask(
                (
                    compute_history_motion_metrics(full_df, history_steps)
                    if config.history_only
                    else compute_window_motion_metrics(full_df, history_steps, future_steps)
                ),
                config,
            )
        if smart_config is not None and smart_config.enabled:
            smart_metrics = compute_smart_motion_metrics(full_df, history_steps, future_steps)
            remove = remove | smart_motion_window_mask(smart_metrics, smart_config)

        keep = ~remove

        if keep.any():
            filtered = full_df.loc[keep].reset_index(drop=True)

            scores = compute_maneuver_scores_df(filtered, history_steps)
            maneuver_scores.append(scores)
            m = (
                compute_history_motion_metrics(filtered, history_steps)
                if config.history_only
                else compute_window_motion_metrics(filtered, history_steps, future_steps)
            )
            radius_col = "history_max_radius_km" if config.history_only else "max_radius_km"
            radius_kept.extend(m[radius_col].tolist())

            total_out += len(filtered)
            table = pa.Table.from_pandas(filtered, preserve_index=False)
            if writer is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(dst, table.schema, compression="snappy")
            writer.write_table(table)
            del filtered, table

        del full_df, remove, keep
        batch_idx += 1
        if batch_idx % 20 == 0:
            print(
                f"  {src.name}: batch {batch_idx} | in={total_in:,} out={total_out:,}",
                flush=True,
            )

    print(
        f"  {src.name}: done ({batch_idx} batches) | in={total_in:,} out={total_out:,}",
        flush=True,
    )

    if writer is not None:
        writer.close()
    elif pf.metadata.num_row_groups > 0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        pf.read_row_group(0).to_pandas().iloc[:0].to_parquet(dst, index=False)

    elapsed = time.perf_counter() - t0
    all_scores = np.concatenate(maneuver_scores) if maneuver_scores else np.array([])

    return {
        "source": str(src),
        "output": str(dst),
        "rows_in": total_in,
        "rows_out": total_out,
        "removed": total_in - total_out,
        "removed_fraction": float((total_in - total_out) / max(total_in, 1)),
        "elapsed_sec": elapsed,
        "motion_summary": {
            "median_max_radius_km": float(np.median(radius_kept)) if radius_kept else 0.0,
        },
        "maneuver_score_median_kept": float(np.median(all_scores)) if len(all_scores) else 0.0,
        "maneuver_score_p90_kept": float(np.percentile(all_scores, 90)) if len(all_scores) else 0.0,
    }


def simulate_training_sample(
    train_filtered: Path,
    sample_size: int,
    maneuver_fraction: float,
    seed: int,
) -> dict:
    """Report 400k training draw: maneuver oversample + counts."""
    from proj.project.window_data import load_windows

    df = load_windows(
        train_filtered,
        sample_size=sample_size,
        maneuver_oversample=True,
        maneuver_fraction=maneuver_fraction,
        seed=seed,
    )
    scores = compute_maneuver_scores_df(df)
    history_steps, _ = infer_window_shape(df)

    # Build minimal x for motion class from history sog/dcog columns
    sog_cols = [f"x_t{t:03d}_sog" for t in range(history_steps)]
    dcog_cols = [f"x_t{t:03d}_dcog" for t in range(history_steps)]
    x_proxy = np.zeros((len(df), history_steps, 2), dtype=np.float64)
    if all(c in df.columns for c in sog_cols):
        x_proxy[:, :, 0] = df[sog_cols].to_numpy()
    if all(c in df.columns for c in dcog_cols):
        x_proxy[:, :, 1] = df[dcog_cols].to_numpy()

    # classify needs feature indices - use proxy with sog at 0, dcog at 1
    from proj.project.window_data import FEATURE_COLS, feature_index

    class_cols = ["sog", "dcog"]
    x_full = np.zeros((len(df), history_steps, len(FEATURE_COLS)), dtype=np.float64)
    x_full[:, :, feature_index(FEATURE_COLS, "sog")] = x_proxy[:, :, 0]
    x_full[:, :, feature_index(FEATURE_COLS, "dcog")] = x_proxy[:, :, 1]
    buckets = classify_window_motion(x_full)

    return {
        "sample_size": len(df),
        "maneuver_fraction_requested": maneuver_fraction,
        "maneuver_score_median": float(np.median(scores)),
        "maneuver_score_p90": float(np.percentile(scores, 90)),
        "buckets": {k: int(v.sum()) for k, v in buckets.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply training filters to combined parquets.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/combined"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/combined_filtered"),
    )
    parser.add_argument("--max-confined-radius-km", type=float, default=0.5)
    parser.add_argument(
        "--min-displacement-km",
        type=float,
        default=1.0,
        help="Min net displacement over history (default) or future if --filter-use-future.",
    )
    parser.add_argument("--min-mean-sog-kn", type=float, default=0.5)
    parser.add_argument(
        "--filter-use-future",
        action="store_true",
        help="Legacy filter: use future window motion (leaks labels; not recommended).",
    )
    parser.add_argument(
        "--require-min-history-displacement-km",
        type=float,
        default=0.0,
        help="Require 24h history net displacement >= this km (default: 0 = off).",
    )
    parser.add_argument(
        "--require-min-history-path-km",
        type=float,
        default=0.0,
        help="Require 24h history path length >= this km (default: 0 = off).",
    )
    parser.add_argument(
        "--max-history-loop-ratio",
        type=float,
        default=None,
        help="Drop ferries/local loops: remove if history net/path below this (e.g. 0.35).",
    )
    parser.add_argument("--min-path-for-loop-km", type=float, default=10.0)
    parser.add_argument(
        "--smart-motion",
        action="store_true",
        help="Require last 16h/8h of history show real movement (drops arrived-and-stopped).",
    )
    parser.add_argument("--smart-min-16h-net-km", type=float, default=8.0)
    parser.add_argument("--smart-min-last-8h-net-km", type=float, default=2.0)
    parser.add_argument("--smart-max-loop-ratio", type=float, default=0.35)
    parser.add_argument(
        "--skip-stationary",
        action="store_true",
        help="Do not re-apply confined/stationary filter (input is already filtered).",
    )
    parser.add_argument("--sample", type=int, default=400_000)
    parser.add_argument("--maneuver-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = StationaryFilterConfig(
        enabled=not args.skip_stationary,
        history_only=not args.filter_use_future,
        max_confined_radius_km=args.max_confined_radius_km,
        min_displacement_km=args.min_displacement_km,
        min_mean_sog_kn=args.min_mean_sog_kn,
        require_min_history_displacement_km=args.require_min_history_displacement_km,
        require_min_history_path_km=args.require_min_history_path_km,
        max_history_loop_ratio=args.max_history_loop_ratio,
        min_path_for_loop_km=args.min_path_for_loop_km,
    )

    smart_config = SmartMotionFilterConfig(
        enabled=args.smart_motion,
        min_history_16h_net_km=args.smart_min_16h_net_km,
        min_history_last_8h_net_km=args.smart_min_last_8h_net_km,
        max_history_loop_ratio=args.smart_max_loop_ratio,
    )

    report: dict = {
        "filter": {
            "history_only": config.history_only,
            "max_confined_radius_km": config.max_confined_radius_km,
            "min_displacement_km": config.min_displacement_km,
            "min_mean_sog_kn": config.min_mean_sog_kn,
            "require_min_history_displacement_km": config.require_min_history_displacement_km,
            "require_min_history_path_km": config.require_min_history_path_km,
            "max_history_loop_ratio": config.max_history_loop_ratio,
            "min_path_for_loop_km": config.min_path_for_loop_km,
        },
        "smart_motion": {
            "enabled": smart_config.enabled,
            "min_history_16h_net_km": smart_config.min_history_16h_net_km,
            "min_history_last_8h_net_km": smart_config.min_history_last_8h_net_km,
            "max_history_loop_ratio": smart_config.max_history_loop_ratio,
            "audit_min_future_8h_net_km": smart_config.audit_min_future_8h_net_km,
            "audit_min_future_12h_net_km": smart_config.audit_min_future_12h_net_km,
        },
        "splits": {},
    }

    print("=== Applying stationary filter to combined splits ===")
    scope = "history-only" if config.history_only else "history+future"
    print(
        f"mode={scope} | radius≤{config.max_confined_radius_km} km & "
        f"displacement<{config.min_displacement_km} km\n"
    )
    if smart_config.enabled:
        print(
            f"smart-motion: last-16h net≥{smart_config.min_history_16h_net_km} km, "
            f"last-8h net≥{smart_config.min_history_last_8h_net_km} km, "
            f"loop ratio≥{smart_config.max_history_loop_ratio}\n"
        )

    for name in ("train.parquet", "val.parquet", "test.parquet"):
        src = args.input_dir / name
        if not src.exists():
            print(f"Skip missing: {src}")
            continue
        dst = args.output_dir / name
        print(f"Processing {name}...")
        report["splits"][name] = filter_parquet_incremental(
            src, dst, config, smart_config=smart_config
        )

    train_out = args.output_dir / "train.parquet"
    if train_out.exists():
        print("\n=== Simulating training sample (maneuver oversample) ===")
        report["training_sample"] = simulate_training_sample(
            train_out,
            sample_size=args.sample,
            maneuver_fraction=args.maneuver_fraction,
            seed=args.seed,
        )
        print(json.dumps(report["training_sample"], indent=2))

    report_path = args.output_dir / "filter_report.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for name, stats in report["splits"].items():
        print(
            f"{name}: {stats['rows_in']:,} → {stats['rows_out']:,} "
            f"(removed {stats['removed']:,}, {stats['removed_fraction']*100:.1f}%)"
        )
    print(f"\nFiltered data: {args.output_dir}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
