#!/usr/bin/env python3
"""Rebuild February model_ready_windows.parquet from existing coastal_segments.

No download / day re-clean. Uses the PROCESS_* streaming window builder (lat/lon
dedupe fix already applied in resample_one_trajectory).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PROCESS_DIR = PROJECT / "processing"

FEB_DATASETS = [
    {
        "label": "east_coast_feb",
        "segments": PROJECT
        / "data/processed/Eastern coast/ais_east_coast_feb_long_horizon/coastal_segments.parquet",
        "process_file": PROCESS_DIR / "PROCESS_noaa_long_coastal Eastern coast.py",
        "module_name": "process_east_coast_rebuild",
    },
    {
        "label": "mexican_coast_feb",
        "segments": PROJECT
        / "data/processed/Mexcany Beach/ais_mexican_coast_feb_long_horizon/coastal_segments.parquet",
        "process_file": PROCESS_DIR / "PROCESS_noaa_long_coastal Mexcany Beach.py",
        "module_name": "process_mexican_coast_rebuild",
    },
    {
        "label": "west_coast_feb",
        "segments": PROJECT
        / "data/processed/West Coast/ais_west_coast_feb_long_horizon/coastal_segments.parquet",
        "process_file": PROCESS_DIR / "PROCESS_noaa_long_coastal West Coast.py",
        "module_name": "process_west_coast_rebuild",
    },
]


def load_process(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load process module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def rebuild_one(
    *,
    label: str,
    segments_path: Path,
    process_file: Path,
    module_name: str,
    history_hours: float,
    future_hours: float,
    resample_minutes: int,
    max_windows_per_traj: int,
) -> None:
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing segments for {label}: {segments_path}")
    if not process_file.exists():
        raise FileNotFoundError(f"Missing process script for {label}: {process_file}")

    windows_path = segments_path.parent / "model_ready_windows.parquet"
    print(f"\n=== {label} ===", flush=True)
    print(f"  segments: {segments_path}", flush=True)
    print(f"  output:   {windows_path}", flush=True)

    t0 = time.perf_counter()
    process = load_process(module_name, process_file)
    segments = pd.read_parquet(segments_path)
    n_traj = segments["traj_id"].nunique() if "traj_id" in segments.columns else "?"
    print(f"  loaded {len(segments):,} rows | trajectories: {n_traj}", flush=True)

    # Sanity: segments must have distinct lon (unlike the corrupt windows).
    sample = segments[["lat", "lon"]].dropna().head(1000)
    frac_eq = float((sample["lat"].round(4) == sample["lon"].round(4)).mean()) if len(sample) else 0.0
    print(f"  segment lat==lon fraction (first 1000): {frac_eq:.4f}", flush=True)
    if frac_eq > 0.5:
        raise RuntimeError(f"{label}: coastal_segments look corrupt (lat≈lon). Aborting.")

    if windows_path.exists():
        # Drop corrupt Feb windows; do not keep a second multi-GB copy on disk.
        size_gb = windows_path.stat().st_size / (1024**3)
        windows_path.unlink()
        print(f"  deleted old windows ({size_gb:.1f} GB)", flush=True)

    count = process.build_sequence_windows(
        segments,
        history_hours=history_hours,
        future_hours=future_hours,
        resample_minutes=resample_minutes,
        max_windows_per_traj=max_windows_per_traj,
        # Match finalize: do not drop windows that exit the bbox mid-horizon.
        lat_range=None,
        lon_range=None,
        output_path=windows_path,
    )
    elapsed = time.perf_counter() - t0
    print(f"  wrote {count:,} windows in {elapsed / 60:.1f} min", flush=True)

    # Post-check: lat/lon must differ in the new windows.
    check = pd.read_parquet(
        windows_path,
        columns=["x_t000_lat", "x_t000_lon", "y_t000_lat", "y_t000_lon"],
    ).head(5000)
    win_eq = float((check["x_t000_lat"].round(4) == check["x_t000_lon"].round(4)).mean())
    print(f"  window x_t000 lat==lon fraction (first 5000): {win_eq:.4f}", flush=True)
    if win_eq > 0.05:
        raise RuntimeError(
            f"{label}: rebuilt windows still have lat≈lon ({win_eq:.3f}). Fix failed — aborting."
        )
    print(f"  OK: lon looks distinct from lat", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["east_coast_feb", "mexican_coast_feb", "west_coast_feb"],
        nargs="*",
        default=None,
        help="Optional subset of February coasts (default: all three).",
    )
    parser.add_argument("--history-hours", type=float, default=24.0)
    parser.add_argument("--future-hours", type=float, default=12.0)
    parser.add_argument("--resample-minutes", type=int, default=10)
    parser.add_argument("--max-windows-per-traj", type=int, default=200)
    args = parser.parse_args()

    wanted = set(args.only) if args.only else {d["label"] for d in FEB_DATASETS}
    overall = time.perf_counter()
    for ds in FEB_DATASETS:
        if ds["label"] not in wanted:
            continue
        rebuild_one(
            label=ds["label"],
            segments_path=ds["segments"],
            process_file=ds["process_file"],
            module_name=ds["module_name"],
            history_hours=args.history_hours,
            future_hours=args.future_hours,
            resample_minutes=args.resample_minutes,
            max_windows_per_traj=args.max_windows_per_traj,
        )
    print(f"\nAll requested rebuilds done in {(time.perf_counter() - overall) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
