#!/usr/bin/env python3
"""
Rebuild model_ready_windows.parquet from existing coastal_segments.parquet files.

By default only builds windows for datasets that are missing them.
Use --all to overwrite existing windows (e.g. after changing history/future hours).

Usage:
  cd <project root>

  # Build only missing windows (new 24h/12h defaults)
  python scripts/rebuild_windows.py

  # Rebuild ALL windows with new params (overwrites existing)
  python scripts/rebuild_windows.py --all

  # One specific dataset
  python scripts/rebuild_windows.py \\
      --segments "data/processed/Eastern coast/ais_east_coast_7d_long_horizon/coastal_segments.parquet"

  # Different window size
  python scripts/rebuild_windows.py --all --history-hours 6 --future-hours 5
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Region bounding boxes (lat_min, lat_max, lon_min, lon_max)
# Must match the values used during the original data cleaning.
# ---------------------------------------------------------------------------
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "east_coast":     (35.0,  43.5,  -76.5, -68.0),
    "west_coast":     (32.0,  49.5, -126.0, -117.0),
    "mexican_coast":  (14.0,  32.5, -118.0,  -86.0),
    "mexico_pacific": (14.0,  32.5, -118.0, -105.0),
    "mexico_gulf":    (18.0,  26.5,  -98.0,  -86.0),
    "gulf":           (24.0,  31.0,  -98.0,  -80.0),
    "california":     (32.0,  38.5, -124.5, -117.0),
    "pnw":            (45.0,  49.5, -126.0, -122.0),
    "danish":         (54.5,  58.0,    7.5,   13.0),
}

# ---------------------------------------------------------------------------
# Features — must match PROCESS_noaa_long_coastal scripts
# ---------------------------------------------------------------------------
WINDOW_FEATURE_COLS = [
    "lat", "lon", "sog", "cog_sin", "cog_cos",
    "heading_sin", "heading_cos", "heading_missing",
    "dt_sec", "dlat", "dlon", "dsog", "dcog",
    "v_north_kmh", "v_east_kmh",
]
TARGET_COLS = ["lat", "lon"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def infer_region(segments_path: Path) -> str | None:
    """Extract region name from folder pattern ais_<region>[_<tag>]_long_horizon."""
    folder = segments_path.parent.name
    m = re.match(r"^ais_(.+?)(?:_\d+[dm])?_long_horizon$", folder)
    if not m:
        return None
    return m.group(1)


def get_bbox(segments_path: Path) -> tuple[tuple[float, float], tuple[float, float]] | None:
    region = infer_region(segments_path)
    if region and region in REGIONS:
        lat_min, lat_max, lon_min, lon_max = REGIONS[region]
        return (lat_min, lat_max), (lon_min, lon_max)
    return None


# ---------------------------------------------------------------------------
# Core: resample + window generation
# ---------------------------------------------------------------------------

def patch_missing_motion_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute any motion columns absent from older segment files."""
    if "v_north_kmh" not in df.columns or "v_east_kmh" not in df.columns:
        if "sog" in df.columns and "cog_sin" in df.columns and "cog_cos" in df.columns:
            sog_kmh = df["sog"] * 1.852
            df = df.copy()
            df["v_north_kmh"] = sog_kmh * df["cog_cos"]
            df["v_east_kmh"] = sog_kmh * df["cog_sin"]
        else:
            df = df.copy()
            df["v_north_kmh"] = 0.0
            df["v_east_kmh"] = 0.0
    return df


def resample_trajectory(traj: pd.DataFrame, every_minutes: int, max_gap_steps: int) -> pd.DataFrame:
    traj = traj.sort_values("timestamp").drop_duplicates("timestamp").copy()
    traj = traj.set_index("timestamp")

    numeric_cols = list(dict.fromkeys(c for c in WINDOW_FEATURE_COLS + TARGET_COLS if c in traj.columns))
    meta_cols = [c for c in ["mmsi", "traj_id", "vessel_type"] if c in traj.columns]

    numeric = traj[numeric_cols].resample(f"{every_minutes}min").mean().interpolate(
        method="time",
        limit_direction="both",
        limit=max_gap_steps,
    )
    meta = traj[meta_cols].resample(f"{every_minutes}min").ffill().bfill()

    out = pd.concat([meta, numeric], axis=1).dropna(subset=["lat", "lon"])
    return out.reset_index()


def build_windows(
    segments: pd.DataFrame,
    *,
    history_hours: float,
    future_hours: float,
    resample_minutes: int,
    max_windows_per_traj: int | None,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
    max_gap_steps: int,
    strict_bbox_future: bool = False,
) -> pd.DataFrame:
    t0 = time.perf_counter()

    history_steps = int(round(history_hours * 60 / resample_minutes))
    future_steps = int(round(future_hours * 60 / resample_minutes))
    total_steps = history_steps + future_steps

    print(
        f"  Window params: history={history_hours}h ({history_steps} steps) | "
        f"future={future_hours}h ({future_steps} steps) | resample={resample_minutes}min",
        flush=True,
    )
    if strict_bbox_future and lat_range:
        print(f"  Bbox filter (future positions): lat {lat_range}, lon {lon_range}", flush=True)
    else:
        print("  Bbox filter: disabled (predictions may exit the region frame)", flush=True)

    rng = np.random.default_rng(42)
    n_feat = len(WINDOW_FEATURE_COLS)

    # Collect compact numpy slices instead of Python dicts.
    # float32 = 4 bytes/value vs Python float = 28 bytes → 7× less memory.
    hist_list: list[np.ndarray] = []   # each: (history_steps, n_feat)
    fut_list:  list[np.ndarray] = []   # each: (future_steps, 2)
    meta: dict[str, list] = {
        "traj_id": [], "mmsi": [], "start_time": [], "split_time": [],
        "target_end_time": [], "history_steps": [], "future_steps": [],
        "resample_minutes": [],
    }
    skipped_bbox = 0
    grouped = list(segments.groupby("traj_id", sort=False))

    for idx, (traj_id, traj) in enumerate(grouped, 1):
        regular = resample_trajectory(traj, resample_minutes, max_gap_steps)
        if len(regular) < total_steps:
            continue

        possible_starts = np.arange(0, len(regular) - total_steps + 1)
        if max_windows_per_traj and len(possible_starts) > max_windows_per_traj:
            possible_starts = np.sort(
                rng.choice(possible_starts, size=max_windows_per_traj, replace=False)
            )

        feat = regular[WINDOW_FEATURE_COLS].to_numpy(dtype=np.float32)
        tgt  = regular[TARGET_COLS].to_numpy(dtype=np.float32)
        ts   = regular["timestamp"].to_numpy()
        mmsi_val = int(regular["mmsi"].iloc[0])

        for si in possible_starts:
            hist = feat[si : si + history_steps]
            fut  = tgt[si + history_steps : si + total_steps]

            if strict_bbox_future and lat_range is not None and lon_range is not None:
                if (
                    fut[:, 0].min() < lat_range[0] or fut[:, 0].max() > lat_range[1]
                    or fut[:, 1].min() < lon_range[0] or fut[:, 1].max() > lon_range[1]
                ):
                    skipped_bbox += 1
                    continue

            hist_list.append(hist.copy())
            fut_list.append(fut.copy())
            meta["traj_id"].append(traj_id)
            meta["mmsi"].append(mmsi_val)
            meta["start_time"].append(ts[si])
            meta["split_time"].append(ts[si + history_steps - 1])
            meta["target_end_time"].append(ts[si + total_steps - 1])
            meta["history_steps"].append(history_steps)
            meta["future_steps"].append(future_steps)
            meta["resample_minutes"].append(resample_minutes)

        if idx % 200 == 0 or idx == len(grouped):
            print(
                f"\r  Trajectories {idx}/{len(grouped)} | windows {len(hist_list):,} | "
                f"skipped_bbox {skipped_bbox:,}",
                end="",
                flush=True,
            )

    print()
    elapsed = format_duration(time.perf_counter() - t0)
    n = len(hist_list)
    print(
        f"  Done: {n:,} windows | {skipped_bbox:,} skipped (exited bbox) | {elapsed}",
        flush=True,
    )
    if n == 0:
        return pd.DataFrame()

    # Stack → flatten → build DataFrame in one pass (avoids per-column copies)
    hist_arr = np.stack(hist_list)          # (n, history_steps, n_feat)
    del hist_list
    fut_arr  = np.stack(fut_list)           # (n, future_steps, 2)
    del fut_list

    hist_cols = [f"x_t{t:03d}_{col}" for t in range(history_steps) for col in WINDOW_FEATURE_COLS]
    fut_cols  = [f"y_t{t:03d}_{coord}" for t in range(future_steps) for coord in ("lat", "lon")]

    data = np.hstack([
        hist_arr.reshape(n, history_steps * n_feat),
        fut_arr.reshape(n, future_steps * 2),
    ])
    del hist_arr, fut_arr

    df = pd.DataFrame(data, columns=hist_cols + fut_cols)
    for k, v in meta.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# Per-dataset entry point
# ---------------------------------------------------------------------------

def rebuild_one(
    segments_path: Path,
    *,
    history_hours: float,
    future_hours: float,
    resample_minutes: int,
    max_windows_per_traj: int | None,
    max_gap_steps: int,
    overwrite: bool,
    strict_bbox_future: bool = False,
) -> None:
    windows_path = segments_path.parent / "model_ready_windows.parquet"

    if windows_path.exists() and not overwrite:
        print(f"  SKIP (windows already exist — use --all to overwrite): {windows_path}")
        return

    bbox = get_bbox(segments_path)
    lat_range = bbox[0] if bbox else None
    lon_range = bbox[1] if bbox else None

    print(f"\n=== {segments_path.parent.name} ===")
    print(f"  segments: {segments_path}", flush=True)

    segments = pd.read_parquet(segments_path)
    segments = patch_missing_motion_cols(segments)
    n_traj = segments["traj_id"].nunique() if "traj_id" in segments.columns else "?"
    print(f"  rows: {len(segments):,} | trajectories: {n_traj}", flush=True)

    windows = build_windows(
        segments,
        history_hours=history_hours,
        future_hours=future_hours,
        resample_minutes=resample_minutes,
        max_windows_per_traj=max_windows_per_traj,
        lat_range=lat_range,
        lon_range=lon_range,
        max_gap_steps=max_gap_steps,
        strict_bbox_future=strict_bbox_future,
    )

    if windows.empty:
        print(
            f"  WARNING: 0 windows generated. Segments probably too short for "
            f"{history_hours}h + {future_hours}h = {history_hours + future_hours}h window.",
            flush=True,
        )
        return

    windows.to_parquet(windows_path, index=False)
    print(f"  Saved: {windows_path} ({len(windows):,} windows)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild model_ready_windows.parquet from existing coastal_segments.parquet.\n"
            "Run from the project root directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--segments",
        default=None,
        help="Path to one coastal_segments.parquet (default: all under --data-root).",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed",
        help="Root directory to search for coastal_segments.parquet files.",
    )
    parser.add_argument(
        "--all",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing model_ready_windows.parquet files.",
    )
    parser.add_argument("--history-hours", type=float, default=24.0)
    parser.add_argument("--future-hours", type=float, default=12.0)
    parser.add_argument("--resample-minutes", type=int, default=10)
    parser.add_argument(
        "--max-windows-per-traj",
        type=int,
        default=200,
        help="Cap windows per trajectory (0 = no cap).",
    )
    parser.add_argument(
        "--max-gap-steps",
        type=int,
        default=6,
        help="Max consecutive interpolated steps in resampled grid (default 6 = 1h at 10-min).",
    )
    parser.add_argument(
        "--strict-bbox-future",
        action="store_true",
        default=False,
        help=(
            "Skip windows where any future (predicted) position exits the region bounding box. "
            "Off by default — vessels may legitimately depart the area, and the model should "
            "learn to predict where they go. Enable only for strictly in-frame training sets."
        ),
    )
    args = parser.parse_args()

    max_w = args.max_windows_per_traj if args.max_windows_per_traj > 0 else None

    kwargs = dict(
        history_hours=args.history_hours,
        future_hours=args.future_hours,
        resample_minutes=args.resample_minutes,
        max_windows_per_traj=max_w,
        max_gap_steps=args.max_gap_steps,
        overwrite=args.overwrite,
        strict_bbox_future=args.strict_bbox_future,
    )

    if args.segments:
        rebuild_one(Path(args.segments), **kwargs)
    else:
        data_root = Path(args.data_root)
        segment_files = sorted(data_root.rglob("coastal_segments.parquet"))
        if not segment_files:
            print(f"No coastal_segments.parquet found under {data_root}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(segment_files)} segment file(s) under {data_root}")
        overall_start = time.perf_counter()

        for seg_path in segment_files:
            rebuild_one(seg_path, **kwargs)

        print(f"\nAll done in {format_duration(time.perf_counter() - overall_start)}.")


if __name__ == "__main__":
    main()