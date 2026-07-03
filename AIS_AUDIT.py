from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def group_col(df: pd.DataFrame) -> str:
    return "traj_id" if "traj_id" in df.columns else "mmsi"


def q(series: pd.Series, vals=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> pd.Series:
    return series.dropna().quantile(vals)


def possible_windows(duration_hours: pd.Series, history_hours: float, future_hours: float, resample_minutes: int) -> pd.Series:
    # After resampling to a regular grid, an approximate number of sliding windows per trajectory.
    total_hours = history_hours + future_hours
    step_hours = resample_minutes / 60.0
    return np.floor((duration_hours - total_hours) / step_hours + 1).clip(lower=0).astype(int)


def save_hist(series: pd.Series, out: Path, title: str, xlabel: str, bins: int = 50, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def audit_segments(path: Path, output_dir: Path, history_hours: float, future_hours: float, resample_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    g = group_col(df)

    # Per trajectory / continuous segment statistics.
    stats = df.groupby(g).agg(
        mmsi=("mmsi", "first") if "mmsi" in df.columns else (g, "first"),
        n_points=("timestamp", "size"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        median_dt_sec=("dt_sec", "median") if "dt_sec" in df.columns else ("timestamp", lambda s: np.nan),
        mean_dt_sec=("dt_sec", "mean") if "dt_sec" in df.columns else ("timestamp", lambda s: np.nan),
        p95_dt_sec=("dt_sec", lambda s: s.quantile(0.95)) if "dt_sec" in df.columns else ("timestamp", lambda s: np.nan),
        total_distance_km=("dist_km", "sum") if "dist_km" in df.columns else ("timestamp", lambda s: np.nan),
        mean_sog_kmh=("sog", "mean") if "sog" in df.columns else ("timestamp", lambda s: np.nan),
    ).reset_index().rename(columns={g: "traj_id"})

    stats["duration_hours"] = (stats["end_time"] - stats["start_time"]).dt.total_seconds() / 3600.0
    stats["duration_days"] = stats["duration_hours"] / 24.0
    stats["points_per_hour"] = stats["n_points"] / stats["duration_hours"].replace(0, np.nan)
    stats["possible_windows"] = possible_windows(stats["duration_hours"], history_hours, future_hours, resample_minutes)

    # Per vessel statistics: how much tracking total does each MMSI contribute?
    vessel = stats.groupby("mmsi").agg(
        n_segments=("traj_id", "nunique"),
        total_points=("n_points", "sum"),
        total_observed_hours_sum_segments=("duration_hours", "sum"),
        longest_segment_hours=("duration_hours", "max"),
        first_seen=("start_time", "min"),
        last_seen=("end_time", "max"),
        total_possible_windows=("possible_windows", "sum"),
        total_distance_km=("total_distance_km", "sum"),
    ).reset_index()
    vessel["calendar_span_hours"] = (vessel["last_seen"] - vessel["first_seen"]).dt.total_seconds() / 3600.0

    # Global summary text.
    summary_lines = []
    summary_lines.append(f"segments file: {path}")
    summary_lines.append(f"rows / AIS reports: {len(df):,}")
    summary_lines.append(f"unique MMSI vessels: {df['mmsi'].nunique():,}" if "mmsi" in df.columns else "unique MMSI vessels: unknown")
    summary_lines.append(f"continuous trajectories/segments: {len(stats):,}")
    summary_lines.append(f"time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    summary_lines.append("")
    summary_lines.append("Trajectory duration quantiles [hours]:")
    summary_lines.append(str(q(stats["duration_hours"])))
    summary_lines.append("")
    summary_lines.append("Points per trajectory quantiles:")
    summary_lines.append(str(q(stats["n_points"])))
    summary_lines.append("")
    summary_lines.append("Possible sliding-window samples per trajectory quantiles:")
    summary_lines.append(str(q(stats["possible_windows"])))
    summary_lines.append("")
    summary_lines.append(f"Total possible windows before cap: {int(stats['possible_windows'].sum()):,}")
    summary_lines.append(f"Trajectories with >=1 window for {history_hours}h->{future_hours}h: {(stats['possible_windows'] > 0).sum():,} / {len(stats):,}")
    summary_lines.append("")
    summary_lines.append("Top 20 longest trajectories:")
    summary_lines.append(str(stats.sort_values("duration_hours", ascending=False).head(20)))
    summary_lines.append("")
    summary_lines.append("Top 20 vessels by total observed segment-hours:")
    summary_lines.append(str(vessel.sort_values("total_observed_hours_sum_segments", ascending=False).head(20)))

    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "trajectory_audit.csv", index=False)
    vessel.to_csv(output_dir / "vessel_audit.csv", index=False)
    (output_dir / "audit_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    save_hist(stats["duration_hours"], output_dir / "hist_duration_hours.png", "Continuous trajectory duration", "hours", logy=True)
    save_hist(stats["n_points"], output_dir / "hist_points_per_trajectory.png", "AIS reports per trajectory", "points", logy=True)
    save_hist(stats["possible_windows"], output_dir / "hist_possible_windows.png", "Possible model windows per trajectory", "windows", logy=True)

    return stats, vessel


def audit_windows(path: Path, output_dir: Path) -> None:
    if not path.exists():
        return
    w = pd.read_parquet(path, columns=[c for c in pd.read_parquet(path, nrows=0).columns if c in {"traj_id", "mmsi", "start_time", "split_time", "target_end_time", "history_steps", "future_steps", "resample_minutes"}])
    w["start_time"] = pd.to_datetime(w["start_time"], errors="coerce")
    w["target_end_time"] = pd.to_datetime(w["target_end_time"], errors="coerce")
    by_traj = w.groupby("traj_id").agg(
        mmsi=("mmsi", "first"),
        n_windows=("traj_id", "size"),
        first_window_start=("start_time", "min"),
        last_window_end=("target_end_time", "max"),
    ).reset_index()
    by_mmsi = by_traj.groupby("mmsi").agg(
        n_traj_with_windows=("traj_id", "nunique"),
        n_windows=("n_windows", "sum"),
    ).reset_index().sort_values("n_windows", ascending=False)

    by_traj.to_csv(output_dir / "windows_by_trajectory.csv", index=False)
    by_mmsi.to_csv(output_dir / "windows_by_vessel.csv", index=False)
    save_hist(by_traj["n_windows"], output_dir / "hist_actual_windows_per_trajectory.png", "Actual saved windows per trajectory", "windows", logy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed AIS data coverage and sample yield.")
    parser.add_argument("--segments", required=True, help="Path to coastal_segments.parquet")
    parser.add_argument("--windows", default=None, help="Optional path to model_ready_windows.parquet")
    parser.add_argument("--out", default=None, help="Output folder. Default: <segments_parent>/audit")
    parser.add_argument("--history-hours", type=float, default=6.0)
    parser.add_argument("--future-hours", type=float, default=5.0)
    parser.add_argument("--resample-minutes", type=int, default=10)
    args = parser.parse_args()

    segments_path = Path(args.segments)
    out = Path(args.out) if args.out else segments_path.parent / "audit"
    audit_segments(segments_path, out, args.history_hours, args.future_hours, args.resample_minutes)

    windows_path = Path(args.windows) if args.windows else segments_path.parent / "model_ready_windows.parquet"
    audit_windows(windows_path, out)

    print(f"Saved audit outputs to: {out}")
    print(f"Read: {out / 'audit_summary.txt'}")


if __name__ == "__main__":
    main()
