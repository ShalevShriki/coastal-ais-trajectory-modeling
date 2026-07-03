from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from proj.project.coast_paths import COAST_CONFIGS, parse_dataset_folder, region_from_dataset_path


@dataclass(frozen=True)
class FrameBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_tuple(cls, bounds: tuple[float, float, float, float]) -> FrameBounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        return cls(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    @classmethod
    def from_dict(cls, bounds: dict[str, float]) -> FrameBounds:
        return cls(
            lat_min=bounds["south"],
            lat_max=bounds["north"],
            lon_min=bounds["west"],
            lon_max=bounds["east"],
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


def bounds_for_coast_region(coast_name: str, region: str) -> FrameBounds:
    coast = COAST_CONFIGS[coast_name]
    if region not in coast.regions:
        raise KeyError(f"Region {region!r} not defined for coast {coast_name!r}.")
    return FrameBounds.from_tuple(coast.regions[region])


def bounds_for_segments_path(segments_path: Path, coast_name: str) -> FrameBounds:
    region = region_from_dataset_path(segments_path)
    if region is None:
        raise ValueError(f"Could not infer region from path: {segments_path}")
    return bounds_for_coast_region(coast_name, region)


def points_in_frame(
    df: pd.DataFrame,
    bounds: FrameBounds,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.Series:
    return (
        (df[lat_col] >= bounds.lat_min)
        & (df[lat_col] <= bounds.lat_max)
        & (df[lon_col] >= bounds.lon_min)
        & (df[lon_col] <= bounds.lon_max)
    )


def discover_segments_path(coast_name: str, dataset_tag: str | None = None) -> Path:
    coast = COAST_CONFIGS[coast_name]
    tag_part = f"_{dataset_tag}" if dataset_tag else ""
    candidate = (
        coast.processed_root
        / f"ais_{coast.default_region}{tag_part}_long_horizon"
        / "coastal_segments.parquet"
    )
    if candidate.exists():
        return candidate

    pattern = f"ais_{coast.default_region}*{tag_part or ''}*_long_horizon"
    matches = sorted(coast.processed_root.glob(f"{pattern}/coastal_segments.parquet"))
    if matches:
        return matches[-1]

    available = sorted(coast.processed_root.glob("ais_*_long_horizon/coastal_segments.parquet"))
    hint = "\n".join(f"  - {path.parent}" for path in available) or "  (none)"
    raise FileNotFoundError(
        f"No segments file for {coast_name} (tag={dataset_tag!r}).\n"
        f"Looked for: {candidate}\nAvailable:\n{hint}"
    )


def dataset_label_from_path(segments_path: Path) -> str:
    _, suffix = parse_dataset_folder(segments_path.parent.name)
    if suffix:
        day_match = __import__("re").fullmatch(r"(\d+)d", suffix)
        if day_match:
            return f"{day_match.group(1)} dey"
        return suffix
    return "unknown"


def segment_frame_stats(
    segments_path: Path,
    bounds: FrameBounds,
    *,
    strict_frame: bool = False,
) -> pd.DataFrame:
    """Per traj_id: wall duration vs hours accumulated only at in-frame AIS reports."""
    schema_names = pq.read_schema(segments_path).names
    cols = [
        c
        for c in (
            "traj_id",
            "mmsi",
            "timestamp",
            "lat",
            "lon",
            "dt_sec",
            "dist_km",
            "sog",
            "vessel_type",
        )
        if c in schema_names
    ]
    df = pd.read_parquet(segments_path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    inside = points_in_frame(df, bounds)
    df["in_frame"] = inside
    if strict_frame:
        df = df[inside].copy()
        inside = pd.Series(True, index=df.index)

    df = df.sort_values(["traj_id", "timestamp"])
    dt_seg = df.groupby("traj_id")["timestamp"].diff().dt.total_seconds()
    df["in_frame_dt_sec"] = np.where(
        df["in_frame"] & dt_seg.notna() & (dt_seg > 0),
        dt_seg,
        0.0,
    )

    stats = (
        df.groupby("traj_id")
        .agg(
            mmsi=("mmsi", "first"),
            n_points=("timestamp", "size"),
            n_in_frame=("in_frame", "sum"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            in_frame_hours=("in_frame_dt_sec", lambda s: s.sum() / 3600.0),
            **(
                {"total_distance_km": ("dist_km", "sum")}
                if "dist_km" in df.columns
                else {}
            ),
            **({"mean_sog_kmh": ("sog", "mean")} if "sog" in df.columns else {}),
        )
        .reset_index()
    )
    stats["duration_hours"] = (
        stats["end_time"] - stats["start_time"]
    ).dt.total_seconds() / 3600.0
    stats["pct_points_in_frame"] = stats["n_in_frame"] / stats["n_points"].replace(0, np.nan)
    stats["pct_time_in_frame"] = stats["in_frame_hours"] / stats["duration_hours"].replace(0, np.nan)
    stats["n_out_of_frame"] = stats["n_points"] - stats["n_in_frame"]
    return stats


def vessel_in_frame_hours(stats: pd.DataFrame) -> pd.DataFrame:
    """Longest in-frame segment hours and totals per MMSI."""
    vessel = (
        stats.groupby("mmsi")
        .agg(
            n_segments=("traj_id", "nunique"),
            longest_in_frame_hours=("in_frame_hours", "max"),
            longest_duration_hours=("duration_hours", "max"),
            total_in_frame_hours=("in_frame_hours", "sum"),
            total_duration_hours=("duration_hours", "sum"),
            min_pct_points_in_frame=("pct_points_in_frame", "min"),
            mean_pct_points_in_frame=("pct_points_in_frame", "mean"),
            n_out_of_frame_points=("n_out_of_frame", "sum"),
        )
        .reset_index()
    )
    vessel["fully_in_frame"] = vessel["min_pct_points_in_frame"] >= 1.0
    return vessel
