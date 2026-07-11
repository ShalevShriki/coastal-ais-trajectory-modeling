from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from proj.project.coast_paths import COAST_CONFIGS, resolve_windows_path

WINDOW_FEATURE_COLS = [
    "lat",
    "lon",
    "sog",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "heading_missing",
    "dt_sec",
    "dlat",
    "dlon",
    "dsog",
    "dcog",
    "v_north_kmh",
    "v_east_kmh",
]

FEATURE_COLS = WINDOW_FEATURE_COLS

# Backward-compatible aliases when parquet was built with older column names.
FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "v_north_kmh": ("v_north_knots", "v_north_kmh"),
    "v_east_kmh": ("v_east_knots", "v_east_kmh"),
    "v_north_knots": ("v_north_knots", "v_north_kmh"),
    "v_east_knots": ("v_east_knots", "v_east_kmh"),
}


def infer_feature_cols(df: pd.DataFrame, step: int = 0) -> list[str]:
    """Read feature names from parquet window columns (handles legacy naming)."""
    prefix = f"x_t{step:03d}_"
    available = [col[len(prefix) :] for col in df.columns if col.startswith(prefix)]
    if not available:
        raise ValueError(f"No history feature columns found with prefix {prefix!r}.")

    resolved: list[str] = []
    seen: set[str] = set()

    for canonical in WINDOW_FEATURE_COLS:
        candidates = FEATURE_ALIASES.get(canonical, (canonical,))
        for name in candidates:
            if name in available and name not in seen:
                resolved.append(name)
                seen.add(name)
                break

    for name in available:
        if name not in seen:
            resolved.append(name)
            seen.add(name)

    return resolved


def feature_index(feature_cols: list[str], name: str) -> int:
    candidates = FEATURE_ALIASES.get(name, (name,))
    for candidate in candidates:
        if candidate in feature_cols:
            return feature_cols.index(candidate)
    raise KeyError(f"Feature {name!r} not found in {feature_cols}")


def default_window_path(
    region: str = "east_coast",
    coast: str | None = None,
) -> Path:
    _, coast_config, _ = resolve_windows_path(coast, region, None)
    return (
        coast_config.processed_root
        / f"ais_{region}_long_horizon"
        / "model_ready_windows.parquet"
    )


def load_windows(
    path: Path,
    sample_size: int | None = None,
    *,
    maneuver_oversample: bool = False,
    maneuver_fraction: float = 0.3,
    motion_balanced_sample: bool = False,
    straight_fraction: float = 0.15,
    other_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run PROCESS_noaa_long_coastal.py or INCREMENTAL_PROCESS.py first."
        )
    if sample_size is None:
        return pd.read_parquet(path)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows

    if total_rows <= sample_size:
        return pd.read_parquet(path)

    # Read only enough row groups to cover sample_size rows.
    # Convert each group to pandas immediately so the PyArrow table is released
    # before the next group is read — avoids holding multiple large tables in RAM.
    num_groups = pf.metadata.num_row_groups
    frac = sample_size / total_rows
    n_groups = max(1, min(num_groups, int(np.ceil(num_groups * frac))))
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(num_groups, size=n_groups, replace=False).tolist())

    frames: list[pd.DataFrame] = []
    for g in chosen:
        frames.append(pf.read_row_group(g).to_pandas())

    df = pd.concat(frames, ignore_index=True)
    del frames

    if len(df) > sample_size:
        if motion_balanced_sample:
            df = motion_stratified_sample(
                df,
                sample_size,
                straight_fraction=straight_fraction,
                other_fraction=other_fraction,
                seed=seed,
            )
        elif maneuver_oversample:
            df = maneuver_balanced_sample(
                df, sample_size, maneuver_fraction=maneuver_fraction, seed=seed
            )
        else:
            df = df.sample(sample_size, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Stationary / confined-window filter
# ---------------------------------------------------------------------------

@dataclass
class StationaryFilterConfig:
    """Drop windows where the vessel barely leaves a small geographic cell."""

    enabled: bool = True
    # Use only history features for filtering (no future lat/lon/SOG leakage).
    history_only: bool = True
    # Max distance from anchor to any history point (the "cell" radius).
    max_confined_radius_km: float = 0.5
    # Net displacement over history (history_only) or future (legacy mode).
    min_displacement_km: float = 1.0
    # Optional: mean SOG below this (knots) reinforces stationary classification.
    min_mean_sog_kn: float = 0.5
    # Require meaningful history motion (net START→END of 24h history).
    require_min_history_displacement_km: float = 0.0
    # Require total history path length (filters micro-wiggles that pass net disp).
    require_min_history_path_km: float = 0.0
    # Drop local loops: net/path below this when path exceeds min_path_for_loop_km.
    max_history_loop_ratio: float | None = None
    min_path_for_loop_km: float = 10.0

    @property
    def min_future_displacement_km(self) -> float:
        """Backward-compatible alias used by older scripts."""
        return self.min_displacement_km


@dataclass
class SmartMotionFilterConfig:
    """
    History-only filter for vessels that are still moving into the forecast window.

    Catches "arrived and stopped" windows: high 24h displacement but the last 8h
    of history are nearly stationary (vessel reached destination before t=0).
  """

    enabled: bool = True
    # Net displacement over the last 16h of history (steps at 10 min).
    min_history_16h_net_km: float = 8.0
    # Net displacement over the last 8h of history ending at anchor.
    min_history_last_8h_net_km: float = 2.0
    # Drop small/local loops only (low net/path in a confined area).
    # Large loops (long path or wide radius, e.g. ferries) are kept.
    max_history_loop_ratio: float = 0.35
    min_path_for_loop_km: float = 10.0
    min_big_loop_path_km: float = 50.0
    max_local_loop_radius_km: float = 20.0
    # Audit thresholds (future motion, for reports only — not used to drop rows).
    audit_min_future_8h_net_km: float = 5.0
    audit_min_future_12h_net_km: float = 5.0


def _track_path_length_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if lat.shape[1] < 2:
        return np.zeros(lat.shape[0], dtype=np.float64)
    seg = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    return seg.sum(axis=1)


def _history_step_slices(history_steps: int) -> tuple[int, int, int]:
    """Return (start_16h, start_8h, anchor) step indices for 10-min windows."""
    anchor = history_steps - 1
    steps_16h = min(history_steps, int(round(16 * 60 / 10)))
    steps_8h = min(history_steps, int(round(8 * 60 / 10)))
    return max(0, anchor - steps_16h + 1), max(0, anchor - steps_8h + 1), anchor


def _future_horizon_step(future_steps: int, hours: float) -> int:
    step = int(round(hours * 60 / 10)) - 1
    return max(0, min(step, future_steps - 1))


def compute_smart_motion_metrics(
    df: pd.DataFrame,
    history_steps: int | None = None,
    future_steps: int | None = None,
) -> pd.DataFrame:
    """
    History + future motion stats for smart filtering and audit.

    Filter uses history columns only. Future columns are for offline audit.
    """
    if history_steps is None or future_steps is None:
        history_steps, future_steps = infer_window_shape(df)

    lat_hist = df[[f"x_t{t:03d}_lat" for t in range(history_steps)]].to_numpy(dtype=np.float64)
    lon_hist = df[[f"x_t{t:03d}_lon" for t in range(history_steps)]].to_numpy(dtype=np.float64)
    lat_fut = df[[f"y_t{t:03d}_lat" for t in range(future_steps)]].to_numpy(dtype=np.float64)
    lon_fut = df[[f"y_t{t:03d}_lon" for t in range(future_steps)]].to_numpy(dtype=np.float64)

    i16, i8, anchor = _history_step_slices(history_steps)
    anchor_lat = lat_hist[:, anchor]
    anchor_lon = lon_hist[:, anchor]

    history_full_net_km = haversine_km(
        lat_hist[:, 0], lon_hist[:, 0], anchor_lat, anchor_lon
    )
    history_16h_net_km = haversine_km(
        lat_hist[:, i16], lon_hist[:, i16], anchor_lat, anchor_lon
    )
    history_last_8h_net_km = haversine_km(
        lat_hist[:, i8], lon_hist[:, i8], anchor_lat, anchor_lon
    )
    history_path_km = _track_path_length_km(lat_hist, lon_hist)
    dist_from_anchor = haversine_km(
        anchor_lat[:, np.newaxis],
        anchor_lon[:, np.newaxis],
        lat_hist,
        lon_hist,
    )
    history_max_radius_km = dist_from_anchor.max(axis=1)

    step_8h = _future_horizon_step(future_steps, 8.0)
    step_12h = _future_horizon_step(future_steps, 12.0)
    future_8h_net_km = haversine_km(
        anchor_lat, anchor_lon, lat_fut[:, step_8h], lon_fut[:, step_8h]
    )
    future_12h_net_km = haversine_km(
        anchor_lat, anchor_lon, lat_fut[:, step_12h], lon_fut[:, step_12h]
    )

    loop_ratio = np.where(
        history_path_km > 10.0,
        history_full_net_km / np.maximum(history_path_km, 1e-6),
        1.0,
    )
    arrived_then_stopped = (history_full_net_km >= 20.0) & (history_last_8h_net_km < 3.0)

    return pd.DataFrame(
        {
            "history_full_net_km": history_full_net_km,
            "history_16h_net_km": history_16h_net_km,
            "history_last_8h_net_km": history_last_8h_net_km,
            "history_path_km": history_path_km,
            "history_max_radius_km": history_max_radius_km,
            "history_loop_ratio": loop_ratio,
            "arrived_then_stopped": arrived_then_stopped,
            "future_8h_net_km": future_8h_net_km,
            "future_12h_net_km": future_12h_net_km,
        }
    )


def smart_motion_window_mask(
    metrics: pd.DataFrame,
    config: SmartMotionFilterConfig,
) -> np.ndarray:
    """True = remove window (not suitable for trajectory forecasting)."""
    remove = (
        metrics["history_16h_net_km"] < config.min_history_16h_net_km
    ) | (metrics["history_last_8h_net_km"] < config.min_history_last_8h_net_km)

    big_loop = (
        (metrics["history_path_km"] >= config.min_big_loop_path_km)
        | (metrics["history_max_radius_km"] >= config.max_local_loop_radius_km)
    )
    local_loop = (
        (metrics["history_path_km"] > config.min_path_for_loop_km)
        & (metrics["history_loop_ratio"] < config.max_history_loop_ratio)
        & ~big_loop
    )
    remove |= local_loop
    return remove.to_numpy()


def smart_motion_keep_mask(
    metrics: pd.DataFrame,
    config: SmartMotionFilterConfig,
) -> np.ndarray:
    if not config.enabled:
        return np.ones(len(metrics), dtype=bool)
    return ~smart_motion_window_mask(metrics, config)

    if lat.shape[1] < 2:
        return np.zeros(lat.shape[0], dtype=np.float64)
    seg = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    return seg.sum(axis=1)


def compute_history_motion_metrics(
    df: pd.DataFrame,
    history_steps: int | None = None,
) -> pd.DataFrame:
    """Motion stats from the history window only (safe for training filters)."""
    if history_steps is None:
        history_steps, _ = infer_window_shape(df)

    lat_hist = df[[f"x_t{t:03d}_lat" for t in range(history_steps)]].to_numpy(dtype=np.float64)
    lon_hist = df[[f"x_t{t:03d}_lon" for t in range(history_steps)]].to_numpy(dtype=np.float64)
    anchor_lat = lat_hist[:, -1]
    anchor_lon = lon_hist[:, -1]

    dist_from_anchor = haversine_km(
        anchor_lat[:, np.newaxis],
        anchor_lon[:, np.newaxis],
        lat_hist,
        lon_hist,
    )
    history_max_radius_km = dist_from_anchor.max(axis=1)
    history_displacement_km = haversine_km(
        lat_hist[:, 0], lon_hist[:, 0], anchor_lat, anchor_lon
    )
    history_path_length_km = _track_path_length_km(lat_hist, lon_hist)

    sog_cols = [f"x_t{t:03d}_sog" for t in range(history_steps)]
    if all(c in df.columns for c in sog_cols):
        history_mean_sog_kn = df[sog_cols].to_numpy(dtype=np.float64).mean(axis=1)
    else:
        history_mean_sog_kn = np.full(len(df), np.nan, dtype=np.float64)

    return pd.DataFrame(
        {
            "history_max_radius_km": history_max_radius_km,
            "history_displacement_km": history_displacement_km,
            "history_path_length_km": history_path_length_km,
            "history_mean_sog_kn": history_mean_sog_kn,
        }
    )


def compute_window_motion_metrics(
    df: pd.DataFrame,
    history_steps: int | None = None,
    future_steps: int | None = None,
) -> pd.DataFrame:
    """
  Per-window motion stats from history + future lat/lon (and SOG when present).

    Confined windows: high path_length with tiny max_radius = wiggling in place.
    """
    if history_steps is None or future_steps is None:
        history_steps, future_steps = infer_window_shape(df)

    lat_hist = df[[f"x_t{t:03d}_lat" for t in range(history_steps)]].to_numpy(dtype=np.float64)
    lon_hist = df[[f"x_t{t:03d}_lon" for t in range(history_steps)]].to_numpy(dtype=np.float64)
    lat_fut = df[[f"y_t{t:03d}_lat" for t in range(future_steps)]].to_numpy(dtype=np.float64)
    lon_fut = df[[f"y_t{t:03d}_lon" for t in range(future_steps)]].to_numpy(dtype=np.float64)

    anchor_lat = lat_hist[:, -1]
    anchor_lon = lon_hist[:, -1]
    all_lat = np.concatenate([lat_hist, lat_fut], axis=1)
    all_lon = np.concatenate([lon_hist, lon_fut], axis=1)

    dist_from_anchor = haversine_km(
        anchor_lat[:, np.newaxis],
        anchor_lon[:, np.newaxis],
        all_lat,
        all_lon,
    )
    max_radius_km = dist_from_anchor.max(axis=1)
    future_displacement_km = haversine_km(
        anchor_lat, anchor_lon, lat_fut[:, -1], lon_fut[:, -1]
    )
    history_displacement_km = haversine_km(
        lat_hist[:, 0], lon_hist[:, 0], anchor_lat, anchor_lon
    )
    path_length_km = _track_path_length_km(all_lat, all_lon)

    sog_cols = [f"x_t{t:03d}_sog" for t in range(history_steps)]
    sog_cols += [f"y_t{t:03d}_sog" for t in range(future_steps)]
    if all(c in df.columns for c in sog_cols):
        mean_sog_kn = df[sog_cols].to_numpy(dtype=np.float64).mean(axis=1)
    else:
        mean_sog_kn = np.full(len(df), np.nan, dtype=np.float64)

    return pd.DataFrame(
        {
            "max_radius_km": max_radius_km,
            "future_displacement_km": future_displacement_km,
            "history_displacement_km": history_displacement_km,
            "path_length_km": path_length_km,
            "mean_sog_kn": mean_sog_kn,
        }
    )


def stationary_window_mask(
    metrics: pd.DataFrame,
    config: StationaryFilterConfig,
) -> np.ndarray:
    """
    True = boring / confined window (candidate for removal).

    When config.history_only is True (default), uses only history motion stats.
    """
    if config.history_only:
        radius_col = "history_max_radius_km"
        disp_col = "history_displacement_km"
        sog_col = "history_mean_sog_kn"
        path_col = "history_path_length_km"
    else:
        radius_col = "max_radius_km"
        disp_col = "future_displacement_km"
        sog_col = "mean_sog_kn"
        path_col = "path_length_km"

    confined = metrics[radius_col] <= config.max_confined_radius_km
    short_motion = metrics[disp_col] < config.min_displacement_km
    primary = confined & short_motion

    if sog_col in metrics.columns and np.isfinite(metrics[sog_col]).any():
        slow = metrics[sog_col] < config.min_mean_sog_kn
        secondary = slow & short_motion & confined
        remove = (primary | secondary).to_numpy()
    else:
        remove = primary.to_numpy()

    if config.require_min_history_displacement_km > 0 and disp_col in metrics.columns:
        remove |= (
            metrics[disp_col] < config.require_min_history_displacement_km
        ).to_numpy()

    if config.require_min_history_path_km > 0 and path_col in metrics.columns:
        remove |= (
            metrics[path_col] < config.require_min_history_path_km
        ).to_numpy()

    if (
        config.max_history_loop_ratio is not None
        and path_col in metrics.columns
        and disp_col in metrics.columns
    ):
        path = metrics[path_col].to_numpy(dtype=np.float64)
        net = metrics[disp_col].to_numpy(dtype=np.float64)
        loop_ratio = np.where(path > config.min_path_for_loop_km, net / np.maximum(path, 1e-6), 1.0)
        remove |= (
            (path > config.min_path_for_loop_km)
            & (loop_ratio < config.max_history_loop_ratio)
        )

    return remove


def filter_stationary_windows(
    df: pd.DataFrame,
    config: StationaryFilterConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not config.enabled or df.empty:
        return df, {"removed": 0.0, "kept_fraction": 1.0}

    metrics = (
        compute_history_motion_metrics(df)
        if config.history_only
        else compute_window_motion_metrics(df)
    )
    remove = stationary_window_mask(metrics, config)
    n_removed = int(remove.sum())
    n_total = len(df)

    radius_col = "history_max_radius_km" if config.history_only else "max_radius_km"
    stats = {
        "windows_total": float(n_total),
        "windows_removed": float(n_removed),
        "windows_kept": float(n_total - n_removed),
        "removed_fraction": float(n_removed / max(n_total, 1)),
        "kept_fraction": float((n_total - n_removed) / max(n_total, 1)),
        "history_only": float(config.history_only),
        "median_max_radius_km_removed": float(metrics.loc[remove, radius_col].median())
        if n_removed
        else 0.0,
        "median_max_radius_km_kept": float(metrics.loc[~remove, radius_col].median())
        if n_removed < n_total
        else 0.0,
    }

    if n_removed == 0:
        return df, stats

    return df.loc[~remove].reset_index(drop=True), stats


def print_stationary_filter_stats(stats: dict[str, float], config: StationaryFilterConfig) -> None:
    scope = "history-only" if config.history_only else "history+future"
    print(
        f"Stationary filter ({scope}): radius≤{config.max_confined_radius_km:.2f} km & "
        f"disp<{config.min_displacement_km:.2f} km "
        f"(SOG<{config.min_mean_sog_kn:.1f} kn reinforces)"
    )
    if config.require_min_history_displacement_km > 0:
        print(f"  + require history net displacement ≥ {config.require_min_history_displacement_km:.1f} km")
    if config.require_min_history_path_km > 0:
        print(f"  + require history path length ≥ {config.require_min_history_path_km:.1f} km")
    if config.max_history_loop_ratio is not None:
        print(
            f"  + drop history loops with net/path < {config.max_history_loop_ratio:.2f} "
            f"(when path > {config.min_path_for_loop_km:.1f} km)"
        )
    print(
        f"  removed {int(stats['windows_removed']):,} / {int(stats['windows_total']):,} "
        f"({stats['removed_fraction'] * 100:.1f}%) | "
        f"kept {int(stats['windows_kept']):,}"
    )


def add_stationary_filter_args(parser) -> None:
    parser.add_argument(
        "--filter-stationary",
        action="store_true",
        help="Drop confined/low-motion windows (wiggling in place near a dock).",
    )
    parser.add_argument(
        "--max-confined-radius-km",
        type=float,
        default=0.5,
        help="Max radius from anchor to classify as confined (default: 0.5 km).",
    )
    parser.add_argument(
        "--min-future-displacement-km",
        type=float,
        default=1.0,
        help="Min net displacement to keep a confined window (default: 1.0 km).",
    )
    parser.add_argument(
        "--filter-use-future",
        action="store_true",
        help="Legacy: allow future trajectory in stationary filter (not recommended).",
    )
    parser.add_argument(
        "--min-mean-sog-kn",
        type=float,
        default=0.5,
        help="Mean SOG below this reinforces stationary removal (default: 0.5 kn).",
    )
    parser.add_argument(
        "--require-min-history-displacement-km",
        type=float,
        default=0.0,
        help="Drop windows whose 24h history net displacement is below this (history-only, no label leak).",
    )
    parser.add_argument(
        "--require-min-history-path-km",
        type=float,
        default=0.0,
        help="Drop windows whose 24h history path length is below this.",
    )
    parser.add_argument(
        "--max-history-loop-ratio",
        type=float,
        default=None,
        help="Drop history loops where net/path is below this (e.g. 0.35 for ferries).",
    )
    parser.add_argument(
        "--min-path-for-loop-km",
        type=float,
        default=10.0,
        help="Only apply loop-ratio filter when history path exceeds this (default: 10 km).",
    )


def stationary_filter_from_args(args) -> StationaryFilterConfig:
    return StationaryFilterConfig(
        enabled=getattr(args, "filter_stationary", False),
        history_only=not getattr(args, "filter_use_future", False),
        max_confined_radius_km=getattr(args, "max_confined_radius_km", 0.5),
        min_displacement_km=getattr(
            args,
            "min_future_displacement_km",
            getattr(args, "min_displacement_km", 1.0),
        ),
        min_mean_sog_kn=getattr(args, "min_mean_sog_kn", 0.5),
        require_min_history_displacement_km=getattr(
            args, "require_min_history_displacement_km", 0.0
        ),
        require_min_history_path_km=getattr(args, "require_min_history_path_km", 0.0),
        max_history_loop_ratio=getattr(args, "max_history_loop_ratio", None),
        min_path_for_loop_km=getattr(args, "min_path_for_loop_km", 10.0),
    )


def load_windows_filtered(
    path: Path,
    sample_size: int | None = None,
    motion_filter: StationaryFilterConfig | None = None,
    *,
    maneuver_oversample: bool = False,
    maneuver_fraction: float = 0.3,
    motion_balanced_sample: bool = False,
    straight_fraction: float = 0.15,
    other_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    df = load_windows(
        path,
        sample_size=sample_size,
        maneuver_oversample=maneuver_oversample,
        maneuver_fraction=maneuver_fraction,
        motion_balanced_sample=motion_balanced_sample,
        straight_fraction=straight_fraction,
        other_fraction=other_fraction,
        seed=seed,
    )
    if motion_filter is not None and motion_filter.enabled:
        df, stats = filter_stationary_windows(df, motion_filter)
        print_stationary_filter_stats(stats, motion_filter)
    return df


def compute_maneuver_scores_df(df: pd.DataFrame, history_steps: int | None = None) -> np.ndarray:
    """Higher score = more course/speed changes in history."""
    if history_steps is None:
        history_steps, _ = infer_window_shape(df)

    dcog_cols = [f"x_t{t:03d}_dcog" for t in range(history_steps)]
    dsog_cols = [f"x_t{t:03d}_dsog" for t in range(history_steps)]
    score = np.zeros(len(df), dtype=np.float64)
    if all(c in df.columns for c in dcog_cols):
        score += np.abs(df[dcog_cols].to_numpy(dtype=np.float64)).mean(axis=1)
    if all(c in df.columns for c in dsog_cols):
        score += np.abs(df[dsog_cols].to_numpy(dtype=np.float64)).mean(axis=1)
    return score


def maneuver_balanced_sample(
    df: pd.DataFrame,
    sample_size: int,
    *,
    maneuver_fraction: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """70/30 style split: random + high-maneuver windows."""
    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    history_steps, _ = infer_window_shape(df)
    scores = compute_maneuver_scores_df(df, history_steps)
    rng = np.random.default_rng(seed)

    n_maneuver = int(round(sample_size * maneuver_fraction))
    n_random = sample_size - n_maneuver

    order = np.argsort(scores)
    maneuver_pool = order[-max(n_maneuver * 4, n_maneuver) :]
    random_pool = order[: max(len(order) - len(maneuver_pool), 1)]

    m_pick = rng.choice(
        maneuver_pool,
        size=min(n_maneuver, len(maneuver_pool)),
        replace=len(maneuver_pool) < n_maneuver,
    )
    r_pick = rng.choice(
        random_pool,
        size=min(n_random, len(random_pool)),
        replace=len(random_pool) < n_random,
    )
    idx = np.unique(np.concatenate([m_pick, r_pick]))
    if len(idx) < sample_size:
        remaining = np.setdiff1d(np.arange(len(df)), idx)
        extra = rng.choice(
            remaining,
            size=min(sample_size - len(idx), len(remaining)),
            replace=False,
        )
        idx = np.concatenate([idx, extra])
    return df.iloc[idx[:sample_size]].reset_index(drop=True)


def motion_bucket_masks_df(
    df: pd.DataFrame,
    history_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Classify windows from parquet columns (same buckets as classify_window_motion)."""
    if history_steps is None:
        history_steps, _ = infer_window_shape(df)

    sog_cols = [f"x_t{t:03d}_sog" for t in range(history_steps)]
    dcog_cols = [f"x_t{t:03d}_dcog" for t in range(history_steps)]
    mean_sog = df[sog_cols].to_numpy(dtype=np.float64).mean(axis=1)
    max_dcog = np.abs(df[dcog_cols].to_numpy(dtype=np.float64)).max(axis=1)

    anchored = mean_sog < 1.0
    maneuver = max_dcog > 15.0
    straight = (mean_sog >= 5.0) & (max_dcog < 5.0) & ~anchored
    other = ~(anchored | maneuver | straight)
    return {
        "straight": straight,
        "maneuver": maneuver,
        "anchored": anchored,
        "other": other,
    }


def motion_stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    *,
    straight_fraction: float = 0.15,
    other_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Oversample straight/other buckets; fill remainder uniformly at random."""
    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    masks = motion_bucket_masks_df(df)
    rng = np.random.default_rng(seed)

    n_straight = int(round(sample_size * straight_fraction))
    n_other = int(round(sample_size * other_fraction))

    def pick(mask: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            return np.array([], dtype=int)
        return rng.choice(pool, size=n, replace=len(pool) < n)

    parts = [
        pick(masks["straight"], n_straight),
        pick(masks["other"], n_other),
    ]
    nonempty = [p for p in parts if len(p)]
    if not nonempty:
        # Prefetch row-groups / aggressive stationary filter can leave zero
        # straight/other rows. Fall back to uniform random sampling.
        print(
            "Motion-stratified sample: no straight/other rows available — "
            "falling back to uniform sample"
        )
        return df.sample(sample_size, random_state=seed).reset_index(drop=True)

    chosen = np.unique(np.concatenate(nonempty))
    if len(chosen) < sample_size:
        pool = np.setdiff1d(np.arange(len(df)), chosen)
        need = sample_size - len(chosen)
        if len(pool) > 0:
            extra = rng.choice(pool, size=need, replace=len(pool) < need)
            chosen = np.unique(np.concatenate([chosen, extra]))

    out = df.iloc[chosen[:sample_size]].reset_index(drop=True)
    picked_masks = motion_bucket_masks_df(out)
    print(
        "Motion-stratified sample: "
        f"straight={int(picked_masks['straight'].sum()):,} "
        f"maneuver={int(picked_masks['maneuver'].sum()):,} "
        f"anchored={int(picked_masks['anchored'].sum()):,} "
        f"other={int(picked_masks['other'].sum()):,} "
        f"(requested straight/other frac {straight_fraction:.0%}/{other_fraction:.0%})"
    )
    return out


def prepare_training_frame(
    train_df: pd.DataFrame,
    *,
    sample_size: int | None,
    maneuver_oversample: bool = False,
    maneuver_fraction: float = 0.3,
    motion_balanced_sample: bool = False,
    straight_fraction: float = 0.15,
    other_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply oversampling / balancing on the training split only."""
    if motion_balanced_sample and sample_size:
        return motion_stratified_sample(
            train_df,
            sample_size,
            straight_fraction=straight_fraction,
            other_fraction=other_fraction,
            seed=seed,
        )
    if maneuver_oversample and sample_size:
        return maneuver_balanced_sample(
            train_df,
            sample_size,
            maneuver_fraction=maneuver_fraction,
            seed=seed,
        )
    if sample_size and len(train_df) > sample_size:
        return train_df.sample(sample_size, random_state=seed).reset_index(drop=True)
    return train_df.reset_index(drop=True)


def make_train_val_test_frames(
    df: pd.DataFrame,
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    split_by: str = "trajectory",
    train_sample_size: int | None = None,
    maneuver_oversample: bool = False,
    maneuver_fraction: float = 0.3,
    motion_balanced_sample: bool = False,
    straight_fraction: float = 0.15,
    other_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split first, then balance/subsample training rows only."""
    split_col = split_column(df, split_by)
    train_ids, val_ids, test_ids = trajectory_splits(
        df,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        seed=seed,
        split_by=split_by,
    )
    train_df = df.loc[mask_by_split(df, train_ids, split_col)].copy()
    val_df = df.loc[mask_by_split(df, val_ids, split_col)].copy()
    test_df = df.loc[mask_by_split(df, test_ids, split_col)].copy()
    train_df = prepare_training_frame(
        train_df,
        sample_size=train_sample_size,
        maneuver_oversample=maneuver_oversample,
        maneuver_fraction=maneuver_fraction,
        motion_balanced_sample=motion_balanced_sample,
        straight_fraction=straight_fraction,
        other_fraction=other_fraction,
        seed=seed,
    )
    return train_df, val_df, test_df, split_col


def compute_naive_cumulative_delta(
    x: np.ndarray,
    future_steps: int,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Constant last-step delta extrapolation (cumulative offset from anchor)."""
    cols = feature_cols or FEATURE_COLS
    dlat_idx = feature_index(cols, "dlat")
    dlon_idx = feature_index(cols, "dlon")
    dlat = x[:, -1, dlat_idx].astype(np.float32)
    dlon = x[:, -1, dlon_idx].astype(np.float32)
    steps = np.arange(1, future_steps + 1, dtype=np.float32)
    return np.stack(
        [dlat[:, np.newaxis] * steps[np.newaxis, :], dlon[:, np.newaxis] * steps[np.newaxis, :]],
        axis=-1,
    )


def compute_kinematic_cumulative_delta(
    x: np.ndarray,
    future_steps: int,
    *,
    feature_cols: list[str] | None = None,
    step_minutes: float = 10.0,
) -> np.ndarray:
    """SOG+COG constant-velocity extrapolation as cumulative lat/lon offset from anchor."""
    cols = feature_cols or FEATURE_COLS
    lat_idx = feature_index(cols, "lat")
    sog_idx = feature_index(cols, "sog")
    cog_sin_idx = feature_index(cols, "cog_sin")
    cog_cos_idx = feature_index(cols, "cog_cos")

    last = x[:, -1, :]
    lat0 = last[:, lat_idx].astype(np.float64)
    sog_kn = last[:, sog_idx].astype(np.float64)
    cog_sin = last[:, cog_sin_idx].astype(np.float64)
    cog_cos = last[:, cog_cos_idx].astype(np.float64)

    speed_kmh = sog_kn * NM_TO_KM
    step_hours = step_minutes / 60.0
    north_km = speed_kmh * cog_cos * step_hours
    east_km = speed_kmh * cog_sin * step_hours

    km_per_deg_lat = 111.322
    cos_lat = np.cos(np.deg2rad(lat0)).clip(min=1e-3)
    dlat_step = (north_km / km_per_deg_lat).astype(np.float32)
    dlon_step = (east_km / (km_per_deg_lat * cos_lat)).astype(np.float32)

    steps = np.arange(1, future_steps + 1, dtype=np.float32)
    return np.stack(
        [dlat_step[:, np.newaxis] * steps[np.newaxis, :], dlon_step[:, np.newaxis] * steps[np.newaxis, :]],
        axis=-1,
    )


def baseline_cumulative_delta(
    x: np.ndarray,
    future_steps: int,
    *,
    kinematic: bool = False,
    feature_cols: list[str] | None = None,
    step_minutes: float = 10.0,
) -> np.ndarray:
    if kinematic:
        return compute_kinematic_cumulative_delta(
            x, future_steps, feature_cols=feature_cols, step_minutes=step_minutes
        )
    return compute_naive_cumulative_delta(x, future_steps, feature_cols=feature_cols)


def compute_sample_weights(
    x_raw: np.ndarray,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Boost loss on windows with larger |dcog|/|dsog| in history."""
    cols = feature_cols or FEATURE_COLS
    weight = np.ones(len(x_raw), dtype=np.float32)
    try:
        dcog_idx = feature_index(cols, "dcog")
        dsog_idx = feature_index(cols, "dsog")
        score = np.abs(x_raw[:, :, dcog_idx]).mean(axis=1) + np.abs(
            x_raw[:, :, dsog_idx]
        ).mean(axis=1)
        scale = float(np.median(score) + 1e-6)
        weight = (1.0 + score / scale).astype(np.float32)
    except KeyError:
        pass
    return weight


def classify_window_motion(
    x_raw: np.ndarray,
    feature_cols: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Stratified test buckets: straight / maneuver / anchored / other."""
    cols = feature_cols or FEATURE_COLS
    sog_idx = feature_index(cols, "sog")
    dcog_idx = feature_index(cols, "dcog")

    mean_sog = x_raw[:, :, sog_idx].mean(axis=1)
    max_dcog = np.abs(x_raw[:, :, dcog_idx]).max(axis=1)

    anchored = mean_sog < 1.0
    maneuver = max_dcog > 15.0
    straight = (mean_sog >= 5.0) & (max_dcog < 5.0) & ~anchored
    other = ~(anchored | maneuver | straight)
    return {
        "straight": straight,
        "maneuver": maneuver,
        "anchored": anchored,
        "other": other,
    }


def evaluate_stratified_positions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_raw: np.ndarray,
    label_prefix: str,
    *,
    anchor: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
) -> list[dict[str, float]]:
    masks = classify_window_motion(x_raw, feature_cols)
    results: list[dict[str, float]] = []
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        results.append(
            evaluate_final_position(
                y_true[mask],
                y_pred[mask],
                f"{label_prefix} [{name}, n={int(mask.sum())}]",
                anchor=None if anchor is None else anchor[mask],
            )
        )
    return results


def infer_window_shape(df: pd.DataFrame) -> tuple[int, int]:
    if "history_steps" in df.columns and "future_steps" in df.columns:
        return int(df["history_steps"].iloc[0]), int(df["future_steps"].iloc[0])

    history_steps = 0
    future_steps = 0
    for col in df.columns:
        if col.startswith("x_t"):
            history_steps = max(history_steps, int(col.split("_")[0][3:]) + 1)
        if col.startswith("y_t"):
            future_steps = max(future_steps, int(col.split("_")[0][3:]) + 1)
    if history_steps <= 0 or future_steps <= 0:
        raise ValueError("Could not infer history_steps / future_steps from window columns.")
    return history_steps, future_steps


def hours_to_window_steps(hours: float, step_minutes: float = 10.0) -> int:
    return max(1, int(round(hours * 60 / step_minutes)))


def resolve_window_hours(
    df: pd.DataFrame,
    *,
    history_hours: float | None = None,
    future_hours: float | None = None,
    step_minutes: float | None = None,
) -> tuple[int, int, int, int, float]:
    """Map hour targets to step counts against parquet window columns.

    Returns (full_history_steps, full_future_steps, use_history_steps, use_future_steps, step_minutes).
    History is always taken as the suffix ending at the anchor (last full-history step).
    Future is always taken as the prefix from forecast start.
    """
    full_h, full_f = infer_window_shape(df)
    if step_minutes is None:
        step_minutes = (
            float(df["resample_minutes"].iloc[0])
            if "resample_minutes" in df.columns
            else 10.0
        )
    use_h = (
        hours_to_window_steps(history_hours, step_minutes)
        if history_hours is not None
        else full_h
    )
    use_f = (
        hours_to_window_steps(future_hours, step_minutes)
        if future_hours is not None
        else full_f
    )
    if use_h > full_h:
        raise ValueError(
            f"history_hours={history_hours} needs {use_h} steps but parquet has {full_h}"
        )
    if use_f > full_f:
        raise ValueError(
            f"future_hours={future_hours} needs {use_f} steps but parquet has {full_f}"
        )
    return full_h, full_f, use_h, use_f, step_minutes


def build_window_arrays(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    history_steps: int | None = None,
    future_steps: int | None = None,
    full_history_steps: int | None = None,
    target_mode: str = "anchor_offset",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = feature_cols or infer_feature_cols(df)
    full_h, full_f = infer_window_shape(df)
    if full_history_steps is not None:
        full_h = full_history_steps
    if history_steps is None:
        history_steps = full_h
    if future_steps is None:
        future_steps = full_f
    if history_steps > full_h:
        raise ValueError(f"history_steps={history_steps} exceeds parquet history {full_h}")
    if future_steps > full_f:
        raise ValueError(f"future_steps={future_steps} exceeds parquet future {full_f}")

    n_samples = len(df)
    n_features = len(feature_cols)
    x = np.empty((n_samples, history_steps, n_features), dtype=np.float32)
    y = np.empty((n_samples, future_steps, 2), dtype=np.float32)

    h_start = full_h - history_steps
    for t in range(history_steps):
        src_t = h_start + t
        cols = [f"x_t{src_t:03d}_{col}" for col in feature_cols]
        x[:, t, :] = df[cols].to_numpy(dtype=np.float32)

    for t in range(future_steps):
        y[:, t, 0] = df[f"y_t{t:03d}_lat"].to_numpy(dtype=np.float32)
        y[:, t, 1] = df[f"y_t{t:03d}_lon"].to_numpy(dtype=np.float32)

    anchor = x[:, -1, :2].copy()
    y_delta = build_targets(y, anchor, target_mode=target_mode)
    return x, y, y_delta, anchor


def build_targets(
    y_abs: np.ndarray,
    anchor: np.ndarray,
    *,
    target_mode: str = "anchor_offset",
) -> np.ndarray:
    """Future targets: anchor-offset (P_t - P_0) or step-delta (P_t - P_{t-1})."""
    if target_mode == "anchor_offset":
        return (y_abs - anchor[:, np.newaxis, :]).astype(np.float32)
    if target_mode == "step_delta":
        return absolute_to_step_delta(y_abs, anchor)
    raise ValueError(f"target_mode must be 'anchor_offset' or 'step_delta', got {target_mode!r}")


def absolute_to_step_delta(y_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Local step displacements d_t = P_t - P_{t-1}; d_0 = P_0 - anchor."""
    step = np.empty_like(y_abs, dtype=np.float32)
    step[:, 0, :] = y_abs[:, 0, :] - anchor
    if y_abs.shape[1] > 1:
        step[:, 1:, :] = y_abs[:, 1:, :] - y_abs[:, :-1, :]
    return step


def step_delta_to_anchor_offset(step_delta: np.ndarray) -> np.ndarray:
    """Cumulative sum of step-deltas → offsets from anchor."""
    return np.cumsum(step_delta, axis=1).astype(np.float32)


def deltas_to_absolute(
    deltas: np.ndarray,
    anchor: np.ndarray,
    *,
    target_mode: str = "anchor_offset",
) -> np.ndarray:
    if target_mode == "step_delta":
        return anchor[:, np.newaxis, :] + np.cumsum(deltas, axis=1)
    return anchor[:, np.newaxis, :] + deltas


def cumulative_delta_to_step_delta(cumulative: np.ndarray) -> np.ndarray:
    step = np.empty_like(cumulative)
    step[:, 0, :] = cumulative[:, 0, :]
    if cumulative.shape[1] > 1:
        step[:, 1:, :] = cumulative[:, 1:, :] - cumulative[:, :-1, :]
    return step.astype(np.float32)


def baseline_step_delta(
    x: np.ndarray,
    future_steps: int,
    *,
    kinematic: bool = False,
    feature_cols: list[str] | None = None,
    step_minutes: float = 10.0,
) -> np.ndarray:
    """Per-step naive/kinematic displacements for residual step-delta training."""
    cumulative = baseline_cumulative_delta(
        x,
        future_steps,
        kinematic=kinematic,
        feature_cols=feature_cols,
        step_minutes=step_minutes,
    )
    return cumulative_delta_to_step_delta(cumulative)


def hour_steps_from_minutes(step_minutes: float, hours: float = 1.0) -> int:
    return max(1, int(round(hours * 60 / step_minutes)))


def one_hour_displacement_from_future(
    y_abs: np.ndarray,
    anchor: np.ndarray,
    hour_step: int,
) -> np.ndarray:
    """Displacement P_{t+1h} - P_0 for each sample, shape (N, 2)."""
    return (y_abs[:, hour_step, :] - anchor).astype(np.float32)


def chunk_displacement_from_future(
    y_abs: np.ndarray,
    anchor: np.ndarray,
    chunk_end_step: int,
) -> np.ndarray:
    """Displacement to chunk end step relative to anchor: P_{t+chunk} - P_0."""
    return (y_abs[:, chunk_end_step, :] - anchor).astype(np.float32)


def naive_one_hour_displacement(
    x: np.ndarray,
    hour_step: int,
    *,
    kinematic: bool = False,
    feature_cols: list[str] | None = None,
    step_minutes: float = 10.0,
) -> np.ndarray:
    cumulative = baseline_cumulative_delta(
        x,
        hour_step + 1,
        kinematic=kinematic,
        feature_cols=feature_cols,
        step_minutes=step_minutes,
    )
    return cumulative[:, hour_step, :].astype(np.float32)


def naive_chunk_displacement(
    x: np.ndarray,
    chunk_end_step: int,
    *,
    kinematic: bool = False,
    feature_cols: list[str] | None = None,
    step_minutes: float = 10.0,
) -> np.ndarray:
    return naive_one_hour_displacement(
        x, chunk_end_step, kinematic=kinematic, feature_cols=feature_cols, step_minutes=step_minutes
    )


def append_synthetic_hour_to_history(
    x: np.ndarray,
    hour_displacement: np.ndarray,
    *,
    steps_per_hour: int,
    step_minutes: float = 10.0,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Shift history window forward by one hour, appending synthetic AIS feature rows.

    Used for recursive 1-hour sliding-window inference (Experiment E).
    """
    cols = feature_cols or FEATURE_COLS
    lat_i = feature_index(cols, "lat")
    lon_i = feature_index(cols, "lon")
    sog_i = feature_index(cols, "sog")
    cog_sin_i = feature_index(cols, "cog_sin")
    cog_cos_i = feature_index(cols, "cog_cos")
    heading_sin_i = feature_index(cols, "heading_sin")
    heading_cos_i = feature_index(cols, "heading_cos")
    heading_miss_i = feature_index(cols, "heading_missing")
    dt_i = feature_index(cols, "dt_sec")
    dlat_i = feature_index(cols, "dlat")
    dlon_i = feature_index(cols, "dlon")
    dsog_i = feature_index(cols, "dsog")
    dcog_i = feature_index(cols, "dcog")
    vn_i = feature_index(cols, "v_north_kmh")
    ve_i = feature_index(cols, "v_east_kmh")

    batch, total_steps, n_feat = x.shape
    drop = min(steps_per_hour, total_steps)
    keep = total_steps - drop
    out = np.zeros((batch, total_steps, n_feat), dtype=np.float32)
    out[:, :keep, :] = x[:, drop:, :]

    last_lat = x[:, -1, lat_i].astype(np.float64)
    last_lon = x[:, -1, lon_i].astype(np.float64)
    dlat_total = hour_displacement[:, 0].astype(np.float64)
    dlon_total = hour_displacement[:, 1].astype(np.float64)
    dt_sec = step_minutes * 60.0

    prev_lat = last_lat.copy()
    prev_lon = last_lon.copy()
    prev_sog = x[:, -1, sog_i].astype(np.float64)

    for s in range(steps_per_hour):
        frac = (s + 1) / steps_per_hour
        lat = last_lat + dlat_total * frac
        lon = last_lon + dlon_total * frac
        dlat = lat - prev_lat
        dlon = lon - prev_lon
        dist_km = np.sqrt((dlat * 111.322) ** 2 + (dlon * 111.322 * np.cos(np.deg2rad(lat))) ** 2)
        sog_kn = (dist_km / NM_TO_KM) / (step_minutes / 60.0)
        cog_rad = np.arctan2(dlon * np.cos(np.deg2rad(lat)), dlat)
        dsog = sog_kn - prev_sog

        idx = keep + s
        out[:, idx, lat_i] = lat.astype(np.float32)
        out[:, idx, lon_i] = lon.astype(np.float32)
        out[:, idx, sog_i] = sog_kn.astype(np.float32)
        out[:, idx, cog_sin_i] = np.sin(cog_rad).astype(np.float32)
        out[:, idx, cog_cos_i] = np.cos(cog_rad).astype(np.float32)
        out[:, idx, heading_sin_i] = out[:, idx, cog_sin_i]
        out[:, idx, heading_cos_i] = out[:, idx, cog_cos_i]
        out[:, idx, heading_miss_i] = 0.0
        out[:, idx, dt_i] = dt_sec
        out[:, idx, dlat_i] = dlat.astype(np.float32)
        out[:, idx, dlon_i] = dlon.astype(np.float32)
        out[:, idx, dsog_i] = dsog.astype(np.float32)
        out[:, idx, dcog_i] = 0.0
        out[:, idx, vn_i] = (sog_kn * NM_TO_KM * np.cos(cog_rad)).astype(np.float32)
        out[:, idx, ve_i] = (sog_kn * NM_TO_KM * np.sin(cog_rad)).astype(np.float32)

        prev_lat = lat
        prev_lon = lon
        prev_sog = sog_kn

    return out


def scale_history_features(
    x_train: np.ndarray,
    x_other: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    n_train, history_steps, n_features = x_train.shape
    flat_train = x_train.reshape(n_train * history_steps, n_features)
    scaler.fit(flat_train)

    x_train_scaled = scaler.transform(flat_train).reshape(n_train, history_steps, n_features)
    scaled_other = []
    for arr in x_other:
        n = arr.shape[0]
        flat = arr.reshape(n * history_steps, n_features)
        scaled_other.append(scaler.transform(flat).reshape(n, history_steps, n_features))
    return x_train_scaled.astype(np.float32), [a.astype(np.float32) for a in scaled_other], scaler


def split_column(df: pd.DataFrame, split_by: str) -> str:
    if split_by == "mmsi":
        if "mmsi" not in df.columns:
            raise KeyError("split_by='mmsi' but column 'mmsi' is missing from windows frame")
        return "mmsi"
    if "traj_id" in df.columns:
        return "traj_id"
    raise KeyError("No trajectory id column found (expected 'traj_id')")


def trajectory_splits(
    df: pd.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    *,
    split_by: str = "trajectory",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_col = split_column(df, split_by)
    groups = df[split_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    test_count = max(1, int(len(groups) * test_fraction))
    val_count = max(1, int(len(groups) * val_fraction)) if val_fraction > 0 else 0

    test_ids = groups[:test_count]
    val_ids = groups[test_count : test_count + val_count]
    train_ids = groups[test_count + val_count :]
    return train_ids, val_ids, test_ids


def mask_by_split(
    df: pd.DataFrame,
    ids: np.ndarray,
    split_col: str | None = None,
    *,
    split_by: str = "trajectory",
) -> np.ndarray:
    col = split_col or split_column(df, split_by)
    return df[col].isin(ids).to_numpy()


def horizon_step_index(df: pd.DataFrame, horizon_hours: float, future_steps: int) -> int:
    if "resample_minutes" in df.columns:
        resample = int(df["resample_minutes"].iloc[0])
        step = int(round(horizon_hours * 60 / resample))
    elif "future_hours" in df.columns:
        total_hours = float(df["future_hours"].iloc[0])
        step = int(round(horizon_hours / total_hours * future_steps))
    else:
        step = int(round(horizon_hours))

    step = max(1, min(step, future_steps))
    return step - 1


def naive_position_at_horizon(
    x: np.ndarray,
    y: np.ndarray,
    horizon_step: int,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrapolate position at a specific future step using the last observed delta."""
    cols = feature_cols or FEATURE_COLS
    lat_idx = feature_index(cols, "lat")
    lon_idx = feature_index(cols, "lon")
    dlat_idx = feature_index(cols, "dlat")
    dlon_idx = feature_index(cols, "dlon")

    last = x[:, -1, :]
    steps_ahead = horizon_step + 1
    y_true = y[:, horizon_step, :]
    y_pred = np.column_stack(
        [
            last[:, lat_idx] + last[:, dlat_idx] * steps_ahead,
            last[:, lon_idx] + last[:, dlon_idx] * steps_ahead,
        ]
    )
    return y_true, y_pred


def kinematic_position_at_horizon(
    x: np.ndarray,
    y: np.ndarray,
    horizon_step: int,
    *,
    feature_cols: list[str] | None = None,
    step_minutes: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrapolate using last SOG and COG (constant velocity)."""
    cols = feature_cols or FEATURE_COLS
    lat_idx = feature_index(cols, "lat")
    lon_idx = feature_index(cols, "lon")
    delta = compute_kinematic_cumulative_delta(
        x,
        horizon_step + 1,
        feature_cols=cols,
        step_minutes=step_minutes,
    )
    last = x[:, -1, :]
    y_true = y[:, horizon_step, :]
    y_pred = np.column_stack(
        [
            last[:, lat_idx] + delta[:, horizon_step, 0],
            last[:, lon_idx] + delta[:, horizon_step, 1],
        ]
    )
    return y_true, y_pred


def naive_final_position(
    x: np.ndarray,
    y: np.ndarray,
    future_steps: int,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return naive_position_at_horizon(x, y, future_steps - 1, feature_cols=feature_cols)


def window_horizon_hours(df: pd.DataFrame, future_steps: int) -> float:
    if "resample_minutes" in df.columns:
        resample = int(df["resample_minutes"].iloc[0])
        return future_steps * resample / 60.0
    if "future_hours" in df.columns:
        return float(df["future_hours"].iloc[0])
    return float(future_steps)


# ---------------------------------------------------------------------------
# Great-circle distance metrics (degrees → km / nm)
# ---------------------------------------------------------------------------

NM_TO_KM = 1.852
EARTH_RADIUS_KM = 6371.0
DEFAULT_MIN_NORMALIZE_KM = 10.0


def future_path_length_km(y_true: np.ndarray) -> np.ndarray:
    """Sum of step distances along the true future trajectory, shape (N,)."""
    if y_true.ndim != 3 or y_true.shape[1] < 2:
        return np.zeros(len(y_true), dtype=np.float64)
    seg = haversine_km(
        y_true[:, :-1, 0],
        y_true[:, :-1, 1],
        y_true[:, 1:, 0],
        y_true[:, 1:, 1],
    )
    return seg.sum(axis=1)


def future_displacement_km(
    y_true: np.ndarray,
    anchor: np.ndarray,
    *,
    horizon_step: int | None = None,
) -> np.ndarray:
    """Great-circle distance from anchor to true position at horizon (or final step)."""
    if y_true.ndim == 3:
        step = (y_true.shape[1] - 1) if horizon_step is None else horizon_step
        y_end = y_true[:, step, :]
    else:
        y_end = y_true
    return haversine_km(anchor[:, 0], anchor[:, 1], y_end[:, 0], y_end[:, 1])


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometers."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in nautical miles."""
    return haversine_km(lat1, lon1, lat2, lon2) / NM_TO_KM


def evaluate_final_position(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    *,
    anchor: np.ndarray | None = None,
    min_normalize_km: float = DEFAULT_MIN_NORMALIZE_KM,
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    distance_km = haversine_km(
        y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1]
    )
    distance_nm = distance_km / NM_TO_KM

    out: dict[str, float] = {
        "model": label,
        "mae_lat": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "mae_lon": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "rmse_lat": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_lon": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "r2_lat": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_lon": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "mean_error_km": float(distance_km.mean()),
        "median_error_km": float(np.median(distance_km)),
        "p90_error_km": float(np.percentile(distance_km, 90)),
        "p95_error_km": float(np.percentile(distance_km, 95)),
        "mean_error_nm": float(distance_nm.mean()),
        "median_error_nm": float(np.median(distance_nm)),
        "p90_error_nm": float(np.percentile(distance_nm, 90)),
        "p95_error_nm": float(np.percentile(distance_nm, 95)),
    }

    if anchor is not None:
        denom = np.maximum(
            future_displacement_km(y_true, anchor),
            min_normalize_km,
        )
        nfde = distance_km / denom
        out["mean_nfde"] = float(nfde.mean())
        out["median_nfde"] = float(np.median(nfde))
        out["p90_nfde"] = float(np.percentile(nfde, 90))
        out["mean_future_displacement_km"] = float(denom.mean())

    return out


def evaluate_full_trajectory(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    *,
    min_normalize_km: float = DEFAULT_MIN_NORMALIZE_KM,
) -> dict[str, float]:
    distances_km = haversine_km(
        y_true[:, :, 0], y_true[:, :, 1], y_pred[:, :, 0], y_pred[:, :, 1]
    )
    distances_nm = distances_km / NM_TO_KM
    ade_per_window = distances_km.mean(axis=1)
    path_len = np.maximum(future_path_length_km(y_true), min_normalize_km)
    nade = ade_per_window / path_len

    return {
        "model": label,
        "mean_ade_km": float(distances_km.mean()),
        "median_ade_km": float(np.median(distances_km)),
        "final_step_mean_error_km": float(distances_km[:, -1].mean()),
        "mean_ade_nm": float(distances_nm.mean()),
        "median_ade_nm": float(np.median(distances_nm)),
        "final_step_mean_error_nm": float(distances_nm[:, -1].mean()),
        "mean_nade": float(nade.mean()),
        "median_nade": float(np.median(nade)),
        "p90_nade": float(np.percentile(nade, 90)),
        "mean_true_path_km": float(path_len.mean()),
    }


def print_position_metrics(metrics: dict[str, float]) -> None:
    print(f"\n{metrics['model']}")

    if "mae_lat" in metrics:
        print(f"  MAE lat/lon:     {metrics['mae_lat']:.6f} / {metrics['mae_lon']:.6f} deg")
        print(f"  RMSE lat/lon:    {metrics['rmse_lat']:.6f} / {metrics['rmse_lon']:.6f} deg")
        print(f"  R² lat/lon:      {metrics['r2_lat']:.4f} / {metrics['r2_lon']:.4f}")
        print(
            f"  Position error:  mean {metrics['mean_error_km']:.3f} km "
            f"({metrics['mean_error_nm']:.3f} nm) | "
            f"median {metrics['median_error_km']:.3f} km "
            f"({metrics['median_error_nm']:.3f} nm) | "
            f"p90 {metrics['p90_error_km']:.3f} km "
            f"({metrics['p90_error_nm']:.3f} nm) | "
            f"p95 {metrics['p95_error_km']:.3f} km "
            f"({metrics['p95_error_nm']:.3f} nm)"
        )
        if "mean_nfde" in metrics:
            print(
                f"  Normalized FDE:  mean {metrics['mean_nfde']:.1%} | "
                f"median {metrics['median_nfde']:.1%} | "
                f"p90 {metrics['p90_nfde']:.1%} "
                f"(÷ max(true displacement, {DEFAULT_MIN_NORMALIZE_KM:.0f} km))"
            )

    if "mean_ade_km" in metrics:
        print(
            f"  Trajectory ADE:  mean {metrics['mean_ade_km']:.3f} km "
            f"({metrics['mean_ade_nm']:.3f} nm) | "
            f"median {metrics['median_ade_km']:.3f} km "
            f"({metrics['median_ade_nm']:.3f} nm)"
        )
        if "mean_nade" in metrics:
            print(
                f"  Normalized ADE:  mean {metrics['mean_nade']:.1%} | "
                f"median {metrics['median_nade']:.1%} | "
                f"p90 {metrics['p90_nade']:.1%} "
                f"(÷ max(true path length, {DEFAULT_MIN_NORMALIZE_KM:.0f} km), "
                f"avg path {metrics['mean_true_path_km']:.1f} km)"
            )
