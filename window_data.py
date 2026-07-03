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
    # Max distance from anchor to any history/future point (the "cell" radius).
    max_confined_radius_km: float = 0.5
    # Net displacement anchor → final future position.
    min_future_displacement_km: float = 1.0
    # Optional: mean SOG below this (knots) reinforces stationary classification.
    min_mean_sog_kn: float = 0.5


def _track_path_length_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if lat.shape[1] < 2:
        return np.zeros(lat.shape[0], dtype=np.float64)
    seg = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    return seg.sum(axis=1)


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

    Primary rule: stays inside a small cell AND barely moves net over 12h future.
    Secondary: very low SOG + low future displacement.
    """
    confined = metrics["max_radius_km"] <= config.max_confined_radius_km
    short_future = metrics["future_displacement_km"] < config.min_future_displacement_km
    primary = confined & short_future

    if np.isfinite(metrics["mean_sog_kn"]).any():
        slow = metrics["mean_sog_kn"] < config.min_mean_sog_kn
        secondary = slow & short_future & confined
        return (primary | secondary).to_numpy()
    return primary.to_numpy()


def filter_stationary_windows(
    df: pd.DataFrame,
    config: StationaryFilterConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not config.enabled or df.empty:
        return df, {"removed": 0.0, "kept_fraction": 1.0}

    metrics = compute_window_motion_metrics(df)
    remove = stationary_window_mask(metrics, config)
    n_removed = int(remove.sum())
    n_total = len(df)

    stats = {
        "windows_total": float(n_total),
        "windows_removed": float(n_removed),
        "windows_kept": float(n_total - n_removed),
        "removed_fraction": float(n_removed / max(n_total, 1)),
        "kept_fraction": float((n_total - n_removed) / max(n_total, 1)),
        "median_max_radius_km_removed": float(metrics.loc[remove, "max_radius_km"].median())
        if n_removed
        else 0.0,
        "median_max_radius_km_kept": float(metrics.loc[~remove, "max_radius_km"].median())
        if n_removed < n_total
        else 0.0,
    }

    if n_removed == 0:
        return df, stats

    return df.loc[~remove].reset_index(drop=True), stats


def print_stationary_filter_stats(stats: dict[str, float], config: StationaryFilterConfig) -> None:
    print(
        f"Stationary filter: radius≤{config.max_confined_radius_km:.2f} km & "
        f"future disp<{config.min_future_displacement_km:.2f} km "
        f"(SOG<{config.min_mean_sog_kn:.1f} kn reinforces)"
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
        help="Min net future displacement to keep a confined window (default: 1.0 km).",
    )
    parser.add_argument(
        "--min-mean-sog-kn",
        type=float,
        default=0.5,
        help="Mean SOG below this reinforces stationary removal (default: 0.5 kn).",
    )


def stationary_filter_from_args(args) -> StationaryFilterConfig:
    return StationaryFilterConfig(
        enabled=getattr(args, "filter_stationary", False),
        max_confined_radius_km=getattr(args, "max_confined_radius_km", 0.5),
        min_future_displacement_km=getattr(args, "min_future_displacement_km", 1.0),
        min_mean_sog_kn=getattr(args, "min_mean_sog_kn", 0.5),
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
    chosen = np.unique(np.concatenate([p for p in parts if len(p)]))
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


def compute_naive_cumulative_delta(
    x: np.ndarray,
    future_steps: int,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Constant-velocity cumulative offset from anchor for each future step."""
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


def build_window_arrays(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    history_steps: int | None = None,
    future_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = feature_cols or infer_feature_cols(df)
    if history_steps is None or future_steps is None:
        history_steps, future_steps = infer_window_shape(df)

    n_samples = len(df)
    n_features = len(feature_cols)
    x = np.empty((n_samples, history_steps, n_features), dtype=np.float32)
    y = np.empty((n_samples, future_steps, 2), dtype=np.float32)

    # Read all features for one timestep at a time — 144 block reads vs 2160
    # individual column reads.  Each block is (n_samples, 15) which is small
    # enough to stay in cache and avoids repeated pandas overhead.
    for t in range(history_steps):
        cols = [f"x_t{t:03d}_{col}" for col in feature_cols]
        x[:, t, :] = df[cols].to_numpy(dtype=np.float32)

    for t in range(future_steps):
        y[:, t, 0] = df[f"y_t{t:03d}_lat"].to_numpy(dtype=np.float32)
        y[:, t, 1] = df[f"y_t{t:03d}_lon"].to_numpy(dtype=np.float32)

    anchor = x[:, -1, :2].copy()
    y_delta = y - anchor[:, np.newaxis, :]
    return x, y, y_delta, anchor


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


def trajectory_splits(
    df: pd.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_col = "traj_id" if "traj_id" in df.columns else "mmsi"
    trajectories = df[split_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(trajectories)

    test_count = max(1, int(len(trajectories) * test_fraction))
    val_count = max(1, int(len(trajectories) * val_fraction)) if val_fraction > 0 else 0

    test_ids = trajectories[:test_count]
    val_ids = trajectories[test_count : test_count + val_count]
    train_ids = trajectories[test_count + val_count :]
    return train_ids, val_ids, test_ids


def mask_by_split(df: pd.DataFrame, ids: np.ndarray, split_col: str = "traj_id") -> np.ndarray:
    if split_col not in df.columns:
        split_col = "mmsi"
    return df[split_col].isin(ids).to_numpy()


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
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    distance_km = haversine_km(
        y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1]
    )
    distance_nm = distance_km / NM_TO_KM

    return {
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


def evaluate_full_trajectory(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float]:
    distances_km = haversine_km(
        y_true[:, :, 0], y_true[:, :, 1], y_pred[:, :, 0], y_pred[:, :, 1]
    )
    distances_nm = distances_km / NM_TO_KM

    return {
        "model": label,
        "mean_ade_km": float(distances_km.mean()),
        "median_ade_km": float(np.median(distances_km)),
        "final_step_mean_error_km": float(distances_km[:, -1].mean()),
        "mean_ade_nm": float(distances_nm.mean()),
        "median_ade_nm": float(np.median(distances_nm)),
        "final_step_mean_error_nm": float(distances_nm[:, -1].mean()),
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

    if "mean_ade_km" in metrics:
        print(
            f"  Trajectory ADE:  mean {metrics['mean_ade_km']:.3f} km "
            f"({metrics['mean_ade_nm']:.3f} nm) | "
            f"median {metrics['median_ade_km']:.3f} km "
            f"({metrics['median_ade_nm']:.3f} nm)"
        )
