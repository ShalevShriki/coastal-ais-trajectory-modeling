from __future__ import annotations

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


def load_windows(path: Path, sample_size: int | None = None) -> pd.DataFrame:
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
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df


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
