from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proj.project.coast_paths import COAST_CONFIGS, resolve_windows_path, results_output_dir
from proj.project.window_data import (
    FEATURE_COLS,
    build_window_arrays,
    horizon_step_index,
    infer_feature_cols,
    infer_window_shape,
    load_windows,
    mask_by_split,
    naive_position_at_horizon,
    trajectory_splits,
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def flatten_history(x: np.ndarray) -> np.ndarray:
    n_samples, history_steps, n_features = x.shape
    return x.reshape(n_samples, history_steps * n_features)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> dict[str, float]:
    distance_km = haversine_km(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
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
    }


def print_metrics(metrics: dict[str, float]) -> None:
    print(f"\n{metrics['model']}")
    print(f"  MAE lat/lon:     {metrics['mae_lat']:.6f} / {metrics['mae_lon']:.6f} deg")
    print(f"  RMSE lat/lon:    {metrics['rmse_lat']:.6f} / {metrics['rmse_lon']:.6f} deg")
    print(f"  R² lat/lon:      {metrics['r2_lat']:.4f} / {metrics['r2_lon']:.4f}")
    print(
        f"  Position error:  mean {metrics['mean_error_km']:.3f} km | "
        f"median {metrics['median_error_km']:.3f} km | "
        f"p90 {metrics['p90_error_km']:.3f} km | "
        f"p95 {metrics['p95_error_km']:.3f} km"
    )


def save_error_histogram(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    errors_km = haversine_km(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_km, bins=60, color="#377eb8", edgecolor="white", alpha=0.9)
    ax.axvline(
        errors_km.mean(),
        color="#e41a1c",
        linestyle="--",
        label=f"mean = {errors_km.mean():.3f} km",
    )
    ax.axvline(
        np.median(errors_km),
        color="#4daf4a",
        linestyle="--",
        label=f"median = {np.median(errors_km):.3f} km",
    )
    ax.set_xlabel("Prediction error (kilometers)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_scatter_plot(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str, max_points: int = 5000
) -> None:
    import matplotlib.pyplot as plt

    if len(y_true) > max_points:
        idx = np.random.default_rng(42).choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dim, name in zip(axes, [0, 1], ["Latitude", "Longitude"]):
        ax.scatter(y_true[:, dim], y_pred[:, dim], s=6, alpha=0.25, color="#377eb8")
        lo = min(y_true[:, dim].min(), y_pred[:, dim].min())
        hi = max(y_true[:, dim].max(), y_pred[:, dim].max())
        ax.plot([lo, hi], [lo, hi], color="#e41a1c", linestyle="--", linewidth=1.5, label="perfect")
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_regression(
    input_path: Path,
    coast_name: str | None,
    region: str,
    sample_size: int | None,
    test_fraction: float,
    seed: int,
    horizon_hours: float,
) -> Path:
    start = time.perf_counter()

    input_path, coast, region = resolve_windows_path(coast_name, region, input_path)

    print(f"Loading {input_path}...", flush=True)
    df = load_windows(input_path, sample_size=sample_size)

    output_dir = results_output_dir(coast, input_path, "LINEAR_REGRESSION", df)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {coast.name} | {output_dir.parent.name} ===", flush=True)
    history_steps, future_steps = infer_window_shape(df)
    horizon_step = horizon_step_index(df, horizon_hours, future_steps)
    actual_horizon_hours = (
        (horizon_step + 1) * int(df["resample_minutes"].iloc[0]) / 60
        if "resample_minutes" in df.columns
        else horizon_hours
    )

    feature_cols = infer_feature_cols(df)
    x, y, _, anchor = build_window_arrays(
        df,
        feature_cols=feature_cols,
        history_steps=history_steps,
        future_steps=future_steps,
    )
    y_target = y[:, horizon_step, :]
    y_target_delta = y_target - anchor

    train_ids, _, test_ids = trajectory_splits(df, test_fraction=test_fraction, val_fraction=0.0, seed=seed)
    train_mask = mask_by_split(df, train_ids)
    test_mask = mask_by_split(df, test_ids)

    x_train = flatten_history(x[train_mask])
    x_test = flatten_history(x[test_mask])
    y_train = y_target_delta[train_mask]
    y_test_delta = y_target_delta[test_mask]
    y_test_true = y_target[test_mask]
    x_test_raw = x[test_mask]

    print(
        f"Samples: {len(df):,} | train: {train_mask.sum():,} | test: {test_mask.sum():,} | "
        f"horizon: {actual_horizon_hours:.1f} h (step {horizon_step + 1}/{future_steps})",
        flush=True,
    )

    print("Training linear regression...", flush=True)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    y_pred_delta = model.predict(x_test)
    y_pred = y_pred_delta + anchor[test_mask]

    y_baseline_true, y_baseline_pred = naive_position_at_horizon(
        x_test_raw, y[test_mask], horizon_step, feature_cols=feature_cols
    )

    metrics = [
        evaluate_predictions(y_baseline_true, y_baseline_pred, "Naive baseline (constant step delta)"),
        evaluate_predictions(
            y_test_true,
            y_pred,
            f"Linear regression ({actual_horizon_hours:.1f} h ahead)",
        ),
    ]
    for m in metrics:
        print_metrics(m)

    regressor = model.named_steps["regressor"]
    results = {
        "input": str(input_path),
        "coast": coast.name,
        "days_label": output_dir.parent.name,
        "region": region,
        "samples_total": len(df),
        "samples_train": int(train_mask.sum()),
        "samples_test": int(test_mask.sum()),
        "history_steps": history_steps,
        "future_steps": future_steps,
        "horizon_hours_requested": horizon_hours,
        "horizon_hours_actual": actual_horizon_hours,
        "horizon_step_index": horizon_step,
        "features": feature_cols,
        "flattened_feature_count": x_train.shape[1],
        "targets": ["delta_lat", "delta_lon"],
        "test_fraction": test_fraction,
        "sample_size": sample_size,
        "metrics": metrics,
        "intercept_lat": float(regressor.intercept_[0]),
        "intercept_lon": float(regressor.intercept_[1]),
        "runtime_sec": round(time.perf_counter() - start, 2),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    save_error_histogram(
        y_test_true,
        y_pred,
        output_dir / "error_hist.png",
        f"Linear Regression — Position Error ({actual_horizon_hours:.1f} h ahead, test set)",
    )
    save_scatter_plot(
        y_test_true,
        y_pred,
        output_dir / "scatter.png",
        f"Linear Regression — Position ({actual_horizon_hours:.1f} h ahead, test set)",
    )

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved plots:   {output_dir / 'error_hist.png'}")
    print(f"               {output_dir / 'scatter.png'}")
    print(f"Runtime: {format_duration(time.perf_counter() - start)}", flush=True)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear regression on NOAA coastal windows to predict ship position 1 hour ahead."
    )
    parser.add_argument("--input", type=Path, default=None, help="Path to model_ready_windows.parquet.")
    parser.add_argument(
        "--coast",
        choices=sorted(COAST_CONFIGS.keys()),
        default=None,
        help="Coastal area (default: Eastern coast, or inferred from --input).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Region label used by PROCESS_noaa_long_coastal.py (if --input not set).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Random sample size for faster runs (0 = use all windows).",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test trajectory fraction.")
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=1.0,
        help="How far ahead to predict (must fit inside the future window in the parquet).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.region is None:
        if args.coast is not None:
            region = COAST_CONFIGS[args.coast].default_region
        else:
            region = COAST_CONFIGS["Eastern coast"].default_region
    else:
        region = args.region

    sample_size = None if args.sample <= 0 else args.sample
    run_regression(
        input_path=args.input,
        coast_name=args.coast,
        region=region,
        sample_size=sample_size,
        test_fraction=args.test_fraction,
        seed=args.seed,
        horizon_hours=args.horizon_hours,
    )


if __name__ == "__main__":
    main()
