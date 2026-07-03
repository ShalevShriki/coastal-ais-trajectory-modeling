"""
compare_rnn_models.py
---------------------
Generates a side-by-side comparison of a flat-head RNN and an
autoregressive (AR) RNN trained on the same dataset.

Outputs (all written to --output-dir):
  comparison_training_curves.png  — overlaid loss curves
  comparison_metrics_bar.png      — MAE / RMSE / mean-error-nm bars
  comparison_compute.png          — param count, GPU memory, throughput
  comparison_error_hist.png       — overlaid error histograms
  comparison_scatter.png          — lat/lon scatter (2×2 grid)
  comparison_map.html             — folium map with sample trajectories
  comparison_summary.json         — all numbers in one place

Usage
-----
  python compare_rnn_models.py \
      --rnn-metrics   <path/to/lstm_metrics.json> \
      --rnn-ar-metrics <path/to/lstm_ar_metrics.json>

  # Or auto-discover the most recent results under data/results/
  python compare_rnn_models.py --auto-discover --rnn-type lstm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_metric(metrics_list: list[dict], key: str) -> dict | None:
    for m in metrics_list:
        if key in m:
            return m
    return None


def get_position_metrics(results: dict) -> dict:
    """Pull the model's own position-error dict (not the baseline)."""
    for m in results["metrics"]:
        if "mae_lat" in m and "Naive" not in m.get("model", ""):
            return m
    return {}


def get_trajectory_metrics(results: dict) -> dict:
    for m in results["metrics"]:
        if "mean_ade_nm" in m:
            return m
    return {}


def load_traj_json(path: Path) -> dict | None:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def discover_metrics(results_root: Path, rnn_type: str) -> tuple[Path | None, Path | None]:
    flat_candidates = sorted(
        results_root.rglob(f"{rnn_type}_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    ar_candidates = sorted(
        results_root.rglob(f"{rnn_type}_ar_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return (
        flat_candidates[0] if flat_candidates else None,
        ar_candidates[0] if ar_candidates else None,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLORS = {"rnn": "#2196F3", "rnn_ar": "#FF5722"}
LABELS = {"rnn": "LSTM (flat head)", "rnn_ar": "LSTM-AR (autoregressive)"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(rnn: dict, rnn_ar: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, key, ylabel in zip(axes, ["train_loss", "val_loss"], ["Train loss (Huber)", "Val loss (Huber)"]):
        for tag, results in [("rnn", rnn), ("rnn_ar", rnn_ar)]:
            epochs = [r["epoch"] for r in results["training_history"]]
            values = [r[key] for r in results["training_history"]]
            ax.plot(epochs, values, label=LABELS[tag], color=COLORS[tag])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Training History Comparison", fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# 2. Metric bars
# ---------------------------------------------------------------------------

def plot_metrics_bar(rnn_m: dict, rnn_ar_m: dict, out: Path) -> None:
    keys = [
        ("mae_lat",          "MAE lat (°)"),
        ("mae_lon",          "MAE lon (°)"),
        ("rmse_lat",         "RMSE lat (°)"),
        ("rmse_lon",         "RMSE lon (°)"),
        ("mean_error_km",    "Mean error (km)"),
        ("median_error_km",  "Median error (km)"),
        ("p90_error_km",     "P90 error (km)"),
        ("mean_error_nm",    "Mean error (nm)"),
        ("median_error_nm",  "Median error (nm)"),
    ]

    labels = [k[1] for k in keys]
    rnn_vals   = [rnn_m.get(k[0], 0.0) for k in keys]
    rnn_ar_vals = [rnn_ar_m.get(k[0], 0.0) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width / 2, rnn_vals,   width, label=LABELS["rnn"],   color=COLORS["rnn"],   alpha=0.85)
    b2 = ax.bar(x + width / 2, rnn_ar_vals, width, label=LABELS["rnn_ar"], color=COLORS["rnn_ar"], alpha=0.85)

    ax.bar_label(b1, fmt="%.4f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.4f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Position Prediction Error — Flat Head vs Autoregressive")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# 3. Compute comparison
# ---------------------------------------------------------------------------

def plot_compute(rnn: dict, rnn_ar: dict, out: Path) -> None:
    rnn_c   = rnn.get("compute", {})
    rnn_ar_c = rnn_ar.get("compute", {})

    metrics = [
        ("param_count",                    "Parameters"),
        ("peak_gpu_mb",                    "Peak GPU (MB)"),
        ("avg_throughput_samples_per_sec", "Throughput (samples/s)"),
        ("total_train_sec",                "Total training time (s)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))

    for ax, (key, title) in zip(axes, metrics):
        vals  = [rnn_c.get(key, 0), rnn_ar_c.get(key, 0)]
        bars  = ax.bar(
            [LABELS["rnn"], LABELS["rnn_ar"]],
            vals,
            color=[COLORS["rnn"], COLORS["rnn_ar"]],
            alpha=0.85,
        )
        ax.bar_label(bars, fmt=lambda v: f"{v:,.0f}" if v >= 1 else f"{v:.2f}", padding=2, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels([LABELS["rnn"], LABELS["rnn_ar"]], rotation=12, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Compute Metrics", fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# 4. Error histograms
# ---------------------------------------------------------------------------

def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def plot_error_hist(rnn_traj: dict, rnn_ar_traj: dict, horizon_step: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for tag, traj in [("rnn", rnn_traj), ("rnn_ar", rnn_ar_traj)]:
        true_arr = np.array(traj["y_true"])   # (N, future_steps, 2)
        pred_arr = np.array(traj["y_pred"])

        t_true = true_arr[:, horizon_step, :]
        t_pred = pred_arr[:, horizon_step, :]
        errors = haversine_nm(t_true[:, 0], t_true[:, 1], t_pred[:, 0], t_pred[:, 1])

        ax.hist(errors, bins=60, alpha=0.55, color=COLORS[tag], label=LABELS[tag])
        ax.axvline(errors.mean(), color=COLORS[tag], linestyle="--", linewidth=1.5,
                   label=f"{LABELS[tag]} mean={errors.mean():.2f} nm")

    ax.set_xlabel("Prediction error (nautical miles)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution at Horizon Step {horizon_step + 1}")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# 5. Scatter plots
# ---------------------------------------------------------------------------

def plot_scatter(rnn_traj: dict, rnn_ar_traj: dict, horizon_step: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (tag, traj) in enumerate([("rnn", rnn_traj), ("rnn_ar", rnn_ar_traj)]):
        true_arr = np.array(traj["y_true"])
        pred_arr = np.array(traj["y_pred"])
        t_true = true_arr[:, horizon_step, :]
        t_pred = pred_arr[:, horizon_step, :]

        # subsample for clarity
        if len(t_true) > 3000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(t_true), 3000, replace=False)
            t_true, t_pred = t_true[idx], t_pred[idx]

        for col, (dim, dim_name) in enumerate([(0, "Latitude"), (1, "Longitude")]):
            ax = axes[row, col]
            ax.scatter(t_true[:, dim], t_pred[:, dim], s=5, alpha=0.25, color=COLORS[tag])
            lo = min(t_true[:, dim].min(), t_pred[:, dim].min())
            hi = max(t_true[:, dim].max(), t_pred[:, dim].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
            ax.set_xlabel(f"True {dim_name}")
            ax.set_ylabel(f"Predicted {dim_name}")
            ax.set_title(f"{LABELS[tag]} — {dim_name}")
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Predicted vs True Position Scatter", fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# 6. Folium map
# ---------------------------------------------------------------------------

def build_map(rnn_traj: dict, rnn_ar_traj: dict, n_vessels: int, out: Path) -> None:
    try:
        import folium
    except ImportError:
        print("  folium not installed — skipping map")
        return

    rng = np.random.default_rng(7)
    n = min(n_vessels, len(rnn_traj["y_true"]), len(rnn_ar_traj["y_true"]))
    idx = rng.choice(min(len(rnn_traj["y_true"]), len(rnn_ar_traj["y_true"])), size=n, replace=False)

    rnn_true  = np.array(rnn_traj["y_true"])[idx]
    rnn_pred  = np.array(rnn_traj["y_pred"])[idx]
    ar_pred   = np.array(rnn_ar_traj["y_pred"])[idx]
    anchors   = np.array(rnn_traj["anchor"])[idx]

    center_lat = float(anchors[:, 0].mean())
    center_lon = float(anchors[:, 1].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px;border:2px solid grey;border-radius:5px;font-size:13px">
        <b>Legend</b><br>
        <span style="color:#1a1a2e">&#9679;</span> Last known position<br>
        <span style="color:#2196F3">&#9644;</span> Ground truth trajectory<br>
        <span style="color:#FF9800">&#9644;</span> LSTM flat-head prediction<br>
        <span style="color:#4CAF50">&#9644;</span> LSTM-AR prediction<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    for i in range(n):
        anchor_pt = [float(anchors[i, 0]), float(anchors[i, 1])]

        # anchor marker
        folium.CircleMarker(
            anchor_pt, radius=3, color="#1a1a2e", fill=True, fill_opacity=0.9, weight=1
        ).add_to(m)

        gt_points   = [[float(rnn_true[i, t, 0]), float(rnn_true[i, t, 1])]
                       for t in range(rnn_true.shape[1])]
        rnn_points  = [[float(rnn_pred[i, t, 0]), float(rnn_pred[i, t, 1])]
                       for t in range(rnn_pred.shape[1])]
        ar_points   = [[float(ar_pred[i, t, 0]),  float(ar_pred[i, t, 1])]
                       for t in range(ar_pred.shape[1])]

        folium.PolyLine([anchor_pt] + gt_points,  color="#2196F3", weight=1.5, opacity=0.7).add_to(m)
        folium.PolyLine([anchor_pt] + rnn_points, color="#FF9800", weight=1.5, opacity=0.7).add_to(m)
        folium.PolyLine([anchor_pt] + ar_points,  color="#4CAF50", weight=1.5, opacity=0.7).add_to(m)

    m.save(str(out))
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary(rnn: dict, rnn_ar: dict) -> dict:
    rnn_pos  = get_position_metrics(rnn)
    ar_pos   = get_position_metrics(rnn_ar)
    rnn_traj_m  = get_trajectory_metrics(rnn)
    ar_traj_m   = get_trajectory_metrics(rnn_ar)

    def delta(a, b, key):
        av, bv = a.get(key, 0), b.get(key, 0)
        if av == 0:
            return None
        return round((bv - av) / av * 100, 2)

    return {
        "dataset": {
            "coast": rnn.get("coast"),
            "region": rnn.get("region"),
            "horizon_hours_actual": rnn.get("horizon_hours_actual"),
            "samples_test": rnn.get("samples_test"),
        },
        "flat_head": {
            "metrics_position": rnn_pos,
            "metrics_trajectory": rnn_traj_m,
            "compute": rnn.get("compute", {}),
            "epochs_ran": rnn.get("training", {}).get("epochs_ran"),
            "best_val_loss": rnn.get("training", {}).get("best_val_loss"),
        },
        "autoregressive": {
            "metrics_position": ar_pos,
            "metrics_trajectory": ar_traj_m,
            "compute": rnn_ar.get("compute", {}),
            "epochs_ran": rnn_ar.get("training", {}).get("epochs_ran"),
            "best_val_loss": rnn_ar.get("training", {}).get("best_val_loss"),
            "teacher_forcing_ratio": rnn_ar.get("architecture", {}).get("teacher_forcing_ratio"),
        },
        "ar_vs_flat_delta_pct": {
            "mean_error_km":   delta(rnn_pos, ar_pos, "mean_error_km"),
            "median_error_km": delta(rnn_pos, ar_pos, "median_error_km"),
            "p90_error_km":    delta(rnn_pos, ar_pos, "p90_error_km"),
            "mean_error_nm":   delta(rnn_pos, ar_pos, "mean_error_nm"),
            "median_error_nm": delta(rnn_pos, ar_pos, "median_error_nm"),
            "p90_error_nm":    delta(rnn_pos, ar_pos, "p90_error_nm"),
            "mean_ade_km":     delta(rnn_traj_m, ar_traj_m, "mean_ade_km"),
            "mean_ade_nm":     delta(rnn_traj_m, ar_traj_m, "mean_ade_nm"),
            "param_overhead":  delta(rnn.get("compute", {}), rnn_ar.get("compute", {}), "param_count"),
            "gpu_overhead":    delta(rnn.get("compute", {}), rnn_ar.get("compute", {}), "peak_gpu_mb"),
            "throughput_drop": delta(rnn.get("compute", {}), rnn_ar.get("compute", {}),
                                     "avg_throughput_samples_per_sec"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare flat-head vs AR-RNN results.")
    parser.add_argument("--rnn-metrics",    type=Path, default=None)
    parser.add_argument("--rnn-ar-metrics", type=Path, default=None)
    parser.add_argument("--rnn-trajs",      type=Path, default=None,
                        help="Optional: path to *_sample_trajectories.json from RNN run.")
    parser.add_argument("--rnn-ar-trajs",   type=Path, default=None,
                        help="Optional: path to *_ar_sample_trajectories.json from RNN_AR run.")
    parser.add_argument("--auto-discover",  action="store_true",
                        help="Search data/results/ for the latest metrics files.")
    parser.add_argument("--rnn-type",       default="lstm",
                        help="Used by --auto-discover (default: lstm).")
    parser.add_argument("--output-dir",     type=Path, default=None,
                        help="Where to write comparison outputs (default: next to rnn-metrics).")
    parser.add_argument("--n-vessels",      type=int, default=30,
                        help="Number of sample vessel trajectories to show on the map.")
    args = parser.parse_args()

    results_root = PROJECT_ROOT / "proj" / "project" / "data" / "results"

    if args.auto_discover:
        flat_path, ar_path = discover_metrics(results_root, args.rnn_type)
        if flat_path is None or ar_path is None:
            sys.exit(
                f"Could not auto-discover metrics for rnn_type={args.rnn_type!r} "
                f"under {results_root}. Run both models first."
            )
        args.rnn_metrics    = flat_path
        args.rnn_ar_metrics = ar_path

    if args.rnn_metrics is None or args.rnn_ar_metrics is None:
        parser.error("Provide --rnn-metrics and --rnn-ar-metrics, or use --auto-discover.")

    rnn    = load_json(args.rnn_metrics)
    rnn_ar = load_json(args.rnn_ar_metrics)

    # Attach training history if stored separately in the JSON (we embed it inline).
    # The JSON written by both model scripts includes "metrics" but NOT "training_history".
    # We reconstruct it from the "training" block for the curves.
    # NOTE: epoch-level history is NOT written to the metrics JSON currently, so the
    # training curves plot is skipped when history is absent.
    rnn["training_history"]    = rnn.get("history") or []
    rnn_ar["training_history"] = rnn_ar.get("history") or []

    outdir: Path = args.output_dir or args.rnn_metrics.parent
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nComparing:")
    print(f"  Flat head : {args.rnn_metrics}")
    print(f"  AR        : {args.rnn_ar_metrics}")
    print(f"  Output    : {outdir}")
    print()

    rnn_pos   = get_position_metrics(rnn)
    rnn_ar_pos = get_position_metrics(rnn_ar)

    horizon_step = int(rnn.get("horizon_step_index", 0))

    # ---- Training curves (only if history is embedded) ----
    if rnn["training_history"] and rnn_ar["training_history"]:
        print("Plotting training curves...")
        plot_training_curves(rnn, rnn_ar, outdir / "comparison_training_curves.png")
    else:
        print("Training history not embedded in metrics JSON — skipping loss curves.")

    # ---- Metric bars ----
    print("Plotting metric bars...")
    plot_metrics_bar(rnn_pos, rnn_ar_pos, outdir / "comparison_metrics_bar.png")

    # ---- Compute comparison ----
    print("Plotting compute comparison...")
    plot_compute(rnn, rnn_ar, outdir / "comparison_compute.png")

    # ---- Trajectory-based plots (need sample traj JSONs) ----
    rnn_traj_path = args.rnn_trajs or (
        args.rnn_metrics.parent / f"{args.rnn_type}_sample_trajectories.json"
    )
    rnn_ar_traj_path = args.rnn_ar_trajs or (
        args.rnn_ar_metrics.parent / f"{args.rnn_type}_ar_sample_trajectories.json"
    )

    rnn_traj    = load_traj_json(rnn_traj_path)
    rnn_ar_traj = load_traj_json(rnn_ar_traj_path)

    if rnn_traj and rnn_ar_traj:
        print("Plotting error histograms...")
        plot_error_hist(rnn_traj, rnn_ar_traj, horizon_step, outdir / "comparison_error_hist.png")

        print("Plotting scatter plots...")
        plot_scatter(rnn_traj, rnn_ar_traj, horizon_step, outdir / "comparison_scatter.png")

        print("Building folium map...")
        build_map(rnn_traj, rnn_ar_traj, args.n_vessels, outdir / "comparison_map.html")
    else:
        missing = []
        if not rnn_traj:
            missing.append(str(rnn_traj_path))
        if not rnn_ar_traj:
            missing.append(str(rnn_ar_traj_path))
        print(f"Missing trajectory files — skipping histogram/scatter/map:\n  " + "\n  ".join(missing))

    # ---- Summary JSON ----
    summary = build_summary(rnn, rnn_ar)
    summary_path = outdir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved → {summary_path}")

    # ---- Console summary table ----
    print("\n" + "=" * 72)
    print(f"{'Metric':<30} {'Flat head':>15} {'AR':>15} {'AR vs flat':>10}")
    print("=" * 72)
    rows = [
        ("mae_lat",          "MAE lat (°)"),
        ("mae_lon",          "MAE lon (°)"),
        ("mean_error_km",    "Mean error (km)"),
        ("median_error_km",  "Median error (km)"),
        ("p90_error_km",     "P90 error (km)"),
        ("mean_error_nm",    "Mean error (nm)"),
        ("median_error_nm",  "Median error (nm)"),
    ]
    for key, label in rows:
        rv = rnn_pos.get(key, float("nan"))
        av = rnn_ar_pos.get(key, float("nan"))
        pct = (av - rv) / rv * 100 if rv != 0 else float("nan")
        arrow = "▼" if pct < 0 else "▲"
        print(f"  {label:<28} {rv:>15.4f} {av:>15.4f} {arrow}{abs(pct):>8.1f}%")

    rnn_c   = rnn.get("compute", {})
    ar_c    = rnn_ar.get("compute", {})
    print("-" * 72)
    print(f"  {'Parameters':<28} {rnn_c.get('param_count', 0):>15,} {ar_c.get('param_count', 0):>15,}")
    print(f"  {'Peak GPU (MB)':<28} {rnn_c.get('peak_gpu_mb', 0):>15.1f} {ar_c.get('peak_gpu_mb', 0):>15.1f}")
    print(f"  {'Throughput (samp/s)':<28} {rnn_c.get('avg_throughput_samples_per_sec', 0):>15.1f} "
          f"{ar_c.get('avg_throughput_samples_per_sec', 0):>15.1f}")
    print(f"  {'Train time (s)':<28} {rnn_c.get('total_train_sec', 0):>15.1f} "
          f"{ar_c.get('total_train_sec', 0):>15.1f}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
