"""
visualize_model_results.py
--------------------------
Rich visualisations for RNN / RNN-AR trajectory prediction results.

Reads metrics JSON + sample-trajectory JSON produced by RNN.py / RNN_AR.py and
writes:

  error_vs_horizon.png          — median / mean / P90 error vs hours ahead
  error_hist_multi_horizon.png  — histograms at selected horizons
  ade_boxplot_horizons.png      — boxplots of per-sample error at each horizon
  trajectory_basemap.png        — static map with sample tracks + AR step markers
  map_ar_unroll.html            — folium map showing full AR unroll + middle points
  map_horizon_{N}h.html         — folium map focused on one horizon (per --horizons)

Usage
-----
  python visualize_model_results.py \\
      --metrics  data/results/.../lstm_metrics.json \\
      --trajs    data/results/.../lstm_sample_trajectories.json \\
      --ar-metrics data/results/.../lstm_ar_metrics.json \\
      --ar-trajs   data/results/.../lstm_ar_sample_trajectories.json

  python visualize_model_results.py --auto-discover
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proj.project.window_data import haversine_km, haversine_nm

COLORS = {
    "gt": "#1565C0",
    "anchor": "#1a1a2e",
    "rnn": "#FF9800",
    "rnn_ar": "#2E7D32",
    "rnn_ar_mid": "#66BB6A",
    "transformer": "#7B1FA2",
}
LABELS = {
    "rnn": "LSTM (flat)",
    "rnn_ar": "LSTM-AR",
    "transformer": "Transformer",
}


def active_models(
    traj_rnn: dict | None,
    traj_ar: dict | None,
    traj_transformer: dict | None,
) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for tag, traj in [
        ("rnn", traj_rnn),
        ("rnn_ar", traj_ar),
        ("transformer", traj_transformer),
    ]:
        if traj is not None:
            out.append((tag, traj))
    return out


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_traj(path: Path) -> dict:
    data = load_json(path)
    return {
        "y_true": np.array(data["y_true"], dtype=np.float64),
        "y_pred": np.array(data["y_pred"], dtype=np.float64),
        "anchor": np.array(data["anchor"], dtype=np.float64),
    }


def step_minutes(metrics: dict) -> float:
    future_steps = metrics["future_steps"]
    window_hours = metrics.get("window_hours") or metrics.get("horizon_hours_actual", 12.0)
    return window_hours * 60.0 / future_steps


def step_to_hours(step: int, step_min: float) -> float:
    return (step + 1) * step_min / 60.0


def resolve_horizon_steps(metrics: dict, horizon_hours: list[float]) -> list[tuple[float, int]]:
    sm = step_minutes(metrics)
    future_steps = metrics["future_steps"]
    out: list[tuple[float, int]] = []
    for h in horizon_hours:
        step = int(round(h * 60.0 / sm)) - 1
        step = max(0, min(step, future_steps - 1))
        out.append((step_to_hours(step, sm), step))
    return out


def per_sample_errors_at_step(y_true: np.ndarray, y_pred: np.ndarray, step: int) -> np.ndarray:
    return haversine_nm(
        y_true[:, step, 0], y_true[:, step, 1],
        y_pred[:, step, 0], y_pred[:, step, 1],
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_error_vs_horizon(
    metrics: dict,
    models: list[tuple[str, dict]],
    out: Path,
) -> None:
    sm = step_minutes(metrics)
    n_steps = metrics["future_steps"]
    hours = np.array([step_to_hours(t, sm) for t in range(n_steps)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    stat_fns = [
        ("median", np.median),
        ("mean", np.mean),
        ("P90", lambda x: np.percentile(x, 90)),
    ]

    for ax, (stat_name, fn) in zip(axes, stat_fns):
        for tag, traj in models:
            yt, yp = traj["y_true"], traj["y_pred"]
            vals = np.array([
                fn(haversine_nm(yt[:, t, 0], yt[:, t, 1], yp[:, t, 0], yp[:, t, 1]))
                for t in range(n_steps)
            ])
            ax.plot(hours, vals, label=LABELS[tag], color=COLORS[tag], linewidth=2)
        ax.set_xlabel("Hours ahead")
        ax.set_ylabel("Error (nautical miles)")
        ax.set_title(f"{stat_name.title()} position error")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)

    coast = metrics.get("coast", "")
    fig.suptitle(f"Error vs prediction horizon — {coast}", fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


def plot_multi_horizon_histograms(
    metrics: dict,
    models: list[tuple[str, dict]],
    horizon_hours: list[float],
    out: Path,
) -> None:
    horizons = resolve_horizon_steps(metrics, horizon_hours)
    n = len(horizons)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (h_label, step) in zip(axes, horizons):
        for tag, traj in models:
            errs = per_sample_errors_at_step(traj["y_true"], traj["y_pred"], step)
            ax.hist(errs, bins=50, alpha=0.55, color=COLORS[tag], label=LABELS[tag])
        ax.set_xlabel("Error (nm)")
        ax.set_ylabel("Count")
        ax.set_title(f"{h_label:.1f} h ahead (step {step + 1})")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes[len(horizons):]:
        ax.set_visible(False)

    fig.suptitle("Error distribution at selected horizons", fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


def plot_ade_boxplot(
    metrics: dict,
    models: list[tuple[str, dict]],
    horizon_hours: list[float],
    out: Path,
) -> None:
    horizons = resolve_horizon_steps(metrics, horizon_hours)
    labels = [f"{h:.0f}h" for h, _ in horizons]

    if not models:
        return

    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, (tag, traj) in enumerate(models):
        data = [
            per_sample_errors_at_step(traj["y_true"], traj["y_pred"], step)
            for _, step in horizons
        ]
        offset = (mi - (n_models - 1) / 2) * width
        bp = ax.boxplot(
            data, positions=x + offset, widths=width * 0.9, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[tag])
            patch.set_alpha(0.7)

    legend_patches = [
        plt.matplotlib.patches.Patch(color=COLORS[tag], alpha=0.7, label=LABELS[tag])
        for tag, _ in models
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Per-sample error (nm)")
    ax.set_title("Error distribution across test windows")
    ax.legend(handles=legend_patches)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save(fig, out)


def plot_trajectory_basemap(
    models: list[tuple[str, dict]],
    traj_ar: dict | None,
    n_vessels: int,
    out: Path,
    *,
    mid_every_steps: int = 6,
) -> None:
    try:
        import contextily as ctx
        from pyproj import Transformer
    except ImportError:
        print("  contextily/pyproj not available — skipping basemap plot")
        return

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    ref = traj_ar or (models[0][1] if models else None)
    if ref is None:
        return

    rng = np.random.default_rng(7)
    n = min(n_vessels, len(ref["y_true"]))
    idx = rng.choice(len(ref["y_true"]), size=n, replace=False)

    all_lats = ref["anchor"][idx, 0].tolist()
    all_lons = ref["anchor"][idx, 1].tolist()
    for i in idx:
        all_lats.extend(ref["y_true"][i, :, 0].tolist())
        all_lons.extend(ref["y_true"][i, :, 1].tolist())

    pad_lat = (max(all_lats) - min(all_lats)) * 0.08 or 0.5
    pad_lon = (max(all_lons) - min(all_lons)) * 0.08 or 0.5
    bounds = {
        "south": min(all_lats) - pad_lat,
        "north": max(all_lats) + pad_lat,
        "west": min(all_lons) - pad_lon,
        "east": max(all_lons) + pad_lon,
    }

    xmin, ymin = to_merc.transform(bounds["west"], bounds["south"])
    xmax, ymax = to_merc.transform(bounds["east"], bounds["north"])
    dx, dy = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    for src in (ctx.providers.CartoDB.Positron, ctx.providers.OpenStreetMap.Mapnik):
        try:
            ctx.add_basemap(ax, source=src, zoom="auto")
            break
        except Exception:
            continue

    for i in idx:
        anchor = ref["anchor"][i]
        ax_pt, ay_pt = to_merc.transform(anchor[1], anchor[0])
        ax.scatter(ax_pt, ay_pt, s=60, marker="s", color=COLORS["anchor"],
                   edgecolors="white", linewidths=1.2, zorder=6)

        gt = ref["y_true"][i]
        gx, gy = to_merc.transform(gt[:, 1], gt[:, 0])
        ax.plot(gx, gy, color=COLORS["gt"], linewidth=1.8, alpha=0.75, zorder=4)
        ax.scatter(gx[-1], gy[-1], s=40, marker="o", color=COLORS["gt"], zorder=5)

        if traj_ar is not None:
            pred = traj_ar["y_pred"][i]
            px, py = to_merc.transform(pred[:, 1], pred[:, 0])
            ax.plot(px, py, color=COLORS["rnn_ar"], linewidth=1.8, alpha=0.85, zorder=4)
            mid_idx = list(range(0, len(pred), mid_every_steps))
            if (len(pred) - 1) not in mid_idx:
                mid_idx.append(len(pred) - 1)
            ax.scatter(px[mid_idx], py[mid_idx], s=22, color=COLORS["rnn_ar_mid"],
                       edgecolors="white", linewidths=0.6, zorder=5, alpha=0.95)
            ax.scatter(px[-1], py[-1], s=50, marker="D", color=COLORS["rnn_ar"], zorder=5)

        for tag, traj in models:
            if tag == "rnn_ar":
                continue
            pred = traj["y_pred"][i]
            px, py = to_merc.transform(pred[:, 1], pred[:, 0])
            linestyle = "--" if tag == "rnn" else "-."
            marker = "X" if tag == "rnn" else "P"
            ax.plot(px, py, color=COLORS[tag], linewidth=1.5, alpha=0.8, linestyle=linestyle, zorder=3)
            ax.scatter(px[-1], py[-1], s=50, marker=marker, color=COLORS[tag], zorder=5)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["anchor"], markersize=8, label="Anchor"),
        Line2D([0], [0], color=COLORS["gt"], linewidth=2, label="Ground truth"),
    ]
    for tag, _ in models:
        if tag == "rnn_ar":
            legend_elems.append(Line2D([0], [0], color=COLORS["rnn_ar"], linewidth=2, label=LABELS["rnn_ar"]))
            legend_elems.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["rnn_ar_mid"],
                       markersize=7, label="AR intermediate steps")
            )
        else:
            linestyle = "--" if tag == "rnn" else "-."
            legend_elems.append(
                Line2D([0], [0], color=COLORS[tag], linewidth=2, linestyle=linestyle, label=LABELS[tag])
            )
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9)
    ax.set_title(
        f"Sample trajectories (n={n}) — AR middle points every {mid_every_steps} steps",
        fontweight="bold",
    )
    ax.set_axis_off()
    fig.tight_layout()
    _save(fig, out)


def _legend_html(extra: str = "") -> str:
    return f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px;border:2px solid grey;border-radius:5px;font-size:12px">
        <b>Legend</b><br>
        <span style="color:{COLORS['anchor']}">&#9632;</span> Anchor (last known)<br>
        <span style="color:{COLORS['gt']}">&#9644;</span> Ground truth<br>
        <span style="color:{COLORS['rnn']}">&#9644;</span> LSTM flat prediction<br>
        <span style="color:{COLORS['rnn_ar']}">&#9644;</span> LSTM-AR prediction<br>
        <span style="color:{COLORS['transformer']}">&#9644;</span> Transformer prediction<br>
        <span style="color:{COLORS['rnn_ar_mid']}">&#9679;</span> AR intermediate steps<br>
        {extra}
    </div>
    """


def build_ar_unroll_map(
    models: list[tuple[str, dict]],
    traj_ar: dict,
    n_vessels: int,
    out: Path,
    *,
    mid_every_steps: int = 1,
    step_min: float = 10.0,
) -> None:
    try:
        import folium
    except ImportError:
        print("  folium not installed — skipping map")
        return

    rng = np.random.default_rng(7)
    n = min(n_vessels, len(traj_ar["y_true"]))
    idx = rng.choice(len(traj_ar["y_true"]), size=n, replace=False)

    anchors = traj_ar["anchor"][idx]
    center_lat = float(anchors[:, 0].mean())
    center_lon = float(anchors[:, 1].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
    m.get_root().html.add_child(folium.Element(_legend_html()))

    for j, i in enumerate(idx):
        anchor_pt = [float(traj_ar["anchor"][i, 0]), float(traj_ar["anchor"][i, 1])]
        folium.CircleMarker(
            anchor_pt, radius=4, color=COLORS["anchor"], fill=True,
            fill_opacity=0.9, weight=1, popup=f"Vessel {j + 1} anchor",
        ).add_to(m)

        gt = traj_ar["y_true"][i]
        gt_pts = [[float(gt[t, 0]), float(gt[t, 1])] for t in range(gt.shape[0])]
        folium.PolyLine([anchor_pt] + gt_pts, color=COLORS["gt"], weight=2, opacity=0.75).add_to(m)

        ar_pred = traj_ar["y_pred"][i]
        ar_pts = [[float(ar_pred[t, 0]), float(ar_pred[t, 1])] for t in range(ar_pred.shape[0])]
        folium.PolyLine([anchor_pt] + ar_pts, color=COLORS["rnn_ar"], weight=2, opacity=0.8).add_to(m)

        for t in range(0, ar_pred.shape[0], mid_every_steps):
            pt = [float(ar_pred[t, 0]), float(ar_pred[t, 1])]
            err_km = float(haversine_km(gt[t, 0], gt[t, 1], ar_pred[t, 0], ar_pred[t, 1]))
            h_ahead = step_to_hours(t, step_min)
            folium.CircleMarker(
                pt, radius=3, color=COLORS["rnn_ar_mid"], fill=True, fill_opacity=0.85, weight=1,
                popup=f"AR step {t + 1} ({h_ahead:.1f} h) | err {err_km:.2f} km",
            ).add_to(m)

        for tag, traj in models:
            if tag == "rnn_ar":
                continue
            pred = traj["y_pred"][i]
            pts = [[float(pred[t, 0]), float(pred[t, 1])] for t in range(pred.shape[0])]
            dash = "6" if tag == "rnn" else "4 6"
            folium.PolyLine(
                [anchor_pt] + pts, color=COLORS[tag], weight=1.5, opacity=0.65, dash_array=dash,
            ).add_to(m)

    m.save(str(out))
    print(f"  saved → {out}")


def build_horizon_map(
    models: list[tuple[str, dict]],
    traj_ar: dict | None,
    horizon_label: float,
    step: int,
    n_vessels: int,
    out: Path,
    *,
    step_min: float = 10.0,
) -> None:
    try:
        import folium
    except ImportError:
        print("  folium not installed — skipping map")
        return

    ref = traj_ar or (models[0][1] if models else None)
    if ref is None:
        return

    rng = np.random.default_rng(7)
    n = min(n_vessels, len(ref["y_true"]))
    idx = rng.choice(len(ref["y_true"]), size=n, replace=False)

    anchors = ref["anchor"][idx]
    center_lat = float(anchors[:, 0].mean())
    center_lon = float(anchors[:, 1].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
    extra = f"<br><i>Markers show position at {horizon_label:.1f} h</i>"
    m.get_root().html.add_child(folium.Element(_legend_html(extra)))

    for i in idx:
        anchor_pt = [float(ref["anchor"][i, 0]), float(ref["anchor"][i, 1])]
        folium.CircleMarker(
            anchor_pt, radius=3, color=COLORS["anchor"], fill=True, fill_opacity=0.9, weight=1,
        ).add_to(m)

        gt = ref["y_true"][i]
        gt_slice = gt[: step + 1]
        gt_pts = [[float(gt_slice[t, 0]), float(gt_slice[t, 1])] for t in range(gt_slice.shape[0])]
        folium.PolyLine([anchor_pt] + gt_pts, color=COLORS["gt"], weight=2, opacity=0.8).add_to(m)

        gt_end = [float(gt[step, 0]), float(gt[step, 1])]
        folium.CircleMarker(
            gt_end, radius=5, color=COLORS["gt"], fill=True,
            fill_opacity=0.9, popup=f"GT @ {horizon_label:.1f}h",
        ).add_to(m)

        for tag, traj in models:
            if tag == "rnn_ar":
                continue
            pred = traj["y_pred"][i]
            pred_slice = pred[: step + 1]
            pts = [[float(pred_slice[t, 0]), float(pred_slice[t, 1])] for t in range(pred_slice.shape[0])]
            dash = "5" if tag == "rnn" else "4 6"
            folium.PolyLine(
                [anchor_pt] + pts, color=COLORS[tag], weight=1.5, opacity=0.7, dash_array=dash,
            ).add_to(m)
            end = [float(pred[step, 0]), float(pred[step, 1])]
            err = float(haversine_km(gt[step, 0], gt[step, 1], pred[step, 0], pred[step, 1]))
            folium.RegularPolygonMarker(
                end, number_of_sides=4, radius=5, color=COLORS[tag], fill=True,
                fill_opacity=0.9, popup=f"{LABELS[tag]} err {err:.2f} km",
            ).add_to(m)

        if traj_ar is not None:
            pred = traj_ar["y_pred"][i]
            pred_slice = pred[: step + 1]
            ar_pts = [[float(pred_slice[t, 0]), float(pred_slice[t, 1])] for t in range(pred_slice.shape[0])]
            folium.PolyLine([anchor_pt] + ar_pts, color=COLORS["rnn_ar"], weight=2, opacity=0.8).add_to(m)
            for t in range(0, step + 1):
                pt = [float(pred[t, 0]), float(pred[t, 1])]
                err = float(haversine_km(gt[t, 0], gt[t, 1], pred[t, 0], pred[t, 1]))
                h_ahead = step_to_hours(t, step_min)
                folium.CircleMarker(
                    pt, radius=2 if t < step else 5, color=COLORS["rnn_ar_mid"], fill=True,
                    fill_opacity=0.7 if t < step else 0.95, weight=1,
                    popup=f"AR step {t + 1} ({h_ahead:.1f}h) err {err:.2f} km",
                ).add_to(m)

    m.save(str(out))
    print(f"  saved → {out}")


def discover_pair(results_root: Path, rnn_type: str = "lstm"):
    flat_m = sorted(
        results_root.rglob(f"{rnn_type}_metrics.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    ar_m = sorted(
        results_root.rglob(f"{rnn_type}_ar_metrics.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not flat_m:
        return None
    fm = flat_m[0]
    ft = fm.parent / f"{rnn_type}_sample_trajectories.json"
    if not ar_m:
        return fm, ft, None, None
    am = ar_m[0]
    at = am.parent / f"{rnn_type}_ar_sample_trajectories.json"
    return fm, ft, am, at


def main() -> None:
    parser = argparse.ArgumentParser(description="Rich visualisations for RNN / RNN-AR results.")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--trajs", type=Path, default=None)
    parser.add_argument("--ar-metrics", type=Path, default=None)
    parser.add_argument("--ar-trajs", type=Path, default=None)
    parser.add_argument("--transformer-metrics", type=Path, default=None)
    parser.add_argument("--transformer-trajs", type=Path, default=None)
    parser.add_argument("--auto-discover", action="store_true")
    parser.add_argument("--rnn-type", default="lstm")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--n-vessels", type=int, default=25)
    parser.add_argument("--ar-mid-every", type=int, default=6)
    args = parser.parse_args()

    results_root = PROJECT_ROOT / "proj" / "project" / "data" / "results"
    horizon_hours = [float(x.strip()) for x in args.horizons.split(",") if x.strip()]

    if args.auto_discover:
        discovered = discover_pair(results_root, args.rnn_type)
        if discovered is None:
            sys.exit(f"No {args.rnn_type}_metrics.json found under {results_root}")
        args.metrics, args.trajs, args.ar_metrics, args.ar_trajs = discovered

    if args.metrics is None:
        parser.error("Provide --metrics or use --auto-discover.")

    metrics = load_json(args.metrics)
    traj_rnn = load_traj(args.trajs) if args.trajs and args.trajs.exists() else None
    traj_ar = load_traj(args.ar_trajs) if args.ar_trajs and args.ar_trajs.exists() else None
    traj_transformer = (
        load_traj(args.transformer_trajs)
        if args.transformer_trajs and args.transformer_trajs.exists()
        else None
    )
    models = active_models(traj_rnn, traj_ar, traj_transformer)

    if not models:
        sys.exit("No trajectory JSON found. Run RNN.py / RNN_AR.py / transformers.py first.")

    outdir = args.output_dir or (args.metrics.parent.parent / "visualizations")
    outdir.mkdir(parents=True, exist_ok=True)
    sm = step_minutes(metrics)

    print(f"\nVisualizing results → {outdir}")
    print(f"  RNN     : {args.metrics}")
    if args.ar_metrics:
        print(f"  AR      : {args.ar_metrics}")
    if args.transformer_metrics:
        print(f"  Xformer : {args.transformer_metrics}")
    print(f"  Models  : {', '.join(LABELS[t] for t, _ in models)}")
    print()

    print("Error vs horizon curve...")
    plot_error_vs_horizon(metrics, models, outdir / "error_vs_horizon.png")

    print("Multi-horizon histograms...")
    plot_multi_horizon_histograms(
        metrics, models, horizon_hours, outdir / "error_hist_multi_horizon.png",
    )

    print("ADE boxplots...")
    plot_ade_boxplot(metrics, models, horizon_hours, outdir / "ade_boxplot_horizons.png")

    print("Static basemap...")
    plot_trajectory_basemap(
        models, traj_ar, args.n_vessels, outdir / "trajectory_basemap.png",
        mid_every_steps=args.ar_mid_every,
    )

    if traj_ar is not None:
        print("AR unroll map (all middle points)...")
        build_ar_unroll_map(
            models, traj_ar, args.n_vessels, outdir / "map_ar_unroll.html",
            mid_every_steps=1, step_min=sm,
        )

    print("Per-horizon maps...")
    for h_label, step in resolve_horizon_steps(metrics, horizon_hours):
        build_horizon_map(
            models, traj_ar, h_label, step, args.n_vessels,
            outdir / f"map_horizon_{h_label:.0f}h.html",
            step_min=sm,
        )

    print("\nDone.")
    if traj_ar is not None:
        print(f"  AR middle points: {outdir / 'map_ar_unroll.html'}")
    print(f"  Horizon maps:     {outdir / 'map_horizon_*.html'}\n")


if __name__ == "__main__":
    main()
