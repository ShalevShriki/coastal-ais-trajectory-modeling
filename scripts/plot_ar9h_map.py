#!/usr/bin/env python3
"""Folium + PNG map for exp_final AR 9h predictions (history + GT + prediction)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import haversine_km, trajectory_splits

SUBROOT = PROJECT
RESULTS = SUBROOT / "data/results/USA Combined/unknown/exp_final/AR_9h/RNN_AR_LSTM"
DEFAULT_TRAJS = RESULTS / "lstm_ar_sample_trajectories.json"
DEFAULT_OUT = RESULTS / "map_ar9h_examples.html"
HISTORY_STEPS = 54  # 9h @ 10 min


def load_trajs(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(d[k], dtype=np.float64) for k in ("y_true", "y_pred", "anchor")}


def load_history_9h(
    input_path: Path,
    anchor: np.ndarray,
    *,
    sample_size: int = 400_000,
    seed: int = 42,
    history_steps: int = HISTORY_STEPS,
    full_history: int = 144,
) -> np.ndarray:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(input_path)
    names = pf.schema.names
    id_col = "traj_id" if "traj_id" in names else "mmsi"
    h_start = full_history - history_steps
    hist_cols = []
    for t in range(h_start, full_history):
        hist_cols.extend([f"x_t{t:03d}_lat", f"x_t{t:03d}_lon"])
    read_cols = [id_col] + hist_cols

    total = pf.metadata.num_rows
    if total <= sample_size:
        import pandas as pd
        df = pd.read_parquet(input_path, columns=read_cols)
    else:
        import pandas as pd
        n_groups = pf.metadata.num_row_groups
        frac = sample_size / total
        n_pick = max(1, min(n_groups, int(np.ceil(n_groups * frac))))
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(n_groups, size=n_pick, replace=False).tolist())
        df = pd.concat(
            [pf.read_row_group(g, columns=read_cols).to_pandas() for g in chosen],
            ignore_index=True,
        )
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=seed).reset_index(drop=True)

    last_lat = f"x_t{full_history - 1:03d}_lat"
    last_lon = f"x_t{full_history - 1:03d}_lon"
    dlat = df[last_lat].to_numpy() - anchor[0]
    dlon = df[last_lon].to_numpy() - anchor[1]
    row = int(np.argmin(dlat * dlat + dlon * dlon))

    hist = np.empty((history_steps, 2), dtype=np.float64)
    for t in range(history_steps):
        src = h_start + t
        hist[t, 0] = df.iloc[row][f"x_t{src:03d}_lat"]
        hist[t, 1] = df.iloc[row][f"x_t{src:03d}_lon"]
    return hist


def pick_indices(traj: dict, *, seed: int = 7) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    anchor = traj["anchor"]
    gt = traj["y_true"]
    pred = traj["y_pred"]
    err = haversine_km(
        gt[:, -1, 0], gt[:, -1, 1], pred[:, -1, 0], pred[:, -1, 1]
    )
    net = haversine_km(anchor[:, 0], anchor[:, 1], gt[:, -1, 0], gt[:, -1, 1])
    moving = net > 5.0
    low_pool = np.where(moving & (err < 15))[0]
    mid_pool = np.where(moving & (err >= 15) & (err < 40))[0]
    if len(low_pool) == 0:
        low_pool = np.argsort(err)[:10]
    if len(mid_pool) == 0:
        mid_pool = np.argsort(err)[-10:]
    return int(rng.choice(low_pool)), int(rng.choice(mid_pool))


def build_folium_map(
    traj: dict,
    histories: dict[int, np.ndarray],
    indices: list[tuple[int, str]],
    out_html: Path,
) -> None:
    import folium
    from folium import Element

    anchors = traj["anchor"][[i for i, _ in indices]]
    m = folium.Map(
        location=[float(anchors[:, 0].mean()), float(anchors[:, 1].mean())],
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
    )

    legend = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
    border:2px solid #333;border-radius:8px;padding:12px;font-size:13px;line-height:1.6;">
    <b>AR 9h → 12h forecast</b><br>
    <span style="color:#757575">&#9644;</span> Gray — 9h history<br>
    <span style="color:orange">&#9733;</span> Orange — NOW (anchor)<br>
    <span style="color:#1565C0">&#9644;</span> Blue — true future (12h)<br>
    <span style="color:#1B5E20">&#9644;</span> Green — AR prediction<br>
    <span style="color:#F44336">&#9679;</span> Red — true endpoint @ 12h<br>
    <span style="color:#FF9800">&#9679;</span> Orange — predicted endpoint @ 12h
    </div>
    """
    m.get_root().html.add_child(Element(legend))

    for row_i, title in indices:
        hist = histories[row_i]
        anchor = traj["anchor"][row_i]
        gt = traj["y_true"][row_i]
        pred = traj["y_pred"][row_i]
        err = float(
            haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1])
        )

        layer = folium.FeatureGroup(name=f"{title} (FDE {err:.1f} km)", show=True)
        hist_pts = [[float(p[0]), float(p[1])] for p in hist]
        anchor_pt = [float(anchor[0]), float(anchor[1])]
        gt_pts = [[float(gt[t, 0]), float(gt[t, 1])] for t in range(len(gt))]
        pred_pts = [[float(pred[t, 0]), float(pred[t, 1])] for t in range(len(pred))]

        folium.PolyLine(hist_pts, color="#9E9E9E", weight=3, opacity=0.85, popup="9h history").add_to(layer)
        folium.PolyLine([anchor_pt] + gt_pts, color="#1565C0", weight=4, opacity=0.9, popup="Ground truth 12h").add_to(layer)
        folium.PolyLine([anchor_pt] + pred_pts, color="#1B5E20", weight=3, opacity=0.85, dash_array="6 4", popup="AR prediction").add_to(layer)

        folium.Marker(anchor_pt, icon=folium.Icon(color="orange", icon="star", prefix="fa"), popup="NOW").add_to(layer)
        folium.CircleMarker(
            gt_pts[-1], radius=10, color="#B71C1C", fill=True, fill_color="#F44336", fill_opacity=1, weight=2,
            popup=f"True @ 12h",
        ).add_to(layer)
        folium.CircleMarker(
            pred_pts[-1], radius=9, color="#E65100", fill=True, fill_color="#FF9800", fill_opacity=1, weight=2,
            popup=f"Predicted @ 12h | err {err:.2f} km",
        ).add_to(layer)
        folium.Marker(
            location=[(anchor_pt[0] + hist_pts[0][0]) / 2, (anchor_pt[1] + hist_pts[0][1]) / 2],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:700;background:rgba(255,255,255,0.92);'
                f'padding:3px 6px;border-radius:4px;border:1px solid #555;">{title}</div>'
            ),
        ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def build_png_map(
    traj: dict,
    histories: dict[int, np.ndarray],
    indices: list[tuple[int, str]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    try:
        import contextily as ctx
        from pyproj import Transformer
        to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        use_basemap = True
    except ImportError:
        use_basemap = False

    fig, axes = plt.subplots(1, len(indices), figsize=(7 * len(indices), 7))
    if len(indices) == 1:
        axes = [axes]

    for ax, (row_i, title) in zip(axes, indices):
        hist = histories[row_i]
        anchor = traj["anchor"][row_i]
        gt = traj["y_true"][row_i]
        pred = traj["y_pred"][row_i]
        err = float(haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1]))

        lats = np.concatenate([hist[:, 0], [anchor[0]], gt[:, 0], pred[:, 0]])
        lons = np.concatenate([hist[:, 1], [anchor[1]], gt[:, 1], pred[:, 1]])
        pad_lat = max((lats.max() - lats.min()) * 0.15, 0.05)
        pad_lon = max((lons.max() - lons.min()) * 0.15, 0.05)

        if use_basemap:
            xmin, ymin = to_merc.transform(lons.min() - pad_lon, lats.min() - pad_lat)
            xmax, ymax = to_merc.transform(lons.max() + pad_lon, lats.max() + pad_lat)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            try:
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom="auto")
            except Exception:
                use_basemap = False

        def plot_line(pts, **kw):
            if use_basemap:
                x, y = to_merc.transform(pts[:, 1], pts[:, 0])
                ax.plot(x, y, **kw)
            else:
                ax.plot(pts[:, 1], pts[:, 0], **kw)

        def plot_pt(lat, lon, **kw):
            if use_basemap:
                x, y = to_merc.transform(lon, lat)
                ax.scatter(x, y, **kw)
            else:
                ax.scatter(lon, lat, **kw)

        true_track = np.vstack([anchor[None, :], gt])
        pred_track = np.vstack([anchor[None, :], pred])
        plot_line(hist, color="#9E9E9E", linewidth=2, label="9h history")
        plot_line(true_track, color="#1565C0", linewidth=2.5, label="Ground truth")
        plot_line(pred_track, color="#1B5E20", linewidth=2.5, linestyle="--", label="AR prediction")
        plot_pt(anchor[0], anchor[1], s=80, marker="*", color="orange", edgecolors="k", zorder=5)
        plot_pt(gt[-1, 0], gt[-1, 1], s=60, marker="o", color="#F44336", edgecolors="k", zorder=5)
        plot_pt(pred[-1, 0], pred[-1, 1], s=60, marker="D", color="#FF9800", edgecolors="k", zorder=5)

        ax.set_title(f"{title}\nFDE @ 12h = {err:.1f} km", fontsize=11)
        if not use_basemap:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_axis_off()

    legend = [
        Line2D([0], [0], color="#9E9E9E", linewidth=2, label="9h history"),
        Line2D([0], [0], color="#1565C0", linewidth=2, label="Ground truth"),
        Line2D([0], [0], color="#1B5E20", linewidth=2, linestyle="--", label="AR prediction"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="orange", markersize=12, label="NOW"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10)
    fig.suptitle("AR 9h context → 12h prediction (USA Combined, smart_motion)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map AR 9h experiment trajectories.")
    parser.add_argument("--trajs", type=Path, default=DEFAULT_TRAJS)
    parser.add_argument("--input", type=Path, default=Path("data/processed/combined_filtered_smart/train.parquet"))
    parser.add_argument("--html", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--png", type=Path, default=RESULTS / "map_ar9h_examples.png")
    args = parser.parse_args()

    traj = load_trajs(args.trajs)
    low_i, mid_i = pick_indices(traj)
    indices = [(low_i, "Low error"), (mid_i, "Higher error")]
    histories = {
        i: load_history_9h(args.input, traj["anchor"][i])
        for i, _ in indices
    }

    build_folium_map(traj, histories, indices, args.html)
    build_png_map(traj, histories, indices, args.png)
    print(f"Saved HTML: {args.html}")
    print(f"Saved PNG:  {args.png}")
    print("Open HTML in browser, or view PNG in the IDE.")


if __name__ == "__main__":
    main()
