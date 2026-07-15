#!/usr/bin/env python3
"""Folium + PNG map for exp_coastal AR 12h predictions (history + GT + prediction)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import haversine_km

SUBROOT = PROJECT
RESULTS = SUBROOT / "data/results/USA Combined/unknown/exp_coastal/AR_12h/RNN_AR_LSTM"
DEFAULT_TRAJS = RESULTS / "lstm_ar_sample_trajectories.json"
DEFAULT_OUT = RESULTS / "map_ar12h_examples.html"
HISTORY_STEPS = 72  # 12h @ 10 min


def load_trajs(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(d[k], dtype=np.float64) for k in ("y_true", "y_pred", "anchor")}


def load_history_12h(
    input_path: Path,
    anchor: np.ndarray,
    *,
    sample_size: int = 400_000,
    seed: int = 42,
    history_steps: int = HISTORY_STEPS,
    full_history: int = 144,
) -> np.ndarray:
    import pandas as pd
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
        df = pd.read_parquet(input_path, columns=read_cols)
    else:
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


def pick_indices(
    traj: dict,
    *,
    seed: int = 7,
    n_each: int = 2,
    min_net_km: float = 5.0,
    prefer_long: bool = False,
) -> list[tuple[int, str]]:
    rng = np.random.default_rng(seed)
    anchor = traj["anchor"]
    gt = traj["y_true"]
    pred = traj["y_pred"]
    err = haversine_km(gt[:, -1, 0], gt[:, -1, 1], pred[:, -1, 0], pred[:, -1, 1])
    net = haversine_km(anchor[:, 0], anchor[:, 1], gt[:, -1, 0], gt[:, -1, 1])
    moving = net > min_net_km
    low_pool = np.where(moving & (err < 15))[0]
    mid_pool = np.where(moving & (err >= 15) & (err < 40))[0]
    high_pool = np.where(moving & (err >= 40))[0]
    if len(low_pool) == 0:
        low_pool = np.argsort(err)[:20]
    if len(mid_pool) == 0:
        mid_pool = np.argsort(err)[len(err) // 3 : 2 * len(err) // 3]
    if len(high_pool) == 0:
        high_pool = np.argsort(err)[-20:]

    picks: list[tuple[int, str]] = []
    for pool, label in [
        (low_pool, "Low error"),
        (mid_pool, "Medium error"),
        (high_pool, "High error"),
    ]:
        if prefer_long and len(pool) > 0:
            # Prefer longest true displacements within each error band
            order = pool[np.argsort(-net[pool])]
            chosen = order[: min(n_each, len(order))]
        else:
            chosen = rng.choice(pool, size=min(n_each, len(pool)), replace=False)
        for j, idx in enumerate(chosen):
            suffix = f" #{j + 1}" if n_each > 1 else ""
            picks.append((int(idx), f"{label}{suffix} ({net[idx]:.0f} km)"))
    return picks


def pick_longest_tracks(traj: dict, *, n: int = 6, min_net_km: float = 40.0) -> list[tuple[int, str]]:
    """Top-N longest true 12h displacements, labeled with FDE."""
    anchor = traj["anchor"]
    gt = traj["y_true"]
    pred = traj["y_pred"]
    err = haversine_km(gt[:, -1, 0], gt[:, -1, 1], pred[:, -1, 0], pred[:, -1, 1])
    net = haversine_km(anchor[:, 0], anchor[:, 1], gt[:, -1, 0], gt[:, -1, 1])
    pool = np.where(net >= min_net_km)[0]
    if len(pool) == 0:
        pool = np.arange(len(net))
    order = pool[np.argsort(-net[pool])][:n]
    picks = []
    for rank, idx in enumerate(order, start=1):
        picks.append(
            (
                int(idx),
                f"Long #{rank} ({net[idx]:.0f} km true, FDE {err[idx]:.0f} km)",
            )
        )
    return picks


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
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    legend = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
    border:2px solid #333;border-radius:8px;padding:12px;font-size:13px;line-height:1.6;
    max-width:300px;box-shadow:0 2px 8px rgba(0,0,0,.15);">
    <b>AR 12h → 12h forecast</b><br>
    <span style="color:#757575">&#9644;</span> Gray — 12h history<br>
    <span style="color:orange">&#9733;</span> Orange — NOW (anchor)<br>
    <span style="color:#1565C0">&#9644;</span> Blue — true future (12h)<br>
    <span style="color:#1B5E20">&#9644;</span> Green dashed — AR prediction<br>
    <span style="color:#F44336">&#9679;</span> Red — true endpoint @ 12h<br>
    <span style="color:#FF9800">&#9679;</span> Orange — predicted endpoint @ 12h<br>
    <small>Toggle layers (top-right) to focus on one example.</small>
    </div>
    """
    # Keep existing legend string replacement minimal — already similar
    m.get_root().html.add_child(Element(legend))

    for row_i, title in indices:
        hist = histories[row_i]
        anchor = traj["anchor"][row_i]
        gt = traj["y_true"][row_i]
        pred = traj["y_pred"][row_i]
        err = float(haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1]))

        layer = folium.FeatureGroup(name=f"{title} (FDE {err:.1f} km)", show=True)
        hist_pts = [[float(p[0]), float(p[1])] for p in hist]
        anchor_pt = [float(anchor[0]), float(anchor[1])]
        gt_pts = [[float(gt[t, 0]), float(gt[t, 1])] for t in range(len(gt))]
        pred_pts = [[float(pred[t, 0]), float(pred[t, 1])] for t in range(len(pred))]

        folium.PolyLine(hist_pts, color="#9E9E9E", weight=3, opacity=0.85, popup="12h history").add_to(layer)
        folium.PolyLine(
            [anchor_pt] + gt_pts, color="#1565C0", weight=4, opacity=0.9, popup="Ground truth 12h"
        ).add_to(layer)
        folium.PolyLine(
            [anchor_pt] + pred_pts,
            color="#1B5E20",
            weight=3,
            opacity=0.85,
            dash_array="6 4",
            popup="AR prediction",
        ).add_to(layer)

        folium.Marker(
            anchor_pt, icon=folium.Icon(color="orange", icon="star", prefix="fa"), popup="NOW"
        ).add_to(layer)
        folium.CircleMarker(
            gt_pts[-1],
            radius=10,
            color="#B71C1C",
            fill=True,
            fill_color="#F44336",
            fill_opacity=1,
            weight=2,
            popup="True @ 12h",
        ).add_to(layer)
        folium.CircleMarker(
            pred_pts[-1],
            radius=9,
            color="#E65100",
            fill=True,
            fill_color="#FF9800",
            fill_opacity=1,
            weight=2,
            popup=f"Predicted @ 12h | err {err:.2f} km",
        ).add_to(layer)
        folium.Marker(
            location=[(anchor_pt[0] + hist_pts[0][0]) / 2, (anchor_pt[1] + hist_pts[0][1]) / 2],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;font-weight:700;background:rgba(255,255,255,0.92);'
                    f'padding:3px 6px;border-radius:4px;border:1px solid #555;">{title}<br>'
                    f"FDE {err:.1f} km</div>"
                )
            ),
        ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    parser = argparse.ArgumentParser(description="Map AR 12h experiment trajectories.")
    parser.add_argument("--trajs", type=Path, default=DEFAULT_TRAJS)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/combined_filtered_smart_coastal/train.parquet")
    )
    parser.add_argument("--html", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-each", type=int, default=2, help="Examples per error band")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--long-tracks",
        action="store_true",
        help="Select the longest true 12h displacements (instead of error bands).",
    )
    parser.add_argument("--n-long", type=int, default=6, help="How many long tracks to show")
    parser.add_argument(
        "--min-net-km",
        type=float,
        default=40.0,
        help="Minimum true net displacement (km) for --long-tracks",
    )
    args = parser.parse_args()

    traj = load_trajs(args.trajs)
    if args.long_tracks:
        indices = pick_longest_tracks(traj, n=args.n_long, min_net_km=args.min_net_km)
    else:
        indices = pick_indices(traj, seed=args.seed, n_each=args.n_each, prefer_long=True)
    print(f"Selected {len(indices)} examples:")
    for i, title in indices:
        err = float(
            haversine_km(
                traj["y_true"][i, -1, 0],
                traj["y_true"][i, -1, 1],
                traj["y_pred"][i, -1, 0],
                traj["y_pred"][i, -1, 1],
            )
        )
        net = float(
            haversine_km(
                traj["anchor"][i, 0],
                traj["anchor"][i, 1],
                traj["y_true"][i, -1, 0],
                traj["y_true"][i, -1, 1],
            )
        )
        print(f"  [{i}] {title}: FDE={err:.1f} km  net={net:.1f} km  anchor={traj['anchor'][i]}")

    histories = {i: load_history_12h(args.input, traj["anchor"][i]) for i, _ in indices}
    build_folium_map(traj, histories, indices, args.html)
    print(f"Saved HTML: {args.html}")


if __name__ == "__main__":
    main()
