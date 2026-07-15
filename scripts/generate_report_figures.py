#!/usr/bin/env python3
"""Generate report figures + clear Folium trajectory maps for the final report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import haversine_km

OUT = PROJECT / "data/results/USA Combined/unknown/exp_coastal/report_figures"
COASTAL = PROJECT / "data/results/USA Combined/unknown/exp_coastal"
DATA = PROJECT / "data/processed/combined_filtered_smart_coastal/train.parquet"
# Trajectories / maps in this script are from this coastal AR run:
MODEL_NAME = "AR LSTM 12h"
MODEL_TAG = "exp_coastal/AR_12h_noland"  # no land penalty; coastal filtered data
MODEL_LABEL = f"{MODEL_NAME} ({MODEL_TAG})"


def load_fde(run: str, rel: str) -> float:
    d = json.loads((COASTAL / rel).read_text())
    for m in d["metrics"]:
        name = m.get("model", "")
        if "[" in name:
            continue
        if "ahead)" in name or "recursive 12h position" in name:
            return float(m["median_error_km"])
    raise KeyError(run)


def fig_model_ranking() -> Path:
    rows = [
        ("Kinematic\nbaseline", 102.5),  # report number; confirm below if present
        ("Flat LSTM\n24h", load_fde("flat", "flat_lstm/RNN/lstm_metrics.json")),
        ("Transformer\n24h", load_fde("xf", "transformer/Transformer/transformer_metrics.json")),
        ("AR LSTM 12h\n(no land pen.)", load_fde("ar12n", "AR_12h_noland/RNN_AR_LSTM/lstm_ar_metrics.json")),
        ("AR LSTM 18h", load_fde("ar18", "AR_18h/RNN_AR_LSTM/lstm_ar_metrics.json")),
        ("AR LSTM 12h", load_fde("ar12", "AR_12h/RNN_AR_LSTM/lstm_ar_metrics.json")),
        ("Adaptive", load_fde("ad", "adaptive_multiscale/RNN_AR_adaptive/adaptive_ar_metrics.json")),
        ("AR LSTM 24h", load_fde("ar24", "AR_24h/RNN_AR_LSTM/lstm_ar_metrics.json")),
        ("AR LSTM 9h", load_fde("ar9", "AR_9h/RNN_AR_LSTM/lstm_ar_metrics.json")),
        ("Sliding\n3h×4", load_fde("sl", "sliding_3h/RNN_recursive_sliding/recursive_sliding_metrics.json")),
    ]
    # try overwrite kinematic from flat metrics if available
    try:
        d = json.loads((COASTAL / "flat_lstm/RNN/lstm_metrics.json").read_text())
        for m in d["metrics"]:
            if "inematic" in m.get("model", "") and "median_error_km" in m:
                rows[0] = ("Kinematic\nbaseline", float(m["median_error_km"]))
                break
    except Exception:
        pass

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#9E9E9E"] + ["#1565C0" if i == 1 else "#42A5F5" for i in range(1, len(rows))]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median FDE @ 12h (km)")
    ax.set_title("Coastal suite — model ranking (median Final Displacement Error)")
    ax.axhline(vals[1], color="#E65100", linestyle="--", linewidth=1, alpha=0.8, label="Best neural (Flat LSTM)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_model_ranking_fde.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ar_context_sweep() -> Path:
    """Clean history-length sweep: all points use λ_land=0.1 (no no-land ablation)."""
    runs = [
        ("9h", "AR_9h/RNN_AR_LSTM/lstm_ar_metrics.json"),
        ("12h", "AR_12h/RNN_AR_LSTM/lstm_ar_metrics.json"),
        ("18h", "AR_18h/RNN_AR_LSTM/lstm_ar_metrics.json"),
        ("24h", "AR_24h/RNN_AR_LSTM/lstm_ar_metrics.json"),
    ]
    xs = [9, 12, 18, 24]
    fdes = [load_fde(h, p) for h, p in runs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, fdes, "o-", color="#1565C0", linewidth=2.2, markersize=9, label="AR LSTM (λ_land = 0.1)")
    for x, y in zip(xs, fdes):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
    best_i = int(np.argmin(fdes))
    ax.scatter(
        [xs[best_i]],
        [fdes[best_i]],
        s=160,
        facecolors="none",
        edgecolors="#E65100",
        linewidths=2.2,
        zorder=5,
    )
    ax.set_xticks(xs)
    ax.set_xlabel("History context (hours)")
    ax.set_ylabel("Median FDE @ 12h (km)")
    ax.set_title("Fixed-context AR LSTM — history length sweep (coastal)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = OUT / "fig_ar_context_sweep.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_adaptive_alphas() -> Path:
    w = json.loads(
        (COASTAL / "adaptive_multiscale/RNN_AR_adaptive/context_alpha_weights.json").read_text()
    )
    alpha = np.asarray(w["per_sample"], dtype=np.float32)
    labels = [f"{h:g}h" for h in w["context_hours"]]
    means = alpha.mean(axis=0)
    argmax = (alpha.argmax(axis=1)[:, None] == np.arange(4)[None, :]).mean(axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, means, color="#1565C0", edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("Mean α")
    axes[0].set_title("Adaptive gate — mean soft weights")
    axes[0].set_ylim(0, 0.45)
    for i, v in enumerate(means):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, argmax, color="#E65100", edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Argmax share (%)")
    axes[1].set_title("Adaptive gate — which context wins most often")
    for i, v in enumerate(argmax):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Adaptive multi-scale gate behavior (coastal test)", fontweight="bold")
    fig.tight_layout()
    path = OUT / "fig_adaptive_alphas.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def load_history(anchor: np.ndarray, history_steps: int = 72) -> np.ndarray:
    import pandas as pd
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(DATA)
    full_history = 144
    h_start = full_history - history_steps
    cols = []
    for t in range(h_start, full_history):
        cols.extend([f"x_t{t:03d}_lat", f"x_t{t:03d}_lon"])
    # sample row groups like plot script
    total = pf.metadata.num_rows
    sample_size = min(400_000, total)
    n_groups = pf.metadata.num_row_groups
    frac = sample_size / total
    n_pick = max(1, min(n_groups, int(np.ceil(n_groups * frac))))
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(n_groups, size=n_pick, replace=False).tolist())
    df = pd.concat([pf.read_row_group(g, columns=cols).to_pandas() for g in chosen], ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
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


def _path_metrics(traj: dict) -> dict[str, np.ndarray]:
    """Net displacement, path length, directness, max cross-track (km)."""
    anchor = traj["anchor"]
    gt = traj["y_true"]
    pred = traj["y_pred"]
    full = np.concatenate([anchor[:, None, :], gt], axis=1)
    dstep = haversine_km(full[:, :-1, 0], full[:, :-1, 1], full[:, 1:, 0], full[:, 1:, 1])
    plen = dstep.sum(axis=1)
    net = haversine_km(anchor[:, 0], anchor[:, 1], gt[:, -1, 0], gt[:, -1, 1])
    err = haversine_km(gt[:, -1, 0], gt[:, -1, 1], pred[:, -1, 0], pred[:, -1, 1])
    direct = net / np.maximum(plen, 1e-6)

    # max distance from GT points to chord NOW → true end (local EN km)
    ct = np.zeros(len(anchor), dtype=np.float64)
    for i in range(len(anchor)):
        ke = 111.32 * np.cos(np.radians(anchor[i, 0]))
        kn = 111.32
        e_e = (gt[i, -1, 1] - anchor[i, 1]) * ke
        e_n = (gt[i, -1, 0] - anchor[i, 0]) * kn
        length = float(np.hypot(e_e, e_n)) + 1e-9
        pts_e = (gt[i, :, 1] - anchor[i, 1]) * ke
        pts_n = (gt[i, :, 0] - anchor[i, 0]) * kn
        ct[i] = float(np.max(np.abs(e_e * pts_n - e_n * pts_e) / length))

    # coarse heading change every 1h (6 steps) — less noisy than per-step
    turn = np.zeros(len(anchor), dtype=np.float64)
    for i in range(len(anchor)):
        pts = full[i, ::6]
        br = []
        for a, b in zip(pts[:-1], pts[1:]):
            phi1, phi2 = np.radians(a[0]), np.radians(b[0])
            dlon = np.radians(b[1] - a[1])
            y = np.sin(dlon) * np.cos(phi2)
            x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
            br.append(np.degrees(np.arctan2(y, x)))
        if len(br) >= 2:
            db = (np.diff(br) + 180.0) % 360.0 - 180.0
            turn[i] = float(np.abs(db).sum())

    return {
        "err": err,
        "net": net,
        "plen": plen,
        "direct": direct,
        "ct": ct,
        "turn": turn,
        "maneuver_score": (1.0 - np.clip(direct, 0, 1)) * np.log1p(ct) * np.log1p(turn),
    }


def pick_clear_examples(traj: dict, n: int = 15) -> list[tuple[int, str]]:
    """Diverse gallery (~15): straight / maneuver × good–bad FDE for report picking."""
    m = _path_metrics(traj)
    err, net, direct, ct, turn, score = (
        m["err"],
        m["net"],
        m["direct"],
        m["ct"],
        m["turn"],
        m["maneuver_score"],
    )
    long = net >= 80.0
    mid_travel = net >= 45.0
    # maneuver: curved / low directness with meaningful travel
    maneuver = mid_travel & ((direct < 0.75) | (ct > 20.0)) & (turn > 80.0)
    straight = long & (direct >= 0.90) & (ct < 15.0)
    used: set[int] = set()
    picks: list[tuple[int, str]] = []

    def _take(mask: np.ndarray, label: str, k: int, order_by: np.ndarray, descending: bool = True) -> None:
        pool = np.where(mask)[0]
        pool = np.array([i for i in pool if i not in used], dtype=int)
        if len(pool) == 0:
            return
        order = pool[np.argsort(-order_by[pool] if descending else order_by[pool])]
        for j, idx in enumerate(order[:k]):
            used.add(int(idx))
            tag = (
                f"{label} #{j+1} | true {net[idx]:.0f} km | "
                f"dir={direct[idx]:.2f} ct={ct[idx]:.0f}km | FDE {err[idx]:.1f} km"
            )
            picks.append((int(idx), tag))

    # Straight / long tracks across FDE bands
    _take(straight & (err < 20), "Straight (good)", 3, net)
    _take(straight & (err >= 20) & (err < 50), "Straight (medium)", 2, net)
    _take(long & (err >= 50), "Straight/long (high FDE)", 2, net)
    # Maneuvers across FDE bands
    _take(maneuver & (err < 25), "Maneuver (good pred)", 2, score)
    _take(maneuver & (err >= 25) & (err < 55), "Maneuver (medium pred)", 3, score)
    _take(maneuver & (err >= 55), "Maneuver (hard)", 3, score)
    # fill leftovers with highest-score unused maneuvers / long tracks
    if len(picks) < n:
        _take(maneuver, "Maneuver (extra)", n - len(picks), score)
    if len(picks) < n:
        _take(mid_travel, "Extra track", n - len(picks), net)
    return picks[:n]


def build_clear_folium(traj: dict, indices: list[tuple[int, str]], out_html: Path, history_steps: int = 144) -> None:
    import folium
    from folium import Element
    from folium.plugins import Fullscreen, MeasureControl

    histories = {i: load_history(traj["anchor"][i], history_steps=history_steps) for i, _ in indices}
    anchors = traj["anchor"][[i for i, _ in indices]]
    # real geographic basemap (OpenStreetMap) + optional satellite
    m = folium.Map(
        location=[float(np.median(anchors[:, 0])), float(np.median(anchors[:, 1]))],
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True,
        overlay=False,
    ).add_to(m)
    folium.TileLayer("CartoDB positron", name="Light map", control=True).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:18px;left:18px;z-index:9999;background:white;
    border:2px solid #222;border-radius:8px;padding:12px 14px;font:13px/1.55 system-ui;
    max-width:360px;box-shadow:0 2px 10px rgba(0,0,0,.18);">
    <b>Model: {MODEL_LABEL}</b><br>
    Coastal test samples · 12h future prediction<br>
    <b>Trajectory legend</b><br>
    <span style="color:#757575;font-weight:700;">——</span> Gray = history (past)<br>
    <span style="color:#1565C0;font-weight:700;">——</span> Blue = true future (12h)<br>
    <span style="color:#2E7D32;font-weight:700;">- - -</span> Green dashed = model prediction<br>
    <span style="color:orange;">★</span> Orange star = NOW (history→future swap)<br>
    <span style="color:#C62828;">●</span> Red = true endpoint @ 12h<br>
    <span style="color:#EF6C00;">◆</span> Orange = predicted endpoint @ 12h<br>
    Thin red line = endpoint error (FDE)<br>
    <small>Toggle tracks / Street·Satellite in layer control.</small>
    </div>
    """
    m.get_root().html.add_child(Element(legend))

    all_lats: list[float] = []
    all_lons: list[float] = []

    for show_i, (row_i, title) in enumerate(indices):
        hist = histories[row_i]
        anchor = traj["anchor"][row_i]
        gt = traj["y_true"][row_i]
        pred = traj["y_pred"][row_i]
        err = float(haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1]))
        net = float(haversine_km(anchor[0], anchor[1], gt[-1, 0], gt[-1, 1]))
        # show maneuvers + one good track by default so the map is not empty/cluttered
        default_on = show_i < 2 or "Maneuver" in title
        layer = folium.FeatureGroup(name=title, show=default_on)
        hist_pts = [[float(a), float(b)] for a, b in hist]
        now = [float(anchor[0]), float(anchor[1])]
        gt_pts = [[float(gt[t, 0]), float(gt[t, 1])] for t in range(len(gt))]
        pred_pts = [[float(pred[t, 0]), float(pred[t, 1])] for t in range(len(pred))]
        true_end = gt_pts[-1]
        pred_end = pred_pts[-1]
        for lat, lon in hist_pts + [now] + gt_pts + pred_pts:
            all_lats.append(lat)
            all_lons.append(lon)

        folium.PolyLine(hist_pts, color="#757575", weight=5, opacity=0.95, tooltip="History (past)").add_to(layer)
        folium.PolyLine([now] + gt_pts, color="#1565C0", weight=5, opacity=0.95, tooltip="True future").add_to(layer)
        folium.PolyLine(
            [now] + pred_pts,
            color="#2E7D32",
            weight=4,
            opacity=0.9,
            dash_array="8 6",
            tooltip="Prediction",
        ).add_to(layer)
        folium.PolyLine(
            [true_end, pred_end],
            color="#C62828",
            weight=3,
            opacity=0.85,
            tooltip=f"FDE = {err:.2f} km",
        ).add_to(layer)

        folium.Marker(
            now,
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
            tooltip="NOW — swap history → future",
            popup=f"<b>NOW (anchor)</b><br>lat={now[0]:.4f}<br>lon={now[1]:.4f}",
        ).add_to(layer)
        folium.CircleMarker(
            true_end,
            radius=9,
            color="#B71C1C",
            fill=True,
            fill_color="#E53935",
            fill_opacity=1,
            weight=2,
            tooltip="True endpoint @ 12h",
            popup=f"<b>True @ 12h</b><br>true travel={net:.1f} km",
        ).add_to(layer)
        folium.RegularPolygonMarker(
            pred_end,
            number_of_sides=4,
            radius=9,
            rotation=45,
            color="#E65100",
            fill=True,
            fill_color="#FF9800",
            fill_opacity=1,
            weight=2,
            tooltip=f"Predicted endpoint | FDE {err:.1f} km",
            popup=f"<b>Predicted @ 12h</b><br>FDE = <b>{err:.2f} km</b>",
        ).add_to(layer)

        folium.Marker(
            location=[now[0] + 0.12, now[1] + 0.12],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font:700 11px system-ui;background:rgba(255,255,255,0.95);'
                    f'padding:4px 7px;border:1px solid #333;border-radius:4px;white-space:nowrap;">'
                    f"{title}<br><span style='color:#C62828'>FDE {err:.1f} km</span></div>"
                )
            ),
        ).add_to(layer)
        layer.add_to(m)

    if all_lats:
        m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]], padding=(30, 30))
    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def _lonlat_to_webmerc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 lon/lat → Web Mercator meters (EPSG:3857)."""
    R = 6378137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0)) * R
    return x, y


def _draw_track_on_basemap(
    ax,
    hist: np.ndarray,
    anchor: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    title: str,
    err_km: float,
    show_legend: bool = False,
) -> None:
    import contextily as cx

    hx, hy = _lonlat_to_webmerc(hist[:, 1], hist[:, 0])
    ax_x, ax_y = _lonlat_to_webmerc(np.array([anchor[1]]), np.array([anchor[0]]))
    gx, gy = _lonlat_to_webmerc(np.r_[anchor[1], gt[:, 1]], np.r_[anchor[0], gt[:, 0]])
    px, py = _lonlat_to_webmerc(np.r_[anchor[1], pred[:, 1]], np.r_[anchor[0], pred[:, 0]])
    tx, ty = _lonlat_to_webmerc(np.array([gt[-1, 1]]), np.array([gt[-1, 0]]))
    qx, qy = _lonlat_to_webmerc(np.array([pred[-1, 1]]), np.array([pred[-1, 0]]))

    ax.plot(hx, hy, color="#757575", lw=3.0, solid_capstyle="round", label="History", zorder=3)
    ax.plot(gx, gy, color="#1565C0", lw=3.2, solid_capstyle="round", label="True future", zorder=4)
    ax.plot(px, py, color="#2E7D32", lw=2.8, ls="--", solid_capstyle="round", label="Prediction", zorder=4)
    ax.plot([tx[0], qx[0]], [ty[0], qy[0]], color="#C62828", lw=2.2, label="FDE", zorder=5)
    ax.scatter(ax_x, ax_y, c="orange", s=160, marker="*", zorder=6, edgecolors="k", linewidths=0.6, label="NOW")
    ax.scatter(tx, ty, c="#E53935", s=70, zorder=6, edgecolors="k", linewidths=0.6, label="True end")
    ax.scatter(qx, qy, c="#FF9800", s=70, marker="D", zorder=6, edgecolors="k", linewidths=0.6, label="Pred end")

    xs = np.r_[hx, gx, px]
    ys = np.r_[hy, gy, py]
    pad_x = max((xs.max() - xs.min()) * 0.18, 2500.0)
    pad_y = max((ys.max() - ys.min()) * 0.18, 2500.0)
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    ax.set_aspect("equal")

    try:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution_size=6, zoom="auto")
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution_size=6, zoom="auto")
        except Exception as e:
            ax.text(0.5, 0.02, f"(basemap unavailable: {e})", transform=ax.transAxes, ha="center", fontsize=8)

    ax.set_axis_off()
    short = title.split("|")[0].strip()
    ax.set_title(f"{MODEL_NAME} — {short}\nFDE = {err_km:.1f} km", fontsize=11, pad=8)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)


def export_basemap_track_pngs(
    traj: dict,
    indices: list[tuple[int, str]],
    out_dir: Path,
    *,
    history_steps: int = 144,
) -> list[Path]:
    """Report-ready PNGs with OSM basemap (same legend as the HTML map)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    singles = out_dir / "map_track_pngs"
    singles.mkdir(parents=True, exist_ok=True)

    # Individual high-res images
    for k, (idx, title) in enumerate(indices, start=1):
        hist = load_history(traj["anchor"][idx], history_steps=history_steps)
        anchor = traj["anchor"][idx]
        gt = traj["y_true"][idx]
        pred = traj["y_pred"][idx]
        err = float(haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1]))
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        _draw_track_on_basemap(ax, hist, anchor, gt, pred, title=title, err_km=err, show_legend=True)
        safe = f"track_{k:02d}_idx{idx}_fde{err:.0f}km.png"
        path = singles / safe
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
        print(f"  PNG {path.name}")

    # Grid collage for the PDF — all gallery tracks (up to 15)
    grid_idx = indices[:15]
    n = len(grid_idx)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14.5, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (idx, title) in zip(axes, grid_idx):
        hist = load_history(traj["anchor"][idx], history_steps=history_steps)
        anchor = traj["anchor"][idx]
        gt = traj["y_true"][idx]
        pred = traj["y_pred"][idx]
        err = float(haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1]))
        _draw_track_on_basemap(ax, hist, anchor, gt, pred, title=title, err_km=err, show_legend=False)
    for ax in axes[n:]:
        ax.axis("off")
    # shared legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#757575", lw=3, label="History"),
        Line2D([0], [0], color="#1565C0", lw=3, label="True future"),
        Line2D([0], [0], color="#2E7D32", lw=2.5, ls="--", label="Prediction"),
        Line2D([0], [0], color="#C62828", lw=2, label="FDE"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="orange", markeredgecolor="k", markersize=12, label="NOW"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935", markeredgecolor="k", markersize=8, label="True end"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#FF9800", markeredgecolor="k", markersize=8, label="Pred end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, framealpha=0.95)
    fig.suptitle(
        f"{MODEL_LABEL} — tracks on map (history / truth / prediction)",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    collage = out_dir / "fig_tracks_on_map.png"
    fig.savefig(collage, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(collage)
    print(f"  collage {collage.name}")
    return paths


def fig_static_track_panels(traj: dict, indices: list[tuple[int, str]], out_png: Path, history_steps: int = 144) -> Path:
    """Static PNG panels for the PDF report (no basemap dependency)."""
    n = len(indices)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (idx, title) in zip(axes, indices):
        hist = load_history(traj["anchor"][idx], history_steps=history_steps)
        anchor = traj["anchor"][idx]
        gt = traj["y_true"][idx]
        pred = traj["y_pred"][idx]
        err = float(haversine_km(gt[-1, 0], gt[-1, 1], pred[-1, 0], pred[-1, 1]))
        ax.plot(hist[:, 1], hist[:, 0], color="#757575", lw=2.5, label="History")
        ax.plot(
            np.r_[anchor[1], gt[:, 1]],
            np.r_[anchor[0], gt[:, 0]],
            color="#1565C0",
            lw=2.5,
            label="True future",
        )
        ax.plot(
            np.r_[anchor[1], pred[:, 1]],
            np.r_[anchor[0], pred[:, 0]],
            color="#2E7D32",
            lw=2.2,
            ls="--",
            label="Prediction",
        )
        ax.plot([gt[-1, 1], pred[-1, 1]], [gt[-1, 0], pred[-1, 0]], color="#C62828", lw=1.8, label="FDE")
        ax.scatter([anchor[1]], [anchor[0]], c="orange", s=90, marker="*", zorder=5, edgecolors="k", label="NOW")
        ax.scatter([gt[-1, 1]], [gt[-1, 0]], c="#E53935", s=50, zorder=5, edgecolors="k", label="True end")
        ax.scatter([pred[-1, 1]], [pred[-1, 0]], c="#FF9800", s=50, marker="D", zorder=5, edgecolors="k", label="Pred end")
        ax.set_title(f"{title}\nFDE = {err:.1f} km", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes[len(indices) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("AR 12h coastal — example tracks (history / truth / prediction)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating report figures...")
    p1 = fig_model_ranking()
    print(" ", p1)
    p2 = fig_ar_context_sweep()
    print(" ", p2)
    p3 = fig_adaptive_alphas()
    print(" ", p3)

    traj_path = COASTAL / "AR_12h_noland/RNN_AR_LSTM/lstm_ar_sample_trajectories.json"
    if not traj_path.exists():
        traj_path = COASTAL / "AR_12h/RNN_AR_LSTM/lstm_ar_sample_trajectories.json"
    print(f"Model: {MODEL_LABEL}")
    print(f"Building maps from {traj_path.name} ...")
    raw = json.loads(traj_path.read_text())
    traj = {k: np.asarray(raw[k], dtype=np.float64) for k in ("y_true", "y_pred", "anchor")}
    indices = pick_clear_examples(traj, n=15)
    for i, t in indices:
        print(f"  [{i}] {t}")

    # Catalog for choosing report figures
    catalog = OUT / "TRACK_CHOICES.txt"
    lines = [
        f"Model tested: {MODEL_LABEL}",
        "Architecture: encoder-decoder AR LSTM (RNN_AR_LSTM)",
        "History context: 12 hours (72 steps @ 10 min)",
        "Prediction horizon: 12 hours (72 steps)",
        "Data: USA Combined coastal-filtered (combined_filtered_smart_coastal)",
        "Land penalty: 0.0 (AR_12h_noland run under exp_coastal)",
        "Source predictions: lstm_ar_sample_trajectories.json",
        "",
        "Pick any of these PNGs for the report:",
        f"  Folder: {OUT / 'map_track_pngs'}",
        f"  Interactive HTML: {OUT / 'map_clear_tracks_ar12h.html'}",
        "",
    ]
    for k, (idx, title) in enumerate(indices, start=1):
        lines.append(f"  {k:02d}. idx={idx:3d}  {title}")
    catalog.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(" ", catalog)

    # Full 24h gray history on a real OSM basemap
    html = OUT / "map_clear_tracks_ar12h.html"
    build_clear_folium(traj, indices, html, history_steps=144)
    print(" ", html)

    # Report PNGs with real map tiles (same tracks / legend as HTML)
    print("Exporting basemap PNGs for the report...")
    # clear previous track pngs so numbering matches this gallery
    singles = OUT / "map_track_pngs"
    if singles.exists():
        for old in singles.glob("track_*.png"):
            old.unlink()
    export_basemap_track_pngs(traj, indices, OUT, history_steps=144)

    # Static lat/lon panels (no tiles) as backup
    man_idx = [(i, t) for i, t in indices if "Maneuver" in t]
    other = [(i, t) for i, t in indices if "Maneuver" not in t][:2]
    panel_idx = (man_idx + other)[:6] if man_idx else indices[:6]
    png = OUT / "fig_example_tracks_panels.png"
    fig_static_track_panels(traj, panel_idx, png, history_steps=144)
    print(" ", png)

    # Copy train/val loss curve into report_figures
    import shutil

    src_hist = traj_path.parent / "lstm_ar_training_history.png"
    if src_hist.exists():
        dst_hist = OUT / "fig_train_val_loss_ar12h.png"
        shutil.copy2(src_hist, dst_hist)
        print(" ", dst_hist)

    guide = OUT / "WHERE_TO_PUT_FIGURES.txt"
    guide.write_text(
        f"""Where to put figures in final_report
=====================================

TRACK MAPS MODEL: {MODEL_LABEL}
  (= coastal AR LSTM with 12h history, no land penalty)

1) fig_model_ranking_fde.png          → §6.1
2) fig_ar_context_sweep.png           → §6.2
3) fig_adaptive_alphas.png            → §6.4
4) map_track_pngs/*.png (15 options)  → pick 2–4 for §6 / Discussion
   See TRACK_CHOICES.txt for the catalog.
   fig_tracks_on_map.png = collage of 6
5) map_clear_tracks_ar12h.html        → interactive appendix
6) fig_train_val_loss_ar12h.png       → §4.4 training (top panel)
""",
        encoding="utf-8",
    )
    print(" ", guide)
    print("Done.")


if __name__ == "__main__":
    main()
