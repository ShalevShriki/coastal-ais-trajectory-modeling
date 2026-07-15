#!/usr/bin/env python3
"""Plot test vessels with explicit start / true end / predicted end / error.

By default selects vessels that actually moved in the 12h future window
(not near-stationary dots), so the prediction task is visible.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import haversine_km

MODELS = [
    ("Transformer", "Transformer", "transformer_sample_trajectories.json", "#7B1FA2"),
    ("RNN", "RNN", "lstm_sample_trajectories.json", "#FF9800"),
    ("RNN_AR", "RNN_AR_LSTM", "lstm_ar_sample_trajectories.json", "#2E7D32"),
]

START_COLOR = "#1B5E20"
TRUE_COLOR = "#1565C0"
PRED_COLOR = "#C62828"
ERROR_COLOR = "#D84315"
NET_ARROW_COLOR = "#00897B"


def load_traj(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "y_true": np.asarray(data["y_true"], dtype=np.float64),
        "y_pred": np.asarray(data["y_pred"], dtype=np.float64),
        "anchor": np.asarray(data["anchor"], dtype=np.float64),
    }


def path_length_km(track: np.ndarray) -> float:
    return float(
        sum(
            haversine_km(track[t, 0], track[t, 1], track[t + 1, 0], track[t + 1, 1])
            for t in range(len(track) - 1)
        )
    )


def motion_stats(anchor: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    true_track = np.vstack([anchor[None, :], gt])
    net = float(haversine_km(anchor[0], anchor[1], gt[-1, 0], gt[-1, 1]))
    path = path_length_km(true_track)
    return net, path


def km_to_deg_span(km: float, mid_lat: float) -> tuple[float, float]:
    km_per_deg_lat = 111.0
    km_per_deg_lon = max(111.0 * math.cos(math.radians(mid_lat)), 1e-6)
    return km / km_per_deg_lat, km / km_per_deg_lon


def select_vessel_indices(
    traj: dict,
    *,
    n_vessels: int,
    seed: int,
    min_net_km: float,
    min_path_km: float,
    vessel_ids: list[int] | None,
) -> tuple[np.ndarray, list[dict]]:
    n_total = len(traj["y_true"])
    meta: list[dict] = []

    if vessel_ids is not None:
        indices = np.array(vessel_ids, dtype=int)
    else:
        candidates: list[tuple[int, float, float]] = []
        for i in range(n_total):
            net, path = motion_stats(traj["anchor"][i], traj["y_true"][i])
            if net >= min_net_km and path >= min_path_km:
                candidates.append((i, net, path))

        if len(candidates) < n_vessels:
            # Relax thresholds gradually.
            for net_thr, path_thr in [(5, 5), (2, 5), (1, 1)]:
                candidates = []
                for i in range(n_total):
                    net, path = motion_stats(traj["anchor"][i], traj["y_true"][i])
                    if net >= net_thr and path >= path_thr:
                        candidates.append((i, net, path))
                if len(candidates) >= n_vessels:
                    break

        if not candidates:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(n_total, size=min(n_vessels, n_total), replace=False))
        else:
            # Pick spread across net displacement (short / medium / long transit).
            candidates.sort(key=lambda x: x[1])
            rng = np.random.default_rng(seed)
            if len(candidates) <= n_vessels:
                chosen = candidates
            else:
                quantiles = np.linspace(0, len(candidates) - 1, n_vessels)
                chosen = []
                used: set[int] = set()
                for q in quantiles:
                    i = int(round(q))
                    idx, net, path = candidates[i]
                    if idx in used:
                        continue
                    used.add(idx)
                    chosen.append((idx, net, path))
                # Fill if quantile picks collided.
                if len(chosen) < n_vessels:
                    remaining = [c for c in candidates if c[0] not in used]
                    rng.shuffle(remaining)
                    chosen.extend(remaining[: n_vessels - len(chosen)])
            indices = np.array([c[0] for c in chosen], dtype=int)

    for i in indices:
        net, path = motion_stats(traj["anchor"][i], traj["y_true"][i])
        meta.append({"id": int(i), "net_km": net, "path_km": path})
    return indices, meta


def set_axis_for_motion(
    ax: plt.Axes,
    true_track: np.ndarray,
    pred_track: np.ndarray,
    *,
    net_km: float,
    path_km: float,
    min_view_km: float,
) -> None:
    """Zoom so movement is visible; avoid tiny dot panels for micro-moves."""
    view_km = max(min_view_km, net_km * 1.35, path_km * 0.45, 8.0)
    center_lat = float(true_track[:, 0].mean())
    center_lon = float(true_track[:, 1].mean())
    dlat, dlon = km_to_deg_span(view_km, center_lat)
    ax.set_xlim(center_lon - dlon, center_lon + dlon)
    ax.set_ylim(center_lat - dlat, center_lat + dlat)
    ax.set_aspect("equal", adjustable="box")


def plot_one_vessel(
    ax: plt.Axes,
    *,
    vessel_id: int,
    anchor: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_color: str,
    model_name: str,
    min_view_km: float,
) -> None:
    true_track = np.vstack([anchor[None, :], gt])
    pred_track = np.vstack([anchor[None, :], pred])
    true_end = gt[-1]
    pred_end = pred[-1]

    fde = float(haversine_km(true_end[0], true_end[1], pred_end[0], pred_end[1]))
    true_len = path_length_km(true_track)
    pred_len = path_length_km(pred_track)
    net_disp = float(haversine_km(anchor[0], anchor[1], true_end[0], true_end[1]))

    set_axis_for_motion(
        ax, true_track, pred_track,
        net_km=net_disp, path_km=true_len, min_view_km=min_view_km,
    )
    ax.grid(True, alpha=0.3, linestyle=":")

    ax.plot(true_track[:, 1], true_track[:, 0], color=TRUE_COLOR, lw=2.8, zorder=3)
    ax.plot(pred_track[:, 1], pred_track[:, 0], color=pred_color, lw=2.4, ls=(0, (6, 4)), zorder=2)

    ax.scatter(anchor[1], anchor[0], s=150, marker="s", c=START_COLOR, edgecolors="white", linewidths=1.5, zorder=6)
    ax.scatter(true_end[1], true_end[0], s=170, marker="o", c=TRUE_COLOR, edgecolors="white", linewidths=1.5, zorder=6)
    ax.scatter(pred_end[1], pred_end[0], s=190, marker="X", c=pred_color, edgecolors="white", linewidths=1.2, zorder=6)

    ax.plot(
        [true_end[1], pred_end[1]], [true_end[0], pred_end[0]],
        color=ERROR_COLOR, lw=2.0, ls=":", zorder=4,
    )

    # Net displacement arrow START -> TRUE END
    if net_disp >= 0.3:
        ax.add_patch(
            FancyArrowPatch(
                (anchor[1], anchor[0]), (true_end[1], true_end[0]),
                arrowstyle="-|>", mutation_scale=14, lw=2.2,
                color=NET_ARROW_COLOR, alpha=0.85, zorder=5,
            )
        )

    motion_tag = "MOVING" if net_disp >= 5 else ("SHORT MOVE" if net_disp >= 1 else "NEAR-STATIONARY")
    ax.text(
        0.98, 0.98,
        f"{motion_tag}\nnet {net_disp:.1f} km\npath {true_len:.1f} km",
        transform=ax.transAxes, ha="right", va="top", fontsize=7, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#FFFDE7" if net_disp < 5 else "#E8F5E9", edgecolor="#888"),
    )

    ax.annotate(
        f"ERROR {fde:.1f} km",
        xy=((true_end[1] + pred_end[1]) / 2, (true_end[0] + pred_end[0]) / 2),
        fontsize=8, color=ERROR_COLOR, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=ERROR_COLOR, alpha=0.9),
    )

    summary = (
        f"vessel {vessel_id} | {model_name}\n"
        f"Net displacement {net_disp:.1f} km | Path length {true_len:.1f} km\n"
        f"Pred path {pred_len:.1f} km | 12h FDE {fde:.1f} km"
    )
    ax.text(
        0.02, 0.02, summary,
        transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#999"),
        family="monospace",
    )
    ax.set_title(f"12h forecast", fontsize=9, fontweight="bold")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def plot_model(
    name: str,
    traj_path: Path,
    out: Path,
    *,
    indices: np.ndarray,
    pred_color: str,
    min_view_km: float,
    selection_note: str,
) -> None:
    traj = load_traj(traj_path)
    ncols = len(indices)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.4), squeeze=False)

    for col, i in enumerate(indices):
        plot_one_vessel(
            axes[0, col],
            vessel_id=int(i),
            anchor=traj["anchor"][i],
            gt=traj["y_true"][i],
            pred=traj["y_pred"][i],
            pred_color=pred_color,
            model_name=name,
            min_view_km=min_view_km,
        )

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=START_COLOR, markersize=10, label="START (last known pos)"),
        Line2D([0], [0], color=TRUE_COLOR, lw=2.5, label="TRUE 12h path"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRUE_COLOR, markersize=10, label="TRUE END"),
        Line2D([0], [0], color=pred_color, lw=2, ls="--", label="PREDICTED 12h path"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=pred_color, markersize=10, label="PRED END"),
        Line2D([0], [0], color=NET_ARROW_COLOR, lw=2, label="Green arrow = net displacement START→TRUE END"),
        Line2D([0], [0], color=ERROR_COLOR, lw=2, ls=":", label="Orange dotted = FDE error"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"v1/experiment1 — {name}\n{selection_note}",
        y=1.08, fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT / "data/results/USA Combined/unknown/v1/experiment1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "data/results/USA Combined/unknown/v1/experiment1/visualizations",
    )
    parser.add_argument("--n-vessels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-net-km", type=float, default=15.0, help="Min net START→TRUE END displacement")
    parser.add_argument("--min-path-km", type=float, default=15.0, help="Min true path length in 12h")
    parser.add_argument("--min-view-km", type=float, default=12.0, help="Minimum map span so motion is visible")
    parser.add_argument("--vessel-ids", type=str, default=None, help="Comma-separated indices (overrides selection)")
    parser.add_argument("--suffix", type=str, default="", help="Output filename suffix, e.g. _moving")
    args = parser.parse_args()

    ref_path = args.base / "Transformer" / "transformer_sample_trajectories.json"
    ref = load_traj(ref_path)
    vessel_ids = None
    if args.vessel_ids:
        vessel_ids = [int(x.strip()) for x in args.vessel_ids.split(",")]

    indices, meta = select_vessel_indices(
        ref,
        n_vessels=args.n_vessels,
        seed=args.seed,
        min_net_km=args.min_net_km,
        min_path_km=args.min_path_km,
        vessel_ids=vessel_ids,
    )

    ids_str = ", ".join(str(m["id"]) for m in meta)
    nets = ", ".join(f'{m["net_km"]:.0f}km' for m in meta)
    selection_note = (
        f"Selected vessels with real 12h motion (min net≥{args.min_net_km:g} km, path≥{args.min_path_km:g} km) | "
        f"ids [{ids_str}] | net displacements [{nets}]"
    )
    print(selection_note)

    suffix = args.suffix or "_moving"
    for model_name, subdir, traj_file, color in MODELS:
        traj_path = args.base / subdir / traj_file
        if not traj_path.exists():
            print(f"skip {model_name}: missing {traj_path}")
            continue
        plot_model(
            model_name,
            traj_path,
            args.output_dir / f"random_vessels_{model_name.lower()}{suffix}.png",
            indices=indices,
            pred_color=color,
            min_view_km=args.min_view_km,
            selection_note=selection_note,
        )

    # Also overwrite the default filenames when using moving selection.
    if suffix == "_moving":
        for model_name, subdir, traj_file, color in MODELS:
            traj_path = args.base / subdir / traj_file
            src = args.output_dir / f"random_vessels_{model_name.lower()}_moving.png"
            dst = args.output_dir / f"random_vessels_{model_name.lower()}.png"
            if src.exists():
                src.replace(dst)
                print(f"updated {dst}")


if __name__ == "__main__":
    main()
