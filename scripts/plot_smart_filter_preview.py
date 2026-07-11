#!/usr/bin/env python3
"""Preview smart-motion filter: random KEPT vs REJECTED tracks before applying."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import (
    SmartMotionFilterConfig,
    compute_smart_motion_metrics,
    smart_motion_keep_mask,
)

HIST = "#1565C0"
FUT = "#7B1FA2"
START = "#1B5E20"
REJECT_BG = "#FFEBEE"
KEEP_BG = "#E8F5E9"


def load_pool(path: Path, n: int, seed: int) -> tuple[pd.DataFrame, int, int]:
    names = pq.ParquetFile(path).schema.names
    hs = sum(1 for c in names if c.startswith("x_t") and c.endswith("_lat"))
    fs = sum(1 for c in names if c.startswith("y_t") and c.endswith("_lat"))
    cols = [f"x_t{t:03d}_{ax}" for t in range(hs) for ax in ("lat", "lon")]
    cols += [f"y_t{t:03d}_{ax}" for t in range(fs) for ax in ("lat", "lon")]
    df = pd.read_parquet(path, columns=cols)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True), hs, fs


def reject_reason(m: pd.Series, cfg: SmartMotionFilterConfig) -> str:
    reasons = []
    if m["history_16h_net_km"] < cfg.min_history_16h_net_km:
        reasons.append(f"16h<{cfg.min_history_16h_net_km:g}km")
    if m["history_last_8h_net_km"] < cfg.min_history_last_8h_net_km:
        reasons.append(f"last8h<{cfg.min_history_last_8h_net_km:g}km")
    if (
        m["history_path_km"] > cfg.min_path_for_loop_km
        and m["history_loop_ratio"] < cfg.max_history_loop_ratio
        and m["history_path_km"] < cfg.min_big_loop_path_km
        and m["history_max_radius_km"] < cfg.max_local_loop_radius_km
    ):
        reasons.append("local_loop")
    return "+".join(reasons) if reasons else "?"


def plot_panel(ax: plt.Axes, row: pd.Series, hs: int, fs: int, title: str, *, kept: bool) -> None:
    hist = np.column_stack([
        [row[f"x_t{t:03d}_lat"] for t in range(hs)],
        [row[f"x_t{t:03d}_lon"] for t in range(hs)],
    ])
    fut = np.column_stack([
        [row[f"y_t{t:03d}_lat"] for t in range(fs)],
        [row[f"y_t{t:03d}_lon"] for t in range(fs)],
    ])
    anchor = hist[-1]
    full = np.vstack([hist, fut])
    mid_lat = float(full[:, 0].mean())
    span = max(float(np.ptp(full[:, 0])), float(np.ptp(full[:, 1])), 0.05)
    dlat = max(span * 0.6, 0.08)
    dlon = dlat / max(math.cos(math.radians(mid_lat)), 0.2)
    cx, cy = float(full[:, 1].mean()), float(full[:, 0].mean())
    ax.set_xlim(cx - dlon, cx + dlon)
    ax.set_ylim(cy - dlat, cy + dlat)
    ax.set_facecolor(KEEP_BG if kept else REJECT_BG)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle=":")

    ax.plot(hist[:, 1], hist[:, 0], color=HIST, lw=2.2, zorder=3)
    ax.scatter(anchor[1], anchor[0], s=90, marker="s", c=START, edgecolors="white", zorder=5)
    ax.plot(fut[:, 1], fut[:, 0], color=FUT, lw=2, ls="--", zorder=2)
    ax.scatter(fut[-1, 1], fut[-1, 0], s=70, marker="o", c=FUT, zorder=5)
    ax.set_title(title, fontsize=7, fontweight="bold")
    ax.set_xlabel("lon", fontsize=6)
    ax.set_ylabel("lat", fontsize=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "data/processed/combined_filtered/train.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "data/results/USA Combined/unknown/smart_motion_audit/preview_kept_vs_rejected.png",
    )
    parser.add_argument("--pool", type=int, default=12000)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smart-min-16h-net-km", type=float, default=8.0)
    parser.add_argument("--smart-min-last-8h-net-km", type=float, default=2.0)
    parser.add_argument("--smart-max-loop-ratio", type=float, default=0.35)
    args = parser.parse_args()

    cfg = SmartMotionFilterConfig(
        min_history_16h_net_km=args.smart_min_16h_net_km,
        min_history_last_8h_net_km=args.smart_min_last_8h_net_km,
        max_history_loop_ratio=args.smart_max_loop_ratio,
    )
    pool, hs, fs = load_pool(args.input, args.pool, args.seed)
    metrics = compute_smart_motion_metrics(pool, hs, fs)
    keep = smart_motion_keep_mask(metrics, cfg)

    rng = np.random.default_rng(args.seed)
    kept_idx = np.flatnonzero(keep)
    rej_idx = np.flatnonzero(~keep)
    rng.shuffle(kept_idx)
    rng.shuffle(rej_idx)
    show_k = kept_idx[: args.per_group]
    show_r = rej_idx[: args.per_group]

    ncols = args.per_group
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.5), squeeze=False)

    for col, i in enumerate(show_r):
        m = metrics.iloc[i]
        title = (
            f"REJECT #{i}\n"
            f"why: {reject_reason(m, cfg)}\n"
            f"16h={m.history_16h_net_km:.1f} last8h={m.history_last_8h_net_km:.1f}km\n"
            f"fut12h={m.future_12h_net_km:.1f}km (true)"
        )
        plot_panel(axes[0, col], pool.iloc[i], hs, fs, title, kept=False)

    for col, i in enumerate(show_k):
        m = metrics.iloc[i]
        title = (
            f"KEEP #{i}\n"
            f"16h={m.history_16h_net_km:.1f} last8h={m.history_last_8h_net_km:.1f}km\n"
            f"fut12h={m.future_12h_net_km:.1f}km (true)"
        )
        plot_panel(axes[1, col], pool.iloc[i], hs, fs, title, kept=True)

    keep_pct = keep.mean() * 100
    fig.suptitle(
        "Smart filter PREVIEW (before applying to full dataset)\n"
        f"Green START square → blue=24h history → purple dashed=12h TRUE future\n"
        f"Top row: random REJECTED examples | Bottom row: random KEPT examples\n"
        f"Overall: keeps {keep_pct:.0f}% of combined_filtered "
        f"(removes {100-keep_pct:.0f}%)\n"
        f"Filter: last-16h net≥{cfg.min_history_16h_net_km}km, "
        f"last-8h net≥{cfg.min_history_last_8h_net_km}km, "
        f"drop small local loops only (keep path≥{cfg.min_big_loop_path_km}km or radius≥{cfg.max_local_loop_radius_km}km)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"pool keep rate: {keep.mean()*100:.1f}%")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
