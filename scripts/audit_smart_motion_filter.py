#!/usr/bin/env python3
"""
Audit smart-motion filter on a random sample of windows.

Samples N vessels from input parquet, applies history-only smart filter rules,
and checks (offline) whether they actually move in the next 8h / 12h future.

Outputs:
  - audit_smart_motion_<N>.json
  - audit_smart_motion_<N>_grid.png
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
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import (
    SmartMotionFilterConfig,
    compute_smart_motion_metrics,
    smart_motion_keep_mask,
)

START = "#1B5E20"
HIST = "#1565C0"
FUT8 = "#00897B"
FUT12 = "#6A1B9A"


def load_geo_sample(path: Path, n: int, seed: int) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema.names
    hs = sum(1 for c in names if c.startswith("x_t") and c.endswith("_lat"))
    fs = sum(1 for c in names if c.startswith("y_t") and c.endswith("_lat"))
    cols = [f"x_t{t:03d}_{ax}" for t in range(hs) for ax in ("lat", "lon")]
    cols += [f"y_t{t:03d}_{ax}" for t in range(fs) for ax in ("lat", "lon")]
    df = pd.read_parquet(path, columns=cols)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def track_latlon(df_row: pd.Series, prefix: str, steps: int) -> np.ndarray:
    lat = [df_row[f"{prefix}{t:03d}_lat"] for t in range(steps)]
    lon = [df_row[f"{prefix}{t:03d}_lon"] for t in range(steps)]
    return np.column_stack([lat, lon])


def plot_vessel(ax: plt.Axes, row: pd.Series, hs: int, fs: int, meta: dict) -> None:
    hist = track_latlon(row, "x_t", hs)
    fut = track_latlon(row, "y_t", fs)
    anchor = hist[-1]
    full = np.vstack([hist, fut])
    mid_lat = float(full[:, 0].mean())
    span_km = max(meta["future_12h_net_km"], meta["history_16h_net_km"], 12.0)
    dlat = span_km / 111.0 * 0.75
    dlon = span_km / (111.0 * max(math.cos(math.radians(mid_lat)), 0.2)) * 0.75
    ax.set_xlim(full[:, 1].min() - dlon, full[:, 1].max() + dlon)
    ax.set_ylim(full[:, 0].min() - dlat, full[:, 0].max() + dlat)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linestyle=":")

    ax.plot(hist[:, 1], hist[:, 0], color=HIST, lw=2, label="24h history")
    ax.scatter(anchor[1], anchor[0], s=80, c=START, marker="s", zorder=5)
    ax.plot(fut[:, 1], fut[:, 0], color=FUT12, lw=2, ls="--", label="12h future (true)")
    ax.scatter(fut[-1, 1], fut[-1, 0], s=70, c=FUT12, marker="o", zorder=5)

    step_8 = min(int(round(8 * 60 / 10)) - 1, fs - 1)
    ax.scatter(fut[step_8, 1], fut[step_8, 0], s=60, c=FUT8, marker="^", zorder=5)

    status = "PASS" if meta["smart_keep"] else "REJECT"
    ax.set_title(
        f"#{meta['index']} {status}\n"
        f"16h hist {meta['history_16h_net_km']:.1f}km | last8h {meta['history_last_8h_net_km']:.1f}km\n"
        f"fut8h {meta['future_8h_net_km']:.1f}km | fut12h {meta['future_12h_net_km']:.1f}km",
        fontsize=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "data/processed/combined_filtered/train.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "data/results/USA Combined/unknown/smart_motion_audit",
    )
    parser.add_argument("--sample-pool", type=int, default=5000,
                        help="Random pool drawn from parquet before picking audit set.")
    parser.add_argument("--n-audit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smart-min-16h-net-km", type=float, default=8.0)
    parser.add_argument("--smart-min-last-8h-net-km", type=float, default=2.0)
    parser.add_argument("--smart-max-loop-ratio", type=float, default=0.35)
    parser.add_argument("--audit-min-future-km", type=float, default=5.0)
    args = parser.parse_args()

    config = SmartMotionFilterConfig(
        min_history_16h_net_km=args.smart_min_16h_net_km,
        min_history_last_8h_net_km=args.smart_min_last_8h_net_km,
        max_history_loop_ratio=args.smart_max_loop_ratio,
        audit_min_future_8h_net_km=args.audit_min_future_km,
        audit_min_future_12h_net_km=args.audit_min_future_km,
    )

    pool = load_geo_sample(args.input, args.sample_pool, args.seed)
    hs = sum(1 for c in pool.columns if c.startswith("x_t") and c.endswith("_lat"))
    fs = sum(1 for c in pool.columns if c.startswith("y_t") and c.endswith("_lat"))
    metrics = compute_smart_motion_metrics(pool, hs, fs)
    keep = smart_motion_keep_mask(metrics, config)

    # Prefer kept windows for audit, fill with rejected if needed.
    rng = np.random.default_rng(args.seed)
    kept_idx = np.flatnonzero(keep)
    rejected_idx = np.flatnonzero(~keep)
    rng.shuffle(kept_idx)
    rng.shuffle(rejected_idx)
    n_keep_show = min(args.n_audit, len(kept_idx))
    n_reject_show = min(max(0, args.n_audit - n_keep_show), len(rejected_idx))
    audit_idx = np.concatenate([kept_idx[:n_keep_show], rejected_idx[:n_reject_show]])
    rng.shuffle(audit_idx)
    audit_idx = audit_idx[: args.n_audit]

    rows = []
    for j, i in enumerate(audit_idx):
        m = metrics.iloc[i]
        rows.append({
            "audit_rank": j,
            "pool_index": int(i),
            "smart_keep": bool(keep[i]),
            "history_16h_net_km": float(m["history_16h_net_km"]),
            "history_last_8h_net_km": float(m["history_last_8h_net_km"]),
            "history_full_net_km": float(m["history_full_net_km"]),
            "history_path_km": float(m["history_path_km"]),
            "history_loop_ratio": float(m["history_loop_ratio"]),
            "arrived_then_stopped": bool(m["arrived_then_stopped"]),
            "future_8h_net_km": float(m["future_8h_net_km"]),
            "future_12h_net_km": float(m["future_12h_net_km"]),
            "future_8h_moves": bool(m["future_8h_net_km"] >= config.audit_min_future_8h_net_km),
            "future_12h_moves": bool(m["future_12h_net_km"] >= config.audit_min_future_12h_net_km),
        })

    kept_rows = [r for r in rows if r["smart_keep"]]
    summary = {
        "input": str(args.input),
        "sample_pool": len(pool),
        "n_audit": len(rows),
        "smart_config": {
            "min_history_16h_net_km": config.min_history_16h_net_km,
            "min_history_last_8h_net_km": config.min_history_last_8h_net_km,
            "max_history_loop_ratio": config.max_history_loop_ratio,
        },
        "horizon_recommendation": {
            "keep_parquet_windows": "24h history / 12h future (no rebuild required)",
            "rationale": (
                "Smart filter uses last-16h and last-8h history motion to drop vessels "
                "that already arrived and stopped. Shortening to 16h/8h is optional at train time."
            ),
        },
        "pool_stats": {
            "smart_keep_fraction": float(keep.mean()),
            "future_8h_median_km_kept": float(metrics.loc[keep, "future_8h_net_km"].median()) if keep.any() else 0,
            "future_12h_median_km_kept": float(metrics.loc[keep, "future_12h_net_km"].median()) if keep.any() else 0,
            "pct_future_8h_ge_audit": float((metrics.loc[keep, "future_8h_net_km"] >= config.audit_min_future_8h_net_km).mean()) if keep.any() else 0,
            "pct_future_12h_ge_audit": float((metrics.loc[keep, "future_12h_net_km"] >= config.audit_min_future_12h_net_km).mean()) if keep.any() else 0,
            "pct_arrived_then_stopped_kept": float(metrics.loc[keep, "arrived_then_stopped"].mean()) if keep.any() else 0,
        },
        "audit_kept_only": {
            "n": len(kept_rows),
            "pct_future_8h_ge_audit": float(np.mean([r["future_8h_moves"] for r in kept_rows])) if kept_rows else 0,
            "pct_future_12h_ge_audit": float(np.mean([r["future_12h_moves"] for r in kept_rows])) if kept_rows else 0,
            "median_future_8h_km": float(np.median([r["future_8h_net_km"] for r in kept_rows])) if kept_rows else 0,
            "median_future_12h_km": float(np.median([r["future_12h_net_km"] for r in kept_rows])) if kept_rows else 0,
        },
        "vessels": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"audit_smart_motion_{len(rows)}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pool_stats"], indent=2))
    print(json.dumps(summary["audit_kept_only"], indent=2))
    print(f"wrote {json_path}")

    # Plot grid (max 20 panels for readability)
    show_n = min(20, len(rows))
    ncols = 5
    nrows = int(math.ceil(show_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.8 * nrows), squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= show_n:
            ax.axis("off")
            continue
        i = audit_idx[k]
        meta = rows[k] | {"index": int(i)}
        plot_vessel(ax, pool.iloc[i], hs, fs, meta)
    fig.suptitle(
        "Smart-motion audit (green square=START, blue=history, dashed=future)\n"
        f"Filter: 16h net≥{config.min_history_16h_net_km}km, last8h≥{config.min_history_last_8h_net_km}km",
        fontsize=10,
    )
    fig.tight_layout()
    png_path = args.output_dir / f"audit_smart_motion_{len(rows)}_grid.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
