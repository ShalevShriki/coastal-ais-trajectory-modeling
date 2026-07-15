#!/usr/bin/env python3
"""Correlate adaptive context alphas with motion features on the test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1]))

from proj.project.coast_paths import resolve_windows_path
from proj.project.window_data import load_windows_filtered, make_train_val_test_frames


def motion_stats_from_frame(df) -> dict[str, np.ndarray]:
    dcog_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_dcog")]
    dsog_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_dsog")]
    sog_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_sog")]
    return {
        "mean_sog": df[sog_cols].mean(axis=1).to_numpy(),
        "maneuver": df[dcog_cols].abs().mean(axis=1).to_numpy(),
        "accel": df[dsog_cols].abs().mean(axis=1).to_numpy(),
        "max_turn": df[dcog_cols].abs().max(axis=1).to_numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-json",
        type=Path,
        default=PROJECT
        / "data/results/USA Combined/unknown/exp_coastal/adaptive_multiscale/RNN_AR_adaptive/context_alpha_weights.json",
    )
    parser.add_argument("--sample", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=20_000, help="Subsample test rows for analysis")
    args = parser.parse_args()

    alpha_all = np.array(json.loads(args.alpha_json.read_text())["per_sample"], dtype=np.float32)
    labels = ["9h", "12h", "18h", "24h"]

    input_path, _, _ = resolve_windows_path(
        "USA Combined",
        "usa_combined",
        PROJECT / "data/processed/combined_filtered_smart_coastal/train.parquet",
    )
    prefetch = int(args.sample / 0.7) + 1000
    df = load_windows_filtered(input_path, sample_size=prefetch, seed=args.seed)
    _, _, test_df, _ = make_train_val_test_frames(
        df,
        test_fraction=0.2,
        val_fraction=0.1,
        seed=args.seed,
        split_by="trajectory",
        train_sample_size=args.sample,
    )

    n = min(len(test_df), len(alpha_all))
    if n < len(alpha_all):
        print(f"Note: using first {n} test rows (alpha file has {len(alpha_all)})")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(n, size=min(args.max_rows, n), replace=False)
    sub = test_df.iloc[pick]
    alpha = alpha_all[pick]
    feats = motion_stats_from_frame(sub)

    print(f"=== Adaptive alpha vs motion (n={len(alpha)}) ===\n")
    print("Spearman r:")
    for fname, fvals in feats.items():
        rs = [stats.spearmanr(fvals, alpha[:, j]).statistic for j in range(4)]
        print(f"  {fname:<12} " + " ".join(f"{labels[j]}={rs[j]:+.3f}" for j in range(4)))

    for j, lbl in enumerate(labels):
        hi = alpha[:, j] >= np.percentile(alpha[:, j], 75)
        lo = alpha[:, j] <= np.percentile(alpha[:, j], 25)
        print(f"\nTop/bottom 25% alpha_{lbl}:")
        for fname, fvals in feats.items():
            print(f"  {fname}: low={fvals[lo].mean():.3f} high={fvals[hi].mean():.3f}")

    man = feats["maneuver"]
    for name, mask in [
        ("low motion", man <= np.percentile(man, 33)),
        ("mid motion", (man > np.percentile(man, 33)) & (man <= np.percentile(man, 66))),
        ("high motion", man > np.percentile(man, 66)),
    ]:
        m = alpha[mask].mean(axis=0)
        print(
            f"mean alpha [{name}]: "
            f"9h={m[0]:.3f} 12h={m[1]:.3f} 18h={m[2]:.3f} 24h={m[3]:.3f}"
        )

    argmax = alpha.argmax(axis=1)
    print("\nArgmax context %:", {labels[j]: f"{(argmax == j).mean() * 100:.1f}%" for j in range(4)})


if __name__ == "__main__":
    main()
