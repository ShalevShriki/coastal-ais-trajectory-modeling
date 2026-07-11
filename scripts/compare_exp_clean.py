#!/usr/bin/env python3
"""Compare exp_clean experiment metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "data/results/USA Combined/unknown/exp_clean"

EXPERIMENTS = [
    ("B1", "B1_flat", "RNN", "lstm_metrics.json", "Flat LSTM"),
    ("B2", "B2_transformer", "Transformer", "transformer_metrics.json", "Transformer"),
    ("A0", "A0_ar_anchor_no_tc", "RNN_AR_LSTM", "lstm_ar_metrics.json", "AR anchor, no TC"),
    ("A1", "A1_ar_anchor_tc", "RNN_AR_LSTM", "lstm_ar_metrics.json", "AR anchor + TC"),
    ("A2", "A2_ar_anchor_residual", "RNN_AR_LSTM", "lstm_ar_metrics.json", "AR anchor + residual"),
    ("M1", "M1_ar_anchor_res_tc", "RNN_AR_LSTM", "lstm_ar_metrics.json", "AR anchor + res + TC"),
    ("M2", "M2_ar_step_delta_res_tc", "RNN_AR_LSTM", "lstm_ar_metrics.json", "AR step-delta + res + TC"),
    ("M3", "M3_ar_sliding_3h_res", "RNN_recursive_sliding", "recursive_sliding_metrics.json", "Sliding 3h + res"),
]


def load_row(path: Path, label: str, exp_id: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    main = straight = maneuver = None
    for m in data.get("metrics", []):
        name = m.get("model", "")
        if "Naive" in name or "Kinematic" in name:
            continue
        if "[straight" in name:
            straight = m
        elif "[maneuver" in name:
            maneuver = m
        elif "full predicted" in name or "full recursive" in name:
            continue
        elif main is None:
            main = m
    if not main:
        return None
    return {
        "id": exp_id,
        "label": label,
        "target_mode": data.get("target_mode", data.get("model_family", "—")),
        "fde_med": main.get("median_error_km"),
        "fde_mean": main.get("mean_error_km"),
        "nfde_med": main.get("median_nfde"),
        "straight_fde": straight.get("median_error_km") if straight else None,
        "maneuver_fde": maneuver.get("median_error_km") if maneuver else None,
        "epochs": data.get("training", {}).get("epochs_ran"),
    }


def main() -> None:
    rows = []
    missing = []
    for exp_id, tag, subdir, fname, label in EXPERIMENTS:
        path = RESULTS / tag / subdir / fname
        row = load_row(path, label, exp_id)
        if row:
            rows.append(row)
        else:
            missing.append(str(path))

    out = RESULTS / "comparison_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# exp_clean comparison (12h FDE, combined_filtered_smart)",
        "",
        "| ID | Model | FDE med | FDE mean | nFDE | Straight | Maneuver | Epochs |",
        "|----|-------|---------|----------|------|----------|----------|--------|",
    ]
    for r in rows:
        def f(v):
            return "—" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
        lines.append(
            f"| {r['id']} | {r['label']} | {f(r['fde_med'])} | {f(r['fde_mean'])} | "
            f"{f(r['nfde_med'])} | {f(r['straight_fde'])} | {f(r['maneuver_fde'])} | {f(r['epochs'])} |"
        )
    if missing:
        lines += ["", "## Pending", *[f"- {m}" for m in missing]]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out}")
    if not rows and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
