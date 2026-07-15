#!/usr/bin/env python3
"""Compare AR ablation experiments A–E (+ flat baseline) on smart_motion data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "data/results/USA Combined/unknown"

EXPERIMENTS = {
    "flat": ("ar_exp/flat", "RNN", "lstm_metrics.json", "Flat LSTM (direct)"),
    "A": ("ar_exp/A", "RNN_AR_LSTM", "lstm_ar_metrics.json", "A: AR anchor, no residual/schedule"),
    "B": ("ar_exp/B", "RNN_AR_LSTM", "lstm_ar_metrics.json", "B: AR anchor + residual"),
    "C": ("ar_exp/C", "RNN_AR_LSTM", "lstm_ar_metrics.json", "C: AR anchor + residual + schedule"),
    "D": ("ar_exp/D", "RNN_AR_LSTM", "lstm_ar_metrics.json", "D: AR step-delta + residual + schedule"),
    "E": ("ar_exp/E", "RNN_recursive_1h", "recursive_1h_metrics.json", "E: recursive 1h sliding window"),
}

# Also include prior smart_motion runs for reference
REFERENCE = {
    "v1_RNN": ("v1/smart_motion", "RNN", "lstm_metrics.json"),
    "v1_AR": ("v1/smart_motion", "RNN_AR_LSTM", "lstm_ar_metrics.json"),
    "v1r_AR": ("v1_residual/smart_motion", "RNN_AR_LSTM", "lstm_ar_metrics.json"),
}


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_row(data: dict, label: str) -> dict | None:
    if data is None:
        return None
    main = None
    straight = None
    maneuver = None
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
        elif main is None and ("position" in name or "recursive" in name):
            main = m
    if main is None:
        return None
    row = {
        "experiment": label,
        "epochs": data.get("training", {}).get("epochs_ran"),
        "target_mode": data.get("target_mode", data.get("model_family", "—")),
        "fde_med_km": main.get("median_error_km"),
        "fde_mean_km": main.get("mean_error_km"),
        "nfde_med": main.get("median_nfde"),
        "ade_med_km": None,
    }
    for m in data.get("metrics", []):
        if "full predicted" in m.get("model", "") or "full recursive" in m.get("model", ""):
            row["ade_med_km"] = m.get("median_ade_km")
    if straight:
        row["straight_fde_med"] = straight.get("median_error_km")
    if maneuver:
        row["maneuver_fde_med"] = maneuver.get("median_error_km")
    return row


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "ar_exp" / "comparison_summary.md")
    parser.add_argument("--include-reference", action="store_true", help="Include prior v1/smart_motion runs")
    args = parser.parse_args()

    rows: list[dict] = []
    missing: list[str] = []

    for exp_id, (tag, subdir, fname, label) in EXPERIMENTS.items():
        path = RESULTS / tag / subdir / fname
        data = load_metrics(path)
        if data is None:
            missing.append(f"{exp_id} ({path})")
            continue
        row = extract_row(data, label)
        if row:
            row["id"] = exp_id
            rows.append(row)

    if args.include_reference:
        for ref_id, (tag, subdir, fname) in REFERENCE.items():
            path = RESULTS / tag / subdir / fname
            row = extract_row(load_metrics(path), ref_id)
            if row:
                row["id"] = ref_id
                rows.append(row)

    if not rows:
        print("No completed experiment metrics found.", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        sys.exit(1)

    lines = [
        "# AR Ablation Comparison (12h FDE, smart_motion data)",
        "",
        "| ID | Experiment | FDE med (km) | FDE mean | nFDE med | ADE med | Straight FDE | Maneuver FDE |",
        "|----|------------|--------------|----------|----------|---------|--------------|--------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('id', '')} | {r['experiment'][:40]} | "
            f"{fmt(r['fde_med_km'])} | {fmt(r['fde_mean_km'])} | {fmt(r['nfde_med'], 3)} | "
            f"{fmt(r['ade_med_km'])} | {fmt(r.get('straight_fde_med'))} | {fmt(r.get('maneuver_fde_med'))} |"
        )

    lines.extend([
        "",
        "## Planned comparisons",
        "- **flat vs A**: direct vs autoregressive",
        "- **A vs B**: effect of residual",
        "- **B vs C**: effect of TF + curriculum",
        "- **C vs D**: anchor-offset vs step-delta",
        "- **D vs E**: internal AR decoder vs recursive 1h",
        "",
    ])
    if missing:
        lines.append("## Pending runs")
        for m in missing:
            lines.append(f"- {m}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {args.output}")
    for line in lines[3:3 + len(rows) + 1]:
        print(line)


if __name__ == "__main__":
    main()
