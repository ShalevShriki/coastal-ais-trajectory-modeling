#!/usr/bin/env python3
"""Build comparison table for separate-encoder adaptive gate experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "data/results/USA Combined/unknown/exp_coastal"


def load_gate_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    gate = data.get("test_gate_summary") or {}
    return {
        "gate_mode": data.get("gate_mode", "?"),
        "median_fde": gate.get("median_fde_km"),
        "mean_fde": gate.get("mean_fde_km"),
        "median_ade": gate.get("median_ade_km"),
        "mean_ade": gate.get("mean_ade_km"),
        "most_selected": gate.get("most_selected_context"),
        "selection_share": gate.get("selection_share_pct"),
        "gate_entropy": gate.get("gate_entropy_mean"),
        "softmax_entropy": gate.get("softmax_entropy_mean"),
        "alpha_mean": gate.get("alpha_mean", {}),
        "argmax_pct": gate.get("argmax_pct", {}),
        "cosines": gate.get("representation_cosines", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--softmax-metrics",
        type=Path,
        default=RESULTS / "adaptive_separate_encoders_softmax/RNN_AR_diff_encoder/diff_encoder_adaptive_metrics.json",
    )
    parser.add_argument(
        "--hard-metrics",
        type=Path,
        default=RESULTS / "adaptive_separate_encoders_hard/RNN_AR_diff_encoder/diff_encoder_adaptive_metrics.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "adaptive_separate_encoders_comparison.md",
    )
    args = parser.parse_args()

    rows = []
    for label, path in [("Softmax", args.softmax_metrics), ("Hard", args.hard_metrics)]:
        if not path.exists():
            print(f"Missing: {path}")
            continue
        s = load_gate_summary(path)
        rows.append((label, s))

    if not rows:
        raise SystemExit("No metrics files found — run training first.")

    lines = [
        "# Separate-encoder adaptive gate comparison",
        "",
        "| Model | Gate | Median FDE | Mean FDE | Median ADE | Mean ADE | Most Selected | Selection % | Gate H |",
        "|-------|------|------------|----------|------------|----------|---------------|-------------|--------|",
    ]
    for label, s in rows:
        lines.append(
            f"| Separate Encoders | {label} | "
            f"{s['median_fde']:.2f} | {s['mean_fde']:.2f} | "
            f"{s['median_ade']:.2f} | {s['mean_ade']:.2f} | "
            f"{s['most_selected']} | {s['selection_share']:.1f}% | "
            f"{s['gate_entropy']:.3f} |"
        )

    lines.extend(["", "## Representation cosine (mean hidden states)", ""])
    for label, s in rows:
        c = s["cosines"]
        lines.append(
            f"- **{label}**: cos(9,12)={c.get('cos_9_12', 0):.3f}, "
            f"cos(12,18)={c.get('cos_12_18', 0):.3f}, "
            f"cos(18,24)={c.get('cos_18_24', 0):.3f}"
        )

    lines.extend(["", "## Alpha / selection breakdown", ""])
    for label, s in rows:
        lines.append(f"### {label}")
        lines.append(f"- mean α: {s['alpha_mean']}")
        lines.append(f"- argmax %: {s['argmax_pct']}")
        if s.get("softmax_entropy") is not None:
            lines.append(f"- softmax entropy (pre-hard): {s['softmax_entropy']:.3f}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
