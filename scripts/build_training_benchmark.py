#!/usr/bin/env python3
"""Merge per-model metrics JSON files into one benchmark comparison file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick_metrics(metrics: list[dict]) -> dict:
    out: dict = {}
    for row in metrics:
        name = row.get("model", "")
        if "Naive baseline" in name:
            out["naive_12h"] = {k: v for k, v in row.items() if k != "model"}
        elif (
            "full predicted" not in name
            and "[straight" not in name
            and "[maneuver" not in name
            and "[anchored" not in name
            and "[other" not in name
            and "position (" in name
        ):
            out["model_12h"] = {k: v for k, v in row.items() if k != "model"}
        elif "[straight" in name:
            out["straight_12h"] = {k: v for k, v in row.items() if k != "model"}
    return out


def load_model_entry(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    hist = data.get("history", [])
    best_val = data["training"]["best_val_loss"]
    best_epoch = next(
        (h["epoch"] for h in hist if abs(h["val_loss"] - best_val) < 1e-9),
        None,
    )
    return {
        "metrics_path": str(path),
        "architecture": data.get("architecture", {}),
        "training": {**data["training"], "best_epoch": best_epoch},
        "compute": data.get("compute", {}),
        "runtime_sec": data.get("runtime_sec"),
        "samples": {
            "total": data["samples_total"],
            "train": data["samples_train"],
            "val": data["samples_val"],
            "test": data["samples_test"],
        },
        "test_metrics": pick_metrics(data.get("metrics", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training benchmark JSON.")
    parser.add_argument("--label", required=True, help="Benchmark label, e.g. baseline_v1")
    parser.add_argument("--description", default="", help="Human-readable run description")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flat-lstm", type=Path, help="RNN lstm_metrics.json")
    parser.add_argument("--lstm-ar", type=Path, help="RNN_AR lstm_ar_metrics.json")
    parser.add_argument("--transformer", type=Path, help="Transformer transformer_metrics.json")
    args = parser.parse_args()

    models: dict[str, dict] = {}
    if args.flat_lstm:
        models["flat_lstm"] = load_model_entry(args.flat_lstm)
    if args.lstm_ar:
        models["lstm_ar"] = load_model_entry(args.lstm_ar)
    if args.transformer:
        models["transformer"] = load_model_entry(args.transformer)

    report = {
        "label": args.label,
        "description": args.description,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(models)} models)")


if __name__ == "__main__":
    main()
