#!/usr/bin/env python3
"""Regenerate training history plots from metrics JSON (no retraining needed)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1]))

from proj.project.models.plot_utils import save_training_history_plot


def plot_from_metrics(metrics_path: Path, output_path: Path | None = None) -> Path:
    with metrics_path.open(encoding="utf-8") as f:
        results = json.load(f)

    history = results.get("history") or results.get("training_history")
    if not history:
        raise ValueError(f"No history found in {metrics_path}")

    training = results.get("training", {})
    improvements = training.get("improvements", {})
    architecture = results.get("architecture", {})

    future_steps = int(results.get("future_steps", 72))
    resample_minutes = 10
    autoregressive = architecture.get("variant") == "autoregressive"

    if output_path is None:
        stem = metrics_path.stem
        if stem.endswith("_metrics"):
            stem = stem[: -len("_metrics")]
        output_path = metrics_path.with_name(f"{stem}_training_history.png")

    eval_summary = results.get("eval_summary") or {}
    test_loss_at_best = eval_summary.get("test_loss_at_best")
    test_epoch = eval_summary.get("best_epoch")

    model_type = architecture.get("type", "model").upper()
    title = f"{model_type}{'-AR' if autoregressive else ''} Training History"

    save_training_history_plot(
        history,
        output_path,
        title=title,
        loss_label="Huber + Haversine loss",
        autoregressive=autoregressive,
        future_steps=future_steps,
        resample_minutes=resample_minutes,
        curriculum_start_hours=float(improvements.get("curriculum_start_hours", 6.0)),
        curriculum_enabled=bool(improvements.get("curriculum", True)),
        teacher_forcing_start=float(
            improvements.get("teacher_forcing_start", architecture.get("teacher_forcing_ratio", 0.3))
        ),
        use_scheduled_teacher_forcing=bool(improvements.get("scheduled_teacher_forcing", True)),
        test_loss_at_best=float(test_loss_at_best) if test_loss_at_best is not None else None,
        test_epoch=int(test_epoch) if test_epoch is not None else None,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+", help="Path(s) to *_metrics.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path (single input only)")
    args = parser.parse_args()

    if args.output is not None and len(args.metrics) != 1:
        parser.error("--output requires exactly one metrics file")

    for metrics_path in args.metrics:
        out = plot_from_metrics(metrics_path, args.output)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
