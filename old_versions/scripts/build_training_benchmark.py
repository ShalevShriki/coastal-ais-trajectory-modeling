#!/usr/bin/env python3
"""Merge per-model metrics JSON files into one benchmark comparison file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric_row(metrics: list[dict], predicate) -> dict | None:
    for row in metrics:
        if predicate(row.get("model", "")):
            return {k: v for k, v in row.items() if k != "model"}
    return None


def pick_metrics(metrics: list[dict]) -> dict:
    """Extract absolute + normalized FDE/ADE and stratified buckets."""
    out: dict = {}

    naive = _metric_row(metrics, lambda n: "Naive baseline" in n)
    if naive:
        out["naive_12h"] = naive

    kinematic = _metric_row(metrics, lambda n: "Kinematic baseline" in n)
    if kinematic:
        out["kinematic_12h"] = kinematic

    model_12h = _metric_row(
        metrics,
        lambda n: (
            "full predicted" not in n
            and "[" not in n
            and "position (" in n
            and "baseline" not in n.lower()
        ),
    )
    if model_12h:
        out["model_12h"] = model_12h

    full_traj = _metric_row(metrics, lambda n: "full predicted trajectory" in n)
    if full_traj:
        out["model_full_traj"] = full_traj

    for bucket in ("straight", "maneuver", "anchored", "other"):
        row = _metric_row(metrics, lambda n, b=bucket: f"[{b}," in n)
        if row:
            out[f"{bucket}_12h"] = row

    return out


def build_summary(test_metrics: dict) -> dict:
    """Compact comparison keys for tables (km + normalized ratios)."""
    summary: dict = {}

    def _fde(row: dict | None, prefix: str) -> None:
        if not row:
            return
        summary[f"{prefix}_fde_km_mean"] = row.get("mean_error_km")
        summary[f"{prefix}_fde_km_median"] = row.get("median_error_km")
        if "mean_nfde" in row:
            summary[f"{prefix}_nfde_mean"] = row["mean_nfde"]
            summary[f"{prefix}_nfde_median"] = row.get("median_nfde")

    def _ade(row: dict | None, prefix: str) -> None:
        if not row:
            return
        summary[f"{prefix}_ade_km_mean"] = row.get("mean_ade_km")
        summary[f"{prefix}_ade_km_median"] = row.get("median_ade_km")
        if "mean_nade" in row:
            summary[f"{prefix}_nade_mean"] = row["mean_nade"]
            summary[f"{prefix}_nade_median"] = row.get("median_nade")

    _fde(test_metrics.get("naive_12h"), "naive")
    _fde(test_metrics.get("kinematic_12h"), "kinematic")
    _fde(test_metrics.get("model_12h"), "model")
    _ade(test_metrics.get("model_full_traj"), "model")

    for bucket in ("straight", "maneuver", "anchored", "other"):
        _fde(test_metrics.get(f"{bucket}_12h"), bucket)

    return summary


def load_model_entry(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    hist = data.get("history", [])
    best_val = data["training"]["best_val_loss"]
    best_epoch = next(
        (h["epoch"] for h in hist if abs(h["val_loss"] - best_val) < 1e-9),
        None,
    )
    test_metrics = pick_metrics(data.get("metrics", []))
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
        "split_by": data.get("split_by"),
        "test_metrics": test_metrics,
        "summary": build_summary(test_metrics),
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
