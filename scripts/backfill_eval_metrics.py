#!/usr/bin/env python3
"""Backfill test loss (once @ best checkpoint) and regenerate training plots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1]))

from proj.project.coast_paths import resolve_windows_path
from proj.project.models.RNN_AR import (
    FEATURE_COLS,
    ShipTrajectoryARRNN,
    WindowDataset,
    evaluate_loss,
)
from proj.project.models.training_utils import TrajectoryLoss, TrainingImprovementConfig
from proj.project.window_data import (
    build_window_arrays,
    load_windows_filtered,
    make_train_val_test_frames,
    resolve_window_hours,
)
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
    autoregressive = architecture.get("variant") == "autoregressive"
    eval_summary = results.get("eval_summary") or {}

    if output_path is None:
        stem = metrics_path.stem
        if stem.endswith("_metrics"):
            stem = stem[: -len("_metrics")]
        output_path = metrics_path.with_name(f"{stem}_training_history.png")

    model_type = architecture.get("type", "model").upper()
    title = f"{model_type}{'-AR' if autoregressive else ''} Training History"
    save_training_history_plot(
        history,
        output_path,
        title=title,
        loss_label="Huber + Haversine loss",
        autoregressive=autoregressive,
        future_steps=future_steps,
        resample_minutes=10,
        curriculum_start_hours=float(improvements.get("curriculum_start_hours", 6.0)),
        curriculum_enabled=bool(improvements.get("curriculum", True)),
        teacher_forcing_start=float(
            improvements.get("teacher_forcing_start", architecture.get("teacher_forcing_ratio", 0.3))
        ),
        use_scheduled_teacher_forcing=bool(improvements.get("scheduled_teacher_forcing", True)),
        test_loss_at_best=float(eval_summary["test_loss_at_best"])
        if eval_summary.get("test_loss_at_best") is not None
        else None,
        test_epoch=int(eval_summary["best_epoch"]) if eval_summary.get("best_epoch") is not None else None,
    )
    return output_path


def _project_root() -> Path:
    return PROJECT


def infer_checkpoint_path(metrics_path: Path, results: dict) -> Path:
    arch = results.get("architecture", {})
    rnn_type = arch.get("type", "lstm")
    variant = arch.get("variant", "")
    coast = results["coast"]
    run_tag = results.get("run_tag") or ""
    days_labels = [results.get("days_label", "unknown"), "unknown"]
    days_labels = list(dict.fromkeys(l for l in days_labels if l))

    if variant == "autoregressive" or metrics_path.name.endswith("_ar_metrics.json"):
        ckpt_name = f"ship_trajectory_{rnn_type}_ar.pt"
    elif "adaptive" in metrics_path.name:
        ckpt_name = "adaptive_ar.pt"
    elif "transformer" in metrics_path.name:
        ckpt_name = "ship_trajectory_transformer.pt"
    elif "recursive" in metrics_path.name:
        ckpt_name = "recursive_sliding.pt"
    else:
        ckpt_name = f"ship_trajectory_{rnn_type}.pt"

    candidates: list[Path] = []
    for days_label in days_labels:
        base = _project_root() / "data" / "models" / coast / days_label
        if run_tag:
            candidates.append(base / run_tag / ckpt_name)
        candidates.append(base / ckpt_name)

    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _training_config(results: dict) -> TrainingImprovementConfig:
    improvements = results.get("training", {}).get("improvements", {})
    if not improvements and isinstance(results.get("training"), dict):
        improvements = results["training"]
    return TrainingImprovementConfig(
        haversine_weight=float(improvements.get("haversine_weight", 0.5)),
        relative_loss_weight=float(improvements.get("relative_loss_weight", 0.0)),
        min_path_km=float(improvements.get("min_path_km", 10.0)),
        difficulty_weighting=bool(improvements.get("difficulty_weighting", True)),
        maneuver_oversample=bool(improvements.get("maneuver_oversample", False)),
        maneuver_fraction=float(improvements.get("maneuver_fraction", 0.3)),
        motion_balanced_sample=bool(improvements.get("motion_balanced_sample", False)),
        straight_fraction=float(improvements.get("straight_fraction", 0.15)),
        other_fraction=float(improvements.get("other_fraction", 0.15)),
        residual_naive=bool(improvements.get("residual_naive", False)),
        kinematic_baseline=bool(improvements.get("kinematic_baseline", True)),
        split_by=str(improvements.get("split_by", results.get("split_by", "trajectory"))),
        curriculum=bool(improvements.get("curriculum", True)),
        curriculum_start_hours=float(improvements.get("curriculum_start_hours", 6.0)),
        scheduled_teacher_forcing=bool(improvements.get("scheduled_teacher_forcing", True)),
        teacher_forcing_start=float(improvements.get("teacher_forcing_start", 0.3)),
        teacher_forcing_end=float(improvements.get("teacher_forcing_end", 0.0)),
    )


def evaluate_ar_checkpoint(
    metrics_path: Path,
    checkpoint_path: Path | None = None,
    *,
    sample_size: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    with metrics_path.open(encoding="utf-8") as f:
        results = json.load(f)

    history = results.get("history") or results.get("training_history")
    if not history:
        raise ValueError(f"No training history in {metrics_path}")

    checkpoint_path = checkpoint_path or infer_checkpoint_path(metrics_path, results)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    splits = results.get("splits", {})
    seed = int(splits.get("seed", 42))
    test_fraction = float(splits.get("test_fraction", 0.2))
    val_fraction = float(splits.get("val_fraction", 0.1))
    training_config = _training_config(results)
    target_mode = results.get("target_mode", "anchor_offset")

    input_path = Path(results["input"])
    if not input_path.is_absolute():
        input_path = _project_root() / input_path
    coast_name = results.get("coast")
    region = results.get("region", "usa_combined")
    input_path, coast, _region = resolve_windows_path(coast_name, region, input_path)

    arch = results["architecture"]
    history_steps = int(results["history_steps"])
    future_steps = int(results["future_steps"])
    batch_size = int(batch_size or results.get("training", {}).get("batch_size", 256))

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    def _apply_checkpoint_scaler(x: np.ndarray) -> np.ndarray:
        ckpt_scaler = StandardScaler()
        ckpt_scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
        ckpt_scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
        ckpt_scaler.n_features_in_ = len(ckpt_scaler.mean_)
        n, steps, n_feat = x.shape
        flat = x.reshape(n * steps, n_feat)
        return ckpt_scaler.transform(flat).reshape(n, steps, n_feat).astype(np.float32)

    df = load_windows_filtered(input_path, sample_size=sample_size, seed=seed)
    _, _, _, _, step_minutes = resolve_window_hours(df)
    history_hours = history_steps * step_minutes / 60
    future_hours = future_steps * step_minutes / 60
    full_history_steps, _full_future, history_steps, future_steps, step_minutes = resolve_window_hours(
        df,
        history_hours=history_hours,
        future_hours=future_hours,
    )

    train_df, val_df, test_df, _split_col = make_train_val_test_frames(
        df,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        seed=seed,
        split_by=training_config.split_by,
        train_sample_size=sample_size,
        maneuver_oversample=training_config.maneuver_oversample,
        maneuver_fraction=training_config.maneuver_fraction,
        motion_balanced_sample=training_config.motion_balanced_sample,
        straight_fraction=training_config.straight_fraction,
        other_fraction=training_config.other_fraction,
    )

    x_train, _, y_delta_train, anchor_train = build_window_arrays(
        train_df,
        history_steps=history_steps,
        future_steps=future_steps,
        full_history_steps=full_history_steps,
        target_mode=target_mode,
    )
    _x_val, _, y_delta_val, anchor_val = build_window_arrays(
        val_df,
        history_steps=history_steps,
        future_steps=future_steps,
        full_history_steps=full_history_steps,
        target_mode=target_mode,
    )
    x_test, _, y_delta_test, anchor_test = build_window_arrays(
        test_df,
        history_steps=history_steps,
        future_steps=future_steps,
        full_history_steps=full_history_steps,
        target_mode=target_mode,
    )

    x_train = _apply_checkpoint_scaler(x_train)
    x_test = _apply_checkpoint_scaler(x_test)

    test_dataset = WindowDataset(x_test, y_delta_test, anchor_test, None, None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_eval_size = min(len(x_train), max(len(val_df), 10_000))
    train_eval_idx = np.random.default_rng(seed).choice(len(x_train), size=train_eval_size, replace=False)
    train_eval_dataset = WindowDataset(x_train, y_delta_train, anchor_train, None, None)
    train_eval_loader = DataLoader(
        Subset(train_eval_dataset, train_eval_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ShipTrajectoryARRNN(
        input_dim=len(FEATURE_COLS),
        future_steps=future_steps,
        hidden_dim=int(arch.get("hidden_dim", ckpt.get("hidden_dim", 128))),
        num_layers=int(arch.get("num_layers", ckpt.get("num_layers", 2))),
        dropout=float(arch.get("dropout", ckpt.get("dropout", 0.2))),
        rnn_type=str(arch.get("type", ckpt.get("rnn_type", "lstm"))),
    ).to(torch_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = TrajectoryLoss(
        haversine_weight=training_config.haversine_weight,
        relative_weight=training_config.relative_loss_weight,
        min_path_km=training_config.min_path_km,
        target_mode=target_mode,
    )

    test_loss_at_best = evaluate_loss(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=torch_device,
        residual_naive=training_config.residual_naive,
    )
    train_eval_at_best = evaluate_loss(
        model=model,
        dataloader=train_eval_loader,
        criterion=criterion,
        device=torch_device,
        residual_naive=training_config.residual_naive,
    )

    best_epoch = int(min(history, key=lambda row: row["val_loss"])["epoch"])
    best_val_loss = float(min(row["val_loss"] for row in history))

    eval_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss_at_best": float(test_loss_at_best),
        "train_eval_loss_at_best_unweighted": float(train_eval_at_best),
    }
    return {
        "results": results,
        "eval_summary": eval_summary,
        "checkpoint_path": str(checkpoint_path),
    }


def backfill_metrics_file(
    metrics_path: Path,
    *,
    checkpoint_path: Path | None = None,
    sample_size: int | None = None,
    replot: bool = True,
) -> Path:
    payload = evaluate_ar_checkpoint(
        metrics_path,
        checkpoint_path,
        sample_size=sample_size,
    )
    results = payload["results"]
    results["eval_summary"] = payload["eval_summary"]

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"{metrics_path.name}: test_loss_at_best={payload['eval_summary']['test_loss_at_best']:.6f} "
        f"@ epoch {payload['eval_summary']['best_epoch']} "
        f"(checkpoint: {payload['checkpoint_path']})"
    )
    print(
        f"  train_eval @ best (unweighted): "
        f"{payload['eval_summary']['train_eval_loss_at_best_unweighted']:.6f}"
    )

    if replot:
        out = plot_from_metrics(metrics_path)
        print(f"  replotted: {out}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+", help="Path(s) to *_metrics.json")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint .pt path")
    parser.add_argument("--sample", type=int, default=None, help="Train sample cap used in original run")
    parser.add_argument("--no-replot", action="store_true", help="Only update metrics JSON")
    args = parser.parse_args()

    for metrics_path in args.metrics:
        backfill_metrics_file(
            metrics_path,
            checkpoint_path=args.checkpoint,
            sample_size=args.sample,
            replot=not args.no_replot,
        )


if __name__ == "__main__":
    main()
