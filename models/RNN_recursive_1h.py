"""
Recursive sliding-window trajectory forecasting (1h or multi-hour chunks).

Train: history (24h) -> displacement over next chunk (default 1h).
Inference: apply recursively to cover full 12h forecast with synthetic feature updates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from proj.project.coast_paths import COAST_CONFIGS, models_output_dir, resolve_windows_path, results_output_dir
from proj.project.models.plot_utils import save_training_history_plot
from proj.project.models.training_utils import (
    TrainingImprovementConfig,
    add_training_improvement_args,
    apply_residual_prediction,
    enrich_history_row,
    make_land_penalty,
    training_config_from_args,
    training_improvements_dict,
    unpack_window_batch,
)
from proj.project.window_data import (
    FEATURE_COLS,
    append_synthetic_hour_to_history,
    chunk_displacement_from_future,
    evaluate_final_position,
    evaluate_full_trajectory,
    evaluate_stratified_positions,
    haversine_km,
    hour_steps_from_minutes,
    infer_window_shape,
    kinematic_position_at_horizon,
    load_windows_filtered,
    add_stationary_filter_args,
    make_train_val_test_frames,
    naive_chunk_displacement,
    naive_one_hour_displacement,
    naive_position_at_horizon,
    one_hour_displacement_from_future,
    print_position_metrics,
    scale_history_features,
    stationary_filter_from_args,
    window_horizon_hours,
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


class HourWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_hour: np.ndarray,
        anchor: np.ndarray,
        naive_hour: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_hour = torch.tensor(y_hour, dtype=torch.float32)
        self.anchor = torch.tensor(anchor, dtype=torch.float32)
        self.naive_hour = (
            torch.tensor(naive_hour, dtype=torch.float32) if naive_hour is not None else None
        )
        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        items = [self.x[idx], self.y_hour[idx], self.anchor[idx]]
        if self.naive_hour is not None:
            items.append(self.naive_hour[idx])
        if self.sample_weights is not None:
            items.append(self.sample_weights[idx])
        return tuple(items)


class HourDisplacementRNN(nn.Module):
    """LSTM encoder + linear head predicting one-hour (lat, lon) displacement."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=rnn_dropout,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.RNN(
                input_dim, hidden_dim, num_layers=num_layers,
                nonlinearity="tanh", batch_first=True, dropout=rnn_dropout,
            )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class HourDisplacementLoss(nn.Module):
    """Huber on 1h displacement + scaled endpoint distance in km."""

    def __init__(
        self,
        haversine_weight: float = 0.5,
        huber_delta: float = 0.01,
        land_penalty: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.haversine_weight = float(haversine_weight)
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")
        self.land_penalty = land_penalty

    def forward(
        self,
        pred_disp: torch.Tensor,
        true_disp: torch.Tensor,
        anchor: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        huber = self.huber(pred_disp, true_disp).mean(dim=1)
        pred_end = anchor + pred_disp
        true_end = anchor + true_disp
        dlat_km = (pred_end[:, 0] - true_end[:, 0]) * 111.322
        dlon_km = (pred_end[:, 1] - true_end[:, 1]) * 111.322 * torch.cos(
            torch.deg2rad((pred_end[:, 0] + true_end[:, 0]) * 0.5)
        ).clamp(min=1e-3)
        dist_km = torch.sqrt(dlat_km * dlat_km + dlon_km * dlon_km + 1e-6) / 50.0
        w = self.haversine_weight
        loss = (1.0 - w) * huber + w * dist_km
        if sample_weight is not None:
            loss = loss * sample_weight
        total = loss.mean()
        if self.land_penalty is not None:
            # SoftLandPenalty expects (B, T, 2); chunk endpoint is a single step.
            total = total + self.land_penalty(pred_end.unsqueeze(1))
        return total


def train_one_epoch(
    model: HourDisplacementRNN,
    dataloader: DataLoader,
    criterion: HourDisplacementLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    residual_naive: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch_x, batch_y, batch_anchor, batch_naive, batch_weight = unpack_window_batch(
            batch, device
        )
        pred = model(batch_x)
        if residual_naive and batch_naive is not None:
            pred = apply_residual_prediction(pred, batch_naive, residual=True)
        if batch_weight is not None:
            loss = criterion(pred, batch_y, batch_anchor, batch_weight)
        else:
            loss = criterion(pred, batch_y, batch_anchor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_hour_loss(
    model: HourDisplacementRNN,
    dataloader: DataLoader,
    criterion: HourDisplacementLoss,
    device: torch.device,
    *,
    residual_naive: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch_x, batch_y, batch_anchor, batch_naive, batch_weight = unpack_window_batch(
            batch, device
        )
        pred = model(batch_x)
        if residual_naive and batch_naive is not None:
            pred = apply_residual_prediction(pred, batch_naive, residual=True)
        if batch_weight is not None:
            loss = criterion(pred, batch_y, batch_anchor, batch_weight)
        else:
            loss = criterion(pred, batch_y, batch_anchor)
        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


def scale_batch(x_raw: np.ndarray, scaler) -> np.ndarray:
    n, t, f = x_raw.shape
    flat = x_raw.reshape(n * t, f)
    return scaler.transform(flat).reshape(n, t, f).astype(np.float32)


@torch.no_grad()
def recursive_rollout_forecast(
    model: HourDisplacementRNN,
    x_raw: np.ndarray,
    anchor: np.ndarray,
    scaler,
    *,
    residual_naive: bool,
    steps_per_chunk: int,
    step_minutes: float,
    chunk_hours: float,
    forecast_hours: int = 12,
    kinematic_baseline: bool = True,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Chain chunk-wise predictions into a full absolute trajectory."""
    model.eval()
    n_samples = len(x_raw)
    steps_per_hour = hour_steps_from_minutes(step_minutes, 1.0)
    total_steps = forecast_hours * steps_per_hour
    chunk_end_step = steps_per_chunk - 1
    n_chunks = forecast_hours // int(chunk_hours)
    if n_chunks * int(chunk_hours) != forecast_hours:
        raise ValueError(f"forecast_hours={forecast_hours} must be divisible by chunk_hours={chunk_hours}")

    y_pred = np.zeros((n_samples, total_steps, 2), dtype=np.float32)
    current_pos = anchor.copy().astype(np.float32)
    history = x_raw.copy()

    for chunk_idx in range(n_chunks):
        chunk_disp = np.zeros((n_samples, 2), dtype=np.float32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_scaled = scale_batch(history[start:end], scaler)
            x_t = torch.tensor(x_scaled, device=device)
            pred = model(x_t).cpu().numpy()
            if residual_naive:
                naive_chunk = naive_chunk_displacement(
                    history[start:end],
                    chunk_end_step,
                    kinematic=kinematic_baseline,
                    step_minutes=step_minutes,
                )
                pred = pred + naive_chunk
            chunk_disp[start:end] = pred

        base_step = chunk_idx * steps_per_chunk
        for step in range(steps_per_chunk):
            frac = (step + 1) / steps_per_chunk
            idx = base_step + step
            y_pred[:, idx, :] = current_pos + chunk_disp * frac

        current_pos = current_pos + chunk_disp
        new_history = np.zeros_like(history)
        for i in range(n_samples):
            new_history[i] = append_synthetic_hour_to_history(
                history[i : i + 1],
                chunk_disp[i : i + 1],
                steps_per_hour=steps_per_chunk,
                step_minutes=step_minutes,
            )[0]
        history = new_history

    return y_pred


def recursive_rollout_12h(*args, **kwargs):
    """Backward-compatible alias."""
    return recursive_rollout_forecast(*args, **kwargs)


def run_recursive_sliding(
    *,
    input_path: Path | None,
    coast_name: str | None,
    region: str,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
    rnn_type: str = "lstm",
    batch_size: int = 256,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    patience: int = 10,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 42,
    sample_size: int | None = None,
    horizon_hours: float = 12.0,
    chunk_hours: float = 1.0,
    motion_filter=None,
    training_config: TrainingImprovementConfig | None = None,
    run_tag: str | None = None,
) -> Path:
    training_config = training_config or TrainingImprovementConfig()
    start_time = time.perf_counter()

    input_path, coast, region = resolve_windows_path(coast_name, region, input_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prefetch_size = int(sample_size / 0.7) + 1000 if sample_size else None
    print(f"Loading windows from: {input_path}")
    df = load_windows_filtered(
        input_path,
        sample_size=prefetch_size,
        motion_filter=motion_filter,
        maneuver_oversample=False,
        motion_balanced_sample=False,
        seed=seed,
    )

    model_tag = "RNN_recursive_sliding"
    output_dir = results_output_dir(coast, input_path, model_tag, df, run_tag=run_tag)
    model_dir = models_output_dir(coast, input_path, df, run_tag=run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    history_steps, future_steps = infer_window_shape(df)
    step_minutes = float(df["resample_minutes"].iloc[0]) if "resample_minutes" in df.columns else 10.0
    steps_per_chunk = hour_steps_from_minutes(step_minutes, chunk_hours)
    chunk_end_step = steps_per_chunk - 1
    forecast_hours = int(round(horizon_hours))
    n_chunks = forecast_hours // int(chunk_hours)
    if n_chunks < 1 or n_chunks * int(chunk_hours) != forecast_hours:
        raise ValueError(
            f"forecast_hours={forecast_hours} must be a positive multiple of chunk_hours={chunk_hours}"
        )

    from proj.project.window_data import build_window_arrays, compute_sample_weights, horizon_step_index

    horizon_step = horizon_step_index(df, horizon_hours, future_steps)

    train_df, val_df, test_df, split_col = make_train_val_test_frames(
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

    def _load_split(split_df):
        x, y_abs, _, anchor = build_window_arrays(split_df, history_steps=history_steps, future_steps=future_steps)
        y_chunk = chunk_displacement_from_future(y_abs, anchor, chunk_end_step)
        naive_h = (
            naive_chunk_displacement(
                x, chunk_end_step, kinematic=training_config.kinematic_baseline, step_minutes=step_minutes
            )
            if training_config.residual_naive
            else None
        )
        return x, y_abs, y_chunk, anchor, naive_h

    x_train, _, y_hour_train, anchor_train, naive_train = _load_split(train_df)
    x_val, _, y_hour_val, anchor_val, naive_val = _load_split(val_df)
    x_test, y_test_abs, y_hour_test, anchor_test, naive_test = _load_split(test_df)
    x_test_raw = x_test.copy()

    train_weights = compute_sample_weights(x_train) if training_config.difficulty_weighting else None
    x_train, [x_val, x_test], scaler = scale_history_features(x_train, [x_val, x_test])

    train_ds = HourWindowDataset(x_train, y_hour_train, anchor_train, naive_train, train_weights)
    val_ds = HourWindowDataset(x_val, y_hour_val, anchor_val, naive_val)
    test_ds = HourWindowDataset(x_test, y_hour_test, anchor_test, naive_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    train_eval_size = min(len(train_ds), max(len(val_ds), 10_000))
    train_eval_idx = np.random.default_rng(seed).choice(len(train_ds), size=train_eval_size, replace=False)
    train_eval_loader = DataLoader(
        Subset(
            HourWindowDataset(x_train, y_hour_train, anchor_train, naive_train),
            train_eval_idx.tolist(),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = HourDisplacementRNN(
        input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim, num_layers=num_layers,
        dropout=dropout, rnn_type=rnn_type,
    ).to(device)
    criterion = HourDisplacementLoss(
        haversine_weight=training_config.haversine_weight,
        land_penalty=make_land_penalty(training_config.land_penalty_weight, device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    print(
        f"\n=== {coast.name} | {output_dir.parent.name} | "
        f"recursive sliding {chunk_hours:g}h x {n_chunks} chunks ==="
    )
    print(
        f"Train {chunk_hours:g}h displacement | eval {forecast_hours}h recursive | "
        f"steps/chunk={steps_per_chunk} | residual={'on' if training_config.residual_naive else 'off'} | "
        f"land_penalty={training_config.land_penalty_weight:.3f}"
    )

    best_val = float("inf")
    best_state = None
    stale = 0
    history = []
    total_train_time = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            residual_naive=training_config.residual_naive,
        )
        val_loss = evaluate_hour_loss(
            model, val_loader, criterion, device, residual_naive=training_config.residual_naive,
        )
        train_eval_loss = evaluate_hour_loss(
            model, train_eval_loader, criterion, device, residual_naive=training_config.residual_naive,
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        total_train_time += elapsed
        history.append(
            enrich_history_row(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr, epoch_sec=elapsed,
                train_eval_loss=train_eval_loss,
            )
        )
        print(
            f"| epoch {epoch:3d} | time: {elapsed:6.2f}s | "
            f"train {train_loss:8.5f} | val {val_loss:8.5f} | train_eval {train_eval_loss:8.5f} | "
            f"lr {lr:.2e}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_epoch = int(min(history, key=lambda row: row["val_loss"])["epoch"])
    test_loss_at_best = evaluate_hour_loss(
        model, test_loader, criterion, device, residual_naive=training_config.residual_naive,
    )
    print(f"Test loss @ best epoch {best_epoch}: {test_loss_at_best:.6f}")

    steps_per_hour = hour_steps_from_minutes(step_minutes, 1.0)
    y_true_traj = y_test_abs[:, : forecast_hours * steps_per_hour, :].copy()
    y_pred_traj = recursive_rollout_forecast(
        model, x_test_raw, anchor_test, scaler,
        residual_naive=training_config.residual_naive,
        steps_per_chunk=steps_per_chunk,
        step_minutes=step_minutes,
        chunk_hours=chunk_hours,
        forecast_hours=forecast_hours,
        kinematic_baseline=training_config.kinematic_baseline,
        device=device,
        batch_size=batch_size,
    )

    eval_true = y_true_traj[:, horizon_step, :]
    eval_pred = y_pred_traj[:, horizon_step, :]
    baseline_true, baseline_pred = naive_position_at_horizon(x_test_raw, y_test_abs, horizon_step)
    kin_true, kin_pred = kinematic_position_at_horizon(
        x_test_raw, y_test_abs, horizon_step, step_minutes=step_minutes,
    )

    model_name = f"Recursive {chunk_hours:g}h sliding LSTM"
    metrics = [
        evaluate_final_position(baseline_true, baseline_pred, "Naive baseline", anchor=anchor_test),
        evaluate_final_position(kin_true, kin_pred, "Kinematic baseline", anchor=anchor_test),
        evaluate_final_position(
            eval_true, eval_pred,
            f"{model_name}: recursive {horizon_hours:.0f}h position",
            anchor=anchor_test,
        ),
        evaluate_full_trajectory(y_true_traj, y_pred_traj, f"{model_name}: full recursive trajectory"),
    ]
    metrics.extend(
        evaluate_stratified_positions(
            eval_true, eval_pred, x_test_raw,
            f"{model_name}: recursive {horizon_hours:.0f}h position",
            anchor=anchor_test,
        )
    )
    for item in metrics:
        print_position_metrics(item)

    metrics_path = output_dir / "recursive_sliding_metrics.json"
    results = {
        "input": str(input_path),
        "coast": coast.name,
        "run_tag": run_tag,
        "experiment_id": run_tag.split("/")[-1] if run_tag else None,
        "model_family": "recursive_sliding",
        "chunk_hours": float(chunk_hours),
        "forecast_hours": forecast_hours,
        "steps_per_chunk": steps_per_chunk,
        "n_recursive_chunks": n_chunks,
        "architecture": {"type": rnn_type, "hidden_dim": hidden_dim, "num_layers": num_layers},
        "training": {
            "epochs_ran": len(history),
            "best_val_loss": float(best_val),
            "improvements": training_improvements_dict(training_config),
        },
        "metrics": metrics,
        "history": history,
        "eval_summary": {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "test_loss_at_best": float(test_loss_at_best),
        },
        "runtime_sec": round(time.perf_counter() - start_time, 2),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_training_history_plot(
        history,
        output_dir / "recursive_sliding_training_history.png",
        title=f"Recursive {chunk_hours:g}h sliding — chunk train loss",
        loss_label=f"{chunk_hours:g}h displacement loss",
        autoregressive=False,
        curriculum_enabled=False,
        test_loss_at_best=test_loss_at_best,
        test_epoch=best_epoch,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "feature_cols": FEATURE_COLS,
            "history_steps": history_steps,
            "chunk_hours": chunk_hours,
            "steps_per_chunk": steps_per_chunk,
            "forecast_hours": forecast_hours,
        },
        model_dir / "ship_trajectory_recursive_sliding.pt",
    )

    print(f"\nSaved: {metrics_path}")
    print(f"Runtime: {format_duration(time.perf_counter() - start_time)}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive sliding-window chunk forecaster.")
    parser.add_argument("--chunk-hours", type=float, default=1.0, help="Hours per recursive chunk (e.g. 1 or 3).")
    parser.add_argument("--coast", choices=sorted(COAST_CONFIGS.keys()), default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rnn-type", default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--horizon-hours", type=float, default=12.0)
    parser.add_argument("--run-tag", type=str, default=None)
    add_stationary_filter_args(parser)
    add_training_improvement_args(parser)
    args = parser.parse_args()

    region = args.region or (
        COAST_CONFIGS[args.coast].default_region if args.coast else COAST_CONFIGS["Eastern coast"].default_region
    )

    run_recursive_sliding(
        input_path=args.input,
        coast_name=args.coast,
        region=region,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        sample_size=None if args.sample <= 0 else args.sample,
        horizon_hours=args.horizon_hours,
        chunk_hours=args.chunk_hours,
        motion_filter=stationary_filter_from_args(args),
        training_config=training_config_from_args(args),
        run_tag=args.run_tag,
    )


run_recursive_1h = run_recursive_sliding


if __name__ == "__main__":
    main()
