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
from torch.utils.data import Dataset, DataLoader

from proj.project.coast_paths import COAST_CONFIGS, models_output_dir, resolve_windows_path, results_output_dir
from proj.project.window_data import (
    FEATURE_COLS,
    build_window_arrays,
    compute_naive_cumulative_delta,
    compute_sample_weights,
    evaluate_final_position,
    evaluate_full_trajectory,
    evaluate_stratified_positions,
    haversine_km,
    horizon_step_index,
    infer_window_shape,
    load_windows_filtered,
    add_stationary_filter_args,
    stationary_filter_from_args,
    mask_by_split,
    naive_position_at_horizon,
    print_position_metrics,
    scale_history_features,
    trajectory_splits,
    window_horizon_hours,
)
from proj.project.models.training_utils import (
    TrajectoryLoss,
    TrainingImprovementConfig,
    add_training_improvement_args,
    apply_residual_prediction,
    curriculum_train_steps,
    scheduled_teacher_forcing,
    training_config_from_args,
    training_improvements_dict,
    unpack_window_batch,
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Each sample is one trajectory window.

    x:
        Past AIS features.
        Shape: (history_steps, num_features)

    y_delta:
        Future position deltas relative to the last known position.
        Shape: (future_steps, 2)

    anchor:
        Last known absolute position [lat, lon].
        Shape: (2,)
    """

    def __init__(
        self,
        x: np.ndarray,
        y_delta: np.ndarray,
        anchor: np.ndarray,
        sample_weights: np.ndarray | None = None,
        naive_delta: np.ndarray | None = None,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_delta = torch.tensor(y_delta, dtype=torch.float32)
        self.anchor = torch.tensor(anchor, dtype=torch.float32)
        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else None
        )
        self.naive_delta = (
            torch.tensor(naive_delta, dtype=torch.float32)
            if naive_delta is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        items = [self.x[idx], self.y_delta[idx], self.anchor[idx]]
        if self.naive_delta is not None:
            items.append(self.naive_delta[idx])
        if self.sample_weights is not None:
            items.append(self.sample_weights[idx])
        return tuple(items)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_rnn(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if rnn_type == "rnn":
        return nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout,
        )
    if rnn_type == "gru":
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
    if rnn_type == "lstm":
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
    raise ValueError("rnn_type must be one of: 'rnn', 'gru', 'lstm'")


class ShipTrajectoryARRNN(nn.Module):
    """
    Encoder-decoder RNN for ship trajectory prediction.

    The encoder reads the full history and produces a hidden state.
    The decoder unrolls autoregressively: at each step it receives its own
    previous delta prediction and passes the updated hidden state forward.
    This means the model can generate trajectories of any length at inference
    time without retraining, unlike the flat-head variant.

    During training, teacher forcing randomly replaces the decoder's input
    with the ground-truth delta to stabilise early learning.

    Input shape:
        x: (batch_size, history_steps, input_dim)

    Output shape:
        y_hat_delta: (batch_size, future_steps, 2)
    """

    def __init__(
        self,
        input_dim: int,
        future_steps: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ):
        super().__init__()

        self.future_steps = future_steps
        self.rnn_type = rnn_type.lower()

        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = _build_rnn(self.rnn_type, input_dim, hidden_dim, num_layers, rnn_dropout)
        # Decoder input: the (Δlat, Δlon) predicted at the previous step.
        self.decoder = _build_rnn(self.rnn_type, 2, hidden_dim, num_layers, rnn_dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        _, enc_hidden = self.encoder(x)

        batch_size = x.size(0)
        dec_input = x.new_zeros(batch_size, 1, 2)
        hidden = enc_hidden

        outputs: list[torch.Tensor] = []
        for t in range(self.future_steps):
            dec_out, hidden = self.decoder(dec_input, hidden)
            delta = self.output_proj(dec_out[:, 0, :])  # (batch, 2)
            outputs.append(delta.unsqueeze(1))

            use_teacher = (
                target is not None
                and teacher_forcing_ratio > 0.0
                and torch.rand(1, device=x.device).item() < teacher_forcing_ratio
            )
            dec_input = target[:, t : t + 1, :] if use_teacher else delta.unsqueeze(1).detach()

        return torch.cat(outputs, dim=1)  # (batch, future_steps, 2)


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ShipTrajectoryARRNN,
    dataloader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float,
    *,
    residual_naive: bool = False,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, batch_naive, batch_weight = unpack_window_batch(
            batch, device
        )
        tf_target = batch_y_delta
        if residual_naive and batch_naive is not None:
            tf_target = batch_y_delta - batch_naive

        y_hat_delta = model(
            batch_x,
            target=tf_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        y_hat_delta = apply_residual_prediction(
            y_hat_delta, batch_naive, residual=residual_naive
        )
        if isinstance(criterion, TrajectoryLoss):
            loss = criterion(y_hat_delta, batch_y_delta, batch_anchor, batch_weight)
        else:
            loss = criterion(y_hat_delta, batch_y_delta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(
    model: ShipTrajectoryARRNN,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    *,
    residual_naive: bool = False,
) -> float:
    model.eval()

    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, batch_naive, batch_weight = unpack_window_batch(
            batch, device
        )

        y_hat_delta = model(batch_x)
        y_hat_delta = apply_residual_prediction(
            y_hat_delta, batch_naive, residual=residual_naive
        )
        if isinstance(criterion, TrajectoryLoss):
            loss = criterion(y_hat_delta, batch_y_delta, batch_anchor, batch_weight)
        else:
            loss = criterion(y_hat_delta, batch_y_delta)

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_absolute_positions(
    model: ShipTrajectoryARRNN,
    dataloader: DataLoader,
    device: torch.device,
    *,
    residual_naive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true_list = []
    y_pred_list = []

    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, batch_naive, _ = unpack_window_batch(
            batch, device
        )

        y_hat_delta = model(batch_x)
        y_hat_delta = apply_residual_prediction(
            y_hat_delta, batch_naive, residual=residual_naive
        )

        y_hat_delta = y_hat_delta.cpu().numpy()
        y_true_delta = batch_y_delta.cpu().numpy()
        anchor = batch_anchor.cpu().numpy()[:, np.newaxis, :]

        y_true_abs = anchor + y_true_delta
        y_pred_abs = anchor + y_hat_delta

        y_true_list.append(y_true_abs)
        y_pred_list.append(y_pred_abs)

    return np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------

def save_training_plot(history: list[dict[str, float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("AR-RNN Training History")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_histogram(
    y_true_final: np.ndarray,
    y_pred_final: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    errors_km = haversine_km(
        y_true_final[:, 0],
        y_true_final[:, 1],
        y_pred_final[:, 0],
        y_pred_final[:, 1],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_km, bins=60)
    ax.axvline(errors_km.mean(), linestyle="--", label=f"mean = {errors_km.mean():.3f} km")
    ax.axvline(np.median(errors_km), linestyle="--", label=f"median = {np.median(errors_km):.3f} km")
    ax.set_xlabel("Prediction error (kilometers)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_scatter_plot(
    y_true_final: np.ndarray,
    y_pred_final: np.ndarray,
    output_path: Path,
    title: str,
    max_points: int = 5000,
) -> None:
    import matplotlib.pyplot as plt

    if len(y_true_final) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_true_final), size=max_points, replace=False)
        y_true_final = y_true_final[idx]
        y_pred_final = y_pred_final[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, dim, name in zip(axes, [0, 1], ["Latitude", "Longitude"]):
        ax.scatter(y_true_final[:, dim], y_pred_final[:, dim], s=6, alpha=0.25)

        low = min(y_true_final[:, dim].min(), y_pred_final[:, dim].min())
        high = max(y_true_final[:, dim].max(), y_pred_final[:, dim].max())

        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5, label="perfect")
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_rnn_ar(
    input_path: Path,
    coast_name: str | None,
    region: str,
    rnn_type: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    sample_size: int | None,
    horizon_hours: float,
    teacher_forcing_ratio: float,
    motion_filter=None,
    training_config: TrainingImprovementConfig | None = None,
) -> Path:
    training_config = training_config or TrainingImprovementConfig()
    start_time = time.perf_counter()

    input_path, coast, region = resolve_windows_path(coast_name, region, input_path)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Load processed windows
    # -----------------------------------------------------------------------

    print(f"Loading windows from: {input_path}")
    df = load_windows_filtered(
        input_path,
        sample_size=sample_size,
        motion_filter=motion_filter,
        maneuver_oversample=training_config.maneuver_oversample,
        maneuver_fraction=training_config.maneuver_fraction,
        motion_balanced_sample=training_config.motion_balanced_sample,
        straight_fraction=training_config.straight_fraction,
        other_fraction=training_config.other_fraction,
        seed=seed,
    )

    tag = f"RNN_AR_{rnn_type.upper()}"
    output_dir = results_output_dir(coast, input_path, tag, df)
    model_dir = models_output_dir(coast, input_path, df)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    history_steps, future_steps = infer_window_shape(df)
    window_hours = window_horizon_hours(df, future_steps)
    horizon_step = horizon_step_index(df, horizon_hours, future_steps)
    actual_horizon_hours = (
        (horizon_step + 1) * int(df["resample_minutes"].iloc[0]) / 60
        if "resample_minutes" in df.columns
        else horizon_hours
    )

    x, y_abs, y_delta, anchor = build_window_arrays(
        df,
        history_steps=history_steps,
        future_steps=future_steps,
    )
    naive_delta = compute_naive_cumulative_delta(x, future_steps)
    naive_for_dataset = naive_delta if training_config.residual_naive else None

    if "traj_id" in df.columns:
        num_trajectories = df["traj_id"].nunique()
    else:
        num_trajectories = df["mmsi"].nunique()

    print(f"\n=== {coast.name} | {output_dir.parent.name} ===")
    print(f"Samples: {len(df):,}")
    print(f"Trajectories: {num_trajectories:,}")
    print(
        f"Window: history={history_steps} steps | "
        f"future={future_steps} steps | "
        f"window={window_hours:.1f} h | "
        f"eval horizon={actual_horizon_hours:.1f} h (step {horizon_step + 1}/{future_steps})"
    )
    print(f"Features: {FEATURE_COLS}")

    # -----------------------------------------------------------------------
    # Split by trajectory
    # -----------------------------------------------------------------------

    train_ids, val_ids, test_ids = trajectory_splits(
        df,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_mask = mask_by_split(df, train_ids)
    val_mask = mask_by_split(df, val_ids)
    test_mask = mask_by_split(df, test_ids)

    x_train = x[train_mask]
    x_val = x[val_mask]
    x_test = x[test_mask]

    y_delta_train = y_delta[train_mask]
    y_delta_val = y_delta[val_mask]
    y_delta_test = y_delta[test_mask]

    anchor_train = anchor[train_mask]
    anchor_val = anchor[val_mask]
    anchor_test = anchor[test_mask]

    y_test_abs = y_abs[test_mask]

    x_test_raw = x_test.copy()

    train_weights = (
        compute_sample_weights(x[train_mask])
        if training_config.difficulty_weighting
        else None
    )

    naive_train = naive_delta[train_mask] if naive_for_dataset is not None else None
    naive_val = naive_delta[val_mask] if naive_for_dataset is not None else None
    naive_test = naive_delta[test_mask] if naive_for_dataset is not None else None

    x_train, [x_val, x_test], scaler = scale_history_features(
        x_train,
        [x_val, x_test],
    )

    train_dataset = WindowDataset(
        x_train, y_delta_train, anchor_train, train_weights, naive_train
    )
    val_dataset = WindowDataset(x_val, y_delta_val, anchor_val, None, naive_val)
    test_dataset = WindowDataset(x_test, y_delta_test, anchor_test, None, naive_test)

    if len(train_dataset) == 0:
        raise ValueError("No training windows found. Run preprocessing first or lower split fractions.")

    print(
        f"Windows — train: {len(train_dataset):,} | "
        f"val: {len(val_dataset):,} | "
        f"test: {len(test_dataset):,}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------------------------
    # Model, loss, optimizer
    # -----------------------------------------------------------------------

    model = ShipTrajectoryARRNN(
        input_dim=len(FEATURE_COLS),
        future_steps=future_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
    ).to(device)

    criterion = TrajectoryLoss(haversine_weight=training_config.haversine_weight)
    resample_minutes = (
        int(df["resample_minutes"].iloc[0]) if "resample_minutes" in df.columns else 10
    )
    base_tf = teacher_forcing_ratio
    print(
        f"Loss: Huber + Haversine (w={training_config.haversine_weight:.2f}) | "
        f"TF start={base_tf:.2f} scheduled={'on' if training_config.scheduled_teacher_forcing else 'off'} | "
        f"residual naive={'on' if training_config.residual_naive else 'off'} | "
        f"motion sample={'balanced' if training_config.motion_balanced_sample else 'uniform'}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    param_count = sum(p.numel() for p in model.parameters())

    print("\nModel:")
    print(model)
    print(f"Trainable parameters: {param_count:,}")
    print(f"Teacher forcing ratio: {teacher_forcing_ratio}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("\nTraining...")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    total_train_time = 0.0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        if training_config.scheduled_teacher_forcing:
            epoch_tf = scheduled_teacher_forcing(
                epoch,
                epochs,
                start=base_tf,
                end=training_config.teacher_forcing_end,
            )
        else:
            epoch_tf = teacher_forcing_ratio

        if training_config.curriculum:
            train_steps = curriculum_train_steps(
                epoch,
                epochs,
                future_steps,
                resample_minutes=resample_minutes,
                start_hours=training_config.curriculum_start_hours,
            )
            criterion.set_train_steps(train_steps)
        else:
            criterion.set_train_steps(None)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=epoch_tf,
            residual_naive=training_config.residual_naive,
        )

        train_steps_cur = criterion.train_steps
        criterion.set_train_steps(None)
        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            residual_naive=training_config.residual_naive,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.perf_counter() - epoch_start
        total_train_time += epoch_elapsed

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": current_lr,
                "epoch_sec": float(epoch_elapsed),
            }
        )

        print(
            f"| epoch {epoch:3d} | "
            f"time: {epoch_elapsed:6.2f}s | "
            f"train loss {train_loss:10.6f} | "
            f"valid loss {val_loss:10.6f} | "
            f"lr {current_lr:.2e} | tf {epoch_tf:.2f}"
            + (
                f" | train steps {train_steps_cur}/{future_steps}"
                if training_config.curriculum and train_steps_cur
                else ""
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    print("-" * 70)

    criterion.set_train_steps(None)

    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    avg_throughput = len(train_dataset) * len(history) / max(total_train_time, 1e-6)
    print(f"Peak GPU memory: {peak_gpu_mb:.1f} MB")
    print(f"Avg throughput:  {avg_throughput:.1f} samples/sec")

    if best_state is not None:
        model.load_state_dict(best_state)

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------

    y_true_traj, y_pred_traj = predict_absolute_positions(
        model=model,
        dataloader=test_loader,
        device=device,
        residual_naive=training_config.residual_naive,
    )

    # Save sample trajectories for map visualisation in the comparison script.
    n_map = min(200, len(y_true_traj))
    rng_map = np.random.default_rng(seed)
    map_idx = rng_map.choice(len(y_true_traj), size=n_map, replace=False)
    traj_path = output_dir / f"{rnn_type}_ar_sample_trajectories.json"
    with traj_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "y_true": y_true_traj[map_idx].tolist(),
                "y_pred": y_pred_traj[map_idx].tolist(),
                "anchor": anchor_test[map_idx].tolist(),
            },
            f,
        )

    eval_true = y_true_traj[:, horizon_step, :]
    eval_pred = y_pred_traj[:, horizon_step, :]

    baseline_true, baseline_pred = naive_position_at_horizon(
        x_test_raw,
        y_test_abs,
        horizon_step,
    )

    model_name = f"{rnn_type.upper()}-AR trajectory model"

    metrics = [
        evaluate_final_position(
            baseline_true,
            baseline_pred,
            "Naive baseline: constant last-step delta",
        ),
        evaluate_final_position(
            eval_true,
            eval_pred,
            f"{model_name}: position ({actual_horizon_hours:.1f} h ahead)",
        ),
        evaluate_full_trajectory(
            y_true_traj,
            y_pred_traj,
            f"{model_name}: full predicted trajectory",
        ),
    ]
    metrics.extend(
        evaluate_stratified_positions(
            eval_true,
            eval_pred,
            x_test_raw,
            f"{model_name}: position ({actual_horizon_hours:.1f} h ahead)",
        )
    )

    for item in metrics:
        print_position_metrics(item)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    model_path = model_dir / f"ship_trajectory_{rnn_type}_ar.pt"
    metrics_path = output_dir / f"{rnn_type}_ar_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "feature_cols": FEATURE_COLS,
            "history_steps": history_steps,
            "future_steps": future_steps,
            "window_hours": window_hours,
            "horizon_hours_requested": horizon_hours,
            "horizon_hours_actual": actual_horizon_hours,
            "horizon_step_index": horizon_step,
            "rnn_type": rnn_type,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "model_variant": "autoregressive",
        },
        model_path,
    )

    results = {
        "input": str(input_path),
        "coast": coast.name,
        "days_label": output_dir.parent.name,
        "region": region,
        "samples_total": int(len(df)),
        "samples_train": int(train_mask.sum()),
        "samples_val": int(val_mask.sum()),
        "samples_test": int(test_mask.sum()),
        "features": FEATURE_COLS,
        "history_steps": int(history_steps),
        "future_steps": int(future_steps),
        "window_hours": float(window_hours),
        "horizon_hours_requested": float(horizon_hours),
        "horizon_hours_actual": float(actual_horizon_hours),
        "horizon_step_index": int(horizon_step),
        "architecture": {
            "type": rnn_type,
            "variant": "autoregressive",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        },
        "training": {
            "batch_size": batch_size,
            "epochs_ran": len(history),
            "learning_rate": learning_rate,
            "patience": patience,
            "best_val_loss": float(best_val_loss),
            "device": str(device),
            "improvements": training_improvements_dict(training_config),
        },
        "splits": {
            "test_fraction": test_fraction,
            "val_fraction": val_fraction,
            "seed": seed,
            "split_by": "trajectory",
        },
        "compute": {
            "param_count": int(param_count),
            "peak_gpu_mb": float(peak_gpu_mb),
            "avg_throughput_samples_per_sec": float(avg_throughput),
            "total_train_sec": float(total_train_time),
        },
        "metrics": metrics,
        "history": history,
        "runtime_sec": round(time.perf_counter() - start_time, 2),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_training_plot(
        history,
        output_dir / f"{rnn_type}_ar_training_history.png",
    )

    save_error_histogram(
        eval_true,
        eval_pred,
        output_dir / f"{rnn_type}_ar_error_hist.png",
        f"{model_name} — Position Error ({actual_horizon_hours:.1f} h ahead)",
    )

    save_scatter_plot(
        eval_true,
        eval_pred,
        output_dir / f"{rnn_type}_ar_scatter.png",
        f"{model_name} — Position ({actual_horizon_hours:.1f} h ahead)",
    )

    print("\nSaved files:")
    print(f"  model:   {model_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  trajs:   {traj_path}")
    print(f"  plots:   {output_dir / f'{rnn_type}_ar_training_history.png'}")
    print(f"           {output_dir / f'{rnn_type}_ar_error_hist.png'}")
    print(f"           {output_dir / f'{rnn_type}_ar_scatter.png'}")
    print(f"Runtime: {format_duration(time.perf_counter() - start_time)}")

    return metrics_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an autoregressive encoder-decoder RNN/LSTM/GRU for ship trajectory prediction."
    )

    parser.add_argument(
        "--coast",
        choices=sorted(COAST_CONFIGS.keys()),
        default=None,
        help="Coastal area (default: Eastern coast, or inferred from --input).",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to model_ready_windows.parquet.",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region label used by the preprocessing script, if --input is not given.",
    )

    parser.add_argument(
        "--rnn-type",
        type=str,
        default="lstm",
        choices=["rnn", "gru", "lstm"],
        help="Recurrent architecture to use.",
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Use a random sample for faster runs. 0 means use all windows.",
    )

    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=1.0,
        help="How far ahead to evaluate (must fit inside the future window in the parquet).",
    )

    parser.add_argument(
        "--teacher-forcing",
        type=float,
        default=0.5,
        help="Probability of using ground-truth delta as decoder input during training (0=free-running, 1=always teacher).",
    )

    add_stationary_filter_args(parser)
    add_training_improvement_args(parser)

    args = parser.parse_args()

    if args.region is None:
        if args.coast is not None:
            region = COAST_CONFIGS[args.coast].default_region
        else:
            region = COAST_CONFIGS["Eastern coast"].default_region
    else:
        region = args.region

    run_rnn_ar(
        input_path=args.input,
        coast_name=args.coast,
        region=region,
        rnn_type=args.rnn_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        sample_size=None if args.sample <= 0 else args.sample,
        horizon_hours=args.horizon_hours,
        teacher_forcing_ratio=args.teacher_forcing,
        motion_filter=stationary_filter_from_args(args),
        training_config=training_config_from_args(args),
    )


if __name__ == "__main__":
    main()
