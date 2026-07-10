"""Adaptive multi-scale autoregressive RNN — learns soft weights over 9/12/18/24h contexts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from proj.project.coast_paths import COAST_CONFIGS, models_output_dir, resolve_windows_path, results_output_dir
from proj.project.models.plot_utils import save_training_history_plot
from proj.project.models.RNN_AR import (
    WindowDataset,
    format_duration,
    save_error_histogram,
    save_scatter_plot,
)
from proj.project.models.vessel_type_utils import (
    NUM_VESSEL_CLASSES,
    VESSEL_CLASS_NAMES,
    resolve_vessel_type_lookup,
    vessel_class_indices_for_mmsi,
)
from proj.project.models.training_utils import (
    TrajectoryLoss,
    TrainingImprovementConfig,
    add_training_improvement_args,
    curriculum_train_steps,
    enrich_history_row,
    scheduled_teacher_forcing,
    training_config_from_args,
    training_improvements_dict,
    unpack_window_batch,
)
from proj.project.window_data import (
    FEATURE_COLS,
    build_window_arrays,
    compute_sample_weights,
    deltas_to_absolute,
    evaluate_final_position,
    evaluate_full_trajectory,
    evaluate_stratified_positions,
    horizon_step_index,
    hours_to_window_steps,
    kinematic_position_at_horizon,
    load_windows_filtered,
    add_stationary_filter_args,
    make_train_val_test_frames,
    naive_position_at_horizon,
    print_position_metrics,
    resolve_window_hours,
    scale_history_features,
    stationary_filter_from_args,
)

DEFAULT_CONTEXT_HOURS = (9.0, 12.0, 18.0, 24.0)


def _build_rnn(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    rnn_dropout = dropout if num_layers > 1 else 0.0
    if rnn_type == "rnn":
        return nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=rnn_dropout,
        )
    if rnn_type == "gru":
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
    if rnn_type == "lstm":
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
    raise ValueError("rnn_type must be one of: 'rnn', 'gru', 'lstm'")


def _last_layer_hidden(hidden, rnn_type: str) -> torch.Tensor:
    if rnn_type == "lstm":
        return hidden[0][-1]
    return hidden[-1]


def _hidden_from_vector(vec: torch.Tensor, num_layers: int, rnn_type: str):
    h0 = vec.unsqueeze(0).repeat(num_layers, 1, 1)
    if rnn_type == "lstm":
        return (h0, torch.zeros_like(h0))
    return h0


class AdaptiveWindowDataset(Dataset):
    """Window batch with optional per-sample vessel class for gate conditioning."""

    def __init__(
        self,
        x: np.ndarray,
        y_delta: np.ndarray,
        anchor: np.ndarray,
        vessel_class: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_delta = torch.tensor(y_delta, dtype=torch.float32)
        self.anchor = torch.tensor(anchor, dtype=torch.float32)
        self.vessel_class = (
            torch.tensor(vessel_class, dtype=torch.long) if vessel_class is not None else None
        )
        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        items = [self.x[idx], self.y_delta[idx], self.anchor[idx]]
        if self.vessel_class is not None:
            items.append(self.vessel_class[idx])
        if self.sample_weights is not None:
            items.append(self.sample_weights[idx])
        return tuple(items)


def unpack_adaptive_batch(
    batch: tuple,
    device: torch.device,
    *,
    gate_vessel_type: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if gate_vessel_type:
        if len(batch) == 5:
            batch_x, batch_y_delta, batch_anchor, vessel_class, batch_weight = batch
            vessel_class = vessel_class.to(device).long()
            batch_weight = batch_weight.to(device)
        elif len(batch) == 4:
            batch_x, batch_y_delta, batch_anchor, fourth = batch
            if fourth.dtype == torch.long:
                vessel_class = fourth.to(device)
                batch_weight = None
            else:
                vessel_class = None
                batch_weight = fourth.to(device)
        else:
            batch_x, batch_y_delta, batch_anchor = batch
            vessel_class = None
            batch_weight = None
        return (
            batch_x.to(device),
            batch_y_delta.to(device),
            batch_anchor.to(device),
            vessel_class,
            batch_weight,
        )

    batch_x, batch_y_delta, batch_anchor, batch_naive, batch_weight = unpack_window_batch(batch, device)
    return batch_x, batch_y_delta, batch_anchor, None, batch_weight


def train_one_epoch_adaptive(
    model: "AdaptiveMultiScaleARRNN",
    dataloader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float,
    *,
    gate_vessel_type: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, vessel_class, batch_weight = unpack_adaptive_batch(
            batch, device, gate_vessel_type=gate_vessel_type
        )
        y_hat_delta = model(
            batch_x,
            target=batch_y_delta,
            teacher_forcing_ratio=teacher_forcing_ratio,
            vessel_class=vessel_class,
        )
        if isinstance(criterion, TrajectoryLoss):
            loss = criterion(y_hat_delta, batch_y_delta, batch_anchor, batch_weight)
        else:
            loss = criterion(y_hat_delta, batch_y_delta)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss_adaptive(
    model: "AdaptiveMultiScaleARRNN",
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    *,
    gate_vessel_type: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, vessel_class, batch_weight = unpack_adaptive_batch(
            batch, device, gate_vessel_type=gate_vessel_type
        )
        y_hat_delta = model(batch_x, vessel_class=vessel_class)
        if isinstance(criterion, TrajectoryLoss):
            loss = criterion(y_hat_delta, batch_y_delta, batch_anchor, batch_weight)
        else:
            loss = criterion(y_hat_delta, batch_y_delta)
        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


class AdaptiveMultiScaleARRNN(nn.Module):
    """Encode 9/12/18/24h history suffixes; gate context; AR-decode 12h future."""

    def __init__(
        self,
        input_dim: int,
        future_steps: int,
        context_steps: list[int],
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
        *,
        gate_vessel_type: bool = False,
        num_vessel_classes: int = NUM_VESSEL_CLASSES,
        vessel_embed_dim: int = 16,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.context_steps = list(context_steps)
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gate_vessel_type = gate_vessel_type
        self.vessel_embed_dim = vessel_embed_dim if gate_vessel_type else 0

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = _build_rnn(self.rnn_type, input_dim, hidden_dim, num_layers, rnn_dropout)
        self.decoder = _build_rnn(self.rnn_type, 2, hidden_dim, num_layers, rnn_dropout)
        if gate_vessel_type:
            self.vessel_embed = nn.Embedding(num_vessel_classes, vessel_embed_dim)
            gate_in_dim = hidden_dim * len(self.context_steps) + vessel_embed_dim
        else:
            self.vessel_embed = None
            gate_in_dim = hidden_dim * len(self.context_steps)
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.context_steps)),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode_context(
        self,
        x: torch.Tensor,
        vessel_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused context vector (B,H) and alpha weights (B,K)."""
        h_vecs = []
        for n_steps in self.context_steps:
            x_slice = x[:, -n_steps:, :]
            _, enc_hidden = self.encoder(x_slice)
            h_vecs.append(_last_layer_hidden(enc_hidden, self.rnn_type))
        h_stack = torch.stack(h_vecs, dim=1)
        gate_in = h_stack.reshape(x.size(0), -1)
        if self.gate_vessel_type:
            if vessel_class is None:
                vessel_class = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            gate_in = torch.cat([gate_in, self.vessel_embed(vessel_class)], dim=-1)
        alpha = torch.softmax(self.gate(gate_in), dim=-1)
        h_context = (alpha.unsqueeze(-1) * h_stack).sum(dim=1)
        return h_context, alpha

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        return_alpha: bool = False,
        vessel_class: torch.Tensor | None = None,
    ):
        h_context, alpha = self.encode_context(x, vessel_class=vessel_class)
        hidden = _hidden_from_vector(h_context, self.num_layers, self.rnn_type)

        batch_size = x.size(0)
        dec_input = x.new_zeros(batch_size, 1, 2)
        outputs: list[torch.Tensor] = []
        for t in range(self.future_steps):
            dec_out, hidden = self.decoder(dec_input, hidden)
            delta = self.output_proj(dec_out[:, 0, :])
            outputs.append(delta.unsqueeze(1))
            use_teacher = (
                target is not None
                and teacher_forcing_ratio > 0.0
                and torch.rand(1, device=x.device).item() < teacher_forcing_ratio
            )
            dec_input = target[:, t : t + 1, :] if use_teacher else delta.unsqueeze(1).detach()

        y_hat = torch.cat(outputs, dim=1)
        if return_alpha:
            return y_hat, alpha
        return y_hat


@torch.no_grad()
def predict_with_alphas(
    model: AdaptiveMultiScaleARRNN,
    dataloader: DataLoader,
    device: torch.device,
    *,
    gate_vessel_type: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_pred_list, alpha_list = [], [], []
    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, vessel_class, _ = unpack_adaptive_batch(
            batch, device, gate_vessel_type=gate_vessel_type
        )
        y_hat_delta, alpha = model(batch_x, return_alpha=True, vessel_class=vessel_class)
        y_true_abs = deltas_to_absolute(
            batch_y_delta.cpu().numpy(),
            batch_anchor.cpu().numpy(),
            target_mode="anchor_offset",
        )
        y_pred_abs = deltas_to_absolute(
            y_hat_delta.cpu().numpy(),
            batch_anchor.cpu().numpy(),
            target_mode="anchor_offset",
        )
        y_true_list.append(y_true_abs)
        y_pred_list.append(y_pred_abs)
        alpha_list.append(alpha.cpu().numpy())
    return (
        np.concatenate(y_true_list, axis=0),
        np.concatenate(y_pred_list, axis=0),
        np.concatenate(alpha_list, axis=0),
    )


def run_adaptive_ar(
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
    future_hours: float = 12.0,
    context_hours: tuple[float, ...] = DEFAULT_CONTEXT_HOURS,
    teacher_forcing_ratio: float = 0.3,
    motion_filter=None,
    training_config: TrainingImprovementConfig | None = None,
    run_tag: str | None = None,
    gate_vessel_type: bool = False,
    vessel_lookup_path: Path | None = None,
) -> Path:
    training_config = training_config or TrainingImprovementConfig()
    start_time = time.perf_counter()

    input_path, coast, region = resolve_windows_path(coast_name, region, input_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

    tag = "RNN_AR_adaptive"
    output_dir = results_output_dir(coast, input_path, tag, df, run_tag=run_tag)
    model_dir = models_output_dir(coast, input_path, df, run_tag=run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    full_history_steps, _, history_steps, future_steps, step_minutes = resolve_window_hours(
        df, history_hours=24.0, future_hours=future_hours
    )
    context_steps = [hours_to_window_steps(h, step_minutes) for h in context_hours]
    horizon_step = horizon_step_index(df, horizon_hours, future_steps)
    actual_horizon_hours = (horizon_step + 1) * step_minutes / 60

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

    def _arrays(split_df):
        return build_window_arrays(
            split_df,
            history_steps=history_steps,
            future_steps=future_steps,
            full_history_steps=full_history_steps,
        )

    x_train, _, y_delta_train, anchor_train = _arrays(train_df)
    x_val, _, y_delta_val, anchor_val = _arrays(val_df)
    x_test, y_test_abs, y_delta_test, anchor_test = _arrays(test_df)
    x_test_raw = x_test.copy()

    train_weights = (
        compute_sample_weights(x_train) if training_config.difficulty_weighting else None
    )
    x_train, [x_val, x_test], scaler = scale_history_features(x_train, [x_val, x_test])

    vessel_lookup = None
    vc_train = vc_val = vc_test = None
    if gate_vessel_type:
        vessel_lookup = resolve_vessel_type_lookup(
            project_root=PACKAGE_ROOT,
            lookup_path=vessel_lookup_path,
        )
        vc_train = vessel_class_indices_for_mmsi(train_df["mmsi"].to_numpy(), vessel_lookup)
        vc_val = vessel_class_indices_for_mmsi(val_df["mmsi"].to_numpy(), vessel_lookup)
        vc_test = vessel_class_indices_for_mmsi(test_df["mmsi"].to_numpy(), vessel_lookup)
        known = (vc_train > 0).mean()
        print(f"Vessel class known for {known * 100:.1f}% of train windows (coarse AIS buckets)")

    def _make_ds(x, y, anchor, vc, weights=None):
        if gate_vessel_type:
            return AdaptiveWindowDataset(x, y, anchor, vc, weights)
        return WindowDataset(x, y, anchor, weights)

    train_ds = _make_ds(x_train, y_delta_train, anchor_train, vc_train, train_weights)
    val_ds = _make_ds(x_val, y_delta_val, anchor_val, vc_val)
    test_ds = _make_ds(x_test, y_delta_test, anchor_test, vc_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    train_eval_size = min(len(train_ds), max(len(val_ds), 10_000))
    train_eval_idx = np.random.default_rng(seed).choice(len(train_ds), size=train_eval_size, replace=False)
    train_eval_loader = DataLoader(
        Subset(_make_ds(x_train, y_delta_train, anchor_train, vc_train), train_eval_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = AdaptiveMultiScaleARRNN(
        input_dim=len(FEATURE_COLS),
        future_steps=future_steps,
        context_steps=context_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
        gate_vessel_type=gate_vessel_type,
    ).to(device)

    criterion = TrajectoryLoss(haversine_weight=training_config.haversine_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    resample_minutes = int(step_minutes)

    print(
        f"\n=== {coast.name} | adaptive multi-scale AR | contexts={list(context_hours)}h -> {future_hours}h future ==="
    )
    if gate_vessel_type:
        print("Gate conditioning: vessel type embedding enabled")
    print(f"Windows — train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}")

    best_val = float("inf")
    best_state = None
    stale = 0
    history = []
    total_train_time = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_tf = (
            scheduled_teacher_forcing(epoch, epochs, start=teacher_forcing_ratio, end=0.0)
            if training_config.scheduled_teacher_forcing
            else teacher_forcing_ratio
        )
        if training_config.curriculum:
            train_steps = curriculum_train_steps(
                epoch, epochs, future_steps, resample_minutes=resample_minutes
            )
            criterion.set_train_steps(train_steps)
        else:
            criterion.set_train_steps(None)

        train_loss = train_one_epoch_adaptive(
            model, train_loader, criterion, optimizer, device,
            teacher_forcing_ratio=epoch_tf,
            gate_vessel_type=gate_vessel_type,
        )
        train_steps_cur = criterion.train_steps
        criterion.set_train_steps(None)
        val_loss = evaluate_loss_adaptive(
            model, val_loader, criterion, device, gate_vessel_type=gate_vessel_type
        )
        train_eval_loss = evaluate_loss_adaptive(
            model, train_eval_loader, criterion, device, gate_vessel_type=gate_vessel_type
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        total_train_time += elapsed
        history.append(
            enrich_history_row(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr, epoch_sec=elapsed,
                train_steps=train_steps_cur if training_config.curriculum else future_steps,
                future_steps=future_steps, teacher_forcing=epoch_tf, train_eval_loss=train_eval_loss,
            )
        )
        print(
            f"| epoch {epoch:3d} | time {elapsed:6.2f}s | train {train_loss:8.5f} | "
            f"val {val_loss:8.5f} | train_eval {train_eval_loss:8.5f} | tf {epoch_tf:.2f}"
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
    test_loss_at_best = evaluate_loss_adaptive(
        model, test_loader, criterion, device, gate_vessel_type=gate_vessel_type
    )
    print(f"Test loss @ best epoch {best_epoch}: {test_loss_at_best:.6f}")

    y_true_traj, y_pred_traj, alpha_test = predict_with_alphas(
        model, test_loader, device, gate_vessel_type=gate_vessel_type
    )
    eval_true = y_true_traj[:, horizon_step, :]
    eval_pred = y_pred_traj[:, horizon_step, :]

    kin_true, kin_pred = kinematic_position_at_horizon(
        x_test_raw, y_test_abs, horizon_step, step_minutes=step_minutes
    )
    metrics = [
        evaluate_final_position(kin_true, kin_pred, "Kinematic baseline", anchor=anchor_test),
        evaluate_final_position(
            eval_true, eval_pred,
            f"Adaptive AR ({actual_horizon_hours:.1f}h ahead)",
            anchor=anchor_test,
        ),
        evaluate_full_trajectory(y_true_traj, y_pred_traj, "Adaptive AR: full trajectory"),
    ]
    metrics.extend(
        evaluate_stratified_positions(
            y_true_traj[:, horizon_step, :],
            y_pred_traj[:, horizon_step, :],
            x_test_raw,
            "Adaptive AR",
            anchor=anchor_test,
        )
    )
    for item in metrics:
        print_position_metrics(item)

    alpha_labels = [f"alpha_{int(h)}h" for h in context_hours]
    alpha_path = output_dir / "context_alpha_weights.json"
    alpha_payload: dict = {
        "context_hours": list(context_hours),
        "labels": alpha_labels,
        "alpha_mean": {lbl: float(alpha_test[:, i].mean()) for i, lbl in enumerate(alpha_labels)},
        "alpha_median": {lbl: float(np.median(alpha_test[:, i])) for i, lbl in enumerate(alpha_labels)},
        "per_sample": alpha_test.tolist(),
    }
    if gate_vessel_type and vc_test is not None:
        alpha_by_class: dict[str, dict[str, float]] = {}
        for cls_id, cls_name in VESSEL_CLASS_NAMES.items():
            mask = vc_test == cls_id
            if not mask.any():
                continue
            alpha_by_class[cls_name] = {
                lbl: float(alpha_test[mask, i].mean()) for i, lbl in enumerate(alpha_labels)
            }
        alpha_payload["vessel_class_names"] = VESSEL_CLASS_NAMES
        alpha_payload["per_sample_vessel_class"] = vc_test.tolist()
        alpha_payload["alpha_mean_by_vessel_class"] = alpha_by_class
    with alpha_path.open("w", encoding="utf-8") as f:
        json.dump(alpha_payload, f, indent=2)

    metrics_path = output_dir / "adaptive_ar_metrics.json"
    results = {
        "input": str(input_path),
        "coast": coast.name,
        "context_hours": list(context_hours),
        "future_hours": float(future_hours),
        "history_steps": int(history_steps),
        "future_steps": int(future_steps),
        "horizon_hours_actual": float(actual_horizon_hours),
        "gate_vessel_type": gate_vessel_type,
        "vessel_class_names": VESSEL_CLASS_NAMES if gate_vessel_type else None,
        "training": training_improvements_dict(training_config),
        "metrics": metrics,
        "history": history,
        "eval_summary": {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "test_loss_at_best": float(test_loss_at_best),
        },
        "alpha_summary": {lbl: float(alpha_test[:, i].mean()) for i, lbl in enumerate(alpha_labels)},
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_training_history_plot(
        history,
        output_dir / "adaptive_ar_training_history.png",
        title="Adaptive multi-scale AR — training",
        loss_label="Trajectory loss",
        autoregressive=True,
        curriculum_enabled=training_config.curriculum,
        test_loss_at_best=test_loss_at_best,
        test_epoch=best_epoch,
    )
    save_error_histogram(
        eval_true, eval_pred,
        output_dir / "adaptive_ar_error_hist.png",
        title=f"Adaptive AR error @ {actual_horizon_hours:.1f}h",
    )
    save_scatter_plot(
        eval_true, eval_pred,
        output_dir / "adaptive_ar_scatter.png",
        title=f"Adaptive AR @ {actual_horizon_hours:.1f}h",
    )

    model_ckpt_name = "adaptive_ar_vessel_type.pt" if gate_vessel_type else "adaptive_ar.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "context_hours": list(context_hours),
            "context_steps": context_steps,
            "future_steps": future_steps,
            "gate_vessel_type": gate_vessel_type,
        },
        model_dir / model_ckpt_name,
    )

    print(f"\nSaved: {metrics_path}")
    print(f"Alpha weights: {alpha_path}")
    print(f"Runtime: {format_duration(time.perf_counter() - start_time)}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive multi-scale autoregressive RNN.")
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
    parser.add_argument("--future-hours", type=float, default=12.0)
    parser.add_argument("--teacher-forcing", type=float, default=0.3)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument(
        "--gate-vessel-type",
        action="store_true",
        help="Condition context gate on coarse AIS vessel type (ferry/cargo/tanker/...).",
    )
    parser.add_argument(
        "--vessel-lookup",
        type=Path,
        default=None,
        help="Optional MMSI->vessel_type lookup parquet (built automatically if missing).",
    )
    add_stationary_filter_args(parser)
    add_training_improvement_args(parser)
    args = parser.parse_args()

    region = args.region or (
        COAST_CONFIGS[args.coast].default_region if args.coast else COAST_CONFIGS["Eastern coast"].default_region
    )
    run_adaptive_ar(
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
        future_hours=args.future_hours,
        teacher_forcing_ratio=args.teacher_forcing,
        motion_filter=stationary_filter_from_args(args),
        training_config=training_config_from_args(args),
        run_tag=args.run_tag,
        gate_vessel_type=args.gate_vessel_type,
        vessel_lookup_path=args.vessel_lookup,
    )


if __name__ == "__main__":
    main()
