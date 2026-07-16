"""Adaptive AR LSTM with independent encoders per context length (9/12/18/24h).

Two gate modes on the same architecture:
  - softmax: soft mixture over contexts
  - hard: Gumbel-Softmax (train) / argmax one-hot (eval)

Experiments:
  exp_coastal/adaptive_separate_encoders_softmax
  exp_coastal/adaptive_separate_encoders_hard
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from proj.project.coast_paths import COAST_CONFIGS, models_output_dir, resolve_windows_path, results_output_dir
from proj.project.models.plot_utils import save_training_history_plot
from proj.project.models.RNN_AR import (
    WindowDataset,
    format_duration,
    save_error_histogram,
    save_scatter_plot,
)
from proj.project.models.training_utils import (
    TrajectoryLoss,
    TrainingImprovementConfig,
    add_training_improvement_args,
    curriculum_train_steps,
    enrich_history_row,
    make_land_penalty,
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
    print_position_metrics,
    resolve_window_hours,
    scale_history_features,
    stationary_filter_from_args,
)

CONTEXT_KEYS = ("9h", "12h", "18h", "24h")
CONTEXT_STEPS = {"9h": 54, "12h": 72, "18h": 108, "24h": 144}
DEFAULT_CONTEXT_HOURS = (9.0, 12.0, 18.0, 24.0)


def gate_temperature(epoch: int, total_epochs: int, start: float = 1.0, end: float = 0.2) -> float:
    if total_epochs <= 1:
        return end
    frac = epoch / max(total_epochs - 1, 1)
    return start + (end - start) * frac


def gate_entropy(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(alpha * torch.log(alpha + eps)).sum(dim=-1).mean()


def summarize_gate(
    alpha: np.ndarray,
    gate_logits: np.ndarray | None,
    context_hours: tuple[float, ...],
) -> dict:
    labels = [f"{int(h)}h" for h in context_hours]
    argmax = alpha.argmax(axis=1)
    n = len(alpha)
    out: dict = {
        "alpha_mean": {lbl: float(alpha[:, i].mean()) for i, lbl in enumerate(labels)},
        "argmax_pct": {lbl: float(100.0 * (argmax == i).mean()) for i, lbl in enumerate(labels)},
        "gate_entropy_mean": float(
            -(alpha * np.log(alpha + 1e-8)).sum(axis=1).mean()
        ),
    }
    if gate_logits is not None:
        soft = torch.softmax(torch.from_numpy(gate_logits), dim=-1).numpy()
        out["softmax_entropy_mean"] = float(
            -(soft * np.log(soft + 1e-8)).sum(axis=1).mean()
        )
    most_idx = int(np.argmax([out["argmax_pct"][lbl] for lbl in labels]))
    out["most_selected_context"] = labels[most_idx]
    out["selection_share_pct"] = out["argmax_pct"][labels[most_idx]]
    return out


class DiffEncoderAdaptiveARRNN(nn.Module):
    """Four independent LSTM encoders + shared gate + shared AR decoder."""

    def __init__(
        self,
        input_dim: int,
        future_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        gate_mode: str = "softmax",
    ):
        super().__init__()
        if gate_mode not in ("softmax", "hard"):
            raise ValueError("gate_mode must be 'softmax' or 'hard'")
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gate_mode = gate_mode
        self.context_steps = dict(CONTEXT_STEPS)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.encoders = nn.ModuleDict(
            {
                key: nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=rnn_dropout,
                    batch_first=True,
                )
                for key in CONTEXT_KEYS
            }
        )
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def _encode_all(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        assert x.ndim == 3, f"expected [B,T,D], got {tuple(x.shape)}"
        batch_size, full_steps, _ = x.shape
        assert full_steps >= self.context_steps["24h"], (
            f"need >= {self.context_steps['24h']} history steps, got {full_steps}"
        )

        h_list: list[torch.Tensor] = []
        c_list: list[torch.Tensor] = []
        for key in CONTEXT_KEYS:
            n_steps = self.context_steps[key]
            x_slice = x[:, -n_steps:, :]
            assert x_slice.shape == (batch_size, n_steps, x.shape[-1])
            _, (h, c) = self.encoders[key](x_slice)
            assert h.shape == (self.num_layers, batch_size, self.hidden_dim)
            assert c.shape == (self.num_layers, batch_size, self.hidden_dim)
            h_list.append(h)
            c_list.append(c)

        gate_input = torch.cat([h[-1] for h in h_list], dim=-1)
        assert gate_input.shape == (batch_size, 4 * self.hidden_dim)
        gate_logits = self.gate(gate_input)
        assert gate_logits.shape == (batch_size, 4)
        return h_list, c_list, gate_logits

    def _apply_gate(
        self,
        gate_logits: torch.Tensor,
        *,
        training: bool,
        temperature: float,
    ) -> torch.Tensor:
        if self.gate_mode == "softmax":
            return torch.softmax(gate_logits, dim=-1)
        if training:
            return F.gumbel_softmax(gate_logits, tau=temperature, hard=True, dim=-1)
        selected = gate_logits.argmax(dim=-1)
        return F.one_hot(selected, num_classes=4).to(gate_logits.dtype)

    def _mix_states(
        self,
        h_list: list[torch.Tensor],
        c_list: list[torch.Tensor],
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = alpha.shape[0]
        hidden_states = torch.stack([h.permute(1, 0, 2) for h in h_list], dim=1)
        cell_states = torch.stack([c.permute(1, 0, 2) for c in c_list], dim=1)
        assert hidden_states.shape == (batch_size, 4, self.num_layers, self.hidden_dim)
        alpha_expanded = alpha[:, :, None, None]
        mixed_h = (hidden_states * alpha_expanded).sum(dim=1).permute(1, 0, 2).contiguous()
        mixed_c = (cell_states * alpha_expanded).sum(dim=1).permute(1, 0, 2).contiguous()
        assert mixed_h.shape == (self.num_layers, batch_size, self.hidden_dim)
        return mixed_h, mixed_c

    @torch.no_grad()
    def representation_cosines(self, x: torch.Tensor) -> dict[str, float]:
        self.eval()
        h_list, _, _ = self._encode_all(x)
        vecs = [h[-1] for h in h_list]
        return {
            "cos_9_12": float(F.cosine_similarity(vecs[0], vecs[1], dim=-1).mean().item()),
            "cos_12_18": float(F.cosine_similarity(vecs[1], vecs[2], dim=-1).mean().item()),
            "cos_18_24": float(F.cosine_similarity(vecs[2], vecs[3], dim=-1).mean().item()),
        }

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        *,
        temperature: float = 1.0,
        return_gate: bool = False,
    ):
        h_list, c_list, gate_logits = self._encode_all(x)
        alpha = self._apply_gate(gate_logits, training=self.training, temperature=temperature)
        mixed_h, mixed_c = self._mix_states(h_list, c_list, alpha)
        hidden: tuple[torch.Tensor, torch.Tensor] = (mixed_h, mixed_c)

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
        if return_gate:
            return y_hat, alpha, gate_logits
        return y_hat


def train_one_epoch(
    model: DiffEncoderAdaptiveARRNN,
    dataloader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float,
    temperature: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, _, batch_weight = unpack_window_batch(batch, device)
        y_hat = model(
            batch_x,
            target=batch_y_delta,
            teacher_forcing_ratio=teacher_forcing_ratio,
            temperature=temperature,
        )
        loss = criterion(y_hat, batch_y_delta, batch_anchor, batch_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(
    model: DiffEncoderAdaptiveARRNN,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, _, batch_weight = unpack_window_batch(batch, device)
        y_hat = model(batch_x)
        loss = criterion(y_hat, batch_y_delta, batch_anchor, batch_weight)
        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_gate_and_metrics(
    model: DiffEncoderAdaptiveARRNN,
    dataloader: DataLoader,
    device: torch.device,
    *,
    horizon_step: int,
    context_hours: tuple[float, ...],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    alpha_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    cos_samples = 0
    cos_acc = {"cos_9_12": 0.0, "cos_12_18": 0.0, "cos_18_24": 0.0}

    for batch in dataloader:
        batch_x, batch_y_delta, batch_anchor, _, _ = unpack_window_batch(batch, device)
        y_hat, alpha, gate_logits = model(batch_x, return_gate=True)
        y_true_abs = deltas_to_absolute(
            batch_y_delta.cpu().numpy(), batch_anchor.cpu().numpy(), target_mode="anchor_offset"
        )
        y_pred_abs = deltas_to_absolute(
            y_hat.cpu().numpy(), batch_anchor.cpu().numpy(), target_mode="anchor_offset"
        )
        y_true_list.append(y_true_abs)
        y_pred_list.append(y_pred_abs)
        alpha_list.append(alpha.cpu().numpy())
        logits_list.append(gate_logits.cpu().numpy())

        cos = model.representation_cosines(batch_x)
        n = batch_x.size(0)
        cos_samples += n
        for k in cos_acc:
            cos_acc[k] += cos[k] * n

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    alpha_all = np.concatenate(alpha_list, axis=0)
    logits_all = np.concatenate(logits_list, axis=0)
    cos_mean = {k: v / max(cos_samples, 1) for k, v in cos_acc.items()}

    gate_summary = summarize_gate(alpha_all, logits_all, context_hours)
    gate_summary["representation_cosines"] = cos_mean

    fde = haversine_batch(y_true[:, horizon_step], y_pred[:, horizon_step])
    ade = haversine_batch(y_true, y_pred).mean(axis=1)
    gate_summary["median_fde_km"] = float(np.median(fde))
    gate_summary["mean_fde_km"] = float(np.mean(fde))
    gate_summary["median_ade_km"] = float(np.median(ade))
    gate_summary["mean_ade_km"] = float(np.mean(ade))
    return gate_summary, y_true, y_pred, alpha_all, cos_mean


def haversine_batch(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    from proj.project.window_data import haversine_km

    if y_true.ndim == 2:
        return haversine_km(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
    return np.array(
        [
            haversine_km(y_true[i, :, 0], y_true[i, :, 1], y_pred[i, :, 0], y_pred[i, :, 1])
            for i in range(len(y_true))
        ]
    )


def run_diff_encoder_adaptive(
    *,
    input_path: Path | None,
    coast_name: str | None,
    region: str,
    gate_mode: str,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
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
) -> Path:
    training_config = training_config or TrainingImprovementConfig()
    start_time = time.perf_counter()

    input_path, coast, region = resolve_windows_path(coast_name, region, input_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Gate mode: {gate_mode}")

    prefetch_size = int(sample_size / 0.7) + 1000 if sample_size else None
    df = load_windows_filtered(
        input_path,
        sample_size=prefetch_size,
        motion_filter=motion_filter,
        maneuver_oversample=False,
        motion_balanced_sample=False,
        seed=seed,
    )

    tag = "RNN_AR_diff_encoder"
    output_dir = results_output_dir(coast, input_path, tag, df, run_tag=run_tag)
    model_dir = models_output_dir(coast, input_path, df, run_tag=run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    full_history_steps, _, history_steps, future_steps, step_minutes = resolve_window_hours(
        df, history_hours=24.0, future_hours=future_hours
    )
    context_steps = [hours_to_window_steps(h, step_minutes) for h in context_hours]
    assert context_steps == [CONTEXT_STEPS[k] for k in CONTEXT_KEYS], context_steps
    horizon_step = horizon_step_index(df, horizon_hours, future_steps)
    actual_horizon_hours = (horizon_step + 1) * step_minutes / 60

    train_df, val_df, test_df, _ = make_train_val_test_frames(
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

    train_weights = compute_sample_weights(x_train) if training_config.difficulty_weighting else None
    x_train, [x_val, x_test], scaler = scale_history_features(x_train, [x_val, x_test])

    train_ds = WindowDataset(x_train, y_delta_train, anchor_train, train_weights)
    val_ds = WindowDataset(x_val, y_delta_val, anchor_val)
    test_ds = WindowDataset(x_test, y_delta_test, anchor_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    train_eval_size = min(len(train_ds), max(len(val_ds), 10_000))
    train_eval_idx = np.random.default_rng(seed).choice(len(train_ds), size=train_eval_size, replace=False)
    train_eval_loader = DataLoader(
        Subset(WindowDataset(x_train, y_delta_train, anchor_train), train_eval_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DiffEncoderAdaptiveARRNN(
        input_dim=len(FEATURE_COLS),
        future_steps=future_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        gate_mode=gate_mode,
    ).to(device)

    criterion = TrajectoryLoss(
        haversine_weight=training_config.haversine_weight,
        land_penalty=make_land_penalty(training_config.land_penalty_weight, device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    resample_minutes = int(step_minutes)

    print(
        f"\n=== {coast.name} | diff-encoder adaptive ({gate_mode}) | "
        f"contexts={list(context_hours)}h -> {future_hours}h future ==="
    )
    print(f"Windows — train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict] = []
    total_train_time = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_tf = (
            scheduled_teacher_forcing(epoch, epochs, start=teacher_forcing_ratio, end=0.0)
            if training_config.scheduled_teacher_forcing
            else teacher_forcing_ratio
        )
        temp = gate_temperature(epoch, epochs) if gate_mode == "hard" else 1.0
        if training_config.curriculum:
            criterion.set_train_steps(
                curriculum_train_steps(epoch, epochs, future_steps, resample_minutes=resample_minutes)
            )
        else:
            criterion.set_train_steps(None)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch_tf, temp
        )
        criterion.set_train_steps(None)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_eval_loss = evaluate_loss(model, train_eval_loader, criterion, device)

        val_gate, _, _, _, _ = collect_gate_and_metrics(
            model, val_loader, device, horizon_step=horizon_step, context_hours=context_hours
        )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        total_train_time += elapsed
        row = enrich_history_row(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            epoch_sec=elapsed,
            train_steps=future_steps,
            future_steps=future_steps,
            teacher_forcing=epoch_tf,
            train_eval_loss=train_eval_loss,
        )
        row["gate_temperature"] = temp
        row["val_gate"] = val_gate
        history.append(row)
        print(
            f"| epoch {epoch:3d} | time {elapsed:6.2f}s | train {train_loss:8.5f} | "
            f"val {val_loss:8.5f} | gate_H {val_gate['gate_entropy_mean']:.3f} | "
            f"argmax {val_gate['most_selected_context']} ({val_gate['selection_share_pct']:.1f}%)"
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
    test_loss_at_best = evaluate_loss(model, test_loader, criterion, device)
    test_gate, y_true_traj, y_pred_traj, alpha_test, cos_mean = collect_gate_and_metrics(
        model, test_loader, device, horizon_step=horizon_step, context_hours=context_hours
    )

    eval_true = y_true_traj[:, horizon_step, :]
    eval_pred = y_pred_traj[:, horizon_step, :]
    kin_true, kin_pred = kinematic_position_at_horizon(
        x_test_raw, y_test_abs, horizon_step, step_minutes=step_minutes
    )
    metrics = [
        evaluate_final_position(kin_true, kin_pred, "Kinematic baseline", anchor=anchor_test),
        evaluate_final_position(
            eval_true,
            eval_pred,
            f"Diff-encoder adaptive ({actual_horizon_hours:.1f}h ahead)",
            anchor=anchor_test,
        ),
        evaluate_full_trajectory(y_true_traj, y_pred_traj, "Diff-encoder adaptive: full trajectory"),
    ]
    metrics.extend(
        evaluate_stratified_positions(
            y_true_traj[:, horizon_step, :],
            y_pred_traj[:, horizon_step, :],
            x_test_raw,
            "Diff-encoder adaptive",
            anchor=anchor_test,
        )
    )
    for item in metrics:
        print_position_metrics(item)

    alpha_labels = [f"alpha_{int(h)}h" for h in context_hours]
    alpha_path = output_dir / "context_alpha_weights.json"
    alpha_payload = {
        "gate_mode": gate_mode,
        "context_hours": list(context_hours),
        "labels": alpha_labels,
        "alpha_mean": test_gate["alpha_mean"],
        "argmax_pct": test_gate["argmax_pct"],
        "gate_entropy_mean": test_gate["gate_entropy_mean"],
        "softmax_entropy_mean": test_gate.get("softmax_entropy_mean"),
        "representation_cosines": cos_mean,
        "per_sample": alpha_test.tolist(),
    }
    alpha_path.write_text(json.dumps(alpha_payload, indent=2), encoding="utf-8")

    metrics_path = output_dir / "diff_encoder_adaptive_metrics.json"
    results = {
        "input": str(input_path),
        "coast": coast.name,
        "gate_mode": gate_mode,
        "context_hours": list(context_hours),
        "future_hours": float(future_hours),
        "history_steps": int(history_steps),
        "future_steps": int(future_steps),
        "horizon_hours_actual": float(actual_horizon_hours),
        "training": training_improvements_dict(training_config),
        "metrics": metrics,
        "history": history,
        "eval_summary": {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "test_loss_at_best": float(test_loss_at_best),
        },
        "test_gate_summary": test_gate,
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    save_training_history_plot(
        history,
        output_dir / "diff_encoder_adaptive_training_history.png",
        title=f"Diff-encoder adaptive ({gate_mode}) — training",
        loss_label="Trajectory loss",
        autoregressive=True,
        curriculum_enabled=training_config.curriculum,
        test_loss_at_best=test_loss_at_best,
        test_epoch=best_epoch,
    )
    save_error_histogram(
        eval_true,
        eval_pred,
        output_dir / "diff_encoder_adaptive_error_hist.png",
        title=f"Diff-encoder adaptive ({gate_mode}) @ {actual_horizon_hours:.1f}h",
    )
    save_scatter_plot(
        eval_true,
        eval_pred,
        output_dir / "diff_encoder_adaptive_scatter.png",
        title=f"Diff-encoder adaptive ({gate_mode}) @ {actual_horizon_hours:.1f}h",
    )

    ckpt_name = f"diff_encoder_adaptive_{gate_mode}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "context_hours": list(context_hours),
            "context_steps": context_steps,
            "future_steps": future_steps,
            "gate_mode": gate_mode,
        },
        model_dir / ckpt_name,
    )

    print(f"\nSaved: {metrics_path}")
    print(f"Alpha weights: {alpha_path}")
    print(f"Test gate: {json.dumps(test_gate, indent=2)}")
    print(f"Runtime: {format_duration(time.perf_counter() - start_time)}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive AR with separate encoders per context length."
    )
    parser.add_argument("--coast", choices=sorted(COAST_CONFIGS.keys()), default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument(
        "--gate-mode",
        required=True,
        choices=["softmax", "hard"],
        help="softmax = soft mixture; hard = Gumbel-Softmax train / argmax eval",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
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
    add_stationary_filter_args(parser)
    add_training_improvement_args(parser)
    args = parser.parse_args()

    region = args.region or (
        COAST_CONFIGS[args.coast].default_region
        if args.coast
        else COAST_CONFIGS["Eastern coast"].default_region
    )
    run_diff_encoder_adaptive(
        input_path=args.input,
        coast_name=args.coast,
        region=region,
        gate_mode=args.gate_mode,
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
        future_hours=args.future_hours,
        teacher_forcing_ratio=args.teacher_forcing,
        motion_filter=stationary_filter_from_args(args),
        training_config=training_config_from_args(args),
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
