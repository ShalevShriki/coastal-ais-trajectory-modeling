"""Shared training helpers: combined loss, curriculum, scheduled teacher forcing."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

EARTH_RADIUS_KM = 6371.0


def haversine_km_torch(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    a = a.clamp(0.0, 1.0)
    # atan2 formulation is more stable for autograd than arcsin(sqrt(a)).
    return (
        EARTH_RADIUS_KM
        * 2.0
        * torch.atan2(torch.sqrt(a), torch.sqrt((1.0 - a).clamp_min(1e-6)))
    )


def local_km_error_torch(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    """Stable local planar distance (km) for training; exact Haversine for eval."""
    lat_mid = torch.deg2rad((lat1 + lat2) * 0.5)
    dlat_km = (lat2 - lat1) * 111.322
    dlon_km = (lon2 - lon1) * 111.322 * torch.cos(lat_mid).clamp(min=1e-3)
    return torch.sqrt(dlat_km * dlat_km + dlon_km * dlon_km + 1e-6)


class TrajectoryLoss(nn.Module):
    """
    Combined Huber (delta in degrees) + Haversine (km) on absolute positions.

    Supports curriculum via set_train_steps() and per-sample difficulty weights.
    Optional relative term: ADE / true path length (clamped) for scale-invariant training.
    Optional soft land penalty via bilinear sampling of a coarse land raster.
    """

    def __init__(
        self,
        haversine_weight: float = 0.5,
        huber_delta: float = 0.01,
        relative_weight: float = 0.0,
        min_path_km: float = 10.0,
        target_mode: str = "anchor_offset",
        land_penalty: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.haversine_weight = float(haversine_weight)
        self.relative_weight = float(relative_weight)
        self.min_path_km = float(min_path_km)
        self.target_mode = str(target_mode)
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")
        self.land_penalty = land_penalty
        self.train_steps: int | None = None

    def set_train_steps(self, steps: int | None) -> None:
        self.train_steps = steps

    def forward(
        self,
        pred_delta: torch.Tensor,
        true_delta: torch.Tensor,
        anchor: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = self.train_steps or pred_delta.shape[1]
        pred_delta = pred_delta[:, :steps, :]
        true_delta = true_delta[:, :steps, :]

        huber = self.huber(pred_delta, true_delta).mean(dim=(1, 2))

        anchor_exp = anchor.unsqueeze(1)
        if self.target_mode == "step_delta":
            pred_abs = anchor_exp + torch.cumsum(pred_delta, dim=1)
            true_abs = anchor_exp + torch.cumsum(true_delta, dim=1)
        else:
            pred_abs = anchor_exp + pred_delta
            true_abs = anchor_exp + true_delta
        dist_km = local_km_error_torch(
            true_abs[..., 0], true_abs[..., 1], pred_abs[..., 0], pred_abs[..., 1]
        )
        # Scale km to ~degree-scale so Huber and Haversine gradients stay balanced.
        haversine = dist_km.mean(dim=1) / 50.0

        w = self.haversine_weight
        loss = (1.0 - w) * huber + w * haversine

        if self.relative_weight > 0.0 and steps > 1:
            step_dist = local_km_error_torch(
                true_abs[:, 1:, 0],
                true_abs[:, 1:, 1],
                true_abs[:, :-1, 0],
                true_abs[:, :-1, 1],
            )
            path_len = step_dist.sum(dim=1).clamp_min(self.min_path_km)
            relative_ade = dist_km.mean(dim=1) / path_len
            loss = loss + self.relative_weight * relative_ade

        if sample_weight is not None:
            loss = loss * sample_weight

        total = loss.mean()
        if self.land_penalty is not None:
            total = total + self.land_penalty(pred_abs)
        return total


@dataclass
class TrainingImprovementConfig:
    haversine_weight: float = 0.5
    relative_loss_weight: float = 0.0
    min_path_km: float = 10.0
    difficulty_weighting: bool = True
    maneuver_oversample: bool = True
    maneuver_fraction: float = 0.3
    motion_balanced_sample: bool = False
    straight_fraction: float = 0.15
    other_fraction: float = 0.15
    residual_naive: bool = False
    kinematic_baseline: bool = True
    split_by: str = "trajectory"
    curriculum: bool = True
    curriculum_start_hours: float = 6.0
    scheduled_teacher_forcing: bool = True
    teacher_forcing_start: float = 0.3
    teacher_forcing_end: float = 0.0
    land_penalty_weight: float = 0.0


def make_land_penalty(weight: float, device: torch.device | None = None):
    """Build SoftLandPenalty or None. Grid is cached under data/processed/."""
    if weight is None or float(weight) <= 0:
        return None
    from pathlib import Path

    from proj.project.models.land_mask_utils import SoftLandPenalty, build_or_load_land_grid

    project = Path(__file__).resolve().parents[1]
    grid = build_or_load_land_grid(project / "data/processed/land_grid_us.npz")
    penalty = SoftLandPenalty(grid, weight=float(weight))
    if device is not None:
        penalty = penalty.to(device)
    return penalty


def apply_residual_prediction(
    pred: torch.Tensor,
    naive_delta: torch.Tensor | None,
    *,
    residual: bool,
) -> torch.Tensor:
    if residual and naive_delta is not None:
        return pred + naive_delta
    return pred


def unpack_window_batch(
    batch: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return x, y_delta, anchor, naive_delta (optional), sample_weight (optional)."""
    if len(batch) == 5:
        batch_x, batch_y_delta, batch_anchor, batch_naive, batch_weight = batch
        return (
            batch_x.to(device),
            batch_y_delta.to(device),
            batch_anchor.to(device),
            batch_naive.to(device),
            batch_weight.to(device),
        )
    if len(batch) == 4:
        batch_x, batch_y_delta, batch_anchor, fourth = batch
        batch_x = batch_x.to(device)
        batch_y_delta = batch_y_delta.to(device)
        batch_anchor = batch_anchor.to(device)
        if fourth.dim() == 1:
            return batch_x, batch_y_delta, batch_anchor, None, fourth.to(device)
        return batch_x, batch_y_delta, batch_anchor, fourth.to(device), None
    batch_x, batch_y_delta, batch_anchor = batch
    return (
        batch_x.to(device),
        batch_y_delta.to(device),
        batch_anchor.to(device),
        None,
        None,
    )


def training_improvements_dict(config: TrainingImprovementConfig) -> dict:
    return {
        "haversine_weight": config.haversine_weight,
        "relative_loss_weight": config.relative_loss_weight,
        "min_path_km": config.min_path_km,
        "difficulty_weighting": config.difficulty_weighting,
        "maneuver_oversample": config.maneuver_oversample,
        "maneuver_fraction": config.maneuver_fraction,
        "motion_balanced_sample": config.motion_balanced_sample,
        "straight_fraction": config.straight_fraction,
        "other_fraction": config.other_fraction,
        "residual_naive": config.residual_naive,
        "kinematic_baseline": config.kinematic_baseline,
        "split_by": config.split_by,
        "curriculum": config.curriculum,
        "curriculum_start_hours": config.curriculum_start_hours,
        "scheduled_teacher_forcing": config.scheduled_teacher_forcing,
        "teacher_forcing_start": config.teacher_forcing_start,
        "teacher_forcing_end": config.teacher_forcing_end,
        "land_penalty_weight": config.land_penalty_weight,
    }


def scheduled_teacher_forcing(
    epoch: int,
    total_epochs: int,
    *,
    start: float = 0.3,
    end: float = 0.0,
) -> float:
    if total_epochs <= 1:
        return end
    progress = epoch / max(total_epochs - 1, 1)
    return float(start + (end - start) * progress)


def curriculum_train_steps(
    epoch: int,
    total_epochs: int,
    future_steps: int,
    *,
    resample_minutes: int = 10,
    start_hours: float = 6.0,
    ramp_fraction: float = 0.5,
) -> int:
    """Gradually increase predicted horizon from start_hours to full window."""
    start_steps = max(1, int(round(start_hours * 60 / resample_minutes)))
    end_steps = future_steps
    ramp_epochs = max(1, int(total_epochs * ramp_fraction))
    progress = min(1.0, epoch / ramp_epochs)
    steps = int(round(start_steps + progress * (end_steps - start_steps)))
    return max(1, min(steps, future_steps))


def add_training_improvement_args(parser) -> None:
    parser.add_argument(
        "--haversine-loss-weight",
        type=float,
        default=0.5,
        help="Weight for Haversine km term in combined loss (0=Huber only, 1=Haversine only).",
    )
    parser.add_argument(
        "--relative-loss-weight",
        type=float,
        default=0.0,
        help="Optional ADE/path-length relative term in training loss (0=off, try 0.1–0.2).",
    )
    parser.add_argument(
        "--min-path-km",
        type=float,
        default=10.0,
        help="Floor for path-length normalization in relative loss/metrics (default 10 km).",
    )
    parser.add_argument(
        "--no-difficulty-weighting",
        action="store_true",
        help="Disable per-sample loss weights from |dcog|/|dsog| in history.",
    )
    parser.add_argument(
        "--no-maneuver-oversample",
        action="store_true",
        help="Disable 30%% maneuver-biased sampling when subsampling windows.",
    )
    parser.add_argument(
        "--motion-balanced-sample",
        action="store_true",
        help="Oversample straight/other motion buckets when subsampling windows.",
    )
    parser.add_argument(
        "--straight-fraction",
        type=float,
        default=0.15,
        help="Target fraction of straight windows in motion-balanced sample (default 0.15).",
    )
    parser.add_argument(
        "--other-fraction",
        type=float,
        default=0.15,
        help="Target fraction of 'other' windows in motion-balanced sample (default 0.15).",
    )
    parser.add_argument(
        "--residual-naive",
        action="store_true",
        help="Predict correction to constant-velocity naive trajectory.",
    )
    parser.add_argument(
        "--no-kinematic-baseline",
        action="store_true",
        help="Use last-step dlat/dlon baseline instead of SOG+COG for residuals/metrics.",
    )
    parser.add_argument(
        "--split-by",
        choices=("trajectory", "mmsi"),
        default="trajectory",
        help="Group column for train/val/test split (mmsi = stronger generalization test).",
    )
    parser.add_argument(
        "--maneuver-fraction",
        type=float,
        default=0.3,
        help="Fraction of sample drawn from high-maneuver windows (default 0.3).",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable horizon curriculum (train on full future from epoch 0).",
    )
    parser.add_argument(
        "--curriculum-start-hours",
        type=float,
        default=6.0,
        help="Initial prediction horizon for curriculum (default 6h).",
    )
    parser.add_argument(
        "--no-scheduled-tf",
        action="store_true",
        help="(AR only) Disable scheduled teacher forcing decay.",
    )
    parser.add_argument(
        "--land-penalty-weight",
        type=float,
        default=0.0,
        help="Soft land-mask penalty weight on predicted absolute positions (0=off, try 0.05–0.2).",
    )


def enrich_history_row(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    epoch_sec: float,
    train_steps: int | None = None,
    future_steps: int | None = None,
    teacher_forcing: float | None = None,
    train_eval_loss: float | None = None,
    test_loss: float | None = None,
) -> dict[str, float]:
    row: dict[str, float] = {
        "epoch": float(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "lr": float(lr),
        "epoch_sec": float(epoch_sec),
    }
    if train_steps is not None:
        row["train_steps"] = float(train_steps)
    if future_steps is not None:
        row["future_steps"] = float(future_steps)
    if teacher_forcing is not None:
        row["teacher_forcing"] = float(teacher_forcing)
    if train_eval_loss is not None:
        row["train_eval_loss"] = float(train_eval_loss)
    if test_loss is not None:
        row["test_loss"] = float(test_loss)
    return row


def training_config_from_args(args) -> TrainingImprovementConfig:
    return TrainingImprovementConfig(
        haversine_weight=getattr(args, "haversine_loss_weight", 0.5),
        relative_loss_weight=getattr(args, "relative_loss_weight", 0.0),
        min_path_km=getattr(args, "min_path_km", 10.0),
        difficulty_weighting=not getattr(args, "no_difficulty_weighting", False),
        maneuver_oversample=not getattr(args, "no_maneuver_oversample", False),
        maneuver_fraction=getattr(args, "maneuver_fraction", 0.3),
        motion_balanced_sample=getattr(args, "motion_balanced_sample", False),
        straight_fraction=getattr(args, "straight_fraction", 0.15),
        other_fraction=getattr(args, "other_fraction", 0.15),
        residual_naive=getattr(args, "residual_naive", False),
        kinematic_baseline=not getattr(args, "no_kinematic_baseline", False),
        split_by=getattr(args, "split_by", "trajectory"),
        curriculum=not getattr(args, "no_curriculum", False),
        curriculum_start_hours=getattr(args, "curriculum_start_hours", 6.0),
        scheduled_teacher_forcing=not getattr(args, "no_scheduled_tf", False),
        land_penalty_weight=float(getattr(args, "land_penalty_weight", 0.0) or 0.0),
    )
