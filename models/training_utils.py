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
    return 2 * EARTH_RADIUS_KM * torch.arcsin(torch.sqrt(a.clamp(0.0, 1.0)))


class TrajectoryLoss(nn.Module):
    """
    Combined Huber (delta in degrees) + Haversine (km) on absolute positions.

    Supports curriculum via set_train_steps() and per-sample difficulty weights.
    """

    def __init__(
        self,
        haversine_weight: float = 0.5,
        huber_delta: float = 0.01,
    ):
        super().__init__()
        self.haversine_weight = float(haversine_weight)
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")
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
        pred_abs = anchor_exp + pred_delta
        true_abs = anchor_exp + true_delta
        dist_km = haversine_km_torch(
            true_abs[..., 0], true_abs[..., 1], pred_abs[..., 0], pred_abs[..., 1]
        )
        haversine = dist_km.mean(dim=1)

        w = self.haversine_weight
        loss = (1.0 - w) * huber + w * haversine

        if sample_weight is not None:
            loss = loss * sample_weight

        return loss.mean()


@dataclass
class TrainingImprovementConfig:
    haversine_weight: float = 0.5
    difficulty_weighting: bool = True
    maneuver_oversample: bool = True
    maneuver_fraction: float = 0.3
    curriculum: bool = True
    curriculum_start_hours: float = 6.0
    scheduled_teacher_forcing: bool = True
    teacher_forcing_start: float = 0.3
    teacher_forcing_end: float = 0.0


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


def training_config_from_args(args) -> TrainingImprovementConfig:
    return TrainingImprovementConfig(
        haversine_weight=getattr(args, "haversine_loss_weight", 0.5),
        difficulty_weighting=not getattr(args, "no_difficulty_weighting", False),
        maneuver_oversample=not getattr(args, "no_maneuver_oversample", False),
        maneuver_fraction=getattr(args, "maneuver_fraction", 0.3),
        curriculum=not getattr(args, "no_curriculum", False),
        curriculum_start_hours=getattr(args, "curriculum_start_hours", 6.0),
        scheduled_teacher_forcing=not getattr(args, "no_scheduled_tf", False),
    )
