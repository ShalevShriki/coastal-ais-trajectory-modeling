"""Training plot helpers (no PyTorch dependency)."""
from __future__ import annotations

from pathlib import Path


def _scheduled_teacher_forcing(
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


def _curriculum_train_steps(
    epoch: int,
    total_epochs: int,
    future_steps: int,
    *,
    resample_minutes: int = 10,
    start_hours: float = 6.0,
    ramp_fraction: float = 0.5,
) -> int:
    start_steps = max(1, int(round(start_hours * 60 / resample_minutes)))
    end_steps = future_steps
    ramp_epochs = max(1, int(total_epochs * ramp_fraction))
    progress = min(1.0, epoch / ramp_epochs)
    steps = int(round(start_steps + progress * (end_steps - start_steps)))
    return max(1, min(steps, future_steps))


def _history_train_steps(
    row: dict,
    epoch: int,
    total_epochs: int,
    future_steps: int,
    *,
    resample_minutes: int,
    curriculum_start_hours: float,
    curriculum_enabled: bool,
) -> int:
    if "train_steps" in row:
        return int(row["train_steps"])
    if not curriculum_enabled:
        return future_steps
    return _curriculum_train_steps(
        epoch,
        total_epochs,
        future_steps,
        resample_minutes=resample_minutes,
        start_hours=curriculum_start_hours,
    )


def save_training_history_plot(
    history: list[dict],
    output_path: str | Path,
    *,
    title: str,
    loss_label: str = "Huber + Haversine loss",
    autoregressive: bool = False,
    future_steps: int = 72,
    resample_minutes: int = 10,
    curriculum_start_hours: float = 6.0,
    curriculum_enabled: bool = True,
    teacher_forcing_start: float = 0.3,
    use_scheduled_teacher_forcing: bool = True,
    test_loss_at_best: float | None = None,
    test_epoch: int | None = None,
) -> None:
    """
    Two-panel training plot:
      top    — comparable eval losses (val + optional train-eval, full horizon)
      bottom — training objective with curriculum horizon / teacher forcing context
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_eval = [float(row["train_eval_loss"]) for row in history if "train_eval_loss" in row]
    has_train_eval = len(train_eval) == len(history)
    total_epochs = max(epochs) + 1

    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    best_epoch = epochs[best_idx]
    best_val = val_loss[best_idx]
    if test_epoch is None:
        test_epoch = best_epoch

    train_steps_series = [
        _history_train_steps(
            row,
            int(row["epoch"]),
            total_epochs,
            future_steps,
            resample_minutes=resample_minutes,
            curriculum_start_hours=curriculum_start_hours,
            curriculum_enabled=curriculum_enabled,
        )
        for row in history
    ]
    horizon_hours = [steps * resample_minutes / 60.0 for steps in train_steps_series]
    tf_series = [
        float(row["teacher_forcing"]) if "teacher_forcing" in row else None
        for row in history
    ]
    if autoregressive and not any(v is not None for v in tf_series):
        tf_series = [
            _scheduled_teacher_forcing(
                int(row["epoch"]),
                total_epochs,
                start=teacher_forcing_start,
                end=0.0,
            )
            if use_scheduled_teacher_forcing
            else teacher_forcing_start
            for row in history
        ]
    has_tf = autoregressive and any(v is not None for v in tf_series)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.08},
    )

    top_values = list(val_loss)
    if has_train_eval:
        top_values.extend(train_eval)
    if test_loss_at_best is not None:
        top_values.append(test_loss_at_best)
    y_span = max(top_values) - min(top_values)

    eval_label_suffix = (
        "full horizon, no teacher forcing, unweighted"
        if autoregressive
        else "held-out split, comparable eval, unweighted"
    )
    ax_top.plot(
        epochs,
        val_loss,
        color="#E65100",
        lw=2.2,
        marker="s",
        ms=3.5,
        label=f"Validation ({eval_label_suffix})",
    )
    if has_train_eval:
        ax_top.plot(
            epochs,
            train_eval,
            color="#1565C0",
            lw=2.2,
            marker="o",
            ms=3.5,
            label=f"Train eval ({eval_label_suffix})",
        )
    if test_loss_at_best is not None:
        test_label = (
            "Test @ best checkpoint (full horizon, no teacher forcing, unweighted)"
            if autoregressive
            else "Test @ best checkpoint (held-out split, unweighted)"
        )
        ax_top.scatter(
            [test_epoch],
            [test_loss_at_best],
            s=90,
            c="#2E7D32",
            marker="^",
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
            label=test_label,
        )
        ax_top.annotate(
            f"test {test_loss_at_best:.4f}",
            xy=(test_epoch, test_loss_at_best),
            xytext=(test_epoch + max(1, len(epochs) * 0.03), test_loss_at_best + y_span * 0.08),
            fontsize=8,
            color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.9),
        )
    ax_top.axvline(best_epoch, color="#666", ls=":", lw=1.2, alpha=0.85)
    ax_top.scatter([best_epoch], [best_val], s=70, c="#E65100", edgecolors="white", zorder=5)
    ax_top.annotate(
        f"best val {best_val:.4f} @ epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + max(1, len(epochs) * 0.04), best_val + y_span * 0.06),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#666", lw=0.9),
    )
    ax_top.set_ylabel(loss_label)
    ax_top.set_title(f"{title} — comparable evaluation", fontsize=11, fontweight="bold")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, linestyle=":", alpha=0.35)

    ax_bottom.plot(
        epochs,
        train_loss,
        color="#1E88E5",
        lw=2.0,
        ls="--",
        marker="o",
        ms=3,
        label="Training objective (curriculum + TF + dropout)",
    )
    ax_bottom.set_ylabel(loss_label)
    ax_bottom.set_xlabel("Epoch")
    ax_bottom.grid(True, linestyle=":", alpha=0.35)

    ax_curr = ax_bottom.twinx()
    ax_curr.plot(
        epochs,
        horizon_hours,
        color="#757575",
        lw=1.6,
        ls="-.",
        marker="^",
        ms=3,
        alpha=0.9,
        label="Train horizon (hours)",
    )
    ax_curr.set_ylabel("Train horizon (hours)", color="#616161")
    ax_curr.tick_params(axis="y", labelcolor="#616161")
    ax_curr.set_ylim(0, max(horizon_hours) * 1.15 + 0.1)

    lines, labels = ax_bottom.get_legend_handles_labels()
    lines2, labels2 = ax_curr.get_legend_handles_labels()
    legend_items = lines + lines2
    legend_labels = labels + labels2

    if has_tf:
        ax_tf = ax_bottom.twinx()
        ax_tf.spines["right"].set_position(("axes", 1.12))
        tf_values = [v if v is not None else 0.0 for v in tf_series]
        ax_tf.plot(
            epochs,
            tf_values,
            color="#6A1B9A",
            lw=1.4,
            ls=":",
            marker="x",
            ms=3,
            alpha=0.85,
            label="Teacher forcing ratio",
        )
        ax_tf.set_ylabel("Teacher forcing", color="#6A1B9A")
        ax_tf.tick_params(axis="y", labelcolor="#6A1B9A")
        ax_tf.set_ylim(-0.05, max(tf_values) * 1.2 + 0.05)
        lines3, labels3 = ax_tf.get_legend_handles_labels()
        legend_items += lines3
        legend_labels += labels3

    ax_bottom.legend(legend_items, legend_labels, loc="upper left", fontsize=8)
    ax_bottom.set_title(
        "Training objective changes each epoch (not directly comparable to validation)",
        fontsize=10,
        style="italic",
    )

    note = (
        "Top panel tracks real 12h autoregressive quality. "
        "Bottom panel rises when curriculum extends horizon and teacher forcing is reduced."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5, color="#444")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.08, top=0.96, hspace=0.28)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
