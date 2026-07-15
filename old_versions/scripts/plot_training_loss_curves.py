#!/usr/bin/env python3
"""Plot train/validation loss vs epoch from Slurm training logs."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]

EPOCH_RE = re.compile(
    r"\|\s*epoch\s+(\d+)\s*\|.*?train loss\s+([\d.]+)\s*\|\s*valid loss\s+([\d.]+)"
)

V1_LOGS = {
    "Transformer": PROJECT / "LOG/exp1_v1_baseline_transformer-22261.out",
    "RNN": PROJECT / "LOG/exp1_v1_baseline_rnn-22262.out",
    "RNN_AR": PROJECT / "LOG/exp1_v1_baseline_rnn_ar-22263.out",
}


def parse_log(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float]] = []
    for m in EPOCH_RE.finditer(text):
        rows.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "val_loss": float(m.group(3)),
        })
    if not rows:
        raise ValueError(f"No epoch loss lines found in {path}")
    return rows


def best_epoch(rows: list[dict[str, float]]) -> tuple[int, float]:
    best = min(rows, key=lambda r: r["val_loss"])
    return int(best["epoch"]), float(best["val_loss"])


def plot_single(
    name: str,
    rows: list[dict[str, float]],
    out: Path,
    *,
    train_color: str,
    val_color: str,
) -> None:
    epochs = [r["epoch"] for r in rows]
    train = [r["train_loss"] for r in rows]
    val = [r["val_loss"] for r in rows]
    best_ep, best_val = best_epoch(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, train, color=train_color, lw=2.2, marker="o", ms=3, label="Train loss")
    ax.plot(epochs, val, color=val_color, lw=2.2, marker="s", ms=3, label="Validation loss")
    ax.axvline(best_ep, color="#555", ls=":", lw=1.2, alpha=0.8)
    ax.scatter([best_ep], [best_val], s=80, c=val_color, edgecolors="white", zorder=5)
    ax.annotate(
        f"best val @ epoch {best_ep}\nloss = {best_val:.4f}",
        xy=(best_ep, best_val),
        xytext=(best_ep + max(1, len(epochs) * 0.05), best_val + (max(val) - min(val)) * 0.08),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#555", lw=1),
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (Huber + Haversine composite)", fontsize=11)
    ax.set_title(f"{name} — training loss vs epoch", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(-0.5, max(epochs) + 0.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_combined(
    histories: dict[str, list[dict[str, float]]],
    out: Path,
) -> None:
    colors = {
        "Transformer": "#7B1FA2",
        "RNN": "#FF9800",
        "RNN_AR": "#2E7D32",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    for name, rows in histories.items():
        epochs = [r["epoch"] for r in rows]
        c = colors.get(name, "#333")
        axes[0].plot(epochs, [r["train_loss"] for r in rows], lw=2, label=name, color=c)
        axes[1].plot(epochs, [r["val_loss"] for r in rows], lw=2, label=name, color=c)

    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Train loss vs epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Validation loss vs epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend()

    fig.suptitle("v1/experiment1 — all models", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_grid(
    histories: dict[str, list[dict[str, float]]],
    out: Path,
) -> None:
    names = list(histories.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5), squeeze=False)
    palette = {"train": "#1E88E5", "val": "#E53935"}

    for ax, name in zip(axes[0], names):
        rows = histories[name]
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r["train_loss"] for r in rows], color=palette["train"], lw=2, label="Train")
        ax.plot(epochs, [r["val_loss"] for r in rows], color=palette["val"], lw=2, label="Val")
        best_ep, best_val = best_epoch(rows)
        ax.axvline(best_ep, color="#888", ls=":", lw=1)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=8)
        ax.text(
            0.03, 0.97, f"best val {best_val:.4f}\n@ epoch {best_ep}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.suptitle("v1/experiment1 — loss vs epoch (per model)", y=1.02, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "data/results/USA Combined/unknown/v1/experiment1/visualizations",
    )
    parser.add_argument("--log-dir", type=Path, default=PROJECT / "LOG")
    args = parser.parse_args()

    histories: dict[str, list[dict[str, float]]] = {}
    for name, log_path in V1_LOGS.items():
        histories[name] = parse_log(log_path)

    plot_grid(histories, args.output_dir / "loss_vs_epoch_grid.png")
    plot_combined(histories, args.output_dir / "loss_vs_epoch_combined.png")

    for name, rows in histories.items():
        plot_single(
            name,
            rows,
            args.output_dir / f"loss_vs_epoch_{name.lower()}.png",
            train_color="#1E88E5",
            val_color="#E53935",
        )


if __name__ == "__main__":
    main()
