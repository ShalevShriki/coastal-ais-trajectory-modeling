#!/usr/bin/env python3
"""Diagnose why AR LSTM 18h beats 9/12/24h on the coastal suite.

Compares exp_coastal AR checkpoints with:
  1) Hidden-state saturation (cosine of h vs full-history h with older steps zeroed)
  2) Hour-block occlusion (ΔFDE when zeroing oldest / newest 3h)
  3) Backprop attribution |∂L/∂x_t| mass by history hour
  4) LSTM forget-gate bias (layer 0)

Writes:
  report_figures/fig_ar_why_18h_diagnostics.png
  report_figures/fig_ar_forget_bias.png
  report_figures/fig_ar_why_18h_meta.txt

Example:
  source scripts/exp_coastal/_env.sh && cd \"$SUBROOT\"
  $PYTHON scripts/diagnose_ar_context.py --n-samples 800
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))

from proj.project.models.RNN_AR import ShipTrajectoryARRNN
from proj.project.window_data import (
    FEATURE_COLS,
    build_window_arrays,
    haversine_km,
    load_windows_filtered,
    make_train_val_test_frames,
)

DEFAULT_DATA = PROJECT / "data/processed/combined_filtered_smart_coastal/train.parquet"
DEFAULT_CKPT_ROOT = PROJECT / "data/models/USA Combined/unknown/exp_coastal"
DEFAULT_OUT = PROJECT / "data/results/USA Combined/unknown/exp_coastal/report_figures"

HOURS = (9, 12, 18, 24)
RUN_TAGS = {h: f"AR_{h}h" for h in HOURS}
STEPS_PER_HOUR = 6  # 10-minute resampling
OFFICIAL_MED_FDE = {9: 20.43, 12: 19.99, 18: 19.71, 24: 20.30}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> tuple[ShipTrajectoryARRNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ShipTrajectoryARRNN(
        input_dim=len(ckpt.get("feature_cols", FEATURE_COLS)),
        future_steps=int(ckpt["future_steps"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt.get("dropout", 0.0)),
        rnn_type=str(ckpt.get("rnn_type", "lstm")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def scale_x(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / np.maximum(scale.reshape(1, 1, -1), 1e-6)).astype(
        np.float32
    )


def encoder_h(model: ShipTrajectoryARRNN, x: torch.Tensor) -> torch.Tensor:
    """Last-layer encoder hidden state, shape (B, H)."""
    _, enc = model.encoder(x)
    if isinstance(enc, tuple):
        h, _c = enc
    else:
        h = enc
    return h[-1]


def predict_deltas(model: ShipTrajectoryARRNN, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x, target=None, teacher_forcing_ratio=0.0)


def fde_km(pred_delta: torch.Tensor, y_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Final-step Haversine FDE (km) for a batch."""
    pred = pred_delta.detach().cpu().numpy()
    pred_abs = anchor[:, None, :] + pred
    return haversine_km(pred_abs[:, -1, 0], pred_abs[:, -1, 1], y_abs[:, -1, 0], y_abs[:, -1, 1])


@torch.no_grad()
def median_fde(
    model: ShipTrajectoryARRNN,
    x: np.ndarray,
    y_abs: np.ndarray,
    anchor: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    vals = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        pred = predict_deltas(model, xb)
        vals.append(fde_km(pred, y_abs[i : i + batch_size], anchor[i : i + batch_size]))
    return float(np.median(np.concatenate(vals)))


def zero_hour_block(x: np.ndarray, *, start_h: int, end_h: int) -> np.ndarray:
    """Zero history hours [start_h, end_h) counted from the start of this model's window."""
    out = x.copy()
    t0 = int(start_h * STEPS_PER_HOUR)
    t1 = int(end_h * STEPS_PER_HOUR)
    t1 = min(t1, out.shape[1])
    t0 = max(0, min(t0, t1))
    out[:, t0:t1, :] = 0.0
    return out


def keep_last_hours(x: np.ndarray, keep_h: float) -> np.ndarray:
    """Keep only the newest keep_h hours; zero the rest."""
    out = x.copy()
    keep_steps = int(round(keep_h * STEPS_PER_HOUR))
    if keep_steps >= out.shape[1]:
        return out
    out[:, : out.shape[1] - keep_steps, :] = 0.0
    return out


def occlusion_delta_fde(
    model: ShipTrajectoryARRNN,
    x: np.ndarray,
    y_abs: np.ndarray,
    anchor: np.ndarray,
    device: torch.device,
    hours: int,
    block_h: float = 3.0,
) -> tuple[float, float]:
    base = median_fde(model, x, y_abs, anchor, device)
    oldest = zero_hour_block(x, start_h=0, end_h=block_h)
    newest = zero_hour_block(x, start_h=max(0.0, hours - block_h), end_h=float(hours))
    d_old = median_fde(model, oldest, y_abs, anchor, device) - base
    d_new = median_fde(model, newest, y_abs, anchor, device) - base
    return float(d_old), float(d_new)


def hidden_saturation_hours(
    model: ShipTrajectoryARRNN,
    x: np.ndarray,
    device: torch.device,
    hours: int,
    batch_size: int = 64,
    target_cos: float = 0.95,
) -> float:
    """Min recent hours so mean cos(h_recent, h_full) >= target_cos."""
    # Evaluate on a few prefixes: keep last k hours.
    candidates = list(range(1, hours + 1))
    full_hs = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        full_hs.append(encoder_h(model, xb).detach())
    h_full = torch.cat(full_hs, dim=0)

    best = float(hours)
    for keep_h in candidates:
        masked = keep_last_hours(x, keep_h)
        parts = []
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(masked[i : i + batch_size]).to(device)
            parts.append(encoder_h(model, xb).detach())
        h_k = torch.cat(parts, dim=0)
        cos = nn.functional.cosine_similarity(h_k, h_full, dim=-1).mean().item()
        if cos >= target_cos:
            best = float(keep_h)
            break
    return best


def gradient_hour_mass(
    model: ShipTrajectoryARRNN,
    x: np.ndarray,
    y_delta: np.ndarray,
    device: torch.device,
    hours: int,
    max_batches: int = 8,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Fraction of |∂L/∂x| mass on newest / oldest 3h of history.

    L = mean Huber on all predicted deltas (no TF). Uses cudnn-off for RNN input grads.
    """
    was_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    model.train()  # needed for some autograd paths; dropout still inactive with eval-ish...
    # Keep dropout off:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    hour_mass = np.zeros(hours, dtype=np.float64)
    n_seen = 0
    criterion = nn.SmoothL1Loss(reduction="mean")

    n = min(len(x), max_batches * batch_size)
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        yb = torch.from_numpy(y_delta[i : i + batch_size]).to(device)
        xb = xb.detach().requires_grad_(True)
        pred = model(xb, target=None, teacher_forcing_ratio=0.0)
        loss = criterion(pred, yb)
        model.zero_grad(set_to_none=True)
        if xb.grad is not None:
            xb.grad = None
        loss.backward()
        g = xb.grad.detach().abs().mean(dim=(0, 2)).cpu().numpy()  # (T,)
        # Map steps → hours within this model's window
        for t, gt in enumerate(g):
            h = min(hours - 1, t // STEPS_PER_HOUR)
            hour_mass[h] += float(gt)
        n_seen += xb.shape[0]

    torch.backends.cudnn.enabled = was_cudnn
    model.eval()
    total = hour_mass.sum() + 1e-12
    newest3 = hour_mass[max(0, hours - 3) :].sum() / total
    oldest3 = hour_mass[: min(3, hours)].sum() / total
    return float(newest3), float(oldest3)


def forget_bias_layer0(model: ShipTrajectoryARRNN) -> float:
    """Mean forget-gate bias from encoder LSTM layer 0 (PyTorch i,f,g,o order)."""
    bias = model.encoder.bias_ih_l0.detach().cpu().numpy()
    h = bias.shape[0] // 4
    return float(bias[h : 2 * h].mean())


def make_diagnostics_figure(results: dict, out_path: Path) -> None:
    labels = [f"{h}h" for h in HOURS]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Occlusion
    ax = axes[0, 0]
    old = [results[h]["occ_oldest3"] for h in HOURS]
    new = [results[h]["occ_newest3"] for h in HOURS]
    x = np.arange(len(HOURS))
    w = 0.35
    ax.bar(x - w / 2, old, w, label="Zero oldest 3h", color="#4C78A8")
    ax.bar(x + w / 2, new, w, label="Zero newest 3h", color="#F58518")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ median FDE (km)")
    ax.set_title("Occlusion (higher = more harmful to remove)")
    ax.legend(fontsize=8)

    # Gradient mass
    ax = axes[0, 1]
    gn = [100 * results[h]["grad_newest3"] for h in HOURS]
    go = [100 * results[h]["grad_oldest3"] for h in HOURS]
    ax.bar(x - w / 2, go, w, label="Oldest 3h", color="#4C78A8")
    ax.bar(x + w / 2, gn, w, label="Newest 3h", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of |∂L/∂x| mass")
    ax.set_title("Backprop attribution by history block")
    ax.legend(fontsize=8)

    # Hidden saturation
    ax = axes[1, 0]
    sat = [results[h]["hidden_95_h"] for h in HOURS]
    ax.plot(HOURS, sat, "o-", color="#54A24B", lw=2)
    ax.plot(HOURS, list(HOURS), "--", color="gray", alpha=0.5, label="Full window")
    ax.set_xlabel("Model history (h)")
    ax.set_ylabel("Hours of recent context")
    ax.set_title("Recent hours to reach 95% cos(h, h_full)")
    ax.legend(fontsize=8)
    ax.set_xticks(HOURS)

    # Subsample FDE vs official
    ax = axes[1, 1]
    sub = [results[h]["sub_med_fde"] for h in HOURS]
    off = [OFFICIAL_MED_FDE[h] for h in HOURS]
    ax.plot(HOURS, sub, "o-", label="Diagnostic subsample", color="#B279A2")
    ax.plot(HOURS, off, "s--", label="Official full test", color="#72B7B2")
    ax.set_xlabel("Model history (h)")
    ax.set_ylabel("Median FDE (km)")
    ax.set_title("FDE reference (subsample ≠ full test)")
    ax.legend(fontsize=8)
    ax.set_xticks(HOURS)

    fig.suptitle("Why AR 18h beats 24h: occlusion, grads, hidden saturation", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_forget_bias_figure(results: dict, out_path: Path) -> None:
    labels = [f"{h}h" for h in HOURS]
    vals = [results[h]["forget_bias_l0"] for h in HOURS]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("Mean forget-gate bias (encoder L0)")
    ax.set_title("LSTM forget bias — more negative ⇒ forget early history")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_meta(results: dict, keep_curve: dict, out_path: Path, n: int) -> None:
    lines = [
        f"AR why-18h diagnostics (shared n={n}; official full-test medians differ slightly due to sampling)",
        "Subsample med FDE: "
        + ", ".join(f"{h}h={results[h]['sub_med_fde']:.2f}" for h in HOURS),
        "Official full-test med FDE: "
        + ", ".join(f"{h}={OFFICIAL_MED_FDE[h]:.2f}" for h in HOURS),
        "KEY: occlusion newest3 >> oldest3 for all models; AR24 oldest3 even slightly helpful to REMOVE",
    ]
    for h in HOURS:
        r = results[h]
        lines.append(
            f"  occ {h}h oldest3={r['occ_oldest3']:.2f} newest3={r['occ_newest3']:.2f}"
        )
    for h in HOURS:
        r = results[h]
        lines.append(
            f"  grad {h}h newest3={100 * r['grad_newest3']:.1f}% "
            f"oldest3={100 * r['grad_oldest3']:.1f}%"
        )
    for h in HOURS:
        lines.append(f"  hidden {h}h: 95% cos at {results[h]['hidden_95_h']:.0f}h recent")
    lines.append("AR24 effective context medFDE by keep-last-k:")
    base = keep_curve[max(keep_curve)]
    for k, med in sorted(keep_curve.items()):
        lines.append(f"  keep{k}={med:.2f} (dmean {med - base:+.2f})")
    for h in HOURS:
        lines.append(f"  forget_bias_l0 {h}h={results[h]['forget_bias_l0']:.3f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=int, default=300_000, help="Train-size budget for split")
    parser.add_argument("--n-samples", type=int, default=800, help="Shared diagnostic subsample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = _device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    print(f"data={args.data}")

    prefetch = int(args.sample / 0.7) + 1000
    df = load_windows_filtered(args.data, sample_size=prefetch, seed=args.seed)
    _, _, test_df, _ = make_train_val_test_frames(
        df,
        test_fraction=0.2,
        val_fraction=0.1,
        seed=args.seed,
        split_by="trajectory",
        train_sample_size=args.sample,
    )
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(test_df))
    pick = rng.choice(len(test_df), size=n, replace=False)
    sub = test_df.iloc[pick].reset_index(drop=True)
    print(f"diagnostic subsample n={n} (of {len(test_df)} test rows)")

    results: dict[int, dict] = {}
    models: dict[int, tuple[ShipTrajectoryARRNN, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for h in HOURS:
        ckpt_path = args.ckpt_root / RUN_TAGS[h] / "ship_trajectory_lstm_ar.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model, ckpt = load_checkpoint(ckpt_path, device)
        hist_steps = int(ckpt["history_steps"])
        x, y_abs, y_delta, anchor = build_window_arrays(
            sub,
            feature_cols=list(ckpt.get("feature_cols", FEATURE_COLS)),
            history_steps=hist_steps,
            future_steps=int(ckpt["future_steps"]),
            target_mode=str(ckpt.get("target_mode", "anchor_offset")),
        )
        x = scale_x(x, np.asarray(ckpt["scaler_mean"], dtype=np.float32), np.asarray(ckpt["scaler_scale"], dtype=np.float32))
        models[h] = (model, ckpt, x, y_abs, y_delta, anchor)

        print(f"\n=== AR {h}h (T={hist_steps}) ===")
        sub_fde = median_fde(model, x, y_abs, anchor, device, batch_size=args.batch_size)
        print(f"  subsample med FDE = {sub_fde:.2f} km (official {OFFICIAL_MED_FDE[h]:.2f})")

        occ_old, occ_new = occlusion_delta_fde(model, x, y_abs, anchor, device, h)
        print(f"  occlusion ΔFDE oldest3={occ_old:+.2f} newest3={occ_new:+.2f}")

        sat_h = hidden_saturation_hours(model, x, device, h, batch_size=args.batch_size)
        print(f"  hidden 95% cos @ {sat_h:.0f}h recent")

        g_new, g_old = gradient_hour_mass(model, x, y_delta, device, h)
        print(f"  grad mass newest3={100*g_new:.1f}% oldest3={100*g_old:.1f}%")

        fb = forget_bias_layer0(model)
        print(f"  forget bias L0 = {fb:.3f}")

        results[h] = {
            "sub_med_fde": sub_fde,
            "occ_oldest3": occ_old,
            "occ_newest3": occ_new,
            "hidden_95_h": sat_h,
            "grad_newest3": g_new,
            "grad_oldest3": g_old,
            "forget_bias_l0": fb,
        }

    # AR24 keep-last-k curve
    model24, _ckpt24, x24, y24, _yd24, a24 = models[24]
    keep_curve: dict[int, float] = {}
    print("\n=== AR24 keep-last-k ===")
    for k in (6, 9, 12, 15, 18, 21, 24):
        xk = keep_last_hours(x24, k)
        med = median_fde(model24, xk, y24, a24, device, batch_size=args.batch_size)
        keep_curve[k] = med
        print(f"  keep{k}h med FDE = {med:.2f}")

    fig_main = args.out_dir / "fig_ar_why_18h_diagnostics.png"
    fig_bias = args.out_dir / "fig_ar_forget_bias.png"
    meta = args.out_dir / "fig_ar_why_18h_meta.txt"
    make_diagnostics_figure(results, fig_main)
    make_forget_bias_figure(results, fig_bias)
    write_meta(results, keep_curve, meta, n)

    summary = {
        "n_samples": n,
        "results": results,
        "ar24_keep_last_k": keep_curve,
        "outputs": [str(fig_main), str(fig_bias), str(meta)],
    }
    (args.out_dir / "fig_ar_why_18h_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print("\nWrote:")
    print(f"  {fig_main}")
    print(f"  {fig_bias}")
    print(f"  {meta}")


if __name__ == "__main__":
    main()
