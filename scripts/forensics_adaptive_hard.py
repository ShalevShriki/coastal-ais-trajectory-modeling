#!/usr/bin/env python3
"""Forensics: why separate+hard adaptive beats shared adaptive."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from proj.project.models.RNN_AR_adaptive import AdaptiveMultiScaleARRNN
from proj.project.models.RNN_AR_diff_encoder import CONTEXT_KEYS, DiffEncoderAdaptiveARRNN
from proj.project.window_data import (
    FEATURE_COLS,
    build_window_arrays,
    haversine_km,
    load_windows_filtered,
    make_train_val_test_frames,
)

OUT = PROJECT / "data/results/USA Combined/unknown/exp_coastal/report_figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = PROJECT / "data/processed/combined_filtered_smart_coastal/train.parquet"
CKPT_HARD = (
    PROJECT
    / "data/models/USA Combined/unknown/exp_coastal/adaptive_separate_encoders_hard/diff_encoder_adaptive_hard.pt"
)
CKPT_SHARED = (
    PROJECT / "data/models/USA Combined/unknown/exp_coastal/adaptive_multiscale/adaptive_ar.pt"
)


def forget_bias(lstm: torch.nn.LSTM) -> float:
    b = lstm.bias_ih_l0.detach().cpu().numpy()
    h = b.shape[0] // 4
    return float(b[h : 2 * h].mean())


def batch_fde(pred: np.ndarray, ya: np.ndarray, an: np.ndarray) -> np.ndarray:
    pred_abs = an[:, None, :] + pred
    return haversine_km(pred_abs[:, -1, 0], pred_abs[:, -1, 1], ya[:, -1, 0], ya[:, -1, 1])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    df = load_windows_filtered(DATA, sample_size=int(300000 / 0.7) + 1000, seed=42)
    _, _, test_df, _ = make_train_val_test_frames(
        df,
        test_fraction=0.2,
        val_fraction=0.1,
        seed=42,
        split_by="trajectory",
        train_sample_size=300000,
    )
    rng = np.random.default_rng(42)
    n = min(1200, len(test_df))
    pick = rng.choice(len(test_df), size=n, replace=False)
    sub = test_df.iloc[pick].reset_index(drop=True)
    x_raw, y_abs, y_delta, anchor = build_window_arrays(sub, history_steps=144, future_steps=72)

    sog_idx = FEATURE_COLS.index("sog")
    dcog_idx = FEATURE_COLS.index("dcog")
    mean_sog = x_raw[:, :, sog_idx].mean(1)
    max_dcog = np.abs(x_raw[:, :, dcog_idx]).max(1)
    lat = x_raw[:, :, 0]
    lon = x_raw[:, :, 1]
    dseg = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    straightness = np.clip(
        haversine_km(lat[:, 0], lon[:, 0], lat[:, -1], lon[:, -1]) / np.maximum(dseg.sum(1), 1e-3),
        0,
        1,
    )
    maneuver = max_dcog > 15
    straight = (mean_sog >= 5) & (max_dcog < 5)

    ckpt_h = torch.load(CKPT_HARD, map_location=device, weights_only=False)
    xh = (
        (x_raw - ckpt_h["scaler_mean"].reshape(1, 1, -1))
        / np.maximum(ckpt_h["scaler_scale"].reshape(1, 1, -1), 1e-6)
    ).astype(np.float32)
    model_h = DiffEncoderAdaptiveARRNN(len(FEATURE_COLS), 72, 256, 2, 0.2, "hard").to(device)
    model_h.load_state_dict(ckpt_h["model_state_dict"])
    model_h.eval()

    ckpt_s = torch.load(CKPT_SHARED, map_location=device, weights_only=False)
    xs = (
        (x_raw - ckpt_s["scaler_mean"].reshape(1, 1, -1))
        / np.maximum(ckpt_s["scaler_scale"].reshape(1, 1, -1), 1e-6)
    ).astype(np.float32)
    model_s = AdaptiveMultiScaleARRNN(
        len(FEATURE_COLS),
        72,
        ckpt_s.get("context_steps", [54, 72, 108, 144]),
        256,
        2,
        0.2,
        "lstm",
    ).to(device)
    model_s.load_state_dict(ckpt_s["model_state_dict"])
    model_s.eval()

    fb_hard = {k: forget_bias(model_h.encoders[k]) for k in CONTEXT_KEYS}
    fb_shared = forget_bias(model_s.encoder)
    enc_l2 = {
        k: float(
            torch.sqrt(
                sum(p.detach().float().pow(2).sum() for p in model_h.encoders[k].parameters())
            ).item()
        )
        for k in CONTEXT_KEYS
    }

    alpha_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    fde_h_rows: list[np.ndarray] = []
    fde_s_rows: list[np.ndarray] = []
    cos_h_rows: list[list[float]] = []
    cos_s_rows: list[list[float]] = []
    forced: dict[str, list[np.ndarray]] = {k: [] for k in CONTEXT_KEYS}

    with torch.no_grad():
        for i in range(0, n, 64):
            xb = torch.from_numpy(xh[i : i + 64]).to(device)
            xb2 = torch.from_numpy(xs[i : i + 64]).to(device)
            ya = y_abs[i : i + 64]
            an = anchor[i : i + 64]

            yh, ah, lh = model_h(xb, return_gate=True)
            ys, _ = model_s(xb2, return_alpha=True)
            alpha_rows.append(ah.cpu().numpy())
            logit_rows.append(lh.cpu().numpy())
            fde_h_rows.append(batch_fde(yh.cpu().numpy(), ya, an))
            fde_s_rows.append(batch_fde(ys.cpu().numpy(), ya, an))

            hl, cl, _ = model_h._encode_all(xb)
            vh = [h[-1] for h in hl]
            cos_h_rows.append(
                [
                    F.cosine_similarity(vh[0], vh[1], -1).mean().item(),
                    F.cosine_similarity(vh[1], vh[2], -1).mean().item(),
                    F.cosine_similarity(vh[2], vh[3], -1).mean().item(),
                ]
            )
            vs = []
            for ns in (54, 72, 108, 144):
                _, enc = model_s.encoder(xb2[:, -ns:, :])
                vs.append(enc[0][-1])
            cos_s_rows.append(
                [
                    F.cosine_similarity(vs[0], vs[1], -1).mean().item(),
                    F.cosine_similarity(vs[1], vs[2], -1).mean().item(),
                    F.cosine_similarity(vs[2], vs[3], -1).mean().item(),
                ]
            )

            for ki, key in enumerate(CONTEXT_KEYS):
                alpha = torch.zeros(xb.size(0), 4, device=device)
                alpha[:, ki] = 1.0
                mh, mc = model_h._mix_states(hl, cl, alpha)
                hidden = (mh, mc)
                dec = xb.new_zeros(xb.size(0), 1, 2)
                outs = []
                for _ in range(72):
                    o, hidden = model_h.decoder(dec, hidden)
                    d = model_h.output_proj(o[:, 0, :])
                    outs.append(d.unsqueeze(1))
                    dec = d.unsqueeze(1)
                forced[key].append(batch_fde(torch.cat(outs, 1).cpu().numpy(), ya, an))

    alpha_h = np.concatenate(alpha_rows)
    logits_h = np.concatenate(logit_rows)
    fde_h = np.concatenate(fde_h_rows)
    fde_s = np.concatenate(fde_s_rows)
    sel = alpha_h.argmax(1)
    soft_logits = torch.softmax(torch.from_numpy(logits_h), dim=-1).numpy()
    forced_arr = {k: np.concatenate(v) for k, v in forced.items()}
    forced_mat = np.stack([forced_arr[k] for k in CONTEXT_KEYS], axis=1)
    oracle = forced_mat.min(1)
    oracle_choice = forced_mat.argmin(1)
    delta = fde_h - fde_s
    cos_h = np.mean(np.asarray(cos_h_rows), axis=0)
    cos_s = np.mean(np.asarray(cos_s_rows), axis=0)

    torch.backends.cudnn.enabled = False
    g_hour = np.zeros(24)
    gate_grad_mass = np.zeros(4)
    for i in range(0, min(256, n), 32):
        xb = torch.from_numpy(xh[i : i + 32]).to(device).requires_grad_(True)
        yd = torch.from_numpy(y_delta[i : i + 32]).to(device)
        hl, cl, logits = model_h._encode_all(xb)
        logits.retain_grad()
        alpha = torch.softmax(logits, dim=-1)
        mh, mc = model_h._mix_states(hl, cl, alpha)
        hidden = (mh, mc)
        dec = xb.new_zeros(xb.size(0), 1, 2)
        outs = []
        for _ in range(72):
            o, hidden = model_h.decoder(dec, hidden)
            d = model_h.output_proj(o[:, 0, :])
            outs.append(d.unsqueeze(1))
            dec = d.unsqueeze(1).detach()
        torch.nn.functional.smooth_l1_loss(torch.cat(outs, 1), yd).backward()
        g = xb.grad.detach().abs().mean(0).mean(-1).cpu().numpy()
        for t, gt in enumerate(g):
            g_hour[min(23, t // 6)] += float(gt)
        gate_grad_mass += logits.grad.detach().abs().mean(0).cpu().numpy()
        model_h.zero_grad(set_to_none=True)
    g_hour = g_hour / (g_hour.sum() + 1e-12)
    gate_grad_mass = gate_grad_mass / (gate_grad_mass.sum() + 1e-12)

    # Plots
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(
        x - 0.2,
        [float(np.median(forced_arr[k])) for k in CONTEXT_KEYS],
        0.4,
        color="#4C78A8",
        label="Forced-context med FDE",
    )
    ax2 = ax.twinx()
    ax2.bar(
        x + 0.2,
        [100.0 * float((sel == i).mean()) for i in range(4)],
        0.4,
        color="#54A24B",
        alpha=0.75,
        label="Hard selection %",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(CONTEXT_KEYS))
    ax.set_ylabel("Median FDE (km)")
    ax2.set_ylabel("Selection %")
    ax.set_title("Hard model: forced-context FDE vs selection frequency")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adaptive_hard_forced_vs_selected.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    buckets = [("straight", straight), ("maneuver", maneuver), ("other", ~(straight | maneuver))]
    mat = np.zeros((3, 4))
    for bi, (_, mask) in enumerate(buckets):
        denom = max(int(mask.sum()), 1)
        for ki in range(4):
            mat[bi, ki] = 100.0 * float(((sel == ki) & mask).sum() / denom)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=50)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(CONTEXT_KEYS))
    ax.set_yticks(range(3))
    ax.set_yticklabels([b[0] for b in buckets])
    ax.set_title("Hard gate selection % within motion buckets")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adaptive_hard_selection_by_motion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(np.arange(24), 100 * g_hour, color="#B279A2")
    ax.set_xlabel("History hour (0=oldest … 23=newest)")
    ax.set_ylabel("% of |∂L/∂x| mass")
    ax.set_title("Hard separate-encoder: input gradient mass by history hour")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adaptive_hard_grad_by_hour.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = ["shared"] + list(CONTEXT_KEYS)
    vals = [fb_shared] + [fb_hard[k] for k in CONTEXT_KEYS]
    ax.bar(names, vals, color=["#4C78A8"] + ["#54A24B"] * 4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean forget-gate bias (L0)")
    ax.set_title("Forget bias: shared vs separate encoders")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adaptive_hard_forget_bias.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(list(CONTEXT_KEYS), [enc_l2[k] for k in CONTEXT_KEYS], color="#54A24B")
    ax.set_ylabel("Encoder parameter L2")
    ax.set_title("Separate encoders: weight magnitude (18h collapsed)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_adaptive_hard_encoder_l2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    spearman: dict[str, dict[str, float]] = {}
    for ki, key in enumerate(CONTEXT_KEYS):
        spearman[key] = {
            "mean_sog": float(stats.spearmanr(soft_logits[:, ki], mean_sog).statistic),
            "max_dcog": float(stats.spearmanr(soft_logits[:, ki], max_dcog).statistic),
            "straightness": float(stats.spearmanr(soft_logits[:, ki], straightness).statistic),
        }

    sel_motion: dict = {}
    for ki, key in enumerate(CONTEXT_KEYS):
        m = sel == ki
        if not m.any():
            sel_motion[key] = "never"
            continue
        sel_motion[key] = {
            "n": int(m.sum()),
            "mean_sog": float(mean_sog[m].mean()),
            "mean_max_dcog": float(max_dcog[m].mean()),
            "mean_straightness": float(straightness[m].mean()),
            "pct_maneuver": float(100 * maneuver[m].mean()),
            "pct_straight": float(100 * straight[m].mean()),
            "med_fde": float(np.median(fde_h[m])),
        }

    meta = {
        "n": n,
        "official_full_test": {
            "shared_med_fde": 20.19,
            "hard_med_fde": 19.08,
            "params_shared": 1895046,
            "params_hard": 4312710,
        },
        "sub_med_fde_hard": float(np.median(fde_h)),
        "sub_med_fde_shared": float(np.median(fde_s)),
        "pct_hard_better": float(100 * (delta < 0).mean()),
        "median_delta_fde_hard_minus_shared": float(np.median(delta)),
        "cos_shared": {
            "9_12": float(cos_s[0]),
            "12_18": float(cos_s[1]),
            "18_24": float(cos_s[2]),
        },
        "cos_hard": {
            "9_12": float(cos_h[0]),
            "12_18": float(cos_h[1]),
            "18_24": float(cos_h[2]),
        },
        "forget_bias_hard": fb_hard,
        "forget_bias_shared": fb_shared,
        "encoder_l2": enc_l2,
        "forced_med_fde": {k: float(np.median(forced_arr[k])) for k in CONTEXT_KEYS},
        "selection_pct": {CONTEXT_KEYS[i]: float(100 * (sel == i).mean()) for i in range(4)},
        "shared_argmax_pct_full": {"9h": 0.38, "12h": 1.61, "18h": 11.91, "24h": 86.10},
        "oracle_med_fde": float(np.median(oracle)),
        "gate_matches_oracle_pct": float(100 * (sel == oracle_choice).mean()),
        "oracle_choice_pct": {
            CONTEXT_KEYS[i]: float(100 * (oracle_choice == i).mean()) for i in range(4)
        },
        "grad_newest3_pct": float(100 * g_hour[-3:].sum()),
        "grad_oldest3_pct": float(100 * g_hour[:3].sum()),
        "gate_logit_grad_mass": {
            CONTEXT_KEYS[i]: float(gate_grad_mass[i]) for i in range(4)
        },
        "selection_vs_motion": sel_motion,
        "spearman": spearman,
        "selection_by_motion_pct": {
            buckets[bi][0]: {CONTEXT_KEYS[j]: float(mat[bi, j]) for j in range(4)}
            for bi in range(3)
        },
    }
    (OUT / "fig_adaptive_hard_forensics_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("Wrote forensics to", OUT)


if __name__ == "__main__":
    main()
