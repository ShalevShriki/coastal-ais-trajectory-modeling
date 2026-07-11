# Vessel Trajectory Prediction — Clean Experiment Plan (`exp_clean`)

> **Dataset:** `data/processed/combined_filtered_smart/train.parquet`  
> **Task:** 24h history → 12h future (72 steps @ 10 min)  
> **Split:** by trajectory, seed 42, 400k train subsample  
> **Run tag prefix:** `exp_clean/`  
> **Legacy runs** (`ar_exp/`, `v1/smart_motion/`, etc.) are kept but **not** used for this comparison.

---

## Configuration legend

| Flag | Meaning |
|------|---------|
| **TC** | Scheduled teacher forcing (0.3→0) + horizon curriculum (6h→12h train loss) |
| **Residual** | `--residual-naive` — predict correction over kinematic baseline |
| **Target** | `anchor_offset` (P_t−P_0) or `step_delta` (P_t−P_{t−1}) + cumsum at eval |

**Baselines (B1, B2):** no residual, no TC (`--no-curriculum`).

**Kinematic & naive baselines** are computed automatically at eval (not separate jobs).

---

## Experiment matrix

| ID | Run tag | Model | Residual | TC | Target | Purpose |
|----|---------|-------|----------|-----|--------|---------|
| **B1** | `B1_flat` | Flat LSTM | No | No | anchor | Direct multi-step neural baseline |
| **B2** | `B2_transformer` | Transformer | No | No | anchor | Attention baseline |
| **A0** | `A0_ar_anchor_no_tc` | RNN_AR | No | No | anchor | Pure AR anchor-offset |
| **A1** | `A1_ar_anchor_tc` | RNN_AR | No | Yes | anchor | TC ablation (+ vs A0) |
| **A2** | `A2_ar_anchor_residual` | RNN_AR | Yes | No | anchor | Residual ablation (+ vs A0) |
| **M1** | `M1_ar_anchor_res_tc` | RNN_AR | Yes | Yes | anchor | Full anchor AR (main) |
| **M2** | `M2_ar_step_delta_res_tc` | RNN_AR | Yes | Yes | step_delta | Anchor vs step-delta (vs M1) |
| **M3** | `M3_ar_sliding_3h_res` | Sliding RNN | Yes | No* | 3h chunk | Internal AR vs external recursive (vs M2) |

\*M3 is external recursive rollout — scheduled TF/curriculum do not apply the same way as internal AR decoder. Training is single 3h-chunk displacement; eval is **4 recursive calls** (3h×4=12h).

---

## M3 — Sliding window 3h

- **Train:** predict displacement over next **3 hours** (18 steps) from current 24h window  
- **Inference:** 4 recursive rollouts:

```text
input_0: real history [-24h, 0h]  → predict [0h, 3h]
input_1: shifted + synthetic [0h,3h] → predict [3h, 6h]
input_2: …                          → predict [6h, 9h]
input_3: …                          → predict [9h, 12h]
```

- Synthetic AIS features appended each chunk (`append_synthetic_hour_to_history`)  
- Residual naive recomputed **per chunk** from current history at inference

---

## Controlled comparisons

| Compare | Question |
|---------|----------|
| B1 vs A1 | Direct vs autoregressive (no residual) |
| A0 vs A1 | Effect of TC on anchor AR |
| A0 vs A2 | Effect of residual (no TC) |
| M1 vs M2 | Anchor-offset vs step-delta (both res+TC) |
| M2 vs M3 | Internal AR decoder vs external 3h sliding window |
| B2 vs B1 vs M1 | Attention vs flat vs best AR |

---

## Metrics (all runs)

- median/mean **FDE**, **ADE**, **nFDE** @ 12h  
- Buckets: **straight**, **maneuver**  
- Do not select winner on median FDE alone — check mean FDE and maneuver bucket

---

## Commands

```bash
# Submit full serial suite (cancels legacy ar_exp jobs)
bash scripts/exp_clean/submit_all.sh

# Compare when complete
python scripts/compare_exp_clean.py
```

## Result paths

```text
data/results/USA Combined/unknown/exp_clean/<ID>_<name>/
LOG/exp_clean_*.out
```

---

## Notes from prior smart_motion runs (hypotheses, not conclusions)

- Residual + TC **hurt** AR vs no-residual on median FDE (~21.5 vs ~19.1 km) — A2/M1 will test this cleanly  
- Flat RNN ≈ Transformer ≈ 20 km median on moving-vessel filter  
- Training loss with TC is **not comparable** across epochs — use validation curve (see two-panel training plots)
