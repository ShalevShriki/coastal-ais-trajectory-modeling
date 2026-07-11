# AIS Vessel Trajectory Prediction: Learning Temporal Context for 12-Hour Forecasting

**Final Project for the Deep Learning Course (046211)**

[Student Name 1], [Student Name 2]

---

## Abstract

This project studies AIS vessel trajectory prediction as a **sequence modeling** problem. We ask how much past trajectory context is needed to forecast vessel positions **12 hours** ahead, and whether a model can learn soft preferences over multiple history scales. Using NOAA AIS data from US coastal waters, we build a filtered dataset of moving vessels, compare kinematic and neural baselines, and evaluate flat LSTM, Transformer, and autoregressive LSTM architectures. Completed experiments show that neural models reduce median final displacement error (FDE) from **106 km** (kinematic extrapolation) to **~18 km** (flat LSTM), while a preliminary 24-hour autoregressive model achieves similar performance. Experiments on context-length sweeps, receding-horizon sliding windows, and an adaptive multi-scale autoregressive model are in progress.

---

## 1. Introduction

Maritime traffic monitoring relies on the Automatic Identification System (AIS), which broadcasts vessel position, speed, and course at irregular intervals. Predicting where a vessel will be hours into the future supports collision avoidance, search-and-rescue planning, and port operations. We frame this task as **long-horizon sequence forecasting**: given a resampled history of AIS features, predict future latitude and longitude at 10-minute resolution.

The central research question is not only *which architecture works best*, but **how much temporal context** a sequence model needs for **12-hour** prediction, and whether different motion patterns (straight transit vs. maneuvering) benefit from different history lengths. Most prior trajectory work uses a fixed history window; we additionally propose an **adaptive multi-scale autoregressive RNN** that learns softmax weights over 9 h, 12 h, 18 h, and 24 h history suffixes.

This project connects directly to course topics: RNNs and LSTMs, autoregressive decoding, encoder–decoder structure, attention and Transformers, long-range dependencies, and interpretability through learned gating weights.

**Contributions (completed and planned).**

1. A reproducible NOAA-to-windows pipeline for US combined coasts with stationary and smart-motion filters.
2. A unified evaluation protocol: anchor-offset targets, Haversine FDE/ADE at 12 h, trajectory-level splits, and motion buckets.
3. Baseline comparison showing large gains over kinematic extrapolation (completed).
4. Fixed-context and adaptive-context autoregressive experiments (in progress).

---

## 2. Related Work

*(Brief — expand with citations before submission.)*

Vessel trajectory prediction is commonly approached with physics-based dead reckoning, Kalman filters, or machine learning over AIS sequences. Recurrent models compress history into a hidden state for multi-step forecasting; Transformers attend over the full history without fixed recurrence. Our work differs by **systematically varying context length** (9–24 h) and by learning **per-sample context weights** rather than assuming a single optimal window. Related maritime forecasting papers typically report ADE/FDE in kilometers over horizons from 30 minutes to several hours; our 12-hour horizon at 10-minute resolution is intentionally challenging.

---

## 3. Dataset and Preprocessing

### 3.1 Data source and geography

We use public NOAA AIS daily files (`ais-YYYY-MM-DD.csv.zst`) from the US Coast Guard archive. Raw data are processed per coastal region (West Coast, Eastern coast, Mexcany Beach), then merged into **USA Combined** for training.

### 3.2 Processing pipeline

The final pipeline used for all reported experiments:

```text
NOAA AIS (daily .csv.zst)
    ↓  per-coast incremental processing, segmentation, resampling
Per-coast model_ready_windows.parquet  (24 h history + 12 h future)
    ↓  scripts/combine_datasets.py
data/processed/combined/{train,val,test}.parquet
    ↓  scripts/apply_training_filters.py  (stationary filter, history-only)
data/processed/combined_filtered/
    ↓  smart-motion filter
data/processed/combined_filtered_smart/   ← PRIMARY TRAINING DATA
```

Each window row contains **144 history steps** (24 h @ 10 min) and **72 future steps** (12 h). History uses 15 features per step (`lat`, `lon`, `sog`, `cog_sin`, `cog_cos`, `heading_sin`, `heading_cos`, `heading_missing`, `dt_sec`, `dlat`, `dlon`, `dsog`, `dcog`, `v_north_kmh`, `v_east_kmh`), defined in `window_data.FEATURE_COLS`.

### 3.3 Stationary filter

Applied on **history only** (no future leakage). Windows are removed if the vessel barely moves:

| Rule | Threshold |
|------|-----------|
| Max confined radius | ≤ 0.5 km |
| Min displacement | ≥ 1.0 km |
| Min mean SOG | ≥ 0.5 kn |

### 3.4 Smart-motion filter

We keep trajectories with meaningful motion and drop near-stationary loops:

| Rule | Threshold |
|------|-----------|
| Min 16 h net displacement | ≥ 8 km |
| Min last 8 h net displacement | ≥ 2 km |
| Max history loop ratio | ≤ 0.35 |

**Rows retained** (`filter_report.json`):

| Split | Rows in | Rows kept | Fraction kept |
|-------|---------|-----------|---------------|
| train | 1,480,990 | 535,967 | 36.2% |
| val | 179,911 | 64,556 | 35.9% |
| test | 181,085 | 66,958 | 37.0% |

**Figure (Data section):** `data/results/USA Combined/unknown/smart_motion_audit/preview_kept_vs_rejected.png`

### 3.5 Train/validation/test split

- **Split by trajectory** (`traj_id`), not random rows — prevents leakage across overlapping windows from the same vessel track.
- Default fractions: 70% / 10% / 20% train / val / test, seed 42.
- Training subsample: **400,000** windows (seed 42) for all neural experiments reported here.
- Test set size at evaluation: **108,231** windows (full test split, not subsampled).

---

## 4. Task Formulation

| Setting | Value |
|---------|--------|
| Forecast horizon | 12 h (72 steps @ 10 min) |
| Full parquet window | 24 h history + 12 h future |
| Context sweep (planned) | 9 h, 12 h, 18 h, 24 h history suffix ending at anchor |
| Target parameterization | **Anchor-offset:** predict \(P_t - P_{\text{anchor}}\) (lat/lon degrees) |
| Primary metric | **FDE** — Haversine error at step 72 (12 h) |
| Secondary metrics | ADE (mean over 72 steps), nFDE, nADE, motion buckets |

For shorter history experiments, code takes the **last N hours** of the stored 24 h window; the anchor (last observed position) is unchanged (`window_data.resolve_window_hours()`).

---

## 5. Methods

### 5.1 Kinematic baseline

A non-neural **constant velocity** extrapolation: from the last observed position, speed (SOG), and course (COG), propagate the vessel forward for 12 h. Implemented in `window_data.kinematic_position_at_horizon()` and computed automatically at evaluation. This baseline tests whether neural models learn more than simple physics continuation.

### 5.2 Training loss and optimization

**Loss:** `TrajectoryLoss` (`models/training_utils.py`) — 50% Huber on anchor-offset deltas (δ = 0.01°) + 50% scaled geographic km error on reconstructed absolute positions. See `docs/trajectory_loss.md` for full equations.

**Evaluation** uses exact **Haversine** distance in km (train/eval mismatch is intentional for differentiability).

**Shared training settings (exp_clean B1, B2):**

| Hyperparameter | Value |
|----------------|--------|
| Optimizer | Adam, lr = 1×10⁻³ |
| Batch size | 256 (LSTM), 128 (Transformer) |
| Early stopping | patience = 10 on validation loss |
| Difficulty weighting | on (upweight maneuvering samples) |
| Curriculum / scheduled TF | **off** for B1/B2 (`--no-curriculum`) |

Autoregressive models (Section 5.4) use scheduled teacher forcing (0.3 → 0) and horizon curriculum (6 h → 12 h) unless noted.

### 5.3 Flat LSTM (direct multi-horizon)

**Architecture.** A 2-layer LSTM (hidden dim 256, dropout 0.2) encodes the full 24 h history. The final hidden state feeds a linear head that outputs **all 72 future (Δlat, Δlon) steps at once** — direct multi-horizon prediction.

**Config:** 24 h → 12 h. Run tag: `exp_clean/B1_flat`. Script: `models/RNN.py`.

**Trainable parameters:** 908,688.

### 5.4 Autoregressive LSTM (RNN_AR)

**Architecture.** Encoder LSTM reads history → initial decoder state; decoder LSTM predicts future **step by step** (autoregressive). Each step outputs an anchor-offset delta; absolute position is \( \hat{P}_t = P_{\text{anchor}} + \hat{y}_t \).

**Training.** Scheduled teacher forcing and horizon curriculum (6 h → 12 h) stabilize long rollouts.

**Context sweep (planned).** Same architecture trained with 9 h, 12 h, 18 h, and 24 h history suffixes → 12 h future. Run tags: `exp_final/AR_9h` … `AR_24h`. Script: `models/RNN_AR.py`.

**Preliminary result:** 24 h context, `v1/smart_motion` (with curriculum + TF): 1,631,618 parameters.

### 5.5 Transformer

**Architecture.** Transformer encoder over 24 h history (d_model = 128, 4 layers, 8 heads, FFN = 512); linear head predicts all 72 future steps (same direct multi-horizon setup as flat LSTM).

**Config:** 24 h → 12 h. Run tag: `exp_clean/B2_transformer`. Script: `models/transformers.py`.

**Trainable parameters:** 1,002,000.

### 5.6 Receding-horizon sliding window

*(Method description only — results pending.)*

Train a chunk model: **24 h history → 3 h displacement** (`models/RNN_recursive_1h.py`, `--chunk-hours 3`). At inference, roll forward **4 times** (3 h × 4 = 12 h): after each chunk, shift the history window and append synthetic AIS features for predicted positions. Run tag: `exp_final/sliding_3h`.

**Research question:** Does chunked rollout with error accumulation outperform direct 12 h autoregressive decoding?

### 5.7 Adaptive multi-scale autoregressive RNN

*(Method description only — results pending.)*

`AdaptiveMultiScaleARRNN` (`models/RNN_AR_adaptive.py`) encodes **four suffixes** (9 h, 12 h, 18 h, 24 h) with separate LSTM passes, applies a **softmax gating** layer to produce weights α₉, α₁₂, α₁₈, α₂₄, forms a weighted context vector, and autoregressively decodes 12 h. Per-test-sample α weights are saved to `context_alpha_weights.json` for interpretability analysis.

Run tag: `exp_final/adaptive_multiscale`.

---

## 6. Experiments and Results

### 6.1 Experimental setup

All completed neural runs use:

- **Dataset:** `data/processed/combined_filtered_smart/train.parquet`
- **Coast:** USA Combined
- **Train subsample:** 400,000 windows, seed 42
- **Split:** by trajectory
- **Evaluation:** full test set (108,231 windows), FDE @ 12 h

### 6.2 Baselines vs. neural models (completed)

**Table 1.** Test-set FDE @ 12 h (Haversine km), `combined_filtered_smart`.

| Model | History | Median FDE | Mean FDE | Mean ADE | Status |
|-------|---------|------------|----------|----------|--------|
| Kinematic baseline | — | **106.1** | 143.1 | — | ✓ auto at eval |
| Flat LSTM (B1) | 24 h | **18.1** | 33.3 | 17.4 | ✓ `exp_clean/B1_flat` |
| Transformer (B2) | 24 h | 20.3 | 35.5 | 18.6 | ✓ `exp_clean/B2_transformer` |
| AR LSTM | 24 h | 19.1* | 34.9* | 17.7* | ✓ *preliminary `v1/smart_motion`* |

Neural models reduce median FDE by **~5×** relative to kinematic extrapolation, confirming that 12-hour coastal vessel motion is not well approximated by constant SOG+COG alone.

**Stratified FDE (Flat LSTM B1):**

| Bucket | n (test) | Median FDE (km) |
|--------|----------|-----------------|
| maneuver | 102,727 | 17.3 |
| straight | 1,361 | 47.0 |
| anchored | 5,070 | 9.1 |
| other | 4,020 | 29.1 |

Straight-transit windows are rare after the smart-motion filter (~1.3% of test) but exhibit **higher** median error — consistent with the hypothesis that long, stable tracks may need different context than maneuvering traffic. This motivates the context-length and adaptive-weight experiments.

**Figures (Results):**

| Figure | Path |
|--------|------|
| Flat LSTM training curves | `exp_clean/B1_flat/RNN/lstm_training_history.png` |
| Flat LSTM scatter @ 12 h | `exp_clean/B1_flat/RNN/lstm_scatter.png` |
| Flat LSTM error histogram | `exp_clean/B1_flat/RNN/lstm_error_hist.png` |
| Transformer training curves | `exp_clean/B2_transformer/Transformer/transformer_training_history.png` |
| Transformer scatter @ 12 h | `exp_clean/B2_transformer/Transformer/transformer_scatter.png` |

Flat LSTM slightly outperforms Transformer on median FDE (18.1 vs. 20.3 km) with fewer parameters and ~4× faster training throughput on our GPU setup.

### 6.3 Context-length sweep (AR LSTM)

*(Title only — experiments running / pending.)*

- AR LSTM, 9 h history → 12 h future
- AR LSTM, 12 h history → 12 h future
- AR LSTM, 18 h history → 12 h future
- AR LSTM, 24 h history → 12 h future (`exp_final/AR_*`)

### 6.4 Direct vs. autoregressive vs. sliding rollout

*(Title only — pending.)*

- Flat LSTM vs. AR LSTM (24 h) — partial comparison in Table 1
- Receding-horizon sliding window (3 h × 4) vs. direct AR (`exp_final/sliding_3h`)

### 6.5 Adaptive multi-scale model and context-weight analysis

*(Title only — pending.)*

- Test FDE/ADE vs. fixed-context AR
- Distribution of α₉, α₁₂, α₁₈, α₂₄
- Correlation of α with motion features (straightness, COG variance, SOG change)

### 6.6 Negative ablation (appendix only)

Residual-naive AR (`v1_residual/smart_motion`) **hurt** performance vs. plain AR (~21.5 vs. ~19.1 km median FDE). We do not include it as a final model.

---

## 7. Discussion

*(Partial — update when `exp_final` completes.)*

**Completed findings.**

1. On moving-vessel US coastal AIS data, deep sequence models substantially outperform kinematic extrapolation at 12 h.
2. A flat LSTM that predicts all future steps in one forward pass is a strong reference (~18 km median FDE); Transformer attention does not improve on this metric in our setup.
3. The smart-motion filter retains ~36% of windows, enriching the dataset toward meaningful transit while dropping local loops and near-stationary patterns.
4. Maneuvering traffic dominates the test set; straight buckets show higher error, suggesting context-length effects are worth studying.

**Open questions** (to address in Sections 6.3–6.5):

- Does longer history consistently improve AR FDE, or does extra context add noise for simple tracks?
- Does chunked sliding inference degrade vs. direct AR due to error accumulation?
- Can the adaptive model assign higher weight to longer contexts for maneuvering samples?

---

## 8. Conclusion and Future Work

### 8.1 Conclusion

*(Partial — finalize after `exp_final`.)*

We built an end-to-end pipeline from NOAA AIS to 12-hour trajectory forecasting windows on US combined coasts, with filters that focus training on vessels with meaningful motion. Neural sequence models achieve roughly **18–20 km** median FDE at 12 hours, far below kinematic baselines. The flat LSTM and preliminary autoregressive LSTM perform similarly; the Transformer does not outperform the simpler recurrent encoder on median FDE.

### 8.2 Future work

*(Titles for planned extensions.)*

- Complete context-length sweep and adaptive multi-scale experiments
- Alpha-weight interpretability vs. motion features
- Horizon-wise error curves from saved trajectories
- Unified comparison script across all `exp_final` runs
- Optional: local AIS traffic density as an input feature

---

## 9. Reproducibility

| Component | Location |
|-----------|----------|
| Data loading, splits, metrics | `window_data.py` |
| Coast configs | `coast_paths.py` |
| Flat LSTM | `models/RNN.py` |
| AR LSTM | `models/RNN_AR.py` |
| Adaptive AR | `models/RNN_AR_adaptive.py` |
| Sliding chunk model | `models/RNN_recursive_1h.py` |
| Transformer | `models/transformers.py` |
| Loss, curriculum, TF | `models/training_utils.py` |
| Slurm jobs | `scripts/exp_final/`, `scripts/exp_clean/` |

See `README.md` for run commands and `REPORT_GUIDE.md` for figure paths and experiment status.

---

## 10. References

*(Add before submission.)*

1. [AIS trajectory forecasting — add citation]
2. [LSTM / sequence forecasting — add citation]
3. [Transformer for trajectories — add citation]
4. NOAA AIS data: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/`

---

## 11. Ethics Statement

*(Required course template — fill in student names and LLM sections before submission.)*

### 11.1 Introduction

**Student names:** [Student Name 1], [Student Name 2]

**Project title:** AIS Vessel Trajectory Prediction: Learning Temporal Context for 12-Hour Forecasting

This project develops deep learning models to predict vessel positions from historical AIS broadcasts. The goal is academic — to study sequence modeling and temporal context — not operational deployment. Predictions are approximate and must not be used as the sole basis for navigation or safety-critical decisions.

### 11.2 LLM-assisted stakeholder analysis

*(Complete using course-approved LLM; independent reflection in 11.3 must be your own work.)*

**2a. Stakeholders (3 types):**

- [To be completed]

**2b. Explanation to stakeholders:**

- [To be completed — 1 paragraph max]

**2c. Who is responsible for providing the explanation:**

- [To be completed — 1 paragraph max]

### 11.3 Reflection on AI output

*(Independent creative thinking — do not use LLMs for this section.)*

**3a. What needs to be added/changed in the LLM responses to make the explanation more ethical?**

- [To be completed — 1 paragraph max]

---

*Draft report generated from completed `exp_clean` baselines and project documentation. Sections marked pending await `exp_final` job chain completion.*
