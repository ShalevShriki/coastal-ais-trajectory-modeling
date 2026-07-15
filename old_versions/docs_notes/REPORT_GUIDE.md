# Report Guide — AIS Trajectory Prediction

Reference for writing the final report: **models**, **data processing (final pipeline)**, **files**, **finished experiments**, and **figures** you can use today.

Companion docs:
- [project_research.md](project_research.md) — research questions and experiment design
- [docs/trajectory_loss.md](docs/trajectory_loss.md) — training loss (equations)
- [README.md](README.md) — how to run code and Slurm jobs

---

## 1. One-sentence framing (for abstract / intro)

> This project studies AIS vessel trajectory prediction as a **sequence modeling** problem, focusing on **how much temporal context** is needed for **12-hour** forecasting. We compare fixed-context autoregressive RNNs (9–24h history), a receding-horizon sliding-window model, a Transformer, and an **adaptive multi-scale AR** model that learns soft weights over several history windows.

---

## 2. Final prediction task

| Setting | Value |
|---------|--------|
| **Forecast horizon** | 12 hours (72 steps @ 10 min) |
| **Full parquet window** | 24h history + 12h future (144 + 72 steps) |
| **Context sweep** | 9h, 12h, 18h, 24h history (suffix ending at anchor) |
| **Target** | Anchor-offset: predict `P_t − P_anchor` (lat/lon degrees) |
| **Geography** | USA Combined (East + West + Gulf coasts merged) |
| **Train subsample** | 400,000 windows, seed 42 |
| **Split** | By trajectory (`traj_id`), not random rows |

---

## 3. Data processing (final pipeline only)

What the report should describe — the path actually used for final experiments.

```text
NOAA AIS (daily .csv.zst)
    ↓
Per-coast incremental processing (West / East / Mexcany Beach)
    ↓  segment + resample + sliding windows
Per-coast model_ready_windows.parquet  (24h → 12h)
    ↓
scripts/combine_datasets.py
    ↓
data/processed/combined/{train,val,test}.parquet
    ↓
scripts/apply_training_filters.py   (stationary / history-only filter)
    ↓
data/processed/combined_filtered/
    ↓
Smart-motion filter (scripts/smart_motion_filter.sbatch)
    ↓
data/processed/combined_filtered_smart/   ← PRIMARY TRAINING DATA
```

### 3.1 Stationary filter (`combined_filtered`)

Applied on **history only** (no future leakage). Removes windows where the vessel barely moves.

| Rule | Threshold |
|------|-----------|
| Max confined radius | ≤ 0.5 km |
| Min displacement | ≥ 1.0 km |
| Min mean SOG | ≥ 0.5 kn |

Script: `scripts/apply_training_filters.py`

### 3.2 Smart-motion filter (`combined_filtered_smart`)

Keeps trajectories with **meaningful motion** and drops small local loops / near-stationary patterns.

| Rule | Threshold |
|------|-----------|
| Min 16h net displacement | ≥ 8 km |
| Min last 8h net displacement | ≥ 2 km |
| Max history loop ratio | ≤ 0.35 |

| Split | Rows in | Rows kept | Fraction kept |
|-------|---------|-----------|---------------|
| train | 1,480,990 | **535,967** | 36.2% |
| val | 179,911 | 64,556 | 35.9% |
| test | 181,085 | 66,958 | 37.0% |

Report file: `data/processed/combined_filtered_smart/filter_report.json`

Audit figures (good for **Data** section):
- `data/results/USA Combined/unknown/smart_motion_audit/preview_kept_vs_rejected.png`
- `data/results/USA Combined/unknown/smart_motion_audit/audit_smart_motion_100_grid.png`

### 3.3 Window features (inputs)

15 features per history step: `lat`, `lon`, `sog`, `cog_sin`, `cog_cos`, `heading_sin`, `heading_cos`, `heading_missing`, `dt_sec`, `dlat`, `dlon`, `dsog`, `dcog`, `v_north_kmh`, `v_east_kmh`.

Defined in `window_data.py` → `FEATURE_COLS`.

### 3.4 Variable context slicing

For experiments with shorter history, code takes the **last N hours** of the 24h window (anchor unchanged). Implemented in `window_data.resolve_window_hours()` and `build_window_arrays()`.

---

## 4. Models in the final report

### 4.1 Baseline — kinematic constant velocity

**Not a neural model.** Extrapolates last position using constant SOG + COG for 12h.

- Computed automatically at evaluation in every training script
- Label in metrics: `"Kinematic baseline: constant SOG+COG"`
- Code: `window_data.kinematic_position_at_horizon()`

**Report role:** Shows neural models beat naive physics extrapolation on 12h.

---

### 4.2 Flat LSTM (`models/RNN.py`)

| Item | Detail |
|------|--------|
| **Idea** | Encode full history → single LSTM → linear head outputs **all 72 future steps at once** |
| **Config for report** | 24h history → 12h future |
| **Run tag** | `exp_clean/B1_flat` |
| **Script** | `scripts/exp_clean/train_B1_flat.sbatch` |

**Report role:** Reference for “direct multi-horizon” vs autoregressive.

---

### 4.3 Fixed-context RNN_AR (`models/RNN_AR.py`)

| Item | Detail |
|------|--------|
| **Idea** | LSTM **encoder** reads history → LSTM **decoder** predicts future **step by step** (autoregressive) |
| **Target** | Anchor-offset deltas |
| **Training** | Teacher forcing 0.3→0, horizon curriculum 6h→12h |
| **Context sweep** | 9h, 12h, 18h, 24h → 12h future |
| **Run tags** | `exp_final/AR_9h`, `AR_12h`, `AR_18h`, `AR_24h` |

**Report role:** Main experiment — “how much history helps?”

---

### 4.4 Receding-horizon sliding window (`models/RNN_recursive_1h.py`)

| Item | Detail |
|------|--------|
| **Idea** | Train **24h → 3h** chunk displacement; at inference **roll forward 4 times** (3h×4 = 12h) |
| **Window update** | After each chunk, shift history; synthetic AIS features appended |
| **Run tag** | `exp_final/sliding_3h` |
| **CLI** | `--chunk-hours 3 --horizon-hours 12` |

**Report role:** Compare direct 12h AR vs chunked rollout with error accumulation.

---

### 4.5 Transformer (`models/transformers.py`)

| Item | Detail |
|------|--------|
| **Idea** | Self-attention over full 24h history → predict 12h future |
| **Config** | 24h → 12h, d_model=128, 4 layers, 8 heads |
| **Run tag** | `exp_clean/B2_transformer` |

**Report role:** Attention vs recurrent memory compression.

---

### 4.6 Adaptive multi-scale AR (`models/RNN_AR_adaptive.py`) — main contribution

| Item | Detail |
|------|--------|
| **Idea** | Encode **9h, 12h, 18h, 24h** suffixes separately → **softmax gating** → weighted context → AR decoder → 12h |
| **Interpretability** | Saves per-sample weights α₉, α₁₂, α₁₈, α₂₄ |
| **Run tag** | `exp_final/adaptive_multiscale` |
| **Output** | `context_alpha_weights.json` |

**Report role:** Can the model learn useful context length? Correlate α with straightness / COG variance (planned analysis).

---

### 4.7 Deliberately excluded from final report

| Item | Why |
|------|-----|
| **Residual-naive AR** | Tested (`v1_residual/smart_motion`); hurt FDE vs plain AR — dropped |
| **Step-delta targets** | Exploratory only (`exp_clean` M2 — not run) |
| **6h-future context sweep** | Superseded by 12h-future plan |
| **v1/v2 experiment1 on `combined_filtered`** | Pre-smart-motion; use only as appendix if needed |

---

## 5. Training & evaluation (methods section)

### Loss

`TrajectoryLoss` in `models/training_utils.py` — 50% Huber on anchor-offset deltas + 50% scaled geographic km error. See [docs/trajectory_loss.md](docs/trajectory_loss.md).

### Metrics (report these)

| Metric | Meaning |
|--------|---------|
| **FDE** | Haversine error at **12h** (final step) — primary |
| **ADE** | Mean Haversine error over all 72 future steps |
| **nFDE / nADE** | Error normalized by true path length |
| **Buckets** | straight / maneuver / anchored / other (from history features) |

### Horizon-wise error

Available in some older visualization folders (`error_vs_horizon.png`). Can be regenerated from saved trajectories in `*_sample_trajectories.json`.

---

## 6. Experiment status

### ✅ Finished — use in report now

| Experiment | Run tag | Median FDE @ 12h | Primary result folder |
|------------|---------|------------------|------------------------|
| **Kinematic baseline** | (in metrics below) | **~106 km** | — |
| **Flat LSTM 24h→12h** | `exp_clean/B1_flat` | **~18.1 km** | `data/results/USA Combined/unknown/exp_clean/B1_flat/RNN/` |
| **Transformer 24h→12h** | `exp_clean/B2_transformer` | **~20.3 km** | `.../exp_clean/B2_transformer/Transformer/` |
| **Flat LSTM** (confirmatory) | `v1/smart_motion` | ~20.0 km | `.../v1/smart_motion/RNN/` |
| **AR LSTM 24h** (placeholder until `exp_final/AR_24h`) | `v1/smart_motion` | ~19.1 km | `.../v1/smart_motion/RNN_AR_LSTM/` |
| **Transformer** (confirmatory) | `v1/smart_motion` | ~20.0 km | `.../v1/smart_motion/Transformer/` |
| **AR + residual** (negative result) | `v1_residual/smart_motion` | ~21.5 km (worse) | appendix only |

**Recommendation:** Main results table → **`exp_clean` B1 + B2** on `combined_filtered_smart`. Use **`v1/smart_motion` AR** as preliminary 24h AR until `exp_final/AR_24h` completes.

### 🔄 Running (`exp_final` serial chain)

| Job | Experiment | Status |
|-----|------------|--------|
| `ef_AR9` | AR 9h → 12h | Running |
| `ef_AR12` | AR 12h → 12h | Pending |
| `ef_AR18` | AR 18h → 12h | Pending |
| `ef_AR24` | AR 24h → 12h | Pending |
| `ef_slide3` | Sliding 3h × 4 | Pending |
| `ef_adapt` | Adaptive multi-scale | Pending |

Submit script: `scripts/exp_final/submit_all.sh`  
Logs: `LOG/exp_final_*.out`

### ⏳ Not started / post-training analysis

- Alpha vs motion-feature correlation (needs adaptive run)
- Local AIS density feature (optional, not implemented)
- Unified comparison table script for `exp_final` (can run `scripts/compare_exp_clean.py` pattern when done)

---

## 7. Figures & visualizations for the report

### 7.1 Per-model outputs (every finished training run)

Each completed run typically has:

| File | Use in report |
|------|----------------|
| `*_training_history.png` | Training / validation loss curves (two-panel: val + train objective) |
| `*_scatter.png` | Predicted vs true lat/lon at 12h |
| `*_error_hist.png` | FDE distribution |
| `*_metrics.json` | All numbers for tables |
| `*_sample_trajectories.json` | Map overlays / qualitative examples |

Regenerate training plot:
```bash
python scripts/plot_training_history.py --metrics path/to/*_metrics.json
```

### 7.2 Best figures for **main report** (ready now)

| Figure | Path | Section |
|--------|------|---------|
| Flat LSTM training | `exp_clean/B1_flat/RNN/lstm_training_history.png` | Methods / Results |
| Flat LSTM scatter @ 12h | `exp_clean/B1_flat/RNN/lstm_scatter.png` | Results |
| Flat LSTM error hist | `exp_clean/B1_flat/RNN/lstm_error_hist.png` | Results |
| Transformer training | `exp_clean/B2_transformer/Transformer/transformer_training_history.png` | Results |
| Transformer scatter | `exp_clean/B2_transformer/Transformer/transformer_scatter.png` | Results |
| Smart-filter kept vs rejected | `smart_motion_audit/preview_kept_vs_rejected.png` | **Data** |
| Track map (segments) | `track_maps/png_good_segments_random.png` | Data / qualitative |
| Horizon error curve (example) | `visualizations/error_vs_horizon.png` | Results (error vs time) |
| Multi-model comparison bars | `comparison/comparison_metrics_bar.png` | Results (if models match) |

### 7.3 Good for **appendix** or exploratory

| Folder | Contents |
|--------|----------|
| `v1/experiment1/visualizations/` | Loss grids, random vessel maps (Feb data, pre-smart-motion) |
| `v1/smart_motion/visualizations/` | Loss curves, random vessels on smart-motion |
| `v1_residual/smart_motion/` | Shows residual hurt — 1 paragraph + 1 table row |
| `comparison/` | Cross-run comparison plots (check which runs they include) |

### 7.4 Figures expected after `exp_final` completes

| Run | New figures |
|-----|-------------|
| `exp_final/AR_*` | `lstm_ar_training_history.png`, scatter, error hist per context |
| `exp_final/sliding_3h` | Recursive sliding metrics + training history |
| `exp_final/adaptive_multiscale` | `adaptive_ar_*.png` + **`context_alpha_weights.json`** for α analysis |

---

## 8. Suggested results table (copy into report)

**Dataset:** `combined_filtered_smart`, test set, FDE @ 12h (Haversine km)

| Model | History | Median FDE | Mean FDE | Status |
|-------|---------|------------|----------|--------|
| Kinematic baseline | — | 106.1 | 143.1 | ✅ |
| Flat LSTM | 24h | **18.1** | 33.3 | ✅ `exp_clean/B1` |
| Transformer | 24h | 20.3 | 35.5 | ✅ `exp_clean/B2` |
| AR LSTM | 24h | 19.1* | 35.0* | ✅ *v1/smart_motion; replace with `exp_final/AR_24h` |
| AR LSTM | 18h | — | — | 🔄 pending |
| AR LSTM | 12h | — | — | 🔄 pending |
| AR LSTM | 9h | — | — | 🔄 pending |
| Sliding 3h×4 | 24h | — | — | 🔄 pending |
| Adaptive multi-scale | 9+12+18+24h | — | — | 🔄 pending |

**Bucket example (Flat LSTM, B1):** maneuver n≈103k → median FDE **17.3 km**; straight n≈1.4k → median FDE **47.0 km**.

---

## 9. Key code files (for methods / reproducibility)

| File | Role |
|------|------|
| `window_data.py` | Load windows, filters, splits, `build_window_arrays`, metrics |
| `coast_paths.py` | Coast configs, result path helpers |
| `models/RNN.py` | Flat LSTM |
| `models/RNN_AR.py` | Autoregressive LSTM |
| `models/RNN_AR_adaptive.py` | Adaptive multi-scale AR |
| `models/RNN_recursive_1h.py` | Sliding-window chunk model |
| `models/transformers.py` | Transformer |
| `models/training_utils.py` | `TrajectoryLoss`, curriculum, teacher forcing |
| `models/plot_utils.py` | Two-panel training history plots |
| `scripts/combine_datasets.py` | Merge coasts |
| `scripts/apply_training_filters.py` | Stationary filter |
| `scripts/exp_final/*.sbatch` | Final experiment Slurm jobs |
| `scripts/plot_training_history.py` | Regenerate training plots |
| `scripts/map_filtered_tracks.py` | Track map figures |
| `scripts/audit_smart_motion_filter.py` | Filter audit |

---

## 10. Report outline (mapped to what you have)

| Section | Write now? | Source material |
|---------|------------|-----------------|
| 1. Introduction | ✅ | `project_research.md` §1, §19–20 |
| 2. Related work / course link | ✅ | RNN, AR, attention, context length |
| 3. Data | ✅ | §3 above + filter_report + audit PNGs |
| 4. Task formulation | ✅ | 24h/12h windows, anchor offsets |
| 5. Methods — baselines | ✅ | Kinematic |
| 6. Methods — Flat LSTM | ✅ | `RNN.py`, B1 results |
| 7. Methods — RNN_AR | ✅ partial | Architecture now; full sweep when `exp_final` done |
| 8. Methods — Sliding window | ✅ text | Results pending |
| 9. Methods — Transformer | ✅ | B2 results |
| 10. Methods — Adaptive AR | ✅ | `RNN_AR_adaptive.py`; results pending |
| 11. Methods — Loss & metrics | ✅ | `docs/trajectory_loss.md` |
| 12. Results — baselines vs neural | ✅ | Table §8, B1/B2 figures |
| 13. Results — context-length sweep | ⏳ | `exp_final/AR_*` |
| 14. Results — sliding vs direct | ⏳ | `exp_final/sliding_3h` |
| 15. Results — adaptive + α analysis | ⏳ | adaptive run + feature script |
| 16. Discussion | partial | Straight bucket worse; residual failed |
| 17. Conclusion | partial | Update when `exp_final` done |

---

## 11. Quick checklist before submitting report

- [ ] State dataset: **`combined_filtered_smart`**, not raw NOAA or unfiltered combined
- [ ] State task: **12h forecast**, 10 min resolution
- [ ] Include kinematic baseline (shows why DL is needed)
- [ ] Use **`exp_clean/B1` and `B2`** as primary flat + Transformer numbers
- [ ] Mention smart-motion filter with **~536k train windows (36% kept)**
- [ ] Include at least one **training curve** + one **scatter/error hist**
- [ ] Include **data figure** (filter preview or track map)
- [ ] When `exp_final` finishes: add context-length table + adaptive α plot
- [ ] Do **not** present residual AR as a final model (negative ablation only)

---

## 12. Reference PDF

Course / style reference: `GrossAhead report.pdf` (example report structure in repo root).

---

*Last updated: aligned with `exp_final` job chain and `exp_clean` B1/B2 completions on `combined_filtered_smart`.*
