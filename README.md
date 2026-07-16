# AIS Vessel Trajectory Prediction

Technion course **046211 — Deep Learning** final project.

Authors: **Amitai Gal**, **Shalev Shiriki**

**GitHub (public):** https://github.com/ShalevShriki/coastal-ais-trajectory-modeling

We predict **12-hour vessel trajectories** from Automatic Identification System (AIS) history, and study a core sequence-learning question: **how much temporal context is useful**, and whether a more complex adaptive model uses long history better than simpler fixed-context models.

**Submitted report:** [`report/AIS_report.pdf`](report/AIS_report.pdf) · TeX source [`report/AIS_report.tex`](report/AIS_report.tex)

**Canonical experiment suite:** `exp_coastal` — USA Combined coastal windows after inland filtering (~363k examples), land penalty λ = 0.1.

**Moodle code ZIP:** see [`SUBMISSION.md`](SUBMISSION.md) (code only; no datasets). Pack with `bash scripts/pack_moodle_zip.sh`.  
**Exact train args for every report experiment:** [`EXPERIMENTS.md`](EXPERIMENTS.md) · `bash scripts/exp_coastal/reproduce_experiments.sh`

---

## Table of contents

1. [Quick facts](#1-quick-facts)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Task, windows, and metrics](#4-task-windows-and-metrics)
5. [Data pipeline (from scratch)](#5-data-pipeline-from-scratch)
6. [Training (`exp_coastal`)](#6-training-exp_coastal)
7. [Models](#7-models)
8. [Main results (report)](#8-main-results-report)
9. [Figures, maps, and analysis](#9-figures-maps-and-analysis)
10. [Key file reference](#10-key-file-reference)
11. [Archived material](#11-archived-material)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Quick facts

| Item | Value |
|------|--------|
| Input history | up to **24 h** (144 × 10 min); AR sweeps also use 9 / 12 / 18 h suffixes |
| Prediction horizon | **12 h** (72 × 10 min) |
| Features / step | **15** (lat, lon, SOG, COG sin/cos, heading sin/cos + missing, Δt, Δlat/Δlon, ΔSOG, ΔCOG, v_north/v_east) |
| Target | lat/lon **offsets from the last observed anchor** |
| Dataset | `data/processed/combined_filtered_smart_coastal/train.parquet` (**363,014** windows) |
| Split | by **trajectory ID** (~255k / 35k / 73k train/val/test after sampling rules in training) |
| Shared train setup | LSTM 256×2, dropout 0.2; batch 256 (Transformer 128); lr 1e-3; GPU A6000 |
| Best median FDE | **Flat LSTM 24h → 18.80 km** (kinematic baseline 102.48 km) |

Data source: NOAA AIS  
`https://coast.noaa.gov/htdata/CMSP/AISDataHandler/<year>/AIS_YYYY_MM_DD.zip` / `.csv.zst`

---

## 2. Repository layout

```
project/
├── README.md                 ← this file
├── SUBMISSION.md             ← Moodle ZIP checklist
├── requirements.txt / environment.yml
├── data_urls.example.json    ← paste Drive links → data_urls.json
├── report/                   ← final PDF + TeX + LyX helper markdown
├── models/                   ← trainers (+ RNN_AR_diff_encoder.py separate encoders)
├── processing/               ← NOAA download / clean / segment / windows
├── scripts/
│   ├── exp_coastal/          ← Slurm jobs for the report suite
│   ├── download_processed_data.py  ← fetch coastal parquet from Drive
│   ├── pack_moodle_zip.sh    ← build code-only Moodle ZIP
│   ├── combine_datasets.py
│   ├── apply_training_filters.py
│   ├── filter_inland_windows.py
│   ├── generate_report_figures.py
│   ├── diagnose_ar_context.py ← why AR 18h > 24h (occlusion/grads/h)
│   ├── compare_adaptive_separate_gates.py
│   ├── forensics_adaptive_hard.py ← separate+hard vs shared adaptive
│   ├── analyze_adaptive_*.py
│   ├── plot_ar*_map.py       ← Folium HTML maps
│   └── …
├── window_data.py            ← load / split / scale windows
├── coast_paths.py            ← coastal configs & path helpers
├── coast_frame.py
├── AIS_AUDIT.py              ← data quality audit
├── VISUALIZE.py / EXPORT_COAST_DATA.py
├── data/
│   ├── processed/            ← segments, filters, coastal parquet, land grid
│   ├── models/.../exp_coastal/   ← checkpoints
│   └── results/.../exp_coastal/  ← metrics, plots, report_figures/
├── LOG/                      ← Slurm stdout (created by jobs)
└── old_versions/             ← drafts & earlier experiment suites (not deleted)
```

---

## 3. Setup

```bash
# After unzipping the Moodle package (or cloning GitHub), PYTHONPATH = parent of `proj/`
# Moodle ZIP layout:  <zip_root>/proj/project/...
export PYTHONPATH=/path/to/zip_root          # directory that contains `proj/`
cd /path/to/zip_root/proj/project

pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate coastal-ais-trajectory
```

Dependencies: see `requirements.txt` / `environment.yml` (`pandas`, `numpy`, `torch`, `scikit-learn`, `pyarrow`, `matplotlib`, `folium`, `contextily`, `pyproj`, `requests`, `zstandard`, `global-land-mask`, `scipy`, `gdown`).

### 3.1 Get the processed coastal dataset (required for training)

The Moodle ZIP / GitHub code tree does **not** ship the ~2.8 GB training parquet. Two options:

**A. Recommended — download our processed coastal artifacts**

1. Authors host `train.parquet` + `land_grid_us.npz` on Google Drive (links already in `data_urls.example.json`).
2. Graders / reproducers:

```bash
python scripts/download_processed_data.py
# uses data_urls.json if present, else data_urls.example.json
```

Drive links:

- parquet: https://drive.google.com/file/d/1Avt0LDK9LAhMmdhULeHwZdKbXXi6-7vy/view?usp=sharing
- land grid: https://drive.google.com/file/d/1aXP3c_M4eAN16I5ltPTreEfnbUfC1_s8/view?usp=sharing

This writes:

- `data/processed/combined_filtered_smart_coastal/train.parquet`
- `data/processed/land_grid_us.npz`

**B. Rebuild from public NOAA AIS** (multi-day) — see [§5](#5-data-pipeline-from-scratch).

### 3.2 Cluster paths (Technion Slurm)

Slurm scripts source `scripts/exp_coastal/_env.sh`. On a fresh machine, set `PROJECT` / `SUBROOT` to your clone, or run the `python -u models/...` commands from §6.3 directly. Defaults used by `exp_coastal`:

```bash
DATA=data/processed/combined_filtered_smart_coastal/train.parquet
SAMPLE=300000
RUN_PREFIX=exp_coastal
FUTURE_H=12
HORIZON_H=12
LAND_PENALTY=0.1
```

---

## 4. Task, windows, and metrics

Each row in the parquet is one sliding window:

- **History columns:** `x_t000_<feature>` … `x_t143_<feature>` (24 h @ 10 min)
- **Future targets:** `y_t000_lat`, `y_t000_lon`, … through `y_t071_*` (12 h)
- Models predict **offsets** from the last observed position \(P_0\); reported lat/lon are \(P_0 + \hat y\).

**Distance:** Haversine kilometers.

| Metric | Definition |
|--------|------------|
| **FDE** | Haversine error at the **last** future step (12 h) |
| **ADE** | Mean Haversine error over all 72 future steps |

Ranking in the report uses **median FDE** (errors are heavy-tailed). We also report mean FDE and median ADE.

Train/val/test are split by **`traj_id`** (fallback: MMSI) to avoid leakage across overlapping windows of the same voyage.

### Motion buckets (straight vs maneuver)

Used **only for analysis / stratified metrics** (e.g. history-length vs motion type in the report). **Not** a training label. Features come from the **observed history only** (no future / label leakage).

Canonical implementation: `window_data.classify_window_motion` / `motion_bucket_masks_df`.

| Bucket | Rule (history window) |
|--------|------------------------|
| **Anchored** | mean SOG \< 1 kn |
| **Maneuver** | max \|ΔCOG\| \> **15°** |
| **Straight** | mean SOG ≥ **5 kn** and max \|ΔCOG\| \< **5°**, and not anchored |
| **Other** | everything else (ambiguous; excluded from the straight-vs-maneuver comparison) |

The report also describes path **straightness** \(d_{\mathrm{direct}} / L_{\mathrm{path}}\) as intuition; the **code thresholds above** are what actually define the metric buckets.

**Separate heuristic:** `scripts/generate_report_figures.py` picks example tracks for Folium/gallery maps with path directness + turn distance. That is only for visualization diversity, not the same as the metric buckets.

Optional training-time oversampling (`--maneuver-fraction`, `--straight-fraction`, or `--no-maneuver-oversample`) can bias which windows enter the train subsample; `exp_coastal` runs use `--no-maneuver-oversample`.

---

## 5. Data pipeline (from scratch)

Prefer [§3.1](#31-get-the-processed-coastal-dataset-required-for-training) unless you need to regenerate windows.

Run steps in order. Already-processed data for the report lives under `data/processed/` (coasts + `combined_filtered_smart` + `combined_filtered_smart_coastal`).

### 5.1 Download, clean, segment, build windows (per coast)

Configured regions in `coast_paths.py`:

- **West Coast**
- **Eastern coast**
- **Mexcany Beach** (Mexican Pacific + Gulf)

```bash
python "processing/INCREMENTAL_PROCESS West Coast.py"
python "processing/INCREMENTAL_PROCESS Eastern coast.py"
python "processing/INCREMENTAL_PROCESS Mexcany Beach.py"
```

Outputs (per coast), roughly:

- `data/processed/days/<region>/YYYY-MM-DD.parquet` — daily cleaned frames  
- `data/processed/<Coast>/ais_<region>_long_horizon/coastal_segments.parquet`  
- `data/processed/<Coast>/ais_<region>_long_horizon/model_ready_windows.parquet`

Optional audit:

```bash
python AIS_AUDIT.py --segments data/processed/West\ Coast/.../coastal_segments.parquet
```

### 5.2 Combine coasts → USA Combined windows

```bash
python scripts/combine_datasets.py \
  --data-root data/processed \
  --out data/processed/combined
```

Produces train/val/test parquet pieces under `data/processed/combined/` (recreate if missing; earlier intermediate `combined` / `combined_filtered` may sit in `old_versions/data_processed/`).

### 5.3 Stationary + smart-motion filters

```bash
python scripts/apply_training_filters.py \
  --input-dir data/processed/combined \
  --output-dir data/processed/combined_filtered_smart \
  --smart-motion
```

This removes near-stationary / abnormal-loop trajectories and keeps meaningfully moving history.  
Result used by the report pipeline: **`combined_filtered_smart/`** (~536k windows before inland removal).

### 5.4 Inland / canal removal → coastal set

```bash
python scripts/filter_inland_windows.py \
  --input data/processed/combined_filtered_smart/train.parquet \
  --output data/processed/combined_filtered_smart_coastal/train.parquet \
  --open-water-km 10 \
  --inland-fraction 0.5
```

Rule: drop a window if **>50%** of subsampled history points have **no open water within 10 km** (coarse land/ocean mask). Keeps open water, coastal fringe, and ports.

Or on the cluster:

```bash
sbatch scripts/exp_coastal/filter_inland.sbatch
```

**Final training parquet:**  
`data/processed/combined_filtered_smart_coastal/train.parquet`  
(+ `inland_filter_report.json` next to it)

**Land grid** for soft land penalty: `data/processed/land_grid_us.npz`

---

## 6. Training (`exp_coastal`)

**All expanded CLI arguments** for every report model are collected in [`EXPERIMENTS.md`](EXPERIMENTS.md).  
To print or run them locally:

```bash
bash scripts/exp_coastal/reproduce_experiments.sh            # print all
bash scripts/exp_coastal/reproduce_experiments.sh --run flat  # Flat LSTM
bash scripts/exp_coastal/reproduce_experiments.sh --run ar18  # AR 18h
```

Those flags match `scripts/exp_coastal/train_*.sbatch` (the Slurm jobs used for the report).

### 6.1 Submit the full suite (recommended)

```bash
bash scripts/exp_coastal/submit_all.sh
```

This chains: inland filter → AR 9h → 12h → 18h → 24h → Flat LSTM → Transformer → Adaptive → Sliding 3h.

### 6.2 Individual Slurm jobs

| Job | Script |
|-----|--------|
| Inland filter | `scripts/exp_coastal/filter_inland.sbatch` |
| AR LSTM 9 / 12 / 18 / 24 h | `train_AR_9h.sbatch` … `train_AR_24h.sbatch` |
| Flat LSTM | `train_flat_lstm.sbatch` |
| Transformer | `train_transformer.sbatch` |
| Adaptive multi-scale AR | `train_adaptive.sbatch` |
| Separate-encoder adaptive (softmax / hard) | `train_adaptive_separate_{softmax,hard}.sbatch` · `submit_adaptive_separate.sh` |
| Sliding 3h × 4 | `train_sliding_3h.sbatch` |
| Ablation: AR 12h, no land penalty | `train_AR_12h_noland.sbatch` |

Logs: `LOG/exp_coastal_*.out`

### 6.3 Example local / interactive run (AR 18h)

```bash
export PYTHONPATH=/path/to/project
cd /path/to/proj/project

python -u models/RNN_AR.py \
  --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/AR_18h \
  --sample 300000 \
  --history-hours 18 --future-hours 12 --horizon-hours 12 \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 \
  --epochs 60 --patience 10 \
  --no-maneuver-oversample --target-mode anchor_offset \
  --land-penalty-weight 0.1
```

### 6.4 Where outputs go

| Kind | Path |
|------|------|
| Checkpoints | `data/models/USA Combined/unknown/exp_coastal/<run>/` |
| Metrics / sample trajs / plots | `data/results/USA Combined/unknown/exp_coastal/<run>/` |
| Report figures & HTML maps | `data/results/USA Combined/unknown/exp_coastal/report_figures/` |

Typical result artifacts per run: `*_metrics.json`, `*_sample_trajectories.json`, training-history / scatter / error-hist PNGs.

---

## 7. Models

| Model | Code | Role in the study |
|-------|------|-------------------|
| Kinematic SOG+COG | evaluated inside training utils | Non-learning baseline |
| **AR LSTM** (9/12/18/24h) | `models/RNN_AR.py` | Fixed-context history sweep |
| **Flat LSTM** | `models/RNN.py` | Direct 72-step forecast (no AR unroll) |
| **Transformer** | `models/transformers.py` | Self-attention over 24h history |
| **Sliding 3h×4** | `models/RNN_recursive_1h.py` | Short predictor rolled to 12h |
| **Adaptive multi-scale AR** | `models/RNN_AR_adaptive.py` | Soft gate over 9+12+18+24h encodings (shared encoder) |
| **Separate-encoder adaptive** | `models/RNN_AR_diff_encoder.py` | Four independent encoders + gate (`--gate-mode softmax\|hard`) |

Shared helpers:

- `models/training_utils.py` — loss, loaders, eval, land penalty  
- `models/land_mask_utils.py` — soft land penalty  
- `models/plot_utils.py`, `visualize_model_results.py`  
- `models/compare_rnn_models.py` — side-by-side comparison  
- `models/build_clean_trajectory_map.py` — Folium trajectory maps  
- `models/vessel_type_utils.py` — optional vessel-type gate encoding  

**Training loss (summary):** trajectory / anchor-offset loss with Haversine + relative terms; optional **soft land penalty** (λ = 0.1 in `exp_coastal`); AR models use **teacher forcing** (0.3) with scheduled decay where enabled. Exact formulation lives in `models/training_utils.py` (see also archived notes under `old_versions/docs_notes/`).

---

## 8. Main results (report)

Coastal test set, λ_land = 0.1. Distances in **km**.

| Model | History | Params (approx.) | Median FDE | Mean FDE | Median ADE |
|-------|---------|------------------|------------|----------|------------|
| Kinematic | — | — | 102.48 | 152.45 | — |
| AR LSTM 9h | 54 steps | 1.63M | 20.43 | 36.53 | 7.50 |
| AR LSTM 12h | 72 | 1.63M | 19.99 | 36.70 | 7.39 |
| **AR LSTM 18h** | 108 | 1.63M | **19.71** | 37.30 | **7.35** |
| AR LSTM 24h | 144 | 1.63M | 20.30 | 36.60 | 7.47 |
| **Flat LSTM** | 144 | 0.91M | **18.80** | **35.26** | 7.72 |
| Transformer | 144 | ~1.00M | 19.40 | 36.81 | 8.24 |
| Sliding 3h×4 | rolled | 0.84M | 22.44 | 39.35 | 7.61 |
| Adaptive AR (shared encoder) | 9+12+18+24 | 1.90M | 20.19 | 36.70 | 7.64 |
| Separate encoders + **hard** gate | 9+12+18+24 | 4.31M | **19.08** | 36.90 | 11.38 |
| Separate encoders + softmax gate | 9+12+18+24 | 4.31M | 21.00 | 37.70 | 12.38 |

Softmax separate-encoder collapsed to **9h (~99%)** and underperformed; hard selection spreads over 9/12/24h (~29/37/34%) and never uses 18h. Report comparison focuses on **shared vs separate+hard**.

**Takeaways (aligned with the report):**

1. Neural models cut median FDE from ~102 km to ~19–20 km.  
2. **Flat LSTM** is best overall median FDE.  
3. In the fixed AR sweep, **18h beats 24h** — useful AR context saturates (~12–15h); older history acts mostly as noise.  
4. Shared-encoder **adaptive gating** does not beat simpler models; the gate collapses toward long context with nearly redundant nested encodings.  
5. **Separate encoders + hard gate** improve on shared adaptive (20.19 → 19.08 median FDE) by making context states distinct, but still trail Flat LSTM (18.80) at ~2.3× the parameters.

---

## 9. Figures, maps, and analysis

### 9.1 Regenerating report figures

```bash
python scripts/generate_report_figures.py
```

Writes ranking / history-sweep / α / track panels (and Folium HTML if `folium` is installed) under:

`data/results/USA Combined/unknown/exp_coastal/report_figures/`

Report-figure highlights (classic set used in the PDF):

- `fig_example_tracks_panels.png` / track panel sets  
- `fig_model_ranking_fde.png`  
- `fig_ar_context_sweep.png`  
- `fig_straight_vs_maneuver.png` / AR variants  
- `fig_adaptive_alphas.png`  
- `fig_error_vs_horizon.png`  

Extra diagnostics / presentation maps (not all embedded in the 8-page PDF):

- `fig_ar_why_18h_diagnostics.png`, `fig_ar_forget_bias.png`, `fig_adaptive_gate_forensics.png`, …  
- `map_ar18h_hourly_steps.html`, `map_ar18h_pred_twists.html`, `slide_ar18h_map.html`  
- `map_clear_tracks_*.html`

### 9.2 Interactive prediction maps

```bash
python scripts/plot_ar12h_map.py
python scripts/plot_ar9h_map.py
```

Defaults point at `exp_coastal` AR runs and coastal parquet. Override with `--input` / trajectory JSON paths as needed.

### 9.3 Why AR 18h beats 24h (occlusion / grads / hidden state)

Reproduces the mentor diagnostics that justify the AR history ranking:

```bash
source scripts/exp_coastal/_env.sh && cd "$SUBROOT"
$PYTHON scripts/diagnose_ar_context.py --n-samples 800
```

What it measures on a shared coastal test subsample for AR 9/12/18/24:

1. **Hidden-state saturation** — hours of recent context to reach 95% cosine with full-history \(h\)
2. **Occlusion** — Δ median FDE when zeroing the oldest vs newest 3h
3. **Backprop attribution** — share of \(|\partial L/\partial x_t|\) on newest vs oldest 3h
4. **Forget-gate bias** — encoder LSTM layer-0 forget bias
5. **AR24 keep-last-k** — FDE when only the newest \(k\) hours are kept

Writes under `report_figures/`:

- `fig_ar_why_18h_diagnostics.png`
- `fig_ar_forget_bias.png`
- `fig_ar_why_18h_meta.txt`
- `fig_ar_why_18h_summary.json`

### 9.4 Adaptive gate analysis

```bash
python scripts/analyze_adaptive_alphas.py
python scripts/analyze_adaptive_gate_drivers.py
```

Defaults read  
`.../exp_coastal/adaptive_multiscale/RNN_AR_adaptive/context_alpha_weights.json`.

### 9.5 Separate-encoder adaptive gates (softmax vs hard)

Model: `models/RNN_AR_diff_encoder.py` — four **independent** LSTM encoders (9/12/18/24h), shared gate MLP + AR decoder; same loss/data as the coastal suite. Does **not** modify `RNN_AR.py` / `RNN_AR_adaptive.py`.

```bash
source scripts/exp_coastal/_env.sh && cd "$SUBROOT"
bash scripts/exp_coastal/submit_adaptive_separate.sh   # both jobs in parallel
# or individually:
sbatch scripts/exp_coastal/train_adaptive_separate_softmax.sbatch
sbatch scripts/exp_coastal/train_adaptive_separate_hard.sbatch
```

| Kind | Path |
|------|------|
| Checkpoints | `data/models/.../exp_coastal/adaptive_separate_encoders_{softmax,hard}/` |
| Metrics / plots | `data/results/.../exp_coastal/adaptive_separate_encoders_{softmax,hard}/RNN_AR_diff_encoder/` |
| Comparison table | `data/results/.../exp_coastal/adaptive_separate_encoders_comparison.md` |
| LyX inserts | `report/chatgpt_adaptive_*.md` |

After training:

```bash
python scripts/compare_adaptive_separate_gates.py
python scripts/forensics_adaptive_hard.py   # shared vs separate+hard diagnostics
```

Forensics figures under `report_figures/`:

- `fig_adaptive_separate_vs_shared_fde.png`
- `fig_adaptive_separate_selection.png`
- `fig_adaptive_separate_hidden_cos.png`
- `fig_adaptive_hard_forced_vs_selected.png`
- `fig_adaptive_hard_selection_by_motion.png`
- `fig_adaptive_hard_grad_by_hour.png`
- `fig_adaptive_hard_forget_bias.png`
- `fig_adaptive_hard_encoder_l2.png`
- `fig_adaptive_hard_forensics_meta.json`

### 9.6 Other utilities

```bash
python scripts/plot_training_history.py --help
python scripts/map_filtered_tracks.py --help
python scripts/backfill_eval_metrics.py --help
python models/compare_rnn_models.py --help
python VISUALIZE.py --help          # general visualization CLI
python EXPORT_COAST_DATA.py --help
```

To serve HTML maps locally (example):

```bash
cd "data/results/USA Combined/unknown"
python -m http.server 8765
# open http://<host>:8765/exp_coastal/report_figures/map_ar18h_hourly_steps.html
```

---

## 10. Key file reference

| Path | Purpose |
|------|---------|
| `report/AIS_report.pdf` | Submitted 8-page report |
| `report/AIS_report.tex` | LaTeX source (structure + diagnostics prose) |
| `scripts/exp_coastal/_env.sh` | Shared env for Slurm jobs |
| `scripts/exp_coastal/submit_all.sh` | Full coastal suite |
| `window_data.py` | Window IO, trajectory splits, feature scaling |
| `coast_paths.py` | Coast bounding boxes and path resolution |
| `data/processed/combined_filtered_smart_coastal/` | Final ML input |
| `data/processed/land_grid_us.npz` | Land penalty grid |
| `data/results/.../exp_coastal/*/…_metrics.json` | Per-model metrics |
| `data/results/.../exp_coastal/report_figures/` | Figures + HTML maps |
| `models/RNN_AR_diff_encoder.py` | Separate-encoder adaptive (softmax / hard) |
| `scripts/forensics_adaptive_hard.py` | Shared vs separate+hard forensics |
| `scripts/compare_adaptive_separate_gates.py` | Softmax vs hard comparison table |
| `report/chatgpt_adaptive_hard_forensics_report.md` | LyX insert for hard-gate forensics |

---

## 11. Archived material

Everything that is **not** part of the final coastal / report workflow was **moved** (not deleted) to [`old_versions/`](old_versions/):

| Subfolder | Contents |
|-----------|----------|
| `reports/` | Draft PDFs, LyX, older TeX, guidelines, feedback |
| `docs_notes/` | Research / REPORT / CLAUDE markdown drafts |
| `scripts/` | `exp_final`, `exp_clean`, `exp_context`, `experiment1`, old sbatch |
| `data_results/`, `data_models/` | Pre-coastal runs and checkpoints |
| `data_processed/` | Early `combined` / `combined_filtered` intermediates |
| `logs/` | Development Slurm logs |
| `misc/` | One-off helpers |

See [`old_versions/README.md`](old_versions/README.md).

---

## 12. Troubleshooting

**`ModuleNotFoundError: proj.project...`**  
Set `PYTHONPATH` to the directory that *contains* `proj/` (not `proj/project`).

**`ModuleNotFoundError: folium`**  
`pip install folium` (maps / some report figure HTML paths). Training does not require it.

**Slurm cannot write logs**  
Ensure `LOG/` exists (`mkdir -p LOG`) and paths in `#SBATCH --output` match your clone.

**OOM when loading parquet**  
Training uses `--sample 300000` on the coastal set. Prefer the Slurm scripts (128G mem). For plotting, pass a smaller sample via script flags where available.

**Rebuilding windows is slow**  
Coastal NOAA download + segmenting is multi-day. Prefer the existing `data/processed/<Coast>/` and `combined_filtered_smart_coastal` artifacts for experiments; only re-run early pipeline stages when regenerating data from scratch.

**Want older experiment numbers (`exp_final`, v1, …)**  
Restore paths from `old_versions/data_results` / `old_versions/scripts` — they are not on the active `scripts/` / `data/results` paths anymore.
