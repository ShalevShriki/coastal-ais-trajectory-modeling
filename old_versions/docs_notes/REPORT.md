# Learning Temporal Context for Long-Horizon AIS Vessel Trajectory Prediction

**Final Project Report — Deep Learning Course (046211)**

[Student Name 1], [Student Name 2]

---

## Abstract

This project studies AIS vessel trajectory prediction as a **sequence modeling** problem, focusing on **how much temporal context** is needed for **12-hour** forecasting. We compare fixed-context autoregressive RNNs, receding-horizon sliding-window prediction, a long-context Transformer, and an **adaptive multi-scale RNN_AR** that learns how much history to use for each trajectory.

All reported experiments use **coastal / open-water windows** after inland river–canal removal, with a soft **land penalty** in the training loss. The prediction horizon is fixed at 12 hours; the main experimental variable is history length (9 / 12 / 18 / 24 h). Completed context-sweep results show that neural models far outperform kinematic constant-velocity extrapolation, and that **18 h fixed context currently yields the best median FDE** among finished autoregressive runs. Flat LSTM, Transformer, sliding-window, and adaptive multi-scale experiments are **still running**.

---

## 1. Introduction

### 1.1 Framing

Maritime traffic monitoring relies on the Automatic Identification System (AIS), which broadcasts vessel position, speed, and course. Forecasting where a vessel will be hours ahead supports collision avoidance, traffic planning, and search-and-rescue. We treat this as long-horizon sequence forecasting: given a resampled history of AIS features, predict future latitude and longitude at 10-minute resolution for the next **12 hours**.

### 1.2 Research question

The main goal is not only to compare architectures, but to answer:

> **How much past trajectory context is useful for predicting a vessel trajectory 12 hours into the future, and can an RNN-based model learn the relevant context window size by itself?**

Instead of assuming every trajectory should use the same fixed history, we test whether different motion patterns need different amounts of past information. For example:

- Stable straight motion may need shorter context.
- Turning / maneuvering may need longer context.
- Speed changes may require longer temporal history.
- Dense coastal or port-like regions may require longer context.

**Sub-questions:**

1. Does longer history improve 12-hour trajectory prediction?
2. Is 24 h history always better than 9 h or 12 h?
3. Does long context help mainly in maneuvers and speed changes?
4. Can too much history add noise for simple straight trajectories?
5. Can an RNN learn a soft context-window preference by itself?
6. Which motion features explain the selected context length?

### 1.3 Contribution

The main contribution is an **adaptive multi-scale autoregressive RNN** that learns soft weights over several history windows, enabling both trajectory prediction and analysis of which motion features determine useful context length.

### 1.4 Course connection

The project connects to RNNs, LSTMs, autoregressive decoding, encoder–decoder models, long-range dependencies, temporal context, attention, Transformers, and interpretability through learned weights. An RNN compresses the past into a hidden state; a Transformer can attend over the full past; the adaptive model explicitly learns how much weight to assign to each history window.

### 1.5 Previous work

Classical maritime forecasting uses dead reckoning and Kalman filters with constant-velocity assumptions. Learning-based AIS work applies RNNs/LSTMs for sequential trajectory encoding and, more recently, Transformers that attend over long histories. Most models fix a single history length. We instead compare fixed-context autoregressive LSTMs, a receding-horizon sliding rollout, a long-context Transformer, and an adaptive multi-scale gate that learns how much history to use per trajectory—linking prediction performance to **temporal context length**, a core issue in sequence models (long-range dependence, attention vs recurrent memory).

---

## 2. Dataset and Trajectory Windows

### 2.1 Source and geography

We use public NOAA AIS daily files, processed for US coastal regions and merged into **USA Combined**.

### 2.2 Window definition

Each supervised sample is one sliding window. Let \(\Delta t = 10\) minutes. With full storage:

\[
\text{history: } T_h = 144 \text{ steps } (24\,\mathrm{h}),\qquad
\text{future: } T_f = 72 \text{ steps } (12\,\mathrm{h}).
\]

The **anchor** \(P_0 = (\mathrm{lat}_0, \mathrm{lon}_0)\) is the last observed history position (“now”).

**[Add here: map / track figure of example coastal AIS trajectories — e.g. `track_maps/png_good_segments_random.png`]**

### 2.3 Input features

At each history step \(t\), the model observes a 15-dimensional vector:

\[
\mathbf{x}_t =
\big[
\mathrm{lat},\;
\mathrm{lon},\;
\mathrm{SOG},\;
\sin\mathrm{COG},\;
\cos\mathrm{COG},\;
\sin\psi,\;
\cos\psi,\;
\mathbb{1}_{\psi\text{ missing}},\;
\Delta t_{\mathrm{sec}},\;
\Delta\mathrm{lat},\;
\Delta\mathrm{lon},\;
\Delta\mathrm{SOG},\;
\Delta\mathrm{COG},\;
v_{\mathrm{north}},\;
v_{\mathrm{east}}
\big]^\top
\]

where \(\psi\) is heading. Course and heading use sin/cos encoding to avoid \(0^\circ/360^\circ\) discontinuity.

### 2.4 Prediction target (anchor offset)

The network predicts offsets relative to the anchor:

\[
y_t = P_t - P_0,\qquad
\hat{P}_t = P_0 + \hat{y}_t,
\qquad t = 0,\ldots,71.
\]

For fixed-context experiments with history length \(H\in\{9,12,18,24\}\) hours, we take the **suffix** of the stored 24 h window ending at the same anchor (e.g. \(H=9\) ⇒ last 54 steps).

### 2.5 Filtering (final training distribution)

```text
NOAA AIS → segment + resample → 24h→12h windows
  → combine coasts
  → stationary filter (history-only)
  → smart-motion filter
  → inland removal          → combined_filtered_smart_coastal
```

**Stationary filter** (history only): remove near-stationary windows (confined radius ≤ 0.5 km, displacement ≥ 1 km, mean SOG ≥ 0.5 kn).

**Smart-motion filter:** keep meaningful motion (min 16 h net displacement ≥ 8 km; last 8 h net ≥ 2 km; loop ratio ≤ 0.35).

**Inland removal (scope of this report):** inland rivers, canals, and marsh far from open ocean are out of scope for coastal forecasting. Drop a window if **more than 50%** of subsampled history points have **no open water within 10 km**. Keeps open water, coastal fringe, and ports.

| Quantity | Value |
|----------|--------|
| Rows before inland filter | 535,967 |
| Rows after inland filter | **363,014** |
| Keep fraction | **67.7%** |
| Drop fraction | 32.3% |

**[Add here: figure of inland filter — kept coastal vs rejected inland tracks; also `smart_motion_audit/preview_kept_vs_rejected.png`]**

**Train / val / test:** split by **trajectory ID** (not random rows), seed 42. Training uses a subsample of up to 400k windows. All models in this report train on `combined_filtered_smart_coastal` with land penalty weight \(\lambda_{\mathrm{land}}=0.1\).

---

## 3. Evaluation Metrics

Primary evaluation uses Haversine great-circle distance in kilometers.

### 3.1 FDE (Final Displacement Error)

\[
\mathrm{FDE} = d_{\mathrm{Hav}}\!\left(\hat{P}_{T_f-1},\, P_{T_f-1}\right)
\]

at the **12 h** step. Primary ranking metric: **median FDE** on the test set.

### 3.2 ADE (Average Displacement Error)

\[
\mathrm{ADE} = \frac{1}{T_f}\sum_{t=0}^{T_f-1} d_{\mathrm{Hav}}\!\left(\hat{P}_t,\, P_t\right).
\]

### 3.3 Normalized metrics

\[
\mathrm{nFDE} = \frac{\mathrm{FDE}}{L_{\mathrm{path}}},\qquad
\mathrm{nADE} = \frac{\mathrm{ADE}}{L_{\mathrm{path}}},
\]

where \(L_{\mathrm{path}}\) is the true future path length (with a small floor for stability).

### 3.4 Horizon-wise error

Error as a function of forecast lead time (1 h, 2 h, 3 h, 6 h, 9 h, 12 h) to see early accuracy vs late drift.

**[Add here: horizon-wise error curves (error vs hours ahead) for finished models]**

### 3.5 Bucket-based evaluation

Report FDE/ADE on motion buckets derived from history: **straight**, **maneuver**, **anchored**, **other** (and, when available, stable vs changing speed). A model can look strong on average while failing on a rare but important bucket.

---

## 4. Training Loss (Full Form)

### 4.1 Reconstruction of absolute positions

Given predicted offsets \(\hat{y}_{t}\) and anchor \(P_0\):

\[
\hat{P}_t = P_0 + \hat{y}_t,\qquad
P_t^{\mathrm{true}} = P_0 + y_t.
\]

### 4.2 Huber term (degree space)

With Huber threshold \(\delta = 0.01^\circ\):

\[
L_{\mathrm{Huber}}^{(i)}
=
\frac{1}{T\cdot 2}
\sum_{t=0}^{T-1}
\sum_{c\in\{\mathrm{lat},\mathrm{lon}\}}
\mathrm{Huber}_\delta\!\left(\hat{y}_{t,c}^{(i)} - y_{t,c}^{(i)}\right).
\]

\(T\) is the number of future steps used (full 72, or a curriculum prefix).

### 4.3 Geographic term (local km, training)

\[
d_t^{(i)}
=
\sqrt{
\big(\Delta\mathrm{lat}_t\cdot 111.322\big)^2
+
\big(\Delta\mathrm{lon}_t\cdot 111.322\cdot\cos\bar{\phi}_t\big)^2
},
\]

\[
L_{\mathrm{geo}}^{(i)}
=
\frac{1}{T}\sum_{t=0}^{T-1}\frac{d_t^{(i)}}{50}.
\]

(The factor \(1/50\) balances Huber and km scales. **Evaluation** uses exact Haversine, not this planar approximation.)

### 4.4 Difficulty weight (default on)

From history maneuver score \(s^{(i)}\) (\(|\Delta\mathrm{COG}| + |\Delta\mathrm{SOG}|\) aggregates):

\[
w^{(i)} = 1 + \frac{s^{(i)}}{\mathrm{median}(s)}.
\]

### 4.5 Soft land penalty

A coarse US land raster \(\mathcal{M}(\mathrm{lat},\mathrm{lon})\in[0,1]\) is sampled with bilinear interpolation at each predicted absolute point. With weight \(\lambda_{\mathrm{land}}=0.1\):

\[
L_{\mathrm{land}}
=
\lambda_{\mathrm{land}}\;
\mathbb{E}_{i,t}\!\big[\mathcal{M}(\hat{P}_t^{(i)})\big].
\]

This discourages trajectories that cut over land.

### 4.6 Full objective (as used)

With geo mix weight \(w=0.5\) and relative term off (\(\lambda_{\mathrm{rel}}=0\)):

\[
\boxed{
\begin{aligned}
\mathcal{L}
&=
\frac{1}{N}\sum_{i=1}^{N}
w^{(i)}\Big[
(1-w)\,L_{\mathrm{Huber}}^{(i)}
+
w\,L_{\mathrm{geo}}^{(i)}
\Big]
+
L_{\mathrm{land}}.
\end{aligned}
}
\]

### 4.7 Training dynamics for AR models

- **Scheduled teacher forcing:** decoder input is ground truth with probability \(\tau\), else the model’s previous prediction; \(\tau\) anneals from \(0.3\) to \(0\).
- **Horizon curriculum:** early epochs optimize a shorter future prefix (from 6 h up to full 12 h), then grow \(T\).

**[Add here: two-panel training history plot (train objective vs validation loss) for AR 12 h / AR 18 h]**

---

## 5. Models

### 5.1 Kinematic constant-velocity baseline

**Only baseline.** From last known \((\mathrm{lat},\mathrm{lon},\mathrm{SOG},\mathrm{COG})\), extrapolate constant speed and course for 12 h.

Answers: *Do neural models learn more than simple physical motion continuation?*

### 5.2 Flat LSTM (direct multi-horizon)

\[
\mathbf{h} = \mathrm{LSTM}(\mathbf{x}_{1:T_h}),\qquad
\hat{y}_{0:T_f-1} = \mathrm{Linear}(\mathbf{h}).
\]

Setup: **24 h → 12 h**, one-shot prediction of all 72 future offsets. Used as a reference for direct vs autoregressive decoding (one Flat LSTM is enough; the project focus is context learning in AR models).

**Status: still running** (coastal + land penalty).

### 5.3 Fixed-context RNN_AR (encoder–decoder + anchor)

Encoder LSTM reads history and provides temporal context to an autoregressive decoder that emits offsets step by step:

\[
\hat{y}_t = f_{\mathrm{dec}}(\hat{y}_{<t},\, h_{\mathrm{enc}}),\qquad
\hat{P}_t = P_0 + \hat{y}_t.
\]

Same architecture trained at:

\[
9\,\mathrm{h},\;12\,\mathrm{h},\;18\,\mathrm{h},\;24\,\mathrm{h}
\;\rightarrow\;
12\,\mathrm{h}.
\]

**Status: complete** (see §6.2).

**[Add here: architecture diagram of encoder–decoder AR LSTM with anchor]**

### 5.4 Receding-horizon sliding window

Train **24 h → 3 h** chunk displacement. At inference, roll four times:

```text
history [0,24] → predict [24,27]
shift window   → predict [27,30]
shift window   → predict [30,33]
shift window   → predict [33,36]
```

Predicted points enter the next history window (local updated context, but risk of error accumulation).

**Status: still running** (coastal + land penalty).

**[Add here: schematic of 3 h × 4 receding-horizon rollout]**

### 5.5 Transformer (long context)

Self-attention over **24 h** history → **12 h** future (one run; not a full Transformer context sweep). Asks whether attention over long context helps vs recurrent memory.

**Status: still running**.

### 5.6 Adaptive multi-scale RNN_AR (main contribution)

Encode four suffixes separately:

\[
h_9 = E(x_{-9\mathrm{h}:0}),\;
h_{12}=E(x_{-12\mathrm{h}:0}),\;
h_{18}=E(x_{-18\mathrm{h}:0}),\;
h_{24}=E(x_{-24\mathrm{h}:0}).
\]

Gate:

\[
\boldsymbol{\alpha}
=
\mathrm{softmax}\big(\mathrm{MLP}([h_9;h_{12};h_{18};h_{24}])\big),
\quad
\sum_k \alpha_k = 1.
\]

Fused context and AR decode:

\[
h_{\mathrm{ctx}} = \sum_k \alpha_k\, h_k
\;\longrightarrow\;
12\,\mathrm{h}\ \text{future}.
\]

One \(\boldsymbol{\alpha}\) per sample (not per future step). Saved \(\alpha\) values enable feature analysis (§7).

**Status: still running**.

**[Add here: adaptive multi-scale architecture diagram with α₉…α₂₄]**

---

## 6. Experiments and Results

**Dataset for all rows below:** `combined_filtered_smart_coastal`, land penalty \(\lambda_{\mathrm{land}}=0.1\), trajectory split, seed 42, 12 h FDE (Haversine km).

### 6.1 Kinematic baseline vs neural models

| Model | History | Median FDE | Mean FDE | Median ADE | Status |
|-------|---------|------------|----------|------------|--------|
| Kinematic SOG+COG | — | **102.5** | 152.5 | — | ✓ |
| Flat LSTM | 24 h | — | — | — | **still running** |
| Transformer | 24 h | — | — | — | **still running** |
| AR LSTM | 9 h | 20.43 | 36.53 | 7.50 | ✓ |
| AR LSTM | 12 h | 19.99 | 36.70 | 7.39 | ✓ |
| AR LSTM | **18 h** | **19.71** | 37.30 | **7.35** | ✓ **best so far** |
| AR LSTM | 24 h | 20.30 | 36.60 | 7.47 | ✓ |
| Sliding 3 h×4 | 24 h | — | — | — | **still running** |
| Adaptive multi-scale | 9+12+18+24 h | — | — | — | **still running** |

**Finding (RQ: baseline).** Neural AR models cut median FDE from ~102 km to ~20 km (~5×). Constant-velocity extrapolation is not sufficient for 12-hour coastal forecasting under this data scope.

**[Add here: bar chart of median FDE — kinematic vs finished AR models]**

**[Add here: predicted vs true scatter @ 12 h for best AR (AR 18 h) — `lstm_ar_scatter.png`]**

**[Add here: FDE error histogram for best AR — `lstm_ar_error_hist.png`]**

### 6.2 Fixed-context RNN_AR — which history length works best?

**Table: context-length sweep (completed).**

| History \(H\) | Steps | Median FDE ↓ | Mean FDE | Median ADE | Median FDE (maneuver) | Median FDE (straight) |
|---------------|-------|--------------|----------|------------|------------------------|------------------------|
| 9 h | 54 | 20.43 | 36.53 | 7.50 | **16.73** | 48.26 |
| 12 h | 72 | 19.99 | 36.70 | 7.39 | 17.32 | 45.42 |
| **18 h** | 108 | **19.71** | 37.30 | **7.35** | 18.36 | **39.34** |
| 24 h | 144 | 20.30 | 36.60 | 7.47 | 19.30 | 43.17 |

**Answers to sub-questions (so far, coastal + land penalty):**

1. **Does longer history improve FDE?** Partially: performance improves from 9 → 12 → **18 h**, then **degrades** at 24 h.
2. **Is 24 h always best?** **No.** 24 h is worse than 18 h (and worse than 12 h on median FDE).
3. **Long context on maneuvers?** Maneuver median FDE is *best* at **9 h** (16.7 km) and *worsens* as context grows — opposite of a simple “maneuvers need more history” story on this filtered set.
4. **Noise on straight tracks?** Straight median FDE is high for all \(H\); **18 h** is best among finished runs (39.3 km). Extra history beyond 18 h does not help straight tracks either.

**[Add here: line/bar plot — median FDE vs history hours (9/12/18/24)]**

**[Add here: grouped bars — maneuver vs straight FDE for each context length]**

**Interpretation.** Useful context saturates around **18 h** for overall median FDE on coastal windows. Full 24 h can add outdated regime information that the LSTM encoder compresses into the decoder state. Bucket trends suggest overall ranking and maneuver ranking need not agree: short context remains strong on the large maneuver mass, while mid-long context helps the overall/straight statistics.

### 6.3 Flat LSTM vs RNN_AR (direct vs autoregressive)

**Comparison planned:**

```text
Flat LSTM:  24h → 12h (one-shot)
RNN_AR:     24h → 12h (autoregressive)   [median FDE = 20.30]
```

**Status:** Flat LSTM **still running**. Fill this subsection when complete; research question: *Does AR prediction beat direct multi-horizon decoding?*

**[Add here: side-by-side FDE / ADE table + training curves once Flat LSTM finishes]**

### 6.4 Direct 12 h AR vs sliding 3 h×4

**Comparison planned:**

```text
RNN_AR 24h → 12h direct          [median FDE = 20.30]
Sliding: 24h → 3h, roll ×4
```

**Status:** Sliding window **still running**. Expected tension from the research plan: updated local context vs **error accumulation** when predictions enter the next window.

**[Add here: FDE comparison + horizon-wise curves once sliding finishes]**

### 6.5 RNN_AR vs Transformer

**Comparison planned:**

```text
RNN_AR 24h → 12h                 [median FDE = 20.30]
Transformer 24h → 12h
```

**Status:** Transformer **still running**. Research question: *Does attention over long context help compared to recurrent memory?*

**[Add here: FDE / ADE comparison once Transformer finishes]**

### 6.6 Fixed context vs adaptive multi-scale (most important)

**Comparison planned:**

```text
best fixed-context RNN_AR        [currently AR 18h, median FDE = 19.71]
vs
Adaptive Multi-Scale RNN_AR      [α₉, α₁₂, α₁₈, α₂₄]
```

**Status:** Adaptive model **still running**. This is the central contribution: can the model learn useful context length, and do \(\alpha\) weights match trajectory type?

**[Add here after run: mean α bar chart; α histograms; α vs straightness / COG variance scatter]**

---

## 7. Feature Analysis of Learned Context Weights

*(Section applies once the adaptive model finishes. Structure follows the research plan.)*

Save per test sample:

\[
(\alpha_9,\, \alpha_{12},\, \alpha_{18},\, \alpha_{24}).
\]

Correlate with derived motion features:

| Feature | Definition / intuition |
|---------|------------------------|
| Straightness | \(\mathrm{direct\_distance}/\mathrm{path\_length}\) ≈ 1 ⇒ straight |
| COG variability | mean \(|\Delta\mathrm{COG}|\), COG variance, turn rate |
| SOG variability | mean \(|\Delta\mathrm{SOG}|\), SOG variance |
| Path / net displacement | path length, direct distance |
| Local AIS density *(optional)* | proxy for ports / busy lanes |

**Expected relationships (hypotheses to test):**

```text
high straightness, low COG/SOG variance  → higher α₉ / α₁₂
low straightness, high variability       → higher α₁₈ / α₂₄
high local density                       → longer preferred context
```

**Status: still running** (requires adaptive checkpoint + α dump).

**[Add here: correlation table / heatmaps of α vs motion features]**

**[Add here: qualitative Folium maps — history (blue), ground truth (green), prediction (red); e.g. `map_ar12h_examples.html` screenshots]**

---

## 8. Discussion

### 8.1 What the finished experiments support

On **inland-filtered coastal** data with **land penalty**:

- Neural AR models clearly beat kinematic extrapolation (~102 → ~20 km median FDE).
- Context length **matters**: the best finished fixed window is **18 h**, not 24 h and not 9 h.
- Bucket results complicate a single story: maneuvers favor shorter context on median FDE, while overall/straight statistics peak at 18 h.

### 8.2 Open questions (pending runs)

- Does Flat LSTM match or beat AR 18 h / AR 24 h?
- Does sliding 3 h×4 reduce drift or accumulate error?
- Does Transformer attention exploit 24 h better than RNN_AR 24 h?
- Can adaptive α recover (or beat) the best fixed window and explain *why*?

### 8.3 Limitations

Coarse land mask; single USA Combined merge; fixed 10-minute resampling; unfinished adaptive feature analysis; land-penalty weight not yet ablated in the main table (noland control is outside the primary “filter + penalty” setting of this report).

---

## 9. Conclusion

We study 12-hour AIS forecasting as a **temporal context** problem on coastal windows with inland removal and a soft land penalty. Completed fixed-context AR experiments show that **useful history length is finite**: **18 h** currently minimizes median FDE, while **24 h does not win**. Maneuver vs straight buckets disagree on the preferred length, motivating the adaptive multi-scale model and α–feature analysis. Flat LSTM, Transformer, sliding-window, and adaptive runs are **still running**; final claims about direct vs AR, attention vs recurrence, chunked rollout, and learned context weights will be filled when those jobs complete.

---

## 10. Future Work

- Complete remaining coastal + land-penalty models and update Tables in §6
- Horizon-wise error curves for all finished models
- Full α vs motion-feature analysis (§7)
- Optional local AIS density feature
- Optional land-penalty ablation (same coastal data, \(\lambda_{\mathrm{land}}=0\)) as a control — not mixed into the main ranking

---

## 11. References

*(Add before submission.)*

1. NOAA AIS Data Handler — `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/`
2. [Add AIS trajectory forecasting paper]
3. [Add sequence-to-sequence / LSTM forecasting reference]
4. [Add Transformer / attention reference]
5. Hochreiter & Schmidhuber, “Long short-term memory,” *Neural Computation*, 1997.
6. Vaswani et al., “Attention is all you need,” *NeurIPS*, 2017.

---

## 12. Ethics Statement

*(Course template — complete before submission. 5 points.)*

### 12.1 Introduction

**Student names:** [Student Name 1], [Student Name 2]

**Project title:** Learning Temporal Context for Long-Horizon AIS Vessel Trajectory Prediction

This project develops research models that forecast vessel positions from historical AIS for academic study of sequence modeling and temporal context—not operational navigation. Predictions are approximate and must not be used as the sole basis for safety-critical decisions.

### 12.2 LLM-assisted stakeholder analysis

**2a. Three stakeholder types:** [To complete]

**2b. Explanation to each stakeholder:** [To complete — ≤1 paragraph]

**2c. Who is responsible for the explanation:** [To complete — ≤1 paragraph]

### 12.3 Reflection on AI output

*(Independent thinking — do not use LLMs here.)*

**3a. What should change in the LLM response to make the explanation more ethical?** [To complete — ≤1 paragraph]

---

## Appendix — Figure insertion checklist

| Location | Insert |
|----------|--------|
| §2.2 | Example coastal track map |
| §2.5 | Inland / smart-motion filter visualization |
| §3.4 | Horizon-wise error vs lead time |
| §4.7 | Training / validation loss curves |
| §5.3–5.6 | Architecture diagrams (AR, sliding, adaptive) |
| §6.1 | FDE bar chart + scatter + error hist |
| §6.2 | FDE vs history hours; maneuver vs straight bars |
| §6.3–6.6 | Comparison figures when jobs finish |
| §7 | α distributions and α–feature plots; Folium map screenshots |

## Appendix — Run status (`exp_coastal`, inland filter + land penalty 0.1)

| Experiment | Status |
|------------|--------|
| Kinematic baseline | ✓ (auto at eval) |
| AR 9 / 12 / 18 / 24 h | ✓ |
| Flat LSTM 24 h → 12 h | **still running** |
| Transformer 24 h → 12 h | **still running** |
| Sliding 3 h×4 | **still running** |
| Adaptive multi-scale | **still running** |
| α feature analysis | **still running** (depends on adaptive) |

*When you cut for the official ≤8-page submission, keep §§1, 4–6, 9, 11–12 first; trim appendices and detailed sub-questions as needed.*
