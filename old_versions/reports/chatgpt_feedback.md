# Full experiment + diagnostic results (for report rewrite)

**Course:** Deep Learning 046211  
**Project:** How Much History Do Deep Sequence Models Need for AIS Trajectory Prediction?  
**Authors:** Amitai Gal and Shalev Shiriki  
**Date:** July 14, 2026  
**Hard limit:** **8 pages**  

Use this file as the factual source of truth. Prefer the compact 8-page structure at the end. Incorporate the **mentor diagnostics** (why 18h AR wins; why adaptive fails) — that is the most important new scientific content.

---

## 1. Setup (fixed across coastal ranking)

| Item | Value |
|--|--|
| Data | USA Combined, coastal-filtered (`combined_filtered_smart_coastal`) |
| Windows after filter | 363,014 / 535,967 (67.7%) |
| Split | by `traj_id` — train 255,133 / val 34,549 / test 73,332 |
| Resolution | Δt = 10 min |
| History store | up to 24h = 144 steps |
| Forecast | 12h = 72 steps |
| Features | 15 (lat, lon, sog, cog_sin/cos, heading_sin/cos, heading_missing, dt_sec, dlat, dlon, dsog, dcog, v_north_kmh, v_east_kmh) |
| Target | anchor-offset \(y_t = P_t - P_0\) |
| Primary metric | **median FDE** (Haversine km) at 12h |
| Also report | mean FDE, median ADE |
| Land penalty | \(\lambda_{\mathrm{land}}=0.1\) for ranking |
| GPU | 1× RTX A6000, 128 GB RAM, 4 CPUs |
| Optim | Adam/AdamW, lr=1e-3, patience=10, max epochs=60 |
| AR extras | scheduled TF 0.3→0, horizon curriculum from 6h |

---

## 2. Official full-test ranking (coastal, λ_land=0.1)

| Rank | Model | History | Mode | Params | Median FDE | Mean FDE | Median ADE |
|--|--|--|--|--|--|--|--|
| 1 | **Flat LSTM** | 24h | Direct | 908,688 | **18.80** | **35.26** | 7.72 |
| 2 | Transformer | 24h | Attention | 1,002,000 | 19.40 | 36.81 | 8.24 |
| 3 | **AR LSTM 18h** | 18h | AR | 1,631,618 | **19.71** | 37.30 | **7.35** |
| 4 | AR LSTM 12h | 12h | AR | 1,631,618 | 19.99 | 36.70 | 7.39 |
| 5 | Adaptive Multi-Scale | 9+12+18+24 | Adaptive AR | 1,895,046 | 20.19 | 36.70 | 7.64 |
| 6 | AR LSTM 24h | 24h | AR | 1,631,618 | 20.30 | 36.60 | 7.47 |
| 7 | AR LSTM 9h | 9h | AR | 1,631,618 | 20.43 | 36.53 | 7.50 |
| 8 | Sliding 3h×4 | 24h→3h×4 | Sliding | 839,042 | 22.44 | 39.35 | 7.61 |
| — | Kinematic | last SOG+COG | Baseline | — | **102.48** | 152.45 | — |

### Architecture hyperparams
- **AR / Flat / Sliding / Adaptive encoder-decoder:** LSTM hidden **256**, **2** layers, dropout 0.2 (Flat/Sliding/Adaptive as trained).
- **Transformer:** d_model **128**, **8** heads, **4** encoder layers, FFN **512**, dropout 0.1, batch 128.
- **Adaptive:** shared LSTM + MLP gate on concat\([h_9;h_{12};h_{18};h_{24}]\); `gate_vessel_type=False`.

### Train compute (where logged)
| Model | Peak GPU MB | Train time | samp/s | Best epoch / ran |
|--|--|--|--|--|
| AR 9h | 2024 | ~63 min | 2695 | 29 / 40 |
| AR 12h | 2077 | ~68 min | 2696 | 32 / 43 |
| AR 18h | 2185 | ~76 min | 2615 | 36 / 47 |
| AR 24h | 2292 | ~56 min | 2501 | 22 / 33 |
| Flat | 1975 | **~7 min** | **11401** | 7 / 18 |
| Transformer | 1512 | ~24 min | 3553 | 9 / 20 |

---

## 3. Mentor question A — Why is AR 18h best (not 24h)?

**Claim:** Useful AR context saturates around ~12–15h. Extra hours barely move the final hidden state / gradients and can act as noise. 18h is the best full-test compromise; 24h is not “more memory,” it is mostly unused capacity.

### 3.1 Official AR sweep (same architecture; only history length changes)
Median FDE: 9h **20.43** → 12h **19.99** → 18h **19.71** → 24h **20.30** (non-monotonic).

Motion buckets (existing report finding):
- Maneuvering: **9h best**
- Straight: **18h best**
→ Overall sweet spot near 18h.

### 3.2 Occlusion (zero one history-hour block; Δ mean FDE; n=800 shared windows)
| Model | Zero oldest 3h | Zero newest 3h |
|--|--|--|
| AR 9h | +0.71 km | **+8.99 km** |
| AR 12h | +0.36 | **+6.13** |
| AR 18h | +0.05 | **+6.14** |
| AR 24h | **−0.09** (slightly helps) | **+7.45** |

Interpretation: recent hours dominate. For AR24, removing the *oldest* hours does not hurt — sometimes helps → old history is partly noise.

### 3.3 Backprop attribution \(|\partial L/\partial x_t|\) (final-offset MSE proxy)
Gradient mass:
- AR9: newest3h **56.9%**, oldest3h **14.8%**
- AR12: newest **36.1%**, oldest **10.3%**
- AR18: newest **21.3%**, oldest **8.8%** (more mid-range use)
- AR24: newest **52.6%**, oldest only **1.8%**

AR24’s training signal nearly ignores hours 22–24 back.

### 3.4 Final encoder hidden-state saturation
Hours of *recent* history needed so cosine(sim) of final \(h\) to full-history \(h\) ≥ 0.95:
- AR9 → **7h**, AR12 → **10h**, AR18 → **11h**, AR24 → **13h**

Even the 24h model’s state is essentially set by the last ~13h.

### 3.5 Forget-gate bias (weights)
Mean forget bias (ih+hh), layer 0: AR9 −0.51, AR12 −0.58, AR18 −0.52, **AR24 −0.72** (most negative).  
More negative forget bias ⇒ smaller forget gate ⇒ less retention of early history. AR24 learns to forget, not to exploit the long window.

### 3.6 Figures for this analysis
- `data/results/USA Combined/unknown/exp_coastal/report_figures/fig_ar_why_18h_diagnostics.png`
- `.../fig_ar_forget_bias.png`
- `.../fig_ar_context_sweep.png`
- `.../fig_ar_straight_vs_maneuver.png`

**One-sentence mentor answer:**  
Occlusion + gradients + hidden-state analysis show the AR encoder bottleneck uses mainly the last ~12h; beyond that, history is weakly used and can add noise — hence 18h beats 24h with identical capacity.

---

## 4. Mentor question B — Why is Adaptive Multi-Scale AR not better?

**Claim:** The gate fails for structural reasons (redundant nested representations + soft nearly-uniform mix with a slight 24h bias). It does **not** learn “maneuver→short, straight→long.”

### 4.1 Mechanism
Shared LSTM encodes nested suffixes → \(h_9,h_{12},h_{18},h_{24}\).  
MLP gate → softmax \(\alpha\) → \(h_{\mathrm{ctx}}=\sum_k \alpha_k h_k\) → AR decoder.

### 4.2 Gate statistics (full test, n=73,332)
| | 9h | 12h | 18h | 24h |
|--|--|--|--|--|
| Mean \(\alpha\) | 0.146 | 0.224 | 0.289 | **0.341** |
| Argmax rate | 0.4% | 1.6% | 11.9% | **86.1%** |

- Softmax gap \(\alpha_{24}-\alpha_{18}\): mean **0.052**, median **0.029** (tiny)
- Mean normalized entropy of \(\alpha\): **0.955** (near-uniform *soft* mix; argmax looks like “always 24” only because of a slight edge)
- When argmax=24: mean \(\alpha\approx[0.14,0.22,0.29,0.35]\), median \(\alpha_{24}\approx0.33\) — not one-hot
- Gate final bias already favors long context: **`[-0.30, -0.08, +0.12, +0.18]`**

### 4.3 Hidden states are nearly interchangeable
Mean cosine between context encoder states (n=512):

| | 9h | 12h | 18h | 24h |
|--|--|--|--|--|
| 9h | 1.000 | 0.969 | 0.954 | 0.943 |
| 12h | | 1.000 | 0.972 | 0.961 |
| 18h | | | 1.000 | **0.984** |
| 24h | | | | 1.000 |

\(\|h_{18}-h_{24}\| / \|h_{24}\| \approx 0.085\). Soft mixing barely changes \(h_{\mathrm{ctx}}\).

### 4.4 Policy ablation (force \(\alpha\); n=1200) — gate barely helps
| Policy | Median FDE |
|--|--|
| learned | 19.51 |
| uniform 0.25 | 19.54 |
| always 12h | **19.47** |
| always 24h | 19.72 |
| always 9h | 19.69 |
| always 18h | 19.88 |

Learned ≈ uniform. Fixed 12h matches/beats the learned gate on this subset.

### 4.5 Causal / correlation checks
- Zero lat/lon channels: mean \(|\Delta\alpha|\approx0.021\); still ~92% argmax=24. Geography is a weak soft prior, not a switch.
- Zero motion keep lat/lon: mean \(|\Delta\alpha|\approx0.034\); still ~91.5% argmax=24.
- Spearman with motion is weak (\(|r|\lesssim0.15\)). Strongest soft association: longer path/speed slightly ↑ \(\alpha_{18}\), ↓ \(\alpha_{24}\).
- Earlier RF surrogate (n=15k): **anchor_lon / anchor_lat** top drivers of \(\alpha\) — geographic prior, not maneuver rule.
- Adaptive bucket FDE: straight 46.09 / maneuver 19.12 (same heavy-tailed pattern as others; not evidence of good switching).

### 4.6 Figures
- `.../fig_adaptive_alphas.png`
- `.../fig_adaptive_gate_forensics.png`
- `.../fig_adaptive_context_hidden_sim.png`
- `.../adaptive_multiscale/RNN_AR_adaptive/gate_feature_drivers.html`

**One-sentence mentor answer:**  
Shared nested encodings make \(h_{18}\approx h_{24}\); the gate only learns a slight soft bias to 24h and does not beat a uniform/fixed mix, so adaptivity never becomes a motion-aware history selector.

---

## 5. Why Flat > Transformer > AR (short)

1. **Neural ≫ kinematic (102.5 → ~19 km):** models learn routes/turns/coastal structure beyond constant SOG+COG.  
2. **Flat best:** direct 72-step head avoids AR error accumulation; also fewest/faster params in practice (908k, ~7 min).  
3. **Transformer 2nd:** attention over 24h helps, but does not beat simple Flat here.  
4. **AR 18h 3rd overall / 1st among AR:** see §3.  
5. **Sliding worst among neural:** 3h×4 rollout accumulates error.  
6. **Adaptive underperforms its size (1.90M params):** see §4.

---

## 6. Other report content to keep (from existing draft)

### Data filtering
1. Stationarity filter (low radius / displacement / speed)  
2. Smart-motion filter (remove abnormal loops)  
3. Inland removal (rivers/canals if most history not near open water)

### Loss
\[
\mathcal{L}=\frac1N\sum_i w^{(i)}[0.5 L_{\mathrm{Huber}}^{(i)}+0.5 L_{\mathrm{geo}}^{(i)}]+\lambda_{\mathrm{land}}L_{\mathrm{land}}
\]
Huber for robustness; geo for km-scale errors; soft land penalty.

### Motion buckets
Post-hoc only, history-based features (no future leakage). Straightness \(= d_{\mathrm{direct}}/L_{\mathrm{path}}\). Used for analyzing preferred history length.

### Ethics
Must include: student names, title, LLM stakeholder analysis disclosure (“We used ChatGPT…”), reflection that average error ≠ safe individual cases.

### References (keep these four)
1. Perera & Guedes Soares, EKF vessel traj, ADAPTIVE 2010  
2. Capobianco et al., Deep learning vessel traj RNN, arXiv:2101.02486  
3. Nguyen & Fablet, TrAISformer, arXiv:2109.03958  
4. Nguyen et al., GeoTrackNet, arXiv:1912.00682  

---

## 7. Figures to use in an 8-page report (must-have)

1. Example tracks map/panel — `fig_tracks_report_3panel_percentiles.png` or `fig_example_tracks_panels.png`  
2. Model ranking — `fig_model_ranking_fde.png`  
3. AR history sweep — `fig_ar_context_sweep.png`  
4. Straight vs maneuver — `fig_ar_straight_vs_maneuver.png`  
5. Adaptive alphas — `fig_adaptive_alphas.png`  
6. Error vs horizon — `fig_error_vs_horizon.png`  
7. **New (strongly recommended if space):** `fig_ar_why_18h_diagnostics.png` *or* a 2-panel crop of occlusion + hidden saturation  
8. Optional if space: `fig_adaptive_context_hidden_sim.png`

Folder:  
`data/results/USA Combined/unknown/exp_coastal/report_figures/`

---

## 8. Required 8-page outline (do not exceed)

1. **Abstract** (~120–150 words) — include Flat 18.80, XF 19.40, AR18 best among AR, adaptive fails / geo prior  
2. **Intro + short related work** (kinematic / RNN / Transformer+geo; our contribution = controlled history study)  
3. **Data + filtering** + Fig. examples (offset formula **once**)  
4. **Metrics + motion buckets (short)** — Haversine brief; FDE/ADE; median justification  
5. **Models + ONE comparison table** + loss (compact) + TF/curriculum one sentence  
6. **Results**  
   - Overall ranking + Fig ranking  
   - AR sweep + non-monotonic 18h + **brief diagnostics why** (occlusion/grads/h-state)  
   - Motion buckets fig  
   - Adaptive: α stats + redundancy of \(h_k\) + policy ablation (learned≈uniform)  
7. **Discussion + limitations** (merge) + error-growth fig  
8. **Conclusion** (½ page)  
9. **Ethics** (dense single block with LLM disclosure)  
10. **References**

### Explicit cuts for 8 pages
- Do **not** print Haversine derivation in full  
- Do **not** duplicate offset equation  
- Do **not** have two ranking tables  
- Do **not** re-derive adaptive softmax twice  
- Keep ethics compact  
- Compute/GPU table is optional one line only  

### Headline clarity (put early & in conclusion)
- Best **overall:** Flat LSTM 24h (18.80)  
- Best **AR history length:** 18h (19.71), not 24h  
- Adaptive does **not** solve history selection  

---

## 9. Typos / fixes from prior PDF drafts
- Replace `[Partner Name]` → **Shalev Shiriki**  
- Fix broken “T ransformeruses”  
- Fill ADE for Flat/Transformer/Sliding (7.72 / 8.24 / 7.61) — no `--`  
- Say λ_land=0.1 + trajectory split once in Results  

---

## 10. Suggested abstract (ready to paste)

This project studies vessel trajectory prediction from AIS data as a sequence learning problem. Given a history of position, speed, and direction sampled every 10 minutes, we predict latitude and longitude for the next 12 hours, focusing on how much temporal context sequence models need. We compare kinematic extrapolation, AR LSTMs with 9/12/18/24 h history, a Flat LSTM, a Transformer, a sliding rollout model, and an Adaptive Multi-Scale RNN that gates over multiple history lengths. Neural models reduce median FDE from 102.5 km to about 19–20 km. The best model is a 24 h Flat LSTM (18.80 km); the Transformer follows (19.40 km). In a controlled AR LSTM sweep, 18 h is best (19.71 km) while 24 h does not help. Diagnostics show that encoder states and gradients concentrate on recent history, so longer windows can add noise. The adaptive gate almost always prefers 24 h only by a small soft margin, the four context states are nearly redundant, and forcing uniform/fixed gates matches learned performance — so useful adaptive history selection does not emerge from a simple gate.

---

*End of source pack. Companion report sources: `AIS_report_8pages.tex` (import into LyX) and `AIS_report_8pages.lyx`.*
