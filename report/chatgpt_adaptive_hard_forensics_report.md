# Report insert: Why separate+hard adaptive beats shared adaptive

**Source:** `scripts/forensics_adaptive_hard.py`  
**Meta:** `data/results/USA Combined/unknown/exp_coastal/report_figures/fig_adaptive_hard_forensics_meta.json`  
**Setup:** coastal USA Combined; same loss/decoder/data as `exp_coastal`; forensic subsample n=1200 from the test split (seed 42). Full-test FDE from official metrics.

---

## Report paragraph (paste into LyX)

We diagnosed why the separate-encoder hard-gated adaptive model improves on the original shared-encoder adaptive model by comparing hidden-state similarity, gate selection, motion conditioning, encoder weights, forget-gate biases, forced-context ablations, and input/gate gradient attribution. With a shared encoder the nested context states are nearly identical (cos(h₉,h₁₂)≈0.97, cos(h₁₈,h₂₄)≈0.99), so soft mixing has little to choose among and the gate collapses to 24h (~86% argmax); with four independent encoders the same pairs drop to ≈0.30 / 0.14 / 0.10, hard one-hot selection spreads over 9h / 12h / 24h (≈29% / 35% / 36%) and never uses 18h—consistent with a collapsed 18h encoder (parameter L2 ≈54 vs ≈150–173 for the others; gate-logit gradient mass on 18h ≈0). Straight trajectories are routed almost entirely to 24h (~95%), while maneuvers are split across 9/12/24; Spearman correlations between soft gate probabilities and motion features remain only weak-to-moderate (e.g. p(24h) vs mean SOG ≈+0.18, vs max|ΔCOG| ≈−0.21). Backprop through the hard model still concentrates on recent history (~29% of |∂L/∂x| on the newest 3h vs ~2% on the oldest 3h). Forced single-context decoding ranks 9h best (median FDE ≈20.8 km), then 24h / 12h / 18h, and the learned gate matches this oracle choice on only ~41% of the subsample (oracle median FDE ≈14.2 km). Overall, hard selection with distinct encoders improves full-test median FDE from 20.19 km to 19.08 km (~2.3× parameters: 1.90M→4.31M), by preventing redundant soft collapse rather than by learning a perfect motion-based context policy, and it remains slightly above the best fixed Flat LSTM (18.80 km).

---

## Main result

| Model | Params | Median FDE (full test) |
|-------|-------:|-----------------------:|
| Shared adaptive (original) | 1.90M | 20.19 km |
| **Separate encoders + hard gate** | **4.31M** | **19.08 km** |
| Flat LSTM (best overall) | 0.91M | 18.80 km |

On the forensic subsample: hard 19.03 km vs shared 20.75 km; hard better on **52.6%** of trajectories.

---

## Evidence 1 — Representations become distinct

| Pair | Shared cosine | Separate+hard cosine |
|------|--------------:|---------------------:|
| 9h–12h | 0.972 | 0.295 |
| 12h–18h | 0.978 | 0.141 |
| 18h–24h | 0.988 | 0.105 |

**Figure:** `fig_adaptive_separate_hidden_cos.png`

---

## Evidence 2 — Gate selection no longer collapses to 24h

| Context | Shared argmax % | Hard selection % |
|---------|----------------:|-----------------:|
| 9h | 0.4% | 29.2% |
| 12h | 1.6% | 35.0% |
| 18h | 11.9% | **0.0%** |
| 24h | **86.1%** | 35.8% |

**Figure:** `fig_adaptive_separate_selection.png`

---

## Evidence 3 — Partial motion structure (not a clean rule)

Hard selection within motion buckets:

| Bucket | 9h | 12h | 18h | 24h |
|--------|---:|----:|----:|----:|
| Straight | 5% | 0% | 0% | **95%** |
| Maneuver | 30% | 37% | 0% | 33% |
| Other | 14% | 18% | 0% | 68% |

Spearman r (soft p(context) vs history features):

| Context | mean SOG | max \|ΔCOG\| | straightness |
|---------|---------:|-------------:|-------------:|
| 9h | −0.22 | +0.06 | −0.06 |
| 12h | −0.13 | +0.18 | −0.28 |
| 24h | +0.18 | −0.21 | +0.14 |

**Figure:** `fig_adaptive_hard_selection_by_motion.png`

---

## Evidence 4 — Weights / forget bias (18h collapsed)

Encoder parameter L2: 9h=159, 12h=150, **18h=54**, 24h=173.

Forget-gate bias (encoder L0 mean): shared −0.16; separate 9/12/18/24h = −0.45 / −0.35 / −0.24 / −0.35.

**Figures:** `fig_adaptive_hard_encoder_l2.png`, `fig_adaptive_hard_forget_bias.png`

---

## Evidence 5 — Forced-context ablation + oracle gap

Forced one-hot context (subsample median FDE):

| Forced context | Median FDE |
|----------------|----------:|
| 9h | **20.80 km** |
| 24h | 21.42 km |
| 12h | 23.04 km |
| 18h | 24.92 km |

- Oracle (best forced per sample): **14.17 km**
- Gate matches oracle choice: **41.1%**

**Figure:** `fig_adaptive_hard_forced_vs_selected.png`

---

## Evidence 6 — Backprop attribution

Through separate+hard (soft path for grads):

- Newest 3h of history: **28.7%** of |∂L/∂x| mass  
- Oldest 3h: **2.3%**  
- Gate-logit gradient mass: 9h 43%, 12h 11%, **18h ≈0%**, 24h 46%

**Figure:** `fig_adaptive_hard_grad_by_hour.png`

---

## Takeaway for the report

Hard+separate succeeds **because independent encoders create distinguishable contexts and hard selection forces specialization**, undoing the shared-encoder soft collapse to 24h. It does **not** succeed by learning a strong explicit “maneuver→short / straight→long” policy; gradients still favor recent history, 18h dies, and a large gap remains to an oracle context chooser and to Flat LSTM.

---

## Figures checklist for the report

1. `fig_adaptive_separate_vs_shared_fde.png`
2. `fig_adaptive_separate_selection.png`
3. `fig_adaptive_separate_hidden_cos.png`
4. `fig_adaptive_hard_forced_vs_selected.png` *(optional / appendix)*
5. `fig_adaptive_hard_selection_by_motion.png` *(optional / appendix)*
6. `fig_adaptive_hard_grad_by_hour.png` *(optional / appendix)*

Path prefix: `data/results/USA Combined/unknown/exp_coastal/report_figures/`
