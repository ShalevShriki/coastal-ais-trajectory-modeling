# Adaptive separate-encoder hard gate — LyX paragraph + forensics

Use the **one paragraph** below as the report text. The rest is evidence for ChatGPT / mentor Q&A.

**Figures**

| File | Content |
|------|---------|
| `report_figures/fig_adaptive_separate_vs_shared_fde.png` | Shared vs hard: median FDE / ADE |
| `report_figures/fig_adaptive_separate_selection.png` | Argmax selection shared vs hard |
| `report_figures/fig_adaptive_separate_hidden_cos.png` | Hidden-state cosine (distinctness) |
| `report_figures/fig_adaptive_hard_forced_vs_selected.png` | Forced-context FDE vs how often selected |
| `report_figures/fig_adaptive_hard_selection_by_motion.png` | Selection % inside straight / maneuver / other |
| `report_figures/fig_adaptive_hard_grad_by_hour.png` | Input gradient mass by history hour |
| `report_figures/fig_adaptive_hard_forget_bias.png` | LSTM forget bias shared vs each encoder |
| `report_figures/fig_adaptive_hard_encoder_l2.png` | Encoder weight L2 (18h collapsed) |
| `report_figures/fig_adaptive_hard_forensics_meta.json` | Full numeric dump |

---

## One paragraph (for LyX)

To test whether the original adaptive model failed mainly because a shared encoder made the context states nearly redundant, we trained a follow-up with the same coastal data, loss, and AR decoder but four independent LSTM encoders (9h / 12h / 18h / 24h) and a hard Gumbel–Softmax / argmax gate (~4.31M parameters vs ~1.90M); relative to the shared-encoder baseline (median FDE 20.19 km, ~86% argmax on 24h, soft weights ≈ [0.15, 0.22, 0.29, 0.34], and cos(h₁₈,h₂₄)≈0.99), separate encoders produced clearly distinct states (cos(h₁₈,h₂₄)≈0.10), the hard gate spread selections across 9h / 12h / 24h (≈29% / 37% / 34%) while never using 18h (that encoder’s weight L2 collapsed to ~54 vs ~150–173 for the others, and gate-logit gradient mass on 18h was ≈0), straight trajectories were routed almost entirely to 24h (~95%) whereas maneuvers were split across 9/12/24, input gradients still concentrated on recent history (~29% newest 3h vs ~2% oldest 3h), and Spearman ties between gate probabilities and motion features remained only weak-to-moderate (e.g. p(24h) vs mean SOG ≈ +0.18, vs max|ΔCOG| ≈ −0.21), yet hard selection improved full-test median FDE to 19.08 km—showing that forcing a single specialized context helps once representations are no longer redundant, without recovering a perfect motion oracle (gate matched the best forced-context choice on only ~41% of a test subsample; oracle median FDE ≈14.2 km), and still remaining slightly above the best fixed Flat LSTM (18.80 km).

---

## Why hard+separate wins (short)

1. **Breaks redundancy.** Shared encoder: cos(h₉,h₁₂)≈0.97, cos(h₁₈,h₂₄)≈0.99. Separate: ≈0.30 / 0.14 / 0.10. Soft mixing near-duplicates had almost nothing to choose among; hard selection now has distinct options.
2. **Forces specialization.** One-hot decoder init means each selected encoder must carry the full context alone — no free-riding on a soft blend toward 24h.
3. **Stops the 24h collapse.** Shared argmax ≈86% on 24h. Hard ≈29/35/0/36% on 9/12/18/24.
4. **Partial motion alignment.** Straight → almost always 24h (~95%). Maneuvers → split 9/12/24. Not a clean rule, but not pure geography collapse either.
5. **Capacity.** ~2.3× parameters, almost all in the four encoders.

## Caveats / failures still present

- **18h is dead:** never selected; encoder L2 ~3× smaller; gate grad mass ≈0.
- **Not an oracle:** forced-context ranking on subsample is 9h (20.8) < 24h (21.4) < 12h (23.0) < 18h (24.9); gate only matches the best forced choice ~41% of the time.
- **Motion Spearman still weak** — no strong “maneuver→short” policy.
- **Still below Flat LSTM** (18.80 km).

## Key numbers (full coastal test + n=1200 forensic subsample)

| | Shared adaptive | Separate + Hard |
|--|--:|--:|
| Params | 1.90M | 4.31M |
| Median FDE (full test) | 20.19 km | **19.08 km** |
| Argmax | ~86% @ 24h | ~29/37/0/34% @ 9/12/18/24 |
| cos(h₁₈,h₂₄) | ≈0.99 | ≈0.10 |
| Forget bias L0 | −0.16 | −0.45 / −0.35 / −0.24 / −0.35 (9/12/18/24) |

Subsample (n=1200): hard median FDE 19.03 vs shared 20.75; hard better on ~53% of trajectories; oracle forced-context median FDE ≈14.2 km.

## Reproduce

```bash
source scripts/exp_coastal/_env.sh && cd "$SUBROOT"
$PYTHON scripts/forensics_adaptive_hard.py
```
