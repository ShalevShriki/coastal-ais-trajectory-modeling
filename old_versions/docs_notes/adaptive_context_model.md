# Adaptive Context Model: Architecture, Training, Interpretation, and Main Finding

**Status:** Corrected write-up aligned with the implementation in `models/RNN_AR_adaptive.py` and the `exp_coastal` / `exp_final` results.  
**Primary numbers below:** coastal suite (`combined_filtered_smart_coastal`) unless noted.

---

## 1. Motivation

The adaptive-context model tests the hypothesis:

> Different vessel trajectories may require different amounts of historical information.

Examples:

- A vessel moving in a **stable, near-straight** direction may benefit from a **long** history, because its long-term pattern remains relevant.
- A vessel that recently **changed speed or course** may benefit from a **short** history, because older observations may no longer describe current behavior.

Instead of one fixed history length for the whole dataset, the model receives four **suffixes** of the same 24h window:

\[
X_9,\; X_{12},\; X_{18},\; X_{24}
\]

where \(X_k\) is the last \(k\) hours of AIS observations (at 10 min sampling: 54 / 72 / 108 / 144 steps).

It then learns, **per sample**, how much weight to assign to each length.

**Intended behavior (hypothesis):**

- maneuvering vessel \(\Rightarrow\) higher weight on short history  
- stable / straight vessel \(\Rightarrow\) higher weight on long history  

**Research question:** Can the model select temporal context automatically, instead of using one fixed context for all vessels?

---

## 2. General architecture

Three components:

1. **History encoder** (shared LSTM; see §3)  
2. **Gating MLP**  
3. **Autoregressive trajectory decoder** (LSTM)

Pipeline:

\[
\{X_9,X_{12},X_{18},X_{24}\}
\;\xrightarrow{\text{shared encoder}}\;
\{h_9,h_{12},h_{18},h_{24}\}
\;\xrightarrow{\text{gate MLP + Softmax}}\;
\{\alpha_9,\alpha_{12},\alpha_{18},\alpha_{24}\}
\]
\[
h_{\text{adaptive}}=\sum_k \alpha_k h_k
\;\xrightarrow{\text{AR decoder}}\;
\hat Y \in \mathbb{R}^{72\times 2}
\]

(\(\hat Y\): 12h future at 10 min = **72** steps, not 36.)

Optional variant (`--gate-vessel-type`): concatenate a vessel-type embedding into the gate input. That run (`exp_final/adaptive_vessel_type`) did **not** improve FDE vs plain adaptive.

---

## 3. Encoding the different history windows

### Important implementation detail

There are **not** four independent encoder networks in the code. There is **one shared LSTM encoder**. For each context length \(k\), the model encodes the last \(k\) hours of the same input tensor:

```text
x_slice = x[:, -n_steps:, :]
_, enc_hidden = self.encoder(x_slice)
h_k = last_layer_hidden(enc_hidden)
```

So:

\[
h_k = \mathrm{Encoder}_{\theta}(X_k),\qquad
h_k \in \mathbb{R}^{256}
\]

with the **same** \(\theta\) for all \(k\).

Each \(h_k\) is a learned summary of that suffix (motion, turns, region cues, etc.), not geographic coordinates.

Because windows are nested (\(X_9 \subset X_{12} \subset X_{18} \subset X_{24}\)) and the encoder is shared, the four representations are **highly correlated**, which makes sharp specialization harder.

---

## 4. The gating MLP

### 4.1 Exact architecture (as implemented)

Concatenate the four hidden vectors:

\[
z = [h_9;\, h_{12};\, h_{18};\, h_{24}] \in \mathbb{R}^{1024}
\]

(with vessel-type embedding: \(z \in \mathbb{R}^{1024+16}\)).

The gate is:

\[
\begin{aligned}
z &\xrightarrow{\mathrm{Linear}(1024\to 256)} \mathbb{R}^{256} \\
&\xrightarrow{\mathrm{ReLU}} \\
&\xrightarrow{\mathrm{Dropout}(0.2)} \\
&\xrightarrow{\mathrm{Linear}(256\to 4)} (s_9,s_{12},s_{18},s_{24}).
\end{aligned}
\]

Code (`RNN_AR_adaptive.py`):

```python
self.gate = nn.Sequential(
    nn.Linear(gate_in_dim, hidden_dim),   # 1024 → 256
    nn.ReLU(),
    nn.Dropout(dropout),                  # 0.2
    nn.Linear(hidden_dim, 4),             # 256 → 4 logits
)
```

**Correction vs some drafts:** the hidden width is **256**, not 128. There is also **Dropout** between ReLU and the final linear layer.

### 4.2 Softmax

\[
\alpha_k = \frac{\exp(s_k)}{\sum_{j\in\{9,12,18,24\}} \exp(s_j)},
\qquad
\alpha_k \ge 0,\quad \sum_k \alpha_k = 1.
\]

The gate usually produces a **soft mixture**, not a hard one-hot choice—though in practice one context often dominates (see §9).

### 4.3 Arc diagram (gate only)

```text
h9,h12,h18,h24  (each ℝ^256)
        │
        ▼ concat
   z ∈ ℝ^1024
        │
        ▼ Linear 1024→256
   u ∈ ℝ^256
        │
        ▼ ReLU + Dropout(0.2)
        │
        ▼ Linear 256→4
   logits s ∈ ℝ^4
        │
        ▼ Softmax
   α = (α9, α12, α18, α24)
        │
        ▼ weighted sum
   h_adaptive = Σ α_k h_k   ∈ ℝ^256
```

---

## 5. Adaptive context and decoding

\[
h_{\text{adaptive}} = \sum_k \alpha_k h_k
\]

This vector initializes the **autoregressive LSTM decoder**, which predicts future lat/lon offsets step by step for **72** steps (12h), with teacher forcing during training and free-running at test time.

**Order:** Gate → fused context → Decoder.  
The decoder does **not** choose \(\alpha\); it consumes the gate’s decision.

**Dynamics of \(\alpha\):**

- **Yes**, \(\alpha^{(i)}\) is **sample-dependent** (dynamic across trajectories).  
- **No**, \(\alpha\) is **not** recomputed at every decoder step in this design: one gate decision per window, then the full 12h rollout uses that fixed context.

(A per-step gate would be a different, more complex architecture.)

---

## 6. How the gate is trained

There is **no** label such as “correct history length = 12h.”

The gate is trained **end-to-end** through the trajectory loss only (indirect supervision).

Coastal runs typically use:

\[
\mathcal{L} = (1-w)\,L_{\mathrm{Huber}} + w\,L_{\mathrm{geo}} + L_{\mathrm{land}}
\]

with \(w=0.5\) and optional soft land penalty (`--land-penalty-weight 0.1` in `exp_coastal`).

Gradients flow:

\[
\mathcal{L} \to \text{Decoder} \to h_{\text{adaptive}} \to \alpha \to \text{Gate MLP} \to \text{Encoder}.
\]

All of \(\{\theta_{\text{encoder}},\theta_{\text{gate}},\theta_{\text{decoder}}\}\) update jointly.

---

## 7. Connection to attention and mixture of experts

**Attention-like:** Softmax weights + weighted sum of “values” \(h_k\).  
Difference: scores come from an MLP on concatenated hiddens, not from \(QK^\top / \sqrt{d}\).

**Soft MoE-like:** each context length is an “expert” \(E_k(X)=h_k\), mixed by \(\alpha_k\).  
Caveat: nested overlapping windows + **shared** encoder limit true specialization.

---

## 8. Results (correct numbers)

### 8.1 Adaptive performance

| Suite | Median FDE @ 12h | Median ADE | Notes |
|-------|------------------|------------|--------|
| **`exp_coastal` adaptive** | **20.19 km** | **7.64 km** | Inland filtered; land penalty 0.1 |
| `exp_final` adaptive | 19.66 km | 7.66 km | Mixed smart-motion; no inland filter |

The model is a competent predictor, but it **does not beat** the best simpler models on the same coastal split.

### 8.2 Coastal suite ranking (median FDE @ 12h)

| Rank | Model | FDE |
|------|--------|-----|
| 1 | Flat LSTM (24h→12h) | **18.80 km** |
| 2 | Transformer (24h→12h) | 19.40 km |
| 3 | AR LSTM 12h (no land penalty) | 19.59 km |
| 4 | AR LSTM 18h | 19.71 km |
| 5 | AR LSTM 12h (land penalty 0.1) | 19.99 km |
| 6 | **Adaptive multiscale** | **20.19 km** |
| 7 | AR LSTM 24h | 20.30 km |
| 8 | AR LSTM 9h | 20.43 km |
| 9 | Sliding 3h×4 | 22.44 km |

**Negative result:** extra flexibility (multi-scale gate) did **not** improve FDE / ADE vs flat or strong fixed AR.

---

## 9. What the average alphas mean (use the real values)

### Coastal adaptive (test set)

| Context | Mean \(\bar\alpha_k\) | Argmax share |
|---------|----------------------|--------------|
| 9h | **0.146** | 0.4% |
| 12h | **0.224** | 1.6% |
| 18h | **0.289** | 11.9% |
| **24h** | **0.341** | **86.1%** |

### `exp_final` adaptive (for comparison)

Mean \(\alpha\): 9h 0.106, 12h 0.210, 18h 0.329, 24h 0.355 (also dominated by long context).

**Correction:** Do **not** report a balanced toy average such as \((0.22, 0.29, 0.27, 0.22)\). Empirically the gate is **not** balanced: **~86% of coastal samples have \(\arg\max \alpha = 24\mathrm{h}\)**. Soft weights still mix scales, but the decision is strongly biased toward long context.

Averages alone can hide that concentration; reporting **argmax %** (as above) is required.

---

## 10. What we expected vs what we observed

**Expected:** motion-based rule  
`maneuver → short context`, `straight/stable → long context`.

**Observed:** only **weak** Spearman associations with motion; **Random Forest** says **anchor latitude/longitude** dominate predicting \(\alpha\).

Gate analysis artifacts:

- `.../exp_coastal/.../gate_feature_drivers.html`
- script: `scripts/analyze_adaptive_gate_drivers.py`

### Spearman (pairwise, weak)

Examples of tendencies (coastal analysis, \(|\rho_s|\) typically \(\lesssim 0.17\)):

- Higher `path_km_24h` / `mean_sog` → slightly more \(\alpha_{12},\alpha_{18}\), less \(\alpha_{24}\)
- Higher accel / speed change → slightly more \(\alpha_9\)
- More maneuver (`|dcog|`) → slightly less \(\alpha_{18}\)

These partially nod at the hypothesis but are **too weak** to claim a clear learned rule.

### Random Forest (joint, post-hoc)

RF is **not** part of the neural model. It only tries to predict the already-computed \(\alpha\) from interpretable features.

Top drivers of \(\alpha\) / of \((\alpha_{24}-\alpha_9)\):

1. **`anchor_lon`**, **`anchor_lat`** (region)  
2. then `std_sog`, accel, turns / path length  

So the gate behaves partly as a **geographic prior**, not primarily as a motion→context controller.

That is a form of **shortcut learning**: location is an easy statistical cue for reducing loss; the objective never forced “use maneuver features for \(\alpha\).”

---

## 11. Why the adaptive model did not win

1. **Wrong (or unintended) signal for the gate** — geography over dynamics.  
2. **Nested windows + shared encoder** — experts are not cleanly separable.  
3. **Higher capacity ≠ better generalization** — more ways to fit shortcuts.  
4. **No direct gate supervision** — only trajectory loss.  
5. **Soft averaging** can blur specialization when several \(\alpha_k\) are non-trivial.  
6. Fixed short AR (especially on `exp_final`) or flat LSTM impose a stronger inductive bias that happened to generalize better.

---

## 12. Course concepts (short map)

| Concept | Where it appears |
|---------|------------------|
| Softmax | Gate normalization of 4 logits |
| Attention-like pooling | \(\sum_k \alpha_k h_k\) over temporal scales |
| End-to-end backprop | Gate trained without \(\alpha\) labels |
| Soft MoE | Context lengths as experts |
| Inductive bias | Fixed context vs adaptive mixture |
| Shortcut learning | Location-driven \(\alpha\) |
| Ablation / negative result | Adaptive vs flat / AR / Transformer |
| Post-hoc interpretability | Spearman + RF (not the predictor itself) |

---

## 13. Final interpretation

The adaptive model **does** compute **sample-dependent** weights:

\[
\alpha^{(i)}=\mathrm{Softmax}\bigl(\mathrm{MLP}(z^{(i)})\bigr),\qquad
h_{\mathrm{adaptive}}^{(i)}=\sum_k \alpha_k^{(i)} h_k^{(i)}.
\]

On coastal data it reaches **median FDE 20.19 km** and **median ADE 7.64 km**, but loses to Flat LSTM (**18.80 km**) and strong AR / Transformer baselines.

Empirically \(\alpha\) is dominated by **24h** (~86% argmax). Motion–\(\alpha\) correlations are weak; the strongest post-hoc predictors of the gate are **anchor lat/lon**.

**Main conclusion:**

> Adaptive context selection did not primarily learn a universal motion-based rule for choosing history length. It partially learned a **geographic prior** over context mixtures. Extra architectural flexibility therefore did **not** yield better generalization than simpler fixed-context models.

**Broader lesson:**

> Greater flexibility does not guarantee better generalization; adaptive mechanisms need the right inductive bias (or supervision) to specialize as intended.

---

## Appendix A — Implementation checklist (for report accuracy)

| Claim | Correct value |
|-------|----------------|
| Encoder | **One shared** LSTM (2 layers, hidden 256), not 4 separate nets |
| Gate MLP | \(1024 \to 256 \to \mathrm{ReLU} \to \mathrm{Dropout}(0.2) \to 4\) |
| Future horizon | **72** steps (12h), not 36 |
| Coastal adaptive FDE / ADE | **20.19 / 7.64 km** |
| Mean \(\alpha\) (coastal) | **0.146 / 0.224 / 0.289 / 0.341** |
| Argmax 24h | **~86%** |
| Best coastal model (this suite) | Flat LSTM **18.80 km** |

## Appendix B — Key code / result paths

- Model: `models/RNN_AR_adaptive.py`  
- Coastal metrics: `data/results/USA Combined/unknown/exp_coastal/adaptive_multiscale/RNN_AR_adaptive/adaptive_ar_metrics.json`  
- Per-sample \(\alpha\): `.../context_alpha_weights.json`  
- Gate feature report: `.../gate_feature_drivers.html`
