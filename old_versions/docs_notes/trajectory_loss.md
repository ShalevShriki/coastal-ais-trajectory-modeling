# Trajectory Loss — Full Reference

One place for **what we optimize**, **every symbol**, and **whether to simplify**.

Implementation: `models/training_utils.py` (`TrajectoryLoss`, `HourDisplacementLoss` in `RNN_recursive_1h.py`).

---

## 1. Coordinate system and symbols

| Symbol | Name | Units | Meaning |
|--------|------|-------|---------|
| **lat** | Latitude | degrees (°) | North–south position. + = north of equator. US coasts ≈ 25°–50°N. |
| **lon** | Longitude | degrees (°) | East–west position. US ≈ −125° to −65° (west is negative). |
| **P** | Position | (lat, lon) | A point on Earth. We write **P_t = (lat_t, lon_t)**. |
| **P₀** / **anchor** | Anchor | (lat, lon) | Last position in the **history** window — “where the ship is now” before we predict. |
| **history** | Past track | 144 steps × 15 features | 24 hours of AIS, resampled every **10 minutes**. |
| **future** | Ground truth track | 72 steps × 2 | Next **12 hours** of true (lat, lon), every 10 min. |
| **t** | Time index | 0 … 71 | Future step. t=0 is +10 min, t=71 is +12 h. |
| **Δt** | Step size | 10 min | Fixed resampling interval. |

### History features (inputs — not in the loss directly)

| Feature | Meaning |
|---------|---------|
| **sog** | Speed over ground (knots) |
| **cog_sin, cog_cos** | Course over ground as sin/cos (avoids 0°/360° wrap) |
| **heading_sin, heading_cos** | Vessel heading (where the bow points) |
| **heading_missing** | 1 if heading was missing in AIS, else 0 |
| **dt_sec** | Seconds since previous AIS point |
| **dlat, dlon** | Change in lat/lon since previous step (degrees) |
| **dsog, dcog** | Change in speed / course |
| **v_north_kmh, v_east_kmh** | Velocity in km/h in local north/east frame |

The model sees scaled history; the **loss** compares predicted vs true **future positions** (via deltas below).

---

## 2. What the model predicts

The network outputs a tensor **ŷ** with shape `[batch, 72, 2]` — one (lat, lon) **delta** per future step.

Two target modes (see `window_data.build_targets`):

### Mode A — `anchor_offset` (default, most runs)

True target at step t:

\[
y_t = P_t - P_0 \quad \text{(in degrees, both lat and lon)}
\]

Absolute position from prediction:

\[
\hat{P}_t = P_0 + \hat{y}_t
\]

### Mode B — `step_delta` (experiment M2)

True target at step t:

\[
y_t = P_t - P_{t-1}, \quad y_0 = P_0 - P_{\text{anchor}}
\]

Reconstruct absolute positions by cumulative sum:

\[
\hat{P}_t = P_0 + \sum_{k=0}^{t} \hat{y}_k
\]

### Residual mode (optional, `--residual-naive`)

Model predicts a **correction** on top of a naive baseline (constant velocity from SOG+COG):

\[
\hat{y}_t \leftarrow \hat{y}_t + y^{\text{naive}}_t
\]

Loss is still computed on the **final** \(\hat{y}\) after adding the baseline.

---

## 3. Main training loss — `TrajectoryLoss`

For one sample \(i\), let \(T\) = number of future steps used (72 full horizon, or fewer with curriculum).

### 3.1 Term 1 — Huber on deltas (degrees)

PyTorch `HuberLoss(δ=0.01)` on each component of lat/lon delta, then mean over time and coordinates:

\[
L_{\text{Huber}}^{(i)} = \frac{1}{T \cdot 2} \sum_{t=0}^{T-1} \sum_{c \in \{\text{lat},\text{lon}\}} \text{Huber}_\delta\!\left(\hat{y}_{t,c}^{(i)} - y_{t,c}^{(i)}\right)
\]

**Huber** behaves like MSE for small errors and like MAE for large errors (robust to outliers).  
**δ = 0.01°** ≈ 1.1 km in latitude — errors larger than that are penalized linearly, not quadratically.

**Why degrees?** The model head outputs lat/lon offsets in the same units as the targets.

---

### 3.2 Term 2 — Geographic distance on absolute positions (km, scaled)

Rebuild true and predicted absolute tracks \(\hat{P}_t, P_t\), then per-step **local planar distance** in km:

\[
d_t^{(i)} = \sqrt{ \left(\Delta\text{lat}_\text{km}\right)^2 + \left(\Delta\text{lon}_\text{km}\right)^2 }
\]

where (mid-latitude planar approximation, used in training for stable gradients):

\[
\Delta\text{lat}_\text{km} = (\hat{\text{lat}}_t - \text{lat}_t) \times 111.322
\]
\[
\Delta\text{lon}_\text{km} = (\hat{\text{lon}}_t - \text{lon}_t) \times 111.322 \times \cos\!\left(\frac{\hat{\text{lat}}_t + \text{lat}_t}{2}\right)
\]

**111.322 km/°** ≈ km per degree of latitude (constant). Longitude degrees shrink by \(\cos(\text{lat})\) away from the equator.

Average over time, then **divide by 50** to match Huber scale (~0.02° ≈ 2 km → 2/50 = 0.04):

\[
L_{\text{geo}}^{(i)} = \frac{1}{T} \sum_{t=0}^{T-1} \frac{d_t^{(i)}}{50}
\]

> **Note:** Despite the name `haversine_weight`, training uses this **local km** formula, not full Haversine. **Evaluation** uses exact Haversine (`window_data.haversine_km`).

---

### 3.3 Combined loss (default weights)

Weight \(w = 0.5\) by default (`--haversine-loss-weight`):

\[
L_{\text{base}}^{(i)} = (1 - w)\, L_{\text{Huber}}^{(i)} + w\, L_{\text{geo}}^{(i)}
\]

**Default:** 50% Huber on degree-deltas + 50% scaled km error on positions.

---

### 3.4 Optional — Relative ADE term (off by default)

When `--relative-loss-weight > 0`:

\[
L_{\text{rel}}^{(i)} = \frac{\frac{1}{T}\sum_t d_t^{(i)}}{\max\!\left(L_{\text{path}}^{(i)},\, 10\text{ km}\right)}
\]

where \(L_{\text{path}}^{(i)}\) = sum of step lengths along the **true** future track.

\[
L^{(i)} = L_{\text{base}}^{(i)} + \lambda_{\text{rel}}\, L_{\text{rel}}^{(i)}
\]

Default \(\lambda_{\text{rel}} = 0\) (disabled).

---

### 3.5 Optional — Per-sample difficulty weight

From history maneuvering (`|dcog|` + `|dsog|`):

\[
w^{(i)} = 1 + \frac{\text{score}^{(i)}}{\text{median}(\text{score})}
\]

\[
L^{(i)} \leftarrow w^{(i)} \cdot L^{(i)}
\]

**On by default** unless `--no-difficulty-weighting`. Upweights turning/accelerating vessels.

---

### 3.6 Batch loss

\[
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} L^{(i)}
\]

---

## 4. Full equation (as implemented today)

\[
\boxed{
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} w^{(i)} \left[
(1-w)\, L_{\text{Huber}}^{(i)} + w\, L_{\text{geo}}^{(i)} + \lambda_{\text{rel}}\, L_{\text{rel}}^{(i)}
\right]
}
\]

| Parameter | Default | CLI flag |
|-----------|---------|----------|
| \(w\) (geo weight) | 0.5 | `--haversine-loss-weight` |
| Huber δ | 0.01° | (fixed in code) |
| km scale divisor | 50 | (fixed in code) |
| \(\lambda_{\text{rel}}\) | 0.0 | `--relative-loss-weight` |
| Difficulty weights \(w^{(i)}\) | on | `--no-difficulty-weighting` to disable |
| Curriculum (shorter \(T\) early) | on in code, **off in exp_clean** | `--no-curriculum` |

---

## 5. Recursive chunk loss — `HourDisplacementLoss` (M3)

Used by `RNN_recursive_1h.py` for 1h or 3h chunk models.

Predict **one displacement** \(\Delta P\) from anchor to end of chunk (not full 72-step path):

\[
L_{\text{Huber}} = \text{Huber}(\hat{\Delta P}, \Delta P)
\]
\[
L_{\text{geo}} = \frac{\text{km\_error}(P_0 + \hat{\Delta P},\; P_0 + \Delta P)}{50}
\]
\[
\mathcal{L} = (1-w)\, L_{\text{Huber}} + w\, L_{\text{geo}}
\]

Same 50/50 blend, but only **one endpoint** per training sample (full 12h built at inference by rolling the model forward).

---

## 6. Training loss vs evaluation metrics

| | **Training** (`TrajectoryLoss`) | **Evaluation** (test JSON) |
|--|--------------------------------|---------------------------|
| Distance formula | Local planar km | **Haversine** great-circle km |
| What is compared | All \(T\) future steps (or curriculum prefix) | **FDE** at 12h, **ADE** over full path |
| Units in loss | Mixed: degrees + km/50 | Pure km (and nm) |
| Goal | Smooth, differentiable objective | Interpretable maritime error |

**FDE** (Final Displacement Error): Haversine distance at the last step (12h).  
**ADE** (Average Displacement Error): mean Haversine over all 72 steps.

---

## 7. Should we simplify?

The current loss is **more complex than necessary** for a first baseline. Here is a honest breakdown.

### What is doing useful work

1. **Geographic term** — Optimizing in km (not raw lon) matters: 1° lon at 40°N ≈ 85 km, but 1° lat ≈ 111 km. The \(\cos(\text{lat})\) factor fixes that.
2. **Huber on deltas** — Stable training; model outputs match target parameterization.
3. **anchor_offset targets** — Simple to decode: \(\hat{P}_t = P_0 + \hat{y}_t\).

### What adds complexity (candidates to remove or fix)

| Piece | Issue | Simpler alternative |
|-------|--------|---------------------|
| Two-term blend (Huber + geo) | Two objectives, magic `/50` scaling | **Pick one:** e.g. only mean km error on \(\hat{P}_t\) |
| Local km in train, Haversine in eval | Train/eval mismatch | Use same Haversine in both (or accept mismatch) |
| `haversine_weight` name | Misleading — not Haversine in training | Rename to `geo_weight` |
| Difficulty weights | Harder to compare runs | Disable for clean ablations (`--no-difficulty-weighting`) |
| Relative loss term | Extra hyperparameter | Keep at 0 unless needed |
| Curriculum | Changes \(T\) during training | Already off in `exp_clean` |
| step_delta mode | Harder AR rollout | Stick to anchor_offset unless M2 wins |

### Recommended simple loss (if we refactor)

**Option A — Minimal (recommended starting point)**

\[
\mathcal{L} = \frac{1}{N\,T} \sum_{i,t} \text{Haversine\_km}(\hat{P}_t^{(i)}, P_t^{(i)})
\]

One term, one unit (km), matches evaluation. Use `haversine_km_torch` from `training_utils.py`.

**Option B — Keep robustness, drop magic constants**

\[
\mathcal{L} = \frac{1}{N\,T} \sum_{i,t} \left[ \text{Huber}(\hat{y}_t - y_t) + \lambda \cdot \text{km}(\hat{P}_t, P_t) \right]
\]

with \(\lambda\) chosen so both terms have similar magnitude on a val batch (no `/50` hack).

**Option C — What papers often use for trajectory forecasting**

\[
\mathcal{L}_{\text{ADE}} = \frac{1}{NT}\sum_{i,t} d_t^{(i)} \quad\text{+ optional }\quad \mathcal{L}_{\text{FDE}} = \frac{1}{N}\sum_i d_{T-1}^{(i)}
\]

Directly optimize average and final km error.

### Practical recommendation for this project

1. **Short term (no code change):** Keep current loss for `exp_clean` so results are comparable to `v1/smart_motion`.
2. **Next experiment sweep:** Try **Option A** (pure mean Haversine ADE) on one anchor model (A0) and compare val FDE to the 50/50 blend.
3. **Always report** Haversine FDE/ADE in km — that is what matters operationally.

---

## 8. Quick reference — data flow

```
History (24h, 15 features)  →  Model  →  ŷ (72 × [Δlat, Δlon])
                                              ↓
                                    + anchor P₀, + optional naive baseline
                                              ↓
                                    P̂_t  vs  P_t  (true future)
                                              ↓
                              TrajectoryLoss  →  backprop
                                              ↓
                         Eval: Haversine FDE @ 12h, ADE over path
```

---

## 9. Code map

| File | Role |
|------|------|
| `models/training_utils.py` | `TrajectoryLoss`, `local_km_error_torch`, `haversine_km_torch` |
| `models/RNN_recursive_1h.py` | `HourDisplacementLoss` |
| `window_data.py` | `build_targets`, `haversine_km`, `evaluate_final_position`, `evaluate_full_trajectory` |
| `models/RNN.py`, `RNN_AR.py`, `transformers.py` | Wire up `TrajectoryLoss` + curriculum + weights |
