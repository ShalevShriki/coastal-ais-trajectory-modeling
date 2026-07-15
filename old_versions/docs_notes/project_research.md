# AIS Vessel Trajectory Prediction — Context Window Learning Research Plan

## 1. Project Framing

This project studies **AIS vessel trajectory prediction** as a **sequence modeling problem**.

The main goal is not only to compare different neural models, but to investigate a more specific research question:

> **How much past trajectory context is useful for predicting a vessel trajectory 12 hours into the future, and can an RNN-based model learn the relevant context window size by itself?**

The project focuses on **temporal context length** in sequence models.

Instead of assuming that every trajectory should use the same fixed history window, we want to test whether different motion patterns require different amounts of past information.

For example:

```text
Stable straight motion may only need shorter context.
Turning or maneuvering motion may need longer context.
Speed changes may require longer temporal history.
Dense coastal or port-like regions may require longer context.
```

The final research direction is to build and analyze a model that can **learn how much context it needs**.

---

## 2. Main Prediction Task

The final prediction task is:

```text
Predict 12 hours into the future.
```

The prediction horizon is fixed across the main experiments.

The main variable is the amount of past trajectory history given to the model.

We will compare history windows of:

```text
9 hours
12 hours
18 hours
24 hours
```

So the core setup is:

```text
9h  history -> 12h future
12h history -> 12h future
18h history -> 12h future
24h history -> 12h future
```

We do not use very short windows such as 3h for the main fixed-context experiments, because predicting 12h from only 3h history may be too extreme and less informative. The selected windows are more realistic while still allowing us to study context-length effects.

---

## 3. Main Research Question

The main research question is:

```text
How does temporal context length affect 12-hour AIS trajectory prediction,
and can an autoregressive RNN learn which context window is useful for each trajectory?
```

This breaks into several sub-questions:

```text
1. Does longer history improve 12-hour trajectory prediction?
2. Is 24h history always better than 9h or 12h history?
3. Does long context help mainly in maneuvers and speed changes?
4. Can too much history add noise for simple straight trajectories?
5. Can an RNN learn a soft context-window preference by itself?
6. Which motion features explain the selected context length?
```

---

## 4. Connection to Course Topics

This project connects directly to course topics in deep learning and sequence modeling:

```text
RNNs
LSTMs
Autoregressive sequence prediction
Encoder-decoder models
Long-range dependencies
Temporal context
Attention
Transformers
Representation learning
Interpretability through learned weights
```

The key course-related idea is the problem of **how sequence models use past information**.

An RNN compresses the past into a hidden representation. A Transformer can attend over different parts of the past. The adaptive context model explicitly learns how much weight to assign to different history windows.

---

# 5. Final Experiment List

The final experiment list should be:

```text
1. Kinematic baseline

2. Flat RNN / Flat LSTM:
   24h history -> 12h future

3. Fixed Context RNN_AR + Temporal Context + Anchor:
   9h  history -> 12h future
   12h history -> 12h future
   18h history -> 12h future
   24h history -> 12h future

4. Receding-Horizon Sliding Window:
   24h history -> 3h future
   update window
   repeat 4 times -> 12h future

5. Transformer:
   24h history -> 12h future

6. Adaptive Multi-Scale RNN_AR:
   9h + 12h + 18h + 24h contexts
   -> learned context weights
   -> 12h future
```

The most important experiment is:

```text
Adaptive Multi-Scale RNN_AR
```

because it directly tests whether the model can learn the useful context window size.

---

# 6. Baseline

Use only one baseline:

## Kinematic Constant-Velocity Baseline

The baseline should use the vessel’s last known position, speed, and course to extrapolate the future trajectory.

Input:

```text
last known latitude
last known longitude
last known SOG
last known COG
```

Prediction:

```text
continue moving with the same speed and course for 12 hours
```

This is a strong and meaningful baseline for vessel trajectories because many vessels move smoothly and approximately maintain their speed/course over short-to-medium horizons.

This baseline answers:

```text
Do the neural models actually learn more than simple physical motion continuation?
```

We choose this baseline instead of a naive “repeat last position” baseline because vessels usually keep moving, so a constant-position baseline is often too weak and less informative.

---

# 7. Experiment 1 — Flat RNN / Flat LSTM

## Goal

This is a simple direct prediction model used as a reference.

Setup:

```text
24h history -> 12h future
```

The model receives the full 24h trajectory history and predicts the entire 12h future trajectory in one forward pass.

This is also called:

```text
direct multi-horizon prediction
```

or:

```text
one-shot future prediction
```

## Purpose

The purpose is to compare autoregressive prediction against a simple direct sequence model.

Research question:

```text
Does autoregressive prediction provide an advantage over predicting the full future at once?
```

## Why only one Flat RNN?

We do not run Flat RNN for all context lengths because the project focus is not direct prediction. The main focus is context learning in autoregressive RNNs.

Therefore, one Flat RNN with 24h context is enough as a reference.

---

# 8. Experiment 2 — Fixed Context RNN_AR

## Goal

This is the main fixed-window experiment.

We train the same autoregressive RNN model with different fixed context lengths:

```text
RNN_AR + TC + Anchor, 9h  history -> 12h future
RNN_AR + TC + Anchor, 12h history -> 12h future
RNN_AR + TC + Anchor, 18h history -> 12h future
RNN_AR + TC + Anchor, 24h history -> 12h future
```

## Model Type

Use:

```text
RNN_AR + Temporal Context + Anchor
```

This means:

```text
RNN_AR:
The model predicts the future autoregressively.

Temporal Context:
The model receives an encoded representation of the past trajectory.

Anchor:
The model predicts future positions relative to a fixed anchor, usually the last observed position.
```

## Why RNN_AR?

Autoregressive prediction is natural for trajectory forecasting because the model generates the future sequence step by step.

Instead of outputting the entire future in one shot, it predicts future points sequentially.

General structure:

```text
history -> predict next future point
predicted point -> help predict the next point
repeat until 12h future is produced
```

## Why Temporal Context?

The whole research question is about how much history is useful.

Therefore, the AR decoder should have access to temporal context from the encoder.

Without temporal context, the model would be less relevant to the context-length research question.

## Why Anchor?

The anchor representation helps reduce drift.

Instead of predicting tiny deltas that accumulate error endlessly, the model predicts future positions relative to a stable reference point.

This makes the experiment more about context length and less about uncontrolled error accumulation.

## Research Questions

This experiment answers:

```text
How much history does an autoregressive RNN need?
Is 24h better than 18h, 12h, or 9h?
Does long context help mainly in difficult trajectories?
Does short context perform similarly on stable trajectories?
```

---

# 9. Experiment 3 — Receding-Horizon Sliding Window

## Goal

This experiment tests a different forecasting strategy.

Instead of predicting the full 12h future directly, the model predicts the future in shorter chunks.

Setup:

```text
24h history -> predict next 3h
update the 24h window
predict next 3h
update the 24h window
predict next 3h
update the 24h window
predict next 3h
```

After four chunks, we obtain:

```text
12h future prediction
```

## Detailed Example

Step 1:

```text
history: 0h-24h
predict: 24h-27h
```

Step 2:

```text
history: 3h-27h
predict: 27h-30h
```

Step 3:

```text
history: 6h-30h
predict: 30h-33h
```

Step 4:

```text
history: 9h-33h
predict: 33h-36h
```

The final prediction covers:

```text
24h-36h = 12h future
```

## Important Detail

After the first prediction chunk, the updated history window contains predicted points.

So this method uses its own predictions as part of the future context.

## Model

Use the same general architecture family:

```text
RNN_AR + TC + Anchor
```

but train it for:

```text
24h history -> 3h future
```

Then apply it repeatedly during inference to reach 12h.

## Research Question

This experiment asks:

```text
Is it better to predict 12h directly,
or to predict shorter chunks while updating the context window?
```

## Why This Matters

This is directly related to temporal context.

The sliding-window method repeatedly forgets old information and focuses on the most recent 24h window.

Potential advantage:

```text
The context stays updated and local.
```

Potential risk:

```text
Prediction errors enter the next input window and may accumulate.
```

This gives a meaningful comparison between:

```text
Direct 12h autoregressive rollout
vs
Chunked 3h receding-horizon rollout
```

---

# 10. Experiment 4 — Transformer Long-Context Model

## Goal

Use a Transformer model with long context:

```text
Transformer:
24h history -> 12h future
```

## Purpose

The Transformer serves as a comparison to RNN-based context modeling.

An RNN compresses the history into hidden states, while a Transformer can use self-attention over the entire 24h history.

## Research Question

```text
Can attention over long temporal context improve 12h AIS trajectory prediction?
```

## Why Only One Transformer?

We only need one Transformer experiment because the project focus is not a full Transformer context-length sweep.

Its role is to test whether attention over 24h history provides an advantage over RNN-style temporal memory.

---

# 11. Experiment 5 — Adaptive Multi-Scale RNN_AR

## Goal

This is the most important experiment.

The goal is to build an autoregressive RNN model that learns how much history it needs.

Instead of manually choosing one fixed context window, the model receives several context windows:

```text
9h context
12h context
18h context
24h context
```

The model learns weights over these windows:

```text
alpha_9
alpha_12
alpha_18
alpha_24
```

These weights represent how much the model uses each context length.

## Architecture

Each context window is encoded separately:

```text
h_9  = Encoder(9h history)
h_12 = Encoder(12h history)
h_18 = Encoder(18h history)
h_24 = Encoder(24h history)
```

Then a gating network predicts context weights:

```text
alpha = softmax(MLP([h_9, h_12, h_18, h_24]))
```

where:

```text
alpha = [alpha_9, alpha_12, alpha_18, alpha_24]
```

and:

```text
alpha_9 + alpha_12 + alpha_18 + alpha_24 = 1
```

The final context representation is:

```text
h_context =
    alpha_9  * h_9
  + alpha_12 * h_12
  + alpha_18 * h_18
  + alpha_24 * h_24
```

Then the autoregressive decoder predicts:

```text
h_context -> 12h future trajectory
```

## Important Design Choice

Start with one alpha vector per trajectory sample.

That means the model chooses one soft context distribution for the whole prediction.

Example:

```text
alpha_9  = 0.10
alpha_12 = 0.20
alpha_18 = 0.25
alpha_24 = 0.45
```

This means the model relied mostly on the 24h context for that sample.

Do not start with alpha weights that change at every future prediction step. That is more complex and should only be optional if there is extra time.

## Why This Is the Main Experiment

This experiment directly answers the core research question:

```text
Can an RNN learn the useful context window size by itself?
```

It also gives interpretability.

We can inspect the learned alpha weights and analyze which types of trajectories cause the model to prefer short or long context.

---

# 12. Feature Analysis

The adaptive model should save the learned alpha weights for every validation/test sample:

```text
alpha_9
alpha_12
alpha_18
alpha_24
```

Then we analyze how these weights correlate with trajectory features.

The goal is to understand:

```text
Which motion patterns cause the model to prefer longer or shorter context?
```

---

## 12.1 Original AIS Features

Use available AIS features such as:

```text
timestamp
latitude
longitude
SOG
COG
MMSI / vessel id
```

---

## 12.2 Derived Motion Features

Compute additional features that describe motion behavior.

Important derived features:

```text
delta_lat
delta_lon
delta_SOG
delta_COG
turn_rate
SOG variance
COG variance
straightness_score
path_length
direct_distance
local_AIS_density
```

---

## 12.3 Straightness Score

A useful feature is the straightness score:

```text
straightness_score = direct_distance / path_length
```

where:

```text
direct_distance = distance between the first and last point in the history window
path_length = sum of distances between consecutive points
```

Interpretation:

```text
straightness_score close to 1:
trajectory is almost straight

straightness_score lower than 1:
trajectory is curved or maneuvering
```

Expected relationship:

```text
high straightness_score -> shorter context may be enough
low straightness_score  -> longer context may be needed
```

---

## 12.4 Course Variability

Course variability measures how much the vessel changes direction.

Example features:

```text
delta_COG
mean absolute delta_COG
COG variance
turn_rate
```

Expected relationship:

```text
low COG variance  -> higher alpha_9 or alpha_12
high COG variance -> higher alpha_18 or alpha_24
```

---

## 12.5 Speed Variability

Speed variability measures whether the vessel is maintaining speed, slowing down, or accelerating.

Example features:

```text
delta_SOG
mean absolute delta_SOG
SOG variance
```

Expected relationship:

```text
stable SOG   -> shorter context may be enough
changing SOG -> longer context may be needed
```

---

## 12.6 Local AIS Density

Local AIS density can be used as a proxy for complex geographic regions.

High density may indicate:

```text
ports
coastal areas
traffic lanes
busy maritime regions
```

Expected relationship:

```text
low density  -> open sea / simpler motion / shorter context
high density -> complex area / longer context
```

This feature is optional but can make the analysis stronger.

---

# 13. Evaluation Metrics

Do not evaluate only a single final score.

Use:

```text
ADE
FDE
nADE
nFDE
horizon-wise error
```

## ADE

Average Displacement Error.

Measures the average distance between predicted and true positions across the future trajectory.

## FDE

Final Displacement Error.

Measures the distance between the predicted final point and the true final point at 12h.

## Horizon-Wise Error

This is very important.

Measure error as a function of prediction horizon:

```text
error after 1h
error after 2h
error after 3h
error after 6h
error after 9h
error after 12h
```

This allows us to understand whether some methods perform well early but drift later.

---

# 14. Bucket-Based Evaluation

Evaluate models not only overall, but also on trajectory buckets.

Suggested buckets:

```text
straight trajectories
maneuvering trajectories
stable-speed trajectories
changing-speed trajectories
low-density regions
high-density regions
```

This is important because a model can have good average performance while failing on difficult trajectory types.

For each bucket, report:

```text
ADE
FDE
horizon-wise error
```

---

# 15. Main Comparisons

The final report should compare:

## 15.1 Kinematic Baseline vs Neural Models

Question:

```text
Do neural models outperform simple constant-velocity extrapolation?
```

---

## 15.2 Flat RNN vs RNN_AR

Question:

```text
Is autoregressive prediction better than direct one-shot prediction?
```

Comparison:

```text
Flat RNN:
24h history -> 12h future

RNN_AR:
24h history -> 12h future
```

---

## 15.3 Fixed Context RNN_AR

Question:

```text
Which fixed context length works best?
```

Comparison:

```text
9h  context
12h context
18h context
24h context
```

---

## 15.4 Direct 12h Prediction vs Sliding Window

Question:

```text
Is it better to predict all 12h at once,
or predict four 3h chunks using a receding horizon?
```

Comparison:

```text
RNN_AR 24h -> 12h direct

Sliding Window:
24h -> 3h
update window
repeat 4 times
```

---

## 15.5 RNN_AR vs Transformer

Question:

```text
Does attention over long context help compared to recurrent memory?
```

Comparison:

```text
RNN_AR 24h -> 12h
Transformer 24h -> 12h
```

---

## 15.6 Fixed Context vs Adaptive Context

Question:

```text
Can the model learn the context length instead of choosing it manually?
```

Comparison:

```text
best fixed-context RNN_AR
vs
adaptive multi-scale RNN_AR
```

This is the most important comparison.

---

# 16. Expected Outcomes

Expected findings:

```text
1. The kinematic baseline may be strong on straight stable trajectories.
2. RNN_AR should improve on more complex trajectories.
3. Longer context may help for turns, speed changes, and dense regions.
4. Shorter context may be enough for stable straight motion.
5. Sliding window may help by keeping the context local and updated.
6. Sliding window may also suffer from error accumulation because predictions enter the next window.
7. Transformer may help if attention can exploit long-range context.
8. Adaptive RNN_AR should provide both prediction performance and interpretability.
```

---

# 17. Desired Final Interpretation

The final result should not only be:

```text
Model X achieved the best ADE/FDE.
```

The stronger research result is:

```text
The useful temporal context length depends on the trajectory type.
Stable straight trajectories often require less history, while maneuvers, speed changes, and dense/coastal regions benefit from longer context.
The adaptive multi-scale RNN_AR learns meaningful context weights, allowing us to analyze which features influence the selected temporal context.
```

---

# 18. Implementation Priority

Recommended implementation order:

```text
1. Prepare dataset windows for:
   9h, 12h, 18h, 24h history -> 12h future

2. Implement the kinematic constant-velocity baseline.

3. Implement one Flat RNN / Flat LSTM:
   24h history -> 12h future

4. Implement Fixed Context RNN_AR + TC + Anchor:
   9h, 12h, 18h, 24h history -> 12h future

5. Implement Receding-Horizon Sliding Window:
   train 24h -> 3h
   rollout 4 times to reach 12h

6. Implement Transformer:
   24h history -> 12h future

7. Implement Adaptive Multi-Scale RNN_AR:
   9h + 12h + 18h + 24h contexts
   learned alpha weights
   12h future

8. Save alpha weights for validation/test samples.

9. Compute derived motion features.

10. Analyze alpha weights vs trajectory features.

11. Report overall metrics and bucket-based metrics.
```

---

# 19. Main Framing Sentence

Use this as the project framing sentence:

```text
This project studies AIS vessel trajectory prediction as a sequence modeling problem, focusing on how much temporal context is needed for 12-hour forecasting. We compare fixed-context autoregressive RNNs, receding-horizon sliding-window prediction, a long-context Transformer, and an adaptive multi-scale RNN_AR model that learns how much history to use for each trajectory.
```

---

# 20. One-Sentence Research Contribution

```text
The main contribution is an adaptive multi-scale autoregressive RNN that learns soft weights over several history windows, enabling both improved AIS trajectory prediction and analysis of which motion features determine the useful context length.
```
