# Reproduce report experiments (`exp_coastal`) — exact arguments

All main-report runs share:

| Flag | Value |
|------|--------|
| `--coast` | `"USA Combined"` |
| `--input` | `data/processed/combined_filtered_smart_coastal/train.parquet` |
| `--sample` | `300000` |
| `--future-hours` / `--horizon-hours` | `12` / `12` |
| `--hidden-dim` / `--num-layers` / `--dropout` | `256` / `2` / `0.2` (Transformer differs) |
| `--epochs` / `--patience` | `60` / `10` |
| `--no-maneuver-oversample` | on |
| `--land-penalty-weight` | `0.1` (except AR 12h noland → `0.0`) |

**Source of truth:** `scripts/exp_coastal/train_*.sbatch`  
**One-shot helper:** `scripts/exp_coastal/reproduce_experiments.sh`

```bash
# from proj/project, after dataset download
bash scripts/exp_coastal/reproduce_experiments.sh          # print all exact commands
bash scripts/exp_coastal/reproduce_experiments.sh --run flat
bash scripts/exp_coastal/reproduce_experiments.sh --run ar18
bash scripts/exp_coastal/reproduce_experiments.sh --run all  # full suite (long / GPU)
```

## Commands (expanded)

### AR LSTM 9 / 12 / 18 / 24 h

```bash
python -u models/RNN_AR.py --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/AR_18h \
  --sample 300000 --history-hours 18 --future-hours 12 --horizon-hours 12 \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample --target-mode anchor_offset \
  --land-penalty-weight 0.1
```

Same for 9 / 12 / 24: change `--history-hours` and `--run-tag` to `AR_9h` / `AR_12h` / `AR_24h`.

### Flat LSTM (best median FDE in report)

```bash
python -u models/RNN.py --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/flat_lstm \
  --sample 300000 --history-hours 24 --future-hours 12 --horizon-hours 12 \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --epochs 60 --patience 10 \
  --no-maneuver-oversample --no-curriculum \
  --land-penalty-weight 0.1
```

### Transformer

```bash
python -u models/transformers.py --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/transformer \
  --sample 300000 --history-hours 24 --future-hours 12 --horizon-hours 12 \
  --d-model 128 --nhead 8 --num-encoder-layers 4 --dim-feedforward 512 --dropout 0.1 \
  --batch-size 128 --lr 1e-3 --weight-decay 1e-4 --epochs 60 --patience 10 \
  --no-maneuver-oversample --no-curriculum \
  --land-penalty-weight 0.1
```

### Shared-encoder Adaptive AR

```bash
python -u models/RNN_AR_adaptive.py --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/adaptive_multiscale \
  --sample 300000 --future-hours 12 --horizon-hours 12 \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample \
  --land-penalty-weight 0.1
```

### Sliding 3h × 4

```bash
python -u models/RNN_recursive_1h.py --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/sliding_3h \
  --sample 300000 --chunk-hours 3 --horizon-hours 12 \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --epochs 60 --patience 10 \
  --no-maneuver-oversample --no-curriculum \
  --land-penalty-weight 0.1
```

### Ablation: AR 12h, no land penalty

Same as AR 12h but `--run-tag exp_coastal/AR_12h_noland` and `--land-penalty-weight 0.0`.

### Follow-up: separate-encoder adaptive (hard / softmax)

```bash
python -u models/RNN_AR_diff_encoder.py --coast "USA Combined" \
  --input data/processed/combined_filtered_smart_coastal/train.parquet \
  --run-tag exp_coastal/adaptive_separate_encoders_hard \
  --gate-mode hard \
  --sample 300000 --future-hours 12 --horizon-hours 12 \
  --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample \
  --land-penalty-weight 0.1
```

Softmax: `--gate-mode softmax` and `--run-tag exp_coastal/adaptive_separate_encoders_softmax`.

## Cluster (Slurm)

```bash
bash scripts/exp_coastal/submit_all.sh
```

Uses the same flags via `scripts/exp_coastal/_env.sh` + `train_*.sbatch`.
