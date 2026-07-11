# experiment1 — Feb fixed windows: v1 vs v2 in parallel

## Output dirs
- `data/results/USA Combined/unknown/v1/experiment1/<Model>/`
- `data/results/USA Combined/unknown/v2/experiment1/<Model>/`
- `data/results/USA Combined/unknown/v1_residual/experiment1/<Model>/` — v1 + residual only (isolates residual vs motion-balance)

## Recipes
| | v1 | v1_residual | v2 |
|--|----|-------------|-----|
| residual-naive | OFF | ON | ON |
| motion-balanced sampling | OFF | OFF | ON (15% straight / 15% other) |
| sample size | 400k | 400k | 400k |
| shared | no maneuver oversample, curriculum on, patience 10, 12h horizon |

v1 ≈ March baseline recipe. v2 ≈ residual + motion-balance recipe. **v1_residual** isolates residual learning without changing the training distribution.

Submit v1_residual: `bash scripts/experiment1/submit_v1_residual.sh`
