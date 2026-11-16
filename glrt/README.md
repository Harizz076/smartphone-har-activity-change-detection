# GLRT detectors

This directory contains the project's generalized likelihood-ratio detectors:
two scalar one-class motion detectors and a supervised six-hypothesis activity
classifier.

| Script | Input | Output cadence |
|---|---|---|
| `windowed_scalar_glrt.py` | Supplied UCI HAR or WISDM windows | One decision per window |
| `sequential_scalar_glrt.py` | WISDM timestamped raw samples | One rolling decision per second by default |
| `six_class_glrt.py` | UCI HAR or WISDM accelerometer windows | One of six activities per window |
| `robust_six_class_glrt.py` | Same supplementary classification windows | Student-t or Laplace six-class likelihood |
| `continuous_transition_glrt.py` | Continuous UCI 341 accelerometer/gyroscope streams | Causal score every configured hop |

Run from the repository root:

```bash
python3 glrt/windowed_scalar_glrt.py --dataset all
python3 glrt/sequential_scalar_glrt.py
python3 glrt/six_class_glrt.py --dataset all
python3 analysis/optimize_six_class_glrt.py --dataset all
python3 analysis/compare_robust_likelihoods.py --dataset all
python3 glrt/robust_six_class_glrt.py --dataset all
python3 analysis/evaluate_continuous_glrt.py
```

`results/` contains reproducible held-out outputs. See the corresponding report
files for model equations and measured results.
The primary continuous event-level evaluation is documented in
`analysis/continuous_glrt/README.md`.
The optimizer evaluates all declared feature/covariance/shrinkage combinations
with subject-disjoint training folds and writes its audit under
`analysis/six_class_optimization/`.
