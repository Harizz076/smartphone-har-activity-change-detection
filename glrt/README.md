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
| `improved_six_class_glrt.py` | Existing UCI HAR and WISDM windows | Literature-reviewed dataset-specific Gaussian/RDA likelihood |
| `continuous_transition_glrt.py` | Generic multichannel inertial stream | Causal score every configured hop |
| `rda_glrt.py` | Same six-class features | Friedman's two-parameter regularized Gaussian likelihood |

Run from the repository root:

```bash
python3 glrt/windowed_scalar_glrt.py --dataset all
python3 glrt/sequential_scalar_glrt.py
python3 glrt/six_class_glrt.py --dataset all
python3 analysis/optimize_six_class_glrt.py --dataset all
python3 analysis/compare_robust_likelihoods.py --dataset all
python3 glrt/robust_six_class_glrt.py --dataset all
python3 glrt/improved_six_class_glrt.py --dataset all
python3 analysis/evaluate_rda_glrt.py --dataset all
python3 analysis/evaluate_gravity_features.py --dataset all
```

`results/` contains reproducible held-out outputs. See the corresponding report
files for model equations and measured results.
The optimizer evaluates all declared feature/covariance/shrinkage combinations
with subject-disjoint training folds and writes its audit under
`analysis/six_class_optimization/`.
