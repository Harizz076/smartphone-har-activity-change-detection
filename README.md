# Smartphone HAR Activity Change Detection

An implementation and experimental workspace for smartphone human-activity
recognition (HAR) and activity-change detection, using the UCI HAR and WISDM
datasets. The project will compare simple threshold-based detectors with a
Generalized Likelihood Ratio (GLR) detector.

The project calculates a per-window tri-axial accelerometer-variance feature,
fits thresholds using training data, and reports held-out activity-recognition
results. The `baseline/` directory contains the variance-threshold baselines;
the top-level `glrt/` directory contains one-class GLRT detectors and the
six-class Gaussian GLRT classifier.

## Run

```bash
python3 baseline/variance_threshold_baseline.py
python3 baseline/wisdm_variance_threshold_baseline.py
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

No third-party Python packages are needed. See `baseline/BASELINE_REPORT.md`
and `baseline/WISDM_BASELINE_REPORT.md` for the variance-threshold protocols.
See `glrt/WINDOWED_SCALAR_GLRT_REPORT.md` and
`glrt/SEQUENTIAL_SCALAR_GLRT_REPORT.md` for the one-class detectors, and
`glrt/SIX_CLASS_GLRT_REPORT.md` for six-class classification.
The exhaustive subject-disjoint model search, plots, and statistical audit are
documented in `analysis/six_class_optimization/README.md`.
The Gaussian/Student-t/Laplace comparison is documented in
`analysis/robust_likelihoods/README.md`.
The literature-driven RDA and gravity-aligned feature experiments are under
`analysis/rda_glrt/` and `analysis/gravity_features/`.

## Datasets

- UCI Human Activity Recognition Using Smartphones Dataset
- WISDM Activity Recognition Dataset v1.1

Please consult the license and citation notices included with each dataset
before redistributing or using them in academic work.
