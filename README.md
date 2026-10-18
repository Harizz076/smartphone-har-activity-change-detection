# Smartphone HAR Activity Change Detection

An implementation and experimental workspace for smartphone human-activity
recognition (HAR) and activity-change detection, using the UCI HAR and WISDM
datasets. The project will compare simple threshold-based detectors with a
Generalized Likelihood Ratio (GLR) detector.

The project calculates a per-window tri-axial accelerometer-variance feature,
fits thresholds using training data, and reports held-out activity-recognition
results. The `baseline/` directory contains the variance-threshold baselines;
the top-level `glrt/` directory contains the scalar one-class GLRT detectors.

## Run

```bash
python3 baseline/variance_threshold_baseline.py
python3 baseline/wisdm_variance_threshold_baseline.py
python3 glrt/windowed_scalar_glrt.py --dataset all
python3 glrt/sequential_scalar_glrt.py
```

No third-party Python packages are needed. See `baseline/BASELINE_REPORT.md`
and `baseline/WISDM_BASELINE_REPORT.md` for the variance-threshold protocols.
See `glrt/WINDOWED_SCALAR_GLRT_REPORT.md` and
`glrt/SEQUENTIAL_SCALAR_GLRT_REPORT.md` for the GLRT detectors.

## Datasets

- UCI Human Activity Recognition Using Smartphones Dataset
- WISDM Activity Recognition Dataset v1.1

Please consult the license and citation notices included with each dataset
before redistributing or using them in academic work.
