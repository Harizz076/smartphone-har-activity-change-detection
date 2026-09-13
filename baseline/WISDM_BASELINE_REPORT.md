# Six-class variance-threshold baseline: WISDM

## Protocol

WISDM v1.1 provides raw 20 Hz tri-axial accelerometer samples, but no official
train/test split. To make the evaluation subject-disjoint, users with IDs
divisible by three (3, 6, ..., 36) are held out for test; all remaining users
are used to learn the normalization and thresholds. This gives 3,597 training
windows and 1,628 test windows.

The raw stream is cleaned by recovering semicolon-delimited records, dropping
malformed/non-numeric records, and accepting only finite accelerometer values.
It is split at changes in user or activity and aggregated into non-overlapping
200-sample windows (10 seconds), matching the window duration stated in the
WISDM transformation documentation. Samples remaining in an incomplete window
are not used. This run recovered 5,225 complete windows; it excluded 11,733
malformed records, one non-numeric record, and 41,475 incomplete-run samples.

For each window, the feature is the mean of the sample variance from each
accelerometer axis. It is normalized using the training mean (26.17691046
(m/s²)²) and standard deviation (22.05682065 (m/s²)²). Activities are ordered
by their training mean normalized variance, and each threshold is the midpoint
between adjacent means.

## Learned detector

| Variance interval (z-score) | Activity |
|---|---|
| `< -1.17943282` | Standing |
| `-1.17943282` to `< -0.93510688` | Sitting |
| `-0.93510688` to `< -0.64791990` | Upstairs |
| `-0.64791990` to `< -0.51516404` | Downstairs |
| `-0.51516404` to `< 0.41978816` | Walking |
| `≥ 0.41978816` | Jogging |

## Held-out result

The six-class held-out accuracy is **58.11%**.

| Activity | Test windows | Correct | Accuracy |
|---|---:|---:|---:|
| Walking | 647 | 353 | 54.56% |
| Jogging | 506 | 472 | 93.28% |
| Upstairs | 149 | 36 | 24.16% |
| Downstairs | 146 | 10 | 6.85% |
| Sitting | 90 | 4 | 4.44% |
| Standing | 90 | 71 | 78.89% |

The result is a transparent one-feature baseline, rather than a competitive
HAR classifier. Variance separates vigorous jogging well but substantially
overlaps for sitting/standing and stair/walking activities. The reproducible
implementation is `wisdm_variance_threshold_baseline.py`; the detailed CSV is
`baseline/results/wisdm_six_class_variance_thresholds.csv`.
