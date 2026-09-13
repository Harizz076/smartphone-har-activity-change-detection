# Variance-threshold baseline: UCI HAR

## Goal and scope

This is a deliberately simple baseline for the future GLR detector. It makes a
binary decision per UCI HAR window: **moving** (walking, walking upstairs, or
walking downstairs) versus **still** (sitting, standing, or laying). A single
accelerometer-variance threshold cannot meaningfully separate all six UCI HAR
activities, so reporting it as a six-class classifier would be misleading.

The detector uses only the raw `total_acc_x`, `total_acc_y`, and `total_acc_z`
signals. For each 128-sample (2.56 s) window, it calculates

\[
v = \frac{\operatorname{Var}(a_x)+\operatorname{Var}(a_y)+\operatorname{Var}(a_z)}{3}.
\]

It declares the window moving when `v >= threshold`; otherwise it declares it
still. The threshold is fit exclusively on the UCI HAR training split by
maximizing balanced accuracy. The UCI test split contains subjects not present
in the training set, making this a subject-disjoint held-out evaluation.

## Reproduce

From the project root, run:

```bash
python3 baseline/variance_threshold_baseline.py
```

The program has no third-party dependency. It prints the learned threshold,
confusion matrix, accuracy, moving recall, still specificity, and false-alarm
rate; it also writes per-activity test results to
`baseline/results/uci_har_variance_threshold_activity_breakdown.csv`.

## Metrics

`false-alarm rate = FP / (FP + TN)`, where an FP is a still-labelled window
incorrectly declared moving. This is the appropriate false-alarm definition
for this motion detector. It is not a transition false alarm: UCI HAR's
provided windows have activity labels but do not provide an independently
labelled sequence of transition instants suitable for a change-point-delay
evaluation.

## Held-out test result

The selected training-set threshold was **0.01221445 g²**. On the official UCI
HAR test partition (2,947 windows from held-out subjects), the detector
achieved **99.90% accuracy** and **99.90% balanced accuracy**.

| Actual class | Predicted moving | Predicted still |
|---|---:|---:|
| Moving | 1,386 | 1 |
| Still | 2 | 1,558 |

This gives 99.93% moving recall, 99.87% still specificity, and a **0.13%
false-alarm rate** (2 still windows classified as moving). Per-activity test
accuracy is 100.00% for walking, walking downstairs, standing, and laying;
99.79% for walking upstairs; and 99.59% for sitting. Exact results are in
`baseline/results/uci_har_variance_threshold_activity_breakdown.csv`.

The printed held-out-test figures are the figures to report if rerunning the
experiment.

## Six-class extension: normalized variance thresholds

The script also evaluates the requested six-class version. It first discards
only non-finite variance values (none occurred in either partition), then
z-score normalizes the scalar window variance using the training-set mean
(0.02089821 g²) and standard deviation (0.02713268 g²). The six activities are
ordered by their **training** mean normalized variance and five thresholds are
placed halfway between adjacent class means:

`standing → sitting → laying → walking → walking upstairs → walking downstairs`

The held-out six-class accuracy is only **37.87%**. Its per-class accuracy is:

| Activity | Accuracy |
|---|---:|
| Walking | 66.53% |
| Walking upstairs | 28.24% |
| Walking downstairs | 48.81% |
| Sitting | 7.13% |
| Standing | 73.68% |
| Laying | 3.91% |

Normalization changes the scale but not the ordering of a one-dimensional
variance feature. The poor six-class result is therefore expected: sitting,
standing, and laying all have near-zero motion variance and overlap heavily;
the three walking variants overlap too. This is useful as a transparent
baseline, but a GLR detector or a practical six-class classifier should add
other accelerometer features (axis means, energy, correlations, frequency
content) and, for the full project, gyroscope features.
