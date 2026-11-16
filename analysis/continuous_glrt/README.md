# Continuous UCI 341 GLRT performance

This is the first subject-independent, continuous event-detection evaluation in
the repository. It supersedes the earlier label-guided score contrast when the
question is “how good is the detector?”

## Honest headline

| Operating profile | Detector selected on validation subjects | Precision | Recall | F1 | False alarms/min | Median delay |
|---|---|---:|---:|---:|---:|---:|
| Balanced | 2 s acceleration mean-change scan | 77.9% | 93.1% | **84.8%** | 0.379 | 1.52 s |
| Low false alarm | 4 s acceleration+gyroscope mean/variance scan | **91.8%** | 77.8% | 84.2% | **0.100** | **0.60 s** |

These are event-level results on held-out subjects 5, 10, 15, 20, 25, and 30:
72 transition events in total. Threshold and five-second refractory settings
were selected using validation subjects 4, 9, 14, 19, 24, and 29. The remaining
18 subjects are reserved as training subjects for later learned preprocessing;
the current raw GLRT does not fit parameters from them.

The low-false-alarm extension is a meaningful improvement. Relative to the
balanced profile it keeps nearly the same F1 while reducing false alarms by
about 74% and median delay by about 61%. The price is 11 fewer detected events.

## Uncertainty

Subject-cluster bootstrap 95% intervals (six test subjects) are wide:

- Balanced F1: 0.744–0.933; recall: 0.875–0.986; false alarms/min: 0.103–0.737.
- Low-false-alarm F1: 0.766–0.913; precision: 0.862–0.969;
  false alarms/min: 0.042–0.155.

The subject, not a window or sample, is the resampling unit. These intervals
describe variation among the six held-out subjects; they do not remove dataset
or protocol bias.

## What changed mathematically

At each causal update, the detector examines only a trailing horizon and scans
all allowed split points. For every channel and candidate split, the null has
one Gaussian regime. The balanced detector's alternative has separate pre/post
means with a common variance:

\[
2\log\Lambda_c(k)=N\log\frac{\mathrm{SSE}_{0,c}}
                                      {\mathrm{SSE}_{1,c}(k)}.
\]

The channel statistics are summed at the same candidate split and then
maximized over the split. This preserves the assumption that all channels share
one physical change time.

The low-false-alarm detector additionally permits the variance to change:

\[
2\log\Lambda_c(k)=
N\log\frac{\mathrm{SSE}_{0,c}}{N}
-k\log\frac{\mathrm{SSE}_{L,c}}{k}
-(N-k)\log\frac{\mathrm{SSE}_{R,c}}{N-k}.
\]

It sums this evidence across all six raw accelerometer/gyroscope axes and scans
a four-second trailing horizon. This is still a GLRT: every unknown mean and
variance is replaced by its maximum-likelihood estimate under its corresponding
hypothesis.

## Evaluation definition

- Scores update every 0.2 s; candidate splits are spaced 0.1 s apart.
- Each side of a candidate split has at least 0.5 s of data.
- A rising threshold crossing emits an alarm; a validation-selected refractory
  period suppresses duplicates.
- An alarm matches one transition if its trigger falls inside the labeled
  transition interval expanded by one second on each side. Matching is
  one-to-one.
- Unmatched alarms are false alarms. The denominator is time annotated as one
  of the six basic activities, excluding unannotated gaps and transition time.
- Thresholds are empirical validation stable-score quantiles. No chi-square
  approximation is used after maximizing over split points.

## Where it still fails

Balanced test recall is 83.3% for stand-to-sit and at least 91.7% for each other
transition type. The conservative detector's weakest transition is
stand-to-lie at 50% recall. Its longer horizon and stricter threshold suppress
false alarms well, but miss gradual or weak stand-to-lie changes.

The balanced validation false-alarm rate was 0.144/min but rose to 0.379/min on
test subjects. That threshold-generalization gap is now the main weakness—not
raw event recall. The next improvement should therefore use nested
subject-level calibration and a predeclared multihorizon fusion rule, optimized
for recall at a fixed false-alarm budget. Per-class thresholds would leak class
information into a class-agnostic detector and are not the right fix.

## Reproduction

```bash
python3 analysis/evaluate_continuous_glrt.py
python3 -m unittest discover -s tests -v
```

Outputs:

- [`metrics.json`](metrics.json): configuration, validation selection, test
  results, transition-type recall, and bootstrap intervals.
- [`test_subject_metrics.csv`](test_subject_metrics.csv): subject-level results.
- [`01_test_operating_curves.svg`](01_test_operating_curves.svg): recall versus
  false alarms per stable minute.
- [`02_held_out_score_trace.svg`](02_held_out_score_trace.svg): causal scores,
  thresholds, alarms, and post-hoc transition overlays.

The statistical workflow and cluster-bootstrap reporting used the local
*Statistical Analysis* scientific skill. Reference:

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
*arXiv:2609.00065*. https://doi.org/10.48550/arXiv.2609.00065
