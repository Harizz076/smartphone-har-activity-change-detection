# Continuous scalar one-class GLRT

The continuous detector uses WISDM's timestamped raw samples. Every 5-second
rolling window (100 samples) produces the scalar feature

\[
z_t=\log\left(\frac{\operatorname{Var}(a_x)+\operatorname{Var}(a_y)+
\operatorname{Var}(a_z)}{3}\right).
\]

Still training frames estimate \(\mu_0\) and \(\sigma_0\), giving
\(q_t=(z_t-\mu_0)/\sigma_0\). For each new frame, the detector scans possible
recent change onsets:

\[
G_{t,m}=\left[\max\left(0,\frac{\sum_{i=t-m+1}^{t}q_i}{\sqrt m}\right)\right]^2,
\qquad G_t=\max_{1\le m\le8}G_{t,m}.
\]

The threshold is the upper 1% empirical quantile of sequential scores from
still-only training data. It accounts empirically for correlation between
overlapping windows. A scan restarts only at a user or session boundary, never
at an activity label boundary.

Run:

```bash
python3 glrt/sequential_scalar_glrt.py
```

Held-out results: 96.61% accuracy, 96.17% motion recall, 99.44% still
specificity, and 0.56% still false-alarm rate. Transition-spanning frames are
retained as `MIXED_TRANSITION` but excluded from pure-class metrics.
