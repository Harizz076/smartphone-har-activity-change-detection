# Scalar one-class GLRT motion detector

This baseline detects motion as an upward departure from a stillness reference.
It is evaluated as moving versus still, not as six-class activity recognition.

For each supplied window, the scalar feature is mean tri-axial acceleration
variance:

\[
v=\frac{\operatorname{Var}(a_x)+\operatorname{Var}(a_y)+\operatorname{Var}(a_z)}{3}.
\]

Only still training windows fit \(z=\log(v)\sim\mathcal N(\mu_0,\sigma_0^2)\).
The one-sided, twice-log likelihood-ratio statistic is

\[
G=\left[\max\left(0,\frac{z-\mu_0}{\sigma_0}\right)\right]^2.
\]

Motion is declared when \(G>h\), where \(h\) is the empirical upper 1%
quantile of training-still scores. No motion labels are used to fit the model
or select its threshold.

Run from the project root:

```bash
python3 glrt/windowed_scalar_glrt.py --dataset all
```

| Dataset | Test accuracy | Motion recall | Still specificity | Still false alarm |
|---|---:|---:|---:|---:|
| UCI HAR | 99.86% | 100.00% | 99.74% | 0.26% |
| WISDM | 100.00% | 100.00% | 100.00% | 0.00% |
