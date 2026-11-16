# Six-class Gaussian GLRT classifier

## Scope

This implementation performs supervised, window-level classification among
six activity hypotheses. It is separate from the one-class still-versus-motion
detector and from the continuous binary change-point detector.

- UCI HAR uses its official subject-disjoint train/test partitions and 128
  samples per supplied window at 50 Hz.
- WISDM uses non-overlapping, activity-pure 200-sample windows at 20 Hz. Users
  divisible by three are held out, matching the existing baseline protocol.
- Every learned quantity—feature scaling, class means, and class
  covariances—is fitted using training subjects only.

## Ground-up feature representation

Each tri-axial accelerometer window becomes a 15-dimensional vector:

1. axis means \(\bar a_x,\bar a_y,\bar a_z\), which retain phone orientation
   and gravity/posture information;
2. log axis variances, which represent motion energy on a less skewed scale;
3. within-window correlations \(r_{xy},r_{xz},r_{yz}\);
4. log per-axis first-difference energies (jerk proxies);
5. dominant frequency, normalized spectral entropy, and log band power from
   0.3–5 Hz, with power summed across axes.

The global training mean \(m\) and population standard deviation \(s\) of
each feature standardize every vector:

\[
y_j=\frac{x_j-m_j}{s_j}.
\]

The same fitted \(m,s\) transform held-out windows. Test data never affect
normalization.

## Six hypotheses

For activity \(c\), the class hypothesis is

\[
H_c:y\sim\mathcal N(\mu_c,\Sigma_c).
\]

The unknown nuisance parameters \(\mu_c,\Sigma_c\) are estimated by maximum
likelihood from labelled training windows belonging to class \(c\). To avoid
unstable or singular covariance estimates, the fitted covariance is

\[
\widetilde\Sigma_c
=(1-\lambda)\widehat\Sigma_c
+\lambda\tau_c I+\epsilon I,
\qquad
\tau_c=\frac{\operatorname{tr}(\widehat\Sigma_c)}{d},
\]

with prespecified \(\lambda=0.10\), \(\epsilon=10^{-6}\), and \(d=15\).
Cholesky factorization evaluates the likelihood without explicitly inverting
the covariance matrix.

## Multi-hypothesis GLRT decision

For a candidate vector \(y\), the fitted class log likelihood is

\[
\ell_c(y)=-\frac12\left[
d\log(2\pi)+\log|\widetilde\Sigma_c|
+(y-\mu_c)^T\widetilde\Sigma_c^{-1}(y-\mu_c)
\right].
\]

No class-frequency prior is added. The prediction is therefore maximum
likelihood rather than MAP:

\[
\widehat c=\arg\max_c \ell_c(y).
\]

For every other class \(j\), the pairwise twice-log generalized likelihood
ratio is

\[
G_{\widehat c,j}=2[\ell_{\widehat c}(y)-\ell_j(y)].
\]

The output confidence field is the winner-versus-runner-up margin:

\[
G_{\text{margin}}
=2\left[\ell_{\widehat c}(y)-\max_{j\ne\widehat c}\ell_j(y)\right]\ge0.
\]

The classifier always returns one of the six activities. This version does not
calibrate a rejection/unknown-class threshold.

## Configuration search

The fixed 15-feature, class-full model below is retained as the original
reference. A separate exhaustive experiment evaluates all 192 combinations of
six feature sets, four covariance structures, and eight shrinkage values using
subject-disjoint folds within the training partition. Its selection criterion
is balanced accuracy, with macro F1 and accuracy as tie-breakers. No test label
selects a configuration.

The corrected training-only search uses three balanced subject folds (UCI:
2450/2441/2461 windows; WISDM: 1187/1189/1221 windows) and selects:

- UCI HAR: all 15 features, class-full covariance, shrinkage 0.03;
- WISDM: all 15 features, class-full covariance, shrinkage 0.75.

UCI test accuracy rises from 86.16% to 87.92%. WISDM test accuracy falls from
82.43% to 78.81%, despite improved cross-validation selection. This failure is
reported rather than selecting retrospectively on the test users. Full search
tables, per-subject diagnostics, and figures are in
`analysis/six_class_optimization/README.md`.

## Robust likelihood follow-up

A separate, training-fold-selected comparison replaces the Gaussian class
likelihood with multivariate Student-t or independent Laplace likelihoods while
retaining the full 15-feature representation. Student-t parameters are fitted
by fixed-degrees-of-freedom EM and covariance shrinkage remains selected only
inside subject-disjoint training folds.

Student-t with \(\nu=30\) is selected for both datasets. UCI accuracy rises
from 87.92% to 88.06%; WISDM rises from 78.81% to 80.41%. The independent
Laplace model is markedly worse. WISDM's historical fixed Gaussian result is
82.43%, so the robust model improves the CV-selected Gaussian but does not set
a new WISDM test maximum. These are exploratory post-development
results; full equations, literature provenance, convergence diagnostics, and
per-subject comparisons are in `analysis/robust_likelihoods/README.md`.

## Reproduce

```bash
python3 -m unittest -v tests/test_six_class_glrt.py
python3 glrt/six_class_glrt.py --dataset all
```

The implementation uses only the Python standard library. Outputs are written
to `glrt/results/` as a metrics JSON file and per-window prediction CSV for
each dataset.

## Held-out results

These are the original fixed full/class-full/0.10 reference results, not the
cross-validated selections described above.

| Dataset | Windows | Accuracy | Balanced accuracy | Macro F1 | Variance-only baseline |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 2,947 | 86.16% | 85.88% | 85.77% | 37.87% |
| WISDM | 1,628 | 82.43% | 80.70% | 79.40% | 58.11% |

### UCI HAR recall by activity

| Activity | Recall |
|---|---:|
| Walking | 91.53% |
| Walking upstairs | 80.68% |
| Walking downstairs | 89.52% |
| Sitting | 60.49% |
| Standing | 93.05% |
| Laying | 100.00% |

The largest ambiguity is sitting versus standing: 187 sitting windows are
classified as standing. The three walking hypotheses also overlap, but no
walking window is classified as a still posture.

### WISDM recall by activity

| Activity | Recall |
|---|---:|
| Walking | 78.98% |
| Jogging | 94.27% |
| Upstairs | 48.99% |
| Downstairs | 80.82% |
| Sitting | 81.11% |
| Standing | 100.00% |

Upstairs is the weakest class and is most often confused with downstairs.
The subject-disjoint result is materially stronger than the one-feature
baseline without claiming that the Gaussian feature model is a perfect model
of human activity.

## Verification and limitations

The deterministic unit tests cover Cholesky reconstruction, triangular solve,
an analytically known one-dimensional Gaussian likelihood, frequency-feature
recovery, six separated synthetic classes, and rejection of a missing training
class. Both real evaluations preserve subject-disjoint test data.

This remains a plug-in Gaussian GLRT: finite feature vectors within a class are
only approximately Gaussian, and supplied/constructed windows—not subjects—are
the prediction units. The reported held-out metrics, rather than asymptotic
likelihood-ratio theory, are the evidence for classification performance.

## Methodological tooling reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
