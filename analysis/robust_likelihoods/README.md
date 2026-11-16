# Robust likelihood experiment for six-class HAR

This is an **exploratory supplementary-classification experiment**. Per
`CODEX.md`, the project's primary deliverable remains continuous transition
detection on ordered raw streams; six-class accuracy is not a substitute for
event recall, false alarms per minute, or detection delay.

## Question and candidates

Observation: several class-feature distributions have asymmetric or heavy
tails. Across 90 class-feature cells, the 90th-percentile excess kurtosis is
3.76 on UCI HAR and 2.63 on WISDM. The candidates were fixed before this
comparison:

1. Gaussian reference: class-specific full or diagonal covariance.
2. Multivariate Student-t: class-specific full or diagonal scatter, fixed
   degrees of freedom \(\nu\in\{3,5,10,30\}\), fitted by EM.
3. Independent Laplace: class medians and mean absolute deviations.

All use the same 15 features and global training-fold standardization. Fifty-
five configurations were compared using the same balanced, subject-disjoint
three folds. Selection maximized balanced accuracy, then macro F1, then raw
accuracy. Held-out test labels did not choose the configuration.

## Mathematics

For a feature vector \(y\), the multivariate Student-t log likelihood is

\[
\ell_c(y)=\log\Gamma\!\left(\frac{\nu+d}{2}\right)
-\log\Gamma\!\left(\frac{\nu}{2}\right)
-\frac12\log|\Psi_c|-\frac d2\log(\nu\pi)
-\frac{\nu+d}{2}\log\!\left(1+\frac{\delta_c(y)}{\nu}\right),
\]

where

\[
\delta_c(y)=(y-\mu_c)^T\Psi_c^{-1}(y-\mu_c).
\]

For fixed \(\nu\), the EM weight for training window \(i\) is

\[
w_i=\frac{\nu+d}{\nu+\delta_c(y_i)}.
\]

Large residuals therefore contribute less to the fitted center and scatter.
The final scatter receives the same trace-target shrinkage used by the Gaussian
model. All selected Student-t fold models converged within the 30-iteration
cap.

The independent Laplace log likelihood is

\[
\ell_c(y)=-\sum_j\left[\log(2b_{cj})+
\frac{|y_j-m_{cj}|}{b_{cj}}\right],
\]

with coordinate median \(m_{cj}\) and mean absolute deviation \(b_{cj}\).
Its independence assumption is a strong rival to the covariance-aware models.

In all cases the classification rule is maximum fitted likelihood and the
reported GLR margin is twice the winner-minus-runner-up log likelihood.

## Results

| Dataset | Selected likelihood | CV balanced accuracy | Test accuracy | Test balanced accuracy | Gaussian test accuracy |
|---|---|---:|---:|---:|---:|
| UCI HAR | Student-t, \(\nu=30\), full covariance, shrinkage 0.03 | 87.99% | 88.06% | 87.89% | 87.92% |
| WISDM | Student-t, \(\nu=30\), full covariance, shrinkage 0.50 | 84.72% | 80.41% | 79.71% | 78.81% |

The UCI gain is only 0.14 percentage points and is not uniform: four test
subjects improve, two tie, and three worsen. WISDM gains 1.60 points, with six
users improving, three tying, and three worsening; mean subject-level change
is +1.72 points. WISDM upstairs recall rises from 36.91% to 44.97%, but the
model remains substantially weaker than a deployment-ready classifier.

The WISDM gain is relative to the configuration selected by the current
training-only CV protocol. The historical fixed Gaussian configuration at
shrinkage 0.10 scored 82.43% on the already-viewed WISDM test users, so the
Student-t result does **not** exceed the repository's highest historical WISDM
test score. Retrospectively selecting shrinkage or distribution to beat that
test score would be leakage.

Laplace performs poorly: its best CV balanced accuracy is 82.44% on UCI and
76.36% on WISDM. This does not show that every multivariate Laplace model is
bad; it shows that replacing covariance geometry with independent absolute
deviations is counterproductive here.

The selected \(\nu=30\) is close to Gaussian, which fits the diagnostics:
most class-feature cells are not extremely heavy-tailed, while a minority have
large skewness/kurtosis. Very heavy-tailed \(\nu=3\) or 5 models generally
rank lower.

## Literature boundary and implications

Searches were conducted on 2026-11-16 using web search, arXiv, publisher
records, and the local Kay textbook. Queries covered multivariate Student-t
MLE/EM, Student-t discriminant analysis, robust/compound-Gaussian GLRT change
detection, marginalized likelihood ratios, and non-asymptotic multivariate
change detection. Primary methods sources were prioritized. This was a bounded
engineering literature search, not a systematic review or novelty assessment.

- Liu and Rubin derive EM/ECM/ECME maximum-likelihood estimation for the
  multivariate t and explicitly motivate it for robust multivariate estimation
  ([Statistica Sinica, 1995](https://www3.stat.sinica.edu.tw/statistica/oldpdf/a5n12.pdf)).
- Chen, Kuo, and Lin develop heterogeneous multivariate-t classification with
  EM-estimated parameters for long-tailed/outlier-contaminated populations
  ([primary paper, 2011](https://amath2.nchu.edu.tw/tilin/JATITAS%20Vol9_No2_3.pdf)).
- Mian et al. derive robust LRT/GLRT change detectors under a compound-Gaussian
  model and report advantages over a Gaussian detector for heterogeneous data
  ([ICASSP paper, 2018](https://dihana.cps.unizar.es/proceedings/ICASSP/2018/pdfs/0003429.pdf)).
- Gustafsson's marginalized likelihood-ratio method addresses sensitivity to
  unknown/changing noise levels, but it is a different Bayesian construction
  from the GLRT used here ([IEEE TAC, 1996](https://doi.org/10.1109/9.481608)).
- Barnett shows that asymptotic likelihood-ratio calibration can be inaccurate
  for short multivariate smartphone-like records
  ([arXiv preprint](https://arxiv.org/abs/2304.04644)).
- Kay's *Detection Theory*, Chapters 6 and 12, remains the project's primary
  source for composite GLRT and unknown-change-time derivations.

For the primary continuous detector, the classification result justifies a
future Student-t or compound-Gaussian change-score ablation. It does **not**
justify reusing classification likelihoods or thresholds directly. The scan
maximum must still be calibrated on stable validation streams, and performance
must be reported at matched false alarms per minute.

## Limitations

- The test partitions had already been viewed earlier; results are
  post-development, not confirmatory.
- Feature-set selection preceded this experiment and used the same datasets.
- Candidate grids differ in size, and \(\nu\) was searched on four values.
- Laplace was diagonal/independent; no multivariate Laplace family was fitted.
- Window metrics are clustered within subjects and are not independent trials.
- Distribution fit diagnostics are descriptive; they do not prove a true
  generative family.

## Reproduce

```bash
python3 analysis/compare_robust_likelihoods.py --dataset all
python3 glrt/robust_six_class_glrt.py --dataset all
python3 -m unittest discover -v
```

## Methodological tooling reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
