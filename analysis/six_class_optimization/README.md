# Six-class GLRT model-selection and visual audit

## Question and protocol

This experiment asks whether the original 15-feature, class-full Gaussian
GLRT is unnecessarily complicated and whether one configuration transfers
across UCI HAR and WISDM. It evaluates the Cartesian product of:

- six prespecified feature sets (3, 6, 9, 12, 12, or 15 dimensions);
- class-specific or shared covariance;
- full or diagonal covariance; and
- eight covariance-shrinkage values from 0 to 1.

That is 192 configurations per dataset. Selection maximizes subject-disjoint
out-of-fold balanced accuracy; macro F1 and accuracy break exact ties. Feature
standardization, means, and covariances are refitted inside each fold. Test
labels do not select configurations. The final selected model is refitted on
all training subjects and evaluated once on the existing held-out split.

The test sets had already been examined earlier in project development, so
these numbers are reproducible post-development evaluations, not a pristine
confirmatory study. A new external subject cohort would be needed for that.

## Selected models and results

| Dataset | Training-CV selection | CV balanced accuracy | Test accuracy | Test balanced accuracy | Test macro F1 |
|---|---|---:|---:|---:|---:|
| UCI HAR | full (15D), class-full, shrinkage 0.03 | 87.82% | 87.92% | 87.76% | 87.67% |
| WISDM | full (15D), class-full, shrinkage 0.75 | 84.07% | 78.81% | 78.41% | 76.72% |

The UCI selection improves on the original full/class-full/0.10 result
(86.16% accuracy). The WISDM selection does not: the original configuration
scored 82.43% on the already-viewed test split. Choosing that older model now
because its test score is larger would leak test information into model
selection, so the negative result is retained.

## Findings from the audit

1. **The full representation wins, but only with dataset-specific
   regularization.** Both datasets select all 15 features and class-specific
   full covariance. UCI needs little shrinkage (0.03), whereas WISDM needs
   heavy shrinkage (0.75). One covariance setting should therefore not be
   assumed portable across datasets.
2. **Several “different” motion features are nearly duplicates.** On UCI,
   `log_variance_y` and `log_band_power` have Pearson \(r=0.990\), while
   `log_variance_x` and `log_jerk_x` have \(r=0.989\). On WISDM the analogous
   correlations are 0.988 and 0.984. A full covariance can model this
   dependence, but it also increases estimation sensitivity.
3. **The hardest distinctions remain structurally plausible.** UCI sitting
   recall is 65.58%, primarily because sitting and standing share low motion.
   WISDM upstairs recall is only 36.24%, with substantial confusion into
   jogging and downstairs.
4. **Confidence transfers poorly.** Treating GLR margin as a score for whether
   a prediction is correct gives AUROC 0.831 on UCI but only 0.638 on WISDM.
   UCI accuracy rises monotonically as low-margin windows are rejected. WISDM
   accuracy is non-monotonic (92.64% at 10% coverage, 75.46% at 20%), so its
   margin must not yet be advertised as calibrated confidence.
5. **Subject shift dominates WISDM.** UCI held-out subject accuracies range
   from 66.67% to 98.95%. WISDM ranges from 25.41% to 97.60%; user 30 is a
   catastrophic failure. Aggregate accuracy therefore hides a deployment
   risk that window-level metrics cannot solve.

These are associations in the supplied data. Windows from one subject are
dependent, and feature correlations do not establish causal mechanisms.

## Outputs

Each dataset directory contains:

- all 192 validation scores;
- selected configuration and complete metrics;
- per-window likelihoods, GLR margins, and predictions;
- class feature profiles and strongest correlations;
- accuracy-versus-coverage and per-subject metrics;
- ten SVG figures, rendered PNGs, and a reviewed contact sheet.

Reproduce everything from the repository root:

```bash
python3 analysis/optimize_six_class_glrt.py --dataset all
```

The runner uses only the Python standard library. SVG rendering is optional
and was performed with `rsvg-convert` for visual QA.

## Methodological tooling reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
