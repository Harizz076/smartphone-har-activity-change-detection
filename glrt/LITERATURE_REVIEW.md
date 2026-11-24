# Literature review and evidence-driven GLRT improvements

## Search boundary

This was a bounded engineering review performed on 24–25 November 2026. The
search covered primary work on generalized likelihood ratios, regularized
Gaussian discriminant analysis, robust multivariate likelihoods,
orientation-aware smartphone HAR, and online multichannel change detection.
Publisher records, author manuscripts, arXiv records, and the local Kay
textbook were used. It was not a systematic review or a novelty claim.

The experimental work below is exploratory. The repository's held-out sets had
already been examined earlier, so they are not pristine confirmatory tests.
Every model choice in this round was nevertheless made using subject-disjoint
training folds; held-out labels were not used to choose features or parameters.

## What the literature says

1. **The likelihood-ratio foundation is sound, but fitted covariance is the
   weak point.** Kay's *Detection Theory*, Chapters 6 and 12, derives the GLRT
   by maximizing the likelihood separately under each composite hypothesis.
   Our six-class rule is the multiple-hypothesis analogue: choose the fitted
   class density with maximum likelihood and report twice the winner-minus-
   runner-up log-likelihood as the GLR margin.

2. **Gaussian discrimination should regularize both between-class covariance
   differences and unstable eigenvalues.** Friedman introduced a two-parameter
   family that continuously connects QDA, LDA, and spherical nearest-mean
   models, and recommended selecting both parameters by cross-validation
   ([primary paper](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/friedman1989.pdf)).
   Our previous classifier implemented only eigenvalue shrinkage. The new
   `rda_glrt.py` adds the missing class-to-pooled covariance parameter exactly.

   For class scatter (S_k), pooled within-class scatter (S), class weight
   (W_k), and total weight (W), the implementation uses

   \[
   S_k(\lambda)=(1-\lambda)S_k+\lambda S,
   \qquad
   W_k(\lambda)=(1-\lambda)W_k+\lambda W,
   \]

   \[
   \widehat\Sigma_k(\lambda)=\frac{S_k(\lambda)}{W_k(\lambda)},
   \qquad
   \widehat\Sigma_k(\lambda,\gamma)
   =(1-\gamma)\widehat\Sigma_k(\lambda)
   +\gamma\frac{\operatorname{tr}\widehat\Sigma_k(\lambda)}{p}I.
   \]

3. **Gravity/body separation and magnitude, jerk, and spectral summaries have
   direct precedent in smartphone HAR.** The original UCI HAR methods paper
   separates gravity and body acceleration and uses magnitude, jerk, entropy,
   correlation, and frequency features
   ([Anguita et al., 2013](https://arpi.unipi.it/retrieve/handle/11568/962678/409259/J35%20-%202013.pdf)).
   Work on orientation-independent HAR maps acceleration into a common frame
   because phone orientation increases within-class variability
   ([Guo et al., 2016](https://onlinelibrary.wiley.com/doi/10.1002/dac.3010)).
   The new `gravity_features.py` estimates the gravity direction from each
   window and appends vertical, horizontal, magnitude, jerk, and spectral
   motion summaries. The original axis features remain, so posture information
   is not discarded.

4. **Heavy-tailed likelihoods are a useful robustness check, not a universal
   replacement.** The repository's earlier Student-t experiment follows the
   robust multivariate-estimation literature and improved WISDM modestly while
   barely changing UCI HAR. That result is retained rather than silently
   replacing Gaussian likelihoods everywhere.

5. **Continuous multichannel detection should eventually fuse evidence across
   channels and scales.** Xie and Siegmund combine stream-specific GLR scores
   when only an unknown subset of channels changes
   ([Annals of Statistics](https://doi.org/10.1214/13-AOS1094)); Chen, Wang, and
   Samworth aggregate likelihood-ratio evidence across coordinates and scales
   for online Gaussian mean changes
   ([primary preprint](https://arxiv.org/abs/2003.03668)). These are principled
   future directions for `continuous_transition_glrt.py`, but the existing UCI
   HAR windows must not be stitched into a fake continuous stream and WISDM's
   recording gaps prevent a broad transition benchmark. No extra dataset was
   added to evade that limitation.

## Questions tested

### Candidate A: full two-parameter RDA

Prediction: partial pooling of class covariance matrices will reduce
subject-to-subject covariance noise and improve balanced accuracy.

Rival: activity classes genuinely have different covariance geometry, so any
pooling will introduce more bias than stability.

Result: the training folds selected \(\lambda=0\) for both datasets. The
candidate therefore collapsed exactly to the current class-specific model:
UCI HAR selected \(\gamma=0.03\), and WISDM selected \(\gamma=0.75\). This is a
negative result: covariance pooling alone does not improve the original
15-feature representation.

### Candidate B: gravity-aligned features plus RDA

Prediction: gravity-aligned vertical/horizontal motion will reduce nuisance
orientation variability, especially for WISDM subjects, while covariance
regularization controls the nine additional dimensions.

Rival: phone placement is sufficiently fixed that aligned features are
redundant, or they dilute the axis direction needed to separate posture and
stair activities.

| Dataset | Model selected on training subjects | CV balanced accuracy | Test accuracy | Test balanced accuracy | Test macro F1 |
|---|---|---:|---:|---:|---:|
| UCI HAR | Original 15D reference | 87.82% | **87.92%** | **87.76%** | **87.67%** |
| UCI HAR | 24D gravity + RDA, \(\lambda=.05,\gamma=0\) | **88.26%** | 87.72% | 87.56% | 87.43% |
| WISDM | Original 15D reference | 84.07% | 78.81% | 78.41% | 76.72% |
| WISDM | 24D gravity + RDA, \(\lambda=.25,\gamma=.01\) | **87.41%** | **82.56%** | **80.73%** | **78.15%** |

The WISDM gain is substantial: +3.75 percentage points in held-out accuracy
and +2.32 points in balanced accuracy relative to the model selected by the
same training-only protocol. Upstairs recall rises to 47.0%. On UCI HAR, the
training-fold improvement does not transfer: test accuracy decreases by 0.20
points, mainly through weaker sitting recall. We therefore do **not** claim a
universal improvement. The new feature family is useful for WISDM but remains
an exploratory alternative on UCI HAR.

The WISDM result also slightly exceeds the repository's historical 82.43% test
accuracy, but that historical test result had already been viewed. The honest
comparison remains the predeclared training-CV-selected reference at 78.81%; a
new subject cohort would be needed for confirmatory evidence.

## Decision

- Keep the original class-specific Gaussian GLRT as the stable UCI HAR default.
- Use gravity-aligned features with \(\lambda=.25,\gamma=.01\) as the improved
  WISDM configuration selected by subject-disjoint training folds.
- Retain RDA and gravity features as tested, reusable components rather than
  overwriting the older result files.
- Do not report a continuous event-detection score from the packaged UCI HAR
  windows or WISDM gaps.

Reproduce with:

```bash
python3 analysis/evaluate_rda_glrt.py --dataset all
python3 analysis/evaluate_gravity_features.py --dataset all
python3 -m unittest discover -s tests -v
```

The research-design and statistical workflow used the local Scientific Agent
Skills procedures. Reference: Kassis, T., Agarwal, V., He, Y., Patel, D., &
Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural
Knowledge for Research Agents*. arXiv:2609.00065.
https://doi.org/10.48550/arXiv.2609.00065
