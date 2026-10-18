# GLRT detectors

This directory contains the project's scalar, one-class generalized
likelihood-ratio test detectors. Both models learn only the stillness reference
class and declare motion for a sufficiently large upward shift in log mean
tri-axial acceleration variance.

| Script | Input | Output cadence |
|---|---|---|
| `windowed_scalar_glrt.py` | Supplied UCI HAR or WISDM windows | One decision per window |
| `sequential_scalar_glrt.py` | WISDM timestamped raw samples | One rolling decision per second by default |

Run from the repository root:

```bash
python3 glrt/windowed_scalar_glrt.py --dataset all
python3 glrt/sequential_scalar_glrt.py
```

`results/` contains the reproducible held-out scalar GLRT outputs. See the two
report files for the model equations and measured results.
