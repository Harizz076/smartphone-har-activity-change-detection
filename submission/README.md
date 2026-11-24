# Final submission artifacts

- `report/har_glrt_report.pdf`: five-page A4 IEEE conference report
- `slides/har_glrt_slides.pdf`: nine-slide 16:9 presentation
- `figures/make_figures.py`: source for every analytical figure
- `figures/generated/`: vector PDF and 300 dpi PNG exports

Regenerate the figures from the saved experiment outputs:

```bash
MPLCONFIGDIR=/tmp/har_glrt_mpl \
UV_CACHE_DIR=/tmp/har_glrt_uv_cache \
uv run --isolated --no-project --python 3.13 \
  --with 'matplotlib==3.11.2' --with 'numpy==2.5.4' \
  python submission/figures/make_figures.py
```

Compile the report without modifying `IEEEtran.cls`:

```bash
cd submission/report
TEXINPUTS=../../IEEE_Conference_Template: \
  latexmk -pdf -interaction=nonstopmode -halt-on-error har_glrt_report.tex
```

Compile the slide deck:

```bash
cd submission/slides
latexmk -pdf -interaction=nonstopmode -halt-on-error har_glrt_slides.tex
```
