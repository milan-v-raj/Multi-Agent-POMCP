# Short LaTeX project report

The editable report is `main.tex`; its required images are in `figures/`. The compiled report is `PROJECT_REPORT_SHORT.pdf` (distributed separately from the Overleaf ZIP).

## Overleaf

Create a new project by uploading `PROJECT_REPORT_SHORT_LATEX.zip`. The archive has `main.tex` at its root and includes the three figures.

## Local build

With a standard TeX Live or MiKTeX installation, run from this folder:

```sh
pdflatex main.tex
pdflatex main.tex
```

Run twice to resolve cross-references. References are embedded, so BibTeX is not required. The link-outage figure is derived from `../arq_fixed_suite/results/e5b_link_failure.csv`; the other figures are from the existing completed experiments.
