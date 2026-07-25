# Manuscript: "What Does 99% Accuracy Measure?"

LaTeX source for the paper auditing shortcut learning in the ISOT/Kaggle fake
news corpus. Everything needed to compile is in this folder.

```
paper/
├── main.tex              IEEEtran conference manuscript (single file)
├── refs.bib              65 BibTeX entries, all cited
├── figures/              8 vector PDF figures
├── validate_latex.py     static pre-flight checker (no LaTeX needed)
└── README.md             this file
```

## Compiling on Overleaf (recommended)

1. Create a new project → **Upload Project** → upload a ZIP of this `paper/`
   folder (or drag the individual files in, keeping `figures/` as a subfolder).
2. Set the main document to `main.tex` (Overleaf usually detects it).
3. Menu → **Compiler: pdfLaTeX**. Leave everything else at defaults.
4. Press **Recompile**. The first run resolves citations, so Overleaf runs
   `pdflatex → bibtex → pdflatex → pdflatex` automatically.

`IEEEtran.cls` and `IEEEtran.bst` ship with Overleaf's TeX Live, so no manual
package installation is required. All other packages (`amsmath`, `booktabs`,
`algorithm`, `algpseudocode`, `tikz`, `hyperref`, `xcolor`, `multirow`) are
standard.

## Compiling locally

Requires a TeX distribution (TeX Live 2021+ or MiKTeX) with `IEEEtran`:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or, if `latexmk` is available:

```bash
latexmk -pdf main.tex
```

Expected output: `main.pdf`. A clean run reports no errors; `Overfull \hbox`
warnings on a few wide table rows are cosmetic and do not affect correctness.

## Pre-flight validation

`validate_latex.py` catches the errors a compiler would, without needing LaTeX
installed. Run it after any edit:

```bash
python validate_latex.py
```

It verifies brace balance, environment nesting, that every `\cite` key exists in
`refs.bib`, that every `\ref` resolves, that every figure/table/algorithm float
is discussed in the text, that referenced graphics exist on disk, unescaped `%`
and `_`, duplicate labels and BibTeX keys, and tabular column-count consistency.
Current status: **PASS, 0 errors**.

## Regenerating the figures

The figures are produced from the experiments, not drawn by hand. From the
project root (`Yuvraj_Verma/`), with `data/Fake.csv` and `data/True.csv` in
place:

```bash
python experiments/run_experiments.py      # writes experiments/results/results.json
python experiments/run_shift_analysis.py   # writes experiments/results/shift_analysis.json
python experiments/make_figures.py         # writes paper/figures/*.pdf
```

Total runtime is roughly twelve minutes on one CPU. Every number quoted in the
manuscript comes from those two JSON files; nothing is transcribed by hand.

## Before you submit — required author actions

The manuscript contains a `\authorinput{...}` macro that renders in **red**.
Search for `authorinput` and resolve each occurrence, then delete the macro
definition (near the top of `main.tex`) so any stragglers fail loudly rather
than printing in red:

| Location | What is needed |
|---|---|
| Author block | Contact e-mail for the camera-ready version |
| `\thanks` | ORCID, and any funding/acknowledgement |
| Introduction footnote | Optional verifiable usage count for the dataset, or delete the footnote |
| Acknowledgment section | Acknowledgements, or delete the section |

### Reference verification checklist

Every entry in `refs.bib` refers to a real, well-known publication, and no DOIs
were invented — DOIs are deliberately **omitted** rather than risk a wrong
identifier. Before camera-ready:

1. Open each entry against the publisher's canonical record (DBLP, ACL
   Anthology, or the ACM/IEEE/Springer page).
2. Confirm author spellings, year, venue, volume, and page range.
3. Add `doi = {...}` fields from the canonical record if the venue requires them.
4. Re-run `python validate_latex.py` to confirm nothing broke.

This is ordinary diligence for any submission, but it matters here because
bibliographic metadata was not machine-verified against a live database.

## Notes on scope

The paper deliberately makes a measurement claim, not a modelling claim. If a
reviewer asks for a transformer comparison, that experiment is scoped in
Section "Future Work" and is the single most valuable addition; it was not run,
and the manuscript says so explicitly rather than speculating about its outcome.
