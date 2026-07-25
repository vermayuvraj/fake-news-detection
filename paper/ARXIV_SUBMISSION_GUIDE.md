# arXiv Submission Guide

Written for this manuscript specifically, and for submitting as an
**independent researcher** with no institutional affiliation.

> Policies and interface details change. Where a rule could have shifted since
> writing, this guide points you to the canonical arXiv help page rather than
> asserting a number that may be stale. Treat <https://info.arxiv.org/help/>
> as authoritative.

---

## 1. Account and endorsement

### Create the account
1. Go to <https://arxiv.org/user/register>.
2. Use an e-mail address you will keep long term — it becomes the owner
   identity for the paper and receives all moderation correspondence. A
   personal address is fine; you do not need an academic one.
3. Verify the address from the confirmation e-mail before doing anything else.
4. Link an **ORCID** (free, <https://orcid.org>). It is optional but it makes
   authorship durable and is worth the five minutes, especially without an
   institution behind your name.

### Endorsement — expect to need it
arXiv gates first-time submitters in each subject area through *endorsement*.
Submitters with a recognised academic e-mail domain and submission history are
often auto-endorsed; a new independent submitter to `cs.CL` usually is not.

When you start a submission in a category you are not endorsed for, arXiv shows
an endorsement code and instructions. To obtain one:

- Ask someone who has recently published in that same archive (`cs.CL` or
  `cs.LG`) to endorse you. Eligibility is determined by arXiv, not by seniority.
- Send a short, specific request: your name, the arXiv endorsement code shown on
  your screen, the paper title and abstract, and a link to the manuscript PDF
  plus the code repository. Endorsers are vouching that the work is plausibly
  appropriate for the archive, not peer-reviewing it — make that easy to judge.
- Good candidates: an author of a paper you cite whose work is close to yours, a
  professor or PhD student you have corresponded with, or a co-author on any
  prior work.

Two practical notes. First, endorsement is per-archive, so being endorsed for
`cs.CL` does not cover a later submission to, say, `stat.ML`. Second, do not
mass-mail requests; arXiv treats endorsement solicitation abuse seriously.

---

## 2. Prepare the submission package

arXiv strongly prefers **LaTeX source** over a finished PDF, because it
regenerates the PDF itself and can produce HTML. Submit source.

### What to include

```
main.tex          the manuscript
main.bbl          REQUIRED — see below
refs.bib          optional (harmless; arXiv ignores it if .bbl is present)
figures/          the 8 PDF figures
```

### The `.bbl` file — the single most common failure

**arXiv does not run BibTeX.** If you upload only `main.tex` and `refs.bib`,
every citation renders as `[?]`. You must include the compiled `main.bbl`.

To obtain it:

- **From Overleaf:** compile the project, then open the **Logs and output
  files** panel (next to the Recompile button) → *Other logs & files* →
  download `main.bbl`.
- **Locally:** run `pdflatex main` then `bibtex main`; `main.bbl` appears in the
  folder.

Place `main.bbl` beside `main.tex` in the upload. Optionally, you can inline its
contents into `main.tex` inside `\begin{thebibliography}...\end{thebibliography}`
and drop `.bbl`/`.bib` altogether — this is the most robust option, though it
makes later edits clumsier.

### What to exclude

Delete before zipping: `.aux`, `.log`, `.out`, `.blg`, `.synctex.gz`,
`.fdb_latexmk`, `.fls`, `main.pdf`, `validate_latex.py`, `README.md`, and any
editor folders (`.vscode/`, `.DS_Store`). Auxiliary files can confuse arXiv's
build; scripts and READMEs belong in the code repository, not the paper source.

### Figures
All eight figures are already vector PDF, which is exactly what pdfLaTeX wants.
Keep them in the `figures/` subfolder and keep the relative paths in
`\includegraphics` unchanged. Do not use absolute paths — a path like
`C:/Users/...` compiles on your machine and fails on arXiv.

### Package availability
Every package used (`IEEEtran`, `amsmath`, `booktabs`, `algorithm`,
`algpseudocode`, `tikz`, `hyperref`, `xcolor`, `multirow`, `array`, `cite`,
`url`) is in arXiv's TeX Live. No special action needed.

### Size
The full package is a few hundred kilobytes, far under arXiv's limit. If you
ever exceed it, the limit and the process for requesting an exception are on the
arXiv help pages.

### Before uploading — clear the author-input markers
Search `main.tex` for `authorinput` and resolve every occurrence, then delete
the macro definition near the top. These render in **red** and must not appear
in a public preprint. See `README.md` for the list.

---

## 3. Metadata

You will be asked for the following. Draft it before you start, because the form
times out.

### Title
```
What Does 99% Accuracy Measure? A Reproducible Audit of Shortcut Learning in a
Widely Used Fake News Corpus
```
Plain text only — no LaTeX macros. Use `$...$` only if a formula is unavoidable.

### Authors
```
Yuvraj Verma
```
Format is `Firstname Lastname`, comma-separated for multiple authors. Put
`Independent Researcher` in the affiliation field if one is offered; arXiv does
not require an institution.

### Abstract
Paste the abstract from `main.tex`, converted to plain text: replace `\Fone{}`
with `F1`, `\dataset{}` with `ISOT/Kaggle`, `$=$` with `=`, and remove `\emph{}`
and `\cite{}`. Keep it under arXiv's length limit (roughly 1,920 characters);
the current abstract is close to that, so if it is rejected for length, cut the
sentence beginning "Temporal transfer within the corpus is by contrast..."

### Categories
| Role | Category | Why |
|---|---|---|
| **Primary** | `cs.CL` — Computation and Language | The work is an NLP benchmark/evaluation study |
| Cross-list | `cs.LG` — Machine Learning | Shortcut learning, distribution shift, leakage |
| Optional cross-list | `cs.CY` — Computers and Society | Only if you want misinformation-policy readership |

Choose the primary category carefully: it determines which moderators see the
paper and which mailing list announces it. Cross-lists can be added later; the
primary is awkward to change.

### Comments
This free-text field appears under the abstract. Suggested:
```
12 pages, 8 figures, 9 tables. Code, experiment scripts, and machine-readable
results: https://github.com/vermayuvraj/fake-news-detection
```
Update the page count to match your compiled PDF. If you later submit to a
venue, add the status here (for example "Accepted at ...").

### ACM / MSC classification
Both optional. MSC is for mathematics — leave it blank. For ACM, reasonable
choices are `I.2.7` (Natural Language Processing) and `I.5.4` (Pattern
Recognition: Applications — Text processing). Skipping this costs nothing.

### Journal reference and DOI
Leave blank. These are for work already published elsewhere; you can add them
later via a replacement if the paper is accepted somewhere.

### License
You will pick a license at submission. Options, briefly:
- **arXiv perpetual non-exclusive license** — the conservative default; keeps
  most rights with you and is compatible with subsequent IEEE/ACM submission.
- **CC BY 4.0** — maximally reusable; some publishers dislike it for later
  submission of the same work.

If you intend to submit this to an IEEE or ACM venue afterwards, the perpetual
non-exclusive license is the safer choice. Check the target venue's preprint
policy first — most now permit arXiv preprints, but the license terms matter.

---

## 4. Submitting as an independent researcher

Nothing about arXiv requires an institution, and independent submissions are
routine. What actually differs:

- **You will almost certainly need endorsement** (Section 1). Budget a few days
  for it; this is the step that delays people.
- **Affiliation:** write `Independent Researcher`. Do not invent an affiliation,
  and do not list a former university as current — arXiv moderators do check,
  and a corrected affiliation is a bad first impression.
- **E-mail:** a personal address is fine, but it must be one you monitor.
  Moderation questions go unanswered otherwise and the paper stalls on hold.
- **Credibility comes from the artifacts.** Without an institution, the code
  repository and the reproducibility appendix do the work an affiliation would
  otherwise do. The repository link in the Comments field is not decoration —
  make sure it is public and that the README explains how to reproduce the
  numbers before you submit.

### Mistakes to avoid
1. Uploading without `main.bbl` — citations become `[?]`.
2. Leaving the red `\authorinput` markers in the PDF.
3. Choosing a primary category by guesswork; `cs.CL` is correct here.
4. Abstract containing LaTeX macros, which arXiv renders literally.
5. Submitting a PDF-only package when the source compiles fine.
6. Assuming the paper appears immediately — see the schedule below.
7. Submitting the same work twice instead of replacing v1 with v2.

---

## 5. Final submission, versioning, and tracking

### Preview and validate
After upload, arXiv runs its own build (AutoTeX) and shows you the log plus a
generated PDF. Do not skip this:

1. Read the build log for errors and for any package substitution warnings.
2. Open the generated PDF and check it **page by page**: all eight figures
   present and not placed after the references, all nine tables intact, no `[?]`
   citations, no `??` cross-references, no red text.
3. Confirm the two-column layout has not broken the wide tables
   (`table*` floats should span both columns).

If anything is wrong, fix the source locally, re-run
`python validate_latex.py`, and re-upload before you submit. Uploading a
corrected package pre-submission is free; after announcement it costs a version
number.

### Announcement timing
Submissions are announced on a fixed weekday schedule with an afternoon
Eastern-time cutoff; anything after the cutoff, or over a weekend or US holiday,
rolls to the next announcement. Check
<https://info.arxiv.org/help/availability.html> for the current cutoff rather
than relying on a time quoted here. Practical consequence: if you want the paper
public by a particular day, submit at least two business days early — and
remember endorsement may add days on top of that.

### Moderation
Submissions may be held for moderator review, reclassified to a different
category, or occasionally rejected as out of scope. Holds are normal and not an
accusation. If held, respond promptly and factually to any moderator query.

### Versioning
- A submission is announced as **v1**, permanently citable as
  `arXiv:XXXX.XXXXX v1`.
- To update, use **Replace** on the article's page. This creates v2, v3, and so
  on. Old versions remain publicly accessible forever — that is the point of the
  archive, so make v1 something you are willing to have on record.
- Use replacements for real changes (fixing an error, adding the transformer
  experiment, noting acceptance at a venue). Do not version-bump for typos you
  can live with.
- Add a short note in the Comments field describing what changed in each
  version.

### Withdrawal
You cannot delete a paper from arXiv. You can **withdraw** it, which posts a new
version whose only content is a withdrawal notice explaining why; the earlier
versions stay visible. Withdrawal is appropriate for a fatal error, not for
second thoughts about presentation.

### Tracking status
Log in and open your user page (<https://arxiv.org/user>) to see each
submission's state: *incoming*, *on hold*, *scheduled*, or *announced*. You
receive e-mail at announcement containing the permanent identifier. After that,
Google Scholar and Semantic Scholar typically index the paper within days to a
few weeks, with no action needed from you.

---

## 6. Pre-flight checklist

Run through this immediately before pressing submit.

- [ ] `python validate_latex.py` reports **0 errors**
- [ ] All `\authorinput` markers resolved and the macro definition deleted
- [ ] Contact e-mail filled into the author block
- [ ] Reference metadata spot-checked against publisher records (`README.md`)
- [ ] `main.bbl` present in the upload
- [ ] Auxiliary files (`.aux`, `.log`, `.out`, `.blg`) excluded
- [ ] All 8 figures render in the arXiv-generated PDF
- [ ] No `[?]` citations and no `??` references in the generated PDF
- [ ] Abstract converted to plain text, macros removed, within length limit
- [ ] Primary category `cs.CL`, cross-list `cs.LG`
- [ ] Comments field includes the public repository URL
- [ ] Repository is public and its README reproduces the reported numbers
- [ ] License chosen deliberately with any future venue in mind
