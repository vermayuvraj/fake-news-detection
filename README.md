# Fake News Detection

![tests](https://github.com/vermayuvraj/fake-news-detection/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Author: Yuvraj Verma

Given a news article's title and body, this classifies it as real or fake. The
pipeline cleans the text, turns it into TF-IDF features, compares a few linear
classifiers on a validation set, picks the best one, and reports the metrics on
a held-out test set. It reaches an F1 of 0.99 on the held-out test set, though
most of that comes from the model learning writing style rather than actually
checking facts (I go into this in REPORT.md).

There's an interactive demo in `huggingface-space/` (Gradio) that's ready to
deploy as a Hugging Face Space, or run locally with
`python huggingface-space/app.py`.

I wrote up the reasoning behind the approach, the results, and the limitations
in REPORT.md. This file just explains how to run everything.

Dataset: Fake and Real News Dataset on Kaggle
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

## Results (held-out 20% test set)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9912 |
| Precision | 0.9933 |
| Recall    | 0.9877 |
| F1-score  | 0.9905 |
| AUC-ROC   | 0.9995 |

The selected model is a Passive-Aggressive classifier on TF-IDF (unigrams +
bigrams). The full comparison of all four models and the exact settings used are
written to `reports/metrics.json` every time the pipeline runs.

## How to run

1. Set up the environment (Python 3.10):

   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```

2. Get the data. The raw CSVs (~116 MB) aren't included in this folder, so
   download them and put them in `data/`:

   - Manually: download and unzip from the Kaggle link above and copy
     `Fake.csv` and `True.csv` into `data/`.
   - Or with the Kaggle API: `pip install kaggle`, add your `kaggle.json`
     token, then `python scripts/download_data.py`.

   You should end up with `data/Fake.csv` and `data/True.csv`.

3. Run the pipeline:

   ```
   python main.py
   ```

   It trains and evaluates everything and writes the saved model to `models/`
   and the metrics + plots to `reports/`. Takes under a minute on a laptop CPU.

4. Classify a new article:

   ```
   python -m src.predict --text "Some headline and article body here ..."
   python -m src.predict --file article.txt
   ```

There's also `python scripts/leakage_demo.py`, which shows why I strip the
Reuters source tag from the text (explained in REPORT.md), and
`python scripts/eda.py`, which regenerates the class-balance and article-length
plots in `reports/figures/`.

Unit tests (cleaning and split logic, plus a check on the saved model) run with
`pytest`. They also run automatically on every push via GitHub Actions.

## Folder layout

```
Yuvraj_Verma/
  README.md                     - this file
  REPORT.md                     - approach, results, limitations
  REPRODUCIBILITY_CHECKLIST.md  - reproducibility checklist
  requirements.txt              - pinned dependencies
  config.py                     - seed, split, TF-IDF settings, paths
  main.py                       - runs the pipeline
  src/
    text_clean.py               - text cleaning
    data_prep.py                - loading, dedup, 70/10/20 split
    features.py                 - TF-IDF
    models.py                   - the candidate models
    train.py                    - train / select / evaluate / save
    evaluate.py                 - metrics and plots
    predict.py                  - run the model on new text
  scripts/
    download_data.py            - optional Kaggle download
    leakage_demo.py             - source-leakage check
  data/                         - put Fake.csv + True.csv here
  models/                       - saved model
  reports/                      - metrics.json + figures
```

## Reproducibility

Everything is driven by one seed (`RANDOM_SEED = 42` in `config.py`): the split
and all the models. The cleaning is plain regex, the TF-IDF vocabulary is
sorted, and the solvers are deterministic, so running `python main.py` again
gives the same numbers (I checked two runs are identical). Setting
`PYTHONHASHSEED=42` before running doesn't change anything but is a safe habit.
More detail is in REPRODUCIBILITY_CHECKLIST.md.
