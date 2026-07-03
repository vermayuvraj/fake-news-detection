# Reproducibility Checklist

Project: Fake News Detection
Author: Yuvraj Verma

Every item below is done in this submission. I've noted where to find each one.

## Data

- [x] Dataset named with a public source - Kaggle "Fake and Real News Dataset"
      (link in README).
- [x] How to get the exact data is documented - README step 2 and
      scripts/download_data.py.
- [x] Preprocessing is fully specified - src/text_clean.py and src/data_prep.py,
      explained in REPORT.md.
- [x] Train/val/test split is defined and reproducible - stratified 70/10/20,
      seeded, in src/data_prep.py.
- [x] Data leakage checked and handled - subject/date dropped, Reuters tag
      stripped, duplicates removed before the split (REPORT.md).
- [x] Class balance reported - 46% fake, printed by the pipeline and saved in
      metrics.json.

## Code

- [x] Source code included and runs end to end - `python main.py`.
- [x] Code is commented and organised into small modules under src/.
- [x] Single run command, documented in the README.
- [x] Inference on new text provided - src/predict.py.
- [x] All settings live in one place (config.py) and are also saved into
      metrics.json.

## Environment

- [x] Language and version stated - Python 3.10.11.
- [x] Dependencies pinned to exact versions - requirements.txt.
- [x] Virtual-environment setup instructions in the README.
- [x] No hidden downloads at runtime - stop-words come from scikit-learn's
      built-in list, no NLTK data needed.

## Randomness

- [x] One seed (RANDOM_SEED = 42) controls everything.
- [x] Seed used in the split and in every model that has randomness.
- [x] Python and NumPy RNGs seeded in src/train.py.
- [x] Deterministic solver for Logistic Regression (liblinear); TF-IDF
      vocabulary is sorted so it's order-independent.
- [x] PYTHONHASHSEED=42 suggested in the README (results are the same either
      way).

## Evaluation

- [x] Metrics are the ones the task asks for - Accuracy, Precision, Recall, F1,
      AUC-ROC.
- [x] Model chosen on the validation set; test set used only once.
- [x] Final metrics reported on the held-out test set (metrics.json).
- [x] Results saved as files - metrics.json, the figures, and the model.
- [x] Limitations discussed - REPORT.md.

## Reproduction steps

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py        # or put Fake.csv + True.csv in data/ yourself
python main.py
```

I ran `python main.py` twice on the same machine and got identical test metrics
(Accuracy 0.9912, Precision 0.9933, Recall 0.9877, F1 0.9905, AUC-ROC 0.9995),
and identical validation numbers too, so the whole pipeline is deterministic.
