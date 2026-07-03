# Model Card - Fake News Detection

## Overview

A binary text classifier that labels a news article (title + body) as real or
fake. It is a linear model (Passive-Aggressive) on top of TF-IDF features
(unigrams + bigrams). Chosen for speed, interpretability, and reproducibility.

- Task: binary text classification (fake = 1, real = 0)
- Input: raw article text (title and body)
- Output: predicted label + a score for the "fake" class
- Training data: Kaggle "Fake and Real News Dataset" (C. Bisaillon)

## Metrics (held-out 20% test set)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9912 |
| Precision | 0.9933 |
| Recall    | 0.9877 |
| F1-score  | 0.9905 |
| AUC-ROC   | 0.9995 |

## How it was built

- Combine title + body, lower-case, strip Reuters source tags, remove
  URLs/punctuation/digits, remove English stop-words.
- Remove duplicate articles before splitting (stratified 70/10/20).
- Fit TF-IDF on the training split only; compare four linear models on the
  validation split; report the best on the test split.
- Fixed seed (42); fully reproducible. See the repo for code and the report.

## Intended use and limitations

This model is a demonstration, not a fact-checker. The dataset contrasts a
single wire service (real) with a set of flagged sites (fake), so the classifier
largely learns writing style and source rather than the truthfulness of claims.
Expect much lower accuracy on news from sources or time periods it hasn't seen
(the data is ~2016-2017 US politics). It is English-only and uses no external
knowledge. Do not use it to make real moderation or credibility decisions.

## License

MIT.
