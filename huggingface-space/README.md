---
title: "Fake News Detection"
emoji: "📰"
colorFrom: "green"
colorTo: "red"
sdk: "gradio"
sdk_version: "4.44.1"
app_file: "app.py"
pinned: false
license: "mit"
---

# Fake News Detection

Interactive demo of a TF-IDF + linear classifier that labels a news article as
real or fake. Trained on the Kaggle Fake and Real News dataset.

Paste a headline and body and the model returns its prediction with a
confidence score. See the full project (code, report, reproducibility) on
GitHub: https://github.com/vermayuvraj/fake-news-detection

Note: the dataset pairs one wire service (real) against a set of flagged sites
(fake), so the model mostly learns writing style and source rather than
verifying facts. It's a demo, not a fact-checker.
