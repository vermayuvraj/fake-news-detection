"""TF-IDF feature extraction."""

from sklearn.feature_extraction.text import TfidfVectorizer

import config


def build_vectorizer():
    # Fit on the training split only, then transform val/test.
    return TfidfVectorizer(**config.TFIDF_PARAMS)
