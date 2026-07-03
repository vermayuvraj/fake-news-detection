"""Candidate models and a helper for the positive-class score."""

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import config


def build_models():
    seed = config.RANDOM_SEED
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000, random_state=seed),
        "multinomial_nb": MultinomialNB(alpha=0.1),
        "linear_svc": LinearSVC(C=1.0, random_state=seed),
        "passive_aggressive": PassiveAggressiveClassifier(
            C=1.0, max_iter=1000, random_state=seed),
    }


def positive_scores(model, X):
    # Use probabilities when the model has them, otherwise the signed margin.
    # Either works as a ranking for ROC/AUC.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
