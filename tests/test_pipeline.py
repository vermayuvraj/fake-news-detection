"""Unit tests for the cleaning and splitting logic (no dataset needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.text_clean import clean_text
from src.data_prep import build_clean_frame, split_70_10_20


def test_clean_text_removes_reuters_tag():
    out = clean_text("WASHINGTON (Reuters) - The Senate voted today.")
    assert "reuters" not in out
    assert "washington" not in out       # the dateline city is stripped too
    assert "senate" in out


def test_clean_text_lowercases_and_strips_symbols():
    out = clean_text("Hello, WORLD! Visit http://x.com now 123.")
    assert out == "hello world visit now"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


# Cleaning strips digits, so rows must differ by letters to stay unique.
_DIGIT_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def _num_words(i):
    return " ".join(_DIGIT_WORDS[d] for d in str(i))


def _toy_frame():
    return pd.DataFrame({
        "title": [f"headline {_num_words(i)}" for i in range(200)],
        "text": [f"some article body {_num_words(i)} with several common words"
                 for i in range(200)],
        "label": [i % 2 for i in range(200)],
    })


def test_build_clean_frame_drops_duplicates():
    df = _toy_frame()
    df.loc[0, ["title", "text"]] = df.loc[1, ["title", "text"]].values  # make a dup
    out = build_clean_frame(df)
    assert out["content"].duplicated().sum() == 0


def test_split_ratios_are_70_10_20():
    out = build_clean_frame(_toy_frame())
    (Xtr, _), (Xva, _), (Xte, _) = split_70_10_20(out)
    n = len(out)
    assert abs(len(Xtr) / n - 0.70) < 0.02
    assert abs(len(Xva) / n - 0.10) < 0.02
    assert abs(len(Xte) / n - 0.20) < 0.02


def test_splits_do_not_overlap():
    out = build_clean_frame(_toy_frame())
    (Xtr, _), (Xva, _), (Xte, _) = split_70_10_20(out)
    idx_tr, idx_va, idx_te = set(Xtr.index), set(Xva.index), set(Xte.index)
    assert idx_tr.isdisjoint(idx_va)
    assert idx_tr.isdisjoint(idx_te)
    assert idx_va.isdisjoint(idx_te)


def test_saved_model_predicts_if_present():
    """Integration check: the committed model classifies obvious inputs."""
    model_path = config.MODELS_DIR / "fake_news_model.joblib"
    if not model_path.exists():
        pytest.skip("no trained model present")
    from src.predict import predict_texts
    res = predict_texts([
        "The central bank held interest rates steady according to a statement.",
        "BREAKING you wont believe this shocking viral video watch and share now",
    ])
    assert res[0]["label"] in {"real", "fake"}
    assert set(res[0].keys()) == {"prediction", "label", "fake_score"}
