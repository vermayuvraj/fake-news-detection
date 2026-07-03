"""Classify new article text with the saved model.

Examples:
    python -m src.predict --text "some headline and body ..."
    python -m src.predict --file article.txt
    echo "some text" | python -m src.predict
"""

import argparse
import sys

import joblib

import config
from src.models import positive_scores
from src.text_clean import clean_text


def load_pipeline():
    model_path = config.MODELS_DIR / "fake_news_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Run `python main.py` first.")
    return joblib.load(model_path)


def predict_texts(texts):
    pipeline = load_pipeline()
    # Clean the input the same way as during training.
    cleaned = [clean_text(t, strip_artifacts=config.STRIP_SOURCE_ARTIFACTS) for t in texts]
    preds = pipeline.predict(cleaned)
    scores = positive_scores(pipeline, cleaned)
    return [
        {"prediction": int(p), "label": config.CLASS_NAMES[int(p)], "fake_score": float(s)}
        for p, s in zip(preds, scores)
    ]


def _read_input(args):
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input. Use --text, --file, or pipe text via stdin.")
    return data


def main():
    ap = argparse.ArgumentParser(description="Classify a news article as real or fake.")
    ap.add_argument("--text", type=str, help="Article text to classify.")
    ap.add_argument("--file", type=str, help="Path to a .txt file with the article.")
    args = ap.parse_args()

    result = predict_texts([_read_input(args)])[0]
    print(f"Prediction : {result['label'].upper()} (class {result['prediction']})")
    print(f"Fake-score : {result['fake_score']:.4f} (higher = more likely fake)")


if __name__ == "__main__":
    main()
