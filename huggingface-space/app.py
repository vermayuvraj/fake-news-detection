"""Gradio demo for the Fake News Detection model.

Self-contained: it only needs the model file next to it, plus the packages in
requirements.txt. Run locally with `python app.py`, or deploy as a Hugging Face
Space (it will start automatically).
"""

import re

import gradio as gr
import joblib

MODEL_PATH = "fake_news_model.joblib"
pipeline = joblib.load(MODEL_PATH)

# Same cleaning as training (kept in one place so the demo matches the model).
_DATELINE_RE = re.compile(r"^\s*[A-Za-z .,'/\-]{0,60}?\(reuters\)\s*[-–—]\s*", re.IGNORECASE)
_REUTERS_RE = re.compile(r"\(?\breuters\b\)?", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z']+")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _DATELINE_RE.sub(" ", text)
    text = _REUTERS_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def classify(text):
    if not text or not text.strip():
        return {"real": 0.0, "fake": 0.0}
    cleaned = clean_text(text)
    pred = int(pipeline.predict([cleaned])[0])
    # Turn the margin into a rough 0-1 confidence with a logistic squashing.
    margin = float(pipeline.decision_function([cleaned])[0])
    fake_prob = 1.0 / (1.0 + pow(2.718281828, -margin))
    return {"fake": fake_prob, "real": 1.0 - fake_prob}


examples = [
    ["The Federal Reserve announced it would hold interest rates steady, citing "
     "stable inflation and a resilient labor market, according to a statement "
     "released after the policy meeting."],
    ["BREAKING: You won't believe what this politician did! Watch the shocking "
     "video the mainstream media doesn't want you to see. SHARE before they "
     "delete this!"],
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=10, label="Paste a news article (title + body)"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Fake News Detection",
    description=(
        "TF-IDF + linear classifier trained on the Kaggle Fake and Real News "
        "dataset. Note: the model largely learns writing style/source, so treat "
        "it as a demo, not a fact-checker."
    ),
    examples=examples,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
