"""Project configuration: seed, paths, and model hyperparameters."""

from pathlib import Path

# Fixed seed so the split and the models are reproducible.
RANDOM_SEED = 42

# Paths (relative to this file, so the folder can live anywhere).
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"

# Positive class is "fake" since the task is fake-news detection.
LABEL_FAKE = 1
LABEL_REAL = 0
CLASS_NAMES = ["real", "fake"]

# Train / val / test proportions (70/10/20).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20

# Cleaning options.
MIN_CLEAN_CHARS = 20          # drop articles shorter than this after cleaning
DROP_DUPLICATES = True        # remove duplicate articles before splitting
STRIP_SOURCE_ARTIFACTS = True # remove the Reuters source tag (see text_clean.py)

# TF-IDF settings.
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=50_000,
    sublinear_tf=True,
    strip_accents="unicode",
    stop_words="english",
)

# Metric used to pick the best model on the validation set.
SELECTION_METRIC = "f1"
