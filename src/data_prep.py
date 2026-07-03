"""Load the data, clean it, and make the 70/10/20 split."""

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.text_clean import clean_text


def load_raw():
    if not config.FAKE_CSV.exists() or not config.TRUE_CSV.exists():
        raise FileNotFoundError(
            f"Couldn't find the data. Put Fake.csv and True.csv in {config.DATA_DIR} "
            "(see README) or run scripts/download_data.py."
        )
    fake = pd.read_csv(config.FAKE_CSV)
    true = pd.read_csv(config.TRUE_CSV)
    fake["label"] = config.LABEL_FAKE
    true["label"] = config.LABEL_REAL
    return pd.concat([fake, true], ignore_index=True)


def build_clean_frame(df):
    # Use title + body. We ignore the subject/date columns on purpose: the
    # subject alone perfectly separates real from fake, so using it would be
    # cheating rather than learning anything from the text.
    raw_content = (
        df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    )
    cleaned = raw_content.apply(
        lambda t: clean_text(t, strip_artifacts=config.STRIP_SOURCE_ARTIFACTS)
    )

    out = pd.DataFrame({"content": cleaned, "label": df["label"].values})
    out = out[out["content"].str.len() >= config.MIN_CLEAN_CHARS]

    # Drop duplicates before splitting so an article can't end up in both the
    # train and test sets.
    if config.DROP_DUPLICATES:
        out = out.drop_duplicates(subset="content", keep="first")

    return out.reset_index(drop=True)


def split_70_10_20(df):
    X, y = df["content"], df["label"]

    # Hold out 20% for test first, then split the remaining 80% into 70/10.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_FRAC, stratify=y, random_state=config.RANDOM_SEED
    )
    val_within_temp = config.VAL_FRAC / (config.TRAIN_FRAC + config.VAL_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_within_temp, stratify=y_temp,
        random_state=config.RANDOM_SEED,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def prepare_data(verbose=True):
    raw = load_raw()
    clean = build_clean_frame(raw)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_70_10_20(clean)
    if verbose:
        print(f"[data] raw={len(raw):,}  after clean+dedup={len(clean):,}  "
              f"train/val/test={len(Xtr):,}/{len(Xva):,}/{len(Xte):,}  "
              f"fake_ratio={clean['label'].mean():.3f}")
    return (Xtr, ytr), (Xva, yva), (Xte, yte)
