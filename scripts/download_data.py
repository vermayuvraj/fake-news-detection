"""Download the dataset from Kaggle into ./data (optional helper).

Needs `pip install kaggle` and a kaggle.json token in ~/.kaggle/. If you'd
rather not use the API, just download the CSVs manually (see README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

DATASET = "clmentbisaillon/fake-and-real-news-dataset"


def main():
    if config.FAKE_CSV.exists() and config.TRUE_CSV.exists():
        print(f"Data already in {config.DATA_DIR}. Nothing to do.")
        return

    try:
        import kaggle
    except ImportError:
        raise SystemExit(
            "The 'kaggle' package isn't installed. Run `pip install kaggle` and set "
            "up kaggle.json, or download the two CSVs manually (see README)."
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{DATASET}' into {config.DATA_DIR} ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=str(config.DATA_DIR), unzip=True)
    print("Done.")


if __name__ == "__main__":
    main()
