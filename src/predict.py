import sys
from pathlib import Path

import joblib

from src.error_logging import run_logged


MODEL = Path("models/spam_nb.joblib")


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        raise SystemExit("Usage: python -m src.predict \"email text\"")
    model = joblib.load(MODEL)
    label = model.predict([text])[0]
    score = model.predict_proba([text]).max()
    print(f"{label} {score:.4f}")


if __name__ == "__main__":
    run_logged(main)
