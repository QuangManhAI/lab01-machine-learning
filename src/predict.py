import sys
from pathlib import Path

import joblib

from src.error_logging import run_logged
from src.preprocess_balance import clean_email_text


MODEL = Path("models/spam_nb.joblib")


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        raise SystemExit("Usage: python -m src.predict \"email text\"")
    model = joblib.load(MODEL)
    clean_text = clean_email_text(text)
    label = model.predict([clean_text])[0]
    score = model.predict_proba([clean_text]).max()
    print(f"{label} {score:.4f}")


if __name__ == "__main__":
    run_logged(main)
