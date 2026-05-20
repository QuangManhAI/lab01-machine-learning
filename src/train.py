from pathlib import Path
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.error_logging import run_logged


DATA = Path("data/processed/emails.csv")
METRICS = Path("reports/metrics/classification_report.txt")
FIGURE = Path("reports/figures/confusion_matrix.png")
MODEL = Path("models/spam_nb.joblib")
logger = logging.getLogger(__name__)


def main():
    logger.info("Train start: data=%s", DATA)
    data = pd.read_csv(DATA).fillna("")
    logger.info("Train data loaded: rows=%s labels=%s", len(data), data["label"].value_counts().to_dict())
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )
    logger.info("Train split: train=%s test=%s", len(x_train), len(x_test))

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", min_df=2)),
            ("nb", MultinomialNB()),
        ]
    )
    logger.info("Fit model: TF-IDF + MultinomialNB")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    logger.info("Predictions complete: count=%s", len(predictions))

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, predictions)
    METRICS.write_text(report)
    logger.info("Metrics saved: %s", METRICS)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.tight_layout()
    plt.savefig(FIGURE)
    plt.close()
    logger.info("Confusion matrix saved: %s", FIGURE)
    joblib.dump(model, MODEL)
    logger.info("Model saved: %s", MODEL)

    print(report)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    run_logged(main)
