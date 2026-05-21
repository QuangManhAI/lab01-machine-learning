from pathlib import Path
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.error_logging import run_logged


DATA = Path("data/processed/emails.csv")
METRICS = Path("data/processed/metrics/classification_report.txt")
SPLIT_REPORT = Path("data/processed/metrics/train_test_distribution.csv")
SOURCE_REPORT = Path("data/processed/metrics/per_source_classification_report.csv")
CROSS_SOURCE_REPORT = Path("data/processed/metrics/cross_source_holdout_report.csv")
SUMMARY_REPORT = Path("data/processed/metrics/model_summary.md")
FIGURE = Path("reports/figures/confusion_matrix.png")
MODEL = Path("models/spam_nb.joblib")
logger = logging.getLogger(__name__)


def main():
    logger.info("Train start: data=%s", DATA)
    data = pd.read_csv(DATA).fillna("")
    logger.info("Train data loaded: rows=%s labels=%s", len(data), data["label"].value_counts().to_dict())
    train_data, test_data = split_data(data)
    x_train = train_data["text"]
    y_train = train_data["label"]
    x_test = test_data["text"]
    y_test = test_data["label"]
    logger.info("Train split: train=%s test=%s", len(train_data), len(test_data))

    save_split_report(train_data, test_data)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    logger.info("Baseline complete: accuracy=%s", baseline_accuracy)

    model = build_model()
    logger.info("Fit model: TF-IDF + MultinomialNB")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Predictions complete: count=%s accuracy=%s", len(predictions), accuracy)

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, predictions)
    METRICS.write_text(report)
    save_per_source_report(test_data, predictions)
    cross_source = evaluate_cross_source_holdout(data)
    save_summary_report(data, train_data, test_data, accuracy, baseline_accuracy, report, cross_source)
    logger.info("Metrics saved: %s", METRICS)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.tight_layout()
    plt.savefig(FIGURE)
    plt.close()
    logger.info("Confusion matrix saved: %s", FIGURE)
    joblib.dump(model, MODEL)
    logger.info("Model saved: %s", MODEL)

    print(report)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Saved model to {MODEL}")
    print(f"Saved split/source reports to {METRICS.parent}")


def split_data(data):
    stratify = data["label"]
    source_label = data["source"].astype(str) + "__" + data["label"].astype(str)
    if source_label.value_counts().min() >= 2:
        stratify = source_label
        logger.info("Using source+label stratified split")
    else:
        logger.warning("Using label-only stratified split because some source+label groups have fewer than 2 rows")
    return train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def build_model():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", min_df=2)),
            ("nb", MultinomialNB()),
        ]
    )


def save_split_report(train_data, test_data):
    rows = []
    for split_name, frame in [("train", train_data), ("test", test_data)]:
        table = pd.crosstab(frame["source"], frame["label"])
        for label in ["ham", "spam"]:
            if label not in table.columns:
                table[label] = 0
        table = table[["ham", "spam"]]
        table["total"] = table.sum(axis=1)
        table["split"] = split_name
        rows.append(table.reset_index())
    report = pd.concat(rows, ignore_index=True)[["split", "source", "ham", "spam", "total"]]
    report.to_csv(SPLIT_REPORT, index=False)
    logger.info("Split distribution saved: %s", SPLIT_REPORT)


def save_per_source_report(test_data, predictions):
    rows = []
    scored = test_data.assign(prediction=predictions)
    for source, group in scored.groupby("source"):
        labels = sorted(group["label"].unique())
        source_report = classification_report(
            group["label"],
            group["prediction"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "source": source,
                "rows": len(group),
                "labels": ",".join(labels),
                "accuracy": accuracy_score(group["label"], group["prediction"]),
                "macro_f1": source_report["macro avg"]["f1-score"],
            }
        )
    pd.DataFrame(rows).sort_values("rows", ascending=False).to_csv(SOURCE_REPORT, index=False)
    logger.info("Per-source report saved: %s", SOURCE_REPORT)


def evaluate_cross_source_holdout(data):
    rows = []
    for source, holdout in data.groupby("source"):
        if len(holdout) < 50:
            continue
        train = data[data["source"] != source]
        if train["label"].nunique() < 2:
            continue
        model = build_model()
        model.fit(train["text"], train["label"])
        predictions = model.predict(holdout["text"])
        labels = sorted(holdout["label"].unique())
        report = classification_report(
            holdout["label"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "holdout_source": source,
                "rows": len(holdout),
                "labels": ",".join(labels),
                "accuracy": accuracy_score(holdout["label"], predictions),
                "macro_f1": report["macro avg"]["f1-score"],
            }
        )
    result = pd.DataFrame(rows).sort_values("accuracy")
    result.to_csv(CROSS_SOURCE_REPORT, index=False)
    logger.info("Cross-source holdout report saved: %s", CROSS_SOURCE_REPORT)
    return result


def save_summary_report(data, train_data, test_data, accuracy, baseline_accuracy, report, cross_source):
    source_counts = data["source"].value_counts()
    source_label = pd.crosstab(data["source"], data["label"])
    for label in ["ham", "spam"]:
        if label not in source_label.columns:
            source_label[label] = 0
    one_label_sources = source_label[(source_label["ham"] == 0) | (source_label["spam"] == 0)]
    lines = [
        "# Model Summary",
        "",
        f"Rows: {len(data)}",
        f"Sources: {data['source'].nunique()}",
        f"Train rows: {len(train_data)}",
        f"Test rows: {len(test_data)}",
        f"Accuracy: {accuracy:.4f}",
        f"Most-frequent baseline accuracy: {baseline_accuracy:.4f}",
        "",
        "## Classification Report",
        "",
        "```text",
        report,
        "```",
        "",
        "## Largest Sources",
        "",
        "```text",
        source_counts.head(20).to_string(),
        "```",
        "",
        "## One-Label Sources",
        "",
        "```text",
        one_label_sources.to_string(),
        "```",
        "",
        "## Harder Cross-Source Holdout",
        "",
        "```text",
        cross_source.head(20).to_string(index=False) if not cross_source.empty else "No eligible source holdouts.",
        "```",
        "",
        "## Reading The Score",
        "",
        "- Treat the headline score as a first pass, not final evidence.",
        "- Check `train_test_distribution.csv` to confirm train/test keep similar source and label proportions.",
        "- Check `per_source_classification_report.csv` because a model can perform well overall while failing a smaller source.",
        "- Check `cross_source_holdout_report.csv` for the harsher source-shift test.",
    ]
    SUMMARY_REPORT.write_text("\n".join(lines))
    logger.info("Model summary saved: %s", SUMMARY_REPORT)


if __name__ == "__main__":
    run_logged(main)
