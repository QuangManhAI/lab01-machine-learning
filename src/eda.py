from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.error_logging import run_logged


DATA = Path("data/processed/emails.csv")
FIGURES = Path("reports/figures")
METRICS = Path("data/processed/metrics")
logger = logging.getLogger(__name__)


def save_label_plot(data):
    logger.info("Save label plot")
    counts = data["label"].value_counts()
    counts.plot(kind="bar", color=["#2d6a4f", "#b23a48"])
    plt.title("Email labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "label_distribution.png")
    plt.close()


def save_length_plot(data):
    logger.info("Save length plot")
    data.assign(length=data["text"].str.len()).boxplot(column="length", by="label")
    plt.title("Text length by label")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel("Characters")
    plt.tight_layout()
    plt.savefig(FIGURES / "text_length_by_label.png")
    plt.close()


def save_top_words(data):
    logger.info("Save top words plot")
    vectorizer = CountVectorizer(stop_words="english", max_features=20)
    matrix = vectorizer.fit_transform(data["text"].fillna(""))
    words = pd.Series(matrix.sum(axis=0).A1, index=vectorizer.get_feature_names_out())
    words.sort_values().plot(kind="barh", color="#457b9d")
    plt.title("Top words")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES / "top_words.png")
    plt.close()


def save_source_plot(data):
    logger.info("Save source contribution plot")
    counts = data["source"].value_counts().sort_values()
    height = max(5, len(counts) * 0.35)
    counts.plot(kind="barh", figsize=(10, height), color="#586f7c")
    plt.title("Emails by source")
    plt.xlabel("Count")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.savefig(FIGURES / "source_distribution.png")
    plt.close()


def save_source_label_plot(source_label):
    logger.info("Save source-label contribution plot")
    plot_data = source_label.sort_values("total").set_index("source")[["ham", "spam"]]
    height = max(5, len(plot_data) * 0.35)
    plot_data.plot(kind="barh", stacked=True, figsize=(10, height), color=["#2d6a4f", "#b23a48"])
    plt.title("Ham/spam by source")
    plt.xlabel("Count")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.savefig(FIGURES / "source_label_distribution.png")
    plt.close()


def build_source_label_table(data):
    table = pd.crosstab(data["source"], data["label"])
    for label in ["ham", "spam"]:
        if label not in table.columns:
            table[label] = 0
    table = table[["ham", "spam"]]
    table["total"] = table.sum(axis=1)
    table["spam_rate"] = (table["spam"] / table["total"]).round(4)
    return table.reset_index().sort_values("total", ascending=False)


def save_data_quality_report(data, source_label):
    logger.info("Save EDA data quality report")
    text_lengths = data["text"].fillna("").str.len()
    report = [
        "# Data Quality Report",
        "",
        f"Rows: {len(data)}",
        f"Sources: {data['source'].nunique()}",
        "",
        "## Label Distribution",
        "",
        "```text",
        data["label"].value_counts().to_string(),
        "```",
        "",
        "## Text Length",
        "",
        "```text",
        text_lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string(),
        "```",
        "",
        "## Source Contribution",
        "",
        "```text",
        source_label.to_string(index=False),
        "```",
        "",
        "## Notes",
        "",
        "- Sources with only ham or only spam are useful, but they can make random test metrics look too optimistic.",
        "- Train/test reports should be checked by both label and source before trusting accuracy.",
    ]
    (METRICS / "data_quality_report.md").write_text("\n".join(report))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    logger.info("EDA folder ready: %s", FIGURES)
    data = pd.read_csv(DATA).fillna("")
    logger.info("EDA loaded data: rows=%s labels=%s", len(data), data["label"].value_counts().to_dict())
    source_label = build_source_label_table(data)
    source_label.to_csv(METRICS / "source_label_distribution.csv", index=False)
    data["source"].value_counts().rename_axis("source").reset_index(name="count").to_csv(
        METRICS / "source_distribution.csv",
        index=False,
    )
    save_label_plot(data)
    save_length_plot(data)
    save_top_words(data)
    save_source_plot(data)
    save_source_label_plot(source_label)
    save_data_quality_report(data, source_label)
    logger.info("EDA finished: %s", FIGURES)
    print(f"Saved EDA plots to {FIGURES}")


if __name__ == "__main__":
    run_logged(main)
