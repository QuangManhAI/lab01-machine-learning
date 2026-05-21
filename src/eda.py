from pathlib import Path
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from src.error_logging import run_logged
from src.preprocess_balance import source_family, stop_words_for_vectorizer


DATA = Path("data/processed/emails.csv")
FIGURES = Path("reports/figures")
METRICS = Path("data/processed/metrics")
STAGE = "after_process"
TEXT_COLUMN = "clean_text"
EXCLUDED_SOURCE_PREFIXES = ("w3c_",)
LABEL_COLORS = {"ham": "#2d6a4f", "spam": "#b23a48"}
logger = logging.getLogger(__name__)


def prepare_data(data, text_column):
    data = data[~data["source"].fillna("").str.startswith(EXCLUDED_SOURCE_PREFIXES)].copy()
    if "source_family" not in data.columns:
        data["source_family"] = data["source"].map(source_family)
    if text_column not in data.columns:
        logger.warning("Requested text column %s is missing; using raw text", text_column)
        text_column = "text"
    data["char_count"] = data["text"].fillna("").str.len()
    data["word_count"] = data["text"].fillna("").str.split().str.len()
    data["analysis_text"] = data[text_column].fillna("")
    data["analysis_char_count"] = data["analysis_text"].str.len()
    data["analysis_word_count"] = data["analysis_text"].str.split().str.len()
    if "clean_text" in data.columns:
        data["clean_char_count"] = data["clean_text"].fillna("").str.len()
        data["clean_word_count"] = data["clean_text"].fillna("").str.split().str.len()
    else:
        data["clean_char_count"] = data["analysis_char_count"]
        data["clean_word_count"] = data["analysis_word_count"]
    data["subject_chars"] = data["subject"].fillna("").str.len()
    return data


def save_label_plot(data):
    logger.info("Save label plot")
    counts = data["label"].value_counts()
    counts.plot(kind="bar", color=[LABEL_COLORS.get(label, "#586f7c") for label in counts.index])
    plt.title("Email labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "label_distribution.png")
    plt.close()


def save_source_plot(data):
    logger.info("Save source family contribution plot")
    counts = data["source_family"].value_counts().sort_values()
    height = max(5, len(counts) * 0.45)
    axis = counts.plot(kind="barh", figsize=(10, height), color="#586f7c")
    for index, value in enumerate(counts):
        axis.text(value + max(counts.max() * 0.01, 2), index, str(value), va="center", fontsize=8)
    axis.axvline(1000, color="#b23a48", linestyle="--", linewidth=1)
    plt.title("Emails by source family")
    plt.xlabel("Count")
    plt.ylabel("Source family")
    plt.xlim(0, max(counts.max() * 1.12, 1100))
    plt.tight_layout()
    plt.savefig(FIGURES / "source_family_distribution.png")
    plt.close()


def save_source_label_plot(source_label):
    logger.info("Save source-family label plot")
    plot_data = source_label.sort_values("total").set_index("source_family")[["ham", "spam"]]
    height = max(5, len(plot_data) * 0.45)
    plot_data.plot(kind="barh", stacked=True, figsize=(10, height), color=[LABEL_COLORS["ham"], LABEL_COLORS["spam"]])
    plt.title("Ham/spam by source family")
    plt.xlabel("Count")
    plt.ylabel("Source family")
    plt.tight_layout()
    plt.savefig(FIGURES / "source_family_label_distribution.png")
    plt.close()


def save_wordcloud(data):
    logger.info("Save overall word cloud")
    save_wordcloud_image(data, "top_words_wordcloud.png", "Most frequent terms", "viridis")
    for label, color_map in [("ham", "Greens"), ("spam", "Reds")]:
        label_data = data[data["label"] == label]
        if not label_data.empty:
            save_wordcloud_image(label_data, f"{label}_wordcloud.png", f"Most frequent terms: {label}", color_map)


def save_wordcloud_image(data, filename, title, colormap):
    vectorizer = CountVectorizer(stop_words=stage_stop_words(), max_features=250)
    try:
        matrix = vectorizer.fit_transform(data["analysis_text"].fillna(""))
    except ValueError as exc:
        logger.warning("Skip word cloud %s: %s", filename, exc)
        return
    frequencies = dict(zip(vectorizer.get_feature_names_out(), matrix.sum(axis=0).A1))
    if not frequencies:
        logger.warning("Skip empty word cloud: %s", filename)
        return
    cloud = WordCloud(width=1400, height=800, background_color="white", colormap=colormap).generate_from_frequencies(frequencies)
    plt.figure(figsize=(12, 7))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename)
    plt.close()


def save_text_length_boxplots(data):
    logger.info("Save text length boxplots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    data.boxplot(column="analysis_char_count", by="label", ax=axes[0])
    axes[0].set_title("Characters by label")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Characters")
    families = data.groupby("source_family")["analysis_char_count"].median().sort_values().index.tolist()
    ordered = data.copy()
    ordered["source_family"] = pd.Categorical(ordered["source_family"], categories=families, ordered=True)
    ordered.boxplot(column="analysis_char_count", by="source_family", ax=axes[1], rot=35)
    axes[1].set_title("Characters by source family")
    axes[1].set_xlabel("Source family")
    axes[1].set_ylabel("Characters")
    axes[1].set_xticklabels(families, ha="right")
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(FIGURES / "text_length_boxplots.png")
    plt.close()


def save_scatter(data):
    logger.info("Save text shape scatter")
    sample = data.sample(min(len(data), 6000), random_state=42) if len(data) else data
    plt.figure(figsize=(10, 7))
    for label, group in sample.groupby("label"):
        plt.scatter(
            group["analysis_word_count"].clip(upper=5000),
            group["analysis_char_count"].clip(upper=50000),
            s=10,
            alpha=0.28,
            c=LABEL_COLORS.get(label, "#586f7c"),
            label=label,
        )
    plt.title("Email length shape")
    plt.xlabel("Word count, clipped at 5,000")
    plt.ylabel("Character count, clipped at 50,000")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.savefig(FIGURES / "text_shape_scatter.png")
    plt.close()


def save_preprocessing_scatter(data):
    logger.info("Save raw vs clean length scatter")
    sample = data.sample(min(len(data), 6000), random_state=42) if len(data) else data
    plt.figure(figsize=(10, 7))
    for label, group in sample.groupby("label"):
        plt.scatter(
            group["char_count"].clip(upper=80000),
            group["clean_char_count"].clip(upper=80000),
            s=10,
            alpha=0.28,
            c=LABEL_COLORS.get(label, "#586f7c"),
            label=label,
        )
    plt.title("Raw vs clean text length")
    plt.xlabel("Raw characters, clipped at 80,000")
    plt.ylabel("Clean characters, clipped at 80,000")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.savefig(FIGURES / "raw_vs_clean_length_scatter.png")
    plt.close()


def build_source_label_table(data):
    table = pd.crosstab(data["source_family"], data["label"])
    for label in ["ham", "spam"]:
        if label not in table.columns:
            table[label] = 0
    table = table[["ham", "spam"]]
    table["total"] = table.sum(axis=1)
    table["spam_rate"] = (table["spam"] / table["total"]).round(4)
    return table.reset_index().sort_values("total", ascending=False)


def save_data_quality_report(data, source_label):
    logger.info("Save EDA data quality report")
    report = [
        f"# Data Quality Report: {STAGE}",
        "",
        f"Rows: {len(data)}",
        f"Raw sources: {data['source'].nunique()}",
        f"Source families: {data['source_family'].nunique()}",
        f"Text column for words/plots: {TEXT_COLUMN if TEXT_COLUMN in data.columns else 'text'}",
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
        data[
            ["char_count", "word_count", "analysis_char_count", "analysis_word_count", "subject_chars"]
        ].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string(),
        "```",
        "",
        "## Source Family Contribution",
        "",
        "```text",
        source_label.to_string(index=False),
        "```",
        "",
        "## EDA Reading",
        "",
        "- Source family distribution checks whether one corpus family dominates the training set.",
        "- Ham/spam by source family reveals label leakage risk from one-label corpora.",
        "- Boxplots show whether spam and ham differ by message length rather than content.",
        "- Scatter plot exposes outliers, very long threads, and length-based separation that can inflate metrics.",
        "- Word cloud is only for quick vocabulary inspection; do not treat it as model evidence.",
    ]
    (METRICS / "data_quality_report.md").write_text("\n".join(report))


def save_label_reports(data):
    logger.info("Save separate ham/spam reports")
    for label in ["ham", "spam"]:
        label_data = data[data["label"] == label]
        if label_data.empty:
            continue
        source_counts = label_data["source_family"].value_counts()
        vectorizer = CountVectorizer(stop_words=stage_stop_words(), max_features=25)
        try:
            matrix = vectorizer.fit_transform(label_data["analysis_text"].fillna(""))
            words = pd.Series(matrix.sum(axis=0).A1, index=vectorizer.get_feature_names_out()).sort_values(ascending=False)
        except ValueError:
            words = pd.Series(dtype=int)
        report = [
            f"# {label.upper()} EDA Report",
            "",
            f"Rows: {len(label_data)}",
            f"Source families: {label_data['source_family'].nunique()}",
            "",
            "## Source Family Contribution",
            "",
            "```text",
            source_counts.to_string(),
            "```",
            "",
            "## Length Statistics",
            "",
            "```text",
            label_data[
                ["char_count", "word_count", "analysis_char_count", "analysis_word_count", "subject_chars"]
            ].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string(),
            "```",
            "",
            "## Top Terms",
            "",
            "```text",
            words.to_string(),
            "```",
            "",
            "## Reading",
            "",
            label_reading(label),
        ]
        (METRICS / f"{label}_eda_report.md").write_text("\n".join(report))


def label_reading(label):
    if label == "spam":
        return "- Use this report to check whether spam language is dominated by offers, urgency, money, or corpus-specific artifacts."
    return "- Use this report to check whether ham is dominated by technical mailing-list language, business mail, or one source family."


def stage_stop_words():
    if TEXT_COLUMN == "clean_text":
        return stop_words_for_vectorizer()
    return "english"


def parse_args():
    parser = argparse.ArgumentParser(description="Create EDA figures and metrics for a dataset snapshot.")
    parser.add_argument("--input", default=str(DATA), help="CSV input path.")
    parser.add_argument("--figures", default=str(FIGURES), help="Output folder for image figures.")
    parser.add_argument("--metrics", default=str(METRICS), help="Output folder for text/table metrics.")
    parser.add_argument("--stage", default=STAGE, help="EDA stage name, e.g. before_process or after_process.")
    parser.add_argument("--text-column", default=TEXT_COLUMN, help="Text column used for word clouds and top terms.")
    return parser.parse_args()


def main():
    global DATA, FIGURES, METRICS, STAGE, TEXT_COLUMN
    args = parse_args()
    DATA = Path(args.input)
    FIGURES = Path(args.figures)
    METRICS = Path(args.metrics)
    STAGE = args.stage
    TEXT_COLUMN = args.text_column
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    logger.info("EDA folders ready: figures=%s metrics=%s stage=%s input=%s", FIGURES, METRICS, STAGE, DATA)
    data = prepare_data(pd.read_csv(DATA).fillna(""), TEXT_COLUMN)
    logger.info(
        "EDA loaded data: rows=%s labels=%s text_column=%s",
        len(data),
        data["label"].value_counts().to_dict(),
        TEXT_COLUMN,
    )
    source_label = build_source_label_table(data)
    source_label.to_csv(METRICS / "source_family_label_distribution.csv", index=False)
    data["source_family"].value_counts().rename_axis("source_family").reset_index(name="count").to_csv(
        METRICS / "source_family_distribution.csv",
        index=False,
    )
    save_label_plot(data)
    save_source_plot(data)
    save_source_label_plot(source_label)
    save_wordcloud(data)
    save_text_length_boxplots(data)
    save_scatter(data)
    if "clean_text" in data.columns and TEXT_COLUMN == "clean_text":
        save_preprocessing_scatter(data)
    save_data_quality_report(data, source_label)
    save_label_reports(data)
    logger.info("EDA finished: %s", FIGURES)
    print(f"Saved EDA plots to {FIGURES}")


if __name__ == "__main__":
    run_logged(main)
