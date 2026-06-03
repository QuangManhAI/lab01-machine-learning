from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

import preprocess
import model_from_scratch


LABEL_COLORS = {"ham": "#2d6a4f", "spam": "#b23a48"}


def prepare_data(data: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
    data = data.copy()
    if "source_family" not in data.columns:
        data["source_family"] = data["source"].map(preprocess.source_family)
    if text_column not in data.columns:
        text_column = "text"
    data["char_count"] = data["text"].fillna("").str.len()
    data["word_count"] = data["text"].fillna("").str.split().str.len()
    data["analysis_text"] = data[text_column].fillna("")
    data["analysis_char_count"] = data["analysis_text"].str.len()
    data["analysis_word_count"] = data["analysis_text"].str.split().str.len()
    data["subject_chars"] = data["subject"].fillna("").str.len() if "subject" in data.columns else 0
    return data


def source_label_table(data: pd.DataFrame) -> pd.DataFrame:
    eda_data = prepare_data(data, "clean_text")
    table = pd.crosstab(eda_data["source_family"], eda_data["label"])
    for label in ["ham", "spam"]:
        if label not in table.columns:
            table[label] = 0
    table = table[["ham", "spam"]]
    table["total"] = table.sum(axis=1)
    table["spam_rate"] = (table["spam"] / table["total"]).round(4)
    return table.reset_index().sort_values("total", ascending=False)


def length_summary(data: pd.DataFrame) -> pd.DataFrame:
    eda_data = prepare_data(data, "clean_text")
    return eda_data[["analysis_char_count", "analysis_word_count", "subject_chars"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95]
    )


def top_terms(data: pd.DataFrame, text_column: str = "clean_text", max_features: int = 25) -> pd.DataFrame:
    vectorizer = CountVectorizer(stop_words=preprocess.stop_words_for_vectorizer(), max_features=max_features)
    matrix = vectorizer.fit_transform(data[text_column].fillna(""))
    counts = matrix.sum(axis=0).A1
    return pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": counts}).sort_values("count", ascending=False)


def plot_label_distribution(data: pd.DataFrame, title: str = "Label distribution") -> None:
    counts = data["label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = [LABEL_COLORS.get(label, "#586f7c") for label in counts.index]

    counts.plot(kind="bar", ax=axes[0], color=colors)
    axes[0].set_title(title)
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
    axes[1].set_title("Label ratio")
    plt.tight_layout()
    plt.show()


def plot_source_label_distribution(data: pd.DataFrame, title: str = "Ham/spam by source family", top_n: int = 20) -> None:
    table = source_label_table(data).head(top_n)
    plot_data = table.set_index("source_family")[["ham", "spam"]].sort_values("ham", ascending=True)
    height = max(5, len(plot_data) * 0.42)
    plot_data.plot(kind="barh", stacked=True, figsize=(11, height), color=[LABEL_COLORS["ham"], LABEL_COLORS["spam"]])
    plt.title(title)
    plt.xlabel("Rows")
    plt.ylabel("Source family")
    plt.tight_layout()
    plt.show()


def plot_text_shape_scatter(data: pd.DataFrame, title: str = "Email length shape", sample_size: int = 4000) -> None:
    eda_data = prepare_data(data, "clean_text")
    sample = eda_data.sample(min(len(eda_data), sample_size), random_state=42) if len(eda_data) else eda_data
    plt.figure(figsize=(9, 6))
    for label, group in sample.groupby("label"):
        plt.scatter(
            group["analysis_word_count"].clip(upper=5000),
            group["analysis_char_count"].clip(upper=50000),
            s=10,
            alpha=0.28,
            c=LABEL_COLORS.get(label, "#586f7c"),
            label=label,
        )
    plt.title(title)
    plt.xlabel("Word count, clipped at 5,000")
    plt.ylabel("Character count, clipped at 50,000")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()


def plot_top_terms(data: pd.DataFrame, title: str = "Top terms", text_column: str = "clean_text", max_features: int = 20) -> None:
    terms = top_terms(data, text_column=text_column, max_features=max_features).sort_values("count")
    plt.figure(figsize=(10, 6))
    plt.barh(terms["term"], terms["count"], color="#586f7c")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.show()


def plot_eda_overview(data: pd.DataFrame, title_prefix: str) -> None:
    plot_label_distribution(data, f"{title_prefix}: label distribution")
    plot_source_label_distribution(data, f"{title_prefix}: ham/spam by source family")
    plot_text_shape_scatter(data, f"{title_prefix}: text length shape")
    plot_top_terms(data, f"{title_prefix}: top terms")


def figure_files(after_figures: Path) -> list[Path]:
    return [
        after_figures / "label_distribution.png",
        after_figures / "source_family_label_distribution.png",
        after_figures / "text_shape_scatter.png",
        after_figures / "top_words_wordcloud.png",
    ]


def display_figures(after_figures: Path, project_root: Path) -> None:
    for figure in figure_files(after_figures):
        if figure.exists():
            display(Markdown(f"**{figure.relative_to(project_root)}**"))
            display(Image(filename=str(figure)))
        else:
            print(f"Missing figure: {figure}")


def classification_report_text(y_test, predictions) -> str:
    return classification_report(y_test, predictions, zero_division=0)


def top_tokens(model, top_n: int = 15) -> pd.DataFrame:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["nb"]
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for class_index, label in enumerate(classifier.classes_):
        top_indices = classifier.feature_log_prob_[class_index].argsort()[-top_n:][::-1]
        rows.extend({"label": label, "token": feature_names[index], "rank": rank} for rank, index in enumerate(top_indices, start=1))
    return pd.DataFrame(rows).pivot(index="rank", columns="label", values="token")


def per_source_scores(test_data: pd.DataFrame, predictions) -> pd.DataFrame:
    scored = test_data[["source_family", "label"]].copy()
    scored["prediction"] = predictions
    rows = []
    for source_family, group in scored.groupby("source_family"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            group["label"], group["prediction"], average="macro", zero_division=0
        )
        rows.append(
            {
                "source_family": source_family,
                "rows": len(group),
                "accuracy": accuracy_score(group["label"], group["prediction"]),
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1", "rows"], ascending=[True, False])


def cross_source_holdout(data: pd.DataFrame, text_column: str, metrics_dir: Path) -> pd.DataFrame:
    cross_source_path = metrics_dir / "cross_source_holdout_report.csv"
    if cross_source_path.exists():
        return pd.read_csv(cross_source_path).sort_values("macro_f1")
    return model_from_scratch.SklearnModelChecker().evaluate_cross_source_holdout(data, text_column)
