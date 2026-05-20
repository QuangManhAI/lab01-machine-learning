from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.error_logging import run_logged


DATA = Path("data/processed/emails.csv")
FIGURES = Path("reports/figures")


def save_label_plot(data):
    counts = data["label"].value_counts()
    counts.plot(kind="bar", color=["#2d6a4f", "#b23a48"])
    plt.title("Email labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "label_distribution.png")
    plt.close()


def save_length_plot(data):
    data.assign(length=data["text"].str.len()).boxplot(column="length", by="label")
    plt.title("Text length by label")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel("Characters")
    plt.tight_layout()
    plt.savefig(FIGURES / "text_length_by_label.png")
    plt.close()


def save_top_words(data):
    vectorizer = CountVectorizer(stop_words="english", max_features=20)
    matrix = vectorizer.fit_transform(data["text"].fillna(""))
    words = pd.Series(matrix.sum(axis=0).A1, index=vectorizer.get_feature_names_out())
    words.sort_values().plot(kind="barh", color="#457b9d")
    plt.title("Top words")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES / "top_words.png")
    plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA).fillna("")
    save_label_plot(data)
    save_length_plot(data)
    save_top_words(data)
    print(f"Saved EDA plots to {FIGURES}")


if __name__ == "__main__":
    run_logged(main)
