from __future__ import annotations

import html
import math
import re
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


VALID_LABELS = {"ham", "spam"}
MIN_CLEAN_WORDS = 5
MIN_CLEAN_CHARS = 25
BALANCE_MAX_PER_SOURCE_FAMILY = 1000
BALANCE_RANDOM_SEED = 42

HTML_ARTIFACT_STOPWORDS = {
    "align", "alink", "arial", "bgcolor", "blockquote", "border", "br", "cellpadding", "cellspacing",
    "center", "charset", "class", "colspan", "content", "css", "div", "doctype", "face", "font",
    "fontsize", "height", "href", "html", "http", "https", "img", "javascript", "mailto", "nbsp",
    "rowspan", "script", "span", "src", "style", "table", "tbody", "td", "textarea", "tfoot",
    "thead", "title", "tr", "valign", "width", "www",
}
EMAIL_ARTIFACT_STOPWORDS = {
    "bcc", "cc", "charset", "content", "date", "delivered", "encoding", "envelope", "from", "fw",
    "fwd", "id", "mime", "multipart", "plain", "quoted", "received", "reply", "return", "subject",
    "text", "to", "utf", "version",
}
DATASET_ARTIFACT_STOPWORDS = {
    "com", "email", "emailtoken", "escapelong", "escapenumber", "hextoken", "numbertoken", "urltoken",
}
JAVASCRIPT_STOPWORDS = {
    "addEventListener", "alert", "button", "click", "const", "document", "else", "false", "function",
    "getelementbyid", "innerhtml", "let", "onclick", "return", "true", "var", "window",
}
CUSTOM_STOPWORDS = {
    token.lower()
    for token in (
        set(ENGLISH_STOP_WORDS)
        | HTML_ARTIFACT_STOPWORDS
        | EMAIL_ARTIFACT_STOPWORDS
        | DATASET_ARTIFACT_STOPWORDS
        | JAVASCRIPT_STOPWORDS
    )
}

EXAMPLE_EMAIL = """
<html><body><h1>WIN MONEY NOW!!!</h1>
Click https://example.com/prize and email winner@example.com to claim $5,000 today.
</body></html>
"""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        if tag.lower() in {"br", "p", "div", "li", "tr", "td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1
        if tag.lower() in {"p", "div", "li", "tr", "td", "th"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self._hidden_depth:
            self.parts.append(data)

    def text(self):
        return " ".join(self.parts)


def source_family(source: str) -> str:
    source = str(source)
    if source.startswith("spamassassin_"):
        return "spamassassin"
    return source


def normalize_label(value) -> str:
    text = str(value).strip().lower()
    if text in {"1", "spam", "phishing", "malicious", "bad", "true"} or "spam" in text or "phish" in text:
        return "spam"
    return "ham"


def clean_email_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = _remove_markup_and_scripts(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"=\r?\n", "", text)
    text = re.sub(r"\\[rnt]", " ", text)
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+\b", " urltoken ", text)
    text = re.sub(r"(?i)\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b", " emailtoken ", text)
    text = re.sub(r"(?i)\b[a-f0-9]{24,}\b", " hextoken ", text)
    text = re.sub(r"\b\d+(?:[.,:/-]\d+)*\b", " numbertoken ", text)
    text = re.sub(r"[_=+*#~^`|\\/<>()[\]{}]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9'\s-]", " ", text)
    text = re.sub(r"(?i)\b([a-z])\1{3,}\b", r"\1\1", text)
    text = text.lower()
    tokens = []
    for token in re.findall(r"[a-z0-9][a-z0-9'-]*", text):
        token = token.strip("'-")
        if len(token) >= 2 and token not in CUSTOM_STOPWORDS:
            tokens.append(token)
    return " ".join(tokens)


def add_preprocessing_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in ["subject", "body", "source", "label"]:
        if column not in data.columns:
            data[column] = ""
    data["label"] = data["label"].fillna("").map(normalize_label)
    data["source_family"] = data["source"].map(source_family)
    data["text"] = data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    data["clean_text"] = data["text"].map(clean_email_text)
    data["raw_char_count"] = data["text"].fillna("").str.len()
    data["clean_char_count"] = data["clean_text"].fillna("").str.len()
    data["clean_word_count"] = data["clean_text"].fillna("").str.split().str.len()
    return data


def filter_trainable_rows(data: pd.DataFrame, min_clean_words: int = MIN_CLEAN_WORDS, min_clean_chars: int = MIN_CLEAN_CHARS):
    data = data[data["label"].isin(VALID_LABELS)].copy()
    data = data[data["clean_word_count"] >= min_clean_words]
    data = data[data["clean_char_count"] >= min_clean_chars]
    data = data[data["clean_text"].str.contains(r"[a-z]", regex=True, na=False)]
    return data


def balance_dataset(data: pd.DataFrame, max_per_source_family: int = BALANCE_MAX_PER_SOURCE_FAMILY, random_seed: int = BALANCE_RANDOM_SEED):
    capped = _cap_each_source_family(data, max_per_source_family, random_seed)
    label_counts = capped["label"].value_counts()
    if not {"ham", "spam"}.issubset(label_counts.index):
        raise ValueError(f"Cannot balance without both labels. Label counts: {label_counts.to_dict()}")
    target_per_label = int(label_counts[["ham", "spam"]].min())
    parts = []
    for label in ["ham", "spam"]:
        parts.append(_sample_evenly_by_source_family(capped[capped["label"] == label], target_per_label, random_seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)


def build_raw_export(data: pd.DataFrame) -> pd.DataFrame:
    raw_data = data.copy()
    for column in ["subject", "body", "source", "label"]:
        if column not in raw_data.columns:
            raw_data[column] = ""
    raw_data["label"] = raw_data["label"].map(normalize_label)
    raw_data["source_family"] = raw_data["source"].map(source_family)
    raw_data["text"] = raw_data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    return raw_data.drop_duplicates(subset=["text", "label"])


def process_raw_dataset(raw_data: pd.DataFrame, balance: bool = True):
    full_data = add_preprocessing_columns(raw_data).drop_duplicates(subset=["clean_text", "label"])
    trainable = filter_trainable_rows(full_data)
    processed = balance_dataset(trainable) if balance else trainable.sample(frac=1, random_state=BALANCE_RANDOM_SEED).reset_index(drop=True)
    return full_data, trainable, processed


def export_processed_datasets(raw_data: pd.DataFrame, output_dir: Path = Path("data/processed"), balance: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_export = build_raw_export(raw_data)
    full_data, trainable, processed = process_raw_dataset(raw_export, balance=balance)
    columns = [
        "email_id", "source", "source_family", "source_url", "local_path", "extracted_from", "extracted_path",
        "archive_path", "label", "sender", "recipient", "subject", "body", "text", "clean_text",
        "raw_char_count", "clean_char_count", "clean_word_count",
    ]
    raw_columns = columns[:14]
    for frame, cols in [(raw_export, raw_columns), (full_data, columns), (processed, columns)]:
        for column in cols:
            if column not in frame.columns:
                frame[column] = ""
    raw_export[raw_columns].to_csv(output_dir / "emails_raw.csv", index=False)
    full_data[columns].to_csv(output_dir / "emails_full.csv", index=False)
    processed[columns].to_csv(output_dir / "emails.csv", index=False)
    write_preprocessing_balance_report(full_data, trainable, processed, output_dir / "metrics/preprocessing_balance_report.md")
    return raw_export, full_data, processed


def stop_words_for_vectorizer() -> list[str]:
    return sorted(CUSTOM_STOPWORDS)


def processed_sample(data: pd.DataFrame, rows: int = 3, random_state: int = 42) -> pd.DataFrame:
    sample = data[["label", "source_family", "subject", "body", "clean_text"]].sample(rows, random_state=random_state).copy()
    sample["body"] = sample["body"].str.slice(0, 180)
    sample["clean_text"] = sample["clean_text"].str.slice(0, 180)
    return sample


def example_cleaning() -> tuple[str, str]:
    return EXAMPLE_EMAIL.strip(), clean_email_text(EXAMPLE_EMAIL)


def missing_data_summary(data: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(data)
    missing_count = data.isna().sum()
    missing_percent = (missing_count / total_rows * 100) if total_rows else missing_count.astype(float)
    return (
        pd.DataFrame(
            {
                "column": missing_count.index,
                "missing_count": missing_count.values,
                "missing_percent": missing_percent.round(2).values,
                "dtype": [str(dtype) for dtype in data.dtypes],
            }
        )
        .sort_values(["missing_count", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def plot_missing_data(data: pd.DataFrame, title: str = "Missing data by column", include_zero: bool = True) -> None:
    summary = missing_data_summary(data)
    plot_data = summary if include_zero else summary[summary["missing_count"] > 0]
    plot_data = plot_data.sort_values("missing_count", ascending=True)

    height = max(4, len(plot_data) * 0.35)
    plt.figure(figsize=(10, height))

    if plot_data.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.show()
        return

    bars = plt.barh(plot_data["column"], plot_data["missing_count"], color="#586f7c")
    max_count = max(int(plot_data["missing_count"].max()), 1)
    plt.xlim(0, max_count * 1.12)
    for bar, percent in zip(bars, plot_data["missing_percent"]):
        width = bar.get_width()
        plt.text(width + max_count * 0.01, bar.get_y() + bar.get_height() / 2, f"{int(width)} ({percent:.2f}%)", va="center")

    plt.title(title)
    plt.xlabel("Missing rows")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.show()


def duplicate_data_summary(data: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    duplicate_mask = data.duplicated(subset=subset, keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    total_rows = len(data)
    unique_rows = total_rows - duplicate_rows
    subset_label = "all columns" if subset is None else ", ".join(subset)
    return pd.DataFrame(
        [
            {
                "subset": subset_label,
                "total_rows": total_rows,
                "unique_rows": unique_rows,
                "duplicate_rows": duplicate_rows,
                "duplicate_percent": round((duplicate_rows / total_rows * 100) if total_rows else 0, 2),
            }
        ]
    )


def plot_duplicate_data(data: pd.DataFrame, title: str = "Duplicate rows", subset: list[str] | None = None) -> None:
    summary = duplicate_data_summary(data, subset=subset).iloc[0]
    counts = pd.Series(
        {
            "Unique rows": int(summary["unique_rows"]),
            "Duplicate rows": int(summary["duplicate_rows"]),
        }
    )

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(counts.index, counts.values, color=["#2d6a4f", "#b23a48"])
    max_count = max(int(counts.max()), 1)
    plt.ylim(0, max_count * 1.12)
    for bar in bars:
        height = bar.get_height()
        percent = (height / summary["total_rows"] * 100) if summary["total_rows"] else 0
        plt.text(bar.get_x() + bar.get_width() / 2, height + max_count * 0.02, f"{int(height)} ({percent:.2f}%)", ha="center")

    plt.title(title)
    plt.xlabel("Row type")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()


def raw_to_clean_sample(raw_data: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    if raw_data.empty:
        return pd.DataFrame()
    processed = add_preprocessing_columns(raw_data[["label", "source", "subject", "body"]].head(rows).copy())
    processed["text"] = processed["text"].str.slice(0, 180)
    processed["clean_text"] = processed["clean_text"].str.slice(0, 180)
    return processed[["label", "source_family", "text", "clean_text", "raw_char_count", "clean_char_count", "clean_word_count"]]


def split_data(data: pd.DataFrame):
    stratify = data["label"]
    if {"source", "label"}.issubset(data.columns):
        source_label = data["source"].astype(str) + "__" + data["label"].astype(str)
        if source_label.value_counts().min() >= 2:
            stratify = source_label
    return train_test_split(data, test_size=0.2, random_state=42, stratify=stratify)


def split_training_data(data: pd.DataFrame):
    text_column = "clean_text" if "clean_text" in data.columns else "text"
    train_data, test_data = split_data(data)
    return {
        "text_column": text_column,
        "train_data": train_data,
        "test_data": test_data,
        "x_train": train_data[text_column],
        "y_train": train_data["label"],
        "x_test": test_data[text_column],
        "y_test": test_data["label"],
    }


def train_source_crosstab(train_data: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    return pd.crosstab(train_data["source_family"], train_data["label"]).head(top_n)


def write_preprocessing_balance_report(full_data, trainable_data, balanced_data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preprocessing And Balance Report",
        "",
        f"Raw exported rows before trainable filters: {len(full_data)}",
        f"Trainable rows after clean-text filters: {len(trainable_data)}",
        f"Balanced rows used by EDA/train: {len(balanced_data)}",
        "",
        "## Label Counts",
        "",
        "### Before Balance",
        "```text",
        trainable_data["label"].value_counts().to_string(),
        "```",
        "### After Balance",
        "```text",
        balanced_data["label"].value_counts().to_string(),
        "```",
    ]
    path.write_text("\n".join(lines))


def _remove_markup_and_scripts(text: str) -> str:
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", text)
    if re.search(r"(?is)<[a-z!/][^>]{0,200}>", text):
        parser = _HTMLTextExtractor()
        try:
            parser.feed(text)
            text = parser.text()
        except Exception:
            text = re.sub(r"(?is)<[^>]+>", " ", text)
    return text


def _cap_each_source_family(data, max_per_source_family, random_seed):
    if max_per_source_family <= 0:
        return data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    pieces = []
    for _, group in data.groupby("source_family", sort=True):
        if len(group) <= max_per_source_family:
            pieces.append(group)
        else:
            pieces.append(_sample_preserving_source_labels(group, max_per_source_family, random_seed))
    return pd.concat(pieces, ignore_index=True)


def _sample_preserving_source_labels(group, target, random_seed):
    labels = sorted(group["label"].unique())
    if len(labels) == 1:
        return group.sample(n=target, random_state=random_seed)
    quotas = _balanced_quotas(group["label"].value_counts().to_dict(), target)
    return pd.concat([group[group["label"] == label].sample(n=quota, random_state=random_seed) for label, quota in quotas.items()])


def _sample_evenly_by_source_family(frame, target, random_seed):
    quotas = _balanced_quotas(frame["source_family"].value_counts().sort_index().to_dict(), target)
    return pd.concat([frame[frame["source_family"] == source].sample(n=quota, random_state=random_seed) for source, quota in quotas.items()])


def _balanced_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    target = min(target, sum(available.values()))
    keys = sorted(available)
    base = target // len(keys)
    quotas = {key: min(available[key], base) for key in keys}
    remaining = target - sum(quotas.values())
    while remaining > 0:
        changed = False
        for key in sorted(keys, key=lambda item: (quotas[item], item)):
            if quotas[key] < available[key]:
                quotas[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return {key: quota for key, quota in quotas.items() if quota > 0}
