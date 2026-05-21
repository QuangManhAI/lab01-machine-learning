from __future__ import annotations

import html
import logging
import math
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


logger = logging.getLogger(__name__)

METRICS = Path("data/processed/metrics")
VALID_LABELS = {"ham", "spam"}

HTML_ARTIFACT_STOPWORDS = {
    "align",
    "alink",
    "arial",
    "bgcolor",
    "blockquote",
    "border",
    "br",
    "cellpadding",
    "cellspacing",
    "center",
    "charset",
    "class",
    "colspan",
    "content",
    "css",
    "div",
    "doctype",
    "face",
    "font",
    "fontsize",
    "height",
    "href",
    "html",
    "http",
    "https",
    "img",
    "javascript",
    "mailto",
    "nbsp",
    "rowspan",
    "script",
    "span",
    "src",
    "style",
    "table",
    "tbody",
    "td",
    "textarea",
    "tfoot",
    "thead",
    "title",
    "tr",
    "valign",
    "width",
    "www",
}

EMAIL_ARTIFACT_STOPWORDS = {
    "bcc",
    "cc",
    "charset",
    "content",
    "date",
    "delivered",
    "encoding",
    "envelope",
    "from",
    "fw",
    "fwd",
    "id",
    "mime",
    "multipart",
    "plain",
    "quoted",
    "received",
    "reply",
    "return",
    "subject",
    "text",
    "to",
    "utf",
    "version",
}

DATASET_ARTIFACT_STOPWORDS = {
    "com",
    "email",
    "emailtoken",
    "escapelong",
    "escapenumber",
    "hextoken",
    "numbertoken",
    "urltoken",
}

JAVASCRIPT_STOPWORDS = {
    "addEventListener",
    "alert",
    "button",
    "click",
    "const",
    "document",
    "else",
    "false",
    "function",
    "getelementbyid",
    "innerhtml",
    "let",
    "onclick",
    "return",
    "true",
    "var",
    "window",
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
        if len(token) < 2:
            continue
        if token in CUSTOM_STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


def add_preprocessing_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in ["subject", "body", "source", "label"]:
        if column not in data.columns:
            data[column] = ""
    data["label"] = data["label"].fillna("").str.lower().str.strip()
    data["source_family"] = data["source"].map(source_family)
    data["text"] = data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    data["clean_text"] = data["text"].map(clean_email_text)
    data["raw_char_count"] = data["text"].fillna("").str.len()
    data["clean_char_count"] = data["clean_text"].fillna("").str.len()
    data["clean_word_count"] = data["clean_text"].fillna("").str.split().str.len()
    return data


def filter_trainable_rows(data: pd.DataFrame, min_clean_words: int, min_clean_chars: int) -> pd.DataFrame:
    before = len(data)
    data = data[data["label"].isin(VALID_LABELS)].copy()
    data = data[data["clean_word_count"] >= min_clean_words]
    data = data[data["clean_char_count"] >= min_clean_chars]
    data = data[data["clean_text"].str.contains(r"[a-z]", regex=True, na=False)]
    logger.info("Filtered trainable rows: before=%s after=%s", before, len(data))
    return data


def balance_dataset(
    data: pd.DataFrame,
    max_per_source_family: int,
    random_seed: int,
) -> pd.DataFrame:
    capped = _cap_each_source_family(data, max_per_source_family, random_seed)
    label_counts = capped["label"].value_counts()
    if not {"ham", "spam"}.issubset(label_counts.index):
        raise ValueError(f"Cannot balance without both labels. Label counts: {label_counts.to_dict()}")
    target_per_label = int(label_counts[["ham", "spam"]].min())
    logger.info(
        "Balance target: capped_rows=%s label_counts=%s target_per_label=%s",
        len(capped),
        label_counts.to_dict(),
        target_per_label,
    )
    balanced_parts = []
    for label in ["ham", "spam"]:
        label_frame = capped[capped["label"] == label]
        balanced_parts.append(_sample_evenly_by_source_family(label_frame, target_per_label, random_seed))
    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)


def write_preprocessing_balance_report(
    full_data: pd.DataFrame,
    trainable_data: pd.DataFrame,
    balanced_data: pd.DataFrame,
    path: Path | None = None,
):
    path = path or METRICS / "preprocessing_balance_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_source_label = _source_label_table(trainable_data)
    balanced_source_label = _source_label_table(balanced_data)
    artifact_counts = _artifact_counts(full_data)
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
        "",
        "```text",
        trainable_data["label"].value_counts().to_string(),
        "```",
        "",
        "### After Balance",
        "",
        "```text",
        balanced_data["label"].value_counts().to_string(),
        "```",
        "",
        "## Source Family By Label Before Balance",
        "",
        "```text",
        trainable_source_label.to_string(index=False),
        "```",
        "",
        "## Source Family By Label After Balance",
        "",
        "```text",
        balanced_source_label.to_string(index=False),
        "```",
        "",
        "## HTML/Script Artifact Reduction Check",
        "",
        "```text",
        artifact_counts.to_string(index=False),
        "```",
        "",
        "## Reading",
        "",
        "- `emails_full.csv` keeps the full cleaned export for audit.",
        "- `emails.csv` is the balanced dataset used by EDA and training.",
        "- Balancing is label-first, then source-family-even within each label.",
        "- One-label sources cannot be made internally ham/spam balanced, so ham-only archive sources are sampled lower while spam-capable sources contribute more spam.",
        "- That shape is intentional: equal source-family totals would make the whole dataset ham-heavy again.",
    ]
    path.write_text("\n".join(lines))
    logger.info("Preprocessing/balance report saved: %s", path)


def stop_words_for_vectorizer() -> list[str]:
    return sorted(CUSTOM_STOPWORDS)


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


def _cap_each_source_family(data: pd.DataFrame, max_per_source_family: int, random_seed: int) -> pd.DataFrame:
    if max_per_source_family <= 0:
        return data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    pieces = []
    for source, group in data.groupby("source_family", sort=True):
        if len(group) <= max_per_source_family:
            pieces.append(group)
            continue
        pieces.append(_sample_preserving_source_labels(group, max_per_source_family, random_seed))
        logger.info("Source family capped: source_family=%s before=%s after=%s", source, len(group), max_per_source_family)
    return pd.concat(pieces, ignore_index=True)


def _sample_preserving_source_labels(group: pd.DataFrame, target: int, random_seed: int) -> pd.DataFrame:
    labels = sorted(group["label"].unique())
    if len(labels) == 1:
        return group.sample(n=target, random_state=random_seed)

    quotas = _balanced_quotas(group["label"].value_counts().to_dict(), target)
    pieces = []
    for label, quota in quotas.items():
        label_group = group[group["label"] == label]
        pieces.append(label_group.sample(n=quota, random_state=random_seed))
    return pd.concat(pieces, ignore_index=True)


def _sample_evenly_by_source_family(frame: pd.DataFrame, target: int, random_seed: int) -> pd.DataFrame:
    available = frame["source_family"].value_counts().sort_index().to_dict()
    quotas = _balanced_quotas(available, target)
    pieces = []
    for source, quota in quotas.items():
        group = frame[frame["source_family"] == source]
        pieces.append(group.sample(n=quota, random_state=random_seed))
    return pd.concat(pieces, ignore_index=True)


def _balanced_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0 or not available:
        return {}
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


def _source_label_table(data: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(data["source_family"], data["label"])
    for label in ["ham", "spam"]:
        if label not in table.columns:
            table[label] = 0
    table = table[["ham", "spam"]]
    table["total"] = table.sum(axis=1)
    table["spam_rate"] = (table["spam"] / table["total"]).replace([math.inf, -math.inf], 0).fillna(0).round(4)
    return table.reset_index().sort_values(["total", "source_family"], ascending=[False, True])


def _artifact_counts(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = data["text"].fillna("").str.lower()
    clean = data["clean_text"].fillna("")
    for token in ["html", "font", "nbsp", "script", "javascript", "br", "td", "href", "http", "www"]:
        rows.append(
            {
                "token": token,
                "raw_rows": int(raw.str.contains(rf"\b{re.escape(token)}\b", regex=True).sum()),
                "clean_rows": int(clean.str.contains(rf"\b{re.escape(token)}\b", regex=True).sum()),
            }
        )
    return pd.DataFrame(rows)
