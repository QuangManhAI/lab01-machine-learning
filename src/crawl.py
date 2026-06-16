from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import tarfile
import zipfile
from codecs import lookup
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


HEADERS = {"User-Agent": "email-spam-lab/1.0"}
TEXT_COLUMNS = ["text", "message", "email", "body", "content", "Text", "Message", "Email Text", "Ticket Description"]
LABEL_COLUMNS = ["label", "Label", "spam", "Spam/Ham", "target", "class", "Category"]
RAW_COLUMNS = [
    "email_id", "source", "source_family", "source_url", "local_path", "extracted_from", "extracted_path",
    "archive_path", "label", "sender", "recipient", "subject", "body", "text",
]


def project_root(start: Path | None = None) -> Path:
    root = Path.cwd() if start is None else Path(start)
    if root.name == "notebooks":
        root = root.parent
    return root.resolve()


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = project_root(root)
    return {
        "project_root": root,
        "data": root / "data/processed/emails.csv",
        "raw_data": root / "data/processed/emails_raw.csv",
        "full_data": root / "data/processed/emails_full.csv",
        "model": root / "models/spam_nb.joblib",
        "metrics_dir": root / "data/processed/metrics",
        "after_figures": root / "reports/figures/after_process",
        "corpora_config": root / "config/corpora_sources.json",
        "crawler_config": root / "config/crawler_sources.json",
        "download_dir": root / "data/raw/downloads",
    }


def load_datasets(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = project_paths(root)
    if not paths["data"].exists():
        raise FileNotFoundError(
            f"Missing {paths['data']}. Run `python notebooks/crawl.py`, then process raw data from the notebook."
        )
    data = pd.read_csv(paths["data"]).fillna("")
    raw_data = pd.read_csv(paths["raw_data"]).fillna("") if paths["raw_data"].exists() else pd.DataFrame()
    return data, raw_data, paths


def crawl_to_raw_csv(root: Path | None = None, include_live_archives: bool = True) -> pd.DataFrame:
    paths = project_paths(root)
    items = []
    if paths["corpora_config"].exists():
        items.extend(download_corpora(paths["corpora_config"], paths["download_dir"]))
    if include_live_archives and paths["crawler_config"].exists():
        items.extend(crawl_live_archives(paths["crawler_config"]))
    data = pd.DataFrame(items)
    if data.empty:
        raise RuntimeError("No emails were collected. Check network access and config files.")
    data = build_raw_export(data)
    paths["raw_data"].parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(paths["raw_data"], index=False)
    return data


def download_corpora(config_path: Path, download_dir: Path) -> list[dict]:
    download_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads(config_path.read_text())
    rows = []
    for source in sources:
        source_type = source["type"]
        if source_type == "url_archive":
            rows.extend(url_archive_items(source, download_dir))
        elif source_type == "kaggle":
            rows.extend(kaggle_items(source))
        elif source_type == "huggingface":
            rows.extend(huggingface_items(source))
        else:
            raise ValueError(f"Unknown corpus source type: {source_type}")
    return rows


def crawl_live_archives(config_path: Path) -> list[dict]:
    sources = json.loads(config_path.read_text())
    rows = []
    for source in sources:
        kind = source.get("kind", "html_index")
        urls = source["urls"] if "urls" in source else [source["url"]]
        for url in urls:
            if kind == "freebsd_year_index":
                rows.extend(crawl_freebsd_year_index(source, url))
            else:
                rows.extend(crawl_html_index(source, url))
    return rows


def crawl_freebsd_year_index(source: dict, url: str) -> list[dict]:
    text = fetch_text(url)
    archive_pages = [urljoin(url, href) for href in re.findall(r'href=["\']([^"\']+[.]html)["\']', text)]
    archive_pages = [page for page in unique(archive_pages) if re.search(r"[0-9]{8}[.]freebsd-[a-z0-9-]+[.]html$", page)]
    selected_pages = select_evenly(archive_pages, source.get("max_index_pages", len(archive_pages)))
    rows = []
    for page in selected_pages:
        if len(rows) >= source.get("max_items", float("inf")):
            break
        rows.extend(crawl_html_index({**source, "max_items": source.get("max_items", 1000) - len(rows)}, page))
    return rows[: source.get("max_items", len(rows))]


def crawl_html_index(source: dict, url: str) -> list[dict]:
    text = fetch_text(url)
    hrefs = [urljoin(url, href) for href in re.findall(r'href=["\']([^"\']+)["\']', text)]
    links = []
    for link in unique(hrefs):
        if "getmsg.cgi?fetch=0+0+" in link and link.endswith("+archive"):
            links.append(link)
        elif "getmsg.cgi?fetch=" in link:
            links.append(link if link.endswith("+raw") else f"{link}+raw")
        elif re.search(r"/[0-9]+[.]html$", link):
            links.append(link)
    rows = []
    for link in select_evenly(links, source.get("max_items", len(links))):
        content = fetch_bytes(link)
        if link.endswith("+archive") or link.endswith(".mbox") or content.startswith(b"From ") or b"\nFrom " in content[:10000]:
            for name, raw in read_archive_bytes(content, link):
                rows.append(email_item(source, link, f"{link}#{name}", raw))
        elif link.endswith("+raw"):
            rows.append(email_item(source, link, link, content))
        else:
            rows.append(email_item(source, link, link, html_to_email_bytes(content.decode("utf-8", errors="replace"))))
    return rows[: source.get("max_items", len(rows))]


def url_archive_items(source: dict, download_dir: Path) -> list[dict]:
    path = download_to_file(source, download_dir)
    rows = []
    per_label_limit = source.get("max_items_per_label")
    max_items = source.get("max_items")
    label_counts = {}
    for name, raw in read_archive_path(path):
        label = label_for_path(source, name)
        if per_label_limit is not None and label_counts.get(label, 0) >= per_label_limit:
            continue
        if max_items is not None and len(rows) >= max_items:
            break
        label_counts[label] = label_counts.get(label, 0) + 1
        item = email_item(source, source["url"], name, raw, label)
        item["local_path"] = str(path)
        item["extracted_from"] = str(path)
        item["extracted_path"] = name
        rows.append(item)
    return rows


def kaggle_items(source: dict) -> list[dict]:
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(source["dataset"]))
    return tabular_files_items(source, dataset_path)


def huggingface_items(source: dict) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(source["dataset"], split=source.get("split", "train"))
    rows = []
    for index, row in enumerate(dataset):
        if source.get("max_items") is not None and index >= source["max_items"]:
            break
        text = row_value(row, source.get("text_column"), TEXT_COLUMNS)
        label = row_value(row, source.get("label_column"), LABEL_COLUMNS)
        if text:
            item = text_item(source, source["dataset"], f"{source.get('split', 'train')}/{index}", text, label)
            item["local_path"] = source["dataset"]
            item["extracted_from"] = "huggingface"
            item["extracted_path"] = f"{source.get('split', 'train')}/{index}"
            rows.append(item)
    return rows


def tabular_files_items(source: dict, root: Path) -> list[dict]:
    rows = []
    files = list(root.rglob("*.csv")) + list(root.rglob("*.json")) + list(root.rglob("*.jsonl")) + list(root.rglob("*.parquet"))
    for path in files:
        try:
            data = read_table(path)
        except Exception:
            continue
        text_column = existing_column(data, source.get("text_column"), TEXT_COLUMNS)
        label_column = existing_column(data, source.get("label_column"), LABEL_COLUMNS)
        for index, row in data.iterrows():
            if source.get("max_items") is not None and len(rows) >= source["max_items"]:
                return rows
            text = row[text_column] if text_column else row.astype(str).to_dict()
            label = row[label_column] if label_column else source.get("default_label", source.get("label", "ham"))
            item = text_item(source, str(path), f"{path.name}/{index}", text, label)
            item["local_path"] = str(path)
            item["extracted_from"] = str(path)
            item["extracted_path"] = f"{path.name}/{index}"
            rows.append(item)
    return rows


def download_to_file(source: dict, download_dir: Path) -> Path:
    url = source["url"]
    path = download_dir / source["name"] / Path(url.split("?")[0]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    response = requests.get(url, headers=HEADERS, timeout=60, stream=True, verify=source.get("verify_ssl", True))
    response.raise_for_status()
    with path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    return path


def read_archive_path(path: Path):
    if path.suffix == ".zip" or path.read_bytes()[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    yield name, archive.read(name)
        return
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    file_obj = archive.extractfile(member)
                    if file_obj:
                        yield member.name, file_obj.read()
            return
    except tarfile.TarError:
        pass
    yield from read_archive_bytes(path.read_bytes(), str(path))


def read_archive_bytes(content: bytes, url: str):
    if url.endswith(".gz") or content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)
    if content.startswith(b"From ") or b"\nFrom " in content[:10000]:
        for index, chunk in enumerate(re.split(rb"(?m)^From .*$\n", content)):
            raw = chunk.strip()
            if raw:
                yield f"message-{index}.eml", raw
        return
    if url.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(content)) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    yield name, archive.read(name)
        return
    yield Path(url).name, content


def email_item(source: dict, source_url: str, archive_path: str, raw: bytes, label: str | None = None) -> dict:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    body = read_body(message)
    email_id = hashlib.sha256(source_url.encode() + archive_path.encode() + raw).hexdigest()
    return {
        "email_id": email_id,
        "source": source["name"],
        "source_url": source_url,
        "archive_path": archive_path,
        "label": normalize_label(label or source.get("label", "ham")),
        "sender": str(message.get("from", "")),
        "recipient": str(message.get("to", "")),
        "subject": str(message.get("subject", "")),
        "body": body,
    }


def text_item(source: dict, source_url: str, archive_path: str, text, label) -> dict:
    text = str(text or "").strip()
    raw = text.encode("utf-8", errors="ignore")
    email_id = hashlib.sha256(source_url.encode() + archive_path.encode() + raw).hexdigest()
    return {
        "email_id": email_id,
        "source": source["name"],
        "source_url": source_url,
        "archive_path": archive_path,
        "label": normalize_label(label),
        "sender": "",
        "recipient": "",
        "subject": "",
        "body": text,
    }


def build_raw_export(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in RAW_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data["source_family"] = data["source"].map(lambda value: "spamassassin" if str(value).startswith("spamassassin_") else str(value))
    data["label"] = data["label"].map(normalize_label)
    data["text"] = data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    return data.drop_duplicates(subset=["text", "label"])[RAW_COLUMNS]


def dataset_overview(data: pd.DataFrame, raw_data: pd.DataFrame) -> dict[str, int]:
    return {"processed_rows": len(data), "raw_rows": len(raw_data), "processed_columns": len(data.columns), "raw_columns": len(raw_data.columns)}


def label_counts(data: pd.DataFrame) -> pd.DataFrame:
    return data["label"].value_counts().rename_axis("label").reset_index(name="rows")


def source_counts(data: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    return data["source_family"].value_counts().rename_axis("source_family").reset_index(name="rows").head(top_n)


def report_files(metrics_dir: Path) -> list[Path]:
    return [
        metrics_dir / "preprocessing_balance_report.md",
        metrics_dir / "classification_report.txt",
        metrics_dir / "model_summary.md",
        metrics_dir / "per_source_classification_report.csv",
        metrics_dir / "cross_source_holdout_report.csv",
    ]


def report_status(metrics_dir: Path, project_root_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [{"report": str(path.relative_to(project_root_path)), "status": "exists" if path.exists() else "missing"} for path in report_files(metrics_dir)]
    )


def read_classification_report(metrics_dir: Path) -> str:
    path = metrics_dir / "classification_report.txt"
    return path.read_text() if path.exists() else "Saved classification report is missing."


def read_body(message) -> str:
    parts = message.walk() if message.is_multipart() else [message]
    texts = []
    for part in parts:
        if part.get_content_type() == "text/plain":
            texts.append(decode_text_part(part))
    return "\n".join(texts).strip()


def decode_text_part(part) -> str:
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        return decode_bytes(payload, part.get_content_charset() or "")


def decode_bytes(payload: bytes, charset: str) -> str:
    for candidate in [charset, "utf-8", "latin-1"]:
        if not candidate:
            continue
        try:
            lookup(candidate)
            return payload.decode(candidate, errors="replace")
        except LookupError:
            continue
    return payload.decode("utf-8", errors="replace")


def html_to_email_bytes(text: str) -> bytes:
    title = first_match(text, r"<title>(.*?)</title>")
    body = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.I | re.S)
    body = re.sub(r"<[^>]+>", "\n", body)
    body = html.unescape(body)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return f"Subject: {title}\n\n{chr(10).join(lines)}".encode("utf-8", errors="ignore")


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.I | re.S)
    return html.unescape(match.group(1)).strip() if match else ""


def normalize_label(value) -> str:
    text = str(value).strip().lower()
    if text in {"1", "spam", "phishing", "malicious", "bad", "true"} or "spam" in text or "phish" in text:
        return "spam"
    return "ham"


def label_for_path(source: dict, name: str) -> str:
    if source.get("label") != "path":
        return source.get("label", "ham")
    lowered = name.lower()
    if "/spam/" in lowered or lowered.startswith("spam/") or "spam" in Path(lowered).parts:
        return "spam"
    return "ham"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def existing_column(data: pd.DataFrame, preferred, candidates: list[str]) -> str | None:
    if preferred and preferred in data.columns:
        return preferred
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    return None


def row_value(row, preferred, candidates):
    if preferred and preferred in row:
        return row[preferred]
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
    return ""


def fetch_bytes(url: str) -> bytes:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.content


def fetch_text(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", errors="replace")


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def select_evenly(items: list, max_items: int):
    if max_items is None or len(items) <= max_items:
        return items
    if max_items <= 1:
        return items[:max_items]
    last = len(items) - 1
    indexes = sorted({round(index * last / (max_items - 1)) for index in range(max_items)})
    return [items[index] for index in indexes]


def main():
    parser = argparse.ArgumentParser(description="Download/crawl raw email spam data into data/processed/emails_raw.csv.")
    parser.add_argument("--no-live", action="store_true", help="Skip live mailing-list crawling and use configured corpora only.")
    args = parser.parse_args()
    data = crawl_to_raw_csv(include_live_archives=not args.no_live)
    print(f"Saved {len(data)} raw emails to data/processed/emails_raw.csv")


if __name__ == "__main__":
    main()
