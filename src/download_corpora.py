import gzip
import json
import logging
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
import urllib3
from pymongo import MongoClient, UpdateOne

from src.config import CORPUS_BATCH_SIZE, DB_NAME, DOWNLOAD_TIMEOUT_SECONDS, MONGO_COLLECTION, MONGO_URI
from src.email_utils import email_item, normalize_label, text_item
from src.error_logging import run_logged

CONFIG = Path("config/corpora_sources.json")
RAW_DIR = Path("data/raw/downloads")
HEADERS = {"User-Agent": "email-spam-lab/1.0"}
TEXT_COLUMNS = ["text", "message", "email", "body", "content", "Text", "Message", "Email Text", "Ticket Description"]
LABEL_COLUMNS = ["label", "Label", "spam", "Spam/Ham", "target", "class", "Category"]
logger = logging.getLogger(__name__)


def main():
    logger.info("Start corpus download/extract from %s", CONFIG)
    sources = json.loads(CONFIG.read_text())
    logger.info("Loaded %s corpus sources", len(sources))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Download folder ready: %s", RAW_DIR)
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][MONGO_COLLECTION]
    collection.create_index("email_id", unique=True)
    logger.info("MongoDB ready: %s.%s corpus_batch_size=%s", DB_NAME, MONGO_COLLECTION, CORPUS_BATCH_SIZE)
    total = 0
    for source in sources:
        try:
            logger.info("Start source: %s (%s)", source["name"], source["type"])
            count = process_source(source, collection)
            total += count
            logger.info("Finished source: %s saved=%s", source["name"], count)
            print(f"{source['name']}: saved {count}")
        except Exception:
            logger.exception("Corpus source failed: %s", source.get("name"))
    client.close()
    logger.info("Finished corpus download/extract total=%s", total)
    print(f"Downloaded/extracted total {total} emails into {DB_NAME}.{MONGO_COLLECTION}")


def process_source(source, collection):
    source_type = source["type"]
    if source_type == "url_archive":
        items = url_archive_items(source)
    elif source_type == "kaggle":
        items = kaggle_items(source)
    elif source_type == "huggingface":
        items = huggingface_items(source)
    else:
        raise ValueError(f"Unknown corpus source type: {source_type}")
    return save_items(collection, source, items)


def url_archive_items(source):
    path = download_to_file(source)
    per_label_limit = source.get("max_items_per_label")
    label_counts = {}
    for name, raw in read_archive_path(path):
        label = label_for_path(source, name)
        if per_label_limit is not None:
            if label_counts.get(label, 0) >= per_label_limit:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
        item = email_item(source, source["url"], name, raw, label)
        item["local_path"] = str(path)
        item["extracted_from"] = str(path)
        item["extracted_path"] = name
        yield item


def download_to_file(source):
    url = source["url"]
    path = RAW_DIR / source["name"] / Path(url.split("?")[0]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Download folder: %s", path.parent)
    if path.exists() and path.stat().st_size:
        logger.info("Use cached download: %s bytes=%s", path, path.stat().st_size)
        return path
    logger.info("Downloading: %s -> %s", url, path)
    downloaded = 0
    verify_ssl = source.get("verify_ssl", True)
    if not verify_ssl:
        logger.warning("SSL verification disabled for source: %s url=%s", source["name"], url)
    with request_download(url, verify_ssl) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded and downloaded % (25 * 1024 * 1024) < len(chunk):
                        logger.info("Downloading progress: %s bytes=%s", source["name"], downloaded)
    logger.info("Downloaded: %s bytes=%s", path, path.stat().st_size)
    return path


def request_download(url, verify_ssl):
    if verify_ssl:
        return requests.get(url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return requests.get(
        url,
        headers=HEADERS,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
        stream=True,
        verify=False,
    )


def read_archive_path(path):
    if is_zip(path):
        logger.info("Extract zip: %s", path)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    logger.info("Extract file: %s", name)
                    yield name, archive.read(name)
        return
    try:
        logger.info("Extract tar archive: %s", path)
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    file_obj = archive.extractfile(member)
                    if file_obj:
                        logger.info("Extract file: %s", member.name)
                        yield member.name, file_obj.read()
            return
    except tarfile.TarError:
        logger.info("Not a tar archive, trying raw/gzip/mbox: %s", path)
        pass
    content = path.read_bytes()
    if path.suffix == ".gz" or content.startswith(b"\x1f\x8b"):
        logger.info("Decompress gzip file: %s", path)
        content = gzip.decompress(content)
    if looks_like_mbox(content):
        logger.info("Read mbox content: %s", path)
        yield from read_mbox(content)
        return
    logger.info("Read single email file: %s", path)
    yield path.name, content


def is_zip(path):
    if path.suffix == ".zip":
        return True
    with path.open("rb") as file:
        return file.read(4) == b"PK\x03\x04"


def looks_like_mbox(content):
    return content.startswith(b"From ") or b"\nFrom " in content[:10000]


def read_mbox(content):
    chunks = re.split(rb"(?m)^From .*$\n", content)
    for index, chunk in enumerate(chunks):
        raw = chunk.strip()
        if raw:
            yield f"message-{index}.eml", raw


def label_for_path(source, name):
    if source.get("label") != "path":
        return source.get("label", "ham")
    lowered = name.lower()
    if "/spam/" in lowered or lowered.startswith("spam/") or "spam" in Path(lowered).parts:
        return "spam"
    return "ham"


def kaggle_items(source):
    import kagglehub

    logger.info("Download Kaggle dataset: %s", source["dataset"])
    dataset_path = Path(kagglehub.dataset_download(source["dataset"]))
    logger.info("Kaggle dataset folder: %s", dataset_path)
    yield from tabular_files_items(source, dataset_path)


def huggingface_items(source):
    from datasets import load_dataset

    logger.info("Load Hugging Face dataset: %s split=%s", source["dataset"], source.get("split", "train"))
    dataset = load_dataset(source["dataset"], split=source.get("split", "train"))
    logger.info("Loaded Hugging Face dataset: %s rows=%s", source["dataset"], len(dataset))
    max_items = source.get("max_items")
    for index, row in enumerate(dataset):
        if max_items is not None and index >= max_items:
            break
        text = row_value(row, source.get("text_column"), TEXT_COLUMNS)
        label = row_value(row, source.get("label_column"), LABEL_COLUMNS)
        if text:
            item = text_item(source, source["dataset"], f"{source.get('split', 'train')}/{index}", text, label)
            item["local_path"] = source["dataset"]
            item["extracted_from"] = "huggingface"
            item["extracted_path"] = f"{source.get('split', 'train')}/{index}"
            yield item


def tabular_files_items(source, root):
    files = list(root.rglob("*.csv")) + list(root.rglob("*.json")) + list(root.rglob("*.jsonl")) + list(root.rglob("*.parquet"))
    logger.info("Scan tabular folder: %s files=%s", root, len(files))
    for path in files:
        try:
            logger.info("Read tabular file: %s", path)
            yield from dataframe_items(source, path, read_table(path))
        except Exception:
            logger.exception("Cannot read tabular file: %s", path)


def read_table(path):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def dataframe_items(source, path, data):
    text_column = existing_column(data, source.get("text_column"), TEXT_COLUMNS)
    label_column = existing_column(data, source.get("label_column"), LABEL_COLUMNS)
    logger.info("Parsed table: %s rows=%s text_column=%s label_column=%s", path, len(data), text_column, label_column)
    max_items = source.get("max_items")
    for index, row in data.iterrows():
        if max_items is not None and index >= max_items:
            break
        text = row[text_column] if text_column else row.astype(str).to_dict()
        label = row[label_column] if label_column else source.get("default_label", source.get("label", "ham"))
        item = text_item(source, str(path), f"{path.name}/{index}", text, label)
        item["local_path"] = str(path)
        item["extracted_from"] = str(path)
        item["extracted_path"] = f"{path.name}/{index}"
        yield item


def existing_column(data, preferred, candidates):
    if preferred and preferred in data.columns:
        return preferred
    lowered = {str(column).lower(): column for column in data.columns}
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def row_value(row, preferred, candidates):
    if preferred and preferred in row:
        return row[preferred]
    lowered = {str(key).lower(): key for key in row.keys()}
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
        if candidate.lower() in lowered:
            return row[lowered[candidate.lower()]]
    return ""


def save_items(collection, source, items):
    saved = 0
    operations = []
    for index, item in enumerate(items):
        if source.get("max_items_per_label") is None and source.get("max_items") is not None and index >= source["max_items"]:
            break
        item["label"] = normalize_label(item["label"])
        operations.append(
            UpdateOne(
                {"email_id": item["email_id"]},
                {"$set": item},
                upsert=True,
            )
        )
        saved += 1
        if saved == 1 or saved % 100 == 0:
            logger.info("Saved emails: source=%s count=%s latest_label=%s", source["name"], saved, item["label"])
        if len(operations) >= CORPUS_BATCH_SIZE:
            flush_operations(collection, source, operations, saved)
            operations = []
    flush_operations(collection, source, operations, saved)
    return saved


def flush_operations(collection, source, operations, saved):
    if not operations:
        return
    result = collection.bulk_write(operations, ordered=False)
    logger.info(
        "Mongo corpus flush: source=%s operations=%s upserted=%s modified=%s seen=%s",
        source["name"],
        len(operations),
        result.upserted_count,
        result.modified_count,
        saved,
    )


if __name__ == "__main__":
    run_logged(main)
