from pathlib import Path
import logging

import pandas as pd
from pymongo import MongoClient

from src.config import (
    BALANCE_DATASET,
    BALANCE_MAX_PER_SOURCE_FAMILY,
    BALANCE_RANDOM_SEED,
    DB_NAME,
    MIN_CLEAN_CHARS,
    MIN_CLEAN_WORDS,
    MONGO_COLLECTION,
    MONGO_URI,
)
from src.common.error_logging import run_logged
from src.data.preprocess_balance import (
    add_preprocessing_columns,
    balance_dataset,
    filter_trainable_rows,
    source_family,
    write_preprocessing_balance_report,
)

OUTPUT = Path("data/processed/emails.csv")
FULL_OUTPUT = Path("data/processed/emails_full.csv")
RAW_OUTPUT = Path("data/processed/emails_raw.csv")
EXCLUDED_SOURCE_PREFIXES = ("w3c_",)
logger = logging.getLogger(__name__)


def main():
    logger.info("Export start: mongo=%s.%s output=%s", DB_NAME, MONGO_COLLECTION, OUTPUT)
    client = MongoClient(MONGO_URI)
    rows = list(client[DB_NAME][MONGO_COLLECTION].find({}, {"_id": 0}))
    client.close()
    logger.info("Loaded Mongo rows: %s", len(rows))

    data = pd.DataFrame(rows)
    if data.empty:
        raise SystemExit("No emails found in MongoDB.")

    before = len(data)
    data = data[~data["source"].fillna("").str.startswith(EXCLUDED_SOURCE_PREFIXES)]
    logger.debug("Dropped excluded sources: before=%s after=%s prefixes=%s", before, len(data), EXCLUDED_SOURCE_PREFIXES)
    raw_data = build_raw_export(data)
    data = add_preprocessing_columns(data)
    before = len(data)
    data = data.drop_duplicates(subset=["clean_text", "label"])
    logger.debug("Dropped duplicates by clean_text+label: before=%s after=%s", before, len(data))
    trainable_data = filter_trainable_rows(data, MIN_CLEAN_WORDS, MIN_CLEAN_CHARS)
    if BALANCE_DATASET:
        output_data = balance_dataset(
            trainable_data,
            max_per_source_family=BALANCE_MAX_PER_SOURCE_FAMILY,
            random_seed=BALANCE_RANDOM_SEED,
        )
    else:
        output_data = trainable_data.sample(frac=1, random_state=BALANCE_RANDOM_SEED).reset_index(drop=True)
    write_preprocessing_balance_report(data, trainable_data, output_data)

    columns = [
        "email_id",
        "source",
        "source_family",
        "source_url",
        "local_path",
        "extracted_from",
        "extracted_path",
        "archive_path",
        "label",
        "sender",
        "recipient",
        "subject",
        "body",
        "text",
        "clean_text",
        "raw_char_count",
        "clean_char_count",
        "clean_word_count",
    ]
    raw_columns = [
        "email_id",
        "source",
        "source_family",
        "source_url",
        "local_path",
        "extracted_from",
        "extracted_path",
        "archive_path",
        "label",
        "sender",
        "recipient",
        "subject",
        "body",
        "text",
    ]
    for column in raw_columns:
        if column not in raw_data.columns:
            raw_data[column] = ""
    for column in columns:
        if column not in data.columns:
            data[column] = ""
        if column not in output_data.columns:
            output_data[column] = ""
    raw_data = raw_data[raw_columns]
    data = data[columns]
    output_data = output_data[columns]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Output folder ready: %s", OUTPUT.parent)
    raw_data.to_csv(RAW_OUTPUT, index=False)
    data.to_csv(FULL_OUTPUT, index=False)
    output_data.to_csv(OUTPUT, index=False)
    logger.info("Raw export: rows=%s output=%s", len(raw_data), RAW_OUTPUT)
    logger.info("Clean full export: rows=%s output=%s", len(data), FULL_OUTPUT)
    logger.info(
        "Balanced export: rows=%s labels=%s output=%s",
        len(output_data),
        output_data["label"].value_counts().to_dict(),
        OUTPUT,
    )
    print(f"Saved {len(raw_data)} raw emails before processing to {RAW_OUTPUT}")
    print(f"Saved {len(data)} cleaned full emails to {FULL_OUTPUT}")
    print(f"Saved {len(output_data)} balanced emails to {OUTPUT}")


def build_raw_export(data):
    raw_data = data.copy()
    for column in ["subject", "body", "source", "label"]:
        if column not in raw_data.columns:
            raw_data[column] = ""
    raw_data["label"] = raw_data["label"].fillna("").str.lower().str.strip()
    raw_data["source_family"] = raw_data["source"].map(source_family)
    raw_data["text"] = raw_data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    before = len(raw_data)
    raw_data = raw_data.drop_duplicates(subset=["text", "label"])
    logger.debug("Raw export duplicates dropped by text+label: before=%s after=%s", before, len(raw_data))
    return raw_data


if __name__ == "__main__":
    run_logged(main)
