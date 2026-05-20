from pathlib import Path
import logging

import pandas as pd
from pymongo import MongoClient

from src.config import DB_NAME, MONGO_COLLECTION, MONGO_URI
from src.error_logging import run_logged

OUTPUT = Path("data/processed/emails.csv")
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

    data["text"] = data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    before = len(data)
    data = data.drop_duplicates(subset=["text", "label"])
    logger.info("Dropped duplicates: before=%s after=%s", before, len(data))
    columns = [
        "email_id",
        "source",
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
    for column in columns:
        if column not in data.columns:
            data[column] = ""
    data = data[columns]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Output folder ready: %s", OUTPUT.parent)
    data.to_csv(OUTPUT, index=False)
    logger.info("Export finished: rows=%s output=%s", len(data), OUTPUT)
    print(f"Saved {len(data)} emails to {OUTPUT}")


if __name__ == "__main__":
    run_logged(main)
