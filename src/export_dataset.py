from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from src.config import DB_NAME, MONGO_COLLECTION, MONGO_URI
from src.error_logging import run_logged

OUTPUT = Path("data/processed/emails.csv")


def main():
    client = MongoClient(MONGO_URI)
    rows = list(client[DB_NAME][MONGO_COLLECTION].find({}, {"_id": 0}))
    client.close()

    data = pd.DataFrame(rows)
    if data.empty:
        raise SystemExit("No emails found in MongoDB.")

    data["text"] = data[["subject", "body"]].fillna("").agg(" ".join, axis=1)
    data = data.drop_duplicates(subset=["text", "label"])
    data = data[["email_id", "source", "label", "sender", "recipient", "subject", "body", "text"]]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data)} emails to {OUTPUT}")


if __name__ == "__main__":
    run_logged(main)
