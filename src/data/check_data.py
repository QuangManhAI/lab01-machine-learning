from pathlib import Path

from pymongo import MongoClient

from src.config import DB_NAME, MONGO_COLLECTION, MONGO_URI
from src.common.error_logging import run_logged


def main():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client[DB_NAME][MONGO_COLLECTION]
    count = collection.count_documents({})
    latest = collection.find_one({}, {"_id": 0, "source": 1, "label": 1, "subject": 1})
    client.close()

    csv_path = Path("data/processed/emails.csv")
    full_csv_path = Path("data/processed/emails_full.csv")
    raw_csv_path = Path("data/processed/emails_raw.csv")
    print(f"MongoDB: {DB_NAME}.{MONGO_COLLECTION}")
    print(f"Raw emails: {count}")
    print(f"Raw CSV before process: {raw_csv_path if raw_csv_path.exists() else 'not exported yet'}")
    print(f"Balanced CSV: {csv_path if csv_path.exists() else 'not exported yet'}")
    print(f"Full cleaned CSV: {full_csv_path if full_csv_path.exists() else 'not exported yet'}")
    if latest:
        print(f"Latest sample: [{latest.get('label')}] {latest.get('subject', '')}")


if __name__ == "__main__":
    run_logged(main)
