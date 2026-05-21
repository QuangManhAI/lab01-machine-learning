import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "email_spam_lab")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "raw_emails")
MONGO_BATCH_SIZE = int(os.getenv("MONGO_BATCH_SIZE", "1"))
CORPUS_BATCH_SIZE = int(os.getenv("CORPUS_BATCH_SIZE", "1000"))
CRAWL_DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "5"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "60"))
BALANCE_DATASET = os.getenv("BALANCE_DATASET", "true").lower() in {"1", "true", "yes", "y"}
BALANCE_MAX_PER_SOURCE_FAMILY = int(os.getenv("BALANCE_MAX_PER_SOURCE_FAMILY", "1000"))
BALANCE_RANDOM_SEED = int(os.getenv("BALANCE_RANDOM_SEED", "42"))
MIN_CLEAN_WORDS = int(os.getenv("MIN_CLEAN_WORDS", "5"))
MIN_CLEAN_CHARS = int(os.getenv("MIN_CLEAN_CHARS", "25"))
