import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "email_spam_lab")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "raw_emails")
