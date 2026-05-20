from src.config import DB_NAME, MONGO_COLLECTION, MONGO_URI
from src.error_logging import setup_error_logging


setup_error_logging()


BOT_NAME = "email_spam"

SPIDER_MODULES = ["src.email_spam.spiders"]
NEWSPIDER_MODULE = "src.email_spam.spiders"

ROBOTSTXT_OBEY = False
DOWNLOAD_TIMEOUT = 120
DOWNLOAD_DELAY = 5
CONCURRENT_REQUESTS = 1
ITEM_DELAY_SECONDS = 5
LOG_LEVEL = "INFO"

ITEM_PIPELINES = {
    "src.email_spam.pipelines.MongoPipeline": 300,
}

MONGO_DB = DB_NAME
