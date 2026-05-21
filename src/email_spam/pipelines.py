import logging

from pymongo import MongoClient, UpdateOne

logger = logging.getLogger(__name__)


class MongoPipeline:
    def open_spider(self, spider):
        self.client = MongoClient(spider.settings.get("MONGO_URI"))
        db = self.client[spider.settings.get("MONGO_DB")]
        self.collection = db[spider.settings.get("MONGO_COLLECTION")]
        self.batch_size = spider.settings.getint("MONGO_BATCH_SIZE", 1)
        self.buffer = []
        self.saved = 0
        logger.info("Mongo pipeline open: db=%s collection=%s batch_size=%s", spider.settings.get("MONGO_DB"), spider.settings.get("MONGO_COLLECTION"), self.batch_size)

    def close_spider(self, spider):
        self.flush()
        self.collection.create_index("email_id", unique=True)
        self.client.close()
        logger.info("Mongo pipeline closed: saved=%s", self.saved)

    def process_item(self, item, spider):
        self.buffer.append(
            UpdateOne(
                {"email_id": item["email_id"]},
                {"$set": dict(item)},
                upsert=True,
            )
        )
        if len(self.buffer) >= self.batch_size:
            self.flush()
        return item

    def flush(self):
        if self.buffer:
            result = self.collection.bulk_write(self.buffer, ordered=False)
            self.saved += result.upserted_count + result.modified_count
            logger.debug("Mongo flush: operations=%s upserted=%s modified=%s total_saved=%s", len(self.buffer), result.upserted_count, result.modified_count, self.saved)
            self.buffer = []
