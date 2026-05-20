from pymongo import MongoClient, UpdateOne


class MongoPipeline:
    def open_spider(self, spider):
        self.client = MongoClient(spider.settings.get("MONGO_URI"))
        db = self.client[spider.settings.get("MONGO_DB")]
        self.collection = db[spider.settings.get("MONGO_COLLECTION")]
        self.buffer = []

    def close_spider(self, spider):
        self.flush()
        self.collection.create_index("email_id", unique=True)
        self.client.close()

    def process_item(self, item, spider):
        self.buffer.append(
            UpdateOne(
                {"email_id": item["email_id"]},
                {"$set": dict(item)},
                upsert=True,
            )
        )
        if len(self.buffer) >= 500:
            self.flush()
        return item

    def flush(self):
        if self.buffer:
            self.collection.bulk_write(self.buffer, ordered=False)
            self.buffer = []
