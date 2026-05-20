import hashlib
import json
import logging
import tarfile
import time
import zipfile
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path

import scrapy


class ArchiveEmailSpider(scrapy.Spider):
    name = "archive_email"
    logger = logging.getLogger(__name__)

    def start_requests(self):
        sources = json.loads(Path("config/sources.json").read_text())
        for source in sources:
            yield scrapy.Request(source["url"], cb_kwargs={"source": source})

    def parse(self, response, source):
        for name, raw in self.read_archive(response.body, response.url):
            time.sleep(self.settings.getfloat("ITEM_DELAY_SECONDS", 5))
            message = BytesParser(policy=policy.default).parsebytes(raw)
            body = self.read_body(message)
            email_id = hashlib.sha256(source["url"].encode() + name.encode() + raw).hexdigest()
            yield {
                "email_id": email_id,
                "source": source["name"],
                "source_url": source["url"],
                "archive_path": name,
                "label": source["label"],
                "sender": str(message.get("from", "")),
                "recipient": str(message.get("to", "")),
                "subject": str(message.get("subject", "")),
                "date": str(message.get("date", "")),
                "body": body,
            }

    def read_archive(self, content, url):
        stream = BytesIO(content)
        if url.endswith(".zip"):
            with zipfile.ZipFile(stream) as archive:
                for name in archive.namelist():
                    if not name.endswith("/"):
                        yield name, archive.read(name)
            return
        mode = "r:*"
        with tarfile.open(fileobj=stream, mode=mode) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    file_obj = archive.extractfile(member)
                    if file_obj:
                        yield member.name, file_obj.read()

    def read_body(self, message):
        parts = message.walk() if message.is_multipart() else [message]
        texts = []
        for part in parts:
            if part.get_content_type() == "text/plain":
                try:
                    texts.append(part.get_content())
                except Exception:
                    self.logger.exception("Cannot decode email part")
                    payload = part.get_payload(decode=True) or b""
                    texts.append(payload.decode("utf-8", errors="ignore"))
        return "\n".join(texts).strip()
