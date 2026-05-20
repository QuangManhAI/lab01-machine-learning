import gzip
import json
import logging
import re
import tarfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import scrapy

from src.email_utils import email_item, html_to_email_bytes


class ArchiveEmailSpider(scrapy.Spider):
    name = "archive_email"
    logger = logging.getLogger(__name__)

    def start_requests(self):
        sources = json.loads(Path("config/crawler_sources.json").read_text())
        self.logger.info("Loaded crawler sources: %s", len(sources))
        for source in sources:
            kind = source.get("kind", "archive")
            callback = self.parse_html_index if kind == "html_index" else self.parse
            self.logger.info("Start crawl source: %s kind=%s url=%s", source["name"], kind, source["url"])
            yield scrapy.Request(source["url"], callback=callback, cb_kwargs={"source": source})

    def parse(self, response, source):
        self.logger.info("Parse archive response: source=%s url=%s bytes=%s", source["name"], response.url, len(response.body))
        for name, raw in self.limit(self.read_archive(response.body, response.url), source):
            time.sleep(self.settings.getfloat("ITEM_DELAY_SECONDS", 5))
            self.logger.info("Crawled email: source=%s path=%s bytes=%s", source["name"], name, len(raw))
            yield email_item(source, response.url, name, raw)

    def parse_html_index(self, response, source):
        self.logger.info("Parse html index: source=%s url=%s", source["name"], response.url)
        raw_links = []
        html_links = []
        for href in response.css("a::attr(href)").getall():
            url = urljoin(response.url, href)
            if "getmsg.cgi?fetch=" in url and "fetch=0+0+" not in url:
                raw_links.append(url if url.endswith("+raw") else f"{url}+raw")
            elif re.search(r"/[0-9]+\.html$", url):
                html_links.append(url)
        links = raw_links or html_links
        self.logger.info("Found email links: source=%s raw=%s html=%s selected=%s", source["name"], len(raw_links), len(html_links), min(len(links), source.get("max_items", len(links))))
        for url in links[: source.get("max_items", len(links))]:
            callback = self.parse_raw_email if raw_links else self.parse_html_email
            self.logger.info("Queue email url: source=%s url=%s", source["name"], url)
            yield scrapy.Request(url, callback=callback, cb_kwargs={"source": source})

    def parse_raw_email(self, response, source):
        time.sleep(self.settings.getfloat("ITEM_DELAY_SECONDS", 5))
        self.logger.info("Parse raw email: source=%s url=%s bytes=%s", source["name"], response.url, len(response.body))
        yield email_item(source, response.url, response.url, response.body)

    def parse_html_email(self, response, source):
        time.sleep(self.settings.getfloat("ITEM_DELAY_SECONDS", 5))
        self.logger.info("Parse html email: source=%s url=%s bytes=%s", source["name"], response.url, len(response.body))
        yield email_item(source, response.url, response.url, html_to_email_bytes(response.text))

    def read_archive(self, content, url):
        if url.endswith(".gz") or content.startswith(b"\x1f\x8b"):
            self.logger.info("Decompress crawler gzip response: %s", url)
            content = gzip.decompress(content)
            url = url.removesuffix(".gz")
        if url.endswith(".mbox") or b"\nFrom " in content[:10000] or content.startswith(b"From "):
            self.logger.info("Read crawler mbox response: %s", url)
            yield from self.read_mbox(content)
            return
        stream = BytesIO(content)
        if url.endswith(".zip"):
            self.logger.info("Extract crawler zip response: %s", url)
            with zipfile.ZipFile(stream) as archive:
                for name in archive.namelist():
                    if not name.endswith("/"):
                        yield name, archive.read(name)
            return
        mode = "r:*"
        self.logger.info("Extract crawler tar response: %s", url)
        with tarfile.open(fileobj=stream, mode=mode) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    file_obj = archive.extractfile(member)
                    if file_obj:
                        yield member.name, file_obj.read()

    def read_mbox(self, content):
        chunks = re.split(rb"(?m)^From .*$\n", content)
        for index, chunk in enumerate(chunks):
            raw = chunk.strip()
            if raw:
                yield f"message-{index}.eml", raw

    def limit(self, items, source):
        max_items = source.get("max_items")
        for index, item in enumerate(items):
            if max_items is not None and index >= max_items:
                break
            yield item
