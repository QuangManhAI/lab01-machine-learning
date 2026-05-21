import gzip
import json
import logging
import re
import tarfile
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import scrapy

from src.email_utils import email_item, html_to_email_bytes


class ArchiveEmailSpider(scrapy.Spider):
    name = "archive_email"
    logger = logging.getLogger(__name__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_by_source = defaultdict(
            lambda: {
                "index_pages": 0,
                "discovered_urls": 0,
                "queued_urls": 0,
                "scraped_items": 0,
                "available_items": 0,
            }
        )
        self.seen_urls = set()

    def start_requests(self):
        sources = json.loads(Path("config/crawler_sources.json").read_text())
        self.logger.info("Loaded crawler sources: %s", len(sources))
        for source in sources:
            kind = source.get("kind", "archive")
            callback = self.callback_for_kind(kind)
            urls = source["urls"] if "urls" in source else [source["url"]]
            for url in urls:
                self.logger.info("Start crawl source: %s kind=%s url=%s", source["name"], kind, url)
                yield scrapy.Request(url, callback=callback, cb_kwargs={"source": source})

    def callback_for_kind(self, kind):
        if kind == "html_index":
            return self.parse_html_index
        if kind == "freebsd_year_index":
            return self.parse_freebsd_year_index
        return self.parse

    def parse_freebsd_year_index(self, response, source):
        self.stats_by_source[source["name"]]["index_pages"] += 1
        self.logger.debug("Parse FreeBSD year index: source=%s url=%s", source["name"], response.url)
        archive_pages = []
        for href in response.css("a::attr(href)").getall():
            if re.match(r"[0-9]{8}\.freebsd-[a-z0-9-]+\.html$", href):
                archive_pages.append(urljoin(response.url, href))
        archive_pages = self.unique_urls(archive_pages)
        self.stats_by_source[source["name"]]["available_items"] += len(archive_pages)
        selected = self.select_items(archive_pages, {"name": source["name"], "max_items": source.get("max_index_pages"), "sample_strategy": source.get("sample_strategy", "even")})
        self.logger.info("FreeBSD year index pages: source=%s available=%s selected=%s", source["name"], len(archive_pages), len(selected))
        for url in selected:
            yield from self.queue_url(url, self.parse_html_index, source)

    def parse(self, response, source):
        self.logger.debug("Parse archive response: source=%s url=%s bytes=%s", source["name"], response.url, len(response.body))
        items = list(self.read_archive(response.body, response.url))
        self.stats_by_source[source["name"]]["available_items"] += len(items)
        for name, raw in self.select_items(items, source, use_remaining=True):
            if self.source_full(source):
                break
            self.stats_by_source[source["name"]]["scraped_items"] += 1
            self.logger.debug("Crawled email: source=%s path=%s bytes=%s", source["name"], name, len(raw))
            yield email_item(source, response.url, name, raw)

    def parse_html_index(self, response, source):
        self.stats_by_source[source["name"]]["index_pages"] += 1
        self.logger.debug("Parse html index: source=%s url=%s", source["name"], response.url)
        archive_links = []
        raw_links = []
        html_links = []
        for href in response.css("a::attr(href)").getall():
            url = urljoin(response.url, href)
            if self.is_freebsd_archive_url(url):
                archive_links.append(url)
            elif self.is_freebsd_raw_url(url):
                raw_links.append(url if url.endswith("+raw") else f"{url}+raw")
            elif self.is_message_html_url(response.url, url):
                html_links.append(url)
        archive_links = self.unique_urls(archive_links)
        raw_links = self.unique_urls(raw_links)
        html_links = self.unique_urls(html_links)
        self.logger.info(
            "Found crawl links: source=%s archive=%s raw=%s html=%s",
            source["name"],
            len(archive_links),
            len(raw_links),
            len(html_links),
        )
        if archive_links:
            self.stats_by_source[source["name"]]["available_items"] += len(archive_links)
            for url in self.select_items(archive_links, source, use_remaining=False):
                yield from self.queue_url(url, self.parse_mbox_archive, source)
            return
        links = raw_links + html_links
        self.stats_by_source[source["name"]]["available_items"] += len(links)
        selected = self.select_items(links, source, use_remaining=True)
        for url in selected:
            callback = self.parse_raw_email if self.is_freebsd_raw_url(url) else self.parse_html_email
            yield from self.queue_url(url, callback, source)

    def parse_mbox_archive(self, response, source):
        count = 0
        items = list(self.read_mbox(response.body))
        self.stats_by_source[source["name"]]["available_items"] += len(items)
        for name, raw in self.select_items(items, source, use_remaining=True):
            if self.source_full(source):
                break
            count += 1
            self.stats_by_source[source["name"]]["scraped_items"] += 1
            archive_path = f"{response.url}#{name}"
            self.logger.debug("Parse mbox email: source=%s archive=%s path=%s bytes=%s", source["name"], response.url, name, len(raw))
            yield email_item(source, response.url, archive_path, raw)
        self.logger.debug("Parsed mbox archive: source=%s url=%s messages=%s", source["name"], response.url, count)

    def parse_raw_email(self, response, source):
        if self.source_full(source):
            return
        self.stats_by_source[source["name"]]["scraped_items"] += 1
        self.logger.debug("Parse raw email: source=%s url=%s bytes=%s", source["name"], response.url, len(response.body))
        yield email_item(source, response.url, response.url, response.body)

    def parse_html_email(self, response, source):
        if self.source_full(source):
            return
        self.stats_by_source[source["name"]]["scraped_items"] += 1
        self.logger.debug("Parse html email: source=%s url=%s bytes=%s", source["name"], response.url, len(response.body))
        yield email_item(source, response.url, response.url, html_to_email_bytes(response.text))

    def queue_url(self, url, callback, source):
        if url in self.seen_urls:
            return
        if callback.__name__ in {"parse_raw_email", "parse_html_email"} and self.stats_by_source[source["name"]]["queued_urls"] >= source.get("max_items", float("inf")):
            return
        self.seen_urls.add(url)
        self.stats_by_source[source["name"]]["discovered_urls"] += 1
        self.stats_by_source[source["name"]]["queued_urls"] += 1
        self.logger.debug("Queue crawl url: source=%s url=%s", source["name"], url)
        yield scrapy.Request(url, callback=callback, cb_kwargs={"source": source})

    def is_freebsd_archive_url(self, url):
        return "getmsg.cgi?fetch=0+0+" in url and url.endswith("+archive")

    def is_freebsd_raw_url(self, url):
        return "getmsg.cgi?fetch=" in url and "fetch=0+0+" not in url

    def is_message_html_url(self, index_url, url):
        if not re.search(r"/[0-9]+\.html$", url):
            return False
        return url.startswith(index_url.rsplit("/", 1)[0] + "/")

    def unique_urls(self, urls):
        return list(dict.fromkeys(urls))

    def select_items(self, items, source, use_remaining=False):
        max_items = source.get("max_items")
        if use_remaining and max_items is not None:
            max_items = max(max_items - self.stats_by_source[source["name"]]["scraped_items"], 0)
        if max_items is None or len(items) <= max_items:
            return items
        if source.get("sample_strategy", "even") != "even":
            return items[:max_items]
        if max_items <= 1:
            return items[:max_items]
        last_index = len(items) - 1
        selected_indexes = sorted({round(index * last_index / (max_items - 1)) for index in range(max_items)})
        selected = [items[index] for index in selected_indexes]
        self.logger.info(
            "Sample source evenly: source=%s available=%s selected=%s",
            source["name"],
            len(items),
            len(selected),
        )
        return selected

    def source_full(self, source):
        max_items = source.get("max_items")
        return max_items is not None and self.stats_by_source[source["name"]]["scraped_items"] >= max_items

    def read_archive(self, content, url):
        if url.endswith(".gz") or content.startswith(b"\x1f\x8b"):
            self.logger.debug("Decompress crawler gzip response: %s", url)
            content = gzip.decompress(content)
            url = url.removesuffix(".gz")
        if url.endswith(".mbox") or b"\nFrom " in content[:10000] or content.startswith(b"From "):
            self.logger.debug("Read crawler mbox response: %s", url)
            yield from self.read_mbox(content)
            return
        stream = BytesIO(content)
        if url.endswith(".zip"):
            self.logger.debug("Extract crawler zip response: %s", url)
            with zipfile.ZipFile(stream) as archive:
                for name in archive.namelist():
                    if not name.endswith("/"):
                        yield name, archive.read(name)
            return
        mode = "r:*"
        self.logger.debug("Extract crawler tar response: %s", url)
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

    def closed(self, reason):
        output = Path("data/processed/metrics/crawl_summary.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = {"closed_reason": reason, "sources": dict(self.stats_by_source)}
        output.write_text(json.dumps(summary, indent=2, sort_keys=True))
        counts = {
            source: stats.get("scraped_items", 0)
            for source, stats in summary["sources"].items()
        }
        self.logger.info("Crawler summary saved: %s scraped_by_source=%s", output, counts)
