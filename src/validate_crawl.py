import json
import logging
from pathlib import Path

from src.common.error_logging import run_logged


SUMMARY = Path("data/processed/metrics/crawl_summary.json")
logger = logging.getLogger(__name__)


def main():
    if not SUMMARY.exists():
        raise SystemExit(f"Missing crawl summary: {SUMMARY}")
    summary = json.loads(SUMMARY.read_text())
    failed = []
    for source, stats in summary.get("sources", {}).items():
        scraped = stats.get("scraped_items", 0)
        queued = stats.get("queued_urls", 0)
        if queued == 0 or scraped == 0:
            failed.append(f"{source}: queued={queued} scraped={scraped}")
    if failed:
        message = "Crawler did not collect emails for these sources:\n" + "\n".join(failed)
        logger.error(message)
        raise SystemExit(message)
    counts = {
        source: stats.get("scraped_items", 0)
        for source, stats in summary.get("sources", {}).items()
    }
    logger.info("Crawl validation passed: sources=%s scraped_by_source=%s", len(counts), counts)
    print(f"Crawl validation passed for {len(summary.get('sources', {}))} sources")


if __name__ == "__main__":
    run_logged(main)
