import logging
import subprocess
import sys

from src.error_logging import run_logged

logger = logging.getLogger(__name__)


COMMANDS = [
    [sys.executable, "-m", "src.download_corpora"],
    [sys.executable, "-m", "scrapy", "crawl", "archive_email"],
    [sys.executable, "-m", "src.validate_crawl"],
    [sys.executable, "-m", "src.export_dataset"],
    [
        sys.executable,
        "-m",
        "src.eda",
        "--input",
        "data/processed/emails_raw.csv",
        "--figures",
        "reports/figures/before_process",
        "--metrics",
        "data/processed/metrics/before_process",
        "--stage",
        "before_process",
        "--text-column",
        "text",
    ],
    [
        sys.executable,
        "-m",
        "src.eda",
        "--input",
        "data/processed/emails.csv",
        "--figures",
        "reports/figures/after_process",
        "--metrics",
        "data/processed/metrics/after_process",
        "--stage",
        "after_process",
        "--text-column",
        "clean_text",
    ],
    [sys.executable, "-m", "src.train"],
]


def main():
    for command in COMMANDS:
        text = " ".join(command)
        logger.info("Run pipeline command: %s", text)
        print(text)
        subprocess.run(command, check=True)
        logger.info("Finished pipeline command: %s", text)


if __name__ == "__main__":
    run_logged(main)
