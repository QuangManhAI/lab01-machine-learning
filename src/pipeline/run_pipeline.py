import logging
import subprocess
import sys

from src.common.error_logging import run_logged

logger = logging.getLogger(__name__)


COMMANDS = [
    ("download corpora", [sys.executable, "-m", "src.data.download_corpora"]),
    ("crawl archives", [sys.executable, "-m", "scrapy", "crawl", "archive_email"]),
    ("validate crawl", [sys.executable, "-m", "src.validate_crawl"]),
    ("export/process/balance", [sys.executable, "-m", "src.data.export_dataset"]),
    (
        "EDA before process",
        [
            sys.executable,
            "-m",
            "src.analysis.eda",
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
    ),
    (
        "EDA after process",
        [
            sys.executable,
            "-m",
            "src.analysis.eda",
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
    ),
    ("train model", [sys.executable, "-m", "src.model.train"]),
]


def main():
    for index, (step_name, command) in enumerate(COMMANDS, start=1):
        text = " ".join(command)
        logger.info("Pipeline step %s/%s start: %s", index, len(COMMANDS), step_name)
        logger.debug("Pipeline command: %s", text)
        print(f"[{index}/{len(COMMANDS)}] {step_name}")
        subprocess.run(command, check=True)
        logger.info("Pipeline step %s/%s done: %s", index, len(COMMANDS), step_name)


if __name__ == "__main__":
    run_logged(main)
