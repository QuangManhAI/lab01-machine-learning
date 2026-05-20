import subprocess
import sys

from src.error_logging import run_logged


COMMANDS = [
    [sys.executable, "-m", "scrapy", "crawl", "archive_email"],
    [sys.executable, "-m", "src.export_dataset"],
    [sys.executable, "-m", "src.eda"],
    [sys.executable, "-m", "src.train"],
]


def main():
    for command in COMMANDS:
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    run_logged(main)
