from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
for module_path in [str(NOTEBOOKS_DIR), str(PROJECT_ROOT)]:
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import crawl
import preprocess


DEFAULT_SPAMASSASSIN_MAX_ITEMS = 1000
DEFAULT_ENRON_MAX_ITEMS = 5000


def data_v2_sources(spamassassin_max_items: int, enron_max_items: int) -> list[dict]:
    return [
        {
            "name": "v2_spamassassin_20030228_easy_ham_2",
            "type": "url_archive",
            "label": "ham",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
            "max_items": spamassassin_max_items,
        },
        {
            "name": "v2_kaggle_enron_email_dataset",
            "type": "kaggle",
            "dataset": "wcukierski/enron-email-dataset",
            "default_label": "ham",
            "max_items": enron_max_items,
        },
    ]


def collect_data_v2(
    output_root: Path,
    spamassassin_max_items: int = DEFAULT_SPAMASSASSIN_MAX_ITEMS,
    enron_max_items: int = DEFAULT_ENRON_MAX_ITEMS,
    include_spamassassin: bool = True,
    include_enron: bool = True,
) -> tuple[int, int]:
    download_dir = output_root / "raw" / "downloads"
    processed_dir = output_root / "processed"
    rows = []

    for source in data_v2_sources(spamassassin_max_items, enron_max_items):
        if source["name"].startswith("v2_spamassassin") and not include_spamassassin:
            continue
        if source["name"].startswith("v2_kaggle_enron") and not include_enron:
            continue

        if source["type"] == "url_archive":
            rows.extend(crawl.url_archive_items(source, download_dir))
        elif source["type"] == "kaggle":
            rows.extend(crawl.kaggle_items(source))
        else:
            raise ValueError(f"Unsupported data_v2 source type: {source['type']}")

    if not rows:
        raise RuntimeError("No data_v2 emails were collected. Check selected sources and network credentials.")

    raw_export = crawl.build_raw_export(pd.DataFrame(rows))
    _, full_data, processed = preprocess.export_processed_datasets(
        raw_export,
        output_dir=processed_dir,
        balance=False,
        mixed_sources_only=False,
        extra_ham_samples=0,
    )
    return len(full_data), len(processed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect extra ham-only data into data_v2/processed.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data_v2")
    parser.add_argument("--spamassassin-max-items", type=int, default=DEFAULT_SPAMASSASSIN_MAX_ITEMS)
    parser.add_argument("--enron-max-items", type=int, default=DEFAULT_ENRON_MAX_ITEMS)
    parser.add_argument("--skip-spamassassin", action="store_true", help="Skip SpamAssassin easy_ham_2.")
    parser.add_argument("--skip-enron", action="store_true", help="Skip Kaggle Enron ham.")
    args = parser.parse_args()

    full_rows, processed_rows = collect_data_v2(
        output_root=args.output_root,
        spamassassin_max_items=args.spamassassin_max_items,
        enron_max_items=args.enron_max_items,
        include_spamassassin=not args.skip_spamassassin,
        include_enron=not args.skip_enron,
    )
    print(f"Saved data_v2 full rows: {full_rows:,}")
    print(f"Saved data_v2 processed rows: {processed_rows:,}")
    print(f"Output: {args.output_root / 'processed'}")


if __name__ == "__main__":
    main()
