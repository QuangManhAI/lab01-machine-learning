# Step By Step

## Rules From Project

- Clarify what AI does and what needs human decision.
- Recommend enough, fit with project.
- Do not guess. Ask when a decision is needed.
- Requirements are in `LAB.md`.
- Analysis is in `ANALASIS.md`.

## Chosen Direction

- Build full download + crawler + MongoDB ELT pipeline.
- Direct download/extract is used for packaged datasets.
- Scrapy is used only for live mailing-list archive pages.
- Raw crawled data goes to local MongoDB.
- Processed output is one CSV file for model training.
- EDA and visualization are required before/after data processing.
- Model is Naive Bayes.
- Direct corpus download/extract should run fast with no per-email sleep.
- Scrapy live archive crawling is slow by default: 5 seconds per request/email.
- Progress logs should be written to `PIPELINE.log`.
- All errors should be written to `ERRORS.log`.

## Current Files And Roles

- `RULES.md`: project working rules.
- `LAB.md`: teacher requirements.
- `ANALASIS.md`: project analysis.
- `.env`: local MongoDB config.
- `requirements.txt`: Python dependencies.
- `scrapy.cfg`: Scrapy project entry.
- `config/corpora_sources.json`: direct download/extract dataset sources.
- `config/crawler_sources.json`: Scrapy mailing-list crawl sources.
- `src/config.py`: loads `.env`.
- `src/error_logging.py`: writes failures to `ERRORS.log`.
  Also writes progress/info to `PIPELINE.log`.
- `src/email_spam/settings.py`: Scrapy settings.
- `src/email_spam/pipelines.py`: saves crawled emails to MongoDB.
- `src/download_corpora.py`: downloads/extracts SpamAssassin, Kaggle, Hugging Face, and Enron corpora.
  Logs download folders, cached files, byte progress, extraction filenames, tabular parsing, and MongoDB save counts.
- `src/email_spam/spiders/archive_email.py`: crawls LKML, FreeBSD, and W3C mailing-list pages.
  Logs source start, index parse, discovered email links, queued email URLs, parsed email pages, and archive extraction.
- `src/email_utils.py`: shared email/text parsing helpers.
- `src/export_dataset.py`: exports MongoDB records to CSV.
- `src/check_data.py`: prints MongoDB count and CSV status.
- `src/eda.py`: creates EDA plots.
- `src/train.py`: trains TF-IDF + Multinomial Naive Bayes model.
- `src/predict.py`: predicts one email text.
- `src/run_pipeline.py`: runs download/extract, crawl, export, EDA, train.
- `README.md`: user run guide.
- `stepByStep.md`: this handoff file.

## Environment Config

`.env` currently contains:

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=email_spam_lab
```

Optional env:

```env
MONGO_COLLECTION=raw_emails
MONGO_BATCH_SIZE=1
CORPUS_BATCH_SIZE=1000
CRAWL_DELAY_SECONDS=5
```

Default collection is `raw_emails`.
Default Scrapy Mongo write batch size is `1`, so crawled emails are saved immediately.
Default direct corpus Mongo batch size is `1000`, so downloaded/extracted datasets insert fast.
Default crawler delay is `5` seconds per crawled request/email.

## Install

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Installed successfully during work.

Important dependency fix:

- `Twisted==24.3.0` is pinned because `Twisted 26.4.0` caused Scrapy `2.11.2` to fail with:

```text
ImportError: cannot import name '_setAcceptableProtocols' from 'twisted.internet._sslverify'
```

Fix command already run:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## MongoDB Requirement

MongoDB must be running before writing to the ELT collection.

Check result:

```bash
which mongod
```

If `mongod` is missing, install MongoDB locally. If MongoDB is installed, start it before the full pipeline.

Start command after MongoDB is installed:

```bash
mkdir -p data/mongo
mongod --dbpath data/mongo
```

## Run Full Pipeline

After MongoDB is running:

```bash
.venv/bin/python -m src.run_pipeline
```

Watch progress in another terminal:

```bash
tail -f PIPELINE.log
```

Watch failures:

```bash
tail -f ERRORS.log
```

This runs:

```bash
.venv/bin/python -m src.download_corpora
.venv/bin/python -m scrapy crawl archive_email
.venv/bin/python -m src.export_dataset
.venv/bin/python -m src.eda
.venv/bin/python -m src.train
```

## Run One Step

Crawl:

```bash
.venv/bin/python -m scrapy crawl archive_email
```

Download/extract corpora:

```bash
.venv/bin/python -m src.download_corpora
```

Export CSV:

```bash
.venv/bin/python -m src.export_dataset
```

Check data:

```bash
.venv/bin/python -m src.check_data
```

EDA:

```bash
.venv/bin/python -m src.eda
```

Train:

```bash
.venv/bin/python -m src.train
```

Predict:

```bash
.venv/bin/python -m src.predict "win money now"
```

## Outputs

- Download cache: `data/raw/downloads/`
- Merged dataset: `data/processed/emails.csv`
- Raw MongoDB collection: `email_spam_lab.raw_emails`
- EDA plots:
  - `reports/figures/label_distribution.png`
  - `reports/figures/text_length_by_label.png`
  - `reports/figures/top_words.png`
- Model:
  - `models/spam_nb.joblib`
- Metrics:
  - `reports/metrics/classification_report.txt`
- Error log:
  - `ERRORS.log`
- Progress log:
  - `PIPELINE.log`

## Current Data Sources

`config/corpora_sources.json` contains direct download/extract sources:

- SpamAssassin public corpus ham archives.
- SpamAssassin public corpus spam archives.
- AUEB Enron-Spam archives.
- CMU Enron Email Dataset archive.
- Kaggle spam/email datasets via `kagglehub`.
- Hugging Face spam/email datasets via `datasets`.

`config/crawler_sources.json` contains Scrapy sources:

- LKML weekly HTML archive pages.
- FreeBSD mailing-list weekly archive pages with raw email links.
- W3C mailing-list archive pages.

Direct source formats supported:

- `url_archive`: tar/zip/email archives downloaded by URL.
- `mbox`: raw mbox or gzip mbox responses.
- tabular CSV, JSON, JSONL, parquet from Kaggle downloads.
- Hugging Face dataset rows.

Crawler source formats supported:

- `html_index`: index pages that link to FreeBSD raw emails, LKML HTML email pages, or W3C HTML email pages.

Each source has `max_items` to keep runtime bounded.

Human decision still needed:

- Whether to raise or lower `max_items`.
- Kaggle download may need Kaggle login/token depending on local Kaggle account state and dataset access.
- Direct corpora no longer use per-email sleep. Only Scrapy live crawling uses `CRAWL_DELAY_SECONDS`.

## Verification Already Done

- Dependencies installed in `.venv`.
- `kagglehub`, `datasets`, and `pyarrow` installed.
- Scrapy HTTP handlers import correctly after pinning `Twisted==24.3.0`.
- `python -m compileall src` passed.
- Scrapy detects spider:

```text
archive_email
```

- EDA and train scripts passed on temporary sample data.
- Archive parser passed on synthetic tar.bz2 email archive.
- FreeBSD raw email link extraction passed.
- LKML HTML index extraction passed.
- Direct SpamAssassin download/extract smoke test passed.
- Hugging Face loader-compatible datasets checked.
- W3C archive link extraction passed.
- `.env` values load correctly.
- Test failure wrote to `ERRORS.log`.
- Progress logging writes to `PIPELINE.log`.
- Logging smoke test wrote info/error to `PIPELINE.log` and only error to `ERRORS.log`.

## Important Notes

- Full real ELT can be long because it includes large corpora and live archive crawling.
- Direct corpus extraction should be fast because it uses local archive parsing and MongoDB bulk writes.
- Scrapy crawling is intentionally slow. The live archive crawl can take a long time because each crawled email waits 5 seconds.
- During ingestion, data is first saved in MongoDB. CSV appears only after `src.export_dataset` runs.
- Code style follows the project request: short files, simple flow, no unnecessary comments.
- Watch progress with `tail -f PIPELINE.log`.
- Watch failures with `tail -f ERRORS.log`.
- `data/raw/downloads/`, `data/processed/`, reports, models, and log files are ignored by git.
- `config/sources.json` was removed. Use `config/corpora_sources.json` and `config/crawler_sources.json`.
- AUEB Enron-Spam URLs use `verify_ssl=false` because their HTTPS certificate failed local validation.
- The AUEB `InsecureRequestWarning` is suppressed in code now; the pipeline logs one clear warning instead of printing urllib3 noise.
- Some old emails declare broken charsets like `unknown-8bit` or `DEFAULT_CHARSET`; code now falls back to safe bytes decoding, so these do not go to `ERRORS.log`.
- CMU Enron archive is large and its host may timeout. `DOWNLOAD_TIMEOUT_SECONDS=60` keeps one dead source from blocking the whole run for too long.
- Kaggle sources can fail until local Kaggle auth is configured.

## Todo After Compact

1. Start MongoDB.

```bash
mkdir -p data/mongo
mongod --dbpath data/mongo
```

2. Watch progress logs.

```bash
tail -f PIPELINE.log
```

3. Download/extract packaged corpora.

```bash
.venv/bin/python -m src.download_corpora
```

4. Crawl live mailing-list archives.

```bash
.venv/bin/python -m scrapy crawl archive_email
```

5. Check MongoDB count.

```bash
.venv/bin/python -m src.check_data
```

6. Export merged CSV.

```bash
.venv/bin/python -m src.export_dataset
```

7. Run EDA.

```bash
.venv/bin/python -m src.eda
```

8. Train model.

```bash
.venv/bin/python -m src.train
```

9. Inspect outputs.

- Dataset: `data/processed/emails.csv`
- Figures: `reports/figures/`
- Metrics: `reports/metrics/classification_report.txt`
- Model: `models/spam_nb.joblib`

10. If direct corpus ingestion is too slow, raise `CORPUS_BATCH_SIZE`.

11. If live crawling is too slow, decide whether to lower `max_items` or lower `CRAWL_DELAY_SECONDS` in `.env`.

12. If Kaggle fails, configure Kaggle credentials or temporarily remove Kaggle sources from `config/corpora_sources.json`.
