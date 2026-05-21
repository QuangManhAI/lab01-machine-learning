# Email Spam Classify

Full flow:

1. Download and extract packaged corpora from `config/corpora_sources.json`.
2. Crawl live mailing-list archives from `config/crawler_sources.json`.
3. Store all raw emails in local MongoDB.
4. Export one merged CSV to `data/processed/emails.csv`.
5. Create EDA plots in `reports/figures`.
6. Train a Naive Bayes spam classifier.

Progress is written to `PIPELINE.log`.
Errors are written to `ERRORS.log`.

Direct download/extract sources include:

- SpamAssassin public corpus spam and ham archives
- Kaggle datasets
- Hugging Face datasets
- AUEB Enron-Spam datasets, enron1 through enron6

Scrapy sources include:

- LKML archive pages
- FreeBSD 2025 yearly mailing-list indexes, sampled across weekly archives
- W3C mailing-list archives across multiple periods

Crawler sources use a balanced trial sample by default. Large archive pages are sampled evenly across the whole page instead of taking only the first messages, and the spider writes `data/processed/metrics/crawl_summary.json` when it closes.

The `reports/` folder is image-only. Tables, text reports, and JSON summaries go under `data/processed/metrics/`.

Install:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Configure local MongoDB in `.env`:

```bash
MONGO_URI=mongodb://localhost:27017
DB_NAME=email_spam_lab
CORPUS_BATCH_SIZE=1000
CRAWL_DELAY_SECONDS=0.5
```

Raw crawler data is stored in MongoDB:

```text
email_spam_lab.raw_emails
```

Check where data is:

```bash
.venv/bin/python -m src.check_data
```

Watch logs:

```bash
tail -f PIPELINE.log
tail -f ERRORS.log
```

Start MongoDB:

```bash
mkdir -p data/mongo
mongod --dbpath data/mongo
```

Run all:

```bash
.venv/bin/python -m src.run_pipeline
```

Run one step:

```bash
.venv/bin/python -m src.download_corpora
.venv/bin/python -m scrapy crawl archive_email
.venv/bin/python -m src.validate_crawl
.venv/bin/python -m src.check_data
.venv/bin/python -m src.export_dataset
.venv/bin/python -m src.eda
.venv/bin/python -m src.train
```

Predict:

```bash
.venv/bin/python -m src.predict "win money now"
```
