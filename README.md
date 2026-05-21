# Email Spam Classify

Full flow:

1. Download and extract packaged corpora from `config/corpora_sources.json`.
2. Crawl live mailing-list archives from `config/crawler_sources.json`.
3. Store all raw emails in local MongoDB.
4. Export raw, cleaned full, and balanced training CSVs.
5. Create EDA before processing and after strong processing.
6. Train a Naive Bayes spam classifier on cleaned balanced text.

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

Crawler sources use a balanced trial sample by default. Large archive pages are sampled evenly across the whole page instead of taking only the first messages, and the spider writes `data/processed/metrics/crawl_summary.json` when it closes.

The `reports/` folder is image-only. Tables, text reports, and JSON summaries go under `data/processed/metrics/`.

Export outputs:

- `data/processed/emails_raw.csv`: merged raw export before strong processing, used for before-process EDA.
- `data/processed/emails_full.csv`: full cleaned export for audit.
- `data/processed/emails.csv`: cleaned and balanced dataset used by after-process EDA and training.
- `data/processed/metrics/preprocessing_balance_report.md`: cleaning and balance summary.

EDA outputs:

- `reports/figures/before_process/`: raw EDA images before strong processing and balancing.
- `reports/figures/after_process/`: EDA images after strong processing and balancing.
- `data/processed/metrics/before_process/`: raw EDA tables/text summaries.
- `data/processed/metrics/after_process/`: processed EDA tables/text summaries.

Balancing is controlled by `.env`:

```bash
BALANCE_DATASET=true
BALANCE_MAX_PER_SOURCE_FAMILY=1000
BALANCE_RANDOM_SEED=42
MIN_CLEAN_WORDS=5
MIN_CLEAN_CHARS=25
```

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

Optional Hugging Face token for higher rate limits:

```bash
HF_TOKEN=your_huggingface_token
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
.venv/bin/python -m src.eda --input data/processed/emails_raw.csv --figures reports/figures/before_process --metrics data/processed/metrics/before_process --stage before_process --text-column text
.venv/bin/python -m src.eda --input data/processed/emails.csv --figures reports/figures/after_process --metrics data/processed/metrics/after_process --stage after_process --text-column clean_text
.venv/bin/python -m src.train
```

Predict:

```bash
.venv/bin/python -m src.predict "win money now"
```
