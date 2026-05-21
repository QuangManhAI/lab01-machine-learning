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
- Scrapy live archive crawling uses a fast trial delay by default: 0.5 seconds per request.
- Progress logs should be written to `PIPELINE.log`.
- All errors should be written to `ERRORS.log`.

## Current Files And Roles

- `RULES.md`: project working rules.
- `LAB.md`: teacher requirements.
- `ANALASIS.md`: project analysis.
- `.env`: local MongoDB config.
- `requirements.txt`: Python dependencies, including `wordcloud`.
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
- `src/email_spam/spiders/archive_email.py`: crawls LKML and FreeBSD mailing-list pages.
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
CORPUS_BATCH_SIZE=1000
CRAWL_DELAY_SECONDS=0.5
DOWNLOAD_TIMEOUT_SECONDS=60
```

Optional env: `MONGO_COLLECTION`, `MONGO_BATCH_SIZE`.
Default collection is `raw_emails`.
Default Scrapy Mongo write batch size is `1`, so crawled emails are saved immediately.
Default direct corpus Mongo batch size is `1000`, so downloaded/extracted datasets insert fast.
Default crawler delay is `0.5` seconds per crawled request.

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
.venv/bin/python -m src.validate_crawl
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
  - `reports/figures/source_family_distribution.png`
  - `reports/figures/source_family_label_distribution.png`
  - `reports/figures/top_words_wordcloud.png`
  - `reports/figures/ham_wordcloud.png`
  - `reports/figures/spam_wordcloud.png`
  - `reports/figures/text_length_boxplots.png`
  - `reports/figures/text_shape_scatter.png`
- Model:
  - `models/spam_nb.joblib`
- Metrics:
  - `data/processed/metrics/classification_report.txt`
- Error log:
  - `ERRORS.log`
- Progress log:
  - `PIPELINE.log`

## Current Data Sources

`config/corpora_sources.json` contains direct download/extract sources:

- SpamAssassin public corpus ham archives.
- SpamAssassin public corpus spam archives.
- AUEB Enron-Spam archives: enron1 through enron6, capped at about 1000/source by 500 ham + 500 spam.
- Kaggle spam/email datasets via `kagglehub`.
- Hugging Face spam/email datasets via `datasets`.

`config/crawler_sources.json` contains Scrapy sources:

- LKML weekly HTML archive pages.
- FreeBSD mailing-list weekly archive pages with raw email links.

Direct source formats supported:

- `url_archive`: tar/zip/email archives downloaded by URL.
- `mbox`: raw mbox or gzip mbox responses.
- tabular CSV, JSON, JSONL, parquet from Kaggle downloads.
- Hugging Face dataset rows.

Crawler source formats supported:

- `html_index`: index pages that link to LKML email pages or FreeBSD raw/mbox links.
- `freebsd_year_index`: FreeBSD yearly list index; spider samples weekly archive pages evenly, then reads full weekly mbox archives.

Each source has `max_items` to keep runtime bounded.

Human decision still needed:

- Whether to raise or lower `max_items`.
- Kaggle download may need Kaggle login/token depending on local Kaggle account state and dataset access.
- Direct corpora no longer use per-email sleep. Only Scrapy live crawling uses `CRAWL_DELAY_SECONDS`.
- W3C was removed from crawler config and export filters out old `w3c_` records if they still exist in MongoDB.

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
- `.env` values load correctly.
- Test failure wrote to `ERRORS.log`.
- Progress logging writes to `PIPELINE.log`.
- Logging smoke test wrote info/error to `PIPELINE.log` and only error to `ERRORS.log`.

## Important Notes

- Full real ELT can be long because it includes large corpora and live archive crawling.
- Direct corpus extraction should be fast because it uses local archive parsing and MongoDB bulk writes.
- Scrapy crawling is intentionally polite. The trial profile uses 0.5 seconds per request.
- Crawler config now uses balanced trial sampling. Large archive pages use `max_items=1000` and `sample_strategy=even`, so the crawler samples across the whole archive instead of taking the first 1000 links.
- Current discovered/selected crawl sizes are roughly:
  - `lkml_2022_10_week_1`: 7781 available email pages, 1000 selected.
  - `lkml_2024_02_week_3`: 8438 available email pages, 1000 selected.
  - `freebsd_questions_2025`: 52 weekly pages available, all sampled until about 1000 messages.
  - `freebsd_hackers_2025`: 52 weekly pages available, all sampled until about 1000 messages.
  - `freebsd_current_2025`: 52 weekly pages available, all sampled until about 1000 messages.
  - `freebsd_stable_2025`: 52 weekly pages available, all sampled until about 1000 messages.
  - `freebsd_ports_2025`: 52 weekly pages available, all sampled until about 1000 messages.
- Direct large download sources are capped for trial balance:
  - AUEB Enron: 500 ham + 500 spam per source, enron1 through enron6.
  - Kaggle/Hugging Face large datasets: 1000 rows per source.
- Scrapy writes `data/processed/metrics/crawl_summary.json` when it closes.
- `src.run_pipeline` now validates crawl summary before export/EDA/train, so it does not silently train after a failed or empty crawl.
- During ingestion, data is first saved in MongoDB. CSV appears only after `src.export_dataset` runs.
- Code style follows the project request: short files, simple flow, no unnecessary comments.
- Watch progress with `tail -f PIPELINE.log`.
- Watch failures with `tail -f ERRORS.log`.
- `data/raw/downloads/`, `data/processed/`, `reports/figures/`, models, and log files are ignored by git.
- `reports/` is image-only. Do not put `.csv`, `.txt`, `.md`, or `.json` files under `reports/`.
- `config/sources.json` was removed. Use `config/corpora_sources.json` and `config/crawler_sources.json`.
- AUEB Enron-Spam URLs use `verify_ssl=false` because their HTTPS certificate failed local validation.
- The AUEB `InsecureRequestWarning` is suppressed in code now; the pipeline logs one clear warning instead of printing urllib3 noise.
- Some old emails declare broken charsets like `unknown-8bit` or `DEFAULT_CHARSET`; code now falls back to safe bytes decoding, so these do not go to `ERRORS.log`.
- AUEB archives list `ham/` before `spam/`; using plain `max_items=1000` captured only ham. The config now uses `max_items_per_label=500` for balanced trial data.
- EDA now writes source contribution reports and plots:
  - `data/processed/metrics/source_family_distribution.csv`
  - `data/processed/metrics/source_family_label_distribution.csv`
  - `data/processed/metrics/data_quality_report.md`
  - `data/processed/metrics/ham_eda_report.md`
  - `data/processed/metrics/spam_eda_report.md`
  - `reports/figures/source_family_distribution.png`
  - `reports/figures/source_family_label_distribution.png`
  - `reports/figures/top_words_wordcloud.png`
  - `reports/figures/ham_wordcloud.png`
  - `reports/figures/spam_wordcloud.png`
  - `reports/figures/text_length_boxplots.png`
  - `reports/figures/text_shape_scatter.png`
- SpamAssassin archives are grouped into one EDA source family named `spamassassin`; other sources remain separate.
- EDA separates vocabulary views by label: overall word cloud, ham word cloud, spam word cloud.
- EDA includes `text_shape_scatter.png` to inspect length/word-count outliers and `text_length_boxplots.png` to compare length by label and source family.
- Last regenerated EDA/training after W3C filtering:
  - exported rows: `17967`
  - ham rows: `13110`
  - spam rows: `4857`
  - source family counts:
    - `spamassassin`: `1457`
    - `kaggle_email_spam_classification`: `1000`
    - `lkml_2024_02_week_3`: `1000`
    - `lkml_2022_10_week_1`: `1000`
    - `kaggle_enron_email_dataset`: `1000`
    - `aueb_enron3_spam_ham`: `999`
    - `huggingface_enron_spam`: `996`
    - `aueb_enron2_spam_ham`: `995`
    - `freebsd_hackers_2025`: `995`
    - `aueb_enron1_spam_ham`: `992`
    - `freebsd_questions_2025`: `990`
    - `freebsd_stable_2025`: `990`
    - `freebsd_ports_2025`: `984`
    - `huggingface_kimdongh_spam_dataset`: `981`
    - `aueb_enron4_spam_ham`: `963`
    - `freebsd_current_2025`: `931`
    - `aueb_enron6_spam_ham`: `925`
    - `aueb_enron5_spam_ham`: `769`
- Current model report after W3C filtering:
  - accuracy: `0.94`
  - ham precision/recall/F1: `0.93 / 1.00 / 0.96`
  - spam precision/recall/F1: `0.99 / 0.79 / 0.88`
  - baseline accuracy: `0.7301`
- Important EDA insight from label reports:
  - Spam top terms include HTML/marketing artifacts such as `font`, `br`, `nbsp`, `td`, `free`, `money`, `click`.
  - Ham top terms include technical/community artifacts such as `freebsd`, `patch`, `kernel`, `struct`, plus Enron/business tokens.
  - This means the model may learn both semantic spam cues and formatting/source artifacts; discuss this risk in analysis.
- Training now uses source+label stratified split when possible and writes:
  - `data/processed/metrics/train_test_distribution.csv`
  - `data/processed/metrics/per_source_classification_report.csv`
  - `data/processed/metrics/cross_source_holdout_report.csv`
  - `data/processed/metrics/model_summary.md`
- The headline random split accuracy is not enough for conclusion. Use cross-source holdout to discuss source shift and model generalization.
- CMU Enron maildir was removed from corpus config because it is very large, slow, mostly ham, and bad for quick balanced trial data.
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

6. Export raw, cleaned, and balanced CSV files.

```bash
.venv/bin/python -m src.export_dataset
```

7. Run EDA before processing.

```bash
.venv/bin/python -m src.eda --input data/processed/emails_raw.csv --figures reports/figures/before_process --metrics data/processed/metrics/before_process --stage before_process --text-column text
```

8. Run EDA after strong processing and balancing.

```bash
.venv/bin/python -m src.eda --input data/processed/emails.csv --figures reports/figures/after_process --metrics data/processed/metrics/after_process --stage after_process --text-column clean_text
```

9. Train model.

```bash
.venv/bin/python -m src.train
```

10. Inspect outputs.

- Raw dataset before processing: `data/processed/emails_raw.csv`
- Full cleaned dataset: `data/processed/emails_full.csv`
- Balanced train dataset: `data/processed/emails.csv`
- Before-process EDA figures: `reports/figures/before_process/`
- After-process EDA figures: `reports/figures/after_process/`
- Metrics: `data/processed/metrics/classification_report.txt`
- Model: `models/spam_nb.joblib`

11. If direct corpus ingestion is too slow, raise `CORPUS_BATCH_SIZE`.

12. If live crawling is too slow, decide whether to reintroduce explicit per-source limits or lower `CRAWL_DELAY_SECONDS` in `.env`.

13. If Kaggle fails, configure Kaggle credentials or temporarily remove Kaggle sources from `config/corpora_sources.json`.

## Latest Change: Strong Preprocessing And Dataset Balance

- Added `src/preprocess_balance.py`.
- It handles:
  - HTML tag removal.
  - `<script>`, `<style>`, and `noscript` removal.
  - repeated `html.unescape`.
  - quoted-printable line break cleanup.
  - URL/email/number/hash normalization.
  - punctuation cleanup and lowercase tokenization.
  - stronger stopword removal for English, email headers, HTML artifacts, JavaScript artifacts, and dataset artifacts such as `escapenumber`, `escapelong`, `numbertoken`, `urltoken`, `hextoken`, `www`, `http`, `com`.
  - `source_family` grouping, with all SpamAssassin files grouped as `spamassassin`.
  - trainable row filtering using `MIN_CLEAN_WORDS` and `MIN_CLEAN_CHARS`.
  - label-balanced and source-family-even sampling.
- `.env` now includes:

```env
BALANCE_DATASET=true
BALANCE_MAX_PER_SOURCE_FAMILY=1000
BALANCE_RANDOM_SEED=42
MIN_CLEAN_WORDS=5
MIN_CLEAN_CHARS=25
```

- Export behavior changed:
  - `data/processed/emails_full.csv` is the full cleaned audit export.
  - `data/processed/emails.csv` is the balanced dataset used by EDA and training.
  - `data/processed/metrics/preprocessing_balance_report.md` explains counts before/after cleaning and balancing.
- Current verified export:
  - full cleaned rows: `17582`
  - trainable rows before balance: `17275`
  - balanced rows: `8940`
  - balanced labels: `4470 ham`, `4470 spam`
- Current balance shape:
  - Ham is sampled almost evenly across 18 source families, about `248-249` each.
  - Spam is sampled across 10 spam-capable source families, about `300-547` each depending on available rows.
  - This is intentional because LKML/FreeBSD/Kaggle Enron are ham-only; forcing equal source totals would make the whole dataset ham-heavy again.
- EDA now uses `clean_text` for word clouds and top terms.
- EDA now adds:
  - `reports/figures/raw_vs_clean_length_scatter.png`
  - clean length boxplots/scatter using `clean_char_count` and `clean_word_count`
- Train now uses `clean_text` and a stronger stopword list in `TfidfVectorizer`.
- Predict now cleans input text before calling the saved model.
- `src.check_data` now prints both balanced CSV and full cleaned CSV.
- Verification commands already run successfully:

```bash
.venv/bin/python -m compileall src
.venv/bin/python -m src.export_dataset
.venv/bin/python -m src.eda
.venv/bin/python -m src.train
.venv/bin/python -m src.check_data
```

- Current model after balanced clean-text training:
  - rows: `8940`
  - test rows: `1788`
  - baseline accuracy: `0.4989`
  - accuracy: `0.9793`
  - ham precision/recall/F1: `0.98 / 0.98 / 0.98`
  - spam precision/recall/F1: `0.98 / 0.98 / 0.98`
- Still mention in analysis:
  - Cross-source holdout shows SpamAssassin ham remains hard.
  - Random split accuracy is not enough evidence by itself.
  - Ham still contains strong source/domain language such as Enron, FreeBSD, Linux/kernel terms.

## Latest Change: Pipeline With Before/After EDA

- `src.run_pipeline` now completes the full flow:

```bash
.venv/bin/python -m src.download_corpora
.venv/bin/python -m scrapy crawl archive_email
.venv/bin/python -m src.validate_crawl
.venv/bin/python -m src.export_dataset
.venv/bin/python -m src.eda --input data/processed/emails_raw.csv --figures reports/figures/before_process --metrics data/processed/metrics/before_process --stage before_process --text-column text
.venv/bin/python -m src.eda --input data/processed/emails.csv --figures reports/figures/after_process --metrics data/processed/metrics/after_process --stage after_process --text-column clean_text
.venv/bin/python -m src.train
```

- `src.export_dataset` now writes:
  - `data/processed/emails_raw.csv`: raw merged dataset before strong processing.
  - `data/processed/emails_full.csv`: full cleaned audit dataset.
  - `data/processed/emails.csv`: cleaned and balanced train/EDA dataset.
- EDA now supports CLI args:
  - `--input`
  - `--figures`
  - `--metrics`
  - `--stage`
  - `--text-column`
- Before-process EDA uses raw `text` and basic English stopwords, so dirty artifacts like `font`, `br`, `nbsp`, `http`, `www`, `escapenumber`, and `escapelong` remain visible.
- After-process EDA uses `clean_text` and the stronger stopword/artifact list.
- EDA image folders:
  - `reports/figures/before_process/`
  - `reports/figures/after_process/`
- EDA metric folders:
  - `data/processed/metrics/before_process/`
  - `data/processed/metrics/after_process/`
- Root EDA metric files from the old flow were removed to avoid confusion. Model metrics remain in `data/processed/metrics/`.
- `src.check_data` now prints raw CSV, balanced CSV, and full cleaned CSV.
- Verified commands:

```bash
.venv/bin/python -m compileall src
.venv/bin/python -m src.export_dataset
.venv/bin/python -m src.eda --input data/processed/emails_raw.csv --figures reports/figures/before_process --metrics data/processed/metrics/before_process --stage before_process --text-column text
.venv/bin/python -m src.eda --input data/processed/emails.csv --figures reports/figures/after_process --metrics data/processed/metrics/after_process --stage after_process --text-column clean_text
.venv/bin/python -m src.train
.venv/bin/python -m src.check_data
```

- Current generated counts:
  - raw before-process CSV: `17967`
  - cleaned full CSV: `17582`
  - balanced train/after-process CSV: `8940`
  - balanced labels: `4470 ham`, `4470 spam`
