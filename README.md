# Email Spam Classify

Full flow:

1. Crawl email archives from web addresses in `config/sources.json`.
2. Store raw emails in local MongoDB.
3. Export one merged CSV to `data/processed/emails.csv`.
4. Create EDA plots in `reports/figures`.
5. Train a Naive Bayes spam classifier.

Errors are written to `ERRORS.log`.

Install:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Configure local MongoDB in `.env`:

```bash
MONGO_URI=mongodb://localhost:27017
DB_NAME=email_spam_lab
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
.venv/bin/python -m scrapy crawl archive_email
.venv/bin/python -m src.export_dataset
.venv/bin/python -m src.eda
.venv/bin/python -m src.train
```

Predict:

```bash
.venv/bin/python -m src.predict "win money now"
```
