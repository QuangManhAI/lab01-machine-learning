# Email Spam Classification Lab

Project code is intentionally kept in `notebooks/`:

- `notebooks/crawl.py`: download/crawl raw email data into `data/processed/emails_raw.csv`.
- `notebooks/preprocess.py`: clean text, balance data, build TF-IDF features, train Naive Bayes, and save/predict with the model.
- `notebooks/model_from_scratch.py`: Naive Bayes from scratch, sklearn double-check, model comparison, save/load, and prediction helpers.
- `notebooks/eda.py`: EDA tables, report metrics, top tokens, per-source scores, and cross-source holdout checks.
- `notebooks/lab01.ipynb`: report notebook that calls the Python files above.

Existing `data/`, `models/`, and `reports/` folders contain generated outputs used by the notebook.

## Install

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Crawl Raw Data

```bash
.venv/bin/python notebooks/crawl.py
```

To skip live mailing-list crawling and use configured corpora only:

```bash
.venv/bin/python notebooks/crawl.py --no-live
```

## Run The Lab

Open and run:

```text
notebooks/lab01.ipynb
```

The notebook loads existing processed CSVs, demonstrates preprocessing, trains Naive Bayes from scratch and with sklearn for double-checking, evaluates the model, and saves/reuses the trained classifier.
