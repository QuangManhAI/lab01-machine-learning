# Spam Classifier Project Requirements

Quick notes on what this project is about, inputs/outputs, and step-by-step tasks.

## Purpose
Build a spam email detector. We need to find the best decision threshold on the Validation set so that the False Positive Rate (FPR) is <= 1% (very important to avoid flagging good emails as spam), while keeping the True Positive Rate (TPR) as high as possible on the Test set. We must use scratch models instead of library ones for the final model.

## Input
Raw email dataset containing:
- `email_text`: Email body content.
- `subject`: Email subject line.
- `sender`: Sender's email address.
- `source_family`: Where the email came from.
- `label`: Label of the email (`spam` or `ham`).

## Output
- A fully executed Jupyter notebook `lab01.ipynb` with all outputs, graphs, and results tables.
- Saved best model payload at `models/spam_nb.joblib` (packaged with the scratch model, the TF-IDF vectorizer, decision threshold, and score type).
- Text metrics reports saved under `data/processed/metrics/`.

## Steps to Do

### Setup & Raw Data
- **Step 0**: Import python libraries and append the `src/` folder to `sys.path`.
- **Step 1**: Load raw data and processed data.
- **Step 2**: Plot simple EDA on raw labels and source families.
- **Step 3**: Count missing values and duplicate rows.

### Preprocessing & Splitting
- **Step 4**: Clean text (lowercase, strip HTML, remove punctuation and stop words).
  - **Step 4.1**: Print before/after text of 5 actual samples to check if the cleaning works.
  - **Step 4.2**: Verify that there are no new missing values or duplicates after cleaning.
- **Step 5**: Split data into Train (70%), Val (15%), and Test (15%) stratified by label. Enrich the text by joining `clean_text` with `subject`, `sender`, and `source_family` into a single field `clean_plus_meta`.
- **Step 6**: Downsample ham on the Train set to create a balanced version (1:1), keeping the original unbalanced version.
- **Step 7**: Extract TF-IDF features on the training set.

### Modeling & Evaluation
- **Step 8**: Train three scratch models (Naive Bayes, Logistic Regression, Linear SVM) on both Balanced and Unbalanced train sets.
- **Step 9**: Compare scratch model predictions/probabilities with sklearn equivalents in a single cell to prove correctness.
- **Step 10**: Search validation thresholds to get validation FPR <= 1% and maximize TPR.
- **Step 10.1**: Plot validation ROC curves near target FPR.
- **Step 10.2**: Predict on the Test set using the validation-selected thresholds. Print results table (accuracy, precision, `test_TPR`, `test_FPR`, TN, FP, FN, TP) and plot test confusion matrices.
- **Step 10.2.1**: Write notes comparing balanced and unbalanced training strategies.
- **Step 10.2.2**: Calculate test TPR at exactly/under 1% test FPR constraint.
- **Step 10.3**: Run failure analysis on the best model (inspect nearest errors, check top error-associated tokens, and check source-level errors).

### Save & Deploy
- **Step 11**: Save the best scratch Naive Bayes model payload. Predict on new sample emails to verify it runs in deployment.
