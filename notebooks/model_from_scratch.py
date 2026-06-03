from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import preprocess


class ScratchMultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, x, y):
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        class_count = []
        feature_log_prob = []
        for label in self.classes_:
            class_rows = y_array == label
            x_class = x[class_rows]
            class_count.append(x_class.shape[0])
            feature_count = np.asarray(x_class.sum(axis=0)).ravel()
            smoothed_feature_count = feature_count + self.alpha
            feature_log_prob.append(np.log(smoothed_feature_count / smoothed_feature_count.sum()))
        class_count = np.asarray(class_count, dtype=float)
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        self.feature_log_prob_ = np.vstack(feature_log_prob)
        return self

    def predict_log_proba(self, x):
        return x @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, x):
        log_scores = self.predict_log_proba(x)
        log_scores = log_scores - log_scores.max(axis=1, keepdims=True)
        scores = np.exp(log_scores)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[np.asarray(self.predict_log_proba(x)).argmax(axis=1)]


class ScratchLogisticRegression:
    def __init__(self, learning_rate: float = 0.8, epochs: int = 120, l2: float = 1e-4, positive_label: str = "spam"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.positive_label = positive_label

    def fit(self, x, y):
        y_binary = self._to_binary(y)
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.classes_ = np.array(["ham", "spam"])

        for _ in range(self.epochs):
            scores = x @ self.weights_ + self.bias_
            probabilities = self._sigmoid(scores)
            error = probabilities - y_binary
            gradient_w = (x.T @ error) / n_samples + self.l2 * self.weights_
            gradient_b = error.mean()
            self.weights_ -= self.learning_rate * np.asarray(gradient_w).ravel()
            self.bias_ -= self.learning_rate * gradient_b
        return self

    def predict_proba(self, x):
        spam_probability = self._sigmoid(x @ self.weights_ + self.bias_)
        spam_probability = np.asarray(spam_probability).ravel()
        return np.column_stack([1 - spam_probability, spam_probability])

    def predict(self, x):
        return np.where(self.predict_proba(x)[:, 1] >= 0.5, "spam", "ham")

    def _to_binary(self, y):
        return (np.asarray(y) == self.positive_label).astype(float)

    def _sigmoid(self, values):
        values = np.clip(values, -35, 35)
        return 1 / (1 + np.exp(-values))


class ScratchLinearSVM:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 120, l2: float = 1e-4, positive_label: str = "spam"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.positive_label = positive_label

    def fit(self, x, y):
        y_signed = np.where(np.asarray(y) == self.positive_label, 1.0, -1.0)
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.classes_ = np.array(["ham", "spam"])

        for _ in range(self.epochs):
            scores = x @ self.weights_ + self.bias_
            margins = y_signed * scores
            active = margins < 1
            if active.any():
                active_count = active.sum()
                gradient_w = self.l2 * self.weights_ - (x[active].T @ y_signed[active]) / active_count
                gradient_b = -y_signed[active].sum() / active_count
            else:
                gradient_w = self.l2 * self.weights_
                gradient_b = 0.0
            self.weights_ -= self.learning_rate * np.asarray(gradient_w).ravel()
            self.bias_ -= self.learning_rate * gradient_b
        return self

    def decision_function(self, x):
        return np.asarray(x @ self.weights_ + self.bias_).ravel()

    def predict(self, x):
        return np.where(self.decision_function(x) >= 0, "spam", "ham")


class SklearnModelChecker:
    def __init__(self):
        self.sklearn_models = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Linear SVM": LinearSVC(),
        }
        self.scratch_models = {
            "Multinomial Naive Bayes": ScratchMultinomialNB(alpha=1.0),
            "Logistic Regression": ScratchLogisticRegression(),
            "Linear SVM": ScratchLinearSVM(),
        }
        self.fitted_models_: dict[str, object] = {}

    def make_tfidf_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(stop_words=preprocess.stop_words_for_vectorizer(), min_df=2, ngram_range=(1, 2))

    def build_model(self) -> Pipeline:
        return Pipeline([("tfidf", self.make_tfidf_vectorizer()), ("nb", MultinomialNB())])

    def tfidf_feature_preview(self, x_train: pd.Series, preview_count: int = 30) -> tuple[tuple[int, int], np.ndarray]:
        vectorizer = self.make_tfidf_vectorizer()
        matrix = vectorizer.fit_transform(x_train)
        return matrix.shape, vectorizer.get_feature_names_out()[:preview_count]

    def fit(self, x_train, y_train):
        self.fitted_models_ = {}
        for name, model in self.sklearn_models.items():
            model.fit(x_train, y_train)
            self.fitted_models_[name] = model
        return self

    def predict(self, model_name: str, x_test):
        return self.fitted_models_[model_name].predict(x_test)

    def evaluate(self, x_test, y_test) -> pd.DataFrame:
        rows = []
        for name, model in self.fitted_models_.items():
            predictions = model.predict(x_test)
            rows.append(
                {
                    "algorithm": name,
                    "implementation": "sklearn",
                    "accuracy": accuracy_score(y_test, predictions),
                    "agreement_with_sklearn": 1.0,
                }
            )
        return pd.DataFrame(rows)

    def compare_scratch_models_with_sklearn(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        sample_rows: int = 5000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        train_frame = pd.DataFrame(
            {"text": pd.Series(x_train).reset_index(drop=True), "label": pd.Series(y_train).reset_index(drop=True)}
        )
        test_frame = pd.DataFrame(
            {"text": pd.Series(x_test).reset_index(drop=True), "label": pd.Series(y_test).reset_index(drop=True)}
        )
        if len(train_frame) > sample_rows:
            train_frame, _ = train_test_split(
                train_frame,
                train_size=sample_rows,
                random_state=random_state,
                stratify=train_frame["label"],
            )
        if len(test_frame) > max(1000, sample_rows // 4):
            test_frame, _ = train_test_split(
                test_frame,
                train_size=max(1000, sample_rows // 4),
                random_state=random_state,
                stratify=test_frame["label"],
            )

        vectorizer = self.make_tfidf_vectorizer()
        x_train_matrix = vectorizer.fit_transform(train_frame["text"])
        x_test_matrix = vectorizer.transform(test_frame["text"])
        y_train_sample = train_frame["label"].to_numpy()
        y_test_sample = test_frame["label"].to_numpy()
        checker = SklearnModelChecker().fit(x_train_matrix, y_train_sample)

        rows = []
        for name, scratch_model in self.scratch_models.items():
            scratch_model.fit(x_train_matrix, y_train_sample)
            scratch_predictions = scratch_model.predict(x_test_matrix)
            sklearn_predictions = checker.predict(name, x_test_matrix)
            rows.append(
                {
                    "algorithm": name,
                    "implementation": "from scratch",
                    "accuracy": accuracy_score(y_test_sample, scratch_predictions),
                    "agreement_with_sklearn": (scratch_predictions == sklearn_predictions).mean(),
                }
            )
            rows.append(
                {
                    "algorithm": name,
                    "implementation": "sklearn checker",
                    "accuracy": accuracy_score(y_test_sample, sklearn_predictions),
                    "agreement_with_sklearn": 1.0,
                }
            )
        return pd.DataFrame(rows)

    def train_project_model(self, x_train, y_train, x_test, y_test) -> tuple[Pipeline, np.ndarray, float, float]:
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(x_train, y_train)
        baseline_accuracy = accuracy_score(y_test, baseline.predict(x_test))
        model = self.build_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return model, predictions, baseline_accuracy, accuracy_score(y_test, predictions)

    def compare_models(self, data: pd.DataFrame, text_column: str, sample_rows: int = 50_000) -> pd.DataFrame:
        comparison_data = data.sample(min(len(data), sample_rows), random_state=42)
        comparison_train, comparison_test = train_test_split(
            comparison_data, test_size=0.2, random_state=42, stratify=comparison_data["label"]
        )
        comparison_models = {
            "Naive Bayes": Pipeline([("tfidf", self.make_tfidf_vectorizer()), ("model", MultinomialNB())]),
            "Logistic Regression": Pipeline(
                [("tfidf", self.make_tfidf_vectorizer()), ("model", LogisticRegression(max_iter=1000))]
            ),
            "Linear SVM": Pipeline([("tfidf", self.make_tfidf_vectorizer()), ("model", LinearSVC())]),
        }
        rows = []
        for name, candidate in comparison_models.items():
            candidate.fit(comparison_train[text_column], comparison_train["label"])
            predictions = candidate.predict(comparison_test[text_column])
            precision, recall, f1, _ = precision_recall_fscore_support(
                comparison_test["label"], predictions, average="macro", zero_division=0
            )
            rows.append(
                {
                    "model": name,
                    "accuracy": accuracy_score(comparison_test["label"], predictions),
                    "macro_precision": precision,
                    "macro_recall": recall,
                    "macro_f1": f1,
                }
            )
        return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)

    def evaluate_cross_source_holdout(self, data: pd.DataFrame, text_column: str) -> pd.DataFrame:
        rows = []
        for source, holdout in data.groupby("source"):
            if len(holdout) < 50:
                continue
            train = data[data["source"] != source]
            if train["label"].nunique() < 2:
                continue
            model = self.build_model()
            model.fit(train[text_column], train["label"])
            predictions = model.predict(holdout[text_column])
            report = classification_report(holdout["label"], predictions, output_dict=True, zero_division=0)
            rows.append(
                {
                    "holdout_source": source,
                    "rows": len(holdout),
                    "labels": ",".join(sorted(holdout["label"].unique())),
                    "accuracy": accuracy_score(holdout["label"], predictions),
                    "macro_f1": report["macro avg"]["f1-score"],
                }
            )
        return pd.DataFrame(rows).sort_values("macro_f1")

    def save_model(self, model: Pipeline, model_path: Path) -> Path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return model_path

    def predict_new_emails(self, model_path: Path, emails: list[str] | None = None) -> pd.DataFrame:
        emails = emails or [
            "Win cash now! Click the prize link and claim your bonus today.",
            "Hi team, attached are the meeting notes and next steps from today's planning call.",
        ]
        deployed_model = joblib.load(model_path)
        email_series = pd.Series(emails, name="email_text")
        clean_new_emails = email_series.map(preprocess.clean_email_text)
        predictions = deployed_model.predict(clean_new_emails)
        table = pd.DataFrame({"email_text": email_series, "clean_text": clean_new_emails, "prediction": predictions})
        if hasattr(deployed_model, "predict_proba"):
            table["confidence"] = deployed_model.predict_proba(clean_new_emails).max(axis=1).round(4)
        return table
