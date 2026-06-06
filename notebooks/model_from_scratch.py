from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

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
        return TfidfVectorizer(
            stop_words=preprocess.stop_words_for_vectorizer(),
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

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

    def split_train_validation_test(
        self,
        data: pd.DataFrame,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_validation, test = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data["label"],
        )
        validation_fraction = validation_size / (1 - test_size)
        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=train_validation["label"],
        )
        return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

    def threshold_experiment_specs(
        self,
        train_pool: pd.DataFrame,
        text_column: str,
        random_state: int = 42,
    ) -> list[dict]:
        strategies = [
            ("Unbalanced", train_pool.reset_index(drop=True), "none"),
            ("Downsample balanced", preprocess.balance_dataset(train_pool, random_seed=random_state), "none"),
            ("Weighted balanced", train_pool.reset_index(drop=True), "weighted"),
        ]
        specs = []
        for strategy_name, train_frame, balance_mode in strategies:
            weighted = balance_mode == "weighted"
            estimators = {
                "Naive Bayes": MultinomialNB(alpha=1.0),
                "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced" if weighted else None),
                "Linear SVM": CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced" if weighted else None),
                    method="sigmoid",
                    cv=3,
                ),
            }
            for model_name, estimator in estimators.items():
                fit_params = {}
                if weighted and model_name == "Naive Bayes":
                    fit_params["model__sample_weight"] = compute_sample_weight("balanced", train_frame["label"])
                specs.append(
                    {
                        "training_strategy": strategy_name,
                        "model": model_name,
                        "text_column": text_column,
                        "train_frame": train_frame,
                        "pipeline": Pipeline([("tfidf", self.make_tfidf_vectorizer()), ("model", clone(estimator))]),
                        "fit_params": fit_params,
                    }
                )
        return specs

    def positive_scores(self, pipeline: Pipeline, texts, positive_label: str = "spam") -> tuple[np.ndarray, str]:
        estimator = pipeline.named_steps["model"]
        if hasattr(pipeline, "predict_proba"):
            class_index = list(estimator.classes_).index(positive_label)
            return pipeline.predict_proba(texts)[:, class_index], "probability"
        return pipeline.decision_function(texts), "decision_score"

    def threshold_summary(
        self,
        labels,
        scores,
        target_fpr: float = 0.01,
        target_tpr: float = 0.99,
        positive_label: str = "spam",
    ) -> dict:
        y_true = (pd.Series(labels).reset_index(drop=True) == positive_label).astype(int)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        survey = pd.DataFrame({"threshold": thresholds, "FPR": fpr, "TPR": tpr})
        survey = survey.replace([np.inf, -np.inf], np.nan).dropna(subset=["threshold"]).copy()
        survey["distance_to_target"] = ((survey["FPR"] - target_fpr) ** 2 + (survey["TPR"] - target_tpr) ** 2) ** 0.5
        under_fpr = survey[survey["FPR"] <= target_fpr]
        if under_fpr.empty:
            best_under_fpr = survey.sort_values(["FPR", "TPR"], ascending=[True, False]).iloc[0]
        else:
            best_under_fpr = under_fpr.sort_values(["TPR", "FPR"], ascending=[False, False]).iloc[0]
        tpr_target_rows = survey[survey["TPR"] >= target_tpr]
        lowest_fpr_at_target_tpr = (
            tpr_target_rows.sort_values(["FPR", "TPR"], ascending=[True, False]).iloc[0] if not tpr_target_rows.empty else None
        )
        closest = survey.sort_values("distance_to_target").iloc[0]
        return {
            "threshold": best_under_fpr["threshold"],
            "validation_fpr_at_threshold": best_under_fpr["FPR"],
            "validation_tpr_at_threshold": best_under_fpr["TPR"],
            "closest_threshold": closest["threshold"],
            "closest_validation_fpr": closest["FPR"],
            "closest_validation_tpr": closest["TPR"],
            "threshold_for_tpr_target": (
                lowest_fpr_at_target_tpr["threshold"] if lowest_fpr_at_target_tpr is not None else np.nan
            ),
            "validation_fpr_at_tpr_target": (
                lowest_fpr_at_target_tpr["FPR"] if lowest_fpr_at_target_tpr is not None else np.nan
            ),
            "validation_tpr_at_tpr_target": (
                lowest_fpr_at_target_tpr["TPR"] if lowest_fpr_at_target_tpr is not None else np.nan
            ),
            "can_reach_target_on_validation": bool(((survey["FPR"] <= target_fpr) & (survey["TPR"] >= target_tpr)).any()),
            "roc_fpr": fpr,
            "roc_tpr": tpr,
            "roc_auc": auc(fpr, tpr),
        }

    def metrics_at_threshold(
        self,
        labels,
        scores,
        threshold: float,
        positive_label: str = "spam",
    ) -> dict:
        y_true = (pd.Series(labels).reset_index(drop=True) == positive_label).astype(int)
        y_pred = (np.asarray(scores) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0
        tnr = tn / (tn + fp) if (tn + fp) else 0
        balanced_accuracy = (recall + tnr) / 2
        assert abs(accuracy - accuracy_score(y_true, y_pred)) < 1e-12
        return {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "TPR": recall,
            "FPR": fpr,
            "FNR": fnr,
            "TNR": tnr,
            "balanced_accuracy": balanced_accuracy,
        }

    def error_analysis_frame(
        self,
        frame: pd.DataFrame,
        scores,
        threshold: float,
        positive_label: str = "spam",
        text_column: str = "clean_text",
    ) -> pd.DataFrame:
        analysis = frame.copy().reset_index(drop=True)
        analysis["score"] = np.asarray(scores)
        analysis["prediction"] = np.where(analysis["score"] >= threshold, positive_label, "ham")
        analysis["is_error"] = analysis["label"] != analysis["prediction"]
        analysis["error_type"] = "correct"
        analysis.loc[(analysis["label"] == "ham") & (analysis["prediction"] == positive_label), "error_type"] = "FP"
        analysis.loc[(analysis["label"] == positive_label) & (analysis["prediction"] == "ham"), "error_type"] = "FN"
        analysis["threshold_distance"] = (analysis["score"] - threshold).abs()
        text_source = text_column if text_column in analysis.columns else "text"
        analysis["snippet"] = analysis[text_source].fillna("").astype(str).str.slice(0, 180)
        return analysis

    def source_label_confounding(self, data: pd.DataFrame) -> pd.DataFrame:
        table = pd.crosstab(data["source_family"], data["label"])
        for label in ["ham", "spam"]:
            if label not in table.columns:
                table[label] = 0
        table = table[["ham", "spam"]]
        table["total"] = table.sum(axis=1)
        table["spam_rate"] = table["spam"] / table["total"]
        table["source_profile"] = np.select(
            [
                (table["ham"] > 0) & (table["spam"] == 0),
                (table["spam"] > 0) & (table["ham"] == 0),
                (table["spam_rate"] <= 0.1),
                (table["spam_rate"] >= 0.9),
            ],
            ["ham-only", "spam-only", "mostly-ham", "mostly-spam"],
            default="mixed",
        )
        return table.reset_index().sort_values(["source_profile", "total"], ascending=[True, False])

    def source_error_summary(self, error_frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for source_family, group in error_frame.groupby("source_family"):
            ham_rows = group[group["label"] == "ham"]
            spam_rows = group[group["label"] == "spam"]
            fp = int((group["error_type"] == "FP").sum())
            fn = int((group["error_type"] == "FN").sum())
            tp = int(((group["label"] == "spam") & (group["prediction"] == "spam")).sum())
            tn = int(((group["label"] == "ham") & (group["prediction"] == "ham")).sum())
            rows.append(
                {
                    "source_family": source_family,
                    "rows": len(group),
                    "ham_rows": len(ham_rows),
                    "spam_rows": len(spam_rows),
                    "FP": fp,
                    "FN": fn,
                    "TP": tp,
                    "TN": tn,
                    "FPR": fp / len(ham_rows) if len(ham_rows) else 0,
                    "FNR": fn / len(spam_rows) if len(spam_rows) else 0,
                    "TPR": tp / len(spam_rows) if len(spam_rows) else 0,
                }
            )
        return pd.DataFrame(rows).sort_values(["FP", "FN", "rows"], ascending=[False, False, False])

    def score_overlap_summary(self, error_frame: pd.DataFrame, threshold: float, window: float = 0.05) -> pd.DataFrame:
        near = error_frame[(error_frame["score"] >= threshold - window) & (error_frame["score"] <= threshold + window)]
        rows = []
        for label, group in error_frame.groupby("label"):
            near_group = near[near["label"] == label]
            rows.append(
                {
                    "label": label,
                    "rows": len(group),
                    "near_threshold_rows": len(near_group),
                    "near_threshold_rate": len(near_group) / len(group) if len(group) else 0,
                    "score_min": group["score"].min(),
                    "score_median": group["score"].median(),
                    "score_max": group["score"].max(),
                }
            )
        return pd.DataFrame(rows)

    def conflicting_clean_texts(self, data: pd.DataFrame, text_column: str = "clean_text", top_n: int = 20) -> pd.DataFrame:
        grouped = (
            data.dropna(subset=[text_column])
            .groupby(text_column)
            .agg(
                rows=("label", "size"),
                labels=("label", lambda values: ",".join(sorted(set(values)))),
                sources=("source_family", lambda values: ",".join(sorted(set(map(str, values)))[:5])),
            )
            .reset_index()
        )
        conflicts = grouped[grouped["labels"].str.contains(",")].copy()
        conflicts["text_preview"] = conflicts[text_column].str.slice(0, 180)
        return conflicts.sort_values("rows", ascending=False).head(top_n)[["rows", "labels", "sources", "text_preview"]]

    def top_error_tokens(
        self,
        error_frame: pd.DataFrame,
        error_type: str,
        text_column: str = "clean_text",
        top_n: int = 20,
    ) -> pd.DataFrame:
        subset = error_frame[error_frame["error_type"] == error_type]
        if subset.empty:
            return pd.DataFrame(columns=["error_type", "token", "count"])
        vectorizer = CountVectorizer(stop_words=preprocess.stop_words_for_vectorizer(), max_features=top_n)
        matrix = vectorizer.fit_transform(subset[text_column].fillna(""))
        counts = matrix.sum(axis=0).A1
        return (
            pd.DataFrame({"error_type": error_type, "token": vectorizer.get_feature_names_out(), "count": counts})
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

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

    def run_threshold_experiment(
        self,
        train_modes: dict[str, pd.DataFrame],
        validation_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        text_column: str,
        target_fpr: float = 0.01,
        target_tpr: float = 0.99,
        positive_label: str = "spam",
    ) -> tuple[pd.DataFrame, list[dict], list[dict]]:
        from sklearn.pipeline import Pipeline
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        
        estimators = {
            "Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Logistic Regression (C=0.5, balanced)": LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced'),
            "Linear SVM (Calibrated)": CalibratedClassifierCV(estimator=LinearSVC(dual='auto'), method="sigmoid", cv=3),
            "Linear SVM (Raw)": LinearSVC(C=1.0, dual='auto', max_iter=2000),
        }
        
        rows = []
        roc_curves = []
        confusion_matrices = []
        specs_out = []
        
        for strategy_name, train_frame in train_modes.items():
            for model_name, estimator in estimators.items():
                pipeline = Pipeline([("tfidf", self.make_tfidf_vectorizer()), ("model", clone(estimator))])
                pipeline.fit(train_frame[text_column], train_frame["label"])
                
                val_scores, score_type = self.positive_scores(pipeline, validation_frame[text_column], positive_label)
                threshold_info = self.threshold_summary(
                    validation_frame["label"],
                    val_scores,
                    target_fpr=target_fpr,
                    target_tpr=target_tpr,
                    positive_label=positive_label,
                )
                test_scores, _ = self.positive_scores(pipeline, test_frame[text_column], positive_label)
                test_metrics = self.metrics_at_threshold(
                    test_frame["label"],
                    test_scores,
                    threshold=threshold_info["threshold"],
                    positive_label=positive_label,
                )
                
                row = {
                    "training_strategy": strategy_name,
                    "model": model_name,
                    "score_type": score_type,
                    "train_rows": len(train_frame),
                    "ham_train": int((train_frame["label"] == "ham").sum()),
                    "spam_train": int((train_frame["label"] == "spam").sum()),
                    "validation_rows": len(validation_frame),
                    "test_rows": len(test_frame),
                    "selected_threshold": threshold_info["threshold"],
                    "validation_FPR": threshold_info["validation_fpr_at_threshold"],
                    "validation_TPR": threshold_info["validation_tpr_at_threshold"],
                    "validation_AUC": threshold_info["roc_auc"],
                    "can_reach_99_1_on_validation": threshold_info["can_reach_target_on_validation"],
                    "closest_threshold": threshold_info["closest_threshold"],
                    "closest_validation_FPR": threshold_info["closest_validation_fpr"],
                    "closest_validation_TPR": threshold_info["closest_validation_tpr"],
                    **test_metrics,
                }
                rows.append(row)
                roc_curves.append({
                    "training_strategy": strategy_name,
                    "model": model_name,
                    "fpr": threshold_info["roc_fpr"],
                    "tpr": threshold_info["roc_tpr"],
                    "auc": threshold_info["roc_auc"],
                    "selected_validation_FPR": threshold_info["validation_fpr_at_threshold"],
                    "selected_validation_TPR": threshold_info["validation_tpr_at_threshold"],
                })
                confusion_matrices.append({
                    "training_strategy": strategy_name,
                    "model": model_name,
                    "matrix": np.array([[test_metrics["TN"], test_metrics["FP"]], [test_metrics["FN"], test_metrics["TP"]]]),
                })
                specs_out.append({
                    "training_strategy": strategy_name,
                    "model": model_name,
                    "text_column": text_column,
                    "train_frame": train_frame,
                    "pipeline": pipeline,
                    "fit_params": {}
                })
        
        results = pd.DataFrame(rows)
        round_columns = [
            "selected_threshold", "validation_FPR", "validation_TPR", "validation_AUC",
            "closest_threshold", "closest_validation_FPR", "closest_validation_TPR",
            "accuracy", "precision", "recall", "TPR", "FPR", "FNR", "TNR", "balanced_accuracy",
        ]
        results[round_columns] = results[round_columns].round(4)
        return results, roc_curves, confusion_matrices, specs_out
