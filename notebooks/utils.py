from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import crawl
import eda
import model_from_scratch
import preprocess


@dataclass
class SplitContext:
    text_column: str
    comparison_base_data: pd.DataFrame
    base_train_data: pd.DataFrame
    base_test_data: pd.DataFrame
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    x_train: pd.Series
    y_train: pd.Series
    x_test: pd.Series
    y_test: pd.Series


@dataclass
class ScratchContext:
    x_train_matrix: object
    x_test_matrix: object
    y_train: np.ndarray
    y_test: np.ndarray
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    vectorizer: object


FEATURE_QUALITY_COLUMNS = [
    "label",
    "source_family",
    "subject",
    "text",
    "clean_text",
    "raw_char_count",
    "clean_char_count",
    "clean_word_count",
]


def feature_quality_summary(frame: pd.DataFrame, name: str) -> None:
    available_columns = [column for column in FEATURE_QUALITY_COLUMNS if column in frame.columns]
    print(f"{name}: missing data in model/features columns")
    display(preprocess.missing_data_summary(frame[available_columns]))
    print(f"{name}: duplicate clean_text + label")
    display(preprocess.duplicate_data_summary(frame, subset=["clean_text", "label"]))


def add_threshold_metadata_text(frame: pd.DataFrame, base_text_column: str = "clean_text") -> pd.DataFrame:
    frame = frame.copy()
    index = frame.index
    base_text = frame[base_text_column].fillna("").astype(str)
    subject_text = frame.get("subject", pd.Series("", index=index)).fillna("").map(preprocess.clean_email_text)
    sender_text = (
        frame.get("sender", pd.Series("", index=index))
        .fillna("")
        .astype(str)
        .str.replace(r"[^a-zA-Z0-9@._-]+", " ", regex=True)
        .str.lower()
    )
    source_family_token = _metadata_token(frame.get("source_family", pd.Series("", index=index)), "sourcefamily")
    source_token = _metadata_token(frame.get("source", pd.Series("", index=index)), "source")
    frame["clean_plus_meta"] = (
        base_text
        + " subject "
        + subject_text
        + " sender "
        + sender_text
        + " "
        + source_family_token
        + " "
        + source_token
    )
    return frame


def run_preprocess_step(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_clean_data, before_balance_data, unbalanced_processed_data = preprocess.process_raw_dataset(
        raw_data,
        balance=False,
        mixed_sources_only=True,
        extra_ham_samples=0,
    )

    print(f"Full cleaned rows: {len(full_clean_data):,}")
    print(f"Trainable rows after preprocess, before balance: {len(before_balance_data):,}")
    print("Unbalanced label counts")
    display(crawl.label_counts(before_balance_data))

    feature_quality_summary(before_balance_data, "After preprocess, before balance")
    display(eda.source_label_table(before_balance_data).head(20))
    display(eda.length_summary(before_balance_data))
    return full_clean_data, before_balance_data, unbalanced_processed_data


def load_and_show_data_v2(project_root: Path) -> pd.DataFrame:
    data_v2_processed = project_root / "data_v2" / "processed" / "emails.csv"
    if not data_v2_processed.exists():
        print(f"Data v2 not found yet: {data_v2_processed}")
        print("Run `.venv/bin/python notebooks/crawl_data_v2.py` first, then rerun this cell.")
        return pd.DataFrame()

    data_v2 = pd.read_csv(data_v2_processed).fillna("")
    if "clean_text" not in data_v2.columns:
        data_v2 = preprocess.add_preprocessing_columns(data_v2)
    data_v2 = preprocess.filter_trainable_rows(data_v2).copy().reset_index(drop=True)

    print(f"Data v2 rows after preprocess filter: {len(data_v2):,}")
    display(crawl.label_counts(data_v2))
    display(crawl.source_counts(data_v2, top_n=10))
    display(eda.source_label_table(data_v2).head(10))
    display(eda.length_summary(data_v2))
    eda.plot_eda_overview(data_v2, title_prefix="Data v2 extra ham")
    return data_v2


def prepare_fixed_split(processed_data: pd.DataFrame, random_seed: int = 42) -> SplitContext:
    comparison_base_data = processed_data.copy().reset_index(drop=True)
    split = preprocess.split_training_data(comparison_base_data)
    text_column = split["text_column"]
    base_train_data = split["train_data"].copy().reset_index(drop=True)
    base_test_data = split["test_data"].copy().reset_index(drop=True)

    train_data = preprocess.balance_dataset(base_train_data, random_seed=random_seed).reset_index(drop=True)
    test_data = base_test_data

    context = SplitContext(
        text_column=text_column,
        comparison_base_data=comparison_base_data,
        base_train_data=base_train_data,
        base_test_data=base_test_data,
        train_data=train_data,
        test_data=test_data,
        x_train=train_data[text_column],
        y_train=train_data["label"],
        x_test=test_data[text_column],
        y_test=test_data["label"],
    )

    split_summary = pd.DataFrame(
        [
            _label_count_row("processed_base_train", base_train_data),
            _label_count_row("balanced_train_for_baseline", train_data),
            _label_count_row("fixed_processed_test", test_data),
        ]
    )

    print(f"Text column: {text_column}")
    display(split_summary)
    display(preprocess.train_source_crosstab(base_train_data, top_n=15))
    return context


def run_three_data_model_comparison(
    split_context: SplitContext,
    data_v2: pd.DataFrame,
    model_checker: model_from_scratch.SklearnModelChecker,
    random_seed: int = 42,
) -> dict:
    base_train_data = split_context.base_train_data
    base_test_data = split_context.base_test_data
    text_column = split_context.text_column

    if data_v2.empty:
        data_v2_train_only = pd.DataFrame(columns=base_train_data.columns)
    else:
        test_clean_texts = set(base_test_data[text_column].fillna(""))
        data_v2_train_only = data_v2[~data_v2[text_column].isin(test_clean_texts)].copy()

    v2_merged_train = (
        pd.concat([base_train_data, data_v2_train_only], ignore_index=True)
        .drop_duplicates(subset=[text_column, "label"])
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )
    train_modes = {
        "Balanced": preprocess.balance_dataset(base_train_data, random_seed=random_seed).reset_index(drop=True),
        "Unbalanced": base_train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True),
        "V2 merged": v2_merged_train,
    }

    version_summary = pd.DataFrame(
        [
            _dataset_summary("processed_base_train before mode transform", base_train_data),
            _dataset_summary("data_v2 extra ham added to train only", data_v2_train_only),
            _dataset_summary("V2 merged train", v2_merged_train),
            _dataset_summary("fixed_processed_test", base_test_data),
        ]
    )
    training_summary = pd.DataFrame(
        [
            {
                "training_mode": mode_name,
                "train_rows": len(train_frame),
                "ham_train": int((train_frame["label"] == "ham").sum()),
                "spam_train": int((train_frame["label"] == "spam").sum()),
                "source_families": train_frame["source_family"].nunique(),
            }
            for mode_name, train_frame in train_modes.items()
        ]
    )

    display(version_summary)
    display(training_summary)
    if not v2_merged_train.empty:
        display(pd.crosstab(v2_merged_train["source_family"], v2_merged_train["label"]).sort_index())

    results, matrices = _fit_three_by_three_models(train_modes, base_test_data, text_column, model_checker)
    metric_columns = ["accuracy", "precision", "recall", "TPR", "FPR", "FNR", "TNR"]
    results[metric_columns] = results[metric_columns].round(4)
    display(results.sort_values(["training_mode", "model"]))
    _plot_confusion_matrix_grid(matrices, "training_mode", "3 training datasets x 3 models, fixed processed test")

    return {
        "data_v2_train_only": data_v2_train_only,
        "v2_merged_train": v2_merged_train,
        "train_modes": train_modes,
        "version_summary": version_summary,
        "training_summary": training_summary,
        "results": results,
        "matrices": matrices,
    }


def prepare_scratch_context(
    split_context: SplitContext,
    model_checker: model_from_scratch.SklearnModelChecker,
    model_sample_rows: int = 3_000,
    check_sample_rows: int = 1_000,
    random_seed: int = 42,
) -> ScratchContext:
    model_train_frame = pd.DataFrame(
        {
            "text": pd.Series(split_context.x_train).reset_index(drop=True),
            "label": pd.Series(split_context.y_train).reset_index(drop=True),
        }
    )
    model_test_frame = pd.DataFrame(
        {
            "text": pd.Series(split_context.x_test).reset_index(drop=True),
            "label": pd.Series(split_context.y_test).reset_index(drop=True),
        }
    )

    if len(model_train_frame) > model_sample_rows:
        model_train_frame, _ = train_test_split(
            model_train_frame,
            train_size=model_sample_rows,
            random_state=random_seed,
            stratify=model_train_frame["label"],
        )
    if len(model_test_frame) > check_sample_rows:
        model_test_frame, _ = train_test_split(
            model_test_frame,
            train_size=check_sample_rows,
            random_state=random_seed,
            stratify=model_test_frame["label"],
        )

    scratch_vectorizer = model_checker.make_tfidf_vectorizer()
    x_train_matrix = scratch_vectorizer.fit_transform(model_train_frame["text"])
    x_test_matrix = scratch_vectorizer.transform(model_test_frame["text"])
    y_train = model_train_frame["label"].to_numpy()
    y_test = model_test_frame["label"].to_numpy()

    print(f"Scratch/check train matrix: {x_train_matrix.shape}")
    print(f"Scratch/check test matrix: {x_test_matrix.shape}")
    return ScratchContext(
        x_train_matrix=x_train_matrix,
        x_test_matrix=x_test_matrix,
        y_train=y_train,
        y_test=y_test,
        train_frame=model_train_frame,
        test_frame=model_test_frame,
        vectorizer=scratch_vectorizer,
    )


def train_scratch_models(scratch_context: ScratchContext) -> dict:
    scratch_models = {
        "Multinomial Naive Bayes": model_from_scratch.ScratchMultinomialNB(alpha=1.0),
        "Logistic Regression": model_from_scratch.ScratchLogisticRegression(),
        "Linear SVM": model_from_scratch.ScratchLinearSVM(),
    }

    predictions_by_model = {}
    for model_name, scratch_model in scratch_models.items():
        scratch_model.fit(scratch_context.x_train_matrix, scratch_context.y_train)
        predictions_by_model[model_name] = scratch_model.predict(scratch_context.x_test_matrix)

    scratch_summary = pd.DataFrame(
        [
            {
                "algorithm": model_name,
                "class_called_in_notebook": scratch_model.__class__.__name__,
                "scratch_accuracy": accuracy_score(scratch_context.y_test, predictions_by_model[model_name]),
            }
            for model_name, scratch_model in scratch_models.items()
        ]
    )
    display(scratch_summary)
    _plot_scratch_confusion_matrices(scratch_context.y_test, predictions_by_model)
    return {
        "models": scratch_models,
        "predictions_by_model": predictions_by_model,
        "summary": scratch_summary,
    }


def compare_scratch_with_sklearn(scratch_context: ScratchContext, scratch_results: dict) -> pd.DataFrame:
    sklearn_checker = model_from_scratch.SklearnModelChecker().fit(
        scratch_context.x_train_matrix,
        scratch_context.y_train,
    )

    check_rows = []
    for model_name, scratch_predictions in scratch_results["predictions_by_model"].items():
        sklearn_predictions = sklearn_checker.predict(model_name, scratch_context.x_test_matrix)
        scratch_accuracy = accuracy_score(scratch_context.y_test, scratch_predictions)
        sklearn_accuracy = accuracy_score(scratch_context.y_test, sklearn_predictions)
        mismatch_count = int((scratch_predictions != sklearn_predictions).sum())

        check_rows.append(
            {
                "algorithm": model_name,
                "scratch_accuracy": scratch_accuracy,
                "sklearn_accuracy": sklearn_accuracy,
                "accuracy_diff_abs": abs(scratch_accuracy - sklearn_accuracy),
                "prediction_disagreement_rate": mismatch_count / len(scratch_context.y_test),
                "mismatch_count": mismatch_count,
                "checked_rows": len(scratch_context.y_test),
            }
        )

    model_check_differences = pd.DataFrame(check_rows)
    display(model_check_differences)
    return model_check_differences


def run_threshold_tuning(
    processed_data: pd.DataFrame,
    data_v2: pd.DataFrame,
    model_checker: model_from_scratch.SklearnModelChecker,
    target_fpr: float = 0.01,
    target_tpr: float = 0.99,
    positive_label: str = "spam",
    negative_label: str = "ham",
    random_seed: int = 42,
) -> dict:
    matrix_labels = [negative_label, positive_label]
    base_threshold_split = preprocess.split_training_data(processed_data)
    text_column = base_threshold_split["text_column"]
    train_validation = add_threshold_metadata_text(
        base_threshold_split["train_data"].copy().reset_index(drop=True),
        text_column,
    )
    test_frame = add_threshold_metadata_text(
        base_threshold_split["test_data"].copy().reset_index(drop=True),
        text_column,
    )

    train_pool, validation_frame = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=random_seed,
        stratify=train_validation["label"],
    )
    train_pool = train_pool.reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)

    if data_v2.empty:
        data_v2_train_only = pd.DataFrame(columns=train_pool.columns)
    else:
        data_v2 = add_threshold_metadata_text(data_v2, text_column)
        holdout_clean_texts = set(test_frame[text_column].fillna("")) | set(validation_frame[text_column].fillna(""))
        data_v2_train_only = data_v2[~data_v2[text_column].isin(holdout_clean_texts)].copy()

    v2_threshold_train = (
        pd.concat([train_pool, data_v2_train_only], ignore_index=True)
        .drop_duplicates(subset=[text_column, "label"])
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )
    train_modes = {
        "Balanced": preprocess.balance_dataset(train_pool, random_seed=random_seed).reset_index(drop=True),
        "Unbalanced": train_pool.sample(frac=1, random_state=random_seed).reset_index(drop=True),
        "V2 merged": v2_threshold_train,
    }

    print("Thresholds are selected on validation only; final metrics are evaluated on test only.")
    display(
        pd.DataFrame(
            [
                _label_count_row("train_pool_base", train_pool),
                _label_count_row("data_v2_extra_train_only", data_v2_train_only),
                _label_count_row("validation", validation_frame),
                _label_count_row("test", test_frame),
            ]
        )
    )
    display(
        pd.DataFrame(
            [
                {
                    "training_strategy": mode_name,
                    "train_rows": len(train_frame),
                    "ham_train": int((train_frame["label"] == "ham").sum()),
                    "spam_train": int((train_frame["label"] == "spam").sum()),
                    "source_families": train_frame["source_family"].nunique(),
                }
                for mode_name, train_frame in train_modes.items()
            ]
        )
    )

    (
        threshold_results,
        validation_roc_curves,
        confusion_matrices,
        threshold_specs,
    ) = model_checker.run_threshold_experiment(
        train_modes=train_modes,
        validation_frame=validation_frame,
        test_frame=test_frame,
        text_column=text_column,
        target_fpr=target_fpr,
        target_tpr=target_tpr,
        positive_label=positive_label,
    )
    display_columns = [
        "training_strategy",
        "feature_set",
        "model",
        "text_column",
        "score_type",
        "train_rows",
        "ham_train",
        "spam_train",
        "selected_threshold",
        "validation_FPR",
        "validation_TPR",
        "validation_AUC",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "TPR",
        "FPR",
        "FNR",
        "balanced_accuracy",
        "can_reach_99_1_on_validation",
    ]
    display(threshold_results[display_columns].sort_values(["FPR", "TPR"], ascending=[True, False]))

    _plot_confusion_matrix_grid(
        confusion_matrices,
        "training_strategy",
        "Test confusion matrices after validation-selected thresholds",
        matrix_labels=matrix_labels,
    )

    validation_choice = _print_threshold_choice(threshold_results, target_fpr, target_tpr)
    return {
        "target_fpr": target_fpr,
        "target_tpr": target_tpr,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "matrix_labels": matrix_labels,
        "text_column": text_column,
        "train_pool": train_pool,
        "validation_frame": validation_frame,
        "test_frame": test_frame,
        "data_v2_train_only": data_v2_train_only,
        "v2_threshold_train": v2_threshold_train,
        "train_modes": train_modes,
        "threshold_results": threshold_results,
        "validation_roc_curves": validation_roc_curves,
        "confusion_matrices": confusion_matrices,
        "threshold_specs": threshold_specs,
        "validation_choice": validation_choice,
    }


def plot_low_fpr_roc(threshold_context: dict) -> pd.DataFrame:
    plt.figure(figsize=(10, 7))
    line_styles = {"Balanced": ":", "Unbalanced": "--", "V2 merged": "-"}
    for curve in threshold_context["validation_roc_curves"]:
        plt.plot(
            curve["fpr"],
            curve["tpr"],
            linestyle=line_styles.get(curve["training_strategy"], "-"),
            linewidth=2,
            label=(
                f'{curve["training_strategy"]} / {curve.get("feature_set", "Text only")} / '
                f'{curve["model"]} (AUC={curve["auc"]:.3f})'
            ),
        )
        plt.scatter(curve["selected_validation_FPR"], curve["selected_validation_TPR"], s=35)

    plt.axvline(threshold_context["target_fpr"], color="black", linestyle="--", linewidth=1.5, label="Target FPR = 1%")
    plt.axhline(threshold_context["target_tpr"], color="gray", linestyle=":", linewidth=1.5, label="Target TPR = 99%")
    plt.xlim(0, 0.05)
    plt.ylim(0.85, 1.01)
    plt.title("Validation ROC curves near low false-positive rates")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

    strategy_compare = threshold_context["threshold_results"].pivot_table(
        index=["feature_set", "model"],
        columns="training_strategy",
        values=["TPR", "FPR", "precision", "balanced_accuracy"],
        aggfunc="first",
    )
    display(strategy_compare)
    return strategy_compare


def run_failure_analysis(
    threshold_context: dict,
    before_balance_data: pd.DataFrame,
    model_checker: model_from_scratch.SklearnModelChecker,
) -> dict:
    analysis_choice = threshold_context["validation_choice"]
    text_column = threshold_context["text_column"]
    positive_label = threshold_context["positive_label"]
    test_frame = threshold_context["test_frame"]
    threshold = analysis_choice["selected_threshold"]
    analysis_spec = next(
        spec
        for spec in threshold_context["threshold_specs"]
        if spec["training_strategy"] == analysis_choice["training_strategy"]
        and spec["model"] == analysis_choice["model"]
        and spec.get("feature_set") == analysis_choice.get("feature_set")
    )
    analysis_pipeline = analysis_spec["pipeline"]
    analysis_text_column = analysis_spec.get("text_column", text_column)
    analysis_pipeline.fit(
        analysis_spec["train_frame"][analysis_text_column],
        analysis_spec["train_frame"]["label"],
        **analysis_spec["fit_params"],
    )
    analysis_scores, analysis_score_type = model_checker.positive_scores(
        analysis_pipeline,
        test_frame[analysis_text_column],
        positive_label,
    )
    analysis_errors = model_checker.error_analysis_frame(
        test_frame,
        analysis_scores,
        threshold=threshold,
        positive_label=positive_label,
        text_column=analysis_text_column,
    )
    analysis_metrics = model_checker.metrics_at_threshold(
        test_frame["label"],
        analysis_scores,
        threshold=threshold,
        positive_label=positive_label,
    )
    assert int((test_frame["label"] == "ham").sum()) == analysis_metrics["TN"] + analysis_metrics["FP"]
    assert int((test_frame["label"] == "spam").sum()) == analysis_metrics["FN"] + analysis_metrics["TP"]

    print(
        f'Failure analysis model: {analysis_choice["training_strategy"]} / {analysis_choice["model"]} '
        f"with threshold={threshold:.4f}, score_type={analysis_score_type}"
    )

    source_confounding = model_checker.source_label_confounding(before_balance_data)
    source_profile_summary = (
        source_confounding.groupby("source_profile")
        .agg(
            sources=("source_family", "count"),
            rows=("total", "sum"),
            ham=("ham", "sum"),
            spam=("spam", "sum"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    display(source_profile_summary)
    display(source_confounding.sort_values("total", ascending=False).head(15))

    source_errors = model_checker.source_error_summary(analysis_errors)
    for column in ["FPR", "FNR", "TPR"]:
        source_errors[column] = source_errors[column].round(4)
    display(source_errors.head(15))

    overlap_summary = model_checker.score_overlap_summary(analysis_errors, threshold=threshold, window=0.05)
    for column in ["near_threshold_rate", "score_min", "score_median", "score_max"]:
        overlap_summary[column] = overlap_summary[column].round(4)
    display(overlap_summary)

    _plot_score_overlap(analysis_errors, threshold, analysis_score_type)
    fp_near, fn_near = _display_nearest_errors(analysis_errors)

    conflict_reference_data = before_balance_data
    if analysis_text_column not in conflict_reference_data.columns:
        conflict_reference_data = add_threshold_metadata_text(conflict_reference_data, text_column)
    conflicts = model_checker.conflicting_clean_texts(
        conflict_reference_data,
        text_column=analysis_text_column,
        top_n=20,
    )
    display(conflicts)

    fp_tokens = model_checker.top_error_tokens(analysis_errors, "FP", text_column=analysis_text_column, top_n=20)
    fn_tokens = model_checker.top_error_tokens(analysis_errors, "FN", text_column=analysis_text_column, top_n=20)
    error_tokens = pd.concat([fp_tokens, fn_tokens], ignore_index=True)
    display(error_tokens)

    issue_summary = _build_issue_summary(source_profile_summary, overlap_summary, source_errors, conflicts)
    display(issue_summary)
    return {
        "analysis_choice": analysis_choice,
        "analysis_spec": analysis_spec,
        "analysis_pipeline": analysis_pipeline,
        "analysis_text_column": analysis_text_column,
        "analysis_scores": analysis_scores,
        "analysis_score_type": analysis_score_type,
        "analysis_errors": analysis_errors,
        "analysis_metrics": analysis_metrics,
        "source_confounding": source_confounding,
        "source_profile_summary": source_profile_summary,
        "source_errors": source_errors,
        "overlap_summary": overlap_summary,
        "fp_near": fp_near,
        "fn_near": fn_near,
        "conflicts": conflicts,
        "error_tokens": error_tokens,
        "issue_summary": issue_summary,
    }


def _dataset_summary(name: str, frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"dataset": name, "rows": 0, "ham": 0, "spam": 0, "source_families": 0}
    return {
        "dataset": name,
        "rows": len(frame),
        "ham": int((frame["label"] == "ham").sum()),
        "spam": int((frame["label"] == "spam").sum()),
        "source_families": frame["source_family"].nunique(),
    }


def _metadata_token(values: pd.Series, prefix: str) -> pd.Series:
    normalized = (
        values.fillna("")
        .astype(str)
        .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    normalized = normalized.mask(normalized.eq(""), "missing")
    return prefix + "_" + normalized


def _label_count_row(split: str, frame: pd.DataFrame) -> dict:
    return {
        "split": split,
        "rows": len(frame),
        "ham": int((frame["label"] == "ham").sum()) if not frame.empty else 0,
        "spam": int((frame["label"] == "spam").sum()) if not frame.empty else 0,
    }


def _fit_three_by_three_models(
    train_modes: dict[str, pd.DataFrame],
    test_frame: pd.DataFrame,
    text_column: str,
    model_checker: model_from_scratch.SklearnModelChecker,
) -> tuple[pd.DataFrame, list[dict]]:
    estimators = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(dual="auto"),
    }

    rows = []
    matrices = []
    for mode_name, train_frame in train_modes.items():
        for model_name, estimator in estimators.items():
            pipeline = Pipeline(
                [
                    ("tfidf", model_checker.make_tfidf_vectorizer()),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.fit(train_frame[text_column], train_frame["label"])
            predictions = pipeline.predict(test_frame[text_column])
            matrix = confusion_matrix(test_frame["label"], predictions, labels=["ham", "spam"])
            tn, fp, fn, tp = matrix.ravel()
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0
            fnr = fn / (fn + tp) if (fn + tp) else 0
            tnr = tn / (tn + fp) if (tn + fp) else 0

            assert tn + fp == int((test_frame["label"] == "ham").sum())
            assert fn + tp == int((test_frame["label"] == "spam").sum())
            assert abs(accuracy - accuracy_score(test_frame["label"], predictions)) < 1e-12
            assert abs(precision - (tp / (tp + fp) if (tp + fp) else 0)) < 1e-12
            assert abs(recall - (tp / (tp + fn) if (tp + fn) else 0)) < 1e-12
            assert abs(fpr - (fp / (fp + tn) if (fp + tn) else 0)) < 1e-12

            rows.append(
                {
                    "training_mode": mode_name,
                    "model": model_name,
                    "train_rows": len(train_frame),
                    "ham_train": int((train_frame["label"] == "ham").sum()),
                    "spam_train": int((train_frame["label"] == "spam").sum()),
                    "test_rows": len(test_frame),
                    "ham_test": int((test_frame["label"] == "ham").sum()),
                    "spam_test": int((test_frame["label"] == "spam").sum()),
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
                }
            )
            matrices.append({"training_mode": mode_name, "model": model_name, "matrix": matrix})
    return pd.DataFrame(rows), matrices


def _plot_confusion_matrix_grid(
    matrices: list[dict],
    strategy_key: str,
    title: str,
    matrix_labels: list[str] | None = None,
) -> None:
    labels = matrix_labels or ["ham", "spam"]
    columns = 3
    rows = int(np.ceil(len(matrices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, max(4, rows * 3.8)))
    axes = np.asarray(axes).ravel()
    for axis, payload in zip(axes, matrices):
        display_obj = ConfusionMatrixDisplay(confusion_matrix=payload["matrix"], display_labels=labels)
        display_obj.plot(ax=axis, values_format="d", colorbar=False)
        title_parts = [payload[strategy_key]]
        if "feature_set" in payload:
            title_parts.append(payload["feature_set"])
        title_parts.append(payload["model"])
        axis.set_title("\n".join(title_parts), fontsize=9)
    for axis in axes[len(matrices):]:
        axis.axis("off")
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def _plot_scratch_confusion_matrices(y_test: np.ndarray, predictions_by_model: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(predictions_by_model), figsize=(15, 4))
    for axis, (model_name, scratch_predictions) in zip(axes, predictions_by_model.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            scratch_predictions,
            labels=["ham", "spam"],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(f"{model_name}\nfrom scratch")
    plt.tight_layout()
    plt.show()


def _print_threshold_choice(threshold_results: pd.DataFrame, target_fpr: float, target_tpr: float) -> pd.Series:
    selection_fpr_limit = target_fpr * 0.75
    validation_candidates = threshold_results[threshold_results["validation_FPR"] <= selection_fpr_limit].sort_values(
        ["validation_TPR", "validation_FPR", "train_rows"],
        ascending=[False, True, False],
    )
    if validation_candidates.empty:
        validation_candidates = threshold_results[threshold_results["validation_FPR"] <= target_fpr].sort_values(
            ["validation_TPR", "validation_FPR", "train_rows"],
            ascending=[False, True, False],
        )
    if validation_candidates.empty:
        validation_choice = threshold_results.sort_values(
            ["validation_FPR", "validation_TPR"],
            ascending=[True, False],
        ).iloc[0]
        print("No validation candidate keeps FPR <= 1%; choosing the lowest validation-FPR option for analysis.")
    else:
        validation_choice = validation_candidates.iloc[0]

    choice_feature_set = validation_choice.get("feature_set", "Text only")
    print(
        f'Validation-selected candidate: {validation_choice["training_strategy"]} / '
        f'{choice_feature_set} / {validation_choice["model"]} '
        f"(selection validation FPR limit <= {selection_fpr_limit:.4f}) "
        f'-> validation TPR={validation_choice["validation_TPR"]:.4f}, '
        f'validation FPR={validation_choice["validation_FPR"]:.4f}; '
        f'test TPR={validation_choice["TPR"]:.4f}, test FPR={validation_choice["FPR"]:.4f}, '
        f'threshold={validation_choice["selected_threshold"]:.4f}'
    )

    test_diagnostic = threshold_results[threshold_results["FPR"] <= target_fpr].sort_values("TPR", ascending=False)
    if not test_diagnostic.empty:
        best_test = test_diagnostic.iloc[0]
        best_feature_set = best_test.get("feature_set", "Text only")
        print(
            "Diagnostic only - best observed test TPR with test FPR <= 1%: "
            f'{best_test["training_strategy"]} / {best_feature_set} / {best_test["model"]} '
            f'-> TPR={best_test["TPR"]:.4f}, FPR={best_test["FPR"]:.4f}, '
            f'threshold={best_test["selected_threshold"]:.4f}'
        )
    else:
        best_test = threshold_results.sort_values(["FPR", "TPR"], ascending=[True, False]).iloc[0]
        best_feature_set = best_test.get("feature_set", "Text only")
        print(
            "Diagnostic only - no model keeps test FPR <= 1%. Lowest-FPR option: "
            f'{best_test["training_strategy"]} / {best_feature_set} / {best_test["model"]} '
            f'-> TPR={best_test["TPR"]:.4f}, FPR={best_test["FPR"]:.4f}'
        )

    if ((threshold_results["TPR"] >= target_tpr) & (threshold_results["FPR"] <= target_fpr)).any():
        print("At least one validation-tuned model reaches TPR >= 99% and FPR <= 1% on test.")
    else:
        print(
            "No validation-tuned model reaches TPR >= 99% and FPR <= 1% on test; "
            "the target still requires stronger features/model or more data quality work."
        )
    return validation_choice


def _plot_score_overlap(error_frame: pd.DataFrame, threshold: float, score_type: str) -> None:
    plt.figure(figsize=(10, 5.5))
    for label, group in error_frame.groupby("label"):
        plt.hist(group["score"], bins=40, alpha=0.55, density=True, label=label)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="selected threshold")
    if score_type == "probability":
        plt.xlim(max(0, threshold - 0.25), min(1, threshold + 0.25))
    else:
        plt.xlim(threshold - 1.0, threshold + 1.0)
    plt.title("Score overlap near selected threshold")
    plt.xlabel("Spam score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _display_nearest_errors(analysis_errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    error_columns = [
        "error_type",
        "label",
        "prediction",
        "score",
        "threshold_distance",
        "source_family",
        "subject",
        "snippet",
    ]
    fp_near = analysis_errors[analysis_errors["error_type"] == "FP"].sort_values("threshold_distance").head(10).copy()
    fn_near = analysis_errors[analysis_errors["error_type"] == "FN"].sort_values("threshold_distance").head(10).copy()
    for frame in [fp_near, fn_near]:
        frame["score"] = frame["score"].round(4)
        frame["threshold_distance"] = frame["threshold_distance"].round(4)
    print("False positives closest to threshold")
    display(fp_near[error_columns])
    print("False negatives closest to threshold")
    display(fn_near[error_columns])
    return fp_near, fn_near


def _build_issue_summary(
    source_profile_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    source_errors: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> pd.DataFrame:
    ham_only_rows = int(
        source_profile_summary.loc[source_profile_summary["source_profile"].eq("ham-only"), "rows"].sum()
    )
    ham_only_sources = int(
        source_profile_summary.loc[source_profile_summary["source_profile"].eq("ham-only"), "sources"].sum()
    )
    near_ham = int(overlap_summary.loc[overlap_summary["label"].eq("ham"), "near_threshold_rows"].iloc[0])
    near_spam = int(overlap_summary.loc[overlap_summary["label"].eq("spam"), "near_threshold_rows"].iloc[0])
    conflict_rows = int(conflicts["rows"].sum()) if not conflicts.empty else 0
    top_source_errors = ", ".join(source_errors.head(5)["source_family"].astype(str))

    return pd.DataFrame(
        [
            {
                "issue": "Source-label confounding",
                "evidence": (
                    f"{ham_only_sources} ham-only source families with {ham_only_rows} rows; "
                    "source profiles are highly label-skewed."
                ),
                "impact": (
                    "Model can learn source/domain style instead of pure spam semantics, "
                    "hurting cross-source generalization."
                ),
                "next_action": (
                    "Evaluate by source family, reduce source artifacts, "
                    "and add mixed-source validation/holdout checks."
                ),
            },
            {
                "issue": "Score overlap around threshold",
                "evidence": f"Near threshold (+/-0.05): {near_ham} ham and {near_spam} spam rows.",
                "impact": "Raising TPR pulls overlapping ham across threshold, so FPR exceeds 1%.",
                "next_action": "Add stronger features such as URL/header/domain signals or a better semantic model.",
            },
            {
                "issue": "False positives / false negatives are source-patterned",
                "evidence": f"Top source errors include {top_source_errors}.",
                "impact": (
                    "Some sources dominate residual errors; "
                    "global threshold cannot fix all source-specific score shifts."
                ),
                "next_action": (
                    "Inspect source-specific samples, label noise, "
                    "and consider source-aware validation or normalization."
                ),
            },
            {
                "issue": "Exact conflicting clean_text labels",
                "evidence": (
                    f"{len(conflicts)} conflicting clean_text groups shown; "
                    f"total duplicate conflict rows in top list: {conflict_rows}."
                ),
                "impact": "If conflicts exist, no threshold can classify identical text into both labels correctly.",
                "next_action": "Deduplicate/resolve conflicting labels before final training.",
            },
            {
                "issue": "Token/artifact-driven errors",
                "evidence": "FP/FN top-token tables show the words most associated with residual mistakes.",
                "impact": "TF-IDF may be reacting to artifacts rather than robust intent cues.",
                "next_action": "Extend cleaning and add robust non-text features; review FP/FN samples manually.",
            },
        ]
    )
