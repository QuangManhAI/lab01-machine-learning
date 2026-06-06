import json
from pathlib import Path

notebook_path = Path("notebooks/lab01.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Refactored Cell 49 Code
refactored_code = """import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

TARGET_FPR = 0.01
TARGET_TPR = 0.99
positive_label = "spam"
negative_label = "ham"
matrix_labels = [negative_label, positive_label]

base_threshold_split = preprocess.split_training_data(data)
threshold_train_validation = base_threshold_split["train_data"].copy().reset_index(drop=True)
threshold_test_frame = base_threshold_split["test_data"].copy().reset_index(drop=True)

train_pool, validation_frame = train_test_split(
    threshold_train_validation,
    test_size=0.25,
    random_state=42,
    stratify=threshold_train_validation["label"],
)
train_pool = train_pool.reset_index(drop=True)
validation_frame = validation_frame.reset_index(drop=True)

if DATA_V2_PROCESSED.exists():
    threshold_data_v2 = pd.read_csv(DATA_V2_PROCESSED).fillna("")
    if "clean_text" not in threshold_data_v2.columns:
        threshold_data_v2 = preprocess.add_preprocessing_columns(threshold_data_v2)
    threshold_data_v2 = preprocess.filter_trainable_rows(threshold_data_v2).copy()
    holdout_clean_texts = set(threshold_test_frame[TEXT_COLUMN].fillna("")) | set(validation_frame[TEXT_COLUMN].fillna(""))
    threshold_data_v2_train_only = threshold_data_v2[~threshold_data_v2[TEXT_COLUMN].isin(holdout_clean_texts)].copy()
else:
    threshold_data_v2 = pd.DataFrame()
    threshold_data_v2_train_only = pd.DataFrame()

v2_threshold_train = (
    pd.concat([train_pool, threshold_data_v2_train_only], ignore_index=True)
    .drop_duplicates(subset=[TEXT_COLUMN, "label"])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

threshold_train_modes = {
    "Balanced": preprocess.balance_dataset(train_pool, random_seed=42),
    "Unbalanced": train_pool.sample(frac=1, random_state=42).reset_index(drop=True),
    "V2 merged": v2_threshold_train,
}

print("Thresholds are selected on validation only; final metrics are evaluated on test only.")
display(pd.DataFrame([
    {"split": "train_pool_base", "rows": len(train_pool), "ham": int((train_pool["label"] == "ham").sum()), "spam": int((train_pool["label"] == "spam").sum())},
    {"split": "data_v2_extra_train_only", "rows": len(threshold_data_v2_train_only), "ham": int((threshold_data_v2_train_only["label"] == "ham").sum()) if not threshold_data_v2_train_only.empty else 0, "spam": int((threshold_data_v2_train_only["label"] == "spam").sum()) if not threshold_data_v2_train_only.empty else 0},
    {"split": "validation", "rows": len(validation_frame), "ham": int((validation_frame["label"] == "ham").sum()), "spam": int((validation_frame["label"] == "spam").sum())},
    {"split": "test", "rows": len(threshold_test_frame), "ham": int((threshold_test_frame["label"] == "ham").sum()), "spam": int((threshold_test_frame["label"] == "spam").sum())},
]))

display(pd.DataFrame([
    {"training_strategy": mode_name, "train_rows": len(train_frame), "ham_train": int((train_frame["label"] == "ham").sum()), "spam_train": int((train_frame["label"] == "spam").sum()), "source_families": train_frame["source_family"].nunique()}
    for mode_name, train_frame in threshold_train_modes.items()
]))

# Run threshold experiments using refactored method in model_from_scratch.py
threshold_results, validation_roc_curves, confusion_matrices = model_checker.run_threshold_experiment(
    train_modes=threshold_train_modes,
    validation_frame=validation_frame,
    test_frame=threshold_test_frame,
    text_column=TEXT_COLUMN,
    target_fpr=TARGET_FPR,
    target_tpr=TARGET_TPR,
    positive_label=positive_label,
)

display_columns = [
    "training_strategy", "model", "score_type", "train_rows", "ham_train", "spam_train",
    "selected_threshold", "validation_FPR", "validation_TPR", "validation_AUC",
    "TN", "FP", "FN", "TP", "accuracy", "precision", "TPR", "FPR", "FNR", "balanced_accuracy",
    "can_reach_99_1_on_validation",
]
display(threshold_results[display_columns].sort_values(["FPR", "TPR"], ascending=[True, False]))

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for axis, payload in zip(axes.ravel(), confusion_matrices):
    display_obj = ConfusionMatrixDisplay(confusion_matrix=payload["matrix"], display_labels=matrix_labels)
    display_obj.plot(ax=axis, colorbar=False, values_format="d")
    axis.set_title(f'{payload["training_strategy"]}\\n{payload["model"]}')
plt.suptitle("Test confusion matrices after validation-selected thresholds", y=1.02)
plt.tight_layout()
plt.show()

validation_candidates = threshold_results[threshold_results["validation_FPR"] <= TARGET_FPR].sort_values("validation_TPR", ascending=False)
validation_choice = validation_candidates.iloc[0]
print(
    f'Validation-selected candidate: {validation_choice["training_strategy"]} / {validation_choice["model"]} '
    f'-> validation TPR={validation_choice["validation_TPR"]:.4f}, validation FPR={validation_choice["validation_FPR"]:.4f}; '
    f'test TPR={validation_choice["TPR"]:.4f}, test FPR={validation_choice["FPR"]:.4f}, threshold={validation_choice["selected_threshold"]:.4f}'
)

test_diagnostic = threshold_results[threshold_results["FPR"] <= TARGET_FPR].sort_values("TPR", ascending=False)
if not test_diagnostic.empty:
    best_test = test_diagnostic.iloc[0]
    print(
        f'Diagnostic only - best observed test TPR with test FPR <= 1%: {best_test["training_strategy"]} / {best_test["model"]} '
        f'-> TPR={best_test["TPR"]:.4f}, FPR={best_test["FPR"]:.4f}, threshold={best_test["selected_threshold"]:.4f}'
    )
else:
    best_test = threshold_results.sort_values(["FPR", "TPR"], ascending=[True, False]).iloc[0]
    print(
        f'Diagnostic only - no model keeps test FPR <= 1%. Lowest-FPR option: {best_test["training_strategy"]} / {best_test["model"]} '
        f'-> TPR={best_test["TPR"]:.4f}, FPR={best_test["FPR"]:.4f}'
    )

if ((threshold_results["TPR"] >= TARGET_TPR) & (threshold_results["FPR"] <= TARGET_FPR)).any():
    print("At least one validation-tuned model reaches TPR >= 99% and FPR <= 1% on test.")
else:
    print("No validation-tuned model reaches TPR >= 99% and FPR <= 1% on test; the target still requires stronger features/model or more data quality work.")"""

nb["cells"][49]["source"] = [line + "\n" if idx < len(refactored_code.splitlines()) - 1 else line for idx, line in enumerate(refactored_code.splitlines())]
print("Refactored Cell 49")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Saved refactored lab01.ipynb")
