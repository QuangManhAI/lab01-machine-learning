import json
from pathlib import Path

notebook_path = Path("notebooks/lab01.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_49 = nb["cells"][49]
src_49 = "".join(cell_49["source"])

old_call = """threshold_results, validation_roc_curves, confusion_matrices = model_checker.run_threshold_experiment("""
new_call = """threshold_results, validation_roc_curves, confusion_matrices, threshold_specs = model_checker.run_threshold_experiment("""

if old_call in src_49:
    src_49 = src_49.replace(old_call, new_call)
    cell_49["source"] = [line + "\n" if idx < len(src_49.splitlines()) - 1 else line for idx, line in enumerate(src_49.splitlines())]
    print("Successfully fixed Cell 49 call")
else:
    print("Error: old_call not found in Cell 49 source.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("Saved lab01.ipynb")
