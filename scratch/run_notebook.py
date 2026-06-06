import json
import sys
import traceback
import io
import base64
from pathlib import Path

# Use non-interactive backend for matplotlib to avoid GUI window popups
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

notebook_path = Path("notebooks/lab01.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Setup execution namespace
from IPython.display import display
PROJECT_ROOT = Path.cwd().resolve()
globals_dict = {
    "__name__": "__main__",
    "PROJECT_ROOT": PROJECT_ROOT,
    "display": display,
}

# Add notebooks directory to python path
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
sys.path.insert(0, str(PROJECT_ROOT))

success = True
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if not source.strip():
            continue
            
        print(f"Executing cell {idx}...")
        cell["outputs"] = []
        cell["execution_count"] = idx
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        try:
            # Execute the cell source code
            exec(source, globals_dict)
            execution_failed = False
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(f"Error in cell {idx}: {e}", file=sys.stderr)
            execution_failed = True
            success = False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        # Collect stdout/stderr outputs
        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()
        
        if stdout_val:
            cell["outputs"].append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in stdout_val.splitlines()]
            })
        if stderr_val:
            cell["outputs"].append({
                "output_type": "stream",
                "name": "stderr",
                "text": [line + "\n" for line in stderr_val.splitlines()]
            })
            
        # Capture matplotlib figures
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                cell["outputs"].append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_str,
                        "text/plain": f"<Figure size {fig.get_size_inches()*fig.dpi}>"
                    },
                    "metadata": {}
                })
            plt.close("all")
            
        if execution_failed:
            print(f"Execution failed at cell {idx}. Stopping.")
            break

# Save the updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Finished notebook execution. Success: {success}")
