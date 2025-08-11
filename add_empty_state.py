"""
Automatically add an empty "state" key to metadata.widgets in Jupyter notebooks
to fix GitHub rendering errors without removing outputs.
"""

import nbformat
import sys
from pathlib import Path

def fix_notebook(path: Path):
    try:
        nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return False

    # Ensure metadata exists
    if "widgets" in nb.metadata:
        widgets_meta = nb.metadata["widgets"]
        if "state" not in widgets_meta:
            widgets_meta["state"] = {}
            print(f"[FIXED] Added empty 'state' to {path}")
        else:
            print(f"[OK] 'state' already present in {path}")
    else:
        print(f"[SKIPPED] No widgets metadata in {path}")
        return False

    try:
        nbformat.write(nb, path)
        return True
    except Exception as e:
        print(f"[ERROR] Could not write {path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_empty_state.py <notebook or folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    files = []
    if target.is_file() and target.suffix == ".ipynb":
        files = [target]
    elif target.is_dir():
        files = list(target.rglob("*.ipynb"))
    else:
        print(f"[ERROR] {target} is not a notebook or folder")
        sys.exit(1)

    for file in files:
        fix_notebook(file)