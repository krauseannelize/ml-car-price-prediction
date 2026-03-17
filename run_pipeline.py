"""Execute all notebooks in order."""

import subprocess
import sys

NOTEBOOKS = [
    "notebooks/01-data-gathering.ipynb",
    "notebooks/02-data-cleaning.ipynb",
    "notebooks/03-eda.ipynb",
    "notebooks/04-feature-engineering.ipynb",
    "notebooks/05-preprocessing.ipynb",
    "notebooks/06-train-models.ipynb",
    "notebooks/07-evaluation.ipynb",
]

for nb in NOTEBOOKS:
    print(f"\nRunning {nb}...")
    result = subprocess.run(
        ["jupyter", "execute", nb, "--inplace"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"FAILED: {nb}")
        print(result.stderr)
        sys.exit(1)
    print(f"Done: {nb}")

print("\nPipeline complete.")
