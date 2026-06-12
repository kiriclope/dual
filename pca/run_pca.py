"""
Unified entry point for all three PCA pipelines.

Usage
-----
  python run_pca.py {single,meta,pseudo} [model-args...]
  python run_pca.py single --help     per-model argument list

Models
------
  single   Per-mouse cross-validated PCA with Procrustes alignment.
           Loads X_all_nan_<scale>.pkl (NaN-padded, one slot per mouse).
           Rebuild from raw data with --rebuild.

  meta     Joint cross-validated PCA on the pooled zero-padded matrix.
           Loads X_all_<scale>.pkl (zero-padded) + mouse_slices.pkl.
           Rebuild from raw data with --rebuild.

  pseudo   Condition-averaged pseudo-population PCA.
           Uses the same data format as meta (X_all_<scale>.pkl).
           Basis fit on within-mouse condition averages — no trial-count
           dilution bias vs meta.
           Rebuild from raw data with --rebuild.

All three scripts save results to ../data/pca/ by default.

Examples
--------
  # single with defaults (loads existing X_all_nan_.pkl)
  python run_pca.py single

  # single: rebuild pseudo-population with std normalisation
  python run_pca.py single --rebuild --scale std

  # single: fit on delay epoch, fewer PCs
  python run_pca.py single --epoch DELAY --n-comp 6

  # meta with defaults
  python run_pca.py meta

  # meta: rebuild + different epoch
  python run_pca.py meta --rebuild --epoch DELAY

  # pseudo with defaults
  python run_pca.py pseudo

  # pseudo: MAD normalisation, more PCs
  python run_pca.py pseudo --norm mad --n-comp 10
"""

import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

SCRIPTS = {
    'single': 'run_single.py',
    'meta':   'run_meta.py',
    'pseudo': 'run_pseudo.py',
}


def _print_help():
    print(__doc__)
    for model, script in SCRIPTS.items():
        header = f'\n{"─" * 60}\n  {model} ({script})\n{"─" * 60}'
        print(header)
        result = subprocess.run(
            [PYTHON, script, '--help'],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            print('  ' + line)


if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
    _print_help()
    sys.exit(0)

model = sys.argv[1]
if model not in SCRIPTS:
    print(f"Error: unknown model {model!r}. Choose from: {list(SCRIPTS)}",
          file=sys.stderr)
    sys.exit(1)

sys.exit(
    subprocess.run([PYTHON, SCRIPTS[model]] + sys.argv[2:]).returncode
)
