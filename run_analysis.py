"""
Unified entry point for all analysis pipelines.

Usage
-----
  python run_analysis.py --model MODEL [model-args...]
  python run_analysis.py --model MODEL --help

Models
------
  single    Per-mouse cross-validated PCA with Procrustes alignment.
  meta      Joint cross-validated PCA on the pooled zero-padded matrix.
  pseudo    Condition-averaged pseudo-population PCA.
  overlaps  Cross-condition generalised decoding (CCGD) — decision functions.

All remaining args are forwarded verbatim to the model script.

Examples
--------
  python run_analysis.py --model single
  python run_analysis.py --model single --rebuild --scale std
  python run_analysis.py --model single --epoch DELAY --n-comp 6

  python run_analysis.py --model meta
  python run_analysis.py --model meta --rebuild --epoch DELAY

  python run_analysis.py --model pseudo --norm mad

  python run_analysis.py --model overlaps
  python run_analysis.py --model overlaps --rebuild --scale std
  python run_analysis.py --model overlaps --mice JawsM01 --stages Expert

  python run_analysis.py --model single --help    (model-specific args)
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    'single':   os.path.join(ROOT, 'pca',      'run_single.py'),
    'meta':     os.path.join(ROOT, 'pca',      'run_meta.py'),
    'pseudo':   os.path.join(ROOT, 'pca',      'run_pseudo.py'),
    'overlaps': os.path.join(ROOT, 'overlaps', 'run_overlaps.py'),
}


def _usage():
    print(__doc__)
    sys.exit(0)


def _model_help(model):
    subprocess.run([sys.executable, SCRIPTS[model], '--help'])
    sys.exit(0)


# ── parse --model (only) from argv ────────────────────────────────────────────

argv = sys.argv[1:]

if not argv or '--help' in argv or '-h' in argv:
    # no --model given: show this script's help
    if '--model' not in argv and '-m' not in argv:
        _usage()

# extract --model value without touching the rest
model = None
rest  = []
i = 0
while i < len(argv):
    if argv[i] in ('--model', '-m') and i + 1 < len(argv):
        model = argv[i + 1]
        i += 2
    else:
        rest.append(argv[i])
        i += 1

if model is None:
    print("Error: --model is required.", file=sys.stderr)
    print(f"  Choices: {list(SCRIPTS)}", file=sys.stderr)
    sys.exit(1)

if model not in SCRIPTS:
    print(f"Error: unknown model {model!r}.", file=sys.stderr)
    print(f"  Choices: {list(SCRIPTS)}", file=sys.stderr)
    sys.exit(1)

sys.exit(subprocess.run([sys.executable, SCRIPTS[model]] + rest).returncode)
