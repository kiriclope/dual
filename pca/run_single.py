"""
Run per-mouse cross-validated PCA for all 9 mice and save results to ../data/pca/.

Two-phase pipeline
------------------
Phase 1 (--rebuild):  load raw fluorescence per mouse, build the shared
                      pseudo-population array X_all_nan_<scale>.pkl, and save it.
Phase 2 (default):    load the pre-built pseudo-population and run cv PCA.

Pass --rebuild to run both phases; omit it to skip phase 1 and use an
existing X_all_nan_<scale>.pkl.

Outputs
-------
../data/pca/single_traj_<dum>.pkl    (n_trials, n_comp, n_time)
../data/pca/single_labels_<dum>.pkl  DataFrame aligned with traj
../data/pca/single_weights_<dum>.pkl (n_comp, n_neurons_total)
../data/pca/single_evr_<dum>.pkl     (n_mice, n_comp)

Examples
--------
# default run (loads existing X_all_nan_.pkl, matches notebook default)
python run_single.py

# rebuild pseudo-population then run PCA
python run_single.py --rebuild

# rebuild with std normalisation
python run_single.py --rebuild --scale std

# different fit epoch and number of PCs
python run_single.py --epoch DELAY --n-comp 6

# quick test with one mouse
python run_single.py --mice JawsM01 --rebuild
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

from src.common.options import set_options
from src.common.get_data import get_X_y_days
from src.pca.io import pkl_save, pkl_load
from src.pca.single import cv_pca_single, align_mice

# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_MICE = [
    'JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
    'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04',
]
REF_MOUSE = 'ACCM03'   # Procrustes reference

parser = argparse.ArgumentParser(
    description='Build pseudo-population and/or run per-mouse cv PCA.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

phase = parser.add_argument_group('phase control')
phase.add_argument('--rebuild', action='store_true',
                   help='Rebuild X_all_nan from raw data before running PCA')

io_grp = parser.add_argument_group('data I/O')
io_grp.add_argument('--data-dir', default='../data/pca', dest='data_dir',
                    help='Directory for X_all_nan_.pkl and PCA results')

build_grp = parser.add_argument_group('pseudo-population build  (--rebuild only)')
build_grp.add_argument('--scale', default='',
                       choices=['', 'std'],
                       help="Per-neuron/day scaling after BL correction. "
                            "'' = none (notebook default), 'std' = divide by "
                            "clipped std. Also used as the X_all_nan_<scale>.pkl "
                            "filename tag.")
build_grp.add_argument('--data-type', default='dF', dest='data_type',
                       help='Fluorescence data type passed to get_X_y_days')
build_grp.add_argument('--scaler-bl', default='center_BL', dest='scaler_bl',
                       choices=['center_BL', 'standard_BL', 'robust_BL'],
                       help='Baseline scaler applied inside get_X_y_days')
build_grp.add_argument('--days', nargs='+', default=['first', 'last'],
                       metavar='DAY',
                       help="Days to include, e.g. 'first last' or 'all'")
build_grp.add_argument('--reload', type=int, default=0, choices=[0, 1],
                       help='Force raw-data reload inside get_X_y_days (0=use cache)')

subset_grp = parser.add_argument_group('subset selection')
subset_grp.add_argument('--mice', nargs='+', default=ALL_MICE,
                        metavar='MOUSE', help='Mice to process')

pca_grp = parser.add_argument_group('PCA')
pca_grp.add_argument('--epoch', default='TEST',
                     choices=['TEST', 'DELAY', 'ED', 'CHOICE'],
                     help='Time window used to fit PCA per fold')
pca_grp.add_argument('--stage', default='Expert',
                     choices=['Expert', 'Naive'],
                     help='Learning stage of trials used for PCA fitting')
pca_grp.add_argument('--cv-scale', default='standard',
                     choices=['standard', 'none'], dest='cv_scale',
                     help='Within-fold feature scaling before PCA (none = no scaling)')
pca_grp.add_argument('--n-splits', type=int, default=-1, dest='n_splits',
                     help='CV folds (-1 = LeaveOneOut)')
pca_grp.add_argument('--n-repeats', type=int, default=1, dest='n_repeats',
                     help='CV repeats (ignored when n_splits=-1)')
pca_grp.add_argument('--n-comp', type=int, default=10, dest='n_comp',
                     help='Number of PCs to retain')
pca_grp.add_argument('--correct', dest='correct', action='store_true',
                     help='Restrict to correct trials (default)')
pca_grp.add_argument('--no-correct', dest='correct', action='store_false',
                     help='Include all trials')
pca_grp.add_argument('--factors', nargs='+', default=['odor_pair'],
                     help='Condition columns for trial averaging (Procrustes anchors)')
parser.set_defaults(correct=True)

args = parser.parse_args()

# ── shared options (used by both phases) ─────────────────────────────────────

options_kwargs = {
    'mice': args.mice, 'tasks': ['Dual'],
    'mouse': args.mice[0], 'laser': 0,
    'trials': '', 'reload': args.reload if args.rebuild else 0,
    'data_type': args.data_type,
    'prescreen': None, 'pval': 0.05,
    'preprocess': None, 'scaler_BL': args.scaler_bl,
    'avg_noise': False, 'unit_var_BL': False,
    'random_state': None, 'T_WINDOW': 0.0,
    'l1_ratio': 0.95, 'n_comp': 3, 'pca': 'pca', 'scaler': None,
    'bootstrap': 1, 'n_boots': 128,
    'n_splits': 5, 'n_repeats': 10,
    'class_weight': 0, 'multilabel': 0,
    'mne_estimator': 'generalizing', 'n_jobs': 64,
}
options_kwargs['days'] = args.days
options_base = set_options(**options_kwargs)

# ── phase 1: rebuild pseudo-population ───────────────────────────────────────

def build_X_all(args, options_base):
    """Load per-mouse raw data and assemble the shared pseudo-population array.

    Mirrors the 'Loading Raw Calcium data' section of singlePCA.org.
    Saves X_all_nan_<scale>.pkl and y_all_nan_<scale>.pkl to args.data_dir.
    """
    print('\n── Building X_all ──────────────────────────────────────────────')
    opts = dict(options_base)

    # First pass: count total neurons across all mice
    n_neurons = 0
    for mouse in args.mice:
        opts['mouse'] = mouse
        opts['reload'] = args.reload
        o = set_options(**opts)
        X, _ = get_X_y_days(**o)
        n_neurons += X.shape[1]
    print(f'Auto n_neurons = {n_neurons}')

    X_trials, y_trials = [], []
    counter = 0

    for mouse in args.mice:
        opts['mouse'] = mouse
        opts['reload'] = args.reload
        o = set_options(**opts)
        X, y = get_X_y_days(**o)
        print(f'{mouse}  X {X.shape}  y {y.shape}  n_days {o["n_days"]}')

        X_scale = X.copy()
        std_ = 1
        for day in range(1, o['n_days'] + 1):
            idx0 = (y.day == day) & (y.laser == 0)
            mean_ = 0.0

            if args.scale != '':
                mean_ = np.nanmean(X[idx0], axis=0, keepdims=True)
            if args.scale == 'std':
                std_ = np.nanstd(X[idx0], axis=(0, 2), ddof=0, keepdims=True)
                lo, hi = np.percentile(std_.ravel(), [5, 95])
                std_ = np.clip(std_, lo, hi)

            idx = (y.day == day)
            X_scale[idx] = (X[idx] - mean_) / std_

        X_trial = np.full((X.shape[0], n_neurons, X.shape[-1]), np.nan)
        X_trial[:, counter:counter + X.shape[1], :] = X_scale
        counter += X.shape[1]

        y['mouse'] = mouse
        X_trials.append(X_trial)
        y_trials.append(y)

    X_all = np.concatenate(X_trials, axis=0)
    y_all = pd.concat(y_trials, ignore_index=True)
    print(f'X_all {X_all.shape}  y_all {y_all.shape}')

    pkl_save(X_all, f'X_all_nan_{args.scale}', path=args.data_dir)
    pkl_save(y_all, f'y_all_nan_{args.scale}', path=args.data_dir)
    return X_all, y_all


if args.rebuild:
    X_all, y_all = build_X_all(args, options_base)
else:
    print(f'Loading X_all_nan_{args.scale!r} ...')
    X_all = pkl_load(f'X_all_nan_{args.scale}', path=args.data_dir)
    y_all = pkl_load(f'y_all_nan_{args.scale}', path=args.data_dir)

y_all['sample']     = y_all.sample_odor
y_all['test']       = y_all.test_odor
y_all['distractor'] = y_all.dist_odor
print(f'X_all {X_all.shape}  y_all {y_all.shape}')

# ── run-id string ─────────────────────────────────────────────────────────────

options = set_options(**options_kwargs)

dum = 'pca'
dum += '_' + args.epoch
epoch_bins = options['bins_' + args.epoch]

dum += '_' + args.stage
dum += '_' + args.cv_scale

if args.n_splits == -1:
    folds = LeaveOneOut()
    dum += '_loo'
else:
    folds = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats)
    dum += f'_{args.n_splits}x{args.n_repeats}'

if args.correct:
    dum += '_correct'
for f in args.factors:
    dum += '_' + f
if args.scale:
    dum += f'_scale_{args.scale}'
if args.scaler_bl != 'center_BL':
    dum += '_' + args.scaler_bl

print('dum:', dum)

# ── phase 2: cv PCA ───────────────────────────────────────────────────────────

cv_scale = None if args.cv_scale == 'none' else args.cv_scale
pca_est  = PCA(n_components=args.n_comp, svd_solver='randomized')

t0 = perf_counter()
X_mice, y_mice, w_mice, evr_mice = [], [], [], []

for mouse in args.mice:
    idx = y_all['mouse'] == mouse
    X_m = X_all[idx]
    y_m = y_all.loc[idx].reset_index(drop=True)

    valid = ~np.all(np.isnan(X_m), axis=(0, 2))
    X_m = X_m[:, valid, :]

    Z, y_z, w, evr = cv_pca_single(
        X_m, y_m, pca_est, folds, args.factors,
        epoch=epoch_bins, scale=cv_scale, stage=args.stage,
        correct=args.correct,
    )
    X_mice.append(Z)
    y_mice.append(y_z)
    w_mice.append(w)
    evr_mice.append(evr)

h, m, s = (int((perf_counter()-t0)//3600),
           int(((perf_counter()-t0) % 3600) // 60),
           int((perf_counter()-t0) % 60))
print(f'cv_pca_single done in {h}h {m}m {s}s')

# ── align across mice (Procrustes on score-space anchors) ─────────────────────

ref_idx = args.mice.index(REF_MOUSE) if REF_MOUSE in args.mice else 0
X_aligned, y_aligned = align_mice(X_mice, y_mice, args.factors, ref_idx=ref_idx)

X_single = np.swapaxes(X_aligned, 1, 2)   # (trials, n_comp, time)
y_single = y_aligned
print('X_single:', X_single.shape)

# ── orient PC2 lick-positive ─────────────────────────────────────────────────

bins_choice = options['bins_CHOICE']
mask_lick   = (y_single.laser == 0) & (y_single.performance == 1) & (y_single.choice == 1)
mask_nolick = (y_single.laser == 0) & (y_single.performance == 1) & (y_single.choice == 0)

if np.nanmean(X_single[mask_lick][:, 1, bins_choice]) < np.nanmean(X_single[mask_nolick][:, 1, bins_choice]):
    print('Flipping PC2')
    X_single[:, 1, :] *= -1
else:
    print('PC2 orientation OK')

# ── weights and EVR ───────────────────────────────────────────────────────────

evr_single = np.vstack([evr.mean(0) for evr in evr_mice])
w_single   = np.hstack([w.mean(0) for w in w_mice]) * 100
print('w_single:', w_single.shape, '  evr_single:', evr_single.shape)

# ── save ──────────────────────────────────────────────────────────────────────

pkl_save(X_single,   f'single_traj_{dum}',    path=args.data_dir)
pkl_save(y_single,   f'single_labels_{dum}',  path=args.data_dir)
pkl_save(w_single,   f'single_weights_{dum}', path=args.data_dir)
pkl_save(evr_single, f'single_evr_{dum}',     path=args.data_dir)
print('Results saved to', args.data_dir)
print('dum:', dum)
