"""
Run joint cross-validated meta-PCA across all 9 mice and save results.

The meta model fits a single PCA basis on the pooled, zero-padded multi-mouse
matrix.  Each fold trains on a subset of clean trials, Procrustes-aligns
loadings to a reference fit, then projects all trials.

Two-phase pipeline
------------------
Phase 1 (--rebuild):  load raw fluorescence per mouse, build the shared
                      zero-padded X_all_<scale>.pkl + mouse_slices.pkl.
Phase 2 (default):    load pre-built X_all and run cv_pca_meta.

Outputs (saved to --data-dir = ../data/pca/)
-------------------------------------------
meta_traj_<dum>.pkl      (n_trials, n_comp, n_time)
meta_labels_<dum>.pkl    DataFrame aligned with traj
meta_weights_<dum>.pkl   (n_comp, n_neurons_total)
meta_evr_<dum>.pkl       (n_folds, n_comp)

Examples
--------
# rebuild data from raw, then run
python run_meta.py --rebuild

# default run (loads existing X_all_center from ../data/pca)
python run_meta.py

# different epoch
python run_meta.py --epoch DELAY --n-comp 6
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from time import perf_counter
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

from src.common.options import set_options
from src.pca.io import pkl_save, pkl_load, build_padded_X
from src.pca.meta import cv_pca_meta

# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_MICE = [
    'JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
    'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04',
]

parser = argparse.ArgumentParser(
    description='Run joint cross-validated meta-PCA across all mice.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

phase = parser.add_argument_group('phase control')
phase.add_argument('--rebuild', action='store_true',
                   help='Rebuild X_all_<scale> from raw data before running')

io_grp = parser.add_argument_group('data I/O')
io_grp.add_argument('--data-dir', default='../data/pca', dest='data_dir',
                    help='Directory for X_all_<scale>.pkl and results')

build_grp = parser.add_argument_group('build  (--rebuild only)')
build_grp.add_argument('--scale', default='center',
                       choices=['center', 'std', 'mad', 'none'],
                       help="Per-neuron/day scaling. 'center' = mean only; "
                            "'std' = mean + clipped std; 'none' = no scaling. "
                            "Also used as the X_all_<scale>.pkl filename tag.")
build_grp.add_argument('--data-type', default='dF', dest='data_type')
build_grp.add_argument('--scaler-bl', default='standard_BL', dest='scaler_bl',
                       choices=['center_BL', 'standard_BL', 'robust_BL'])
build_grp.add_argument('--days', nargs='+', default=['first', 'last'],
                       metavar='DAY')
build_grp.add_argument('--reload', type=int, default=0, choices=[0, 1])
build_grp.add_argument('--n-neurons', type=int, default=3319, dest='n_neurons',
                       help='Total neuron slots in the padded matrix')

subset_grp = parser.add_argument_group('subset')
subset_grp.add_argument('--mice', nargs='+', default=ALL_MICE, metavar='MOUSE')

pca_grp = parser.add_argument_group('PCA')
pca_grp.add_argument('--epoch', default='TEST',
                     choices=['TEST', 'DELAY', 'ED', 'CHOICE'])
pca_grp.add_argument('--stage', default='Expert',
                     choices=['Expert', 'Naive'])
pca_grp.add_argument('--n-splits', type=int, default=5, dest='n_splits',
                     help='CV folds (-1 = LeaveOneOut)')
pca_grp.add_argument('--n-repeats', type=int, default=10, dest='n_repeats')
pca_grp.add_argument('--n-comp', type=int, default=10, dest='n_comp')
pca_grp.add_argument('--factors', nargs='+', default=['odor_pair'])
pca_grp.add_argument('--mouse-gain', default=None,
                     choices=['equal_mouse', 'equal_neuron', 'none'],
                     dest='mouse_gain',
                     help='Mouse contribution normalisation (none = off)')

args = parser.parse_args()

scale_tag  = '' if args.scale == 'none' else args.scale
mouse_gain = None if (args.mouse_gain is None or args.mouse_gain == 'none') \
             else args.mouse_gain
build_scale = None if args.scale == 'none' else args.scale

# ── shared options ────────────────────────────────────────────────────────────

options_kwargs = {
    'mice': args.mice, 'tasks': ['Dual'],
    'mouse': args.mice[0], 'laser': 0,
    'trials': '', 'reload': args.reload if args.rebuild else 0,
    'data_type': args.data_type,
    'prescreen': None, 'pval': 0.05,
    'preprocess': True, 'scaler_BL': args.scaler_bl,
    'avg_noise': True, 'unit_var_BL': False,
    'random_state': None, 'T_WINDOW': 0.0,
    'l1_ratio': 0.95, 'n_comp': 3, 'pca': 'pca', 'scaler': None,
    'bootstrap': 1, 'n_boots': 128,
    'n_splits': 5, 'n_repeats': 10,
    'class_weight': 0, 'multilabel': 0,
    'mne_estimator': 'generalizing', 'n_jobs': 64,
}
options_kwargs['days'] = args.days
options_base = set_options(**options_kwargs)

# ── phase 1: rebuild (optional) ───────────────────────────────────────────────

if args.rebuild:
    print(f'\n── Building X_all (scale={args.scale!r}) ──────────────────────')
    X_all, y_all, mouse_slices = build_padded_X(
        options_base, n_neurons_total=args.n_neurons, scale=build_scale,
    )
    pkl_save(X_all,        f'X_all_{scale_tag}',  path=args.data_dir)
    pkl_save(y_all,        f'y_all_{scale_tag}',  path=args.data_dir)
    pkl_save(mouse_slices, 'mouse_slices',          path=args.data_dir)
else:
    print(f'Loading X_all_{scale_tag!r} ...')
    X_all        = pkl_load(f'X_all_{scale_tag}',  path=args.data_dir)
    y_all        = pkl_load(f'y_all_{scale_tag}',  path=args.data_dir)
    mouse_slices = pkl_load('mouse_slices',          path=args.data_dir)

y_all['sample']     = y_all.sample_odor
y_all['test']       = y_all.test_odor
y_all['distractor'] = y_all.dist_odor
print(f'X_all {X_all.shape}  y_all {y_all.shape}')

# ── run-id string ─────────────────────────────────────────────────────────────

options = set_options(**options_kwargs)

dum = 'meta'
dum += '_' + args.epoch
dum += '_' + args.stage
dum += '_' + (scale_tag or 'noscale')

if args.n_splits == -1:
    folds = LeaveOneOut()
    dum += '_loo'
else:
    folds = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats)
    dum += f'_{args.n_splits}x{args.n_repeats}'

for f in args.factors:
    if f != 'odor_pair':
        dum += '_' + f
if mouse_gain is not None:
    dum += '_' + mouse_gain

print('dum:', dum)

# ── phase 2: cv meta-PCA ─────────────────────────────────────────────────────

epoch_bins = options['bins_' + args.epoch]
if_scale   = 0   # X_all already scaled at build time

pca_est = PCA(n_components=args.n_comp, svd_solver='randomized')

t0 = perf_counter()
Z_all, y_meta, W_mean, W_ref, evr_folds = cv_pca_meta(
    X_all, y_all, pca_est, folds, args.factors,
    epoch=epoch_bins,
    learning=args.stage,
    mouse_slices=mouse_slices,
    mouse_gain_mode=mouse_gain,
    scale=0,
    if_scale=if_scale,
    scale_test=0,
)
print(f'cv_pca_meta done in {(perf_counter()-t0)/60:.1f} min')

X_meta = np.swapaxes(Z_all, 1, 2).astype(float)   # (trials, n_comp, time)
print('X_meta:', X_meta.shape, '  evr_folds:', evr_folds.shape)

# ── orient PC2 lick-positive ─────────────────────────────────────────────────

bins_choice = options['bins_CHOICE']
mask_lick   = (y_meta.laser == 0) & (y_meta.learning == args.stage) & \
              (y_meta.performance == 1) & (y_meta.choice == 1)
mask_nolick = (y_meta.laser == 0) & (y_meta.learning == args.stage) & \
              (y_meta.performance == 1) & (y_meta.choice == 0)

if np.nanmean(X_meta[mask_lick][:, 1, bins_choice]) < \
   np.nanmean(X_meta[mask_nolick][:, 1, bins_choice]):
    print('Flipping PC2')
    X_meta[:, 1, :]  *= -1
    W_ref[1, :]      *= -1
    W_mean[1, :]     *= -1
else:
    print('PC2 orientation OK')

w_meta = W_ref  * 100
w_mean = W_mean * 100

# ── save ──────────────────────────────────────────────────────────────────────

pkl_save(X_meta,    f'meta_traj_{dum}',    path=args.data_dir)
pkl_save(y_meta,    f'meta_labels_{dum}',  path=args.data_dir)
pkl_save(w_meta,    f'meta_weights_{dum}', path=args.data_dir)
pkl_save(evr_folds, f'meta_evr_{dum}',     path=args.data_dir)
print('Results saved to', args.data_dir)
print('dum:', dum)
