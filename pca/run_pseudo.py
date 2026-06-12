"""
Run condition-averaged pseudo-population PCA across all 9 mice and save results.

The pseudo model fits a PCA basis on within-mouse condition averages (no
trial-count dilution).  Each fold Procrustes-aligns to a reference fit and
records explained variance.  All trials (clean + perturbed) are projected onto
the reference basis.

Two-phase pipeline
------------------
Phase 1 (--rebuild):  load raw fluorescence per mouse, build the shared
                      zero-padded X_all_<scale>.pkl + mouse_slices.pkl.
Phase 2 (default):    load pre-built X_all and run cv_pca_pseudo.

Outputs (saved to --data-dir = ../data/pca/)
-------------------------------------------
pseudo_traj_<dum>.pkl      (n_trials, n_comp, n_time)
pseudo_labels_<dum>.pkl    DataFrame aligned with traj
pseudo_weights_<dum>.pkl   (n_comp, n_neurons_total)
pseudo_evr_<dum>.pkl       (n_folds, n_comp)

Examples
--------
# rebuild data from raw, then run
python run_pseudo.py --rebuild

# default run (loads existing X_all_center from ../data/pca)
python run_pseudo.py

# different epoch, fewer PCs
python run_pseudo.py --epoch TEST --n-comp 10

# MAD normalisation
python run_pseudo.py --rebuild --scale mad --norm none
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
from src.pca.pseudo import cv_pca_pseudo, remove_ci_subspace

# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_MICE = [
    'JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
    'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04',
]

parser = argparse.ArgumentParser(
    description='Run condition-averaged pseudo-population PCA.',
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
                       choices=['center', 'blcenter', 'std', 'mad', 'none'],
                       help="Per-neuron/day scaling. 'center' = subtract per-day "
                            "mean PSTH at every time bin (removes condition-"
                            "independent + dependent structure → can push e.g. DPA "
                            "negative during the GNG cue); 'blcenter' = subtract "
                            "per-day BASELINE mean only (drift removal without that "
                            "artifact); 'std' = mean + clipped std; 'none' = none. "
                            "Also the X_all_<scale>.pkl filename tag.")
build_grp.add_argument('--data-type', default='dF', dest='data_type')
build_grp.add_argument('--scaler-bl', default='standard_BL', dest='scaler_bl',
                       choices=['center_BL', 'standard_BL', 'robust_BL'])
build_grp.add_argument('--preprocess', type=int, default=1, choices=[0, 1],
                       help='Apply per-day baseline scaling (scaler-bl) in '
                            'get_X_y_days. 0 = raw fluorescence, no baseline '
                            'normalisation.')
build_grp.add_argument('--days', nargs='+', default=['first', 'last'],
                       metavar='DAY')
build_grp.add_argument('--reload', type=int, default=0, choices=[0, 1])
build_grp.add_argument('--n-neurons', type=int, default=3319, dest='n_neurons',
                       help='Total neuron slots in the padded matrix')

subset_grp = parser.add_argument_group('subset')
subset_grp.add_argument('--mice', nargs='+', default=ALL_MICE, metavar='MOUSE')

pca_grp = parser.add_argument_group('PCA')
pca_grp.add_argument('--epoch', default='DELAY',
                     choices=['TEST', 'DELAY', 'ED', 'CHOICE'],
                     help='Time window for fitting PCA per fold')
pca_grp.add_argument('--stage', default='Expert',
                     choices=['Expert', 'Naive'])
pca_grp.add_argument('--norm', default='zscore',
                     choices=['zscore', 'mad', 'none'],
                     help='Per-neuron normalisation before PCA '
                          '(applied to condition averages inside each fold)')
pca_grp.add_argument('--n-splits', type=int, default=5, dest='n_splits',
                     help='CV folds (-1 = LeaveOneOut)')
pca_grp.add_argument('--n-repeats', type=int, default=10, dest='n_repeats')
pca_grp.add_argument('--n-comp', type=int, default=6, dest='n_comp')
pca_grp.add_argument('--remove-ci', type=int, default=0, dest='remove_ci', metavar='Q',
                     help='Project out the top-Q condition-independent (ramp) '
                          'directions per mouse before the fit, stripping the '
                          'common-mode from the task axes without the per-time '
                          'demean artifact. 0 = off.')
pca_grp.add_argument('--pert-ref', dest='pert_foldwise', action='store_false',
                     help='Project perturbed/non-clean trials once through the '
                          'reference basis. Default projects them fold-by-fold '
                          '(Procrustes-aligned, averaged) like the clean trials.')
parser.set_defaults(pert_foldwise=True)
pca_grp.add_argument('--factors', nargs='+',
                     default=['odor_pair', 'tasks'],
                     help='Condition columns for the condition-average step. '
                          'NB: for correct trials choice is collinear with '
                          'odor_pair, so adding it only injects empty (all-zero) '
                          'condition rows — leave it out.')

args = parser.parse_args()

scale_tag   = '' if args.scale == 'none' else args.scale
build_scale = None if args.scale == 'none' else args.scale
norm_val    = None if args.norm == 'none' else args.norm

# ── shared options ────────────────────────────────────────────────────────────

options_kwargs = {
    'mice': args.mice, 'tasks': ['Dual'],
    'mouse': args.mice[0], 'laser': 0,
    'trials': '', 'reload': args.reload if args.rebuild else 0,
    'data_type': args.data_type,
    'prescreen': None, 'pval': 0.05,
    'preprocess': bool(args.preprocess), 'scaler_BL': args.scaler_bl,
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

# ── remove condition-independent (ramp) subspace ───────────────────────────────

if args.remove_ci > 0:
    print(f'Removing top-{args.remove_ci} condition-independent directions per mouse ...')
    X_all = remove_ci_subspace(X_all, y_all, mouse_slices, q=args.remove_ci,
                               factors=args.factors, learning=args.stage)

# ── run-id string ─────────────────────────────────────────────────────────────

options = set_options(**options_kwargs)

dum = 'pseudo'
dum += '_' + args.epoch
dum += '_' + args.stage
dum += '_' + (norm_val or 'nonorm')

if args.n_splits == -1:
    folds = LeaveOneOut()
    dum += '_loo'
else:
    folds = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats)
    dum += f'_{args.n_splits}x{args.n_repeats}'

# center is the canonical default → left untagged; every other scale (incl.
# 'none') is tagged explicitly so runs never collide on the DUM
if args.scale != 'center':
    dum += '_scale_' + args.scale
if not args.preprocess:
    dum += '_raw'
# non-default condition factors change the basis → tag so they never collide
if args.factors != ['odor_pair', 'tasks']:
    dum += '_f-' + '-'.join(args.factors)
# perturbed projected through reference basis (non-default) → tag
if not args.pert_foldwise:
    dum += '_pertref'
# condition-independent subspace removal → tag
if args.remove_ci > 0:
    dum += f'_ci{args.remove_ci}'

print('dum:', dum)

# ── phase 2: cv pseudo-PCA ───────────────────────────────────────────────────

epoch_bins = options['bins_' + args.epoch]
bl_bins    = slice(0, 12)
pca_est    = PCA(n_components=args.n_comp, svd_solver='auto')

t0 = perf_counter()
Z_all, y_meta, W_ref, Z_cond, evr_folds = cv_pca_pseudo(
    X_all, y_all, pca_est, folds, args.factors,
    epoch=epoch_bins, bl_bins=bl_bins,
    learning=args.stage,
    norm=norm_val,
    mouse_slices=mouse_slices,
    pert_foldwise=args.pert_foldwise,
)
print(f'cv_pca_pseudo done in {(perf_counter()-t0)/60:.1f} min')

X_pseudo = np.swapaxes(Z_all, 1, 2).astype(float)   # (trials, n_comp, time)
w_pseudo = W_ref * 100
print('X_pseudo:', X_pseudo.shape, '  evr_folds:', evr_folds.shape)
print('fold-mean EVR (%):', np.round(evr_folds.mean(0) * 100, 1))

# ── orient PC2 lick-positive ─────────────────────────────────────────────────

bins_choice = options['bins_CHOICE']
mask_lick   = (y_meta.laser == 0) & (y_meta.learning == args.stage) & \
              (y_meta.performance == 1) & (y_meta.choice == 1)
mask_nolick = (y_meta.laser == 0) & (y_meta.learning == args.stage) & \
              (y_meta.performance == 1) & (y_meta.choice == 0)

if np.nanmean(X_pseudo[mask_lick][:, 1, bins_choice]) < \
   np.nanmean(X_pseudo[mask_nolick][:, 1, bins_choice]):
    print('Flipping PC2')
    X_pseudo[:, 1, :] *= -1
    W_ref[1, :]       *= -1
else:
    print('PC2 orientation OK')

# ── save ──────────────────────────────────────────────────────────────────────

pkl_save(X_pseudo,  f'pseudo_traj_{dum}',    path=args.data_dir)
pkl_save(y_meta,    f'pseudo_labels_{dum}',  path=args.data_dir)
pkl_save(w_pseudo,  f'pseudo_weights_{dum}', path=args.data_dir)
pkl_save(evr_folds, f'pseudo_evr_{dum}',     path=args.data_dir)
print('Results saved to', args.data_dir)
print('dum:', dum)
