"""
Churchland/Shenoy pseudo-population PCA across mice — clean replacement for
run_meta_all's cv_pca_meta basis.

Key differences from run_meta_all.py
------------------------------------
* basis fit on WITHIN-mouse condition averages (no zero-padding into the mean)
  → removes the trial-count dilution bias of cv_pca_meta
* per-neuron z-score (soft-normalised) so high-variance cells don't dominate
* richer condition space (odor_pair x tasks x choice) so the basis is not
  rank-deficient (cv_pca_meta defined 10 PCs from 4 odor_pair conditions x
  6 TEST bins ≈ 4 independent patterns)
* basis fit on the DELAY epoch (matches the delay-memory hypothesis) — change
  EPOCH below to revert to TEST

Results → results/pseudo_*.pkl
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, '/home/leon/dual_task/dual_data/')
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from time import perf_counter
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold

from src.common.options import set_options
from src.pca.io import pkl_save, pkl_load
from src.pca.pseudo import cv_pca_pseudo, pseudo_population_pca

RESULTS = 'results'
DATA    = '/home/leon/dual_task/dual_data/data/pca'

# ── parameters ────────────────────────────────────────────────────────────────

mice = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
        'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']

kwargs = {
    'mice': mice, 'tasks': ['Dual'], 'mouse': mice[0], 'laser': 0,
    'trials': '', 'reload': 0, 'data_type': 'dF', 'prescreen': None, 'pval': 0.05,
    'preprocess': None, 'scaler_BL': 'center_BL', 'avg_noise': False,
    'unit_var_BL': False, 'random_state': None, 'T_WINDOW': 0.0, 'l1_ratio': 0.95,
    'n_comp': 3, 'pca': 'pca', 'scaler': None, 'bootstrap': 1, 'n_boots': 128,
    'n_splits': 5, 'n_repeats': 10, 'class_weight': 0, 'multilabel': 0,
    'mne_estimator': 'generalizing', 'n_jobs': 64,
}
kwargs['days'] = ['first', 'last']
options = set_options(**kwargs)

stage     = 'Expert'
factors   = ['odor_pair', 'tasks', 'choice']   # rich condition space
EPOCH     = 'DELAY'                              # 'DELAY' | 'TEST' | ...
epoch     = options['bins_' + EPOCH]
bl_bins   = slice(0, 12)                         # pre-stimulus baseline (t 0-2s)
norm      = 'zscore'
n_comp    = 6
n_splits, n_repeats = 5, 10
DUM       = f'pseudo_{EPOCH}_{stage}_{norm}_{n_splits}x{n_repeats}'

folds   = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
pca_est = PCA(n_components=n_comp, svd_solver='auto')

# ── load data ─────────────────────────────────────────────────────────────────

print('Loading data ...')
X_all        = pkl_load('X_all_center', path=DATA)
y_all        = pkl_load('y_all_center', path=DATA)
mouse_slices = pkl_load('mouse_slices', path=DATA)
y_all['sample'] = y_all.sample_odor
y_all['test']   = y_all.test_odor
print(X_all.shape, y_all.shape)

# ── run ───────────────────────────────────────────────────────────────────────

t0 = perf_counter()
Z_all, y_meta, W_ref, Z_cond, evr_folds = cv_pca_pseudo(
    X_all, y_all, pca_est, folds, factors,
    epoch=epoch, bl_bins=bl_bins, learning=stage,
    norm=norm, mouse_slices=mouse_slices,
)
print(f'cv_pca_pseudo done in {(perf_counter()-t0)/60:.1f} min')

X_meta = np.swapaxes(Z_all, 1, 2).astype(float)   # (trials, n_comp, time)
w_meta = W_ref * 100
print('X_meta:', X_meta.shape, ' W_ref:', W_ref.shape, ' evr_folds:', evr_folds.shape)
print('reference EVR (%):', np.round(evr_folds.mean(0) * 100, 1))

# ── fix PC2 sign (lick-positive in CHOICE epoch) ──────────────────────────────

bins_choice = options['bins_CHOICE']
m_lick   = (y_meta.laser == 0) & (y_meta.learning == stage) & (y_meta.performance == 1) & (y_meta.choice == 1)
m_nolick = (y_meta.laser == 0) & (y_meta.learning == stage) & (y_meta.performance == 1) & (y_meta.choice == 0)
if np.nanmean(X_meta[m_lick][:, 1, bins_choice]) < np.nanmean(X_meta[m_nolick][:, 1, bins_choice]):
    print('Flipping PC2')
    X_meta[:, 1, :] *= -1
    w_meta[1, :]    *= -1
else:
    print('PC2 orientation OK')

# ── save ──────────────────────────────────────────────────────────────────────

pkl_save(X_meta,    f'meta_traj_{DUM}',    path=RESULTS)
pkl_save(y_meta,    f'meta_labels_{DUM}',  path=RESULTS)
pkl_save(w_meta,    f'meta_weights_{DUM}', path=RESULTS)
pkl_save(evr_folds, f'meta_evr_{DUM}',     path=RESULTS)
print('Results saved to', RESULTS, 'with tag', DUM)
print('done')
