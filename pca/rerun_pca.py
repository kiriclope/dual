"""
Extracted from singlePCA.org — runs cv_pca per mouse, aligns across mice,
and saves single_traj/labels/weights/evr pkl files to ../data/pca/.
"""
import sys
sys.path.insert(0, '/home/leon/dual_task/dual_data/')
sys.path.insert(0, '/home/leon/dual/')

import os
os.chdir('/home/leon/dual/pca')

import warnings
warnings.filterwarnings("ignore")

import pickle as pkl
import numpy as np
import pandas as pd
import itertools
from time import perf_counter

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from utils.options import set_options

# ── helpers ───────────────────────────────────────────────────────────────────

class StandardScaler:
    def __init__(self, axis=0, if_scale=0):
        self.axis = axis
        self.center_ = None
        self.scale_ = None
        self.if_scale_ = if_scale

    def fit(self, X):
        self.center_ = np.nanmean(X, axis=self.axis, keepdims=True)
        self.scale_ = np.nanstd(X, axis=(self.axis, -1), keepdims=True)
        lo, hi = np.nanpercentile(self.scale_.ravel(), [5, 95])
        self.scale_ = np.clip(self.scale_, lo, hi)
        return self

    def transform(self, X):
        if self.if_scale_:
            return (X - self.center_) / self.scale_
        return (X - self.center_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def pkl_save(obj, name, path='.'):
    os.makedirs(path, exist_ok=True)
    dst = path + '/' + name + '.pkl'
    print('saving to', dst)
    pkl.dump(obj, open(dst, 'wb'))


def pkl_load(name, path='.'):
    src = path + '/' + name + '.pkl'
    print('loading from', src)
    return pkl.load(open(src, 'rb'))


def get_levels(y, factors):
    if isinstance(factors, str):
        factors = [factors]
    levels = {}
    for f in factors:
        s = y[f]
        if np.issubdtype(s.dtype, np.number):
            levels[f] = np.sort(s.dropna().unique())
        else:
            levels[f] = np.sort(s.astype(str).dropna().unique())
    return levels


def _combo_mask(y, factors, combo):
    m = np.ones(len(y), dtype=bool)
    for f, v in zip(factors, combo):
        s = y[f]
        if np.issubdtype(s.dtype, np.number):
            m &= (s.values == v)
        else:
            m &= (s.astype(str).values == str(v))
    return m


def cv_avg_cond(X, y, factors, levels, drop_missing=False):
    if isinstance(factors, str):
        factors = [factors]
    combos = list(itertools.product(*[levels[f] for f in factors]))
    X_avg = []
    for combo in combos:
        idx = _combo_mask(y, factors, combo)
        if not idx.any():
            if drop_missing:
                continue
            raise ValueError(f"Missing combo: {dict(zip(factors, combo))}")
        X_avg.append(np.nanmean(X[idx], axis=0))
    return np.stack(X_avg, axis=0), combos


def anchors_from_Z(Z, y, factors, levels, drop_missing=False):
    if isinstance(factors, str):
        factors = [factors]
    combos = list(itertools.product(*[levels[f] for f in factors]))
    A = []
    for combo in combos:
        idx = _combo_mask(y, factors, combo)
        if not idx.any():
            if drop_missing:
                continue
            raise ValueError(f"Missing combo: {dict(zip(factors, combo))}")
        A.append(np.nanmean(Z[idx], axis=0))
    A = np.stack(A, axis=0)
    return A.reshape(-1, Z.shape[-1])


def cv_pca(X, y, pca, folds, factors,
           epoch=None, scale=None, stage='Expert', correct=True, context=None,
           group_col=None, show_pbar=True, center_on='trialtime',
           align_folds=False):
    if isinstance(factors, str):
        factors = [factors]

    def flat_trial_time(A):
        return A.transpose(0, 2, 1).reshape(-1, A.shape[1])

    idx_correct = (y.performance == 1) & ((y.odr_perf == 1) | (y.tasks == 'DPA')) if correct else True

    if context is not None:
        if context == 'Dual':
            idx_context = (y.tasks != 'DPA')
        else:
            idx_context = (y.tasks == context)
    else:
        idx_context = True

    m_clean = (y.laser == 0) & (y.learning == stage) & idx_context & idx_correct
    m_pert  = ~m_clean

    Xc = X[m_clean]
    yc = y.loc[m_clean].reset_index(drop=True)
    print('Xc', Xc.shape, 'yc', yc.shape)

    X_pert = X[m_pert]
    y_pert = y.loc[m_pert].reset_index(drop=True)
    print('X_pert', X_pert.shape, 'y_pert', y_pert.shape)

    strata = (
        yc['odor_pair'].astype(str) + '_'
        + yc['day'].astype(str) + '_'
        + yc['tasks'].astype(str)
    ).values

    levels = get_levels(yc, factors)

    combos = list(itertools.product(*[levels[f] for f in factors]))
    for combo in combos:
        if not _combo_mask(yc, factors, combo).any():
            raise ValueError(
                f"Combo {dict(zip(factors, combo))} missing globally; "
                "Procrustes anchors will be inconsistent across folds."
            )

    it = folds.split(Xc, strata, groups=None)
    if show_pbar:
        name = y.mouse.unique()[0] if 'mouse' in y.columns else 'X'
        it = tqdm(it, total=folds.get_n_splits(Xc, strata), desc=str(name))

    trial_Z_accum = {}
    Z_cross = []
    w_folds, evr_folds = [], []
    A_ref = None

    for fold_i, (train, test) in enumerate(it):
        X_train, y_train = Xc[train], yc.iloc[train].reset_index(drop=True)
        X_test,  y_test  = Xc[test],  yc.iloc[test].reset_index(drop=True)

        if scale is not None:
            scale_fold = StandardScaler(if_scale=1)
            X_train = scale_fold.fit_transform(X_train)
            X_test  = scale_fold.transform(X_test)
            X_pert_scaled = scale_fold.transform(X_pert)
        else:
            X_pert_scaled = X_pert

        X_train_epoch = X_train if epoch is None else X_train[..., epoch]

        X_avg_train, _ = cv_avg_cond(X_train_epoch, y_train, factors, levels, drop_missing=False)
        X_avg_flat = flat_trial_time(X_avg_train)

        if center_on == 'trialtime':
            X_mean = np.nanmean(flat_trial_time(X_train_epoch), axis=0, keepdims=True)
        else:
            X_mean = np.nanmean(X_avg_flat, axis=0, keepdims=True)

        pca_fold = clone(pca)
        pca_fold.fit(X_avg_flat - X_mean)
        w_folds.append(pca_fold.components_)
        evr_folds.append(pca_fold.explained_variance_ratio_)

        Z_test = pca_fold.transform(flat_trial_time(X_test) - X_mean)
        Z_test = Z_test.reshape(X_test.shape[0], X_test.shape[-1], -1)

        Z_pert_fold = pca_fold.transform(flat_trial_time(X_pert_scaled) - X_mean)
        Z_pert_fold = Z_pert_fold.reshape(X_pert_scaled.shape[0], X_pert_scaled.shape[-1], -1)

        if align_folds:
            Z_train = pca_fold.transform(flat_trial_time(X_train_epoch) - X_mean)
            Z_train = Z_train.reshape(X_train_epoch.shape[0], X_train_epoch.shape[-1], -1)
            A_curr = anchors_from_Z(Z_train, y_train, factors, levels, drop_missing=False)
            if A_ref is None:
                A_ref = A_curr
            else:
                R, _ = orthogonal_procrustes(A_curr, A_ref)
                Z_test = Z_test @ R
                Z_pert_fold = Z_pert_fold @ R

        for i, orig_idx in enumerate(test):
            trial_Z_accum.setdefault(orig_idx, []).append(Z_test[i])

        Z_cross.append(Z_pert_fold)

    sorted_indices = sorted(trial_Z_accum.keys())
    Z_clean = np.stack([np.mean(trial_Z_accum[idx], axis=0) for idx in sorted_indices])
    y_clean = yc.iloc[sorted_indices].reset_index(drop=True)
    print(f'Z_clean: {Z_clean.shape}, y_clean: {y_clean.shape}')

    Zp = np.mean(np.stack(Z_cross, axis=0), axis=0)
    print(f'Z_pert: {Zp.shape}, y_pert: {y_pert.shape}')

    Z_all = np.concatenate([Z_clean, Zp], axis=0)
    y_all = pd.concat([y_clean, y_pert], ignore_index=True)

    return Z_all, y_all, np.array(w_folds), np.array(evr_folds)


# ── parameters ────────────────────────────────────────────────────────────────

SCALE = None

mice = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
        'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
tasks = ['Dual']

kwargs = {
    'mice': mice, 'tasks': tasks,
    'mouse': mice[0], 'laser': 0,
    'trials': '', 'reload': 0, 'data_type': 'dF',
    'prescreen': None, 'pval': 0.05,
    'preprocess': None, 'scaler_BL': 'center_BL',
    'avg_noise': False, 'unit_var_BL': False,
    'random_state': None, 'T_WINDOW': 0.0,
    'l1_ratio': 0.95,
    'n_comp': 3, 'pca': 'pca',
    'scaler': None,
    'bootstrap': 1, 'n_boots': 128,
    'n_splits': 5, 'n_repeats': 10,
    'class_weight': 0, 'multilabel': 0,
    'mne_estimator': 'generalizing',
    'n_jobs': 64,
}
kwargs['days'] = ['first', 'last']
options = set_options(**kwargs)
options['cv_B'] = False

# ── load X_all, y_all ─────────────────────────────────────────────────────────

dum_raw = ''
if SCALE is not None:
    dum_raw = SCALE

X_all = pkl_load('X_all_nan_%s' % dum_raw, path='/home/leon/dual_task/dual_data/data/pca')
y_all = pkl_load('y_all_nan_%s' % dum_raw, path='/home/leon/dual_task/dual_data/data/pca')

y_all['sample']     = y_all.sample_odor
y_all['test']       = y_all.test_odor
y_all['distractor'] = y_all.dist_odor

print(X_all.shape, y_all.shape)

# ── model parameters ──────────────────────────────────────────────────────────

dum = 'pca'

epoch = 'TEST'
if epoch is not None:
    dum  += '_' + epoch
    epoch = options['bins_' + epoch]

stage = 'Expert'
options['learning'] = stage
dum += '_' + stage

scale = 'standard'
dum += '_' + scale

context = None

n_splits, n_repeats = -1, 1
if n_splits == -1:
    folds = LeaveOneOut()
    dum += '_loo'
else:
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

correct = True
if correct:
    dum += '_correct'

factors = ['odor_pair']
for factor in factors:
    dum += '_' + factor

print('dum:', dum)

n_comp = 10
pca_est = PCA(n_components=n_comp, svd_solver='randomized')

# ── run cv_pca per mouse ──────────────────────────────────────────────────────

t0 = perf_counter()
X_mice, y_mice, w_mice, evr_mice = [], [], [], []

for mouse in options['mice']:
    idx = (y_all['mouse'] == mouse)
    X_pca = X_all[idx]
    y_pca = y_all.loc[idx].reset_index(drop=True)

    valid_neurons = ~np.all(np.isnan(X_pca), axis=(0, 2))
    X_pca = X_pca[:, valid_neurons, :]

    Z_mouse, y_mouse, W_mouse, evr_mouse = cv_pca(
        X_pca, y_pca, pca_est, folds, factors,
        epoch=epoch, scale=scale, stage=stage,
        context=context, correct=correct,
    )

    X_mice.append(Z_mouse)
    y_mice.append(y_mouse)
    w_mice.append(W_mouse)
    evr_mice.append(evr_mouse)

h, m, s = int((perf_counter()-t0)//3600), int(((perf_counter()-t0)%3600)//60), int((perf_counter()-t0)%60)
print(f'cv_pca done in {h}h {m}m {s}s')

# ── align across mice with Procrustes ─────────────────────────────────────────

y_pool  = pd.concat(y_mice, ignore_index=True)
levels  = get_levels(y_pool, factors)
ref_idx = 7   # ACCM03 as reference

A_ref = anchors_from_Z(X_mice[ref_idx], y_mice[ref_idx], factors, levels, drop_missing=False)

X_mice_aligned = []
for Z_mouse, y_mouse in zip(X_mice, y_mice):
    A   = anchors_from_Z(Z_mouse, y_mouse, factors, levels, drop_missing=False)
    R, _ = orthogonal_procrustes(A, A_ref)
    X_mice_aligned.append(Z_mouse @ R)

X_aligned = np.concatenate(X_mice_aligned, axis=0)
y_aligned = pd.concat(y_mice, ignore_index=True)

X_single = np.swapaxes(X_aligned, 1, 2)   # (trials, n_comp, time)
y_single = y_aligned
print('X_single:', X_single.shape)

# ── fix PC2 orientation (lick-positive) ───────────────────────────────────────

bins_choice = options['bins_CHOICE']
mask_lick   = (y_single.laser == 0) & (y_single.performance == 1) & (y_single.choice == 1)
mask_nolick = (y_single.laser == 0) & (y_single.performance == 1) & (y_single.choice == 0)

m_lick   = np.nanmean(X_single[mask_lick][:, 1, bins_choice])
m_nolick = np.nanmean(X_single[mask_nolick][:, 1, bins_choice])

if m_lick < m_nolick:
    print('Flipping PC2')
    X_single[:, 1, :] *= -1
else:
    print('PC2 orientation OK')

# ── weights and EVR ───────────────────────────────────────────────────────────

evr_list  = [evr.mean(0) for evr in evr_mice]
evr_single = np.vstack(evr_list)

w_list   = [w.mean(0) for w in w_mice]
w_single = np.hstack(w_list) * 100
print('w_single:', w_single.shape, 'evr_single:', evr_single.shape)

# ── save ──────────────────────────────────────────────────────────────────────

OUT = '../data/pca'
pkl_save(X_single,  'single_traj_'     + dum, path=OUT)
pkl_save(y_single,  'single_labels_'   + dum, path=OUT)
pkl_save(w_single,  'single_weights_'  + dum, path=OUT)
pkl_save(evr_single,'single_evr_'      + dum, path=OUT)
print('all done')
