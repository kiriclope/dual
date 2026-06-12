"""
Per-mouse cross-validated PCA with cross-mouse Procrustes alignment.

Pipeline
--------
1. For each mouse: run cv_pca_single → (Z_mouse, y_mouse, w_folds, evr_folds)
2. Align all mice's Z to a common reference frame via anchors_from_Z + Procrustes
3. Concatenate → (X_single, y_single) saved as single_traj_*.pkl

Key design choices
------------------
- PCA is fit on condition-averaged activity (not raw trials) to focus the
  subspace on task-related variance.
- Perturbed trials (laser != 0) are projected using each fold's fit so they
  land in the same space as clean trials, then averaged over folds.
- Cross-fold alignment uses Procrustes on score-space anchors (condition means
  of Z), not on loadings, because mice have different neuron sets.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.base import clone
from tqdm import tqdm

from src.pca.scalers import StandardScaler
from src.pca.utils import _combo_mask, anchors_from_Z, cv_avg_cond, get_levels


def cv_pca_single(
    X, y, pca, folds, factors,
    epoch=None,
    scale=None,
    stage='Expert',
    correct=True,
    context=None,
    show_pbar=True,
    center_on='trialtime',
    align_folds=False,
):
    """
    Cross-validated PCA for a single mouse.

    Parameters
    ----------
    X : (n_trials, n_neurons, n_time)
    y : DataFrame aligned with X; must have columns laser, learning, performance,
        odr_perf, tasks, odor_pair, day
    pca : sklearn PCA estimator (cloned per fold)
    folds : CV splitter (e.g. LeaveOneOut, RepeatedStratifiedKFold)
    factors : str or list[str]  — condition columns used for averaging and anchors
    epoch : array-like of int or None
        Time indices used for fitting; None = full trial time.
    scale : 'standard' or None
        If 'standard', StandardScaler(if_scale=1) is fit on train set each fold.
    stage : str  — learning stage to select clean trials ('Expert', 'Naive', …)
    correct : bool  — restrict clean trials to performance==1 (and odr_perf for GNG)
    context : str or None  — restrict to a task context (e.g. 'DPA', 'Dual')
    center_on : 'trialtime' | 'cond_avg'
        Whether the PCA mean is computed over all trial×time points or
        over condition averages only.
    align_folds : bool
        Procrustes-align each fold's score space to the first fold's anchors.
        Useful when repeated CV would otherwise flip PC signs across folds.

    Returns
    -------
    Z_all   : (n_trials_all, n_time, n_comp)
    y_all   : DataFrame aligned with Z_all
    w_folds : (n_folds, n_comp, n_neurons)
    evr_folds : (n_folds, n_comp)
    """
    if isinstance(factors, str):
        factors = [factors]

    def flat(A):
        return A.transpose(0, 2, 1).reshape(-1, A.shape[1])

    # ── trial selection ───────────────────────────────────────────────────────
    if correct:
        idx_correct = (y.performance == 1) & ((y.odr_perf == 1) | (y.tasks == 'DPA'))
    else:
        idx_correct = pd.Series(True, index=y.index)

    if context is None:
        idx_context = pd.Series(True, index=y.index)
    elif context == 'Dual':
        idx_context = y.tasks != 'DPA'
    else:
        idx_context = y.tasks == context

    m_clean = (y.laser == 0) & (y.learning == stage) & idx_context & idx_correct
    m_pert = ~m_clean

    Xc, yc = X[m_clean], y.loc[m_clean].reset_index(drop=True)
    X_pert, y_pert = X[m_pert], y.loc[m_pert].reset_index(drop=True)
    print(f'Xc {Xc.shape}  X_pert {X_pert.shape}')

    strata = (
        yc['odor_pair'].astype(str) + '_'
        + yc['day'].astype(str) + '_'
        + yc['tasks'].astype(str)
    ).values

    levels = get_levels(yc, factors)

    # verify all combos present globally (Procrustes consistency)
    for combo in itertools.product(*[levels[f] for f in factors]):
        if not _combo_mask(yc, factors, combo).any():
            raise ValueError(
                f"Combo {dict(zip(factors, combo))} absent from clean trials; "
                "Procrustes anchors would be inconsistent across folds."
            )

    # ── CV loop ───────────────────────────────────────────────────────────────
    it = folds.split(Xc, strata)
    if show_pbar:
        name = y.mouse.unique()[0] if 'mouse' in y.columns else 'X'
        it = tqdm(it, total=folds.get_n_splits(Xc, strata), desc=str(name))

    trial_Z_accum = {}   # orig_idx → list of Z_test[i] arrays (for repeated CV)
    Z_pert_folds = []
    w_folds, evr_folds = [], []
    A_ref = None

    for train, test in it:
        X_train, y_train = Xc[train], yc.iloc[train].reset_index(drop=True)
        X_test, y_test = Xc[test], yc.iloc[test].reset_index(drop=True)

        if scale is not None:
            sc = StandardScaler(if_scale=1)
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            X_pert_sc = sc.transform(X_pert)
        else:
            X_pert_sc = X_pert

        X_train_ep = X_train if epoch is None else X_train[..., epoch]

        X_avg_train, _ = cv_avg_cond(X_train_ep, y_train, factors, levels)
        X_avg_flat = flat(X_avg_train)

        if center_on == 'trialtime':
            X_mean = np.nanmean(flat(X_train_ep), axis=0, keepdims=True)
        else:
            X_mean = np.nanmean(X_avg_flat, axis=0, keepdims=True)

        pca_fold = clone(pca)
        pca_fold.fit(X_avg_flat - X_mean)
        w_folds.append(pca_fold.components_)
        evr_folds.append(pca_fold.explained_variance_ratio_)

        Z_test = pca_fold.transform(flat(X_test) - X_mean)
        Z_test = Z_test.reshape(X_test.shape[0], X_test.shape[-1], -1)

        Z_pert = pca_fold.transform(flat(X_pert_sc) - X_mean)
        Z_pert = Z_pert.reshape(X_pert_sc.shape[0], X_pert_sc.shape[-1], -1)

        if align_folds:
            Z_tr = pca_fold.transform(flat(X_train_ep) - X_mean)
            Z_tr = Z_tr.reshape(X_train_ep.shape[0], X_train_ep.shape[-1], -1)
            A_curr = anchors_from_Z(Z_tr, y_train, factors, levels)
            if A_ref is None:
                A_ref = A_curr
            else:
                R, _ = orthogonal_procrustes(A_curr, A_ref)
                Z_test = Z_test @ R
                Z_pert = Z_pert @ R

        for i, orig_idx in enumerate(test):
            trial_Z_accum.setdefault(orig_idx, []).append(Z_test[i])
        Z_pert_folds.append(Z_pert)

    # ── aggregate ─────────────────────────────────────────────────────────────
    sorted_idx = sorted(trial_Z_accum)
    Z_clean = np.stack([np.mean(trial_Z_accum[i], axis=0) for i in sorted_idx])
    y_clean = yc.iloc[sorted_idx].reset_index(drop=True)

    Zp = np.mean(np.stack(Z_pert_folds, axis=0), axis=0)
    print(f'Z_clean {Z_clean.shape}  Z_pert {Zp.shape}')

    Z_all = np.concatenate([Z_clean, Zp], axis=0)
    y_all = pd.concat([y_clean, y_pert], ignore_index=True)

    return Z_all, y_all, np.array(w_folds), np.array(evr_folds)


def align_mice(X_mice, y_mice, factors, ref_idx=0):
    """
    Procrustes-align per-mouse score arrays to a common reference.

    X_mice : list of (n_trials_m, n_time, n_comp)
    y_mice : list of DataFrames
    ref_idx : index of the reference mouse

    Returns
    -------
    X_aligned : (sum_trials, n_time, n_comp)
    y_aligned : DataFrame
    """
    y_pool = pd.concat(y_mice, ignore_index=True)
    levels = get_levels(y_pool, factors)

    A_ref = anchors_from_Z(X_mice[ref_idx], y_mice[ref_idx], factors, levels)

    aligned = []
    for Z, y in zip(X_mice, y_mice):
        A = anchors_from_Z(Z, y, factors, levels)
        R, _ = orthogonal_procrustes(A, A_ref)
        aligned.append(Z @ R)

    X_aligned = np.concatenate(aligned, axis=0)
    y_aligned = pd.concat(y_mice, ignore_index=True)
    return X_aligned, y_aligned
