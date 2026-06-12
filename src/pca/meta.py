"""
Meta-mouse cross-validated PCA on a shared, padded neuron space.

Pipeline
--------
1. All mice are zero-padded into a global (n_trials, n_neurons_total, n_time)
   matrix, with per-mouse neuron blocks tracked by mouse_slices.
2. A single joint PCA is run on the pooled matrix.  Per-mouse gains
   (1/sqrt(N_m)) ensure equal contribution from each mouse regardless of
   population size.
3. Each fold fits PCA on the training set and aligns its loadings to the
   reference (all-clean-data) loadings via Procrustes before projecting test.
4. Perturbed trials are projected using the all-clean reference fit.

Key difference from singlePCA
------------------------------
singlePCA aligns SCORE SPACES across mice (different neuron axes).
metaPCA aligns LOADING SPACES across CV folds (single neuron axis, fold noise).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm import tqdm

from src.pca.procrustes import (
    apply_rotation_to_loadings,
    apply_rotation_to_scores,
    compute_gain_vector,
    procrustes_rotation,
)
from src.pca.scalers import StandardScaler
from src.pca.utils import cv_avg_cond, get_levels


# ── internal helpers ──────────────────────────────────────────────────────────

def _fit_space(X_train, y_train, pca, factors, levels, gain_vec,
               X_train_full=None, bl_bins=None):
    """
    Fit PCA mean and components on condition-averaged data in weighted space.

    X_train      : (n_train, n_neurons, n_fit_time) — used for condition avg + PCA
    X_train_full : (n_train, n_neurons, n_full_time) — used for mean computation
                   only; falls back to X_train when None
    bl_bins      : slice or array — bins within X_train_full used to compute the
                   centering mean.  Omitting bl_bins uses the all-time mean of
                   X_train_full (or X_train when X_train_full is None).

    Returns mean_w (1, n_neurons), W (n_comp, n_neurons), evr (n_comp,)
    """
    X_avg, _ = cv_avg_cond(X_train, y_train, factors, levels, drop_missing=True)
    X_for_mean = X_train_full if X_train_full is not None else X_train
    if bl_bins is not None:
        X_bl = X_for_mean[:, :, bl_bins]
        X_flat = X_bl.transpose(0, 2, 1).reshape(-1, X_for_mean.shape[1])
    else:
        X_flat = X_for_mean.transpose(0, 2, 1).reshape(-1, X_for_mean.shape[1])
    mean_w = (X_flat * gain_vec[None, :]).mean(axis=0, keepdims=True)

    X_flat_avg = X_avg.transpose(0, 2, 1).reshape(-1, X_avg.shape[1])
    pca.fit(X_flat_avg * gain_vec[None, :] - mean_w)

    return mean_w, pca.components_.copy(), pca.explained_variance_ratio_.copy()


def _project(X_trials, pca, mean_w, gain_vec):
    """
    Project X_trials into the fitted PC space.

    X_trials : (n_trials, n_neurons, n_time)
    Returns Z : (n_trials, n_time, n_comp)
    """
    X_flat = X_trials.transpose(0, 2, 1).reshape(-1, X_trials.shape[1])
    Z = pca.transform(X_flat * gain_vec[None, :] - mean_w)
    return Z.reshape(X_trials.shape[0], X_trials.shape[-1], -1)


# ── public API ────────────────────────────────────────────────────────────────

def pc_mouse_energy(W, mouse_slices):
    """
    Fraction of each PC's loading energy attributed to each mouse.

    W : (n_comp, n_neurons_total)
    Returns DataFrame  rows=PCs, cols=mice
    """
    E = {m: np.sum(W[:, sl] ** 2, axis=1) for m, sl in mouse_slices.items()}
    return pd.DataFrame(E)


def cv_pca_meta(
    X, y, pca, folds, factors,
    epoch=None,
    bl_bins=None,
    levels=None,
    learning='Expert',
    scale=0,
    if_scale=0,
    scale_test=0,
    mouse_slices=None,
    mouse_gain_mode="equal_mouse",
    show_pbar=True,
):
    """
    Cross-validated PCA on the pooled, padded multi-mouse matrix.

    Parameters
    ----------
    X : (n_trials, n_neurons_total, n_time)  — from build_padded_X
    y : DataFrame aligned with X; must have columns laser, learning, performance,
        odor_pair, mouse, day, tasks
    pca : sklearn PCA estimator (cloned per fold)
    folds : CV splitter
    factors : str or list[str]  — condition columns for averaging and anchors
    epoch : array-like of int or None  — time slice used for fitting
    bl_bins : slice or array-like or None  — bins used to compute the centering
        mean.  When provided, each neuron is zeroed to its pre-trial baseline
        activity rather than its all-time mean.  Removes centering bias that
        arises when task conditions have unequal motor activity (e.g. DualGo
        lick responses inflating the lick-neuron mean).
    levels : dict or None  — {factor: values}; computed from y if None
    learning : str  — learning stage used to define clean trials
    scale : int (bool)  — apply StandardScaler before fitting
    if_scale : int (bool)  — passed to StandardScaler (0=center only, 1=z-score)
    scale_test : int (bool)  — apply the same scaler to test trials
    mouse_slices : dict {mouse: slice}  — required when mouse_gain_mode != None
    mouse_gain_mode : 'equal_mouse' | 'equal_neuron' | None
    show_pbar : bool

    Returns
    -------
    Z_all    : (n_trials_all, n_time, n_comp)
    y_all    : DataFrame aligned with Z_all
    W_mean   : (n_comp, n_neurons_total)  — fold loadings averaged after alignment
    W_ref    : (n_comp, n_neurons_total)  — reference loadings from all-clean fit
    evr_folds : (n_folds, n_comp)
    """
    if mouse_gain_mode is not None and mouse_slices is None:
        raise ValueError("mouse_slices is required when mouse_gain_mode is not None")

    n_neurons = X.shape[1]
    gain_vec = compute_gain_vector(n_neurons, mouse_slices, mode=mouse_gain_mode)

    if isinstance(factors, str):
        factors = [factors]

    # ── clean / perturbed split ───────────────────────────────────────────────
    m_clean = (y.laser == 0) & (y.learning == learning) & (y.performance == 1)
    Xc, yc = X[m_clean], y.loc[m_clean].reset_index(drop=True)

    if scale:
        scaler = StandardScaler(if_scale=if_scale)
        scaler.fit(Xc)
        Xc_sc = scaler.transform(Xc)
    else:
        Xc_sc = Xc

    if levels is None:
        levels = get_levels(yc, factors)

    # ── reference fit on all clean data ──────────────────────────────────────
    Xc_fit = Xc_sc if epoch is None else Xc_sc[..., epoch]
    # bl_bins is relative to Xc_fit's time axis when epoch is used; keep full
    # Xc_sc time axis when epoch=None so bl_bins can address the full timeline.
    bl_bins_fit = bl_bins if epoch is None else None
    pca_ref = clone(pca)
    mean_w_ref, W_ref, _ = _fit_space(Xc_fit, yc, pca_ref, factors, levels,
                                       gain_vec, bl_bins=bl_bins_fit)

    # ── project perturbed trials using reference fit ──────────────────────────
    m_pert = ~m_clean
    Zp, yp = None, None
    if np.any(m_pert):
        Xp = X[m_pert]
        if scale and scale_test:
            Xp = scaler.transform(Xp)
        yp = y.loc[m_pert].reset_index(drop=True)
        Zp = _project(Xp, pca_ref, mean_w_ref, gain_vec)

    # ── CV loop ───────────────────────────────────────────────────────────────
    strata = (
        yc["odor_pair"].astype(str)
        + "_" + yc["mouse"].astype(str)
        + "_" + yc["day"].astype(str)
        + "_" + yc["tasks"].astype(str)
    ).to_numpy()

    splitter = folds.split(Xc_sc, strata)
    if show_pbar:
        splitter = tqdm(splitter,
                        total=folds.get_n_splits(Xc_sc, strata),
                        desc="cv_pca_meta")

    Z_folds, y_folds, W_folds, evr_folds = [], [], [], []

    for train_idx, test_idx in splitter:
        X_train = Xc[train_idx]
        y_train = yc.iloc[train_idx].reset_index(drop=True)
        X_test = Xc[test_idx]
        y_test = yc.iloc[test_idx].reset_index(drop=True)

        if scale:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            if scale_test:
                X_test = scaler.transform(X_test)

        X_train_fit = X_train if epoch is None else X_train[..., epoch]

        pca_fold = clone(pca)
        mean_w, W_fold, evr = _fit_space(X_train_fit, y_train, pca_fold, factors, levels, gain_vec)
        evr_folds.append(evr)

        Z_test = _project(X_test, pca_fold, mean_w, gain_vec)

        R = procrustes_rotation(W_fold, W_ref)
        Z_test = apply_rotation_to_scores(Z_test, R)
        W_aligned = apply_rotation_to_loadings(W_fold, R)

        Z_folds.append(Z_test)
        y_folds.append(y_test)
        W_folds.append(W_aligned)

    Z_clean = np.concatenate(Z_folds, axis=0)
    y_clean = pd.concat(y_folds, axis=0, ignore_index=True)
    W_mean = np.mean(np.stack(W_folds, axis=0), axis=0)
    evr_folds = np.asarray(evr_folds)

    if Zp is not None:
        Z_all = np.concatenate([Z_clean, Zp], axis=0)
        y_all = pd.concat([y_clean, yp], axis=0, ignore_index=True)
    else:
        Z_all, y_all = Z_clean, y_clean

    return Z_all, y_all, W_mean, W_ref, evr_folds
