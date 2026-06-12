"""
Churchland/Shenoy-style pseudo-population PCA across mice.

Why this exists
---------------
``meta.cv_pca_meta`` zero-pads every mouse into a shared neuron axis and then
condition-averages with ``np.nanmean`` over the *pooled* trial matrix.  Because
the padding is stored as 0.0 (not NaN), a neuron's condition mean comes out as

    sum_over_mouse_m_trials(x) / N_total_trials_in_condition
        = mean_m(x) * (N_m / N_total)

i.e. each mouse's response is silently attenuated by its share of trials.  With
unequal trial counts (here 768-1152) this conflates trial count with response
amplitude, and the PCA basis is biased toward whichever mice contribute more
trials / more neurons.

The standard cross-animal approach (Churchland, Cunningham, Shenoy 2012) avoids
this entirely:

1. Average **within each mouse** per condition  →  per-mouse PSTH tensor.
2. Soft-normalise / z-score **per neuron** so high-variance cells don't dominate.
3. **Concatenate along the neuron axis** (no zero padding) into one pseudo-
   population  (n_conditions, n_neurons_total, n_time).
4. PCA on the condition-averaged pseudo-population.

Single-trial scores (for trajectory plots with trial SEM, decoding, etc.) are
obtained by projecting each mouse's trials through *its own* loading sub-block.
This is mathematically identical to padding that trial with zeros and projecting
through the full loading vector, so it stays compatible with the existing
downstream code — only the *basis* changes (and is now unbiased).

Public API
----------
pseudo_population_pca : fit the shared basis on within-mouse condition averages
project_trials        : per-mouse single-trial projection onto a fitted basis
cv_pca_pseudo         : drop-in replacement for cv_pca_meta (same return shape)
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.pca.procrustes import apply_rotation_to_scores, procrustes_rotation
from src.pca.utils import cv_avg_cond, get_levels


# ── normalisation ──────────────────────────────────────────────────────────────

def _neuron_norm(X_cond, mode, soft, bl_bins):
    """
    Per-neuron centering mean and scale from the condition-averaged tensor.

    X_cond : (n_cond, n_neurons, n_time)
    mode   : 'zscore' | 'soft' | 'center'
    soft   : additive constant in the denominator (prevents blow-up on quiet
             neurons; Churchland soft-norm uses range + const)
    bl_bins: slice/array or None — bins used for the centering mean.  None uses
             the full (condition x time) mean.

    Returns mean_ (n_neurons,), scale_ (n_neurons,)
    """
    if bl_bins is not None:
        mean_ = np.nanmean(X_cond[:, :, bl_bins], axis=(0, 2))
    else:
        mean_ = np.nanmean(X_cond, axis=(0, 2))

    if mode == 'center':
        scale_ = np.ones(X_cond.shape[1])
    elif mode == 'zscore':
        scale_ = np.nanstd(X_cond, axis=(0, 2)) + soft
    elif mode == 'soft':
        rng = np.nanmax(X_cond, axis=(0, 2)) - np.nanmin(X_cond, axis=(0, 2))
        scale_ = rng + soft
    else:
        raise ValueError(f"Unknown norm mode: {mode!r}")
    return mean_, scale_


# ── build the pseudo-population ─────────────────────────────────────────────────

def build_pseudo_population(
    X, y, factors, mouse_slices,
    levels=None, epoch=None, bl_bins=None,
    norm='zscore', soft=1e-3,
):
    """
    Within-mouse condition averages, per-neuron normalised, concatenated along
    the neuron axis.

    X : (n_trials, n_neurons_total, n_time)  — padded matrix (mouse blocks per
        mouse_slices); only each mouse's own block is read, padding is ignored.
    y : DataFrame aligned with X; needs 'mouse' + the factor columns.
    factors : str or list[str]
    epoch : array-like of int or None — time bins the PCA basis is fit on.
    bl_bins : slice/array or None — bins for per-neuron centering (defaults to
        full window).  Given in the ORIGINAL time axis (applied before epoch).

    Returns
    -------
    P        : (n_cond, n_neurons_total, n_epoch_time)  normalised pseudo-pop
    mean_    : (n_neurons_total,)  per-neuron centering mean
    scale_   : (n_neurons_total,)  per-neuron scale
    combos   : list of condition tuples (row order of P)
    """
    if isinstance(factors, str):
        factors = [factors]
    if levels is None:
        levels = get_levels(y, factors)

    n_total = X.shape[1]
    n_time = X.shape[-1]
    combos = None
    P = np.full((np.prod([len(levels[f]) for f in factors]), n_total, n_time),
                np.nan, dtype=np.float64)
    mean_ = np.zeros(n_total)
    scale_ = np.ones(n_total)

    for mouse, sl in mouse_slices.items():
        m = (y['mouse'].astype(str) == str(mouse)).to_numpy()
        if not m.any():
            continue
        Xm = X[m][:, sl, :]                      # (n_trials_m, n_neurons_m, n_time)
        ym = y.loc[m].reset_index(drop=True)

        # within-mouse condition average; fill_nan keeps row order across mice
        Xavg_m, combos = cv_avg_cond(Xm, ym, factors, levels=levels,
                                     fill_nan=True)   # (n_cond, n_neurons_m, n_time)

        mu, sc = _neuron_norm(Xavg_m, norm, soft, bl_bins)
        mean_[sl], scale_[sl] = mu, sc
        P[:, sl, :] = (Xavg_m - mu[None, :, None]) / sc[None, :, None]

    # a mouse missing a condition leaves NaN in its block for that row; that
    # mouse simply contributes nothing (origin) to that condition after norm
    P = np.nan_to_num(P, nan=0.0)

    if epoch is not None:
        P = P[:, :, epoch]
    return P, mean_, scale_, combos


# ── fit + project ───────────────────────────────────────────────────────────────

def pseudo_population_pca(
    X, y, pca, factors, mouse_slices,
    levels=None, epoch=None, bl_bins=None,
    norm='zscore', soft=1e-3,
):
    """
    Fit the shared PCA basis on the within-mouse condition-averaged pseudo-pop.

    Returns
    -------
    W        : (n_comp, n_neurons_total)   loadings
    mean_    : (n_neurons_total,)          per-neuron centering mean
    scale_   : (n_neurons_total,)          per-neuron scale
    evr      : (n_comp,)                   explained variance ratio
    Z_cond   : (n_cond, n_epoch_time, n_comp)  condition trajectories in PC space
    combos   : list of condition tuples
    """
    P, mean_, scale_, combos = build_pseudo_population(
        X, y, factors, mouse_slices, levels=levels, epoch=epoch,
        bl_bins=bl_bins, norm=norm, soft=soft,
    )
    n_cond, n_total, n_t = P.shape
    P_flat = P.transpose(0, 2, 1).reshape(-1, n_total)   # (n_cond*n_t, n_total)

    pca = clone(pca)
    Z_flat = pca.fit_transform(P_flat)
    W = pca.components_.copy()
    evr = pca.explained_variance_ratio_.copy()
    Z_cond = Z_flat.reshape(n_cond, n_t, -1)
    return W, mean_, scale_, evr, Z_cond, combos


def project_trials(X, y, W, mean_, scale_, mouse_slices):
    """
    Project single trials onto a fitted basis, each mouse through its own block.

    Equivalent to zero-padding the trial and projecting through the full W, but
    without materialising the padding.

    Returns
    -------
    Z   : (n_trials, n_time, n_comp)   row-aligned with y_out
    y_out : DataFrame                  (mouse-grouped order)
    """
    Z_list, y_list = [], []
    for mouse, sl in mouse_slices.items():
        m = (y['mouse'].astype(str) == str(mouse)).to_numpy()
        if not m.any():
            continue
        Xm = X[m][:, sl, :].astype(np.float64)
        Xn = (Xm - mean_[sl][None, :, None]) / scale_[sl][None, :, None]
        # (trials, time, neurons_m) @ (neurons_m, comp)
        Zm = np.einsum('ntk,ck->ntc', Xn.transpose(0, 2, 1), W[:, sl])
        Z_list.append(Zm)
        y_list.append(y.loc[m])
    Z = np.concatenate(Z_list, axis=0)
    y_out = pd.concat(y_list, axis=0, ignore_index=True)
    return Z, y_out


# ── condition-independent (ramp) subspace removal ───────────────────────────────

def remove_ci_subspace(X, y, mouse_slices, q=2, factors=('odor_pair', 'tasks'),
                       learning='Expert'):
    """Project out the top-q condition-independent directions, per mouse block.

    The condition-independent component is the time-varying mean shared by every
    condition (the 'ramp'/common-mode).  Removing it as a fixed-direction
    projection — `X' = X - U (Uᵀ X)`, with `U` the top-q neuron-space directions
    of the per-mouse condition-independent marginal — strips the ramp from all
    downstream axes WITHOUT the per-timepoint demeaning that pushes e.g. DPA
    negative (a trial only loses its own component along `U`).

    X : (n_trials, n_neurons_total, n_time)  padded matrix (modified copy returned)
    q : number of CI directions to remove per mouse
    """
    if q <= 0:
        return X
    if isinstance(factors, str):
        factors = [factors]
    X = X.copy()
    m_clean = (y.laser == 0) & (y.learning == learning) & (y.performance == 1)
    for mouse, sl in mouse_slices.items():
        mm = (y['mouse'].astype(str) == str(mouse)) & m_clean
        if mm.sum() == 0:
            continue
        Xm = X[mm.to_numpy()][:, sl, :]                 # (trials_m, n_block, time)
        ym = y.loc[mm].reset_index(drop=True)
        # condition-independent marginal = mean over equally-weighted conditions
        try:
            Xavg, _ = cv_avg_cond(Xm, ym, list(factors), drop_missing=True)
            R = np.nanmean(Xavg, axis=0)                 # (n_block, time)
        except Exception:
            R = np.nanmean(Xm, axis=0)
        R = np.nan_to_num(R)
        R = R - R.mean(axis=1, keepdims=True)            # focus on the time-varying part
        if not np.any(R):
            continue
        U, _, _ = np.linalg.svd(R, full_matrices=False)  # U: (n_block, k) neuron dirs
        Uq = U[:, :q]                                    # top-q CI directions
        Xblk = X[:, sl, :]                               # (all_trials, n_block, time)
        coef = np.einsum('jq,bjt->bqt', Uq, Xblk)        # project onto CI subspace
        X[:, sl, :] = Xblk - np.einsum('jq,bqt->bjt', Uq, coef)
    return X


# ── drop-in CV wrapper (mirrors cv_pca_meta's signature/return) ──────────────────

def cv_pca_pseudo(
    X, y, pca, folds, factors,
    epoch=None, bl_bins=None, levels=None,
    learning='Expert', norm='zscore', soft=1e-3,
    mouse_slices=None, show_pbar=True, pert_foldwise=True,
):
    """
    Cross-validated pseudo-population PCA.

    The reference basis is fit on within-mouse condition averages of the CLEAN
    trials (laser off, given learning stage, correct).

    Clean trial scores are CROSS-VALIDATED: each fold refits the basis on its
    train split, projects the held-out clean trials through that fold's basis,
    and Procrustes-rotates the scores into the reference frame.  A trial's
    held-out projections are averaged across repeats, giving one out-of-sample
    row per trial (no duplication).

    Perturbed / non-clean trials never enter the clean-only fit, so they are
    already out-of-sample for every basis.  With ``pert_foldwise=True`` (default)
    they are projected through EVERY fold's basis (Procrustes-aligned) and
    averaged over folds — the same pipeline as the clean trials, so clean and
    perturbed scores live in the same averaged-fold frame.  With
    ``pert_foldwise=False`` they are projected once through the reference basis.

    This avoids both the trial-count dilution of cv_pca_meta and its 10x clean
    trial duplication.

    Returns
    -------
    Z_all    : (n_trials, n_time, n_comp)   clean rows held-out + perturbed via ref
    y_all    : DataFrame aligned with Z_all
    W_ref    : (n_comp, n_neurons_total)   reference loadings
    Z_cond   : (n_cond, n_epoch_time, n_comp)  reference condition trajectories
    evr_folds: (n_folds, n_comp)           per-fold EVR (stability diagnostic)
    """
    if mouse_slices is None:
        raise ValueError("mouse_slices is required")
    if isinstance(factors, str):
        factors = [factors]

    m_clean = (y.laser == 0) & (y.learning == learning) & (y.performance == 1)
    yc = y.loc[m_clean].reset_index(drop=True).copy()
    yc['_tid'] = np.arange(len(yc))                 # stable id to scatter held-out scores
    Xc = X[m_clean.to_numpy()]
    if levels is None:
        levels = get_levels(yc, factors)

    # ── reference fit on all clean data (defines the canonical frame) ─────────
    W_ref, mean_ref, scale_ref, _, Z_cond, _ = pseudo_population_pca(
        Xc, yc, pca, factors, mouse_slices,
        levels=levels, epoch=epoch, bl_bins=bl_bins, norm=norm, soft=soft,
    )
    n_comp = W_ref.shape[0]
    n_time = X.shape[-1]

    # ── perturbed / non-clean trials (already out-of-sample for every basis) ──
    m_pert = ~m_clean
    Xp = X[m_pert.to_numpy()]
    yp_full = y.loc[m_pert].reset_index(drop=True).copy()
    n_pert = len(yp_full)
    yp_full['_tid'] = np.arange(n_pert)
    Zp, yp = None, None
    if n_pert and not pert_foldwise:
        # single projection through the reference basis
        Zp, yp = project_trials(Xp, yp_full.drop(columns='_tid'),
                                W_ref, mean_ref, scale_ref, mouse_slices)
    Zp_sum = np.zeros((n_pert, n_time, n_comp))   # foldwise accumulators
    Zp_cnt = np.zeros(n_pert)

    # ── CV: held-out clean projection, aligned to the reference frame ─────────
    strata = (
        yc["odor_pair"].astype(str) + "_" + yc["mouse"].astype(str)
        + "_" + yc["day"].astype(str) + "_" + yc["tasks"].astype(str)
    ).to_numpy()

    splitter = folds.split(Xc, strata)
    if show_pbar:
        from tqdm import tqdm
        splitter = tqdm(splitter, total=folds.get_n_splits(Xc, strata),
                        desc="cv_pca_pseudo")

    Z_sum = np.zeros((len(yc), n_time, n_comp))
    Z_cnt = np.zeros(len(yc))
    evr_folds = []
    for train_idx, test_idx in splitter:
        Xtr = Xc[train_idx]
        ytr = yc.iloc[train_idx].reset_index(drop=True)
        W_f, mean_f, scale_f, evr_f, _, _ = pseudo_population_pca(
            Xtr, ytr, pca, factors, mouse_slices,
            levels=levels, epoch=epoch, bl_bins=bl_bins, norm=norm, soft=soft,
        )
        evr_folds.append(evr_f)

        # project HELD-OUT clean trials through the fold basis, then rotate the
        # scores into the reference frame so folds can be averaged together
        R = procrustes_rotation(W_f, W_ref)
        Z_te, y_te = project_trials(
            Xc[test_idx], yc.iloc[test_idx].reset_index(drop=True),
            W_f, mean_f, scale_f, mouse_slices,
        )
        Z_te = apply_rotation_to_scores(Z_te, R)
        tids = y_te['_tid'].to_numpy()
        Z_sum[tids] += Z_te
        Z_cnt[tids] += 1

        # perturbed trials through this fold's basis (projected every fold)
        if n_pert and pert_foldwise:
            Z_pe, y_pe = project_trials(Xp, yp_full, W_f, mean_f, scale_f, mouse_slices)
            Z_pe = apply_rotation_to_scores(Z_pe, R)
            tp = y_pe['_tid'].to_numpy()
            Zp_sum[tp] += Z_pe
            Zp_cnt[tp] += 1

    evr_folds = np.asarray(evr_folds)

    if n_pert and pert_foldwise:
        Zp = Zp_sum / Zp_cnt[:, None, None]          # mean over all folds
        yp = yp_full.drop(columns='_tid')

    # average each trial's held-out projections across repeats → one CV'd row
    seen = Z_cnt > 0
    Z_clean = np.zeros((len(yc), n_time, n_comp))
    Z_clean[seen] = Z_sum[seen] / Z_cnt[seen][:, None, None]
    if (~seen).any():
        # safety net: any clean trial never held out (cannot happen with KFold)
        idx_fb = np.where(~seen)[0]
        Z_fb, y_fb = project_trials(
            Xc[idx_fb], yc.iloc[idx_fb].reset_index(drop=True),
            W_ref, mean_ref, scale_ref, mouse_slices,
        )
        Z_clean[y_fb['_tid'].to_numpy()] = Z_fb

    y_clean = yc.drop(columns='_tid')

    if Zp is not None:
        Z_all = np.concatenate([Z_clean, Zp], axis=0)
        y_all = pd.concat([y_clean, yp], axis=0, ignore_index=True)
    else:
        Z_all, y_all = Z_clean, y_clean

    return Z_all, y_all, W_ref, Z_cond, evr_folds
