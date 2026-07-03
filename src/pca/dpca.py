"""
Demixed PCA (dPCA, Kobak et al. 2016) on the within-mouse pseudo-population.

Why this exists
---------------
Plain pseudo-PCA mixes the large, high-dimensional condition-INDEPENDENT
stimulus-evoked transients into the task axes, so e.g. the choice axis still
carries the sample/distractor/test timing (removing a few CI directions with
`remove_ci_subspace` only strips the top ~half of that component).

dPCA splits the condition-averaged data into orthogonal marginalisations — a
time-only (condition-independent) part, one part per task factor, and their
interactions — and fits, for each marginalisation, a reduced-rank DECODER that
captures that marginal's variance while demixing it from the others.  Projecting
trials through the choice decoder therefore gives a choice trajectory free of the
stimulus-timing common mode by construction.

For a complete factorial over `factors` the marginalisations sum to the data
(ANOVA-style), so `factors` must form a complete design (e.g. `sample test`,
which spans the four odor pairs).  The choice contrast is the **sample×test
interaction** (lick iff sample and test match), so with `factors=['sample',
'test']` the choice axis is the `sample:test` marginalisation.

Public API
----------
dpca_decode : reduced-rank decoders/encoders per marginalisation of the pop
cv_dpca     : drop-in CV wrapper (mirrors cv_pca_pseudo's return shape)
"""

import numpy as np
import pandas as pd
from itertools import combinations

from src.pca.procrustes import apply_rotation_to_scores, procrustes_rotation
from src.pca.pseudo import build_pseudo_population, project_trials
from src.pca.utils import get_levels


# ── marginalisation ─────────────────────────────────────────────────────────────

def _marginals(A, n_factors):
    """ANOVA-style marginal decomposition of A over its first `n_factors` axes.

    A : (l1, ..., lk, n_neurons, n_time)  — complete-factorial condition averages
    Returns {factor-axis tuple: array of A's shape}; the parts sum to A.  Key
    () is the condition-independent (time-only) marginal, (i,) a main effect,
    (i, j) an interaction, ...
    """
    fax = tuple(range(n_factors))
    marg = {}
    for r in range(n_factors + 1):
        for keep in combinations(fax, r):
            drop = tuple(a for a in fax if a not in keep)
            m = A.mean(axis=drop, keepdims=True) if drop else A.copy()
            m = np.broadcast_to(m, A.shape).copy()
            for r2 in range(r):                      # subtract lower-order parts
                for sub in combinations(keep, r2):
                    m -= marg[sub]
            marg[keep] = m
    return marg


def _marg_name(key, factors):
    """'time' for the condition-independent marginal, else 'sample:test' etc."""
    return 'time' if not key else ':'.join(factors[i] for i in key)


# ── reduced-rank decoders ───────────────────────────────────────────────────────

def dpca_decode(P, sizes, factors, q=2, ridge=1e-2):
    """Fit per-marginalisation reduced-rank decoders on the pseudo-population.

    P     : (n_cond, n_neurons_total, n_time)  condition averages (rows in
            product(levels) order; complete factorial)
    sizes : list[int]  number of levels per factor (n_cond == prod(sizes))
    q     : decoder components kept per marginalisation
    ridge : ridge fraction (× mean spectral power of P) for the regression

    Returns
    -------
    W       : (n_comp, n_neurons_total)   stacked decoders (rows = components)
    evr     : (n_comp,)                   variance fraction per component
    labels  : list[str]                   marginal name per component
    Z_cond  : (n_cond, n_time, n_comp)    reference component trajectories
    """
    n_cond, n_total, n_t = P.shape
    A = P.reshape(*sizes, n_total, n_t)
    marg = _marginals(A, len(sizes))

    # flatten neuron axis to front: (n_total, n_cond*n_time) = (N, S)
    def flat(M):
        return np.moveaxis(M, -2, 0).reshape(n_total, -1)

    X = flat(A)                                          # (N, S)
    Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)  # economy SVD over samples
    Vx = VxT.T
    total_var = float(np.sum(sx ** 2)) + 1e-12
    lam = ridge * np.mean(sx ** 2)
    shrink = sx ** 2 / (sx ** 2 + lam)                   # ridge-shrunk projector
    shrinkD = sx / (sx ** 2 + lam)

    # main effects first, then interactions by order, condition-independent last
    keys = sorted(marg, key=lambda t: (len(t) == 0, len(t), t))

    from sklearn.utils.extmath import randomized_svd
    W, evr, labels, Zc_list = [], [], [], []
    for key in keys:
        Xphi = flat(marg[key])                           # (N, S)
        # reduced-rank regression of the marginal on the full data:
        #   minimise ||Xphi - F D X||²  with rank q, ridge lam.
        # Yhat = M @ Vx.T with M = (Xphi @ Vx) * shrink; Vx is orthogonal so the
        # left singular vectors / values of Yhat = those of M — and we only need
        # the top q, so a truncated SVD of M avoids forming Yhat and the full SVD.
        M = (Xphi @ Vx) * shrink[None, :]                # (N, S)
        F, sf, _ = randomized_svd(M, n_components=q, random_state=0)  # F (N,q)
        D = ((F.T @ Xphi @ Vx) * shrinkD[None, :]) @ Ux.T   # decoder (q, N)
        Zc = ((F.T @ M) @ Vx.T).reshape(q, n_cond, n_t)  # reference comps (q, cond, t)

        name = _marg_name(key, factors)
        W.append(D)
        evr.append(sf ** 2 / total_var)
        labels += [name] * q
        Zc_list.append(np.moveaxis(Zc, 0, -1))           # (n_cond, n_t, q)

    W = np.vstack(W)
    evr = np.concatenate(evr)
    Z_cond = np.concatenate(Zc_list, axis=-1)
    return W, evr, labels, Z_cond


# ── drop-in CV wrapper (mirrors cv_pca_pseudo) ──────────────────────────────────

def cv_dpca(
    X, y, folds, factors,
    epoch=None, bl_bins=None, levels=None,
    learning='Expert', norm='zscore', soft=1e-3,
    mouse_slices=None, q=2, ridge=1e-2, show_pbar=True, perf_filter=True,
):
    """Cross-validated dPCA on the within-mouse pseudo-population.

    Decoders are fit on within-mouse condition averages of the CLEAN trials.
    Clean trial scores are CROSS-VALIDATED: each fold refits the decoders on its
    train split, projects the held-out clean trials, and aligns each marginal's
    component block to the reference frame (Procrustes per marginal).  Averaging
    a trial's held-out projections across repeats gives one out-of-sample row per
    trial.  This is essential here — an in-sample projection makes the reduced-
    rank regression model the (out-of-window) noise and spuriously "decode" test/
    choice before they are presented; held-out projection collapses that to ~0.
    Perturbed / non-clean trials are projected through the reference decoders.

    `epoch` should be None (the whole trial) so each marginal is anchored by its
    variable's real signal at its real time — fitting on a single window where a
    variable has no signal overfits that window's noise.

    Returns (matching cv_pca_pseudo, plus the marginal labels)
    -------
    Z_all, y_all, W_ref, Z_cond, evr_folds, labels
    """
    if mouse_slices is None:
        raise ValueError("mouse_slices is required")
    if isinstance(factors, str):
        factors = [factors]

    m_clean = (y.laser == 0) & (y.learning == learning)
    if perf_filter:                                    # default: fit on correct trials only
        m_clean = m_clean & (y.performance == 1)       # perf_filter=False → include errors (decouples choice)
    yc = y.loc[m_clean].reset_index(drop=True).copy()
    yc['_tid'] = np.arange(len(yc))
    Xc = X[m_clean.to_numpy()]
    if levels is None:
        levels = get_levels(yc, factors)
    sizes = [len(levels[f]) for f in factors]

    # ── reference decoders on all clean data (canonical frame) ───────────────
    P, mean_, scale_, _ = build_pseudo_population(
        Xc, yc, factors, mouse_slices, levels=levels, epoch=epoch,
        bl_bins=bl_bins, norm=norm, soft=soft,
    )
    W_ref, _, labels, Z_cond = dpca_decode(P, sizes, factors, q=q, ridge=ridge)
    n_comp = W_ref.shape[0]
    n_marg = n_comp // q
    n_time = X.shape[-1]

    # ── perturbed / non-clean trials → reference decoders ────────────────────
    m_pert = ~m_clean
    Zp, yp = None, None
    if m_pert.any():
        Zp, yp = project_trials(
            X[m_pert.to_numpy()], y.loc[m_pert].reset_index(drop=True),
            W_ref, mean_, scale_, mouse_slices,
        )

    # ── CV: held-out clean projection, aligned per marginal ──────────────────
    strata = (
        yc["odor_pair"].astype(str) + "_" + yc["mouse"].astype(str)
        + "_" + yc["day"].astype(str) + "_" + yc["tasks"].astype(str)
    ).to_numpy()
    splitter = folds.split(Xc, strata)
    if show_pbar:
        from tqdm import tqdm
        splitter = tqdm(splitter, total=folds.get_n_splits(Xc, strata), desc="cv_dpca")

    Z_sum = np.zeros((len(yc), n_time, n_comp))
    Z_cnt = np.zeros(len(yc))
    evr_folds = []
    for train_idx, test_idx in splitter:
        Ptr, mean_f, scale_f, _ = build_pseudo_population(
            Xc[train_idx], yc.iloc[train_idx].reset_index(drop=True), factors,
            mouse_slices, levels=levels, epoch=epoch, bl_bins=bl_bins,
            norm=norm, soft=soft,
        )
        W_f, evr_f, _, _ = dpca_decode(Ptr, sizes, factors, q=q, ridge=ridge)
        evr_folds.append(evr_f)

        Z_te, y_te = project_trials(
            Xc[test_idx], yc.iloc[test_idx].reset_index(drop=True),
            W_f, mean_f, scale_f, mouse_slices,
        )
        # align each marginal's q-component block to the reference frame
        for mi in range(n_marg):
            a, b = mi * q, (mi + 1) * q
            R = procrustes_rotation(W_f[a:b], W_ref[a:b])
            Z_te[:, :, a:b] = apply_rotation_to_scores(Z_te[:, :, a:b], R)
        tids = y_te['_tid'].to_numpy()
        Z_sum[tids] += Z_te
        Z_cnt[tids] += 1

    evr_folds = np.asarray(evr_folds)
    seen = Z_cnt > 0
    Z_clean = np.zeros((len(yc), n_time, n_comp))
    Z_clean[seen] = Z_sum[seen] / Z_cnt[seen][:, None, None]
    y_clean = yc.drop(columns='_tid')

    if Zp is not None:
        Z_all = np.concatenate([Z_clean, Zp], axis=0)
        y_all = pd.concat([y_clean, yp], axis=0, ignore_index=True)
    else:
        Z_all, y_all = Z_clean, y_clean

    return Z_all, y_all, W_ref, Z_cond, evr_folds, labels
