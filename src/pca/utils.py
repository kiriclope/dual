import itertools

import numpy as np
import pandas as pd


def pad_list(arrays, axis=0, max_len=None):
    """Pad a list of arrays along `axis` with NaNs to a common length."""
    if max_len is None:
        max_len = max(arr.shape[axis] for arr in arrays)
    padded = []
    for arr in arrays:
        n_pad = max_len - arr.shape[axis]
        if n_pad > 0:
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axis] = (0, n_pad)
            arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
        padded.append(arr)
    return padded


def get_levels(y, factors):
    """Return {factor: sorted_unique_values} for the given factor columns."""
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
    """Boolean mask selecting rows where each factor equals the combo value."""
    m = np.ones(len(y), dtype=bool)
    for f, v in zip(factors, combo):
        s = y[f]
        if np.issubdtype(s.dtype, np.number):
            m &= s.values == v
        else:
            m &= s.astype(str).values == str(v)
    return m


def cv_avg_cond(X, y, factors, levels=None, drop_missing=False, fill_nan=False):
    """
    Condition-average X over all combinations of factor levels.

    Parameters
    ----------
    X : (n_trials, n_neurons, n_time)
    y : DataFrame aligned with X
    factors : str or list[str]
    levels : dict {factor: values} — computed from y if None
    drop_missing : skip combos with no matching trials (returned array shorter)
    fill_nan : fill missing combos with NaN rather than raising

    Returns
    -------
    X_avg : (n_combos, n_neurons, n_time)
    combos : list of tuples
    """
    if isinstance(factors, str):
        factors = [factors]
    if levels is None:
        levels = get_levels(y, factors)

    combos = list(itertools.product(*[levels[f] for f in factors]))
    X_avg = []
    valid_combos = []
    for combo in combos:
        idx = _combo_mask(y, factors, combo)
        if not idx.any():
            if drop_missing:
                continue
            if fill_nan:
                X_avg.append(np.full_like(X[0], np.nan))
                valid_combos.append(combo)
                continue
            raise ValueError(f"Missing combo: {dict(zip(factors, combo))}")
        X_avg.append(np.nanmean(X[idx], axis=0))
        valid_combos.append(combo)

    return np.stack(X_avg, axis=0), valid_combos


def anchors_from_Z(Z, y, factors, levels, drop_missing=False):
    """
    Compute per-condition mean of latent trajectories, flattened to 2-D.

    Z : (n_trials, n_time, n_comp)
    Returns A : (n_combos * n_time, n_comp)
    """
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
