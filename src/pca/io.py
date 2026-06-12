import os
import pickle as pkl

import numpy as np
import pandas as pd


def pkl_save(obj, name, path="."):
    os.makedirs(path, exist_ok=True)
    dst = os.path.join(path, name + ".pkl")
    print("saving to", dst)
    pkl.dump(obj, open(dst, "wb"))


def pkl_load(name, path="."):
    src = os.path.join(path, name + ".pkl")
    print("loading from", src)
    return pkl.load(open(src, "rb"))


def build_padded_X(options, n_neurons_total=3319, scale=None):
    """
    Load per-mouse calcium data, optionally scale per day, then zero-pad all
    mice into a shared neuron axis of length n_neurons_total.

    Parameters
    ----------
    options : dict
        Must contain 'mice' and all keys consumed by set_options / get_X_y_days.
    n_neurons_total : int
        Width of the shared neuron dimension.
    scale : str or None
        Day-wise normalisation applied before padding.
        One of: None, 'std', 'mad', 'very_mad', 'std_trial', 'mad_trial'.

    Returns
    -------
    X_big : (sum_trials, n_neurons_total, n_time)  float32
    y_big : DataFrame, one row per trial
    mouse_slices : dict  {mouse_name: slice}  — neuron ranges per mouse
    """
    from src.common.get_data import get_X_y_days
    from src.common.options import set_options

    X_list, y_list, mouse_slices = [], [], {}
    counter = 0

    for mouse in options["mice"]:
        opts = set_options(**{**options, "mouse": mouse})
        X, y = get_X_y_days(**opts)
        y["mouse"] = mouse
        print(mouse, X.shape, y.shape)

        n_m = X.shape[1]
        sl = slice(counter, counter + n_m)
        mouse_slices[str(mouse)] = sl
        counter += n_m

        X_scale = _scale_per_day(X, y, opts["n_days"], scale)

        X_pad = np.zeros((X.shape[0], n_neurons_total, X.shape[-1]), dtype=np.float32)
        X_pad[:, sl, :] = np.nan_to_num(X_scale, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X_pad)
        y_list.append(y)

    if counter > n_neurons_total:
        raise ValueError(f"Total neurons {counter} exceeds n_neurons_total={n_neurons_total}")

    X_big = np.concatenate(X_list, axis=0)
    y_big = pd.concat(y_list, axis=0, ignore_index=True)
    return X_big, y_big, mouse_slices


# ── internal ──────────────────────────────────────────────────────────────────

def _scale_per_day(X, y, n_days, scale):
    X_out = X.copy()
    if scale is None:
        return X_out
    for day in range(1, n_days + 1):
        idx0 = (y.day == day) & (y.laser == 0)
        if scale == 'blcenter':
            # subtract the per-neuron BASELINE mean only (a scalar over the
            # baseline window), not the full per-time mean PSTH — removes day
            # drift without the cross-condition per-timepoint demeaning artifact
            mean_ = np.nanmean(X[idx0][:, :, :12], axis=(0, 2), keepdims=True)
            std_ = 1.0
        else:
            mean_ = np.nanmean(X[idx0], axis=0, keepdims=True)
            std_ = _day_std(X[idx0], scale)
        X_out[y.day == day] = (X[y.day == day] - mean_) / std_
    return X_out


def _day_std(X0, scale):
    if scale == 'std':
        s = np.nanstd(X0, axis=(0, 2), ddof=0, keepdims=True)
        lo, hi = np.percentile(s.ravel(), [5, 95])
        return np.clip(s, lo, hi)
    if scale == 'mad':
        med = np.nanmedian(X0, axis=(0, 2), keepdims=True)
        s = 1.4826 * np.nanmedian(np.abs(X0 - med), axis=(0, 2), keepdims=True)
        lo, hi = np.nanpercentile(s.ravel(), [5, 95])
        return np.clip(s, lo, hi)
    if scale == 'very_mad':
        med = np.nanmedian(X0, axis=(0, 2), keepdims=True)
        s = 1.4826 * np.nanmedian(np.abs(X0 - med), axis=(0, 2), keepdims=True)
        lo, hi = np.nanpercentile(s.ravel(), [25, 75])
        return np.clip(s, lo, hi)
    if scale == 'center':
        return 1.0   # center-only: subtract mean, no std division
    if scale == 'std_trial':
        s = np.nanstd(X0, axis=0, ddof=0, keepdims=True)
        lo, hi = np.percentile(s.ravel(), [5, 95])
        return np.clip(s, lo, hi)
    if scale == 'mad_trial':
        med = np.nanmedian(X0, axis=0, keepdims=True)
        s = 1.4826 * np.nanmedian(np.abs(X0 - med), axis=0, keepdims=True)
        lo, hi = np.nanpercentile(s.ravel(), [5, 95])
        return np.clip(s, lo, hi)
    raise ValueError(f"Unknown scale: {scale!r}")
