import numpy as np
import pandas as pd
from sklearn.base import clone

from src.overlaps.data import dataloader
from src.overlaps.estimator import get_estimator
from src.overlaps.weights import get_space_params_timewise


def normalize_stage(s):
    """Map learning labels (0/1 or string variants) to {'Naive', 'Expert'}."""
    if np.issubdtype(s.dtype, np.number):
        return s.map({0: "Naive", 1: "Expert"})
    return (
        s.astype(str).str.strip().str.lower()
         .map({"naive": "Naive", "expert": "Expert",
               "0": "Naive", "1": "Expert"})
    )


def correct_trials(df):
    """Standard correct, non-laser trial mask."""
    return (df.laser == 0) & (df.performance == 1) & (
        (df.tasks == "DPA") | (df.odr_perf == 1)
    )


def attach_delay_value(X_epoch, y_df, delay_bins=None, trial_mask=None):
    """Return y_df with a per-trial 'value' = mean trajectory over delay_bins."""
    if delay_bins is None:
        raise ValueError(
            "attach_delay_value() requires 'delay_bins' (e.g. options['bins_DELAY'])"
        )
    d = y_df.reset_index(drop=True).copy()
    d["traj"] = list(X_epoch)
    if trial_mask is not None:
        d = d[trial_mask.values if isinstance(trial_mask, pd.Series)
              else trial_mask].copy()
    d["value"] = d["traj"].apply(lambda t: np.nanmean(np.asarray(t)[delay_bins]))
    d["stage"] = normalize_stage(d["learning"])
    d = d[d["stage"].isin(["Naive", "Expert"])]
    return d


def pivot_delta(df, index_cols, value_col):
    """Pivot Naive/Expert columns and compute delta = Expert - Naive."""
    w = (
        df.pivot_table(index=index_cols, columns="stage",
                       values=value_col, aggfunc="mean")
          .dropna(subset=["Naive", "Expert"])
          .reset_index()
    )
    w["delta"] = w["Expert"] - w["Naive"]
    return w


def fit_axis_weights(
    X, y_df, target, stage, context, epoch_bins,
    scaler="standard", l1_ratio=0.0, raw=True,
):
    """Refit one estimator on full within-condition data; return mean w over epoch.

    Parameters
    ----------
    epoch_bins : array-like of int
        Time-bin indices to average the weight vector over.
    scaler, l1_ratio, raw : passed through to get_estimator / get_space_params_timewise

    Returns
    -------
    w : (n_neurons,)  mean weight across epoch_bins
    """
    Xw, yw, _, _, _, _, _ = dataloader(
        X, y_df, target=target, stage=stage, context=context,
        correct=False, strata=False,
    )
    est = get_estimator(clf="logcv", scoring="accuracy", mode="generalizing",
                        scaler=scaler, l1_ratio=l1_ratio, n_jobs=-1)
    est.fit(Xw, yw["labels"].to_numpy())
    ws, _, _ = get_space_params_timewise(est, raw=raw)   # (T, F)
    return ws[epoch_bins].mean(0)                        # (F,)


def fit_axis(X, y_df, target, bins, stage, context="all", estimator=None):
    """Return a unit-norm decoder weight vector for target averaged over bins.

    Parameters
    ----------
    estimator : sklearn Pipeline (mode=None), cloned internally
    bins : array-like of int  — epoch bins to average activity before fitting
    """
    mask = (y_df.learning == stage) & (y_df.laser == 0)
    if context != "all":
        mask &= (y_df.tasks == context)
    Xm = X[mask]
    ym = y_df.loc[mask].reset_index(drop=True).copy()
    ym["labels"] = ym[target].values

    Xe = np.nanmean(Xm[..., bins], axis=-1)
    est = clone(estimator)
    est.fit(Xe, ym["labels"].values)

    if hasattr(est, "named_steps"):
        w = est.named_steps["model"].coef_.ravel()
    else:
        w = est.coef_.ravel()
    w = w / (np.linalg.norm(w) + 1e-12)
    return w


def subspace_angle(w1, w2):
    """Principal angle (degrees) between two 1-D subspaces."""
    c = np.clip(np.dot(w1, w2), -1, 1)
    return np.degrees(np.arccos(abs(c)))


def project_2d(X, y_df, bins_epoch, stage, estimator):
    """Per-trial (sample_proj, choice_proj) in the delay epoch subspace.

    Parameters
    ----------
    estimator : sklearn Pipeline (mode=None), used to fit sample and choice axes

    Returns
    -------
    proj_s : (n_trials,)
    proj_c : (n_trials,)
    ym     : DataFrame aligned with the projections
    """
    mask = (y_df.learning == stage) & (y_df.laser == 0)
    Xm = X[mask]
    ym = y_df.loc[mask].reset_index(drop=True)
    valid = ~np.all(np.isnan(Xm), axis=(0, 2))
    Xm = Xm[:, valid, :]
    Xe = np.nanmean(Xm[..., bins_epoch], axis=-1)

    w_s = clone(estimator).fit(Xe, ym["sample"].values)
    w_c = clone(estimator).fit(Xe, ym["choice"].values)
    ws = w_s.named_steps["model"].coef_.ravel()
    wc = w_c.named_steps["model"].coef_.ravel()
    ws /= np.linalg.norm(ws) + 1e-12
    wc /= np.linalg.norm(wc) + 1e-12

    proj_s = Xe @ ws
    proj_c = Xe @ wc
    return proj_s, proj_c, ym
