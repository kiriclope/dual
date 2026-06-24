import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.overlaps.null import label_permutation_null, weight_shuffle_null
from src.overlaps.weights import (get_decision_values, get_norms_timewise,
                                  get_space_params_timewise, postprocess_decision)


def _zscore_against_null(obs, null):
    mu = null.mean(0)
    sd = null.std(0)
    return (obs - mu) / (sd + 1e-12), mu, sd


def ccgd_validation(
    X_y_data,
    estimator,
    selector=None,
    cv=None,
    signed=False,
    raw=False,
    remove_intercept=True,
    fit_param_epoch=False,
    bins_epoch=None,
    random_state=0,
    null_type=None,
    n_shuffles=100,
    null_reduction="zscore",
    return_weights=False,
):
    """Cross-validated cross-condition generalized decoding.

    Parameters
    ----------
    X_y_data : tuple returned by dataloader()
        (X_within, y_within, X_cross, y_cross, strata)
    estimator : fitted-compatible sklearn estimator (GeneralizingEstimator)
    null_type : None | 'weight' | 'label'
    null_reduction : 'zscore' | 'subtract'

    Returns
    -------
    probas   : (n_trials_all, n_train, n_test)
    dfs      : (n_trials_all, n_train, n_test)
    y_cv     : DataFrame aligned with first axis of probas/dfs
    null_info : dict of per-fold null statistics. If return_weights=True it also
        carries "weights": (n_train_times, n_features) fold-averaged discriminant
        axis in the (raw if raw=True) feature space, for cosine-similarity analyses.
    """
    X_within, y_within, X_cross, y_cross, strata = X_y_data
    y_within_labels = y_within["labels"].to_numpy()
    y_cross_labels = y_cross["labels"].to_numpy()

    has_cross = X_cross.shape[0] > 0
    if not has_cross:
        print("[ccgd_validation] y_cross is empty — skipping cross-condition block.")

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(cv.split(X_within, strata))

    within_probas, within_dfs, y_cv_within = [], [], []
    cross_probas, cross_dfs = [], []
    null_info = {
        "within_mu": [], "within_sd": [],
        "cross_mu": [],  "cross_sd": [],
        "C_per_fold": [],
    }
    weight_accum = []   # per-fold discriminant axes (n_train, n_feat), if requested

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X_within[train_idx], X_within[test_idx]
        y_tr = y_within_labels[train_idx].copy()
        y_te = y_within_labels[test_idx]
        y_cv_within.append(y_within.iloc[test_idx].reset_index(drop=True))

        est = clone(estimator)

        if fit_param_epoch:
            sel = clone(selector)
            sel.fit(X_tr[..., bins_epoch].mean(-1), y_tr)
            best_C = sel.named_steps["model"].C_[0]
            est.base_estimator.named_steps["model"].Cs = [best_C]
            null_info["C_per_fold"].append(best_C)

        est.fit(X_tr, y_tr)
        norms = get_norms_timewise(est, raw=raw)

        if return_weights:
            ws_fold, _, _ = get_space_params_timewise(est, raw=raw)
            weight_accum.append(ws_fold)   # (n_train, n_feat)

        within_obs = get_decision_values(est, X_te, raw=raw,
                                         remove_intercept=remove_intercept)
        within_obs = postprocess_decision(within_obs, norms, y=y_te, signed=signed)

        if has_cross:
            cross_obs = get_decision_values(est, X_cross, raw=raw,
                                            remove_intercept=remove_intercept)
            cross_obs = postprocess_decision(cross_obs, norms,
                                             y=y_cross_labels, signed=signed)
        else:
            cross_obs = None

        within_null = cross_null = None
        if null_type == "weight":
            within_null = weight_shuffle_null(
                est, X_te, y_te, n_shuffles=n_shuffles, raw=raw,
                signed=signed, remove_intercept=remove_intercept,
                random_state=random_state + fold,
            )
            if has_cross:
                cross_null = weight_shuffle_null(
                    est, X_cross, y_cross_labels, n_shuffles=n_shuffles,
                    raw=raw, signed=signed, remove_intercept=remove_intercept,
                    random_state=random_state + fold,
                )
        elif null_type == "label":
            within_null = label_permutation_null(
                est, X_tr, y_tr, X_te, y_te,
                n_shuffles=n_shuffles, raw=raw, signed=signed,
                remove_intercept=remove_intercept,
                random_state=random_state + fold,
            )
            if has_cross:
                cross_null = label_permutation_null(
                    est, X_tr, y_tr, X_cross, y_cross_labels,
                    n_shuffles=n_shuffles, raw=raw, signed=signed,
                    remove_intercept=remove_intercept,
                    random_state=random_state + fold,
                )

        def _calibrate(obs, null):
            if obs is None:
                return None, None, None
            if null is None:
                return obs, None, None
            if null_reduction == "zscore":
                return _zscore_against_null(obs, null)
            if null_reduction == "subtract":
                mu = null.mean(0)
                return obs - mu, mu, None
            return obs, None, None

        within_cal, mu_w, sd_w = _calibrate(within_obs, within_null)
        cross_cal, mu_c, sd_c = _calibrate(cross_obs, cross_null)

        null_info["within_mu"].append(mu_w)
        null_info["within_sd"].append(sd_w)
        null_info["cross_mu"].append(mu_c)
        null_info["cross_sd"].append(sd_c)

        within_dfs.append(within_cal)
        if has_cross:
            cross_dfs.append(cross_cal)

        w_proba = est.predict_proba(X_te)[..., 1]
        w_proba = (y_te[:, None, None] * w_proba
                   + (1 - y_te[:, None, None]) * (1 - w_proba))
        within_probas.append(w_proba)

        if has_cross:
            c_proba = est.predict_proba(X_cross)[..., 1]
            c_proba = (y_cross_labels[:, None, None] * c_proba
                       + (1 - y_cross_labels[:, None, None]) * (1 - c_proba))
            cross_probas.append(c_proba)

    if return_weights and weight_accum:
        # mean discriminant axis across folds (direction is what cosine needs)
        null_info["weights"] = np.mean(np.stack(weight_accum, axis=0), axis=0)

    within_probas = np.vstack(within_probas)
    within_dfs = np.vstack(within_dfs)
    y_cv_within = pd.concat(y_cv_within, axis=0, ignore_index=True)

    if has_cross:
        cross_probas = np.mean(np.stack(cross_probas, axis=0), axis=0)
        cross_dfs = np.mean(np.stack(cross_dfs, axis=0), axis=0)
        y_cv = pd.concat([y_cv_within, y_cross], axis=0, ignore_index=True)
        probas = np.vstack([within_probas, cross_probas])
        dfs = np.vstack([within_dfs, cross_dfs])
    else:
        y_cv = y_cv_within
        probas = within_probas
        dfs = within_dfs

    return probas, dfs, y_cv, null_info
