import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from src.overlaps.weights import (
    _apply_scaler_if_needed,
    get_decision_values,
    get_norms_timewise,
    get_space_params_timewise,
    postprocess_decision,
)


def _dec_from_ws(X_proj, ws, bs=None, remove_intercept=True, means=None, raw=False):
    """Project with given ws; apply mean-offset correction if raw + remove_intercept."""
    if bs is None or remove_intercept:
        bs = np.zeros(ws.shape[0])
    dec = np.einsum("nfu,tf->ntu", X_proj, ws) + bs[None, :, None]
    if raw and remove_intercept and means is not None:
        offsets = np.einsum("tf,tf->t", ws, means)
        dec = dec - offsets[None, :, None]
    return dec


def shuffle_weights_timewise(ws, rng, preserve_norm=True):
    """Permute feature indices independently per time-point."""
    F = ws.shape[1]
    ws_shuff = np.empty_like(ws)
    for t in range(ws.shape[0]):
        perm = rng.permutation(F)
        ws_shuff[t] = ws[t, perm]
        if preserve_norm:
            n0 = np.linalg.norm(ws[t])
            n1 = np.linalg.norm(ws_shuff[t]) + 1e-12
            ws_shuff[t] *= n0 / n1
    return ws_shuff


def weight_shuffle_null(
    est, X, y, n_shuffles=100, raw=False, signed=False,
    remove_intercept=True, random_state=0, n_jobs=-1,
):
    ws, bs, means = get_space_params_timewise(est, raw=raw)
    X_proj = _apply_scaler_if_needed(est, X, raw=raw)
    seeds = np.random.SeedSequence(random_state).generate_state(n_shuffles)

    def _one(seed):
        rng = np.random.default_rng(int(seed))
        ws_s = shuffle_weights_timewise(ws, rng, preserve_norm=True)
        norms_s = np.linalg.norm(ws_s, axis=1)
        dec_s = _dec_from_ws(X_proj, ws_s, bs=bs, remove_intercept=remove_intercept,
                              means=means, raw=raw)
        return postprocess_decision(dec_s, norms_s, y=y, signed=signed)

    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one)(s) for s in seeds)
    return np.stack(out, axis=0)


def label_permutation_null(
    estimator, X_train, y_train, X_test, y_test,
    n_shuffles=100, raw=False, signed=False,
    remove_intercept=True, random_state=0, n_jobs=-1,
):
    seeds = np.random.SeedSequence(random_state).generate_state(n_shuffles)

    def _one(seed):
        rng = np.random.default_rng(int(seed))
        y_perm = rng.permutation(y_train)
        est = clone(estimator)
        est.fit(X_train, y_perm)
        norms = get_norms_timewise(est, raw=raw)
        dec = get_decision_values(est, X_test, raw=raw,
                                  remove_intercept=remove_intercept)
        return postprocess_decision(dec, norms, y=y_test, signed=signed)

    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one)(s) for s in seeds)
    return np.stack(out, axis=0)
