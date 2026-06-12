import numpy as np
from sklearn.pipeline import Pipeline


def get_space_params_timewise(est, raw=False, min_scale=1e-8):
    """Extract per-timepoint weights and intercepts from a fitted
    SlidingEstimator / GeneralizingEstimator.

    Parameters
    ----------
    raw : bool
        Back-project weights through the scaler (w_raw = w_scaled / scale).
    min_scale : float
        Floor on scaler.scale_ to avoid blow-up from near-zero-variance features.

    Returns
    -------
    ws    : (n_train_times, n_features)
    bs    : (n_train_times,)
    means : (n_train_times, n_features) or None
    """
    ws, bs, means = [], [], []
    any_mean = False

    for est_t in est.estimators_:
        if isinstance(est_t, Pipeline):
            clf = est_t.named_steps["model"]
            scaler = est_t.named_steps.get("scaler", None)
        else:
            clf, scaler = est_t, None

        w = np.asarray(clf.coef_).ravel().astype(float)
        b = float(np.asarray(clf.intercept_).item())

        mu = None
        if scaler is not None:
            mean = getattr(scaler, "mean_", None)
            scale = getattr(scaler, "scale_", None)
            mean = np.zeros_like(w) if mean is None else np.asarray(mean)
            scale = np.ones_like(w) if scale is None else np.asarray(scale)
            scale = np.where(scale < min_scale, 1.0, scale)

            if raw:
                w = w / scale
                b = b - np.dot(w, mean)
                mu = mean
                any_mean = True

        ws.append(w)
        bs.append(b)
        means.append(mu)

    ws = np.asarray(ws)
    bs = np.asarray(bs)
    means = np.asarray(means) if any_mean else None
    return ws, bs, means


def _apply_scaler_if_needed(est, X, raw):
    """If raw=False and the pipeline has a scaler, transform X per train-time."""
    if raw:
        return X
    first = est.estimators_[0]
    if not isinstance(first, Pipeline) or "scaler" not in first.named_steps:
        return X

    X_out = np.empty_like(X, dtype=float)
    for t, est_t in enumerate(est.estimators_):
        sc = est_t.named_steps.get("scaler", None)
        X_out[..., t] = X[..., t] if sc is None else sc.transform(X[..., t])
    return X_out


def get_decision_values(est, X, raw=False, remove_intercept=True):
    """Project trials onto the per-time discriminant axis.

    Returns
    -------
    dec : (n_trials, n_train_times) for SlidingEstimator
          (n_trials, n_train_times, n_test_times) for GeneralizingEstimator
    """
    ws, bs, means = get_space_params_timewise(est, raw=raw)

    if remove_intercept:
        bs = np.zeros_like(bs)

    X_proj = _apply_scaler_if_needed(est, X, raw=raw)
    is_sliding = est.__class__.__name__ == "SlidingEstimator"

    if is_sliding:
        dec = np.einsum("nft,tf->nt", X_proj, ws) + bs[None, :]
    else:
        dec = np.einsum("nfu,tf->ntu", X_proj, ws) + bs[None, :, None]

    if raw and remove_intercept and means is not None:
        offsets = np.einsum("tf,tf->t", ws, means)
        if is_sliding:
            dec = dec - offsets[None, :]
        else:
            dec = dec - offsets[None, :, None]

    return dec


def get_norms_timewise(est, raw=False):
    ws, _, _ = get_space_params_timewise(est, raw=raw)
    return np.linalg.norm(ws, axis=1)


def postprocess_decision(dec, norms, y=None, signed=False, eps=1e-6):
    """Normalize by weight norm; optionally flip sign by label (0/1 coding)."""
    norms = norms + eps

    if dec.ndim == 2:
        dec = dec / norms[None, :]
        if signed:
            dec = (2 * np.asarray(y) - 1)[:, None] * dec
    elif dec.ndim == 3:
        dec = dec / norms[None, :, None]
        if signed:
            dec = (2 * np.asarray(y) - 1)[:, None, None] * dec
    else:
        raise ValueError(f"Unexpected dec shape: {dec.shape}")

    return dec
