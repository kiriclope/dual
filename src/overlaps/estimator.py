import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import GeneralizingEstimator, SlidingEstimator


def smooth_and_bin2(X, sigma=1.0):
    """Gaussian-smooth along time then bin by pairs of samples."""
    X_s = gaussian_filter1d(X, sigma=sigma, axis=-1, mode="nearest")
    T = X_s.shape[-1]
    assert T % 2 == 0
    return X_s.reshape(*X_s.shape[:-1], T // 2, 2).mean(-1)


def get_estimator(
    clf="logcv",
    scoring="accuracy",
    mode="generalizing",
    scaler=None,
    l1_ratio=0,
    C=0.1,
    Cs=np.logspace(-4, 4, 10),
    n_jobs=-1,
):
    """Build a (Generalizing|Sliding)Estimator wrapping a logistic pipeline."""
    solver = "liblinear" if l1_ratio in (0, 1) else "saga"

    if clf == "logcv":
        clf_obj = LogisticRegressionCV(
            cv=5, solver=solver, class_weight="balanced",
            n_jobs=None, l1_ratios=(l1_ratio,), Cs=Cs,
        )
    else:
        clf_obj = LogisticRegression(
            solver=solver, class_weight="balanced",
            n_jobs=None, l1_ratio=l1_ratio, C=C,
        )

    steps = []
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "center":
        steps.append(("scaler", StandardScaler(with_std=False)))
    steps.append(("model", clf_obj))
    pipe = Pipeline(steps)

    if mode == "sliding":
        return SlidingEstimator(pipe, scoring=scoring, n_jobs=n_jobs, verbose=False)
    if mode == "generalizing":
        return GeneralizingEstimator(pipe, scoring=scoring, n_jobs=n_jobs, verbose=False)
    return pipe
