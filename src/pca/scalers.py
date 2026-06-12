import numpy as np


class StandardScaler:
    """
    Mean-center (and optionally scale) a (n_trials, n_neurons, n_time) array.

    Clips scale to the [5, 95] percentile range before dividing to avoid
    outlier neurons dominating the normalisation.  if_scale=0 centres only.
    """

    def __init__(self, axis=0, if_scale=0):
        self.axis = axis
        self.if_scale_ = if_scale
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        self.center_ = np.nanmean(X, axis=self.axis, keepdims=True)
        self.scale_ = np.nanstd(X, axis=(self.axis, -1), keepdims=True)
        lo, hi = np.nanpercentile(self.scale_.ravel(), [5, 95])
        self.scale_ = np.clip(self.scale_, lo, hi)
        return self

    def transform(self, X):
        if self.if_scale_:
            return (X - self.center_) / self.scale_
        return X - self.center_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """Median-centre and IQR-scale along axis."""

    def __init__(self, axis=0):
        self.axis = axis
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        self.center_ = np.nanmedian(X, axis=self.axis, keepdims=True)
        q75 = np.nanpercentile(X, 75, axis=self.axis, keepdims=True)
        q25 = np.nanpercentile(X, 25, axis=self.axis, keepdims=True)
        self.scale_ = q75 - q25
        self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
