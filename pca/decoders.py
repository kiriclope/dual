"""THE decoder for Fig 2/3 — defined ONCE and imported everywhere, so panels cannot drift apart.

Before 2026-08-12 the figure mixed methods: the geometry panels used difference-of-class-means
contrast axes while the pooled decoding panels used regularised logistic regression. Both are linear
readouts, but a reviewer comparing a cosine from one panel with an accuracy from another was
comparing quantities built by different estimators. This module makes every panel use the pipeline
already used by the overlaps caches (`overlaps/fig_ccgp_matrices_pseudo.py:61`):

    StandardScaler -> PCA(n) -> LogisticRegression(C=1, class_weight='balanced')

with n = min(20, n_features, n_samples-1) so per-animal populations (a few hundred neurons) use the
same call as the 3,319-neuron pseudo-population.

`fit_axis` returns the decision direction mapped BACK into the input space and normalised to unit
length, so the same object serves as (i) a decoder and (ii) a geometric axis for cosines and for
plotting — the two uses cannot disagree because they are the same vector.
"""
import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

C_REG = 1.0
NOPCA = '--nopca' in sys.argv          # any script run with --nopca drops the denoising step
NPC = 20                               # --npc N tightens/loosens the denoising (default 20)
if '--npc' in sys.argv:
    NPC = int(sys.argv[sys.argv.index('--npc') + 1])
# caches and figures take the suffix, so variants never overwrite each other
SUF = '_nopca' if NOPCA else ('' if NPC == 20 else f'_npc{NPC}')


def make_clf(n_features, n_samples):
    steps = [StandardScaler()]
    if not NOPCA:
        steps.append(PCA(n_components=int(max(1, min(NPC, n_features, n_samples - 1))),
                         random_state=0))
    steps.append(LogisticRegression(C=C_REG, class_weight='balanced', max_iter=3000))
    return make_pipeline(*steps)


def fit_axis(X, y):
    """Fit the standard decoder; return (unit weight vector in X's space, fitted pipeline).

    The pipeline's decision function is  coef . PCA(( x - mean ) / scale), so the direction in the
    input space is (V^T coef) / scale — undoing the projection and the standardisation."""
    X = np.nan_to_num(np.asarray(X, float))
    clf = make_clf(X.shape[1], X.shape[0]).fit(X, y)
    sc, lr = clf.steps[0][1], clf.steps[-1][1]
    wz = lr.coef_[0] if NOPCA else (clf.steps[1][1].components_.T @ lr.coef_[0])
    w = wz / np.where(sc.scale_ > 0, sc.scale_, 1.0)
    n = np.linalg.norm(w)
    return (w / n if n > 0 else w), clf


def bal_acc(clf, X, y):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, clf.predict(np.nan_to_num(np.asarray(X, float)))))
