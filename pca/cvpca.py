"""cvpca.py — THE cross-validated PCA estimator for Fig 2. One implementation, one place to change it.

Until 2026-09-09 this estimator was copy-pasted into seven producer scripts (`exp_dimensionality_fits`,
`exp_cdec_support`, `exp_pceta_cv`, `exp_dimensionality_ci`, `exp_dimensionality_jk`,
`exp_dimensionality_md`, `exp_learning_delta`). `neuron_scale` and `cvpca_spectrum` were byte-identical
in all of them and `split_means` differed only by the optional `mice` subset and `shuffle` switch, so a
change to the estimator had to be made seven times or the caches silently disagreed. They now all import
from here.

WHAT IT COMPUTES. The matrix being decomposed is CONDITIONS x NEURONS (4, 8 or 12 task conditions by
3,319 pseudo-population neurons, each neuron valid only inside its own mouse's column block), from
trials that are laser-off and correct, one learning stage at a time.

  1. split_means   within each mouse AND each condition separately, permute the trials and cut them in
                   half, then average each half. Two independent estimates of the same condition mean,
                   matched on design, mice and window; only the trials differ.
  2. neuron_scale  divide each neuron by its own SD across all that mouse/stage's trials. Computed once
                   from all trials and shared by both halves — it is a per-neuron scale and cannot
                   manufacture condition structure. NOTE it always uses every bound mouse, including
                   under a leave-one-mouse-out jackknife: the dropped mouse's columns are zeros in both
                   halves, so they contribute nothing to the SVD and their scale is irrelevant.
  3. cvpca_spectrum
                   centre both halves across conditions (removes the condition-independent component),
                   take the right singular vectors of ONE half, and report the INNER PRODUCT of the two
                   halves' scores on those directions — a covariance between independent estimates, not
                   a variance. Noise in the two halves is independent and zero-mean, so a direction that
                   captures only noise returns ~0 in expectation and may go negative. Selection and
                   evaluation never touch the same trials, which is exactly what ordinary PCA on
                   all-trial condition means gets wrong: it ranks components by variance measured on the
                   data that fitted them, so noise inflates the top components and the tail looks
                   structured. Both directions (fit on 1 read on 2, fit on 2 read on 1) are averaged.
  4. avg_spec      average over random half-splits. avg_frac clips the negatives to zero and normalises
                   to sum 1 — that fraction is what Fig 2b plots. pr_of gives the participation ratio
                   of the same clipped spectrum.

CALLER SETTINGS ARE NOT DEFAULTS. Each producer keeps its own (nsplits, seed) because those choices are
baked into the cached numbers: Fig 2b and its jackknife/null use 30 splits (seed 7, null seed 11), the
stored FITDATA spectrum and PR use 25 splits (seed 0), the PR jackknife and the learning-delta CIs use
20 (seed 7), the split-level PR CI uses 30 (seed 0). Passing them explicitly at every call site keeps
that visible. rng CONSUMPTION is identical to the pre-unification copies (one permutation per mouse x
condition with at least 2 trials, in condition order, plus one extra permutation per mouse only when
shuffle=True), so every cached value reproduces bit-for-bit — verified against results.pkl on
2026-09-09.

USAGE

    import cvpca
    cvpca.bind(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO,
               VALIDIX=VALIDIX, N=N, MICE=MICE)          # or cvpca.bind_cache(_c, MICE)
    frac = cvpca.avg_frac('Expert', DPA4, AW['md'], nsplits=30, seed=7)

Not converted: `exp_dimensionality.py`, the superseded 12-condition build, whose split_means takes no
condition list and whose spectrum function differs. Leave it alone or port it deliberately.
"""
import numpy as np

_B = {}                       # the bound pseudo-population context


def bind(*, MOUSE, LEARN, LAS, PERF, TSK, SAMP, TESTO, VALIDIX, N, MICE):
    """Bind the pseudo-population label vectors once, before any other call in this module."""
    _B.update(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO,
              VALIDIX=VALIDIX, N=int(N), MICE=list(MICE))


def bind_cache(cache, mice):
    """Bind from a loaded fits_inputs.pkl dict (keys 'L', 'VALIDIX', 'N')."""
    L = cache['L']
    bind(MOUSE=L['MOUSE'], LEARN=L['LEARN'], LAS=L['LAS'], PERF=L['PERF'], TSK=L['TSK'],
         SAMP=L['SAMP'], TESTO=L['TESTO'], VALIDIX=cache['VALIDIX'], N=cache['N'], MICE=mice)


def _ctx():
    if not _B:
        raise RuntimeError('cvpca: call cvpca.bind(...) or cvpca.bind_cache(...) before using it')
    return _B


def trials(stage, cond=None, mouse=None):
    """Indices of the analysed trials: laser off, correct, this stage (optionally this mouse/condition)."""
    b = _ctx()
    k = (b['LEARN'] == stage) & (b['LAS'] == 0) & (b['PERF'] == 1)
    if mouse is not None:
        k &= (b['MOUSE'] == mouse)
    if cond is not None:
        t, s, te = cond
        k &= (b['TSK'] == t) & (b['SAMP'] == s) & (b['TESTO'] == te)
    return np.where(k)[0]


def neuron_scale(stage, M):
    """Per-neuron SD across all of that mouse/stage's analysed trials (1.0 where undefined)."""
    b = _ctx(); sd = np.ones(b['N'])
    for m in b['MICE']:
        val = b['VALIDIX'][(m, stage)]; tr = trials(stage, mouse=m)
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def cond_means(stage, M, conds, mice=None):
    """Condition means over ALL analysed trials — the uncross-validated reference."""
    b = _ctx(); R = np.zeros((len(conds), b['N']))
    for m in (b['MICE'] if mice is None else mice):
        val = b['VALIDIX'][(m, stage)]
        for ci, c in enumerate(conds):
            idx = trials(stage, c, m)
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], axis=0)
    return R


def split_means(stage, conds, M, rng, mice=None, shuffle=False):
    """Two independent condition-mean estimates from disjoint halves of each mouse x condition's trials.

    shuffle=True permutes the trial->condition assignment WITHIN each mouse first (the label-shuffle
    null), keeping the number of trials per condition. A cell with fewer than 2 trials is left as zeros.
    """
    b = _ctx(); n = b['N']
    R1 = np.zeros((len(conds), n)); R2 = np.zeros((len(conds), n))
    for m in (b['MICE'] if mice is None else mice):
        val = b['VALIDIX'][(m, stage)]
        pools = [trials(stage, c, m) for c in conds]
        if shuffle:
            perm = rng.permutation(np.concatenate(pools)); k = 0; new = []
            for p in pools:
                new.append(perm[k:k + len(p)]); k += len(p)
            pools = new
        for ci, idx in enumerate(pools):
            if len(idx) < 2:
                continue
            p = rng.permutation(idx); h = len(p) // 2
            R1[ci][val] = np.nanmean(M[np.ix_(p[:h], val)], 0)
            R2[ci][val] = np.nanmean(M[np.ix_(p[h:], val)], 0)
    return R1, R2


def fold_means(stage, conds, M, rng, K, mice=None):
    """K-fold sibling of split_means: for each fold, (mean of the other K-1 folds, mean of this fold).

    Used by the 5-fold panel-d variant. A mouse x condition cell with fewer than K trials is left as
    zeros, since it cannot fill every fold; the smallest cell in this dataset holds 6 trials.
    """
    b = _ctx(); n = b['N']
    TE = [np.zeros((len(conds), n)) for _ in range(K)]
    TR = [np.zeros((len(conds), n)) for _ in range(K)]
    for m in (b['MICE'] if mice is None else mice):
        val = b['VALIDIX'][(m, stage)]
        for ci, c in enumerate(conds):
            idx = trials(stage, c, m)
            if len(idx) < K:
                continue
            parts = np.array_split(rng.permutation(idx), K)      # sizes differ by at most 1
            for f in range(K):
                rest = np.concatenate([parts[g] for g in range(K) if g != f])
                TE[f][ci][val] = np.nanmean(M[np.ix_(parts[f], val)], 0)
                TR[f][ci][val] = np.nanmean(M[np.ix_(rest, val)], 0)
    return list(zip(TR, TE))


def cvpca_spectrum(S1, S2):
    """Per-component cross-validated (reliable) variance: basis from one half, scores read on both.

    Both halves are centred across conditions first. Returns the average of the two directions; entries
    may be negative, which is the estimator saying that component carries no replicating structure.
    """
    S1 = S1 - S1.mean(0, keepdims=True); S2 = S2 - S2.mean(0, keepdims=True)

    def one(A, B):
        Vt = np.linalg.svd(A, full_matrices=False)[2]
        return ((A @ Vt.T) * (B @ Vt.T)).sum(0)
    a, b = one(S1, S2), one(S2, S1); k = min(len(a), len(b))
    return 0.5 * (a[:k] + b[:k])


def spectra(stage, conds, M, mice=None, nsplits=30, seed=7, shuffle=False):
    """The per-split cross-validated spectra, as a list (raw, may go negative)."""
    sd = neuron_scale(stage, M); rng = np.random.RandomState(seed); out = []
    for _ in range(nsplits):
        R1, R2 = split_means(stage, conds, M, rng, mice, shuffle)
        out.append(cvpca_spectrum(R1 / sd[None, :], R2 / sd[None, :]))
    return out


def avg_spec(stage, conds, M, mice=None, nsplits=30, seed=7, shuffle=False):
    """Split-averaged cross-validated spectrum (raw, may go negative)."""
    sp = spectra(stage, conds, M, mice, nsplits, seed, shuffle)
    return np.sum(sp, axis=0) / len(sp)


def avg_frac(stage, conds, M, mice=None, nsplits=30, seed=7, shuffle=False):
    """Fraction of reliable variance per component — the quantity Fig 2b plots."""
    pos = np.clip(avg_spec(stage, conds, M, mice, nsplits, seed, shuffle), 0, None)
    return pos / (pos.sum() + 1e-12)


def pr_of(cv):
    """Participation ratio of a (clipped) spectrum: (sum)^2 / sum of squares."""
    pos = np.clip(cv, 0, None)
    return float(pos.sum() ** 2 / ((pos ** 2).sum() + 1e-12))
