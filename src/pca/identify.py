"""
Identify which task variable each PC carries.

Over the four odor pairs (0=AC, 1=AD, 2=BD, 3=BC) the three task variables are
mutually orthogonal ±1 contrasts:

    Sample (A vs B)        : {0,1} vs {2,3}
    Choice (lick vs no)    : {0,2} vs {1,3}
    Test   (C vs D)        : {0,3} vs {1,2}

Which PC expresses which contrast is RUN-DEPENDENT (e.g. a DELAY-epoch fit puts
Choice on PC1 and Sample on PC3, whereas a TEST-epoch fit orders them
differently), so it must be measured from the projected trajectories rather than
assumed.

Each contrast is scored in the window where it is behaviourally expressed
(Sample in the delay, Choice/Test at test), then each variable is assigned 1:1
to the PC carrying the largest fraction of its contrast energy.
"""

import numpy as np

# ±1 contrast weights over odor_pair 0..3
CONTRASTS = {
    'Sample': {0: +1, 1: +1, 2: -1, 3: -1},   # A vs B
    'Choice': {0: +1, 1: -1, 2: +1, 3: -1},   # lick vs no-lick
    'Test':   {0: +1, 1: -1, 2: -1, 3: +1},   # C vs D
}

# default analysis windows (project-standard 84-bin layout, 6 Hz)
DEFAULT_WINDOWS = {
    'Sample': np.arange(18, 54),    # BINS_DELAY  — sample held in memory
    'Choice': np.arange(54, 60),    # BINS_TEST   — lick decision at test
    'Test':   np.arange(54, 60),    # BINS_TEST   — test odor presented
}

_BL = slice(0, 12)   # pre-stim baseline


def pc_contrast_scores(X, y, windows=None, stage='Expert', bl_correct=True):
    """Per-PC contrast score for each task variable.

    X : (n_trials, n_comp, n_time)   projected trajectories
    y : DataFrame with laser, learning, performance, odor_pair
    Returns {variable: np.ndarray(n_comp)}.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if bl_correct:
        X = X - X[:, :, _BL].mean(axis=2, keepdims=True)

    mask = (y.laser == 0) & (y.learning == stage) & (y.performance == 1)
    n_comp = X.shape[1]
    scores = {}
    for v, w in CONTRASTS.items():
        win = windows[v]
        cond_mean = np.zeros((4, n_comp))
        for p in range(4):
            sel = (mask & (y.odor_pair == p)).to_numpy()
            if sel.sum() == 0:
                continue
            cond_mean[p] = X[sel][:, :, win].mean(axis=(0, 2))
        weight = np.array([w[p] for p in range(4)])[:, None]
        scores[v] = (cond_mean * weight).sum(0)
    return scores


def identify_pcs(X, y, windows=None, stage='Expert', bl_correct=True):
    """Assign each task variable 1:1 to the PC carrying most of its contrast.

    Returns a list of length n_comp; entry k is the variable name carried by
    PC k (0-indexed), or None if no variable was assigned to it.
    """
    scores = pc_contrast_scores(X, y, windows, stage, bl_correct)
    n_comp = X.shape[1]

    # rank (variable, pc) candidates by the fraction of the variable's contrast
    # energy that lands on that PC (comparable across variables)
    cand = []
    for v, s in scores.items():
        frac = s ** 2 / (np.sum(s ** 2) + 1e-12)
        for k in range(n_comp):
            cand.append((frac[k], v, k))
    cand.sort(reverse=True)

    labels = [None] * n_comp
    used_v, used_k = set(), set()
    for _, v, k in cand:
        if v in used_v or k in used_k:
            continue
        labels[k] = v
        used_v.add(v)
        used_k.add(k)
        if len(used_v) == len(scores):
            break
    return labels


def pc_label(k, labels, prefix='PC'):
    """'PC 1 (Choice)' style label for 0-indexed PC k, or 'PC 1' if unassigned."""
    base = f'{prefix} {k + 1}'
    if labels is not None and k < len(labels) and labels[k]:
        return f'{base} ({labels[k]})'
    return base


# ── mixing between task components ──────────────────────────────────────────────

def coding_vectors(X, y, windows=None, stage='Expert', bl_correct=True):
    """(n_var, n_comp) signed contrast score per variable per PC.

    Row v is variable v's 'coding vector': the direction in PC space along which
    that task variable is expressed.  Returns (C, variable_names).
    """
    scores = pc_contrast_scores(X, y, windows, stage, bl_correct)
    names = list(CONTRASTS.keys())               # Sample, Choice, Test
    C = np.array([scores[v] for v in names])
    return C, names


def variable_mixing(X, y, windows=None, stage='Expert', bl_correct=True):
    """Mixing between task variables = |cos| between their coding vectors.

    Returns
    -------
    M     : (n_var, n_var)  |cosine| similarity; 1 on the diagonal, off-diagonal
            is how much two variables share PC-space directions (0 = demixed).
    C     : (n_var, n_comp) coding vectors
    names : list of variable names (row/col order of M)
    """
    C, names = coding_vectors(X, y, windows, stage, bl_correct)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    M = np.abs(Cn @ Cn.T)
    return M, C, names


def participation_ratio(C):
    """Per-variable effective number of PCs it spreads over: (Σs²)² / Σs⁴.

    1.0 → the variable lives on a single PC (demixed); larger → spread across PCs.
    """
    s2 = C ** 2
    return (s2.sum(1) ** 2) / ((s2 ** 2).sum(1) + 1e-12)


def mixing_index(M):
    """Scalar summary: mean off-diagonal |cos| over variable pairs (0 = demixed)."""
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(M[iu].mean())


def coding_vectors_time(X, y, stage='Expert', bl_correct=True):
    """Per-time-bin contrast score: (n_var, n_comp, n_time).

    Like `coding_vectors` but evaluated at every time bin (no window averaging),
    so the coding direction can be tracked through the trial.
    """
    if bl_correct:
        X = X - X[:, :, _BL].mean(axis=2, keepdims=True)
    mask = (y.laser == 0) & (y.learning == stage) & (y.performance == 1)
    n_comp, n_time = X.shape[1], X.shape[2]
    cond = np.zeros((4, n_comp, n_time))
    for p in range(4):
        sel = (mask & (y.odor_pair == p)).to_numpy()
        if sel.sum():
            cond[p] = X[sel].mean(0)
    names = list(CONTRASTS.keys())
    C = np.zeros((len(names), n_comp, n_time))
    for i, v in enumerate(names):
        w = np.array([CONTRASTS[v][p] for p in range(4)], dtype=float)
        C[i] = np.tensordot(w, cond, axes=(0, 0))    # (n_comp, n_time)
    return C, names


def variable_mixing_time(X, y, stage='Expert', bl_correct=True):
    """Time-resolved variable mixing.

    Returns
    -------
    M     : (n_var, n_var, n_time)  |cos| between coding vectors at each bin
    C     : (n_var, n_comp, n_time) per-bin coding vectors
    energy: (n_var, n_time)         coding-vector norm per bin (signal strength;
            |cos| is ill-defined where this is ~0, e.g. pre-stimulus)
    names : list of variable names
    """
    C, names = coding_vectors_time(X, y, stage, bl_correct)
    energy = np.linalg.norm(C, axis=1)               # (n_var, n_time)
    Cn = C / (energy[:, None, :] + 1e-12)
    M = np.abs(np.einsum('ikt,jkt->ijt', Cn, Cn))
    return M, C, energy, names
