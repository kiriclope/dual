"""AXIS_FRAME — the geometry of the shared frame used by Fig 3: pairwise angles between the three
task axes (sample @ mid-delay, action/lick @ decision, distractor/gng @ mid-delay), per stage.

Each axis is a design contrast applied to condition means built from ONE trial half; the pairing is
computed across INDEPENDENT halves and then ATTENUATION-CORRECTED by each axis's own split-half
reliability — cos_corr = |cos(a1,b2)| / sqrt(|cos(a1,a2)| * |cos(b1,b2)|). Without that correction
finite-trial noise biases every cosine toward 0, so near-orthogonality would be un-falsifiable.

WHY THIS EXISTS (2026-08-12): cross-decoding between two codes can be high while their axes are far
from parallel — transfer only needs the projected signal to clear the noise. gng decodes at d' ~ 6,
so even |cos| ~ 0.38 predicts ~0.87 cross-accuracy (observed 0.84). The cosine is the geometric
statement; the cross-decode is not. This cache supplies the geometric one.

Output: merges {'AXIS_FRAME': {stage: {'cos': 3x3, 'rel': 3-vector, 'raw': 3x3}}} into results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_axis_frame.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import fit_axis, SUF                 # THE shared decoder (see decoders.py)

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
DPA4 = [c for c in ALL12 if c[0] == 'DPA']
DUAL = [c for c in ALL12 if c[0] != 'DPA']
NREP = 12                    # logistic fits are far costlier than the old contrast axes

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
assert {'md', 'decision'} <= set(AW), 'run exp_dimensionality_md.py first (md window missing)'
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

# the three axes of the frame: (label, window, condition set, contrast)
AXES = [('sample', 'md', DPA4, lambda cs: np.array([2 * c[1] - 1 for c in cs], float)),
        ('action', 'decision', DPA4, lambda cs: np.array([2 * (c[1] == c[2]) - 1 for c in cs], float)),
        ('distractor', 'md', DUAL, lambda cs: np.array([1.0 if c[0] == 'DualGo' else -1.0 for c in cs]))]


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def axis_from_half(stage, conds, M, sd, contrast, H, which, K=60, seed=0):
    """LOGISTIC-REGRESSION axis (decoders.fit_axis) on composite pseudo-trials built from one trial
    half. `contrast` supplies the ±1 class labels; conditions with contrast 0 are dropped."""
    rng = np.random.RandomState(seed + which)
    Xs, ys = [], []
    for ci, cd in enumerate(conds):
        if contrast[ci] == 0:
            continue
        X = np.zeros((K, N))
        for m in MICE:
            val = VALIDIX[(m, stage)]
            idx = H[(m, ALL12.index(cd))][which]
            if not len(idx):
                continue
            cm = np.nanmean(M[np.ix_(idx, val)], 0)
            blk = M[np.ix_(rng.choice(idx, K, replace=True), val)]
            bad = ~np.isfinite(blk)
            if bad.any():
                blk[bad] = np.broadcast_to(cm, blk.shape)[bad]
            X[:, val] = blk
        Xs.append(X / sd[None, :]); ys.append(np.full(K, 1 if contrast[ci] > 0 else 0))
    return fit_axis(np.vstack(Xs), np.concatenate(ys))[0]


OUT = {}
for stage in STAGES:
    SD = {wn: neuron_scale(stage, AW[wn]) for wn in ['md', 'decision']}
    raw = np.zeros((3, 3)); rel = np.zeros(3)
    for rep in range(NREP):
        rng = np.random.RandomState(100 + rep)     # was RandomState(None): panel-C cosines drifted
                                                   # at the 2nd decimal on every rebuild
        H = {}                                              # ONE split shared by all axes this rep
        for m in MICE:
            for ci, cd in enumerate(ALL12):
                idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                               & (TSK == cd[0]) & (SAMP == cd[1]) & (TESTO == cd[2]))[0]
                p = rng.permutation(idx); h = len(p) // 2
                H[(m, ci)] = (p[:h], p[h:])
        W = [[axis_from_half(stage, cs, AW[wn], SD[wn], f(cs), H, k) for k in (0, 1)]
             for (_, wn, cs, f) in AXES]
        for i in range(3):
            rel[i] += abs(W[i][0] @ W[i][1])
            for j in range(3):
                raw[i, j] += 0.5 * (abs(W[i][0] @ W[j][1]) + abs(W[i][1] @ W[j][0]))
    raw /= NREP; rel /= NREP
    cos = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cos[i, j] = 1.0 if i == j else raw[i, j] / np.sqrt(max(rel[i] * rel[j], 1e-9))
    OUT[stage] = dict(cos=cos, rel=rel, raw=raw, labels=[a[0] for a in AXES])
    print(f'{stage}: split-half reliability ' +
          ' '.join(f'{AXES[i][0]}={rel[i]:.2f}' for i in range(3)))
    for i in range(3):
        print('   ' + '  '.join(f'{AXES[i][0][:6]}x{AXES[j][0][:6]}={cos[i, j]:.2f}' for j in range(3)))

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['AXIS_FRAME' + SUF] = OUT
pickle.dump(d, open(RES, 'wb'))
print('\nmerged AXIS_FRAME' + SUF + ' into', RES)
