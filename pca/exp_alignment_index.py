"""AI cache — Elsayed-style subspace ALIGNMENT INDEX for Fig 3 panel F (user routing 2026-09-01:
"relevant for main"): are Naive and Expert condition-mean subspaces the SAME subspace?

AI(A→B) = the fraction of stage-B condition-mean variance captured by stage-A's top-K principal
subspace, normalised by the most it could capture (its own top-K); symmetrized over directions.
NULL (Elsayed et al. 2016): random K-dim subspaces drawn with probability proportional to the
POOLED covariance spectrum (v ~ N(0, C_pooled), orthonormalised) — random subspaces in a
correlated space capture far more variance than isotropic ones, so this null is the honest floor.
Condition means: 12 task conditions × N neurons at the combined delay+decision window (correct
laser-off trials, per-mouse neuron fill as everywhere), centred across conditions. K = 3 (the
reliable decision rank; K = 2 printed for the plane). Also computes DPA-vs-dual AI at md.

Merges {'AI'+SUF}: dict(cross_stage={K: (ai, null_mean, null95)}, dpa_dual_md={...}).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_alignment_index.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
NNULL = 1000

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def cond_means(stage, wn, conds):
    M = AW[wn]; R = np.zeros((len(conds), N))
    for ci, (t, s, te) in enumerate(conds):
        for m in MICE:
            val = VALIDIX[(m, stage)]
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], 0)
    return R - R.mean(0, keepdims=True)


def topk(R, K):
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    return Vt[:K].T, (S ** 2)                        # (N, K) basis; squared singular values


def ai_pair(RA, RB, K, rng):
    QA, _ = topk(RA, K); QB, sB = topk(RB, K)
    CB = RB.T @ RB
    ai_ab = np.trace(QA.T @ CB @ QA) / sB[:K].sum()
    QB2, _ = topk(RB, K); CA = RA.T @ RA
    _, sA = topk(RA, K)
    ai_ba = np.trace(QB2.T @ CA @ QB2) / sA[:K].sum()
    ai = 0.5 * (ai_ab + ai_ba)
    # Elsayed null: random subspaces ~ pooled covariance spectrum
    Rp = np.vstack([RA, RB]); Up, Sp, Vp = np.linalg.svd(Rp, full_matrices=False)
    L = Vp.T * Sp[None, :]                           # (N, r) covariance-shaping map
    nulls = []
    for _ in range(NNULL):
        W = L @ rng.randn(L.shape[1], K)
        Q, _ = np.linalg.qr(W)
        a = np.trace(Q.T @ CB @ Q) / sB[:K].sum()
        b = np.trace(Q.T @ CA @ Q) / sA[:K].sum()
        nulls.append(0.5 * (a + b))
    nulls = np.asarray(nulls)
    return float(ai), float(nulls.mean()), float(np.percentile(nulls, 95))


def scaled(R):
    sd = R.std(0); return R / np.where(sd > 1e-9, sd, 1.0)


rng = np.random.RandomState(1100)
print(f'== ALIGNMENT INDEX v2: per-neuron scaled, per-window, rank-matched (SUF="{SUF}") ==', flush=True)
OUT = {'cross_stage': {}, 'dpa_dual_md': {}}
for wn, K in (('md', 2), ('decision', 3), ('delay+dec', 3)):
    RN = scaled(cond_means('Naive', wn, ALL12))
    RE = scaled(cond_means('Expert', wn, ALL12))
    ai, nm, n95 = ai_pair(RN, RE, K, rng)
    OUT['cross_stage'][(wn, K)] = (ai, nm, n95)
    print(f'  cross-stage {wn:10s} K={K}: AI {ai:.2f}  null mean {nm:.2f}  null95 {n95:.2f}', flush=True)
DPA4 = [c for c in ALL12 if c[0] == 'DPA']; DUAL8 = [c for c in ALL12 if c[0] != 'DPA']
for stage in ['Naive', 'Expert']:
    RD = scaled(cond_means(stage, 'md', DPA4)); RU = scaled(cond_means(stage, 'md', DUAL8))
    ai, nm, n95 = ai_pair(RD, RU, 1, rng)          # rank-matched: DPA delay is rank-1
    OUT['dpa_dual_md'][stage] = (ai, nm, n95)
    print(f'  DPA-vs-dual @md {stage} K=2: AI {ai:.2f}  null mean {nm:.2f}  null95 {n95:.2f}', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['AI' + SUF] = OUT
pickle.dump(d, open(RES, 'wb'))
print('merged AI' + SUF, 'into', RES)
