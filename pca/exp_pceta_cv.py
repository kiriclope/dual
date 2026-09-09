"""exp_pceta_cv.py — CROSS-VALIDATED version of Fig 2d (Leon 2026-09-09: "we should cross validate panel d").

Panel d used to PCA the condition means computed from ALL trials, so its per-PC percentages counted the
noise in those means: at the DPA mid-delay it reported PC1 39% / PC2 31% / PC3 29% — three comparable axes —
while the cross-validated spectrum of panel b said one (1.00 / 0 / 0). Same data, different question.

Here each component is FIT ON ONE TRIAL HALF and MEASURED ON THE OTHER, exactly like the panel-b spectrum:

    for each of NSPLIT random half-splits, and in both directions:
        S1, S2  = z-scored, centred condition means of the two disjoint halves
        Vt      = right singular vectors of S1            (the basis comes from half 1 only)
        cross-variance_k = <S1 Vt_k , S2 Vt_k>            (noise does not replicate → ~0)
        eta2_k          = eta² of the HELD-OUT scores S2 Vt_k against the design contrasts

COMPONENT MATCHING (essential — without it the DPA decision matrix is mush). Splits are averaged
componentwise, so the k-th component must mean the same thing in every split. When two components
carry nearly equal variance their ORDER is arbitrary: the DPA decision spectrum is 0.41 / 0.40 / 0.18,
so its first two swap from split to split, and naive averaging blends the choice PC with the sample PC
(PC1 came out 0.67 choice + 0.30 sample, PC2 0.57 sample + 0.35 choice — the two raw rows, mixed in
proportion to how often they swapped). Each split's components are therefore matched to a fixed
reference basis (the full-data PCs of the same normalised condition means) by maximum |cosine|,
with a Hungarian assignment, BEFORE accumulating. The reference only labels the components; it
never enters the fit or the measurement, both of which stay strictly out-of-sample. Matching is a
permutation of the same top-k set, so the variance fractions are unchanged as a multiset and still
reproduce panel b.

Averaged over splits and directions. Writes into each FITDATA entry of results.pkl:
    pceta_cv   (nk × factors)  cross-validated coding matrix, the panel-d cells
    cm_var_cv  (nk,)           cross-validated variance fractions, the panel-d row labels
The uncross-validated `pceta` / `cm_var` stay in place so nothing that reads them breaks.

Run from pca/:  /home/leon/mambaforge/envs/dual/bin/python exp_pceta_cv.py
"""
import os
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NSPLIT = 30
RES = 'figures/pseudo/dimensionality/results.pkl'
AWPKL = 'figures/pseudo/dimensionality/fits_inputs.pkl'
STAGES = ['Naive', 'Expert']
TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
ALL12 = [(t, s, te) for t in TASKS3 for s in (0, 1) for te in (0, 1)]
TASKSETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'],
            'dual': [c for c in ALL12 if c[0] != 'DPA'],
            'all': ALL12}

_c = pickle.load(open(AWPKL, 'rb'))
AW, VALIDIX, N = _c['AW'], _c['VALIDIX'], _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MICE = sorted({m for m, _ in VALIDIX})


def contrasts(conds):
    """orthogonal ±1 factor contrasts — identical to exp_dimensionality_fits.contrasts."""
    s = np.array([c[1] for c in conds], float); te = np.array([c[2] for c in conds], float)
    tk = np.array([TASKS3.index(c[0]) for c in conds])
    C = {'sample': 2 * s - 1, 'test': 2 * te - 1, 'choice': 2 * (s == te) - 1}
    has_dual = bool(np.any((tk == 1) | (tk == 2))); has_dpa = bool(np.any(tk == 0))
    order = ['sample']
    if has_dual:
        C['gng'] = np.select([tk == 1, tk == 2], [1.0, -1.0], default=0.0); order.append('gng')
    order += ['test', 'choice']
    if has_dpa and has_dual:
        C['tasks'] = np.select([tk == 0, tk == 1, tk == 2], [2.0, -1.0, -1.0]); order.append('tasks')
    return C, order


def eta2(zk, C, order):
    zc = zk - zk.mean(); sst = (zc ** 2).sum() + 1e-12
    return [(C[f] @ zc) ** 2 / ((C[f] @ C[f]) * sst) for f in order]


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def split_means(stage, conds, M, rng):
    R1 = np.zeros((len(conds), N)); R2 = np.zeros((len(conds), N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx) < 2:
                continue
            p = rng.permutation(idx); h = len(p) // 2
            R1[ci][val] = np.nanmean(M[np.ix_(p[:h], val)], 0); R2[ci][val] = np.nanmean(M[np.ix_(p[h:], val)], 0)
    return R1, R2


def ref_basis(stage, conds, M, nk):
    """LABELLING basis only: the full-data PCs of the same normalised condition means."""
    sd = neuron_scale(stage, M)
    R = np.zeros((len(conds), N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], 0)
    S = R / sd[None, :]; S = S - S.mean(0, keepdims=True)
    return np.linalg.svd(S, full_matrices=False)[2][:nk]


def cv_pc(stage, conds, M, C, order, nk):
    """cross-validated (variance fraction, eta² matrix): basis from one half, both read on the other,
    components matched to a fixed reference so that averaging across splits compares like with like."""
    sd = neuron_scale(stage, M); rng = np.random.RandomState(0)
    Vref = ref_basis(stage, conds, M, nk)
    var = np.zeros(nk); eta = np.zeros((nk, len(order))); n = 0; nperm = 0
    for _ in range(NSPLIT):
        R1, R2 = split_means(stage, conds, M, rng)
        S1, S2 = R1 / sd[None, :], R2 / sd[None, :]
        S1 = S1 - S1.mean(0, keepdims=True); S2 = S2 - S2.mean(0, keepdims=True)
        for A, B in ((S1, S2), (S2, S1)):                       # both directions, as in cvpca_spectrum
            Vt = np.linalg.svd(A, full_matrices=False)[2][:nk]
            ri, ci = linear_sum_assignment(-np.abs(Vref @ Vt.T))  # ref slot ri <- fitted component ci
            nperm += int(not np.array_equal(ci, np.arange(nk)))
            ZA, ZB = A @ Vt.T, B @ Vt.T                          # scores of the fit half and the held-out half
            var[ri] += (ZA[:, ci] * ZB[:, ci]).sum(0)            # replicating variance per component
            for r, c in zip(ri, ci):
                eta[r] += eta2(ZB[:, c], C, order)               # coding measured on the HELD-OUT half
            n += 1
    pos = np.clip(var / n, 0, None)
    return pos / (pos.sum() + 1e-12), eta / n, nperm / n


d = pickle.load(open(RES, 'rb'))
F = d['FITDATA']
print(f'{"set":5s}{"window":10s}{"stage":7s}{"raw %":24s}{"cross-validated %":24s}{"perm":5s}  top factor per PC')
for tsname, conds in TASKSETS.items():
    C, order = contrasts(conds); nk = len(conds) - 1
    for wn, M in AW.items():
        for stage in STAGES:
            key = (tsname, wn, stage)
            if key not in F:
                continue
            cmv, pce, fperm = cv_pc(stage, conds, M, C, order, nk)
            F[key]['cm_var_cv'] = cmv; F[key]['pceta_cv'] = pce; F[key]['pceta_cv_factors'] = order
            F[key]['pceta_cv_permfrac'] = fperm
            if wn in ('md', 'decision') and stage == 'Expert' and tsname != 'all':
                raw = np.asarray(F[key]['cm_var'], float)[:3]
                top = [f'{order[int(np.argmax(pce[k]))]} {pce[k].max():.2f}' for k in range(min(3, nk))]
                print(f'{tsname:5s}{wn:10s}{stage:7s}{str([f"{v:.0%}" for v in raw]):24s}'
                      f'{str([f"{v:.0%}" for v in cmv[:3]]):24s}{fperm:5.2f}  {top}')
pickle.dump(d, open(RES, 'wb'))
print(f'\nmerged pceta_cv / cm_var_cv into {RES} ({NSPLIT} splits × 2 directions)')
