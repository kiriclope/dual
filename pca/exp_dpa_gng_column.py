"""(1) Restore the clean 3-column DPA PC-coding matrices (sample/test/choice within-fit eta^2) and
(2) store a separate DPA_GNG entry = gng cross-DECODING above-chance per DPA PC (2*(bal-acc-0.5), clipped),
which the render can optionally add as a 'gng' column. Idempotent, from the cache (no X)."""
import os, pickle
os.chdir('/home/leon/dual/pca')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

C = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = C['AW']; VALIDIX = C['VALIDIX']; N = C['N']; L = C['L']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (L[k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
DPA = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
DUAL = [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
GLAB = np.array([1 if c[0] == 'DualGo' else 0 for c in DUAL])
_s = np.array([c[1] for c in DPA], float); _t = np.array([c[2] for c in DPA], float)
WCON = {'sample': 2 * _s - 1, 'test': 2 * _t - 1, 'choice': 2 * (_s == _t) - 1}


def cond_means(stage, M, conds):
    R = np.zeros((len(conds), N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], axis=0)
    return R


def pools(stage, conds, rng):
    tr, te = {}, {}
    for m in MICE:
        for ci, (t, s, te_) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te_))[0]
            p = rng.permutation(idx); h = len(p) // 2
            tr[(ci, m)], te[(ci, m)] = p[:h], p[h:]
    return tr, te


def make_pseudo(pool, stage, conds, K, rng, M):
    Xp = np.zeros((len(conds) * K, N)); crow = np.repeat(np.arange(len(conds)), K)
    for ci in range(len(conds)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; pi = pool[(ci, m)]
            if len(pi):
                Xp[ci * K:(ci + 1) * K, val] = M[np.ix_(pi[rng.randint(0, len(pi), K)], val)]
    return Xp, crow


def eta(contrast, zk):
    zc = zk - zk.mean()
    return (contrast @ zc) ** 2 / ((contrast @ contrast) * (zc ** 2).sum() + 1e-12)


d = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); FIT = d['FITDATA']
DPA_GNG = {}
for wn in ['delay', 'decision', 'delay+dec']:
    M = AW[wn]
    for stage in ['Naive', 'Expert']:
        R = cond_means(stage, M, DPA); mu = R.mean(0); sd = R.std(0) + 1e-9
        Rc = (R - mu) / sd; Rc = Rc - Rc.mean(0)
        Vt = np.linalg.svd(Rc, full_matrices=False)[2]; Z = Rc @ Vt.T; npc = 4   # incl degenerate PC4 (~0% var)
        # (1) within-DPA eta^2 (PC4 = null direction → ~0)
        F = FIT[('DPA', wn, stage)]
        F['pceta'] = np.column_stack([[eta(WCON[f], Z[:, k]) for k in range(npc)] for f in ['sample', 'test', 'choice']])
        F['factors'] = ['sample', 'test', 'choice']
        # (2) gng cross-decode above-chance per PC
        ac = np.zeros((npc, 8))
        for b in range(8):
            rng = np.random.RandomState(500 + b)
            trp, tep = pools(stage, DUAL, rng)
            Xtr, ctr = make_pseudo(trp, stage, DUAL, 24, rng, M); Xte, cte = make_pseudo(tep, stage, DUAL, 24, rng, M)
            Ztr = ((Xtr - mu) / sd) @ Vt[:npc].T; Zte = ((Xte - mu) / sd) @ Vt[:npc].T
            for k in range(npc):
                clf = LinearDiscriminantAnalysis().fit(Ztr[:, [k]], GLAB[ctr])
                ac[k, b] = balanced_accuracy_score(GLAB[cte], clf.predict(Zte[:, [k]]))
        DPA_GNG[(wn, stage)] = np.clip(2 * (ac.mean(1) - 0.5), 0, 1)        # above-chance fraction per PC
        print(f'DPA {wn:9s} {stage:6s}: gng decode bal-acc {np.round(ac.mean(1),2)} -> col {np.round(DPA_GNG[(wn,stage)],2)}')
d['DPA_GNG'] = DPA_GNG
pickle.dump(d, open('figures/pseudo/dimensionality/results.pkl', 'wb'))
print('restored clean DPA matrices + stored DPA_GNG (gng cross-decode column)')
