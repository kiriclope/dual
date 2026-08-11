"""Panel-D inputs at the MID-DELAY window (bins_MD 36-38) — window consistency with Fig 2b/c.

Computes, for DPA (4 conds) and dual (8 conds) x stage, at wn='md':
  - condition-mean PCs (neurons z-scored across condition means, the exp_dpa_gng_column convention),
    eta^2 of each PC on the orthogonal factor contrasts (DPA: sample/test/choice; dual: + gng),
    cm_var = per-PC share of condition-mean variance  -> merged as FITDATA[(set, 'md', stage)]
  - the per-PC gng CROSS-decode column for DPA (dual pseudo-trials projected on DPA PCs, LDA per PC,
    above-chance fraction 2*(bal-acc - 0.5))  -> merged as DPA_GNG[('md', stage)]
Cache-only (~1 min), no X reload.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_pceta_md.py
"""
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
M = AW['md']


def contrasts(conds):
    s = np.array([c[1] for c in conds], float) * 2 - 1
    te = np.array([c[2] for c in conds], float) * 2 - 1
    out = {'sample': s, 'test': te, 'choice': s * te}
    tk = np.array([c[0] for c in conds])
    if 'DualGo' in tk:
        out['gng'] = np.where(tk == 'DualGo', 1.0, -1.0)
    return out


def cond_means(stage, conds):
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


def make_pseudo(pool, stage, conds, K, rng):
    Xp = np.zeros((len(conds) * K, N)); crow = np.repeat(np.arange(len(conds)), K)
    for ci in range(len(conds)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; pi = pool[(ci, m)]
            if len(pi):
                block = M[np.ix_(pi[rng.randint(0, len(pi), K)], val)]
                block[~np.isfinite(block)] = 0.0
                Xp[ci * K:(ci + 1) * K, val] = block
    return Xp, crow


def eta(contrast, zk):
    zc = zk - zk.mean()
    return (contrast @ zc) ** 2 / ((contrast @ contrast) * (zc ** 2).sum() + 1e-12)


d = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
FIT = d['FITDATA']; DPA_GNG = d['DPA_GNG']
NPC = 4
for stage in ['Naive', 'Expert']:
    for conds, sname, forder in [(DPA, 'DPA', ['sample', 'test', 'choice']),
                                 (DUAL, 'dual', ['sample', 'gng', 'test', 'choice'])]:
        R = cond_means(stage, conds); mu = R.mean(0); sd = R.std(0) + 1e-9
        Rc = (R - mu) / sd; Rc = Rc - Rc.mean(0)
        U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
        Z = Rc @ Vt.T
        CN = contrasts(conds)
        pceta = np.column_stack([[eta(CN[f], Z[:, k]) for k in range(NPC)] for f in forder])
        cm_var = np.zeros(NPC); ev = S ** 2 / (S ** 2).sum()
        cm_var[:len(ev)] = ev[:NPC]
        FIT[(sname, 'md', stage)] = dict(pceta=pceta, factors=forder, cm_var=cm_var)
        print(f'{sname:4s} md {stage:6s} cm_var {np.round(cm_var, 2)}')
        for k in range(NPC):
            print(f'   PC{k+1} ' + '  '.join(f'{f}={pceta[k, i]:.2f}' for i, f in enumerate(forder)))
        if sname == 'DPA':                    # per-PC gng cross-decode column at MD
            ac = np.zeros((NPC, 8))
            for b in range(8):
                rng = np.random.RandomState(500 + b)
                trp, tep = pools(stage, DUAL, rng)
                Xtr, ctr = make_pseudo(trp, stage, DUAL, 24, rng)
                Xte, cte = make_pseudo(tep, stage, DUAL, 24, rng)
                Ztr = ((Xtr - mu) / sd) @ Vt[:NPC].T; Zte = ((Xte - mu) / sd) @ Vt[:NPC].T
                for k in range(NPC):
                    clf = LinearDiscriminantAnalysis().fit(Ztr[:, [k]], GLAB[ctr])
                    ac[k, b] = balanced_accuracy_score(GLAB[cte], clf.predict(Zte[:, [k]]))
            DPA_GNG[('md', stage)] = np.clip(2 * (ac.mean(1) - 0.5), 0, 1)
            print(f'   gng x col (above-chance frac): {np.round(DPA_GNG[("md", stage)], 2)}'
                  f'  [bal-acc {np.round(ac.mean(1), 2)}]')

d['FITDATA'] = FIT; d['DPA_GNG'] = DPA_GNG
pickle.dump(d, open('figures/pseudo/dimensionality/results.pkl', 'wb'))
print('merged md pceta/cm_var into FITDATA and md gng column into DPA_GNG')
