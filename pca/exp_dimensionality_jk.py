"""Jackknife-across-mice CIs for the cvPCA participation ratio (Fig 2c error bars).

The split-level PR percentiles (exp_dimensionality_ci.py, PR_CI) quantify trial-split stability only —
they are implausibly tight (Expert delay [2.00, 2.06]) and mix estimators (averaged-spectrum point vs
per-split distribution). The defensible population error bar treats MICE as the exchangeable unit
(neurons partition disjointly by mouse): leave one mouse out (its neurons AND trials), recompute the
averaged cvPCA spectrum -> PR, jackknife SE over the 9 leave-outs, 95% CI = pr +/- 1.96*SE (lower bound
clipped at the PR floor of 1). Merge-dumps PR_JK into results.pkl.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dimensionality_jk.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
ALL12 = [(t, s, te) for t in TASKS3 for s in (0, 1) for te in (0, 1)]
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]

print('loading fits-inputs cache (no 20 GB X reload) …')
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def split_means(stage, conds, M, rng, mice):
    R1 = np.zeros((len(conds), N)); R2 = np.zeros((len(conds), N))
    for m in mice:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx) < 2:
                continue
            p = rng.permutation(idx); h = len(p) // 2
            R1[ci][val] = np.nanmean(M[np.ix_(p[:h], val)], 0); R2[ci][val] = np.nanmean(M[np.ix_(p[h:], val)], 0)
    return R1, R2


def cvpca_spectrum(S1, S2):
    S1 = S1 - S1.mean(0, keepdims=True); S2 = S2 - S2.mean(0, keepdims=True)

    def one(A, B):
        Vt = np.linalg.svd(A, full_matrices=False)[2]
        return ((A @ Vt.T) * (B @ Vt.T)).sum(0)
    a, b = one(S1, S2), one(S2, S1); k = min(len(a), len(b))
    return 0.5 * (a[:k] + b[:k])


def pr_of(cv):
    pos = np.clip(cv, 0, None)
    return float(pos.sum() ** 2 / ((pos ** 2).sum() + 1e-12))


def avg_pr(stage, conds, M, mice, nsplits=20):
    sd = neuron_scale(stage, M); rng = np.random.RandomState(7); spec = None
    for _ in range(nsplits):
        R1, R2 = split_means(stage, conds, M, rng, mice)
        c = cvpca_spectrum(R1 / sd[None, :], R2 / sd[None, :])
        spec = c if spec is None else spec + c
    return pr_of(spec / nsplits)


SETS = {('all', 'delay'): (ALL12, AW['delay']), ('all', 'decision'): (ALL12, AW['decision']),
        ('DPA', 'delay'): (DPA4, AW['delay'])}
print('\n══ jackknife-across-mice PR CIs (leave-one-mouse-out, 20 splits each) ══')
PR_JK = {}
for (ts, wn), (conds, M) in SETS.items():
    for stage in STAGES:
        pr_full = avg_pr(stage, conds, M, MICE)
        jk = np.array([avg_pr(stage, conds, M, [m for m in MICE if m != mo]) for mo in MICE])
        n = len(MICE)
        se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
        ci = [max(1.0, pr_full - 1.96 * se), pr_full + 1.96 * se]
        PR_JK[(ts, wn, stage)] = dict(pr=pr_full, jk=jk, se=se, ci=np.array(ci))
        print(f'  {ts:4s} {wn:9s} {stage:6s}: PR={pr_full:.2f}  SE(jk)={se:.3f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]'
              f'  (leave-outs {jk.min():.2f}-{jk.max():.2f})')

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb')) if os.path.exists(RES) else {}
d['PR_JK'] = PR_JK
pickle.dump(d, open(RES, 'wb'))
print('\nmerged PR_JK into', RES)
