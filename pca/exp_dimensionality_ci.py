"""Hardening stats for the dimensionality main figure (Fig 2) — runs from the fits_inputs.pkl cache
(no 20 GB X reload):

  1. FULL shattering: all 462 balanced dichotomies of the 12 conditions at the decision window
     (57-65), Naive & Expert, real + shuffle null, B=8 pseudo-population resamples. The published SD
     was a 150-dichotomy sample; the full set removes the sampling footnote and gives a per-resample
     spread for a descriptive CI.
  2. PR bootstrap CIs: the cvPCA participation ratio recomputed on each of 30 random half-splits →
     percentile CI of the split-level PRs, for (all-12 delay), (all-12 decision), (DPA delay),
     Naive & Expert. Point estimates stay the averaged-spectrum PRs (exp_dimensionality.py /
     exp_dimensionality_fits.py); these CIs quantify split-to-split spread only.

Merge-dumps SD_FULL and PR_CI into figures/pseudo/dimensionality/results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dimensionality_ci.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvpca                       # THE cvPCA estimator — one implementation (2026-09-09)
from itertools import combinations
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

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
cvpca.bind(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO,
           VALIDIX=VALIDIX, N=N, MICE=MICE)   # the cvPCA estimator lives in cvpca.py (one copy)


# ══ 1. FULL 462-dichotomy shattering (decision window) ═══════════════════════
def all_dich(nc=12):
    seen, out = set(), []
    for s in combinations(range(nc), nc // 2):
        s = frozenset(s); key = min(tuple(sorted(s)), tuple(sorted(set(range(nc)) - s)))
        if key not in seen:
            seen.add(key); out.append(list(s))
    return out                                                             # 462 for nc=12


def cond_pools(stage, conds, rng):
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


def shatter_full(stage, dich, K=24, B=8, null=False):
    """(ndich, B) balanced accuracies — per-dichotomy × per-pseudo-population-resample."""
    acc = np.zeros((len(dich), B)); M = AW['decision']
    for b in range(B):
        rng = np.random.RandomState(2000 + b)
        trp, tep = cond_pools(stage, ALL12, rng)
        Xtr, ctr = make_pseudo(trp, stage, ALL12, K, rng, M); Xte, cte = make_pseudo(tep, stage, ALL12, K, rng, M)
        if null:
            ctr = rng.permutation(ctr); cte = rng.permutation(cte)
        pre = make_pipeline(StandardScaler(), PCA(min(30, Xtr.shape[0] - 1), random_state=0)).fit(Xtr)
        Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
        for di, plus in enumerate(dich):
            clf = LinearDiscriminantAnalysis().fit(Ztr, np.isin(ctr, plus).astype(int))
            acc[di, b] = balanced_accuracy_score(np.isin(cte, plus).astype(int), clf.predict(Zte))
    return acc


print('\n══ 1. FULL shattering (462 dichotomies, decision 57-65) ══')
DICH462 = all_dich()
SD_FULL = {}
for stage in STAGES:
    acc = shatter_full(stage, DICH462); nul = shatter_full(stage, DICH462, null=True)
    per_boot = acc.mean(0)                                                 # mean over dichotomies, per resample
    ci = np.percentile(per_boot, [2.5, 97.5])
    SD_FULL[stage] = dict(acc=acc.mean(1), acc_boot=per_boot, null=nul.mean(1), ndich=len(DICH462),
                          ci=ci)
    print(f'  {stage}: SD = {acc.mean():.3f}  [resample CI {ci[0]:.3f}, {ci[1]:.3f}]  null {nul.mean():.3f}'
          f'  (range over dichotomies {acc.mean(1).min():.2f}-{acc.mean(1).max():.2f})')
dE, dN = SD_FULL['Expert']['acc_boot'], SD_FULL['Naive']['acc_boot']
dd = dE.mean() - dN.mean()
print(f'  Expert−Naive Δ = {dd:+.3f} (per-resample spread ±{np.sqrt(dE.var()+dN.var()):.3f}) — descriptive')

# ══ 2. cvPCA PR split-level CIs ══════════════════════════════════════════════




PR_CI = {}
SETS = {('all', 'delay'): (ALL12, AW['delay']), ('all', 'decision'): (ALL12, AW['decision']),
        ('DPA', 'delay'): (DPA4, AW['delay'])}
for (ts, wn), (conds, M) in SETS.items():
    for stage in STAGES:
        sp = cvpca.spectra(stage, conds, M, nsplits=30, seed=0)
        prs = [cvpca.pr_of(c) for c in sp]
        pr_point = cvpca.pr_of(np.sum(sp, axis=0) / 30)                    # averaged-spectrum PR (the headline)
        ci = np.percentile(prs, [2.5, 97.5])
        PR_CI[(ts, wn, stage)] = dict(pr=pr_point, pr_splits=np.array(prs), ci=ci)
        print(f'  {ts:4s} {wn:9s} {stage:6s}: PR={pr_point:.2f}  [split CI {ci[0]:.2f}, {ci[1]:.2f}]')

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb')) if os.path.exists(RES) else {}
d['SD_FULL'] = SD_FULL; d['PR_CI'] = PR_CI
pickle.dump(d, open(RES, 'wb'))
print('\nmerged SD_FULL + PR_CI into', RES)
