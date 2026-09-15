"""Per-animal companion to the shattering dimension (ED 2c) — reviewer item 2026-09-15.

ED 2c's interval is a pseudo-population RESAMPLE interval of the same nine mice, not an across-animal one. Two
animal-level views of the same estimator (exp_dimensionality_ci.py: 462 balanced dichotomies of the 12 conditions,
decision window 54-62, disjoint train/test halves per mouse x condition, 24 pseudo-trials per condition, scaler +
PCA(30) on the training half, LDA per dichotomy):

  1. SD_LOO  — leave-one-mouse-out jackknife of the pseudo-population SD: the estimator re-run on the eight
               remaining mice, for each held-out mouse and stage (B=8 resamples each, real labels only — the
               shuffle floor 0.50 is established by SD_FULL). Gives a jackknife SE / t(8) for each stage and for
               the Expert-Naive difference (paired over held-out mice).
  2. SD_MOUSE — each mouse's OWN simultaneously recorded population (same design without pseudo-trial
               resampling: real trials, repeated stratified half-splits, B=20), Naive and Expert. The
               animal-level test of "unchanged by learning" is a Wilcoxon over the nine mice. Absolute values
               are lower than the pseudo-population's (fewer trials, fewer neurons) — read the stage contrast,
               not the level.

Merge-dumps SD_LOO and SD_MOUSE into figures/pseudo/dimensionality/results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_shatter_permouse.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from scipy.stats import wilcoxon, t as tdist
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
ALL12 = [(t, s, te) for t in TASKS3 for s in (0, 1) for te in (0, 1)]
NJOBS = 36

print('loading fits-inputs cache (no 20 GB X reload) …')
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
VALIDIX = _c['VALIDIX']; N = _c['N']; M = np.ascontiguousarray(_c['AW']['decision'])
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
del _c


def all_dich(nc=12):
    seen, out = set(), []
    for s in combinations(range(nc), nc // 2):
        s = frozenset(s); key = min(tuple(sorted(s)), tuple(sorted(set(range(nc)) - s)))
        if key not in seen:
            seen.add(key); out.append(list(s))
    return out


DICH462 = all_dich()
COND_IDX = {(m, st, ci): np.where((MOUSE == m) & (LEARN == st) & (LAS == 0) & (PERF == 1)
                                  & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            for m in MICE for st in STAGES for ci, (t, s, te) in enumerate(ALL12)}


def decode_all(Xtr, ctr, Xte, cte):
    pre = make_pipeline(StandardScaler(), PCA(min(30, Xtr.shape[0] - 1), random_state=0)).fit(Xtr)
    Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
    acc = np.zeros(len(DICH462))
    for di, plus in enumerate(DICH462):
        clf = LinearDiscriminantAnalysis().fit(Ztr, np.isin(ctr, plus).astype(int))
        acc[di] = balanced_accuracy_score(np.isin(cte, plus).astype(int), clf.predict(Zte))
    return acc


# ── 1. leave-one-mouse-out pseudo-population (exp_dimensionality_ci.shatter_full on 8 mice) ────────────
def shatter_loo(stage, hold, K=24, B=8):
    mice = [m for m in MICE if m != hold]
    cols = np.sort(np.concatenate([VALIDIX[(m, stage)] for m in mice]))
    out = np.zeros((len(DICH462), B))
    for b in range(B):
        rng = np.random.RandomState(2000 + b)
        Xtr = np.zeros((12 * K, N)); Xte = np.zeros((12 * K, N)); crow = np.repeat(np.arange(12), K)
        for ci in range(12):
            for m in MICE:                                   # held-out mouse still consumes its rng draws
                p = rng.permutation(COND_IDX[(m, stage, ci)]); h = len(p) // 2
                if m == hold:
                    continue
                val = VALIDIX[(m, stage)]
                for X, pi in ((Xtr, p[:h]), (Xte, p[h:])):
                    if len(pi):
                        X[ci * K:(ci + 1) * K, val] = M[np.ix_(pi[rng.randint(0, len(pi), K)], val)]
        out[:, b] = decode_all(Xtr[:, cols], crow, Xte[:, cols], crow)
    return stage, hold, out


# ── 2. each mouse's own population, real trials, repeated half-splits ──────────────────────────────────
def shatter_mouse(stage, m, B=20):
    val = VALIDIX[(m, stage)]; pools = [COND_IDX[(m, stage, ci)] for ci in range(12)]
    out = np.zeros((len(DICH462), B))
    for b in range(B):
        rng = np.random.RandomState(3000 + b); tr, te, ctr, cte = [], [], [], []
        for ci, idx in enumerate(pools):
            p = rng.permutation(idx); h = len(p) // 2
            tr += list(p[:h]); te += list(p[h:]); ctr += [ci] * h; cte += [ci] * (len(p) - h)
        out[:, b] = decode_all(M[np.ix_(tr, val)], np.array(ctr), M[np.ix_(te, val)], np.array(cte))
    return stage, m, out


print(f'\n══ 1. leave-one-mouse-out shattering (18 jobs × 8 resamples × 462 dichotomies, {NJOBS} workers) ══')
res = Parallel(n_jobs=NJOBS, verbose=5)(delayed(shatter_loo)(st, m) for st in STAGES for m in MICE)
SD_LOO = {'acc': {(st, m): a for st, m, a in res}}
for st in STAGES:
    th = np.array([SD_LOO['acc'][(st, m)].mean() for m in MICE]); n = len(th)
    se = np.sqrt((n - 1) / n * ((th - th.mean()) ** 2).sum())
    SD_LOO[st] = dict(loo=th, mean=th.mean(), se=se, ci=(th.mean() - tdist.ppf(.975, n - 1) * se,
                                                        th.mean() + tdist.ppf(.975, n - 1) * se))
    print(f'  {st:6s}: LOO mean {th.mean():.3f}  jackknife SE {se:.4f}  95% CI [{SD_LOO[st]["ci"][0]:.3f}, '
          f'{SD_LOO[st]["ci"][1]:.3f}]  range over held-out mice {th.min():.3f}-{th.max():.3f}')
dl = SD_LOO['Expert']['loo'] - SD_LOO['Naive']['loo']; n = len(dl)
se = np.sqrt((n - 1) / n * ((dl - dl.mean()) ** 2).sum()); tval = dl.mean() / se
p = 2 * tdist.sf(abs(tval), n - 1)
SD_LOO['delta'] = dict(loo=dl, mean=dl.mean(), se=se, t=tval, p=p,
                       ci=(dl.mean() - tdist.ppf(.975, n - 1) * se, dl.mean() + tdist.ppf(.975, n - 1) * se))
print(f'  Expert−Naive Δ = {dl.mean():+.4f}  jackknife SE {se:.4f}  t(8) = {tval:+.2f}  p = {p:.3f}  '
      f'CI [{SD_LOO["delta"]["ci"][0]:+.3f}, {SD_LOO["delta"]["ci"][1]:+.3f}]')

print(f'\n══ 2. per-mouse own-population shattering (18 jobs × 20 half-splits × 462 dichotomies) ══')
res = Parallel(n_jobs=NJOBS, verbose=5)(delayed(shatter_mouse)(st, m) for st in STAGES for m in MICE)
SD_MOUSE = {'acc': {(st, m): a for st, m, a in res}}
for st in STAGES:
    SD_MOUSE[st] = np.array([SD_MOUSE['acc'][(st, m)].mean() for m in MICE])
ntr = {(st, m): int(sum(len(COND_IDX[(m, st, ci)]) for ci in range(12))) for st in STAGES for m in MICE}
SD_MOUSE['ntrials'] = ntr; SD_MOUSE['nneurons'] = {(st, m): len(VALIDIX[(m, st)]) for st in STAGES for m in MICE}
for i, m in enumerate(MICE):
    print(f'  {m:8s} naive {SD_MOUSE["Naive"][i]:.3f} ({ntr[("Naive", m)]} tr)  expert {SD_MOUSE["Expert"][i]:.3f} '
          f'({ntr[("Expert", m)]} tr)  Δ {SD_MOUSE["Expert"][i] - SD_MOUSE["Naive"][i]:+.3f}')
d = SD_MOUSE['Expert'] - SD_MOUSE['Naive']; w = wilcoxon(SD_MOUSE['Expert'], SD_MOUSE['Naive'])
SD_MOUSE['delta'] = dict(mean=d.mean(), median=np.median(d), p=float(w.pvalue), npos=int((d > 0).sum()))
print(f'  median naive {np.median(SD_MOUSE["Naive"]):.3f}  expert {np.median(SD_MOUSE["Expert"]):.3f}  '
      f'Δ median {np.median(d):+.3f}  Wilcoxon p = {w.pvalue:.3f}  {(d > 0).sum()}/9 up')

RES = 'figures/pseudo/dimensionality/results.pkl'
r = pickle.load(open(RES, 'rb')) if os.path.exists(RES) else {}
r['SD_LOO'] = SD_LOO; r['SD_MOUSE'] = SD_MOUSE
pickle.dump(r, open(RES, 'wb'))
print('\nmerged SD_LOO + SD_MOUSE into', RES)
