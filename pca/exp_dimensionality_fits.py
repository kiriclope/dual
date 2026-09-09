"""Per-fit dimensionality: for each (task-set × window × stage) compute the condition-mean scree + PR and the
shattering SD (mean balanced-accuracy over balanced dichotomies). Windows: delay(48-53), decision(57-65),
delay+dec wide(48-65). Task-sets: dual (8 conds), DPA (4 conds). Merges FITDATA into results.pkl."""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir('/home/leon/dual/pca')
import numpy as np
import cvpca                       # THE cvPCA estimator — one implementation (2026-09-09)
from itertools import combinations
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from src.pca.io import pkl_load
from src.common.options import set_options

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
o = set_options()
ALTWIN = '--altwin' in sys.argv                                           # robustness: full delay + test windows
SUF = '_altwin' if ALTWIN else ''
if ALTWIN:                                                                # delay = full delay (21-53); decision = test (57-59)
    WINS = {'delay': np.asarray(o['bins_DELAY']), 'decision': np.asarray(o['bins_TEST']),
            'delay+dec': np.concatenate([np.asarray(o['bins_DELAY']), np.asarray(o['bins_TEST'])])}
else:                                                                     # default: delay = LD (48-53); decision = 57-65
    WINS = {'delay': np.asarray(o['bins_LD']), 'decision': np.arange(54, 63), 'delay+dec': np.arange(48, 63)}   # decision = canonical choice window (9.0–10.5 s) since 2026-09-08
# DUAL_AXSUF + DUAL_FITS_WINS='name:a-b,name2:c-d' (2026-09-08): extra/overriding windows for a variant cache
# fits_inputs{SUF}.pkl / results{SUF}.pkl (one raw-tensor pass for several candidate windows).
if os.environ.get('DUAL_AXSUF'):
    SUF = os.environ['DUAL_AXSUF']
    for _spec in [t for t in os.environ.get('DUAL_FITS_WINS', '').split(',') if t]:
        _nm, _rng = _spec.split(':'); _a, _b = [int(v) for v in _rng.split('-')]; WINS[_nm] = np.arange(_a, _b + 1)
# Persistent cache of the window-averaged pseudo-population matrices + labels, so re-analysis never reloads
# the 20 GB X. Separate cache file per window-set. Built once (first run), then loaded in seconds.
AWPKL = f'figures/pseudo/dimensionality/fits_inputs{SUF}.pkl'
if os.path.exists(AWPKL) and set(pickle.load(open(AWPKL, 'rb'))['AW']) >= set(WINS):
    print('loading fits-inputs cache (no 20 GB X reload) …')
    _c = pickle.load(open(AWPKL, 'rb'))
    AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
    MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                                 ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
else:
    print('loading pseudo-population (one-time; caching window matrices for future runs) …')
    X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
    y = pkl_load('y_all_nan_', path='../data/pca')
    VALID = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
                     path='../data/overlaps')['valid']
    N = X.shape[1]
    MOUSE = y.mouse.to_numpy(); LEARN = y.learning.to_numpy(); LAS = y.laser.to_numpy()
    TSK = y.tasks.to_numpy(); SAMP = y.sample_odor.to_numpy(); TESTO = y.test_odor.to_numpy(); PERF = y.performance.to_numpy()
    VALIDIX = {k: np.where(np.asarray(v))[0] for k, v in VALID.items()}
    AW = {w: np.nanmean(X[:, :, b], axis=2) for w, b in WINS.items()}
    del X
    pickle.dump({'AW': AW, 'VALIDIX': VALIDIX, 'N': N,
                 'L': {'MOUSE': MOUSE, 'LEARN': LEARN, 'LAS': LAS, 'TSK': TSK, 'SAMP': SAMP,
                       'TESTO': TESTO, 'PERF': PERF}}, open(AWPKL, 'wb'))
    print('cached fits inputs →', os.path.abspath(AWPKL))

cvpca.bind(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO,
           VALIDIX=VALIDIX, N=N, MICE=MICE)   # the cvPCA estimator lives in cvpca.py (one copy)

TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
DUAL = [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
DPA = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
ALL12 = [(t, s, te) for t in TASKS3 for s in (0, 1) for te in (0, 1)]


def contrasts(conds):
    """orthogonal ±1 factor contrasts. task splits into gng (Go-vs-NoGo, DPA=0) + tasks (DPA-vs-Dual)."""
    s = np.array([c[1] for c in conds], float); te = np.array([c[2] for c in conds], float)
    tk = np.array([TASKS3.index(c[0]) for c in conds])                     # 0=DPA 1=Go 2=NoGo
    C = {'sample': 2 * s - 1, 'test': 2 * te - 1, 'choice': 2 * (s == te) - 1}
    has_dual = bool(np.any((tk == 1) | (tk == 2))); has_dpa = bool(np.any(tk == 0))
    order = ['sample']
    if has_dual:
        C['gng'] = np.select([tk == 1, tk == 2], [1.0, -1.0], default=0.0); order.append('gng')
    order += ['test', 'choice']
    if has_dpa and has_dual:
        C['tasks'] = np.select([tk == 0, tk == 1, tk == 2], [2.0, -1.0, -1.0]); order.append('tasks')
    return C, order


TASKSETS = {'DPA': DPA, 'dual': DUAL, 'all': ALL12}



def eta2(zk, C, order):
    zc = zk - zk.mean(); sst = (zc ** 2).sum() + 1e-12
    return [(C[f] @ zc) ** 2 / ((C[f] @ C[f]) * sst) for f in order]


# ── shattering SD for a condition set at a window ──
def bal_dich(nc, cap=150):
    half = nc // 2; seen, out = set(), []
    for s in combinations(range(nc), half):
        s = frozenset(s); key = min(tuple(sorted(s)), tuple(sorted(set(range(nc)) - s)))
        if key not in seen:
            seen.add(key); out.append(list(s))
    if len(out) > cap:
        np.random.RandomState(0).shuffle(out); out = out[:cap]              # sample for the 12-cond fit
    return out


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


def shatter(stage, conds, M, dich, K=24, B=8):
    acc = np.zeros((len(dich), B))
    for b in range(B):
        rng = np.random.RandomState(400 + b)
        trp, tep = cond_pools(stage, conds, rng)
        Xtr, ctr = make_pseudo(trp, stage, conds, K, rng, M); Xte, cte = make_pseudo(tep, stage, conds, K, rng, M)
        pre = make_pipeline(StandardScaler(), PCA(min(30, Xtr.shape[0] - 1), random_state=0)).fit(Xtr)
        Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
        for di, plus in enumerate(dich):
            clf = LinearDiscriminantAnalysis().fit(Ztr, np.isin(ctr, plus).astype(int))
            acc[di, b] = balanced_accuracy_score(np.isin(cte, plus).astype(int), clf.predict(Zte))
    return acc.mean(1)


# ── cross-validated (cvPCA) reliable-variance scree per fit: cvpca.avg_spec, 25 splits, seed 0 ──




FITDATA = {}
for tsname, conds in TASKSETS.items():
    C, order = contrasts(conds); dich = bal_dich(len(conds))
    for wn, M in AW.items():
        for stage in STAGES:
            R = cvpca.cond_means(stage, M, conds); Rc = (R - R.mean(0)) / (R.std(0) + 1e-9); Rc = Rc - Rc.mean(0)
            sv, Vt = np.linalg.svd(Rc, full_matrices=False)[1:]; Z = Rc @ Vt.T
            cm_var = sv ** 2 / (sv ** 2).sum()                             # condition-mean var (matches pceta PCs)
            nk = len(conds) - 1
            pceta = np.array([eta2(Z[:, k], C, order) for k in range(nk)])
            cv = cvpca.avg_spec(stage, conds, M, nsplits=25, seed=0)       # cross-validated reliable-variance spectrum
            pos = np.clip(cv, 0, None); cv_var = pos / pos.sum(); pr = float(pos.sum() ** 2 / (pos ** 2).sum())
            sd = shatter(stage, conds, M, dich)
            FITDATA[(tsname, wn, stage)] = dict(var=cv_var, cv=cv, cm_var=cm_var, pr=pr, sd=float(sd.mean()),
                                                sd_arr=sd, pceta=pceta, factors=order, nconds=len(conds), ndich=len(dich))
            print(f'{tsname:4s} {wn:9s} {stage:6s}: cvPR={pr:.2f}  SD={sd.mean():.3f}  cv_var%={np.round(cv_var[:4],2)}')

RESPKL = f'figures/pseudo/dimensionality/results{SUF}.pkl'
d = pickle.load(open(RESPKL, 'rb')) if os.path.exists(RESPKL) else {}
d['FITDATA'] = FITDATA
pickle.dump(d, open(RESPKL, 'wb'))
print('merged FITDATA into', RESPKL)
