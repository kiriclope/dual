"""Per-fit dimensionality: for each (task-set × window × stage) compute the condition-mean scree + PR and the
shattering SD (mean balanced-accuracy over balanced dichotomies). Windows: delay(48-53), decision(57-65),
delay+dec wide(48-65). Task-sets: dual (8 conds), DPA (4 conds). Merges FITDATA into results.pkl."""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir('/home/leon/dual/pca')
import numpy as np
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
WINS = {'delay': np.asarray(o['bins_LD']), 'decision': np.arange(57, 66), 'delay+dec': np.arange(48, 66)}
# Persistent cache of the window-averaged pseudo-population matrices + labels, so re-analysis never reloads
# the 20 GB X. Built once (first run), then loaded in seconds. Rebuild by deleting the file (or if WINS change).
AWPKL = 'figures/pseudo/dimensionality/fits_inputs.pkl'
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


# ── cross-validated (cvPCA) reliable-variance scree per fit — the honest dimensionality (matches main panel) ──
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


def cvpca_spectrum(S1, S2):
    S1 = S1 - S1.mean(0, keepdims=True); S2 = S2 - S2.mean(0, keepdims=True)

    def one(A, B):
        Vt = np.linalg.svd(A, full_matrices=False)[2]
        return ((A @ Vt.T) * (B @ Vt.T)).sum(0)
    a, b = one(S1, S2), one(S2, S1); k = min(len(a), len(b))
    return 0.5 * (a[:k] + b[:k])


def cvpca_fit(stage, conds, M, nsplits=25):
    sd = neuron_scale(stage, M); rng = np.random.RandomState(0); spec = None
    for _ in range(nsplits):
        R1, R2 = split_means(stage, conds, M, rng)
        c = cvpca_spectrum(R1 / sd[None, :], R2 / sd[None, :])
        spec = c if spec is None else spec + c
    return spec / nsplits


FITDATA = {}
for tsname, conds in TASKSETS.items():
    C, order = contrasts(conds); dich = bal_dich(len(conds))
    for wn, M in AW.items():
        for stage in STAGES:
            R = cond_means(stage, M, conds); Rc = (R - R.mean(0)) / (R.std(0) + 1e-9); Rc = Rc - Rc.mean(0)
            sv, Vt = np.linalg.svd(Rc, full_matrices=False)[1:]; Z = Rc @ Vt.T
            cm_var = sv ** 2 / (sv ** 2).sum()                             # condition-mean var (matches pceta PCs)
            nk = len(conds) - 1
            pceta = np.array([eta2(Z[:, k], C, order) for k in range(nk)])
            cv = cvpca_fit(stage, conds, M)                                # cross-validated reliable-variance spectrum
            pos = np.clip(cv, 0, None); cv_var = pos / pos.sum(); pr = float(pos.sum() ** 2 / (pos ** 2).sum())
            sd = shatter(stage, conds, M, dich)
            FITDATA[(tsname, wn, stage)] = dict(var=cv_var, cv=cv, cm_var=cm_var, pr=pr, sd=float(sd.mean()),
                                                sd_arr=sd, pceta=pceta, factors=order, nconds=len(conds), ndich=len(dich))
            print(f'{tsname:4s} {wn:9s} {stage:6s}: cvPR={pr:.2f}  SD={sd.mean():.3f}  cv_var%={np.round(cv_var[:4],2)}')

d = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
d['FITDATA'] = FITDATA
pickle.dump(d, open('figures/pseudo/dimensionality/results.pkl', 'wb'))
print('merged FITDATA into results.pkl')
