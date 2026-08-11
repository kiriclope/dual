"""Support cache for the --cdecode Fig-2 variant: B-spectra error bars + DPA-gng bar for C.

A. SPEC_JK — jackknife-across-mice 95% CIs for the panel-B reliable spectra (DPA/dual x md/decision
   x stage): leave one mouse out (its neurons AND trials), recompute the averaged cvPCA spectrum
   FRACTIONS, jackknife SE per component (exp_dimensionality_jk.py convention), CI clipped to [0,1].
   Stores the full-sample fractions too, so the renderer draws point + CI from ONE source.
B. DPA_GNG_C — the panel-C 'gng in DPA' bar: Go-vs-NoGo cross-decoded from the DPA-STATE SUBSPACE
   (top-3 PCs of the DPA condition means, exp_dpa_gng_column.py convention): dual pseudo-trials
   projected into the subspace, LDA train/test on disjoint trial halves, vs a within-mouse
   label-shuffle null (95th pct). Windows md + decision (matching C's rows).

Merge-dumps {'SPEC_JK', 'DPA_GNG_C'} into results.pkl. Cache-only (~3 min), no X reload.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_cdec_support.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
DUAL = [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
GLAB = np.array([1 if c[0] == 'DualGo' else 0 for c in DUAL])
WINS = ['md', 'decision']

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


def avg_frac(stage, conds, M, mice, nsplits=20):
    sd = neuron_scale(stage, M); rng = np.random.RandomState(7); spec = None
    for _ in range(nsplits):
        R1, R2 = split_means(stage, conds, M, rng, mice)
        c = cvpca_spectrum(R1 / sd[None, :], R2 / sd[None, :])
        spec = c if spec is None else spec + c
    pos = np.clip(spec / nsplits, 0, None)
    return pos / (pos.sum() + 1e-12)


print('══ A. SPEC_JK: jackknife-across-mice CIs for the B spectra ══')
SPEC_JK = {}
for ts, conds in [('DPA', DPA4), ('dual', DUAL)]:
    for wn in WINS:
        for stage in STAGES:
            frac = avg_frac(stage, conds, AW[wn], MICE)
            jk = np.array([avg_frac(stage, conds, AW[wn], [m for m in MICE if m != mo]) for mo in MICE])
            n = len(MICE)
            se = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
            lo = np.clip(frac - 1.96 * se, 0, 1); hi = np.clip(frac + 1.96 * se, 0, 1)
            SPEC_JK[(ts, wn, stage)] = dict(frac=frac, se=se, lo=lo, hi=hi)
            print(f'  {ts:4s} {wn:9s} {stage:6s} frac {np.round(frac[:4], 3)}  '
                  f'CI1 [{lo[0]:.2f},{hi[0]:.2f}]  CI2 [{lo[1]:.2f},{hi[1]:.2f}]', flush=True)

# ── B. DPA_GNG_C: gng decoded from the DPA-state subspace (top-3 PCs), held-out, vs shuffle null ──


def cond_means_all(stage, M, conds):
    R = np.zeros((len(conds), N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], axis=0)
    return R


def dual_pools(stage, rng, shuffle=False):
    tr, te = {}, {}
    P = {}
    for m in MICE:
        for ci, (t, s, te_) in enumerate(DUAL):
            P[(ci, m)] = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                                  & (TSK == t) & (SAMP == s) & (TESTO == te_))[0]
    if shuffle:                                    # within-mouse trial->condition permutation
        for m in MICE:
            allidx = np.concatenate([P[(ci, m)] for ci in range(len(DUAL))])
            perm = rng.permutation(allidx); k = 0
            for ci in range(len(DUAL)):
                n = len(P[(ci, m)]); P[(ci, m)] = perm[k:k + n]; k += n
    for key, idx in P.items():
        p = rng.permutation(idx); h = len(p) // 2
        tr[key], te[key] = p[:h], p[h:]
    return tr, te


def make_pseudo(pool, stage, K, rng, M):
    Xp = np.zeros((len(DUAL) * K, N)); crow = np.repeat(np.arange(len(DUAL)), K)
    for ci in range(len(DUAL)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; pi = pool[(ci, m)]
            if len(pi):
                block = M[np.ix_(pi[rng.randint(0, len(pi), K)], val)]
                bad = ~np.isfinite(block)
                if bad.any():
                    block[bad] = 0.0
                Xp[ci * K:(ci + 1) * K, val] = block
    return Xp, crow


def gng_from_dpa(stage, M, rng, shuffle):
    R = cond_means_all(stage, M, DPA4); mu = R.mean(0); sd = R.std(0) + 1e-9
    Rc = (R - mu) / sd; Rc = Rc - Rc.mean(0)
    Vt = np.linalg.svd(Rc, full_matrices=False)[2][:3]        # DPA-state subspace (drop degenerate PC4)
    trp, tep = dual_pools(stage, rng, shuffle=shuffle)
    Xtr, ctr = make_pseudo(trp, stage, 24, rng, M); Xte, cte = make_pseudo(tep, stage, 24, rng, M)
    Ztr = ((Xtr - mu) / sd) @ Vt.T; Zte = ((Xte - mu) / sd) @ Vt.T
    clf = LinearDiscriminantAnalysis().fit(Ztr, GLAB[ctr])
    return balanced_accuracy_score(GLAB[cte], clf.predict(Zte))


print('\n══ B. DPA_GNG_C: gng from the DPA subspace (held-out, vs shuffle null) ══')
DPA_GNG_C = {}
for wn in WINS:
    for stage in STAGES:
        rng = np.random.RandomState(500)
        real = [gng_from_dpa(stage, AW[wn], rng, False) for _ in range(8)]
        null = [gng_from_dpa(stage, AW[wn], rng, True) for _ in range(12)]
        acc = float(np.mean(real)); n95 = float(np.percentile(null, 95))
        DPA_GNG_C[(wn, stage)] = dict(acc=acc, null95=n95, sig=bool(acc > n95))
        print(f'  {wn:9s} {stage:6s} acc={acc:.2f}  null95={n95:.2f}  sig={acc > n95}', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['SPEC_JK'] = SPEC_JK; d['DPA_GNG_C'] = DPA_GNG_C
pickle.dump(d, open(RES, 'wb'))
print('\nmerged SPEC_JK + DPA_GNG_C into', RES)
