"""Mid-delay (bins_MD 36-38, data 5.5-6.33 s) dimensionality check — post-distractor, PRE-cue/PRE-lick.

User question: at late delay (bins_LD, 7.5-8.83 s) the dual-set gng factor carries 88-96% of the
reliable variance, but on Go trials LD is post-cue/post-lick/post-reward, so that variance bundles
consummatory residue. Re-run at MD, where no lick has happened, to see the uncontaminated split
(expectation: sample vs gng more balanced -> dual delay closer to a true 2-D sample+gng plane).

Does ONE 20 GB X pass to add 'md' (36-38) and 'ed' (21-26, pre-distractor) window matrices to
fits_inputs.pkl (merged in place; existing loaders unaffected), then:
  A. cross-validated factor-contrast decomposition at MD (and ED, LD for comparison), dual + DPA;
  B. cvPCA spectra + PR at MD for dual / DPA / all-12.
Merge-dumps {'MD_CHECK': ...} into results.pkl.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dimensionality_md.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.pca.io import pkl_load
from src.common.options import set_options

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
ALL12 = [(t, s, te) for t in TASKS3 for s in (0, 1) for te in (0, 1)]
DUAL = [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
o = set_options()
NEWWINS = {'md': np.asarray(o['bins_MD']), 'ed': np.asarray(o['bins_ED'])}

AWPKL = 'figures/pseudo/dimensionality/fits_inputs.pkl'
_c = pickle.load(open(AWPKL, 'rb'))
if not set(NEWWINS) <= set(_c['AW']):
    print('extending fits cache with md/ed windows (one-time 20 GB X pass) …')
    X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
    for w, b in NEWWINS.items():
        _c['AW'][w] = np.nanmean(X[:, :, b], axis=2)
    del X
    pickle.dump(_c, open(AWPKL, 'wb'))
    print('cache extended:', sorted(_c['AW']))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
if 'test' not in AW:                                                      # bins_TEST 57-59 = the altwin
    AW['test'] = pickle.load(open('figures/pseudo/dimensionality/fits_inputs_altwin.pkl',  # cache's
                                  'rb'))['AW']['decision']                # 'decision' matrix — no X pass
    pickle.dump(_c, open(AWPKL, 'wb'))
    print('added test window (bins_TEST 57-59) from the altwin cache:', sorted(AW))


def contrasts(conds):
    s = np.array([c[1] for c in conds]) * 2 - 1
    te = np.array([c[2] for c in conds]) * 2 - 1
    out = {'sample': s.astype(float), 'test': te.astype(float), 'choice': (s * te).astype(float)}
    tk = np.array([c[0] for c in conds])
    if 'DualGo' in tk and 'DualNoGo' in tk:
        g = np.where(tk == 'DualGo', 1.0, -1.0)
        out['gng'] = g; out['s×g'] = s * g; out['g×t'] = g * te; out['s×g×t'] = s * g * te
    return out


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
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


def factor_var(stage, conds, M, nsplits=20):
    C = contrasts(conds); nc = len(conds)
    sd = neuron_scale(stage, M); rng = np.random.RandomState(0)
    acc = {f: [] for f in C}
    for _ in range(nsplits):
        R1, R2 = split_means(stage, conds, M, rng)
        R1 = R1 / sd[None, :]; R2 = R2 / sd[None, :]
        R1 = R1 - R1.mean(0, keepdims=True); R2 = R2 - R2.mean(0, keepdims=True)
        for f, c in C.items():
            D1 = c @ R1 / (c @ c); D2 = c @ R2 / (c @ c)
            acc[f].append(float(np.dot(D1, D2) * (c @ c) / nc))
    return {f: (float(np.mean(v)), float(np.std(v))) for f, v in acc.items()}


def cvpca_pr(stage, conds, M, nsplits=25):
    sd = neuron_scale(stage, M); rng = np.random.RandomState(0); spec = None
    for _ in range(nsplits):
        R1, R2 = split_means(stage, conds, M, rng)
        c = cvpca_spectrum(R1 / sd[None, :], R2 / sd[None, :])
        spec = c if spec is None else spec + c
    spec = spec / nsplits; pos = np.clip(spec, 0, None)
    return float(pos.sum() ** 2 / ((pos ** 2).sum() + 1e-12)), pos / pos.sum()


OUT = {}
print('\n══ A. factor decomposition per window (z-scored) ══')
for wn, wlab in [('ed', 'early delay (pre-distr.)'), ('md', 'MID delay (pre-lick)'), ('delay', 'late delay'),
                 ('test', 'test odour (57-59)')]:
    for conds, nm in [(DUAL, 'dual'), (DPA4, 'DPA')]:
        for stage in STAGES:
            fv = factor_var(stage, conds, AW[wn])
            OUT[('fv', wn, nm, stage)] = fv
            tot = sum(max(v[0], 0) for v in fv.values()) + 1e-12
            parts = '  '.join(f'{f}={100*max(v[0],0)/tot:.1f}%' for f, v in fv.items() if max(v[0], 0) / tot > 0.005)
            print(f'  {wlab:26s} {nm:4s} {stage:6s}: {parts}   [sample abs {fv["sample"][0]:.2f}±{fv["sample"][1]:.2f}]')

print('\n══ B. cvPCA PR × window (ED / MD / LD / TEST) × set ══')
for wn, wlab in [('ed', 'ED'), ('md', 'MD'), ('delay', 'LD'), ('test', 'TEST')]:
    for conds, nm in [(DPA4, 'DPA'), (DUAL, 'dual'), (ALL12, 'all')]:
        for stage in STAGES:
            pr, frac = cvpca_pr(stage, conds, AW[wn])
            OUT[(f'pr_{wn}', nm, stage)] = dict(pr=pr, frac=frac)
            print(f'  {wlab:4s} {nm:4s} {stage:6s}: PR={pr:.2f}  fracs={np.round(frac[:5], 3)}')

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['MD_CHECK'] = OUT
pickle.dump(d, open(RES, 'wb'))
print('\nmerged MD_CHECK into', RES)
