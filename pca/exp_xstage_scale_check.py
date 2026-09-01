"""XSTAGE scaling sensitivity check (2026-09-01, from the Codex review of Figs 2-4).

Concern: XSTAGE_DEC (exp_plane_frame.py) scales TRAIN pseudo-trials by the train stage's
per-neuron sd and TEST pseudo-trials by the TEST stage's own sd — so the reported
transfer/within ~0.9 measures functional transfer AFTER per-stage per-neuron renormalisation,
not axis identity in one fixed coordinate system. This script re-runs the same protocol under
both conventions:
  own   — test features scaled by the test stage's sd (the cached/figure convention)
  train — test features scaled by the TRAIN stage's sd (one common coordinate system per decoder)
If the transfer/within ratios agree, the frame claim is scaling-robust; a drop under 'train'
would mean per-neuron gain changes carry part of the story.

Merges {'XSTAGE_SCALECHK'+SUF} into results.pkl for the record.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_xstage_scale_check.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import make_clf, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
NREP = 8

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def split_ABB(stage, rng):
    H = {}
    for m in MICE:
        for ci, (t, s, te) in enumerate(ALL12):
            for pf in (0, 1):
                idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == pf)
                               & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
                p = rng.permutation(idx); h = len(p) // 2; q = h + (len(p) - h) // 2
                H[(m, ci, pf)] = (p[:h], p[h:q], p[q:])
    return H


def _pseudo(M, sd, stage, pools, K, rng):
    X = np.zeros((K, N))
    for m, idx in pools.items():
        if not len(idx):
            continue
        val = VALIDIX[(m, stage)]
        cm = np.nanmean(M[np.ix_(idx, val)], 0)
        blk = M[np.ix_(rng.choice(idx, K, replace=True), val)]
        bad = ~np.isfinite(blk)
        if bad.any():
            blk[bad] = np.broadcast_to(cm, blk.shape)[bad]
        X[:, val] = blk
    return X / sd[None, :]


print(f'== XSTAGE scaling sensitivity (SUF="{SUF}", NREP={NREP}) ==', flush=True)
OUT = {}
for rep in range(NREP):
    rng = np.random.RandomState(240 + rep)          # SAME seeds as the cached XSTAGE_DEC
    Hs = {st: split_ABB(st, rng) for st in STAGES}
    sds = {st: neuron_scale(st, AW['delay+dec']) for st in STAGES}

    def sample_sets(st, part, sd_use):
        Xs, ys = [], []
        for ci, cd in enumerate(ALL12):
            pools = {m: Hs[st][(m, ci, 1)][part] for m in MICE}
            Xs.append(_pseudo(AW['md'], sd_use, st, pools, 24, rng))
            ys.append(np.full(24, cd[1]))
        return np.vstack(Xs), np.concatenate(ys)

    def choice_sets(st, part, sd_use):
        Xs, ys = [], []
        for lickval in (True, False):
            pools = {}
            for m in MICE:
                pool = [Hs[st][(m, ci, pf)][part] for ci in range(len(ALL12)) for pf in (0, 1)]
                idx = np.concatenate(pool) if pool else np.array([], int)
                pools[m] = idx[LICK[idx] == lickval]
            Xs.append(_pseudo(AW['decision'], sd_use, st, pools, 48, rng))
            ys.append(np.full(48, int(lickval)))
        return np.vstack(Xs), np.concatenate(ys)

    for vname, getter in [('sample', sample_sets), ('choice', choice_sets)]:
        for st_tr in STAGES:
            Xtr, ytr = getter(st_tr, 0, sds[st_tr])
            clf = make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr)
            for st_te in STAGES:
                for conv, sd_use in [('own', sds[st_te]), ('train', sds[st_tr])]:
                    Xte, yte = getter(st_te, 2, sd_use)
                    pred = clf.predict(Xte)
                    acc = np.mean([np.mean(pred[yte == c] == c) for c in set(yte)])
                    OUT.setdefault((vname, st_tr, st_te, conv), []).append(float(acc))

for vname in ['sample', 'choice']:
    for conv in ['own', 'train']:
        dia = np.mean([np.mean(OUT[(vname, s, s, conv)]) for s in STAGES])
        off = np.mean([np.mean(OUT[(vname, a, b, conv)]) for a in STAGES for b in STAGES if a != b])
        print(f'  {vname:7s} [{conv:5s}] within {dia:.3f}  cross {off:.3f}  '
              f'transfer/within {(off - .5) / (dia - .5):.3f}', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['XSTAGE_SCALECHK' + SUF] = {k: np.asarray(v) for k, v in OUT.items()}
pickle.dump(d, open(RES, 'wb'))
print('merged XSTAGE_SCALECHK' + SUF, 'into', RES)
