"""PM_XSTAGE cache — per-mouse cross-stage decoding for Fig 3 panel F's per-animal scatters
(user request 2026-08-31: "add per mouse scatters to E and F").

The pooled XSTAGE_DEC (exp_plane_frame.py) shows transfer/within ~0.9 on the pseudo-population;
this cache asks the same question INSIDE each animal: train the mouse's own sample (@md) or
choice (@decision) decoder on one stage's trial half, test on held-out halves of BOTH stages
(neurons registered across stages — VALIDIX is identical per mouse, verified 2026-08-31).
Features are scaled by each stage's own per-neuron sd (the pooled XSTAGE convention).

Per (mouse, variable): NREP reps x both train stages; 'within' = mean of NaiveâNaive & Expert->
Expert held-out accuracy, 'cross' = mean of the two cross-stage reads; per-direction values kept.
Merge-dumps {'PM_XSTAGE'+SUF}: PM_XSTAGE[mouse][var] = dict(within, cross, dirs={(tr,te): acc}).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_permouse_xstage.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import make_clf, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
NREP = 10

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)          # behavioural lick (error trials included)


def zscale(M, val, idx):
    sd = np.nanstd(M[np.ix_(idx, val)], axis=0)
    return np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def halves(rng, idx):
    p = rng.permutation(idx); h = len(p) // 2
    return p[:h], p[h:]


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


# variable -> (window key, positive-pool fn, negative-pool fn) per (mouse, stage)
def pools(mo, st, vn):
    base = (MOUSE == mo) & (LEARN == st) & (LAS == 0)
    if vn == 'sample':
        return 'md', np.where(base & (PERF == 1) & (SAMP == 1))[0], \
                     np.where(base & (PERF == 1) & (SAMP == 0))[0]
    return 'decision', np.where(base & (TSK == 'DPA') & LICK)[0], \
                       np.where(base & (TSK == 'DPA') & ~LICK)[0]


PM_XSTAGE = {}
print(f'== PM_XSTAGE (SUF="{SUF}") ==', flush=True)
for mo in MICE:
    val = VALIDIX[(mo, 'Naive')]
    assert np.array_equal(val, VALIDIX[(mo, 'Expert')]), mo   # registered neurons
    ent = {}
    for vn in ['sample', 'choice']:
        wn = pools(mo, 'Naive', vn)[0]
        M = AW[wn]
        # per-stage feature scale from that stage's correct trials (the pooled convention)
        sds = {}
        for st in STAGES:
            allc = np.where((MOUSE == mo) & (LEARN == st) & (LAS == 0) & (PERF == 1))[0]
            sds[st] = zscale(M, val, allc)
        acc = {}
        rng = np.random.RandomState(500)
        for _ in range(NREP):
            H = {}
            for st in STAGES:
                _, P, Nn = pools(mo, st, vn)
                H[st] = (halves(rng, P), halves(rng, Nn))
            if any(min(len(H[st][0][0]), len(H[st][1][0])) < 3 for st in STAGES):
                continue
            for st_tr in STAGES:
                (p1, _), (n1, _) = H[st_tr]
                Xtr = np.nan_to_num(np.vstack([M[np.ix_(p1, val)], M[np.ix_(n1, val)]])
                                    / sds[st_tr])
                ytr = np.r_[np.ones(len(p1), int), np.zeros(len(n1), int)]
                clf = make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr)
                for st_te in STAGES:
                    (_, p2), (_, n2) = H[st_te]
                    Xte = np.nan_to_num(np.vstack([M[np.ix_(p2, val)], M[np.ix_(n2, val)]])
                                        / sds[st_te])
                    yte = np.r_[np.ones(len(p2), int), np.zeros(len(n2), int)]
                    acc.setdefault((st_tr, st_te), []).append(bacc(clf.predict(Xte), yte))
        if not acc:
            continue
        dirs = {k: float(np.mean(v)) for k, v in acc.items()}
        ent[vn] = dict(within=float(np.mean([dirs[('Naive', 'Naive')], dirs[('Expert', 'Expert')]])),
                       cross=float(np.mean([dirs[('Naive', 'Expert')], dirs[('Expert', 'Naive')]])),
                       dirs=dirs)
    PM_XSTAGE[mo] = ent
    print('  ' + mo + '  ' + '  '.join(
        f"{vn} within {ent[vn]['within']:.2f} cross {ent[vn]['cross']:.2f}"
        for vn in ent), flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PM_XSTAGE' + SUF] = PM_XSTAGE
pickle.dump(d, open(RES, 'wb'))
print('merged PM_XSTAGE' + SUF, 'into', RES)
