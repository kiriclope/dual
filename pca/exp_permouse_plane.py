"""PM_PLANE cache — per-mouse plane-vs-full decoding for the Fig 3 sufficiency panel
(user design 2026-08-31): for EACH mouse and stage, decode each variable from (i) only the 2
coordinates of that mouse's own sample x choice plane and (ii) its full population.

Per (mouse, stage), NREP reps: split every trial pool into disjoint halves; fit the mouse's sample
axis (@md, samp 1v0, correct trials) and behavioural choice axis (@decision, DPA lick v no-lick)
on half 1 (decoders.fit_axis — respects --nopca); Q = orthonormalised [w_s, w_l]. Each variable's
classifier is trained on half 1 and tested on held-out half 2, either on X@Q (plane, 2 features)
or on X (full). Balanced accuracy.

Variables (own canonical windows, pca T_WINDOW=0.5 conventions):
  sample @ md (correct, all tasks) · test @ decision (correct, all tasks) ·
  choice @ decision (DPA, behavioural lick) · dist @ md (dual Go v NoGo, no perf filter).
Merge-dumps {'PM_PLANE'+SUF} into results.pkl: PM_PLANE[(mouse, stage)][var] = (plane, full).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_permouse_plane.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from decoders import fit_axis, make_clf, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
NREP = 10

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)


def sel(mouse, stage, **kw):
    m = (MOUSE == mouse) & (LEARN == stage) & (LAS == 0)
    for k, v in kw.items():
        arr = {'task': TSK, 'samp': SAMP, 'test': TESTO, 'perf': PERF, 'lick': LICK,
               'dual': None}[k]
        if k == 'dual':
            m &= (TSK != 'DPA')
        else:
            m &= np.isin(arr, v) if isinstance(v, (list, tuple, np.ndarray)) else (arr == v)
    return np.where(m)[0]


def zscale(M, val, idx):
    sd = np.nanstd(M[np.ix_(idx, val)], axis=0)
    return np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def halves(rng, idx):
    p = rng.permutation(idx); h = len(p) // 2
    return p[:h], p[h:]


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


PM_PLANE = {}
print(f'══ PM_PLANE (SUF="{SUF}") ══', flush=True)
for stage in STAGES:
    for mo in MICE:
        val = VALIDIX[(mo, stage)]
        allc = sel(mo, stage, perf=1)
        if len(allc) < 20 or not len(val):
            continue
        Mmd, Mdc = AW['md'], AW['decision']
        sdm, sdd = zscale(Mmd, val, allc), zscale(Mdc, val, allc)
        # variable spec: (name, window matrix, sd, pos-pool fn, neg-pool fn)
        VARS = [
            ('sample', Mmd, sdm, lambda: sel(mo, stage, perf=1, samp=1),
                                 lambda: sel(mo, stage, perf=1, samp=0)),
            ('test',   Mdc, sdd, lambda: sel(mo, stage, perf=1, test=1),
                                 lambda: sel(mo, stage, perf=1, test=0)),
            ('choice', Mdc, sdd, lambda: sel(mo, stage, task='DPA', lick=True),
                                 lambda: sel(mo, stage, task='DPA', lick=False)),
            ('dist',   Mmd, sdm, lambda: sel(mo, stage, task='DualGo'),
                                 lambda: sel(mo, stage, task='DualNoGo')),
        ]
        acc = {v[0]: [] for v in VARS}
        rng = np.random.RandomState(300)
        for _ in range(NREP):
            # the mouse's own frame, fit on half 1 of the axis pools
            sP1, sP2 = halves(rng, sel(mo, stage, perf=1, samp=1))
            sN1, sN2 = halves(rng, sel(mo, stage, perf=1, samp=0))
            lP1, lP2 = halves(rng, sel(mo, stage, task='DPA', lick=True))
            lN1, lN2 = halves(rng, sel(mo, stage, task='DPA', lick=False))
            if min(len(sP1), len(sN1), len(lP1), len(lN1)) < 3:
                continue
            Xs = np.vstack([Mmd[np.ix_(sP1, val)] / sdm, Mmd[np.ix_(sN1, val)] / sdm])
            w_s, _ = fit_axis(np.nan_to_num(Xs), np.r_[np.ones(len(sP1), int), np.zeros(len(sN1), int)])
            Xl = np.vstack([Mdc[np.ix_(lP1, val)] / sdd, Mdc[np.ix_(lN1, val)] / sdd])
            w_l, _ = fit_axis(np.nan_to_num(Xl), np.r_[np.ones(len(lP1), int), np.zeros(len(lN1), int)])
            Q = np.linalg.qr(np.stack([w_s, w_l], 1))[0]
            # reuse the SAME half split for the axis pools; fresh halves for the others
            HH = {'sample': ((sP1, sP2), (sN1, sN2)), 'choice': ((lP1, lP2), (lN1, lN2))}
            for vname, M, sd_, fpos, fneg in VARS:
                if vname in HH:
                    (p1, p2), (n1, n2) = HH[vname]
                else:
                    p1, p2 = halves(rng, fpos()); n1, n2 = halves(rng, fneg())
                if min(len(p1), len(n1), len(p2), len(n2)) < 3:
                    continue
                Xtr = np.nan_to_num(np.vstack([M[np.ix_(p1, val)] / sd_, M[np.ix_(n1, val)] / sd_]))
                Xte = np.nan_to_num(np.vstack([M[np.ix_(p2, val)] / sd_, M[np.ix_(n2, val)] / sd_]))
                ytr = np.r_[np.ones(len(p1), int), np.zeros(len(n1), int)]
                yte = np.r_[np.ones(len(p2), int), np.zeros(len(n2), int)]
                c2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                a_pl = bacc(c2.fit(Xtr @ Q, ytr).predict(Xte @ Q), yte)
                cF = make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr)
                a_fu = bacc(cF.predict(Xte), yte)
                # OUT-OF-PLANE complement (2026-08-31): remove the plane component, decode the
                # residual with the full pipeline. NB estimation caveat: only the ESTIMATED plane
                # is removed, and population codes are redundant, so above-chance residual
                # decoding does NOT contradict "the plane carries the geometry" — the comparison
                # of interest is plane vs out-of-plane vs full per variable.
                Xtr_o = Xtr - (Xtr @ Q) @ Q.T
                Xte_o = Xte - (Xte @ Q) @ Q.T
                cO = make_clf(Xtr_o.shape[1], Xtr_o.shape[0]).fit(Xtr_o, ytr)
                a_ou = bacc(cO.predict(Xte_o), yte)
                acc[vname].append((a_pl, a_fu, a_ou))
        ent = {v: tuple(float(np.mean([t[i] for t in acc[v]])) for i in range(3))
               for v in acc if acc[v]}
        PM_PLANE[(mo, stage)] = ent                     # (plane, full, out-of-plane)
        print(f'  {stage:6s} {mo:8s} ' + '  '.join(
            f'{v} {ent[v][0]:.2f}/{ent[v][1]:.2f}/{ent[v][2]:.2f}'
            for v in ['sample', 'test', 'choice', 'dist'] if v in ent), flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PM_PLANE' + SUF] = PM_PLANE
pickle.dump(d, open(RES, 'wb'))
print('merged PM_PLANE' + SUF, 'into', RES)
