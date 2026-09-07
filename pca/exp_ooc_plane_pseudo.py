"""OOC_PLANE_PSEUDO cache — the OUT-OF-CONTEXT plane test on the pooled pseudo-population
(2026-09-07; the pooled twin of exp_ooc_plane.py, whose per-mouse cells are the n = 9 companion).

Question. Does ONE sample x choice plane, fitted in ONE context, capture the decodable memory and
choice signal in EVERY OTHER context (other stage, other trial types, other moments)? This replaces
the within-context plane-vs-full comparison of Fig 3c, which is by construction for sample and choice
(docs/pca/dimensionality.md, 2026-09-07).

Construction (the exp_plane_frame.py conventions, helpers copied per repo rule):
  split_ABB(stage): per (mouse, condition, perf) half A + quarters B1/B2; _pseudo(): K pseudo-trials
  per pool, one trial per mouse per pseudo-trial (3,319 neurons); per-stage neuron scale from
  'delay+dec' (the XSTAGE convention; scaling-sensitivity <= 0.02).
  reference R in {Naive-DPA, Expert-DPA}: sample axis on md pseudo-trials of the 4 DPA conditions
    (correct, partition A); choice axis on decision pseudo-trials of DPA trials lick v no-lick
    (both perf, partition A); Q = qr([w_s, w_l]).
  test contexts T = stage x task x window (sample: ed/md/decision, K = 24 per condition; choice:
    md/decision, K = 48 per class): in-context decoders trained on partition A of T, everything
    scored on partition B2 of T. Diagonal cell (T = R at the axis window) = the within check.
  per cell: full (pipeline on all neurons) · plane (2-feature LR on X@Q, plane fixed, readout refit)
    · iplane (the same 2-D readout on a plane fitted IN CONTEXT: this variable's axis at the tested
    window + the other axis at its canonical window — the fair ceiling, since a full decoder on 3,319
    neurons and 96 pseudo-trials is a worse-regularised readout than a 2-D one) · resid (pipeline on
    X - XQQ^T; redundancy keeps it high, reported not headlined) · transfer (reference axis decoder,
    no refit). Balanced accuracy, NREP reps.
Merge-dumps {'OOC_PLANE_PSEUDO'+SUF}: cells[ref][var][(stage, task, window)] = dict(full, plane,
  iplane, resid, transfer, ratio (vs full), ratio_ic (vs in-context plane), tratio (no refit vs
  in-context plane), *_sd, n_rep).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_ooc_plane_pseudo.py --nopca
"""
import sys, os, warnings, pickle, time
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from decoders import fit_axis, make_clf, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS = ['DPA', 'DualGo', 'DualNoGo']
ALL12 = [(t, s, te) for t in TASKS for s in (0, 1) for te in (0, 1)]
REFS = [('Naive', 'DPA'), ('Expert', 'DPA')]
WIN = {'sample': ['ed', 'md', 'decision'], 'choice': ['md', 'decision']}
AXWIN = {'sample': 'md', 'choice': 'decision'}
NREP = int(sys.argv[sys.argv.index('--nrep') + 1]) if '--nrep' in sys.argv else 8
KS, KL = 24, 48

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (np.asarray(_c['L'][k]) for k in
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


def sample_set(H, st, sd, task, win, part, rng):
    """Correct trials of `task`, 4 conditions x KS pseudo-trials, label = sample."""
    Xs, ys = [], []
    for ci, cd in enumerate(ALL12):
        if cd[0] != task:
            continue
        pools = {m: H[(m, ci, 1)][part] for m in MICE}
        Xs.append(_pseudo(AW[win], sd, st, pools, KS, rng)); ys.append(np.full(KS, cd[1]))
    return np.vstack(Xs), np.concatenate(ys)


def choice_set(H, st, sd, task, win, part, rng):
    """All trials of `task` (both perf), lick v no-lick at the test, KL pseudo-trials per class."""
    Xs, ys = [], []
    for lickval in (True, False):
        pools = {}
        for m in MICE:
            pool = [H[(m, ci, pf)][part] for ci, cd in enumerate(ALL12) if cd[0] == task for pf in (0, 1)]
            idx = np.concatenate(pool) if pool else np.array([], int)
            pools[m] = idx[LICK[idx] == lickval]
        Xs.append(_pseudo(AW[win], sd, st, pools, KL, rng)); ys.append(np.full(KL, int(lickval)))
    return np.vstack(Xs), np.concatenate(ys)


GET = {'sample': sample_set, 'choice': choice_set}
OTHER = {'sample': ('choice', 'decision'), 'choice': ('sample', 'md')}


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def lr2d(Xtr, ytr, Xte, yte, Q):
    c2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    return bacc(c2.fit(Xtr @ Q, ytr).predict(Xte @ Q), yte)


acc = {r: {v: {} for v in WIN} for r in REFS}
t0 = time.time()
print(f'== OOC_PLANE_PSEUDO (SUF="{SUF}", NREP={NREP}) ==', flush=True)
for rep in range(NREP):
    rng = np.random.RandomState(900 + rep)
    Hs = {st: split_ABB(st, rng) for st in STAGES}
    sds = {st: neuron_scale(st, AW['delay+dec']) for st in STAGES}
    planes = {}
    for r in REFS:
        rs, rt = r
        Xs, ys = sample_set(Hs[rs], rs, sds[rs], rt, 'md', 0, rng)
        w_s, c_s = fit_axis(Xs, ys)
        Xl, yl = choice_set(Hs[rs], rs, sds[rs], rt, 'decision', 0, rng)
        w_l, c_l = fit_axis(Xl, yl)
        planes[r] = (np.linalg.qr(np.stack([w_s, w_l], 1))[0], {'sample': c_s, 'choice': c_l})
    for st in STAGES:
        for tk in TASKS:
            # the other axis of each in-context plane, at its canonical window
            w_other = {}
            for v, (ov, ow) in OTHER.items():
                Xo, yo = GET[ov](Hs[st], st, sds[st], tk, ow, 0, rng)
                w_other[v], _ = fit_axis(Xo, yo)
            for v in WIN:
                for win in WIN[v]:
                    Xtr, ytr = GET[v](Hs[st], st, sds[st], tk, win, 0, rng)
                    Xte, yte = GET[v](Hs[st], st, sds[st], tk, win, 2, rng)
                    a_fu = bacc(make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr).predict(Xte), yte)
                    w_own, _ = fit_axis(Xtr, ytr)
                    pair = [w_own, w_other[v]] if v == 'sample' else [w_other[v], w_own]
                    a_ic = lr2d(Xtr, ytr, Xte, yte, np.linalg.qr(np.stack(pair, 1))[0])
                    for r, (Q, axclf) in planes.items():
                        a_pl = lr2d(Xtr, ytr, Xte, yte, Q)
                        Xtr_o = Xtr - (Xtr @ Q) @ Q.T; Xte_o = Xte - (Xte @ Q) @ Q.T
                        a_ou = bacc(make_clf(Xtr_o.shape[1], Xtr_o.shape[0]).fit(Xtr_o, ytr).predict(Xte_o), yte)
                        a_tr = bacc(axclf[v].predict(Xte), yte)
                        acc[r][v].setdefault((st, tk, win), []).append((a_fu, a_pl, a_ou, a_tr, a_ic))
    print(f'  rep {rep + 1}/{NREP} done ({time.time() - t0:.0f}s)', flush=True)


def cr(num, den):
    ok = den > 0.52
    return (float(((num[ok] - 0.5) / (den[ok] - 0.5)).mean()), float(((num[ok] - 0.5) / (den[ok] - 0.5)).std())) if ok.any() else (np.nan, np.nan)


cells = {}
for r in REFS:
    cells[r] = {}
    for v in WIN:
        cells[r][v] = {}
        for ctx, L in acc[r][v].items():
            A = np.array(L)                                   # (rep, 5): full, plane, resid, transfer, iplane
            ratio, ratio_sd = cr(A[:, 1], A[:, 0]); ratio_ic, ratio_ic_sd = cr(A[:, 1], A[:, 4]); tratio, tratio_sd = cr(A[:, 3], A[:, 4])
            cells[r][v][ctx] = dict(full=float(A[:, 0].mean()), plane=float(A[:, 1].mean()), resid=float(A[:, 2].mean()),
                                    transfer=float(A[:, 3].mean()), iplane=float(A[:, 4].mean()),
                                    full_sd=float(A[:, 0].std()), plane_sd=float(A[:, 1].std()), iplane_sd=float(A[:, 4].std()),
                                    ratio=ratio, ratio_sd=ratio_sd, ratio_ic=ratio_ic, ratio_ic_sd=ratio_ic_sd,
                                    tratio=tratio, tratio_sd=tratio_sd, n_rep=int(len(L)))
print('\n== SUMMARY (pseudo-population; ratio_ic = fixed plane / in-context plane; tratio = no refit / in-context plane) ==')
for r in REFS:
    for v in WIN:
        print(f'-- reference {r[0]}-{r[1]} plane, variable {v}')
        for ctx, e in cells[r][v].items():
            tag = 'WITHIN' if (ctx[0], ctx[1]) == r and ctx[2] == AXWIN[v] else 'ooc'
            print(f"   {ctx[0]:6s} {ctx[1]:8s} @{ctx[2]:8s} {tag:6s} full {e['full']:.2f}  iplane {e['iplane']:.2f}  plane {e['plane']:.2f}  "
                  f"resid {e['resid']:.2f}  transfer {e['transfer']:.2f}  ratio_ic {e['ratio_ic']:.2f}±{e['ratio_ic_sd']:.2f}  tratio {e['tratio']:.2f}  ratio_full {e['ratio']:.2f}")

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['OOC_PLANE_PSEUDO' + SUF] = dict(cells=cells, refs=REFS, win=WIN, axwin=AXWIN, nrep=NREP, KS=KS, KL=KL)
pickle.dump(d, open(RES, 'wb'))
print('merged OOC_PLANE_PSEUDO' + SUF, 'into', RES, f'({time.time() - t0:.0f}s)')
