"""OOC_PLANE cache — the OUT-OF-CONTEXT plane test, per mouse (2026-09-07).

Why. The Fig 3c,d plane-vs-full comparison is circular for sample and choice: the plane is spanned
by those variables' own decoder axes, fitted on the same trials as the "full" decoder, so plane = full
is guaranteed (see docs/pca/dimensionality.md, 2026-09-07). The non-circular question is whether ONE
plane, fitted in ONE context, captures the decodable memory and choice signal in EVERY OTHER context
(other stage, other trial types, other moments of the trial).

Design (per mouse; neurons registered across stages, VALIDIX identical per mouse):
  reference R in {Naive-DPA, Expert-DPA}: sample axis = DPA correct trials A v B at 'md'; choice axis
    = DPA trials lick v no-lick at 'decision' (behavioural lick, errors included); both fitted with
    decoders.fit_axis on HALF 1 of the reference pools; Q = orthonormalised [w_s, w_l]. One per-neuron
    scale per reference (sd over the reference stage's correct trials, pooled over the md and decision
    windows) is applied to every window so that Q lives in one coordinate system.
  test contexts T = stage {Naive, Expert} x task {DPA, DualGo, DualNoGo} x window
    (sample: 'ed', 'md', 'decision'; choice: 'md', 'decision'). Pools: sample = correct trials A v B;
    choice = all laser-off trials lick v no-lick. Each (stage, task, variable) pool is split into
    halves ONCE per rep and the split is shared across windows and references, so the diagonal cell
    (T = R at the axis window) is the within-context check and every other cell is out of context.
  per context and rep, all scored on half 2 of T:
    full    = the pipeline trained on half 1 of T (all neurons)
    plane   = a 2-feature logistic regression on X@Q trained on half 1 of T (plane fixed, readout refit)
    iplane  = the same 2-D readout on a plane fitted IN CONTEXT on half 1 of T (the variable's own axis
              at the tested window + the other axis at its canonical window) — the fair ceiling for a
              fixed plane, since a full-population decoder on hundreds of neurons and tens of trials is
              a worse-regularised readout than any 2-D one
    resid   = the pipeline on X - (X@Q)Q^T (plane removed; NB population codes are redundant, so this
              stays high whatever the plane — reported, not headlined)
    transfer = the reference axis decoder applied to half 2 of T with NO refit (axis + boundary).
  Balanced accuracy; NREP reps; per-mouse means; ratios are chance-referenced,
    ratio_ic = (plane - 0.5)/(iplane - 0.5) [primary], ratio = (plane - 0.5)/(full - 0.5), ceilings > 0.52.

Merge-dumps {'OOC_PLANE'+SUF} into results.pkl:
  OOC_PLANE['cells'][mouse][ref][var][(stage, task, window)] = dict(full, plane, iplane, resid, transfer, n)
  OOC_PLANE['summary'][ref][var][(stage, task, window)] = across-mouse stats (see summarise()).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_ooc_plane.py --nopca
"""
import sys, os, warnings, pickle, time
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon, sem
from sklearn.linear_model import LogisticRegression
from decoders import fit_axis, make_clf, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS = ['DPA', 'DualGo', 'DualNoGo']
REFS = [('Naive', 'DPA'), ('Expert', 'DPA')]
WIN = {'sample': ['ed', 'md', 'decision'], 'choice': ['md', 'decision']}
AXWIN = {'sample': 'md', 'choice': 'decision'}
NREP = int(sys.argv[sys.argv.index('--nrep') + 1]) if '--nrep' in sys.argv else 10
MINN = 3

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (np.asarray(_c['L'][k]) for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)          # behavioural lick at the test (errors included)


def pools(mo, st, tk, var):
    base = (MOUSE == mo) & (LEARN == st) & (LAS == 0) & (TSK == tk)
    if var == 'sample':
        return np.where(base & (PERF == 1) & (SAMP == 1))[0], np.where(base & (PERF == 1) & (SAMP == 0))[0]
    return np.where(base & LICK)[0], np.where(base & ~LICK)[0]


def halves(rng, idx):
    p = rng.permutation(idx); h = len(p) // 2
    return p[:h], p[h:]


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def X_of(win, idx, val, sd):
    return np.nan_to_num(AW[win][np.ix_(idx, val)] / sd)


def ref_scale(mo, st, val):
    allc = np.where((MOUSE == mo) & (LEARN == st) & (LAS == 0) & (PERF == 1))[0]
    M = np.vstack([AW['md'][np.ix_(allc, val)], AW['decision'][np.ix_(allc, val)]])
    sd = np.nanstd(M, axis=0)
    return np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def lr2d(Xtr, ytr, Xte, yte, Q):
    c2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    return bacc(c2.fit(Xtr @ Q, ytr).predict(Xte @ Q), yte)


cells = {}
t0 = time.time()
print(f'== OOC_PLANE (SUF="{SUF}", NREP={NREP}) ==', flush=True)
for mo in MICE:
    val = VALIDIX[(mo, 'Naive')]
    assert np.array_equal(val, VALIDIX[(mo, 'Expert')]), mo
    SD = {r: ref_scale(mo, r[0], val) for r in REFS}
    acc = {r: {v: {} for v in WIN} for r in REFS}      # acc[ref][var][ctx] -> list of tuples
    rng = np.random.RandomState(700)
    for rep in range(NREP):
        # one split per (stage, task, variable) pool, shared across windows and references
        H = {}
        for st in STAGES:
            for tk in TASKS:
                for v in WIN:
                    P, N = pools(mo, st, tk, v)
                    H[(st, tk, v)] = (halves(rng, P), halves(rng, N))
        # the reference planes and their axis decoders
        planes = {}
        for r in REFS:
            rs, rt = r
            (sP1, _), (sN1, _) = H[(rs, rt, 'sample')]
            (lP1, _), (lN1, _) = H[(rs, rt, 'choice')]
            if min(len(sP1), len(sN1), len(lP1), len(lN1)) < MINN:
                continue
            Xs = np.vstack([X_of('md', sP1, val, SD[r]), X_of('md', sN1, val, SD[r])])
            w_s, c_s = fit_axis(Xs, np.r_[np.ones(len(sP1), int), np.zeros(len(sN1), int)])
            Xl = np.vstack([X_of('decision', lP1, val, SD[r]), X_of('decision', lN1, val, SD[r])])
            w_l, c_l = fit_axis(Xl, np.r_[np.ones(len(lP1), int), np.zeros(len(lN1), int)])
            planes[r] = (np.linalg.qr(np.stack([w_s, w_l], 1))[0], {'sample': c_s, 'choice': c_l})
        if not planes:
            continue
        for st in STAGES:
            for tk in TASKS:
                # the OTHER axis of the in-context plane, at its canonical window, per reference scale
                (cp1, _), (cn1, _) = H[(st, tk, 'choice')]
                (sp1, _), (sn1, _) = H[(st, tk, 'sample')]
                for v in WIN:
                    (p1, p2), (n1, n2) = H[(st, tk, v)]
                    if min(len(p1), len(n1), len(p2), len(n2)) < MINN:
                        continue
                    ytr = np.r_[np.ones(len(p1), int), np.zeros(len(n1), int)]
                    yte = np.r_[np.ones(len(p2), int), np.zeros(len(n2), int)]
                    for win in WIN[v]:
                        for r, (Q, axclf) in planes.items():
                            Xtr = np.vstack([X_of(win, p1, val, SD[r]), X_of(win, n1, val, SD[r])])
                            Xte = np.vstack([X_of(win, p2, val, SD[r]), X_of(win, n2, val, SD[r])])
                            a_fu = bacc(make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr).predict(Xte), yte)
                            a_pl = lr2d(Xtr, ytr, Xte, yte, Q)
                            Xtr_o = Xtr - (Xtr @ Q) @ Q.T; Xte_o = Xte - (Xte @ Q) @ Q.T
                            a_ou = bacc(make_clf(Xtr_o.shape[1], Xtr_o.shape[0]).fit(Xtr_o, ytr).predict(Xte_o), yte)
                            a_tr = bacc(axclf[v].predict(Xte), yte)
                            # in-context plane: this variable's axis at THIS window + the other axis
                            a_ic = np.nan
                            w_own, _ = fit_axis(Xtr, ytr)
                            if v == 'sample' and min(len(cp1), len(cn1)) >= MINN:
                                Xo = np.vstack([X_of('decision', cp1, val, SD[r]), X_of('decision', cn1, val, SD[r])])
                                w_oth, _ = fit_axis(Xo, np.r_[np.ones(len(cp1), int), np.zeros(len(cn1), int)])
                                a_ic = lr2d(Xtr, ytr, Xte, yte, np.linalg.qr(np.stack([w_own, w_oth], 1))[0])
                            elif v == 'choice' and min(len(sp1), len(sn1)) >= MINN:
                                Xo = np.vstack([X_of('md', sp1, val, SD[r]), X_of('md', sn1, val, SD[r])])
                                w_oth, _ = fit_axis(Xo, np.r_[np.ones(len(sp1), int), np.zeros(len(sn1), int)])
                                a_ic = lr2d(Xtr, ytr, Xte, yte, np.linalg.qr(np.stack([w_oth, w_own], 1))[0])
                            acc[r][v].setdefault((st, tk, win), []).append((a_fu, a_pl, a_ou, a_tr, len(yte), a_ic))
    cells[mo] = {r: {v: {ctx: dict(full=float(np.mean([t[0] for t in L])), plane=float(np.mean([t[1] for t in L])),
                                    resid=float(np.mean([t[2] for t in L])), transfer=float(np.mean([t[3] for t in L])),
                                    n=int(np.mean([t[4] for t in L])), iplane=float(np.nanmean([t[5] for t in L])))
                         for ctx, L in acc[r][v].items()} for v in WIN} for r in REFS}
    ex = cells[mo][('Naive', 'DPA')]['sample']
    line = ' '.join(f"{tk[:4]}{st[0]}@md:{ex[(st, tk, 'md')]['plane']:.2f}/{ex[(st, tk, 'md')]['iplane']:.2f}/{ex[(st, tk, 'md')]['full']:.2f}"
                    for st in STAGES for tk in TASKS if (st, tk, 'md') in ex)
    print(f'  {mo:8s} ({time.time() - t0:4.0f}s) ref Naive-DPA sample@md plane/iplane/full: {line}', flush=True)


def summarise(cells):
    """Across-mouse statistics per (ref, var, ctx)."""
    S = {}
    for r in REFS:
        S[r] = {}
        for v in WIN:
            S[r][v] = {}
            ctxs = sorted({c for mo in cells for c in cells[mo][r][v]})
            for ctx in ctxs:
                rows = [cells[mo][r][v][ctx] for mo in MICE if ctx in cells[mo][r][v]]
                fu = np.array([d['full'] for d in rows]); pl = np.array([d['plane'] for d in rows])
                ou = np.array([d['resid'] for d in rows]); tr = np.array([d['transfer'] for d in rows])
                ic = np.array([d['iplane'] for d in rows])
                ok = fu > 0.52; okc = np.isfinite(ic) & (ic > 0.52)
                ratio = (pl[ok] - 0.5) / (fu[ok] - 0.5) if ok.sum() else np.array([])
                ratio_ic = (pl[okc] - 0.5) / (ic[okc] - 0.5) if okc.sum() else np.array([])
                ent = dict(n_mice=len(rows), full=(fu.mean(), sem(fu)), plane=(pl.mean(), sem(pl)),
                           iplane=(float(np.nanmean(ic)), float(sem(ic[np.isfinite(ic)])) if np.isfinite(ic).sum() > 1 else np.nan),
                           resid=(ou.mean(), sem(ou)), transfer=(tr.mean(), sem(tr)),
                           ratio_n=int(ok.sum()), ratio_median=float(np.median(ratio)) if len(ratio) else np.nan,
                           ratio_ic_n=int(okc.sum()), ratio_ic_median=float(np.median(ratio_ic)) if len(ratio_ic) else np.nan,
                           p_plane_vs_full=float(wilcoxon(pl, fu).pvalue) if len(rows) > 5 and np.any(pl != fu) else np.nan,
                           p_plane_vs_iplane=float(wilcoxon(pl[okc], ic[okc]).pvalue) if okc.sum() > 5 and np.any(pl[okc] != ic[okc]) else np.nan,
                           p_resid_vs_chance=float(wilcoxon(ou - 0.5).pvalue) if len(rows) > 5 and np.any(ou != 0.5) else np.nan,
                           delta_plane_iplane=(float(np.nanmean(pl[okc] - ic[okc])), float(sem(pl[okc] - ic[okc]))) if okc.sum() > 1 else (np.nan, np.nan))
                S[r][v][ctx] = ent
    return S


summary = summarise(cells)
print('\n== SUMMARY (across mice; ratio_ic = fixed plane / in-context plane, chance-referenced) ==')
for r in REFS:
    for v in WIN:
        print(f'-- reference {r[0]}-{r[1]} plane, variable {v}')
        for ctx, e in summary[r][v].items():
            tag = 'WITHIN' if (ctx[0], ctx[1]) == r and ctx[2] == AXWIN[v] else 'ooc'
            print(f"   {ctx[0]:6s} {ctx[1]:8s} @{ctx[2]:8s} {tag:6s} full {e['full'][0]:.2f}  iplane {e['iplane'][0]:.2f}  plane {e['plane'][0]:.2f}  "
                  f"resid {e['resid'][0]:.2f}  transfer {e['transfer'][0]:.2f}  ratio_ic med {e['ratio_ic_median']:.2f} (n={e['ratio_ic_n']}) "
                  f"p(plane=iplane) {e['p_plane_vs_iplane']:.2f}  ratio_full med {e['ratio_median']:.2f}")

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['OOC_PLANE' + SUF] = dict(cells=cells, summary=summary, refs=REFS, win=WIN, axwin=AXWIN, nrep=NREP)
pickle.dump(d, open(RES, 'wb'))
print('merged OOC_PLANE' + SUF, 'into', RES, f'({time.time() - t0:.0f}s)')
