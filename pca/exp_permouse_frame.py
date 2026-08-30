"""PER-MOUSE companions for Fig 3 b/c/d — the same three quantities computed WITHIN each animal, so
the pseudo-population matrices get an n=9 statistic instead of resting on one pooled estimate.

Everything runs on that mouse's OWN neurons (VALIDIX[(mouse, stage)]) and its own trials, from the
cached window matrices — no 20 GB X reload, no overlaps tensor.

  PM_COS  b: attenuation-corrected |cos| between sample@md / action@decision / distractor@md
             (Fig 3a/b windows — this companion sits under the axis-geometry matrices).
  PM_ACT  c: Go-vs-NoGo @md  <->  behavioural lick-vs-no-lick @decision, cross-decoded BOTH ways
             (held-out halves), plus each code's within-task accuracy.
  PM_GEN  d: cross-TASK generalisation over DPA / Go / NoGo, stored as the full 3x3 so the figure
             can normalise per column. Windows deliberately MATCH the pooled matrices this sits
             under — sample @ LATE DELAY, choice/test @ TEST (overlaps/fig_ccgp_matrices_pseudo.py:58)
             — NOT Fig 2's md/decision windows.

Decoder = the shared pipeline in `decoders.py` (StandardScaler -> PCA -> logistic regression), the
same one the pooled overlaps caches use. All estimates average NREP random half-splits.

Output: merges {'PM_COS','PM_ACT','PM_GEN'} into results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_permouse_frame.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import fit_axis, SUF                 # THE shared decoder (see decoders.py)

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
NREP = 12                    # logistic fits are far costlier than the old mean-difference axes

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
assert {'md', 'decision', 'delay', 'test'} <= set(AW), 'run exp_dimensionality_md.py first'
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)          # behavioural lick (error trials included)


def sel(mouse, stage, **kw):
    m = (MOUSE == mouse) & (LEARN == stage) & (LAS == 0)
    for k, v in kw.items():
        arr = {'task': TSK, 'samp': SAMP, 'test': TESTO, 'perf': PERF, 'lick': LICK}[k]
        m &= np.isin(arr, v) if isinstance(v, (list, tuple, np.ndarray)) else (arr == v)
    return np.where(m)[0]


def zscale(M, val, idx):
    sd = np.nanstd(M[np.ix_(idx, val)], axis=0)
    return np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def axis_mid(M, val, sd, pos, neg):
    """LOGISTIC-REGRESSION axis (decoders.fit_axis) on this mouse's own trials, + the training
    midpoint along it. Returns (unit axis, threshold) — same signature as the old mean-difference
    version so the callers below are unchanged."""
    if len(pos) < 3 or len(neg) < 3:
        return None, None
    X = np.vstack([M[np.ix_(pos, val)], M[np.ix_(neg, val)]]) / sd
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    w, _ = fit_axis(X, y)
    if np.linalg.norm(w) < 1e-9:
        return None, None
    a = np.nanmean(M[np.ix_(pos, val)], 0) / sd
    b = np.nanmean(M[np.ix_(neg, val)], 0) / sd
    return w, 0.5 * (a @ w + b @ w)


def bal_acc(M, val, sd, w, thr, pos, neg, sign=1):
    if not len(pos) or not len(neg):
        return np.nan
    pp = (M[np.ix_(pos, val)] / sd) @ w; nn = (M[np.ix_(neg, val)] / sd) @ w
    return 0.5 * (np.mean(sign * (pp - thr) > 0) + np.mean(sign * (nn - thr) < 0))


def halves(rng, idx):
    p = rng.permutation(idx); h = len(p) // 2
    return p[:h], p[h:]


PM_COS, PM_ACT, PM_GEN = {}, {}, {}
for stage in STAGES:
    for mo in MICE:
        val = VALIDIX[(mo, stage)]
        Mmd, Mdc = AW['md'], AW['decision']
        allc = sel(mo, stage, perf=1)
        if len(allc) < 20 or not len(val):
            continue
        sdm, sdd = zscale(Mmd, val, allc), zscale(Mdc, val, allc)
        rng = np.random.RandomState(11)

        # ── b: axis cosines (split-half, attenuation-corrected) ──────────────
        acc_raw = {k: [] for k in ['sa', 'sd', 'ad']}; rel = {k: [] for k in ['s', 'a', 'd']}
        for _ in range(NREP):
            W = {}
            for key, M, val_, sd_, pos, neg in [
                    ('s', Mmd, val, sdm, sel(mo, stage, perf=1, samp=1), sel(mo, stage, perf=1, samp=0)),
                    ('a', Mdc, val, sdd, sel(mo, stage, task='DPA', lick=True),
                     sel(mo, stage, task='DPA', lick=False)),
                    ('d', Mmd, val, sdm, sel(mo, stage, perf=1, task='DualGo'),
                     sel(mo, stage, perf=1, task='DualNoGo'))]:
                p1, p2 = halves(rng, pos); n1, n2 = halves(rng, neg)
                w1 = axis_mid(M, val_, sd_, p1, n1)[0]; w2 = axis_mid(M, val_, sd_, p2, n2)[0]
                W[key] = (w1, w2)
            if any(v[0] is None or v[1] is None for v in W.values()):
                continue
            for k in rel:
                rel[k].append(abs(W[k][0] @ W[k][1]))
            for lab, (i, j) in [('sa', ('s', 'a')), ('sd', ('s', 'd')), ('ad', ('a', 'd'))]:
                acc_raw[lab].append(0.5 * (abs(W[i][0] @ W[j][1]) + abs(W[i][1] @ W[j][0])))
        if not acc_raw['ad']:
            continue
        R = {k: float(np.mean(v)) for k, v in rel.items()}
        PM_COS[(mo, stage)] = {lab: float(np.mean(acc_raw[lab]) /
                                          np.sqrt(max(R[i] * R[j], 1e-9)))
                               for lab, (i, j) in [('sa', ('s', 'a')), ('sd', ('s', 'd')),
                                                   ('ad', ('a', 'd'))]}
        PM_COS[(mo, stage)]['rel'] = R

        # ── c: gng <-> lick cross-decoding, both directions ──────────────────
        got = {k: [] for k in ['w_g', 'w_l', 'g2l', 'l2g']}
        for _ in range(NREP):
            gP, gN = sel(mo, stage, perf=1, task='DualGo'), sel(mo, stage, perf=1, task='DualNoGo')
            lP, lN = sel(mo, stage, task='DPA', lick=True), sel(mo, stage, task='DPA', lick=False)
            gP1, gP2 = halves(rng, gP); gN1, gN2 = halves(rng, gN)
            lP1, lP2 = halves(rng, lP); lN1, lN2 = halves(rng, lN)
            wg, tg = axis_mid(Mmd, val, sdm, gP1, gN1)
            wl, tl = axis_mid(Mdc, val, sdd, lP1, lN1)
            if wg is None or wl is None:
                continue
            got['w_g'].append(bal_acc(Mmd, val, sdm, wg, tg, gP2, gN2))
            got['w_l'].append(bal_acc(Mdc, val, sdd, wl, tl, lP2, lN2))
            # cross: apply each axis in the OTHER window/labelling, threshold re-fit on train half
            _, thr_gl = axis_mid(Mdc, val, sdd, lP1, lN1)
            proj_p = (Mdc[np.ix_(lP1, val)] / sdd) @ wg; proj_n = (Mdc[np.ix_(lN1, val)] / sdd) @ wg
            b = 0.5 * (proj_p.mean() + proj_n.mean()); s = 1 if proj_p.mean() > proj_n.mean() else -1
            got['g2l'].append(bal_acc(Mdc, val, sdd, wg, b, lP2, lN2, sign=s))
            proj_p = (Mmd[np.ix_(gP1, val)] / sdm) @ wl; proj_n = (Mmd[np.ix_(gN1, val)] / sdm) @ wl
            b = 0.5 * (proj_p.mean() + proj_n.mean()); s = 1 if proj_p.mean() > proj_n.mean() else -1
            got['l2g'].append(bal_acc(Mmd, val, sdm, wl, b, gP2, gN2, sign=s))
        if got['g2l']:
            PM_ACT[(mo, stage)] = {k: float(np.nanmean(v)) for k, v in got.items()}

        # ── d: cross-TASK generalisation, full 3x3 per variable ──────────────
        # WINDOWS MUST MATCH THE POOLED MATRICES they sit under (Fig 3d):
        # overlaps/fig_ccgp_matrices_pseudo.py:58 uses sample @ LATE DELAY and choice/test @ TEST.
        # (An earlier version used md / 57-65 here, so the per-mouse scatters were measuring the
        #  same variables at different times than the matrices above them.)
        Mld, Mte = AW['delay'], AW['test']
        sdl, sdt = zscale(Mld, val, allc), zscale(Mte, val, allc)
        VARS = {'sample': (Mld, sdl, lambda t: (sel(mo, stage, perf=1, task=t, samp=1),
                                                sel(mo, stage, perf=1, task=t, samp=0))),
                'choice': (Mte, sdt, lambda t: (sel(mo, stage, task=t, lick=True),
                                                sel(mo, stage, task=t, lick=False))),
                'test':   (Mte, sdt, lambda t: (sel(mo, stage, perf=1, task=t, test=1),
                                                sel(mo, stage, perf=1, task=t, test=0)))}
        for vname, (M, sd_, getter) in VARS.items():
            acc = np.full((3, 3), np.nan); cnt = np.zeros((3, 3))
            tot = np.zeros((3, 3))
            for _ in range(NREP):
                H = {}
                ok = True
                for ti, t in enumerate(TASKS3):
                    P, N = getter(t)
                    if min(len(P), len(N)) < 4:
                        ok = False; break
                    H[ti] = (halves(rng, P), halves(rng, N))
                if not ok:
                    break
                for ti in range(3):
                    w, thr = axis_mid(M, val, sd_, H[ti][0][0], H[ti][1][0])
                    if w is None:
                        continue
                    for tj in range(3):
                        pp = (M[np.ix_(H[tj][0][0], val)] / sd_) @ w
                        nn = (M[np.ix_(H[tj][1][0], val)] / sd_) @ w
                        b = 0.5 * (pp.mean() + nn.mean()); s = 1 if pp.mean() > nn.mean() else -1
                        a = bal_acc(M, val, sd_, w, b, H[tj][0][1], H[tj][1][1], sign=s)
                        if np.isfinite(a):
                            tot[ti, tj] += a; cnt[ti, tj] += 1
            if cnt.min() > 0:
                PM_GEN[(mo, stage, vname)] = tot / cnt
    n = sum(1 for k in PM_COS if k[1] == stage)
    print(f'{stage}: {n} mice with cosines, '
          f'{sum(1 for k in PM_ACT if k[1] == stage)} with action cross-decode, '
          f'{sum(1 for k in PM_GEN if k[1] == stage) // 3} with generalisation')

for stage in STAGES:
    v = [PM_COS[(m, stage)]['ad'] for m in MICE if (m, stage) in PM_COS]
    a = [PM_ACT[(m, stage)] for m in MICE if (m, stage) in PM_ACT]
    print(f'{stage}: |cos|(action,distractor) median {np.median(v):.2f} (n={len(v)}); '
          f'cross-decode g2l {np.nanmean([x["g2l"] for x in a]):.2f} '
          f'l2g {np.nanmean([x["l2g"] for x in a]):.2f}')

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PM_COS' + SUF] = PM_COS; d['PM_ACT' + SUF] = PM_ACT; d['PM_GEN' + SUF] = PM_GEN
pickle.dump(d, open(RES, 'wb'))
print('merged PM_COS/PM_ACT/PM_GEN' + SUF + ' into', RES)
