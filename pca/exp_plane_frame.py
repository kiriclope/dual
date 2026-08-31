"""'One manifold' support caches for Fig 3 (2026-08-31, user request): does the sample x choice
PLANE suffice, and is it the SAME plane across learning?

  PLANE_VAR   fraction of RELIABLE condition variance captured by the sample x choice plane,
              per (set x window x stage). Estimator: axes fit on trial-half A (the figure's own
              construction); condition means from two INDEPENDENT quarter-splits B1/B2 of the
              remaining half; reliable variance = cross-half dot product of centred means
              (noise-unbiased in numerator AND denominator):
                  tot = <R1c, R2c> ,  inplane = <Q'R1c, Q'R2c> ,  frac = inplane / tot
              with Q = orthonormalised [sample axis, choice axis]. Axes trained on A only ->
              no self-inclusion inflation of the in-plane share.
  PLANE_DEC   decoding from the 2-D plane alone vs the full space: per (set x window x variable),
              held-out balanced accuracy using ONLY the two plane coordinates (logistic) vs the
              full-space pipeline (decoders.make_clf). Train on A-pseudo, test on B-pseudo.
  AXIS_XSTAGE is the Expert frame the NAIVE frame? attenuation-corrected cross-stage axis cosines
              (neurons are registered across stages — identical valid masks):
                  corrected = mean(|cos(aN_h1, aE_h2)|, |cos(aN_h2, aE_h1)|) / sqrt(rel_N * rel_E)
              per axis (sample @ md, choice @ decision), rel = split-half |cos| within stage.

Frame construction copied from fig_manifold_main.build_frame (importing the figure would render
it). Respects the decoders --nopca/--npc flags; cache keys carry SUF. Merge-dumps
{'PLANE_VAR','PLANE_DEC','AXIS_XSTAGE'}+SUF into results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_plane_frame.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from decoders import fit_axis, make_clf, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
B_SPECS = [('DPA', 'md'), ('DPA', 'decision'), ('dual', 'md'), ('dual', 'delay'), ('dual', 'decision')]
NREP, KP = 8, 60

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
assert {'md', 'decision', 'delay', 'delay+dec'} <= set(AW), 'run exp_dimensionality_md.py first'
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)          # behavioural lick (error trials included)


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
    """Per (mouse, cond, perf): half A (axis/decoder training) + quarters B1/B2 (evaluation)."""
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


def frame_axes(stage, sd, H, part, rng):
    """Sample (@md, all 12 conds) + behavioural lick (@decision) axes from partition `part`
    (0=A, 1=B1, 2=B2) — the figure's own construction."""
    M = AW['md']
    blocks = {0: [], 1: []}
    for ci, cd in enumerate(ALL12):
        pools = {m: H[(m, ci, 1)][part] for m in MICE}
        blocks[cd[1]].append(_pseudo(M, sd, stage, pools, KP, rng))
    w_s, _ = fit_axis(np.vstack(blocks[1] + blocks[0]),
                      np.r_[np.ones(len(blocks[1]) * KP, int),
                            np.zeros(len(blocks[0]) * KP, int)])
    M = AW['decision']
    out = []
    for lickval in (True, False):
        pools = {}
        for m in MICE:
            pool = [H[(m, ci, pf)][part] for ci in range(len(ALL12)) for pf in (0, 1)]
            idx = np.concatenate(pool) if pool else np.array([], int)
            pools[m] = idx[LICK[idx] == lickval]
        out.append(_pseudo(M, sd, stage, pools, KP, rng))
    w_l, _ = fit_axis(np.vstack(out), np.r_[np.ones(KP, int), np.zeros(KP, int)])
    return w_s, w_l


def cond_means(M, sd, stage, H, conds, part):
    """(n_conds, N) condition means from partition `part`, correct trials, sd-scaled."""
    R = np.zeros((len(conds), N))
    for ci_l, cd in enumerate(conds):
        ci = ALL12.index(cd)
        for m in MICE:
            val = VALIDIX[(m, stage)]; idx = H[(m, ci, 1)][part]
            if len(idx):
                R[ci_l][val] = np.nanmean(M[np.ix_(idx, val)], 0) / sd[val]
    return R


VAR_DEFS = [('sample', lambda cd: cd[1], None),                      # decodable label per cond
            ('dist', lambda cd: int(cd[0] == 'DualGo'), 'dual'),     # dual only
            ('test', lambda cd: cd[2], None),
            ('choice', lambda cd: int(cd[1] == cd[2]), None)]        # correct-trial lick

print(f'══ PLANE_VAR / PLANE_DEC / AXIS_XSTAGE (SUF="{SUF}") ══', flush=True)
PLANE_VAR, PLANE_DEC, AXQ = {}, {}, {}
for stage in STAGES:
    sd = neuron_scale(stage, AW['delay+dec'])
    tot_acc = {}
    for rep in range(NREP):
        rng = np.random.RandomState(40 + rep)
        H = split_ABB(stage, rng)
        w_s, w_l = frame_axes(stage, sd, H, 0, rng)                 # axes from half A
        AXQ.setdefault(stage, []).append((w_s, w_l))
        Q = np.linalg.qr(np.stack([w_s, w_l], 1))[0]                # orthonormal plane basis
        for sname, wn in B_SPECS:
            M = AW[wn]; conds = SETS[sname]
            R1 = cond_means(M, sd, stage, H, conds, 1)
            R2 = cond_means(M, sd, stage, H, conds, 2)
            R1c = R1 - R1.mean(0, keepdims=True); R2c = R2 - R2.mean(0, keepdims=True)
            tot = float((R1c * R2c).sum())
            inp = float(((R1c @ Q) * (R2c @ Q)).sum())
            PLANE_VAR.setdefault((sname, wn, stage), []).append((inp, tot))
        # decode from the plane vs full space (md + decision, both sets)
        for sname in ['DPA', 'dual']:
            for wn in ['md', 'decision']:
                M = AW[wn]; conds = SETS[sname]
                for vname, lab_fn, only in VAR_DEFS:
                    if only and sname != only:
                        continue
                    labs = np.array([lab_fn(cd) for cd in conds])
                    if len(set(labs)) < 2:
                        continue
                    def pseudo_set(part):
                        Xs, ys = [], []
                        for ci_l, cd in enumerate(conds):
                            ci = ALL12.index(cd)
                            pools = {m: H[(m, ci, 1)][part] for m in MICE}
                            Xs.append(_pseudo(M, sd, stage, pools, 24, rng))
                            ys.append(np.full(24, labs[ci_l]))
                        return np.vstack(Xs), np.concatenate(ys)
                    Xtr, ytr = pseudo_set(0); Xte, yte = pseudo_set(2)
                    accs = {}
                    clf2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                    clf2.fit(Xtr @ Q, ytr)
                    pred = clf2.predict(Xte @ Q)
                    accs['plane'] = np.mean([np.mean(pred[yte == c] == c) for c in set(yte)])
                    clfF = make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr)
                    pred = clfF.predict(Xte)
                    accs['full'] = np.mean([np.mean(pred[yte == c] == c) for c in set(yte)])
                    PLANE_DEC.setdefault((sname, wn, vname, stage), []).append(
                        (accs['plane'], accs['full']))
    for sname, wn in B_SPECS:
        v = PLANE_VAR[(sname, wn, stage)]
        fr = np.sum([a for a, _ in v]) / np.sum([b for _, b in v])
        print(f'  PLANE_VAR {stage:6s} {sname:4s} {wn:9s} in-plane {fr:.2f}', flush=True)

# ── XSTAGE_DEC: is the Expert frame the NAIVE frame? Answered by CROSS-STAGE DECODING (the
#    robust check — the corrected cross-stage cosine below EXPLODES at rel~0.15, same disease as
#    the retired per-mouse attenuation correction; kept only as a logged dead-end). Train the
#    sample (@md) / choice (@decision) decoder on one stage's half-A pseudo-trials, test held-out
#    within-stage (B2) and cross-stage (other stage's B2). transfer/within ~ 1 => same functional
#    axis across learning. Registered neurons make the cross-stage projection well-defined. ──
print('── XSTAGE_DEC ──', flush=True)
XSTAGE_DEC = {}
for rep in range(NREP):
    rng = np.random.RandomState(240 + rep)
    Hs = {st: split_ABB(st, rng) for st in STAGES}
    sds = {st: neuron_scale(st, AW['delay+dec']) for st in STAGES}

    def sample_sets(st, part):
        Xs, ys = [], []
        for ci, cd in enumerate(ALL12):
            pools = {m: Hs[st][(m, ci, 1)][part] for m in MICE}
            Xs.append(_pseudo(AW['md'], sds[st], st, pools, 24, rng))
            ys.append(np.full(24, cd[1]))
        return np.vstack(Xs), np.concatenate(ys)

    def choice_sets(st, part):
        Xs, ys = [], []
        for lickval in (True, False):
            pools = {}
            for m in MICE:
                pool = [Hs[st][(m, ci, pf)][part] for ci in range(len(ALL12)) for pf in (0, 1)]
                idx = np.concatenate(pool) if pool else np.array([], int)
                pools[m] = idx[LICK[idx] == lickval]
            Xs.append(_pseudo(AW['decision'], sds[st], st, pools, 48, rng))
            ys.append(np.full(48, int(lickval)))
        return np.vstack(Xs), np.concatenate(ys)

    for vname, getter in [('sample', sample_sets), ('choice', choice_sets)]:
        for st_tr in STAGES:
            Xtr, ytr = getter(st_tr, 0)
            clf = make_clf(Xtr.shape[1], Xtr.shape[0]).fit(Xtr, ytr)
            for st_te in STAGES:
                Xte, yte = getter(st_te, 2)
                pred = clf.predict(Xte)
                acc = np.mean([np.mean(pred[yte == c] == c) for c in set(yte)])
                XSTAGE_DEC.setdefault((vname, st_tr, st_te), []).append(float(acc))
for (vname, a, b), v in sorted(XSTAGE_DEC.items()):
    print(f'  XSTAGE_DEC {vname:7s} train {a:6s} -> test {b:6s}  acc {np.mean(v):.2f}', flush=True)

# cross-stage axis identity: rel within stage from (B1 vs B2)-fit axes; cross from A-fit axes
# (DEAD-END for display: corrected values >1 at rel~0.15 — cached for the record only)
print('── AXIS_XSTAGE (cosine dead-end, record only) ──', flush=True)
REL = {}
for stage in STAGES:
    sd = neuron_scale(stage, AW['delay+dec'])
    rs, rl = [], []
    for rep in range(NREP):
        rng = np.random.RandomState(140 + rep)
        H = split_ABB(stage, rng)
        s1, l1 = frame_axes(stage, sd, H, 1, rng)
        s2, l2 = frame_axes(stage, sd, H, 2, rng)
        rs.append(abs(float(s1 @ s2))); rl.append(abs(float(l1 @ l2)))
    REL[stage] = (float(np.mean(rs)), float(np.mean(rl)))
    print(f'  rel {stage:6s} sample {REL[stage][0]:.2f}  choice {REL[stage][1]:.2f}', flush=True)
raw_s = np.mean([abs(float(a[0] @ b[0])) for a, b in zip(AXQ['Naive'], AXQ['Expert'])])
raw_l = np.mean([abs(float(a[1] @ b[1])) for a, b in zip(AXQ['Naive'], AXQ['Expert'])])
AXIS_XSTAGE = dict(
    sample_raw=float(raw_s), choice_raw=float(raw_l),
    sample=float(raw_s / np.sqrt(REL['Naive'][0] * REL['Expert'][0])),
    choice=float(raw_l / np.sqrt(REL['Naive'][1] * REL['Expert'][1])),
    rel=REL)
print(f"  cross-stage corrected: sample {AXIS_XSTAGE['sample']:.2f} (raw {raw_s:.2f})  "
      f"choice {AXIS_XSTAGE['choice']:.2f} (raw {raw_l:.2f})", flush=True)

# ── PLANE_TRAJ: panel-A-style code traces read THROUGH the plane vs the full axis (2026-08-31).
#    Per stage: fit the frame axes (half A, rep-0 split — the storyboard's construction) plus a
#    dist axis (dual Go v NoGo @ md) and a test axis (@ test window); project the cached per-bin
#    condition means (CMBIN) onto (i) each code's FULL unit axis and (ii) the unit-normalised
#    IN-PLANE component P·w of that axis (P = sample x choice plane projector). Per-class mean
#    trace, baseline-centred (bins 0-11). Descriptive panel: solid (plane) vs dashed (full)
#    coincide where the plane suffices; test's plane trace stays flat. ──
print('── PLANE_TRAJ ──', flush=True)
_res0 = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
CMBIN = _res0['CMBIN']
PLANE_TRAJ = {}
for stage in STAGES:
    sd = neuron_scale(stage, AW['delay+dec'])
    rng = np.random.RandomState(40)                     # rep-0 split = the storyboard's axes
    H = split_ABB(stage, rng)
    w_s, w_l = frame_axes(stage, sd, H, 0, rng)
    Q = np.linalg.qr(np.stack([w_s, w_l], 1))[0]
    P = Q @ Q.T

    def axis_from_pools(M, pos_pools, neg_pools):
        Xp = np.vstack([_pseudo(M, sd, stage, pos_pools, KP, rng),
                        _pseudo(M, sd, stage, neg_pools, KP, rng)])
        w, _ = fit_axis(Xp, np.r_[np.ones(KP, int), np.zeros(KP, int)])
        return w

    def pool(cds, part=0):
        out = {m: [] for m in MICE}
        for cd in cds:
            ci = ALL12.index(cd)
            for m in MICE:
                out[m].append(H[(m, ci, 1)][part])
        return {m: np.concatenate(v) if v else np.array([], int) for m, v in out.items()}

    DUALC = SETS['dual']; DPAC = SETS['DPA']
    w_d = axis_from_pools(AW['md'], pool([c for c in DUALC if c[0] == 'DualGo']),
                          pool([c for c in DUALC if c[0] == 'DualNoGo']))
    w_t = axis_from_pools(AW['test'], pool([c for c in ALL12 if c[2] == 1]),
                          pool([c for c in ALL12 if c[2] == 0]))
    CM = np.asarray(CMBIN[stage], dtype=float)          # (12, N, 84)
    CMs = CM / sd[None, :, None]
    CODES = [('sample', w_s, [[c for c in DPAC if c[1] == 0], [c for c in DPAC if c[1] == 1]]),
             ('dist',   w_d, [[c for c in DUALC if c[0] == 'DualNoGo'],
                              [c for c in DUALC if c[0] == 'DualGo']]),
             ('test',   w_t, [[c for c in DPAC if c[2] == 0], [c for c in DPAC if c[2] == 1]]),
             ('choice', w_l, [[c for c in DPAC if c[1] != c[2]], [c for c in DPAC if c[1] == c[2]]])]
    for cname, w, classes in CODES:
        wp = P @ w
        wp = wp / max(np.linalg.norm(wp), 1e-9)         # unit in-plane component
        for which, wv in [('full', w), ('plane', wp)]:
            for cls, cds in enumerate(classes):
                idx = [ALL12.index(c) for c in cds]
                tr = np.einsum('n,cnt->t', wv, CMs[idx]) / len(idx)
                tr = tr - tr[0:12].mean()
                PLANE_TRAJ[(stage, cname, cls, which)] = tr
        sep_full = np.abs(PLANE_TRAJ[(stage, cname, 1, 'full')] -
                          PLANE_TRAJ[(stage, cname, 0, 'full')]).max()
        sep_pl = np.abs(PLANE_TRAJ[(stage, cname, 1, 'plane')] -
                        PLANE_TRAJ[(stage, cname, 0, 'plane')]).max()
        print(f'  {stage:6s} {cname:7s} max sep full {sep_full:5.2f}  plane {sep_pl:5.2f}  '
              f'in-plane cos {abs(float(w @ wp)):.2f}', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PLANE_VAR' + SUF] = {k: np.asarray(v) for k, v in PLANE_VAR.items()}
d['PLANE_DEC' + SUF] = {k: np.asarray(v) for k, v in PLANE_DEC.items()}
d['AXIS_XSTAGE' + SUF] = AXIS_XSTAGE
d['XSTAGE_DEC' + SUF] = {k: np.asarray(v) for k, v in XSTAGE_DEC.items()}
d['PLANE_TRAJ' + SUF] = PLANE_TRAJ
pickle.dump(d, open(RES, 'wb'))
print('merged PLANE_VAR/PLANE_DEC/AXIS_XSTAGE' + SUF, 'into', RES)
