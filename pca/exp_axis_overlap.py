"""Do the gng (Go/NoGo) axis and the DPA choice (match/lick) axis overlap? (decision window)

Three demixed axes (contrast on z-scored condition means, exp_dpca_count convention):
  gng        — dual set, Go(+1, lick at cue) vs NoGo(-1)
  choice_DPA — DPA set, match(+1, lick at test) vs nonmatch(-1)
  choice_dual— dual set, same contrast
Positive contrast side = lick in all three, so a POSITIVE cosine = lick-with-lick alignment.

Reliability-corrected cosines: axes built on INDEPENDENT trial halves (15 splits);
corrected cos(A,B) = <cos(wA_h1, wB_h2)> / sqrt(rel_A * rel_B), rel = same-axis across-half cosine
(attenuation correction: raw cross-half cosines are deflated by axis estimation noise; corrected
values can slightly exceed 1 when the true overlap is complete).

RESULT (2026-08-11): choice_DPA x choice_dual corrected +0.97 (N) / +1.05 (E) — the SAME axis, task-
invariant. gng x choice corrected +0.14 (N) -> +0.19/0.20 (E) — nearly orthogonal with a small REAL
lick-aligned shared component that grows with learning. NOTE: this replicates (does not discover) the
SETTLED overlaps shared-action result — the cross-temporal weight matrices already showed choice x gng
orthogonal on the delay diagonal but aligned at the action/reward block, +0.15 -> +0.24 paired-t
p=.005 (fig_overlaps_codes_supp.py row 2; Fig 3 panel-A shared-action trajectories). dPCA gives
0.147 -> 0.222. Three pipelines, same number.

Print-only (no cache merge). Run: cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_axis_overlap.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
DUAL = [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
M = AW['decision']

AXES = {  # name -> (cond set, contrast fn over conds)
    'gng': (DUAL, lambda c: 1.0 if c[0] == 'DualGo' else -1.0),
    'choice_DPA': (DPA4, lambda c: 1.0 if c[1] == c[2] else -1.0),
    'choice_dual': (DUAL, lambda c: 1.0 if c[1] == c[2] else -1.0),
}


def neuron_scale(stage):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def half_axes(stage, sd, rng):
    """One split -> {name: (w_half1, w_half2)} (unit vectors)."""
    out = {}
    halves = {}
    for m in MICE:
        for t, s, te in set(c for cs, _ in AXES.values() for c in cs):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            p = rng.permutation(idx); h = len(p) // 2
            halves[(m, (t, s, te))] = (p[:h], p[h:])
    for name, (conds, cf) in AXES.items():
        c = np.array([cf(cd) for cd in conds])
        ws = []
        for hi in (0, 1):
            R = np.zeros((len(conds), N))
            for m in MICE:
                val = VALIDIX[(m, stage)]
                for ci, cd in enumerate(conds):
                    tr = halves[(m, cd)][hi]
                    if len(tr):
                        R[ci][val] = np.nanmean(M[np.ix_(tr, val)], 0)
            R = R / sd[None, :]; R = R - R.mean(0, keepdims=True)
            w = (c @ R) / (c @ c)
            ws.append(w / (np.linalg.norm(w) + 1e-12))
        out[name] = ws
    return out


for stage in STAGES:
    sd = neuron_scale(stage)
    rng = np.random.RandomState(11)
    pairs = [('gng', 'choice_DPA'), ('gng', 'choice_dual'), ('choice_DPA', 'choice_dual')]
    cross = {p: [] for p in pairs}; rel = {n: [] for n in AXES}
    for _ in range(15):
        HA = half_axes(stage, sd, rng)
        for n in AXES:
            rel[n].append(float(HA[n][0] @ HA[n][1]))
        for a, b in pairs:
            cross[(a, b)].append(0.5 * (float(HA[a][0] @ HA[b][1]) + float(HA[a][1] @ HA[b][0])))
    print(f'\n== {stage} (decision window) ==')
    for n in AXES:
        print(f'  reliability cos({n}): {np.mean(rel[n]):.3f}')
    for a, b in pairs:
        raw = np.mean(cross[(a, b)])
        corr = raw / np.sqrt(max(np.mean(rel[a]), 1e-6) * max(np.mean(rel[b]), 1e-6))
        sdv = np.std(cross[(a, b)]) / np.sqrt(max(np.mean(rel[a]), 1e-6) * max(np.mean(rel[b]), 1e-6))
        print(f'  overlap {a:11s} x {b:11s}: raw {raw:+.3f}  CORRECTED {corr:+.3f} (split sd {sdv:.3f})')
