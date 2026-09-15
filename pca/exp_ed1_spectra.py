"""exp_ed1_spectra.py — the twelve-condition spectra and the participation-ratio ladder at the CANONICAL windows,
with the ONE cvPCA implementation of Fig. 2b (cvpca.py), for ED 1a,b (2026-09-15; Leon: "panel a compares the
late delay conditions why? ... is cvpca here the same implementation as in the main figure?").

The legacy caches CV / PR_JK were built by exp_dimensionality.py's own copy of the estimator at the
pre-unification windows (late delay 48–53, decision 57–65) with a different neuron scale (SD over trials x time
of the raw tensor). Here: cvpca.spectra (30 half-splits, seed 7; per-neuron SD of the window-averaged state),
the shuffle null (seed 11), and PR with its leave-one-mouse-out jackknife SE, at mid-delay (bins 33–38) and
decision (54–62) — Fig. 2b's windows — for the DPA set (4 conditions) and the all-tasks set (12 conditions).
Merge-dumps {'ED1_SPEC'} into results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_ed1_spectra.py
"""
import sys, os, pickle, time
sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvpca

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW, VALIDIX, N = _c['AW'], _c['VALIDIX'], _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
cvpca.bind(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO, VALIDIX=VALIDIX, N=N, MICE=MICE)
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
DPA4 = [c for c in ALL12 if c[0] == 'DPA']
SETS = [('DPA', 'md', DPA4), ('all', 'md', ALL12), ('all', 'decision', ALL12)]
STAGES = ['Naive', 'Expert']
out = {}
t0 = time.time()
for ts, wn, conds in SETS:
    M = AW[wn]
    for stage in STAGES:
        spec = cvpca.avg_spec(stage, conds, M, nsplits=30, seed=7)
        pr = cvpca.pr_of(spec)
        jk = np.array([cvpca.pr_of(cvpca.avg_spec(stage, conds, M, [m for m in MICE if m != mo], nsplits=30, seed=7)) for mo in MICE])
        se = np.sqrt((len(MICE) - 1) / len(MICE) * np.sum((jk - jk.mean()) ** 2))
        null = cvpca.avg_spec(stage, conds, M, nsplits=30, seed=11, shuffle=True) if stage == 'Expert' else None
        out[(ts, wn, stage)] = dict(spec=spec, null=null, pr=pr, se=se, jk=jk)
        pos = np.clip(spec, 0, None); frac = pos / pos.sum()
        print(f'{ts:4s} {wn:9s} {stage:6s} PR={pr:.2f} ± {2.306*se:.2f} (t8 CI)  fractions {np.round(frac[:4], 3)}  ({time.time()-t0:.0f}s)', flush=True)
RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb')); d['ED1_SPEC'] = out; pickle.dump(d, open(RES, 'wb'))
print('merged ED1_SPEC into', RES)
