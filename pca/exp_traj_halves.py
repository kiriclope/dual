"""Held-out per-bin condition means, so the Fig-3 trajectory row can use PER-MOUSE decoders without
self-inclusion leakage.

`CMBIN` (from exp_antact_traj.py) averages ALL correct trials, so projecting it onto an axis fitted
from those same trials inflates the trace — the exact leakage that faked a mid-delay lick split
earlier in this project. Here one pass over X splits every (mouse, stage, condition) trial pool in
half with a fixed seed and stores:

  CMBIN_H1[stage]                 (12, N, 84) float32 — per-bin condition means from HALF 1 only
                                  (what gets plotted)
  TRAJ_FIT[(mouse, stage, ci)]    the HALF 0 trial indices (what the per-mouse decoders are fit on)

The two halves are disjoint, so a per-mouse axis fit on TRAJ_FIT can be applied to CMBIN_H1 cleanly.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_traj_halves.py
      (~1 pass over the 20 GB X; merges into results.pkl)
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.pca.io import pkl_load

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SEED = 17

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

print('loading X (20 GB, one pass) …', flush=True)
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
NBINS = X.shape[2]
print('X', X.shape, flush=True)

CMH, FIT = {}, {}
rng = np.random.RandomState(SEED)
for stage in STAGES:
    CM = np.zeros((len(ALL12), N, NBINS), dtype=np.float32)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(ALL12):
            rows = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                            & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            p = rng.permutation(rows); h = len(p) // 2
            FIT[(m, stage, ci)] = p[:h]                        # half 0 → fits the decoder
            plot_rows = p[h:]                                  # half 1 → plotted
            if len(plot_rows):
                CM[ci][val] = np.nanmean(X[np.ix_(plot_rows, val)], axis=0)
    CMH[stage] = CM
    n0 = np.median([len(FIT[(m, stage, ci)]) for m in MICE for ci in range(12)])
    print(f'{stage}: median fit-half trials per (mouse, condition) = {n0:.0f}', flush=True)
del X

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['CMBIN_H1'] = CMH; d['TRAJ_FIT'] = FIT
pickle.dump(d, open(RES, 'wb'))
print('merged CMBIN_H1 + TRAJ_FIT into', RES)
