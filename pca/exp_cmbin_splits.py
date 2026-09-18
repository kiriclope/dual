"""exp_cmbin_splits.py — per-bin condition means for BOTH halves of several random trial splits, so that an
unsupervised basis can be fitted on one half and the trajectories taken from the other (Leon 2026-09-18: "can you get
me a cross validated version of the figure?").

results.pkl['CMBIN_H1'] stores only half 1, which is enough for a decoder fitted from window-averaged trials but not
for fitting a BASIS on per-bin states. One pass over the 20 GB tensor writes, per stage and split, the per-bin
condition means of each half:

  CMBIN_SPLITS[(stage, split, half)]   (12, N, 84) float32   half in {0, 1}, the halves disjoint by construction

Written to its own pickle (~160 MB) so results.pkl stays small.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_cmbin_splits.py [--nsplit 6]
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.pca.io import pkl_load

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
NSPLIT = int(sys.argv[sys.argv.index('--nsplit') + 1]) if '--nsplit' in sys.argv else 6
SEED = 4242
OUT = 'figures/pseudo/dimensionality/cmbin_splits.pkl'

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

POOL = {(m, stage, ci): np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                                 & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
        for m in MICE for stage in STAGES for ci, (t, s, te) in enumerate(ALL12)}
print('loading X (20 GB, one pass) …', flush=True)
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
NBINS = X.shape[2]
print('X', X.shape, flush=True)

SPL = {}
rng = np.random.RandomState(SEED)
for stage in STAGES:
    for sp in range(NSPLIT):
        CM = [np.zeros((len(ALL12), N, NBINS), dtype=np.float32) for _ in range(2)]
        for m in MICE:
            val = VALIDIX[(m, stage)]
            for ci in range(len(ALL12)):
                p = rng.permutation(POOL[(m, stage, ci)]); h = len(p) // 2
                for half, rows in enumerate((p[:h], p[h:])):
                    if len(rows):
                        CM[half][ci][val] = np.nanmean(X[np.ix_(rows, val)], axis=0)
        for half in (0, 1):
            SPL[(stage, sp, half)] = CM[half]
        print(f'  {stage} split {sp + 1}/{NSPLIT} done', flush=True)
del X
pickle.dump(dict(splits=SPL, nsplit=NSPLIT, seed=SEED), open(OUT, 'wb'))
print('wrote', OUT, f'({os.path.getsize(OUT) / 1e6:.0f} MB)')
