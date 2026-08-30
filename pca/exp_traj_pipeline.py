"""Panel-a code traces through THE SHARED PIPELINE (`decoders.py`), so every panel of Fig 3 uses one
decoder — replacing the ORIG_TRACES replay of the overlaps CCGD projections.

WHY PER-BIN. Fitting one axis at a single window and projecting it across the trial drags the
condition-INDEPENDENT ramp onto the trace (measured at 29-45% of the code size): it shifts both
classes together and makes the traces asymmetric. A decoder fit AT each time bin cannot do that —
the common-mode signal at that bin is identical for the two classes, so it is not discriminative and
does not project. That, not the estimator, is what made the overlaps traces clean.

Per (stage, mouse, code, bin): split the trials in half, fit `decoders.make_clf` on one half, take
the decision_function on the OTHER half, both directions, average. Cross-validated by construction;
the per-mouse class means are what the figure plots.

Codes follow the original panel A (overlaps/main_panels.py:486-491):
  sample  sample odour A/B, DPA, correct
  dist    Go vs NoGo, dual, correct
  test    test odour C/D, DPA, correct
  choice  lick vs no-lick, DPA, correct (= match on correct trials)

Output: merges {'PIPE_TRACES': {(stage, code, level): (n_mice, n_bins)}, 'PIPE_XTIME'} into
results.pkl, suffixed by the decoder variant (`--nopca`, `--npc N`).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_traj_pipeline.py
"""
import sys, os, warnings, pickle, time
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import make_clf, SUF
from src.pca.io import pkl_load

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
SEED = 23

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)

# code -> (task subset, per-trial label array)
CODES = {'sample': ('DPA',  SAMP.astype(int)),
         'dist':   ('dual', (TSK == 'DualGo').astype(int)),
         'test':   ('DPA',  TESTO.astype(int)),
         'choice': ('DPA',  MATCH.astype(int))}

print('loading X (20 GB, one pass) …', flush=True)
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
NB = X.shape[2]
print('X', X.shape, flush=True)

OUT = {}
t0 = time.time()
for stage in STAGES:
    for code, (tset, LAB) in CODES.items():
        acc = {0: [], 1: []}
        for m in MICE:
            val = VALIDIX[(m, stage)]
            base = ((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                    & ((TSK == 'DPA') if tset == 'DPA' else (TSK != 'DPA')))
            rows = np.where(base)[0]
            y = LAB[rows]
            if len(rows) < 12 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 4:
                continue
            rng = np.random.RandomState(SEED)
            h = rng.permutation(len(rows)); k = len(h) // 2
            folds = [(h[:k], h[k:]), (h[k:], h[:k])]
            dv = np.full((len(rows), NB), np.nan)
            for tr, te in folds:
                if len(np.unique(y[tr])) < 2:
                    continue
                for b in range(NB):
                    Xb = np.nan_to_num(X[np.ix_(rows, val)][:, :, b])
                    clf = make_clf(Xb.shape[1], len(tr)).fit(Xb[tr], y[tr])
                    dv[te, b] = clf.decision_function(Xb[te])
            for lv in (0, 1):
                sel = (y == lv) & np.isfinite(dv[:, 0])
                if sel.sum() >= 3:
                    acc[lv].append(np.nanmean(dv[sel], axis=0))
        for lv in (0, 1):
            OUT[(stage, code, lv)] = np.asarray(acc[lv])
        n = len(acc[0])
        print(f'{stage:6s} {code:7s} n={n} mice   '
              f'class0 {np.asarray(acc[0]).mean(0).min():+.2f}..{np.asarray(acc[0]).mean(0).max():+.2f}  '
              f'class1 {np.asarray(acc[1]).mean(0).min():+.2f}..{np.asarray(acc[1]).mean(0).max():+.2f}  '
              f'[{time.time() - t0:.0f}s]', flush=True)
del X

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PIPE_TRACES' + SUF] = OUT
d['PIPE_XTIME'] = np.arange(NB) / 6.0 - 0.5
pickle.dump(d, open(RES, 'wb'))
print(f'merged PIPE_TRACES{SUF} into {RES}  [{time.time() - t0:.0f}s]')
