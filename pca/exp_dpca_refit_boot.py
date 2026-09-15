"""Refit-dPCA mouse bootstrap for the axis-mixing panel (ED 3b) — reviewer item 2026-09-15.

ED 3b's animal-level interval resamples the nine mice's neuron columns of the FIXED per-stage dPCA decoders
(`pseudo_weights_…f-sample-test-tasks_dpca`). That keeps the decoders as fitted on all nine mice; a mouse
bootstrap should REFIT dPCA on each resampled set of mice. This script does that with the pipeline's own
primitives (src/pca/pseudo.build_pseudo_population + src/pca/dpca.dpca_decode; factors sample x test x tasks,
zscore, baseline-centred, q = 2, ridge 1e-2, correct laser-off trials — the reference build's settings):

  for each of B draws of nine mice with replacement (the SAME draw for Naive and Expert):
      P_b = the drawn mice's condition-average blocks concatenated along neurons (a mouse drawn twice appears twice)
      W_b = dpca_decode(P_b)   →  |cos| between the first components of every pair of marginals
      Δ_b = cos_Expert − cos_Naive
  percentile 95% CI and two-sided bootstrap p (fraction of draws on either side of 0) per pair.

Writes DPCA_REFIT_BOOT into figures/pseudo/dimensionality/results.pkl (point estimates, per-draw Δ, CI, p, and
the reference cosines recomputed here, which must match fig_ed6_dpca.py's N -> E values).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dpca_refit_boot.py [--B 1000]
"""
import sys, os, warnings, pickle, argparse
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from time import perf_counter
from joblib import Parallel, delayed
from src.pca.io import pkl_load
from src.pca.pseudo import build_pseudo_population
from src.pca.dpca import dpca_decode
from src.pca.utils import get_levels

ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=1000); ap.add_argument('--jobs', type=int, default=36)
args = ap.parse_args()
FACTORS = ['sample', 'test', 'tasks']; STAGES = ['Naive', 'Expert']
MARGS = ['sample', 'test', 'sample:test', 'tasks']; SH = {'sample': 'sample', 'test': 'test', 'sample:test': 'choice', 'tasks': 'task'}
PRS = [(a, b) for a in range(4) for b in range(a + 1, 4)]

t0 = perf_counter()
print('loading X_all_blcenter (10 GB) …', flush=True)
X_all = pkl_load('X_all_blcenter', path='../data/pca'); y_all = pkl_load('y_all_blcenter', path='../data/pca')
SL = pkl_load('mouse_slices', path='../data/pca'); MICE = list(SL.keys())
y_all['sample'] = y_all.sample_odor; y_all['test'] = y_all.test_odor          # as run_pseudo.py
print(f'  X_all {X_all.shape}  ({(perf_counter() - t0) / 60:.1f} min)', flush=True)

P = {}; SIZES = None
for st in STAGES:                                    # the reference build's clean set: correct laser-off trials
    m = ((y_all.laser == 0) & (y_all.learning == st) & (y_all.performance == 1)).to_numpy()
    yc = y_all.loc[m].reset_index(drop=True); levels = get_levels(yc, FACTORS)
    SIZES = [len(levels[f]) for f in FACTORS]
    P[st], _, _, _ = build_pseudo_population(X_all[m], yc, FACTORS, SL, levels=levels, epoch=None,
                                             bl_bins=slice(0, 12), norm='zscore')
    print(f'  {st}: P {P[st].shape}  sizes {SIZES}', flush=True)
del X_all


def cosines(W, labels):
    ix = {mg: labels.index(mg) for mg in MARGS}
    u = {mg: W[ix[mg]] / np.linalg.norm(W[ix[mg]]) for mg in MARGS}
    return np.array([abs(float(u[MARGS[a]] @ u[MARGS[b]])) for a, b in PRS])


def one_draw(pick):
    out = {}
    for st in STAGES:
        Pb = np.concatenate([P[st][:, SL[MICE[k]], :] for k in pick], axis=1)
        W, _, labels, _ = dpca_decode(Pb, SIZES, FACTORS, q=2, ridge=1e-2)
        out[st] = cosines(W, labels)
    return out['Expert'] - out['Naive'], out['Naive'], out['Expert']


REF = {}
for st in STAGES:
    W, _, labels, _ = dpca_decode(P[st], SIZES, FACTORS, q=2, ridge=1e-2); REF[st] = cosines(W, labels)
print('\nreference |cos| (must match fig_ed6_dpca.py panel b):')
for i, (a, b) in enumerate(PRS):
    print(f'  {SH[MARGS[a]]:>6}-{SH[MARGS[b]]:<6} N {REF["Naive"][i]:.3f} -> E {REF["Expert"][i]:.3f}  Δ {REF["Expert"][i] - REF["Naive"][i]:+.3f}')

rng = np.random.RandomState(0)
PICKS = [rng.randint(0, len(MICE), len(MICE)) for _ in range(args.B)]
print(f'\nrefit bootstrap: {args.B} draws × 2 stages on {args.jobs} workers …', flush=True)
t1 = perf_counter()
res = Parallel(n_jobs=args.jobs, verbose=2)(delayed(one_draw)(p) for p in PICKS)
D = np.array([r[0] for r in res]); CN = np.array([r[1] for r in res]); CE = np.array([r[2] for r in res])
print(f'  done in {(perf_counter() - t1) / 60:.1f} min')
ci = np.percentile(D, [2.5, 97.5], axis=0).T
pv = np.array([2 * min((D[:, i] > 0).mean(), (D[:, i] < 0).mean()) for i in range(len(PRS))])
print('\nrefit mouse bootstrap (Δ = expert − naive |cos|):')
for i, (a, b) in enumerate(PRS):
    print(f'  {SH[MARGS[a]]:>6}-{SH[MARGS[b]]:<6} Δ {REF["Expert"][i] - REF["Naive"][i]:+.3f}  boot mean {D[:, i].mean():+.3f}  '
          f'CI [{ci[i, 0]:+.3f}, {ci[i, 1]:+.3f}]  p = {pv[i]:.3f}   (naive {CN[:, i].mean():.3f} [{np.percentile(CN[:, i], 2.5):.3f}, '
          f'{np.percentile(CN[:, i], 97.5):.3f}], expert {CE[:, i].mean():.3f} [{np.percentile(CE[:, i], 2.5):.3f}, {np.percentile(CE[:, i], 97.5):.3f}])')

RES = 'figures/pseudo/dimensionality/results.pkl'
r = pickle.load(open(RES, 'rb')) if os.path.exists(RES) else {}
r['DPCA_REFIT_BOOT'] = dict(pairs=PRS, margs=MARGS, ref=REF, delta=D, cos_naive=CN, cos_expert=CE, ci=ci, p=pv,
                            picks=np.array(PICKS), B=args.B)
pickle.dump(r, open(RES, 'wb'))
print('\nmerged DPCA_REFIT_BOOT into', RES, f'({(perf_counter() - t0) / 60:.1f} min total)')
