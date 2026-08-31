"""Cache the ORIGINAL Fig-3 panel-A traces so `fig_manifold_main.py` can plot them without loading
the 1.9 GB overlaps tensor.

Rebuilding these axes from scratch was the wrong instinct: `overlaps/main_panels.py` already exposes
the validated per-mouse CCGD projections (SAMPLE_D / LICK_D / TEST_D / GNG_D, one shared per-mouse
unit so amplitudes are comparable across codes), and re-fitting them from the cached window matrices
produced axes contaminated by the trial's global ramp (29-45% of the code size). This script simply
replays `main_panels._draw_trace_col`'s computation — per-mouse mean of the projection per class —
and stores the result.

Output: merges {'ORIG_TRACES': {(stage, code, level): (n_mice, n_bins)}, 'ORIG_XTIME', 'ORIG_SPECS'}
into results.pkl.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_traj_orig.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir('/home/leon/dual/overlaps')                       # main_panels uses relative data paths
sys.path.insert(0, '/home/leon/dual/overlaps')             # ...and is not a package module
import numpy as np
import main_panels as MP                                    # loads the ~1.9 GB overlaps tensor

TSUF = ('_pca20' if '--pca' in sys.argv[1:] else '') + \
       ('_antact' if '--antact' in sys.argv[1:] else '')   # --antact → main_panels' choice axis
                                                           #   = anticipatory+action, bins 48-62
SPECS = list(MP.VARS_A) + [MP.VAR_GNG]                       # sample, lick, test, GNG
OUT, META = {}, []
for ttl, D, YY, col, levels, labs, cols, task in SPECS:
    code = 'lick' if ttl == MP.LICK_TITLE else ttl
    META.append(dict(code=code, title=ttl, levels=list(levels), labels=list(labs),
                     colors=list(cols), task=task, column=col))
    # the canonical panel-A trace keeps its historical task filter under the plain key; sample
    # and lick ALSO get per-task-set keys (2026-08-31, for the task-split panel-A variant):
    # '@dual' = pooled DualGo+DualNoGo, '@go' / '@nogo' = the two dual tasks separately
    variants = [(task, '')]
    if code in ('sample', 'lick'):
        variants += [('Dual', '@dual'), ('DualGo', '@go'), ('DualNoGo', '@nogo')]
    for tsk_v, ksuf in variants:
        for stage in MP.STAGES:
            base = ((YY.laser == 0).to_numpy() & (YY.learning == stage).to_numpy()
                    & (YY.performance == 1).to_numpy())
            if tsk_v == 'DPA':
                base = base & (YY.tasks == 'DPA').to_numpy()
            elif tsk_v == 'Dual':
                base = base & (YY.tasks != 'DPA').to_numpy()
            else:
                base = base & (YY.tasks == tsk_v).to_numpy()
            for lv, lab in zip(levels, labs):
                per_mouse = []
                for mo in MP.ALL_MICE:
                    s = base & (YY.mouse == mo).to_numpy() & (YY[col].to_numpy() == lv)
                    if s.sum() >= 3:
                        per_mouse.append(np.nanmean(D[s], 0))
                OUT[(stage, code + ksuf, int(lv))] = np.asarray(per_mouse)
                print(f'{stage:6s} {code + ksuf:12s} {lab:8s} n={len(per_mouse)} '
                      f'range {np.asarray(per_mouse).mean(0).min():+.2f}..'
                      f'{np.asarray(per_mouse).mean(0).max():+.2f}', flush=True)

RES = '/home/leon/dual/pca/figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['ORIG_TRACES' + TSUF] = OUT; d['ORIG_XTIME'] = np.asarray(MP.xtime); d['ORIG_SPECS'] = META
pickle.dump(d, open(RES, 'wb'))
print('merged ORIG_TRACES' + TSUF + ' + ORIG_XTIME + ORIG_SPECS into', RES)
