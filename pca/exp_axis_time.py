"""AXIS_TIME cache — does the frame ROTATE within the trial? cos(axis(t), axis(ref window)) for
the sample / choice / dist decoder axes, from the saved per-bin overlaps decoder weights
(run_overlaps --save-weights; NO tensor load).

Per mouse & stage: unit-normalise the per-bin weight vectors, take |cos| against the axis averaged
over each code's canonical training window (overlaps T_WINDOW=0 conventions: sample = delay bins
18-53, choice = action bins 57-62, dist = bins_MD 33-38); mean ± SEM across the 9 mice.
Merge-dumps {'AXIS_TIME'} into results.pkl:  AXIS_TIME[(code, stage)] = dict(mean, sem); + 'xtime'.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_axis_time.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.pca.io import pkl_load

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
W = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
             path='../data/overlaps')['weights']
CODES = [('sample', 'sample', np.arange(18, 54)),      # delay (overlaps T_WINDOW=0 bins)
         ('choice', 'choice', np.arange(57, 63)),      # action window 57-62 incl.
         ('dist',   'gng',    np.arange(33, 39))]      # bins_MD 33-38

AXIS_TIME = {}
for cname, tg, refwin in CODES:
    for stage in STAGES:
        traces = []
        for m in MICE:
            k = (m, stage, 'all', tg)
            if k not in W:
                continue
            wb = np.asarray(W[k], float)                             # (bins, neurons)
            n = np.linalg.norm(wb, axis=1, keepdims=True)
            wb = wb / np.where(n > 0, n, 1.0)
            ref = wb[refwin].mean(0); ref /= np.linalg.norm(ref)
            traces.append(np.abs(wb @ ref))
        T = np.stack(traces, 0)
        AXIS_TIME[(cname, stage)] = dict(mean=T.mean(0), sem=T.std(0, ddof=1) / np.sqrt(len(T)),
                                         n=len(T))
        mwin = AXIS_TIME[(cname, stage)]['mean']
        print(f'{cname:7s} {stage:6s} n={len(T)}  |cos| @ref {mwin[refwin].mean():.2f}  '
              f'delay-mean {mwin[18:54].mean():.2f}  early(0-11) {mwin[0:12].mean():.2f}', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['AXIS_TIME'] = AXIS_TIME
d['AXIS_TIME_X'] = np.linspace(0, 14, 84)
pickle.dump(d, open(RES, 'wb'))
print('merged AXIS_TIME into', RES)
