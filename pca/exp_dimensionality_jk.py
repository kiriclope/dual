"""Jackknife-across-mice CIs for the cvPCA participation ratio (Fig 2c error bars).

The split-level PR percentiles (exp_dimensionality_ci.py, PR_CI) quantify trial-split stability only —
they are implausibly tight (Expert delay [2.00, 2.06]) and mix estimators (averaged-spectrum point vs
per-split distribution). The defensible population error bar treats MICE as the exchangeable unit
(neurons partition disjointly by mouse): leave one mouse out (its neurons AND trials), recompute the
averaged cvPCA spectrum -> PR, jackknife SE over the 9 leave-outs, 95% CI = pr +/- 1.96*SE (lower bound
clipped at the PR floor of 1). Merge-dumps PR_JK into results.pkl.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dimensionality_jk.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvpca                       # THE cvPCA estimator — one implementation (2026-09-09)

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TASKS3 = ['DPA', 'DualGo', 'DualNoGo']
ALL12 = [(t, s, te) for t in TASKS3 for s in (0, 1) for te in (0, 1)]
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]

print('loading fits-inputs cache (no 20 GB X reload) …')
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
cvpca.bind(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO,
           VALIDIX=VALIDIX, N=N, MICE=MICE)   # the cvPCA estimator lives in cvpca.py (one copy)






def avg_pr(stage, conds, M, mice, nsplits=20):
    return cvpca.pr_of(cvpca.avg_spec(stage, conds, M, mice, nsplits=nsplits, seed=7))


SETS = {('all', 'delay'): (ALL12, AW['delay']), ('all', 'decision'): (ALL12, AW['decision']),
        ('DPA', 'delay'): (DPA4, AW['delay'])}
print('\n══ jackknife-across-mice PR CIs (leave-one-mouse-out, 20 splits each) ══')
PR_JK = {}
for (ts, wn), (conds, M) in SETS.items():
    for stage in STAGES:
        pr_full = avg_pr(stage, conds, M, MICE)
        jk = np.array([avg_pr(stage, conds, M, [m for m in MICE if m != mo]) for mo in MICE])
        n = len(MICE)
        se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
        ci = [max(1.0, pr_full - 1.96 * se), pr_full + 1.96 * se]
        PR_JK[(ts, wn, stage)] = dict(pr=pr_full, jk=jk, se=se, ci=np.array(ci))
        print(f'  {ts:4s} {wn:9s} {stage:6s}: PR={pr_full:.2f}  SE(jk)={se:.3f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]'
              f'  (leave-outs {jk.min():.2f}-{jk.max():.2f})')

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb')) if os.path.exists(RES) else {}
d['PR_JK'] = PR_JK
pickle.dump(d, open(RES, 'wb'))
print('\nmerged PR_JK into', RES)
