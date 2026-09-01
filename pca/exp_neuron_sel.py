"""NEURON_SEL cache — per-neuron selectivity biplot data for Fig 2 panel G (user routing
2026-09-01: the neuron-level display goes in MAIN).

Per neuron (within its own mouse's trials, correct laser-off): d' for SAMPLE at mid-delay
(A vs B) and d' for CHOICE at the decision window (lick vs no-lick, DPA behavioural). A
factorised code shows a CROSS-shaped cloud (neurons selective for one variable or neither);
a mixed code a diagonal/diffuse blob. Model-free (condition means / pooled SD), so no decoder
regularisation can shape the cloud. Also caches the |d'| Pearson correlation across neurons
(independence stat) and a within-mouse class-label shuffle floor for |d'|.

Merges {'NEURON_SEL'+SUF}: per stage dict(ds, dc (N,), mouse_of_neuron, r_abs, null95_abs).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_neuron_sel.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)


def dprime(M, i1, i0, val):
    x1 = M[np.ix_(i1, val)]; x0 = M[np.ix_(i0, val)]
    m1, m0 = np.nanmean(x1, 0), np.nanmean(x0, 0)
    s = np.sqrt(0.5 * (np.nanvar(x1, 0) + np.nanvar(x0, 0)))
    return (m1 - m0) / np.where(s > 1e-9, s, np.nan)


print(f'== NEURON_SEL (SUF="{SUF}") ==', flush=True)
OUT = {}
rng = np.random.RandomState(900)
for stage in ['Naive', 'Expert']:
    ds = np.full(N, np.nan); dc = np.full(N, np.nan); mono = np.full(N, -1, int)
    nulls = []
    for mi, m in enumerate(MICE):
        val = VALIDIX[(m, stage)]
        base = (MOUSE == m) & (LEARN == stage) & (LAS == 0)
        sA = np.where(base & (PERF == 1) & (SAMP == 1))[0]
        sB = np.where(base & (PERF == 1) & (SAMP == 0))[0]
        lk = np.where(base & (TSK == 'DPA') & MATCH & (PERF == 1))[0]
        nl = np.where(base & (TSK == 'DPA') & ~MATCH & (PERF == 1))[0]
        if min(len(sA), len(sB), len(lk), len(nl)) < 5:
            continue
        ds[val] = dprime(AW['md'], sA, sB, val)
        dc[val] = dprime(AW['decision'], lk, nl, val)
        mono[val] = mi
        # shuffle floor: permute class labels within the mouse (one draw per mouse is enough
        # pooled over ~370 neurons/mouse x 9 mice x 20 reps)
        for _ in range(20):
            pool = np.r_[sA, sB]; pp = rng.permutation(len(pool))
            nulls.append(np.abs(dprime(AW['md'], pool[pp[:len(sA)]], pool[pp[len(sA):]], val)))
    ok = np.isfinite(ds) & np.isfinite(dc)
    r_abs = float(np.corrcoef(np.abs(ds[ok]), np.abs(dc[ok]))[0, 1])
    null95 = float(np.nanpercentile(np.concatenate(nulls), 95))
    OUT[stage] = dict(ds=ds, dc=dc, mouse_of_neuron=mono, r_abs=r_abs, null95_abs=null95,
                      n_ok=int(ok.sum()))
    frac_s = float(np.mean(np.abs(ds[ok]) > null95))
    frac_c = float(np.mean(np.abs(dc[ok]) > null95))
    frac_both = float(np.mean((np.abs(ds[ok]) > null95) & (np.abs(dc[ok]) > null95)))
    print(f'  {stage:6s} n={ok.sum()}  |d\'| corr r={r_abs:+.3f}  null95={null95:.2f}  '
          f'frac>null: sample {frac_s:.2f} choice {frac_c:.2f} both {frac_both:.3f} '
          f'(independence predicts {frac_s * frac_c:.3f})', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['NEURON_SEL' + SUF] = OUT
pickle.dump(d, open(RES, 'wb'))
print('merged NEURON_SEL' + SUF, 'into', RES)
