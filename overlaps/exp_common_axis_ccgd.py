"""exp_common_axis_ccgd.py — the fixed-common-axis control on CCGD held-out decision functions (ED 3c,
reinstated 2026-09-15; Leon: "make sure all the results are cross validated").

The old control projected training trials on fold-averaged weights. Here both axes come from the CCGD
pipeline itself: PER-STAGE = Fig. 4's tensor (one decoder per mouse x stage); POOLED = `run_overlaps.py
--pool-stages --scaler none --targets choice --contexts all`, one decoder per mouse fitted on the Naive+Expert
laser-off trials together (neurons registered in both stages), every trial's decision function from the fold
that held it out. Depth = decision function of the decoders trained at the decision window (bins 54–62), read
at late delay (45–53), RAW log-odds, laser-off DPA trials — as Fig. 4b,c.
  push     depth ~ stage + sample + (1|mouse), 36 obs
  coupling per-mouse Spearman Δdepth vs ΔDPA accuracy (GNG-free DPA trials), n = 9
Dumps ed3_cache['common_ccgd'] = {'perstage': {...}, 'pooled': {...}}.
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_common_axis_ccgd.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr
from src.pca.io import pkl_load

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ACT, LD = np.arange(54, 63), np.arange(45, 54)
TENSORS = {'perstage': 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
           'pooled': 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_pooled_targets_choice'}


def depth_table(dum):
    y = pkl_load(f'labels_{dum}', path='../data/overlaps'); X = np.asarray(pkl_load(f'X_{dum}', path='../data/overlaps'))
    ch = (y.target == 'choice').to_numpy(); D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True); del X
    yc['depth'] = D[:, LD].mean(1)
    d = yc[(yc.laser == 0) & (yc.tasks == 'DPA')].copy()
    d['sample'] = np.where(d.odor_pair.isin([0, 1]), 'A', 'B'); d['st'] = (d.learning == 'Expert').astype(int)
    return d


out = {}
for key, dum in TENSORS.items():
    d = depth_table(dum)
    g = d.groupby(['mouse', 'sample', 'st']).agg(depth=('depth', 'mean'), acc=('performance', 'mean'), n=('depth', 'size')).reset_index()
    fit = smf.mixedlm('depth ~ st + C(sample)', g, groups=g['mouse']).fit()
    per = g.pivot_table(index='mouse', columns='st', values=['depth', 'acc'], aggfunc='mean')
    dd = (per['depth'][1] - per['depth'][0]).reindex(MICE); da = (per['acc'][1] - per['acc'][0]).reindex(MICE)
    ok = dd.notna() & da.notna(); rho, p = spearmanr(dd[ok], da[ok])
    out[key] = dict(g=g, push=(float(fit.params['st']), float(fit.pvalues['st'])), mice=list(dd.index[ok]),
                    ddm=dd[ok].to_numpy(), dam=da[ok].to_numpy(), rho=float(rho), p=float(p))
    print(f'{key:9s} push β={fit.params["st"]:+.3f} p={fit.pvalues["st"]:.3f}   coupling ρ={rho:+.2f} p={p:.3f} (n={int(ok.sum())})  '
          f'trials {int(g.n.sum())}')
C = 'figures/overlaps/controls/ed3_cache.pkl'
c = pickle.load(open(C, 'rb')) if os.path.exists(C) else {}
c['common_ccgd'] = out; pickle.dump(c, open(C, 'wb')); print('cached ed3_cache[common_ccgd]')
