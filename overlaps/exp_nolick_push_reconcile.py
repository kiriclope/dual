"""
exp_nolick_push_reconcile.py — reconcile the disagreeing no-lick-push statistics.

The tests on the pooled (A&B) Naive->Expert deepening disagree: the paired t and
two-sided Wilcoxon say ~0.07-0.09 (n.s.), while the percentile/BCa bootstrap CIs
EXCLUDE 0. This script settles which to trust by printing, for the strongest cell
(pooled, trainDELAY, all trials), the 9 per-mouse paired diffs and every interval
side by side.

Verdict: the bootstrap CIs excluding 0 are ANTI-CONSERVATIVE at n=9 — they use the
~1.96 normal quantile + a slightly smaller resampling SE, ~21% narrower than the
honest t 95% CI [-2.14, +0.19] which INCLUDES 0. Report p~0.07-0.09 two-sided as the
honest number (directional one-sided p1~0.037); do NOT cite the bootstrap CI as
significance. Not outlier-driven (rank-based Wilcoxon still 0.074, 7/9 deeper). The
maximal LMM (p~0.062) agrees; the random-intercept LMM (p<1e-4) is pseudo-replication.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python exp_nolick_push_reconcile.py
"""

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import wilcoxon, ttest_rel, ttest_1samp, t as tdist, bootstrap

from src.common.options import set_options
from src.pca.io import pkl_load

DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0'; DATA_IN = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0, trials='', data_type='dF',
    prescreen=None, pval=0.05, preprocess=None, scaler_BL='standard_BL', avg_noise=False,
    unit_var_BL=False, random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
    scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10, class_weight=0,
    multilabel=0, mne_estimator='generalizing', n_jobs=64, days=['first', 'last'])
BINS_BL = options['bins_BL']; BINS_LATE = np.arange(27, 54); bins_delay = options['bins_DELAY']

X = pkl_load(f'X_{DUM}', path=DATA_IN); y = pkl_load(f'labels_{DUM}', path=DATA_IN)
idx_laser = (y.laser == 0)

def choice_axis(bt):
    Xe = X[..., bt, :].mean(-2)[:, 1].astype(float)
    for m in ALL_MICE:
        mk = (y.mouse == m).values; sd = Xe[mk][:, BINS_BL].std()
        if sd > 0: Xe[mk] /= sd
    return Xe
Xe = choice_axis(bins_delay)

def depth(mouse, stage, pairs):
    m = ((y.mouse == mouse) & (y.tasks == 'DPA') & (y.stage == stage) &
         (y.target == 'choice') & idx_laser & y.odor_pair.isin(pairs)).values
    return Xe[m][:, BINS_LATE].mean() if m.sum() else np.nan

nai = np.array([np.nanmean([depth(m, 'Naive', [0, 1]), depth(m, 'Naive', [2, 3])]) for m in ALL_MICE])
exp = np.array([np.nanmean([depth(m, 'Expert', [0, 1]), depth(m, 'Expert', [2, 3])]) for m in ALL_MICE])
d = exp - nai
n = len(d); mean = d.mean(); sd = d.std(ddof=1); sem = sd / np.sqrt(n)

print('POOLED A&B, trainDELAY, all trials — paired diff d = Expert - Naive, per mouse:')
for m, dd in zip(ALL_MICE, d): print(f'  {m:9s} {dd:+.3f}')
print(f'\n  n={n}  mean={mean:+.3f}  sd={sd:.3f}  sem={sem:.3f}  dz={mean/sd:+.3f}  {(d<0).sum()}/{n} deeper')

print(f'\n  paired t  p2={ttest_rel(exp, nai).pvalue:.4f}   (1-samp on d: {ttest_1samp(d, 0).pvalue:.4f})')
print(f'  Wilcoxon  p2={wilcoxon(d).pvalue:.4f}   p1={wilcoxon(d, alternative="less").pvalue:.4f}')

tcrit = tdist.ppf(0.975, df=n - 1)
print(f'\n  t 95% CI (mean +/- {tcrit:.2f}*sem):        [{mean-tcrit*sem:+.3f}, {mean+tcrit*sem:+.3f}]  <- honest, INCLUDES 0')

rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(20000, n)); bs = d[idx].mean(1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f'  percentile bootstrap 95% CI (20k):     [{lo:+.3f}, {hi:+.3f}]  bootSE={bs.std():.3f}  <- anti-conservative')
print(f'  basic/reverse-pct bootstrap 95% CI:    [{2*mean-hi:+.3f}, {2*mean-lo:+.3f}]')
res = bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=20000, method='BCa')
print(f'  BCa bootstrap 95% CI (bias-corrected): [{res.confidence_interval.low:+.3f}, {res.confidence_interval.high:+.3f}]')
print(f'\n  WHY they disagree: percentile uses ~1.96 quantile, t uses {tcrit:.3f} (fat tails, df={n-1}); '
      f'bootSE {bs.std():.3f} < sem {sem:.3f}.')
print('  => bootstrap half-width ~21% narrower than t => flips "includes 0" to "excludes 0" on identical data.')
print('  Report p~0.07-0.09 two-sided (trend); do NOT report the bootstrap CI as significance.')
