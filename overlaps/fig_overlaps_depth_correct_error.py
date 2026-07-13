"""
fig_overlaps_depth_correct_error.py — DIAGNOSTIC: is the DPA choice-code depth
(the "no-lick push" read in fig_overlaps_main_native.py panel C) deeper on CORRECT
than on INCORRECT DPA trials?

Same depth definition as the main figure:
    raw   = X[..., TRAIN_LDTEST, :].mean(-2)[:, 1]          # choice decision fn (n, 84)
    X_bl  = raw / per-mouse BINS_BL std                      # per-mouse BL normalise
    depth = X_bl[:, BINS_LATE].mean()                        # late-delay readout, bins 27–53

Here, instead of collapsing to correct-only trials (as panel C does), we split the
DPA choice-target trials by OUTCOME (performance 1 vs 0) and compare per-mouse depth,
split Naive vs Expert and pooled across stages.

Statistical unit = mouse (paired correct−error), to avoid trial pseudoreplication.
A mouse contributes to a stage only if it has ≥ MIN_TR correct AND ≥ MIN_TR error
DPA trials in that stage. Expert errors are sparse, so the pooled panel is the
best-powered contrast.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_depth_correct_error.py [--ldtest05]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
import statsmodels.formula.api as smf
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load

# ── Style (same reset as the main figure: plot_utils forces poster context) ─────
sns.set_context('notebook')
sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.9, 'lines.linewidth': 1.8,
})

# ── Config (mirrors fig_overlaps_main_native.py) ────────────────────────────────
DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES   = ['Naive', 'Expert']
_pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal_mice[i] for i, m in enumerate(ALL_MICE)}
MIN_TR   = 3                                                            # min trials/cell for a per-mouse estimate

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL   = options['bins_BL']
BINS_LATE = np.arange(27, 54)                                          # depth readout, bins 27–53

# Decoder TRAINING axis — bins over which the generalizing decision fn is averaged.
# Select with --train {ldtest,ldtest05,ld,delay,test} (default ldtest = locked main axis).
def _arg_train():
    for a in sys.argv[1:]:
        if a.startswith('--train'):
            return a.split('=', 1)[1].lower() if '=' in a else a[len('--train'):].lstrip('=').lower()
    if '--ldtest05' in sys.argv[1:]:                                   # back-compat
        return 'ldtest05'
    return 'ldtest'

_TRAIN = _arg_train()
_TRAIN_AXES = {
    'ldtest':   (np.concatenate([options['bins_LD'], options['bins_TEST']]),                  'trainLD_TEST, bins 45–59', ''),
    'ldtest05': (np.concatenate([options['bins_LD'][-3:], options['bins_TEST'][:3]]),          'trainLDTEST05, bins 51–56', '_ldtest05'),
    'ld':       (options['bins_LD'],                                                           'trainLD, bins 45–53',      '_trainLD'),
    'delay':    (options['bins_DELAY'],                                                        'trainDELAY, bins 18–53',   '_trainDELAY'),
    'test':     (options['bins_TEST'],                                                         'trainTEST, bins 54–59',    '_trainTEST'),
}
if _TRAIN not in _TRAIN_AXES:
    raise SystemExit(f'--train must be one of {list(_TRAIN_AXES)} (got {_TRAIN!r})')
TRAIN_LDTEST, AXIS_LABEL, FILE_SUF = _TRAIN_AXES[_TRAIN]

# ── Load tensor, slice choice decision fn on the training axis, normalise ────────
print('loading main tensor …')
X = pkl_load(f'X_{DUM}',      path=DATA_IN)
y = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'  X {X.shape}  y {y.shape}')

raw = X[..., TRAIN_LDTEST, :].mean(-2)[:, 1].astype(float)             # (n, 84)
del X

X_bl = raw.copy()
for mo in ALL_MICE:
    mm = (y.mouse == mo).values
    sd = X_bl[mm][:, BINS_BL].std()
    if sd > 0:
        X_bl[mm] /= sd

depth = X_bl[:, BINS_LATE].mean(1)                                     # (n,) per-trial depth

# ── Per-mouse correct/error DPA choice-code depth, per stage ────────────────────
base_dpa_choice = ((y.laser == 0) & (y.tasks == 'DPA') & (y.target == 'choice')).values
outcome = y.performance.values                                        # DPA: performance == correctness

def cell(mouse, stage, ok):
    m = (base_dpa_choice & (y.mouse == mouse).values &
         (y.stage == stage).values & (outcome == ok))
    return depth[m]

# rows: (mouse, stage) -> {'cor': mean, 'err': mean, 'n_cor', 'n_err'}
rows = {}
for stage in STAGES + ['Pooled']:
    for mouse in ALL_MICE:
        if stage == 'Pooled':
            dc = np.concatenate([cell(mouse, s, 1) for s in STAGES])
            de = np.concatenate([cell(mouse, s, 0) for s in STAGES])
        else:
            dc, de = cell(mouse, stage, 1), cell(mouse, stage, 0)
        rows[(mouse, stage)] = dict(cor=dc.mean() if len(dc) else np.nan,
                                    err=de.mean() if len(de) else np.nan,
                                    n_cor=len(dc), n_err=len(de))


# ── trial-level LMM per stage: depth ~ correct, mouse random intercept AND slope ─
# The random-slope p is the pseudoreplication-robust test (the random-intercept-only
# p is inflated — same lesson as the opto d′ panels); we report the random-slope one.
def trial_lmm(stage):
    m = base_dpa_choice.copy()
    if stage != 'Pooled':
        m = m & (y.stage == stage).values
    d = pd.DataFrame(dict(depth=depth[m], mouse=y.mouse.values[m],
                          correct=outcome[m].astype(int)))
    out = {}
    for tag, re in (('int', None), ('slope', '~C(correct)')):
        try:
            fit = smf.mixedlm('depth ~ C(correct)', d, groups=d['mouse'],
                              re_formula=re).fit()
            out[tag] = (float(fit.params.get('C(correct)[T.1]', np.nan)),
                        float(fit.pvalues.get('C(correct)[T.1]', np.nan)))
        except Exception:
            out[tag] = (np.nan, np.nan)
    return out


lmm = {s: trial_lmm(s) for s in ['Naive', 'Expert', 'Pooled']}

# ── Plot: 3 panels (Naive | Expert | Pooled), paired per-mouse correct vs error ─
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
XPOS = {'cor': 0, 'err': 1}
summary = {}
for ax, stage in zip(axes, ['Naive', 'Expert', 'Pooled']):
    cor, err, used = [], [], []
    for mouse in ALL_MICE:
        r = rows[(mouse, stage)]
        if r['n_cor'] >= MIN_TR and r['n_err'] >= MIN_TR:
            cor.append(r['cor']); err.append(r['err']); used.append(mouse)
            ax.plot([0, 1], [r['cor'], r['err']], '-', color=MOUSE_COLOR[mouse],
                    lw=0.9, alpha=0.6, zorder=2)
            ax.scatter([0, 1], [r['cor'], r['err']], s=55, facecolors=MOUSE_COLOR[mouse],
                       edgecolors='k', linewidths=0.6, zorder=3)
    cor, err = np.array(cor), np.array(err)
    n = len(cor)
    # group mean ± sem
    for xk, vals in (('cor', cor), ('err', err)):
        if n:
            mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
            ax.plot([XPOS[xk] - 0.18, XPOS[xk] + 0.18], [mu, mu], color='k', lw=2.4, zorder=4)
            ax.errorbar(XPOS[xk], mu, yerr=se, color='k', capsize=4, lw=1.6, zorder=4)
    # paired stats over mice
    if n >= 3:
        tp = float(ttest_rel(cor, err).pvalue)
        try:
            wp = float(wilcoxon(cor, err).pvalue)
        except ValueError:
            wp = np.nan
        d_mean = float((cor - err).mean())
    else:
        tp = wp = d_mean = np.nan
    b_sl, p_sl = lmm[stage]['slope']                                   # robust (random-slope) test
    b_in, p_in = lmm[stage]['int']                                     # inflated (random-intercept)
    summary[stage] = dict(n=n, d=d_mean, tp=tp, wp=wp, lmm_slope=(b_sl, p_sl), lmm_int=(b_in, p_in))
    star = '*' if (p_sl == p_sl and p_sl < 0.05) else 'n.s.'           # star on the ROBUST test
    ax.set_title(f'{stage}  (n={n} mice)\n'
                 f'Δ(cor−err)={d_mean:+.2f}  paired-t p={tp:.3f}\n'
                 f'trial LMM (rand-slope) β={b_sl:+.2f} p={p_sl:.3f} {star}', fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['correct', 'incorrect'])
    ax.set_xlim(-0.5, 1.5)
    ax.axhline(0, ls=':', color='0.6', lw=0.8)
    if ax is axes[0]:
        ax.set_ylabel('DPA choice-code depth\n(late-delay, bins 27–53)')
    print(f'{stage:7s} n={n}  Δ(cor−err)={d_mean:+.3f}  paired-t p={tp:.3f}  wilcoxon p={wp:.3f}  '
          f'| trial LMM rand-int β={b_in:+.3f} p={p_in:.3f}  rand-slope β={b_sl:+.3f} p={p_sl:.3f}')

fig.suptitle(f'DPA choice-code depth: correct vs incorrect trials  ({AXIS_LABEL})',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()

OUT = 'figures/overlaps/depth_error'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_depth_correct_error{FILE_SUF}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
