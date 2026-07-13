"""
fig_overlaps_depth_sdt_sample.py — DIAGNOSTIC: within-odor-pair correct-vs-error DPA
choice-code depth, SAMPLE-DISCRIMINATED.

Controls the sample confound in fig_overlaps_depth_sdt.py: the choice-code depth carries a
sample-dependent bias (sample A pushes deeper than B; the reason the main figure keeps A/B
separate), and the SDT error cells are not A/B-balanced. The clean comparison is therefore
CORRECT vs ERROR *within a single odor pair* — same sample, same test, same pairing, only the
outcome differs. The four DPA pairs split by pairing × sample:

  nonpaired (withhold rewarded):  AD = sample A · corr-rej vs false-alarm
                                  BC = sample B · corr-rej vs false-alarm
  paired    (lick rewarded):      AC = sample A · hit vs miss
                                  BD = sample B · hit vs miss

Depth = X_bl[:, BINS_LATE].mean over the late-delay window (bins 27–53, pre-test), per-mouse
BL-normalised. NEGATIVE = no-lick well, POSITIVE = toward lick. DPA, laser off. Unit = mouse
(paired-t; a mouse enters a pair only with ≥ MIN_TR trials in BOTH outcome cells). NOT pooled
across stages (the well deepens with learning → Simpson's-paradox-prone).

Default training axis = trainDELAY (bins 18–53); override with --train {ldtest,ld,delay,test}.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_depth_sdt_sample.py [--train=delay]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load

sns.set_context('notebook')
sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.9, 'lines.linewidth': 1.8,
})

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES   = ['Naive', 'Expert']
_pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal_mice[i] for i, m in enumerate(ALL_MICE)}
SAMPLE_COLOR = {'A': '#332288', 'B': '#44AA99'}
MIN_TR   = 3

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
BINS_LATE = np.arange(27, 54)


def _arg_train():
    for a in sys.argv[1:]:
        if a.startswith('--train'):
            return a.split('=', 1)[1].lower() if '=' in a else a[len('--train'):].lstrip('=').lower()
    return 'delay'


_TRAIN = _arg_train()
_TRAIN_AXES = {
    'ldtest':   (np.concatenate([options['bins_LD'], options['bins_TEST']]),          'trainLD_TEST, bins 45–59', '_ldtest'),
    'ldtest05': (np.concatenate([options['bins_LD'][-3:], options['bins_TEST'][:3]]),  'trainLDTEST05, bins 51–56', '_ldtest05'),
    'ld':       (options['bins_LD'],                                                   'trainLD, bins 45–53',      '_trainLD'),
    'delay':    (options['bins_DELAY'],                                                'trainDELAY, bins 18–53',   '_trainDELAY'),
    'test':     (options['bins_TEST'],                                                 'trainTEST, bins 54–59',    '_trainTEST'),
}
if _TRAIN not in _TRAIN_AXES:
    raise SystemExit(f'--train must be one of {list(_TRAIN_AXES)} (got {_TRAIN!r})')
TRAIN, AXIS_LABEL, FILE_SUF = _TRAIN_AXES[_TRAIN]

print('loading main tensor …')
X = pkl_load(f'X_{DUM}',      path=DATA_IN)
y = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'  X {X.shape}  y {y.shape}')
raw = X[..., TRAIN, :].mean(-2)[:, 1].astype(float)
del X
X_bl = raw.copy()
for mo in ALL_MICE:
    mm = (y.mouse == mo).values
    sd = X_bl[mm][:, BINS_BL].std()
    if sd > 0:
        X_bl[mm] /= sd
depth = X_bl[:, BINS_LATE].mean(1)

base = ((y.laser == 0) & (y.tasks == 'DPA') & (y.target == 'choice')).values
op   = y.odor_pair.values
resp = y.response.values


def cell(mouse, stage, odor_pair, r):
    m = (base & (y.mouse == mouse).values & (y.stage == stage).values &
         (op == odor_pair) & (resp == r))
    return depth[m]


# within-pair contrasts: (label, odor_pair, correct_key, error_key, sample, err_label)
CONTRASTS = [
    ('AD', 1, 'correct_rej',  'incorrect_fa',   'A', 'false-alarm'),   # nonpaired, sample A
    ('BC', 3, 'correct_rej',  'incorrect_fa',   'B', 'false-alarm'),   # nonpaired, sample B
    ('AC', 0, 'correct_hit',  'incorrect_miss', 'A', 'miss'),          # paired,    sample A
    ('BD', 2, 'correct_hit',  'incorrect_miss', 'B', 'miss'),          # paired,    sample B
]


def paired_t(stage, odor_pair, ck, ek):
    va, vb, n = [], [], 0
    for mo in ALL_MICE:
        a, b = cell(mo, stage, odor_pair, ck), cell(mo, stage, odor_pair, ek)
        if len(a) >= MIN_TR and len(b) >= MIN_TR:
            va.append(a.mean()); vb.append(b.mean()); n += 1
    va, vb = np.array(va), np.array(vb)
    p = float(ttest_rel(va, vb).pvalue) if n >= 3 else np.nan
    d = float((va - vb).mean()) if n else np.nan
    return d, p, n


# x layout: 4 pairs, each a correct→error couplet; nonpaired group | paired group
GX = {'AD': (0.0, 0.85), 'BC': (1.9, 2.75), 'AC': (4.3, 5.15), 'BD': (6.2, 7.05)}

np.random.seed(0)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, stage in zip(axes, STAGES):
    ax.axvspan(-0.5, 3.4, color='#FFF3D6', zorder=0)                   # nonpaired group shade
    for lab, odor_pair, ck, ek, samp, errlab in CONTRASTS:
        xc, xe = GX[lab]
        col = SAMPLE_COLOR[samp]
        for mo in ALL_MICE:
            a, b = cell(mo, stage, odor_pair, ck), cell(mo, stage, odor_pair, ek)
            if len(a) >= MIN_TR and len(b) >= MIN_TR:
                ya, yb = a.mean(), b.mean()
                ax.plot([xc, xe], [ya, yb], '-', color=col, lw=0.8, alpha=0.45, zorder=2)
                ax.scatter(xc, ya, s=46, facecolors=col, edgecolors='k', linewidths=0.5, zorder=3)
                ax.scatter(xe, yb, s=46, facecolors='w', edgecolors=col, linewidths=1.3, zorder=3)
        # group means
        for xx, key in ((xc, ck), (xe, ek)):
            vals = [cell(mo, stage, odor_pair, key).mean() for mo in ALL_MICE
                    if len(cell(mo, stage, odor_pair, key)) >= MIN_TR]
            if vals:
                mu = np.mean(vals); se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.plot([xx - 0.2, xx + 0.2], [mu, mu], color='k', lw=2.2, zorder=4)
                ax.errorbar(xx, mu, yerr=se, color='k', capsize=3, lw=1.4, zorder=4)
        d, p, n = paired_t(stage, odor_pair, ck, ek)
        star = ' *' if (p == p and p < 0.05) else ''
        ax.text((xc + xe) / 2, 0.99, f'{lab} ({samp})', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8.5, fontweight='bold', color=col)
        ax.text((xc + xe) / 2, 0.02, f'Δ={d:+.2f}\np={p:.3f}{star}\n(n={n})',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, color='0.2')
        print(f'{stage:7s} {lab} ({samp}) {ck}-vs-{ek}: Δ={d:+.3f} p={p:.3f} n={n}')
    ax.axhline(0, ls=':', color='0.6', lw=0.8)
    ticks, labs = [], []
    for lab, _, _, _, _, errlab in CONTRASTS:
        xc, xe = GX[lab]; ticks += [xc, xe]; labs += ['corr', 'err']
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_xlim(-0.6, 7.7)
    ax.set_title(stage, fontweight='bold', fontsize=12, pad=22)
    if ax is axes[0]:
        ax.set_ylabel('DPA choice-code depth  (late-delay, bins 27–53)\n← deeper no-lick well      toward lick →')

lo = min(ax.get_ylim()[0] for ax in axes); hi = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(lo, hi * 1.08)

leg = [Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=7, label='correct (filled)'),
       Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=7, label='error (open)'),
       Line2D([0], [0], color=SAMPLE_COLOR['A'], lw=3, label='sample A'),
       Line2D([0], [0], color=SAMPLE_COLOR['B'], lw=3, label='sample B')]
axes[1].legend(handles=leg, frameon=False, fontsize=8, loc='lower right', ncol=2, columnspacing=1.0)
fig.text(0.30, 0.95, 'nonpaired (corr-rej vs false-alarm)', ha='center', fontsize=9, style='italic', color='0.35')
fig.text(0.74, 0.95, 'paired (hit vs miss)', ha='center', fontsize=9, style='italic', color='0.35')
fig.suptitle(f'Within-odor-pair correct-vs-error depth, sample-discriminated  ({AXIS_LABEL})',
             fontsize=12, fontweight='bold', y=1.005)
fig.tight_layout(rect=(0, 0, 1, 0.94))

OUT = 'figures/overlaps/depth_error'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_depth_sdt_sample{FILE_SUF}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
