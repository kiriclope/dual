"""
fig_overlaps_depth_sdt.py — DIAGNOSTIC: DPA choice-code depth (the "no-lick well" read in
fig_overlaps_main_native.py) broken out by trial OUTCOME TYPE, stage-resolved.

Motivation: plain correct-vs-error (fig_overlaps_depth_correct_error.py) lumps the two DPA
error types together, but the well hypothesis makes OPPOSITE predictions for them:
  • a deep no-lick well should PREVENT false alarms   (nonpaired trial, animal licks)
  • a deep no-lick well could CAUSE misses            (paired trial, animal withholds)
So the clean test is WITHIN each trial type. DPA pairing → signal-detection cells:
  paired    (match,   lick rewarded):  hit (lick, correct)  | miss (no-lick, error)
  nonpaired (nonmatch, withhold reward): corr-rej (no-lick, correct) | false-alarm (lick, error)
These are exactly the labels in y.response {correct_hit, incorrect_miss, correct_rej, incorrect_fa}.

Depth = same read as the main figure: X_bl[:, BINS_LATE].mean over the late-delay window
(bins 27–53, pre-test); per-mouse BL-normalised. NEGATIVE depth = no-lick push (deep well),
POSITIVE = toward lick. DPA trials only, laser off.

Stat per within-type contrast = per-mouse paired-t (unit = mouse; a mouse enters a contrast
only with ≥ MIN_TR trials in BOTH cells). NOT pooled across Naive/Expert: the no-lick well
deepens strongly with learning, and the error types are unevenly distributed across stages,
so pooling is Simpson's-paradox-prone (false alarms are Naive-heavy, correct-rej Expert-heavy)
— pooling manufactures a stage-composition effect that vanishes once stage is controlled.

Default training axis = trainDELAY (bins 18–53), where the correct-vs-error signal is strongest;
override with --train {ldtest,ldtest05,ld,delay,test}.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_depth_sdt.py [--train=delay]
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

# ── Style (plot_utils forces poster context; reset it) ──────────────────────────
sns.set_context('notebook')
sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.9, 'lines.linewidth': 1.8,
})

# ── Config ──────────────────────────────────────────────────────────────────────
DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES   = ['Naive', 'Expert']
_pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal_mice[i] for i, m in enumerate(ALL_MICE)}
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
BINS_LATE = np.arange(27, 54)                                          # depth readout, bins 27–53


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

# ── Load, slice, normalise, per-trial depth ─────────────────────────────────────
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
resp = y.response.values


def cell(mouse, stage, r):
    m = (base & (y.mouse == mouse).values & (y.stage == stage).values & (resp == r))
    return depth[m]


# ── Categories (paired group | nonpaired group) ─────────────────────────────────
#  key, label, x-pos, is_correct
CATS = [('correct_hit',    'hit',        0.0, True),
        ('incorrect_miss', 'miss',       1.0, False),
        ('correct_rej',    'corr-rej',   2.6, True),
        ('incorrect_fa',   'false-alarm', 3.6, False)]
PAIRS = [('correct_hit', 'incorrect_miss', 'paired: hit vs miss', False),
         ('correct_rej', 'incorrect_fa',   'nonpaired: corr-rej vs false-alarm', True)]  # last = story test


def paired_t(stage, a, b):
    va, vb, n = [], [], 0
    for mo in ALL_MICE:
        da, db = cell(mo, stage, a), cell(mo, stage, b)
        if len(da) >= MIN_TR and len(db) >= MIN_TR:
            va.append(da.mean()); vb.append(db.mean()); n += 1
    va, vb = np.array(va), np.array(vb)
    p = float(ttest_rel(va, vb).pvalue) if n >= 3 else np.nan
    d = float((va - vb).mean()) if n else np.nan
    return d, p, n


# ── Figure ──────────────────────────────────────────────────────────────────────
np.random.seed(0)                                                      # deterministic x-jitter
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=False)
for ax, stage in zip(axes, STAGES):
    ax.axvspan(2.25, 3.95, color='#FFF3D6', zorder=0)                  # highlight nonpaired (story test)
    for key, lab, xpos, is_cor in CATS:
        vals = []
        for mo in ALL_MICE:
            d = cell(mo, stage, key)
            if len(d) >= MIN_TR:
                vals.append(d.mean())
                face = MOUSE_COLOR[mo] if is_cor else 'w'
                ax.scatter(xpos + np.random.uniform(-0.05, 0.05), d.mean(), s=48,
                           facecolors=face, edgecolors=MOUSE_COLOR[mo], linewidths=1.1, zorder=3)
        vals = np.array(vals)
        if len(vals):
            mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.plot([xpos - 0.22, xpos + 0.22], [mu, mu], color='k', lw=2.4, zorder=4)
            ax.errorbar(xpos, mu, yerr=se, color='k', capsize=4, lw=1.6, zorder=4)
    # per-mouse connectors within each pairing group
    for a, b, _, _ in PAIRS:
        xa = dict((k, x) for k, _, x, _ in CATS)[a]; xb = dict((k, x) for k, _, x, _ in CATS)[b]
        for mo in ALL_MICE:
            da, db = cell(mo, stage, a), cell(mo, stage, b)
            if len(da) >= MIN_TR and len(db) >= MIN_TR:
                ax.plot([xa, xb], [da.mean(), db.mean()], '-', color=MOUSE_COLOR[mo],
                        lw=0.8, alpha=0.5, zorder=2)
    ax.axhline(0, ls=':', color='0.6', lw=0.8)
    ax.set_xticks([c[2] for c in CATS]); ax.set_xticklabels([c[1] for c in CATS], fontsize=9)
    ax.set_xlim(-0.5, 4.1)
    ax.set_title(stage, fontweight='bold', fontsize=12)
    if ax is axes[0]:
        ax.set_ylabel('DPA choice-code depth  (late-delay, bins 27–53)\n← deeper no-lick well      toward lick →')
    # stats
    txt = []
    for a, b, lab, is_story in PAIRS:
        d, p, n = paired_t(stage, a, b)
        star = ' *' if (p == p and p < 0.05) else ''
        mark = '★ ' if is_story else '  '
        txt.append(f'{mark}{lab}\n    Δ={d:+.2f}  paired-t p={p:.3f}{star}  (n={n})')
        print(f'{stage:7s} {lab:38s} Δ={d:+.3f} p={p:.3f} n={n}')
    ax.text(0.02, 0.02, '\n'.join(txt), transform=ax.transAxes, ha='left', va='bottom',
            fontsize=7.6, color='0.2')

# shared y-limits for comparability
lo = min(ax.get_ylim()[0] for ax in axes); hi = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(lo, hi)

leg = [Line2D([0], [0], marker='o', color='0.4', mfc='0.4', ls='none', ms=7, label='correct (filled)'),
       Line2D([0], [0], marker='o', color='0.4', mfc='w', ls='none', ms=7, label='error (open)')]
axes[1].legend(handles=leg, frameon=False, fontsize=8, loc='upper right')
fig.suptitle(f'DPA choice-code depth by outcome type (signal-detection cells)  ({AXIS_LABEL})\n'
             'highlighted = nonpaired corr-rej vs false-alarm — the clean test of the well hypothesis',
             fontsize=11.5, fontweight='bold', y=1.04)
fig.tight_layout()

OUT = 'figures/overlaps/depth_error'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_depth_sdt{FILE_SUF}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
