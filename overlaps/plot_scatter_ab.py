"""
Per-animal scatter of sample-A and sample-B end-delay positions
in the (sample code, choice code) plane.

Unit: per-(mouse, sample-identity) mean position at BINS_LATE,
      averaging over the two odor pairs within each identity.
Layout: 1 row × 3 cols (DPA / DualGo / DualNoGo),
        Naive (open) and Expert (filled) overlaid.
One figure per train epoch.
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load

sns.set_style("ticks")

panel_w = 3.2

matplotlib.rcParams.update({
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'axes.labelpad':     4,
    'axes.linewidth':    0.9,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'svg.fonttype':      'none',
})

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/scatter_ab'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

SAMPLE_SPLITS = [
    ('A', [0, 1], '#332288'),
    ('B', [2, 3], '#44AA99'),
]

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_BL   = options['bins_BL']
BINS_LATE = np.arange(27, 54)

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
]

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_all = (y_single.laser == 0)

for train_tag, bins_train in TRAIN_EPOCHS:
    fig_dir = os.path.join(FIG_BASE, train_tag)
    os.makedirs(os.path.join(fig_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(fig_dir, 'svg'), exist_ok=True)
    print(f'\n=== {train_tag} ===')

    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd

    fig, axes = plt.subplots(1, 3, figsize=(3 * panel_w, panel_w),
                             sharex=True, sharey=True, constrained_layout=True)

    for ax, cond in zip(axes, CONDITIONS):
        # pts[stage][label] = {mouse: (x, y)}
        pts = {s: {lab: {} for lab, _, _ in SAMPLE_SPLITS} for s in STAGES}

        for stage in STAGES:
            for label, pairs, _ in SAMPLE_SPLITS:
                for mouse in ALL_MICE:
                    mask_s = (
                        (y_single.mouse  == mouse) &
                        (y_single.tasks  == cond)  &
                        (y_single.stage  == stage) &
                        (y_single.target == 'sample') &
                        y_single.odor_pair.isin(pairs) &
                        idx_all
                    ).values
                    mask_c = (
                        (y_single.mouse  == mouse) &
                        (y_single.tasks  == cond)  &
                        (y_single.stage  == stage) &
                        (y_single.target == 'choice') &
                        y_single.odor_pair.isin(pairs) &
                        idx_all
                    ).values
                    if mask_s.sum() == 0 or mask_c.sum() == 0:
                        continue
                    x = X_ep[mask_s].mean(0)[BINS_LATE].mean()
                    y = X_ep[mask_c].mean(0)[BINS_LATE].mean()
                    pts[stage][label][mouse] = (x, y)

        # Naive → Expert connecting lines per animal
        for label, _, color in SAMPLE_SPLITS:
            common = set(pts['Naive'][label]) & set(pts['Expert'][label])
            for mouse in common:
                xn, yn = pts['Naive'][label][mouse]
                xe, ye = pts['Expert'][label][mouse]
                ax.plot([xn, xe], [yn, ye], color=color,
                        lw=0.8, alpha=0.35, zorder=1)

        # Individual animal dots (Naive = open, Expert = filled)
        for label, _, color in SAMPLE_SPLITS:
            for stage, mfc, zo in [('Naive', 'white', 3), ('Expert', color, 4)]:
                data = pts[stage][label]
                if not data:
                    continue
                xs, ys = zip(*data.values())
                ax.scatter(xs, ys, color=color, facecolors=mfc,
                           s=52, lw=1.3, zorder=zo)

        # Centre of mass per sample identity (larger, black edge)
        for label, _, color in SAMPLE_SPLITS:
            for stage, mfc, zo in [('Naive', 'white', 6), ('Expert', color, 7)]:
                data = pts[stage][label]
                if not data:
                    continue
                xs, ys = zip(*data.values())
                ax.scatter(np.mean(xs), np.mean(ys),
                           color=color, facecolors=mfc,
                           s=200, lw=2.0, edgecolors='black', zorder=zo)

        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(length=3, width=0.9)
        ax.set_xlabel('Sample code (BL σ)')
        ax.set_title(cond, pad=6)

    axes[0].set_ylabel('Choice code (BL σ)')

    handles = [
        Line2D([0], [0], marker='o', color='w', mfc='#332288', mec='#332288',
               ms=7, label='Sample A'),
        Line2D([0], [0], marker='o', color='w', mfc='#44AA99', mec='#44AA99',
               ms=7, label='Sample B'),
        Line2D([0], [0], marker='o', color='w', mfc='white', mec='0.4',
               ms=7, lw=1.1, label='Naive'),
        Line2D([0], [0], marker='o', color='w', mfc='0.4', mec='0.4',
               ms=7, label='Expert'),
    ]
    axes[-1].legend(handles=handles, frameon=False,
                    bbox_to_anchor=(1.03, 1), loc='upper left')

    stem = f'{DUM}_{train_tag}'
    fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}')

print(f'\nScatter AB → {FIG_BASE}/')
