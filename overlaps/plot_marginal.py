"""
Option 1 — 1D marginal: choice-code (and sample-code) vs time during delay,
Naive vs Expert overlaid per condition (DPA / DualGo / DualNoGo).

Layout: 2 rows (sample code / choice code) × 3 cols (conditions).
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
from src.plot.traj import plot_mean_sem

sns.set_style("ticks")

matplotlib.rcParams.update({
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
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
FIG_BASE = './figures/overlaps/marginal'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

STAGE_COLORS = {'Naive': '#9ecae1', 'Expert': '#2171b5'}
STAGE_LW     = {'Naive': 1.6,       'Expert': 1.8}
BAND_ALPHA   = 0.22

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_BL    = options['bins_BL']
BINS_DELAY = options['bins_DELAY']

X_single = pkl_load(f'X_{DUM}', path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_correct = (y_single.laser == 0)

xtime       = np.linspace(0, 14, X_single.shape[-1])
xtime_delay = xtime[BINS_DELAY]

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
]

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

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.0), sharex=True,
                             constrained_layout=True)

    for ci, cond in enumerate(CONDITIONS):
        for ri, target in enumerate(['sample', 'choice']):
            ax = axes[ri, ci]

            for stage in STAGES:
                color = STAGE_COLORS[stage]
                trajs = []
                for mouse in ALL_MICE:
                    mask = (
                        (y_single.mouse  == mouse) &
                        (y_single.tasks  == cond)  &
                        (y_single.stage  == stage) &
                        (y_single.target == target) &
                        idx_correct
                    ).values
                    if mask.sum() == 0:
                        continue
                    trajs.append(X_ep[mask].mean(0))

                if not trajs:
                    continue
                arr = np.stack(trajs, 0)                          # (n_mice, 84)
                mu  = arr.mean(0)[BINS_DELAY]
                sem = arr.std(0, ddof=1)[BINS_DELAY] / np.sqrt(arr.shape[0])
                plot_mean_sem(ax, xtime_delay, mu, sem, color,
                              alpha=BAND_ALPHA, lw=STAGE_LW[stage], label=stage)

            ax.axhline(0, color='0.75', lw=0.6, zorder=0)
            ax.tick_params(length=3, width=0.9)
            if ri == 0:
                ax.set_title(cond, pad=5)
            if ri == 1:
                ax.set_xlabel('Time (s)')
            if ci == 0:
                ax.set_ylabel(f'{target} code (BL σ)')

    handles = [Line2D([0], [0], color=STAGE_COLORS[s], lw=2.0, label=s)
               for s in STAGES]
    axes[0, -1].legend(handles=handles, frameon=False, loc='upper right',
                       labelspacing=0.3, handletextpad=0.5)

    stem = f'{DUM}_{train_tag}'
    fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}')

print(f'\nMarginal → {FIG_BASE}/')
