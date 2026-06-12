"""
Three-panel figure per condition (DPA, DualGo, DualNoGo) × stage (Naive, Expert):
  Panel 1 — sample code:  trajectories for odor A (label=0) vs odor B (label=1)
  Panel 2 — choice code:  trajectories for no-lick (label=0) vs lick (label=1)
  Panel 3 — test code:    trajectories for odor C (label=0) vs odor D (label=1)

Each line is the mean across mice (SEM shading).
Per-mouse baseline normalisation applied before averaging.

Run for three train-epoch slices; saved in separate subfolders:
  codes/trainDELAY/   bins_DELAY (18-53, ~3-9 s, full delay)
  codes/trainCHOICE/  bins_CHOICE (60-65, ~10-11 s, choice)
  codes/trainED/      bins_ED (18-26, ~3-4.3 s, early delay)
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.options import set_options
from src.common.plot_utils import add_vlines
from src.pca.io import pkl_load
from src.plot.traj import plot_mean_sem

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5 ** 0.5 - 1) / 2
width  = 6
height = width * golden_ratio

matplotlib.rcParams.update({
    'figure.figsize':    (width, height),
    'lines.markersize':  5,
    'axes.titlesize':    22,
    'axes.labelsize':    18,
    'xtick.labelsize':   14,
    'ytick.labelsize':   14,
    'axes.titlepad':     20,
    'axes.labelpad':     8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         13,
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM       = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN   = '../data/overlaps'
FIG_BASE  = './figures/overlaps/codes'

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

PANELS = [
    ('sample', 'odor A', 'odor B'),
    ('choice', 'no lick', 'lick'),
    ('test',   'odor C',  'odor D'),
]

# ── Epoch bins ────────────────────────────────────────────────────────────────

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_BL = options['bins_BL']

TRAIN_EPOCHS = [
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
    ('trainTEST',   options['bins_TEST']),
]

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_correct = (
    (y_single.laser == 0) & (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)

xtime = np.linspace(0, 14, X_single.shape[-1])

# ── Colours (matching figure2BC PCA trajectories) ─────────────────────────────

PANEL_COLORS = {
    'sample': ('#332288', '#44AA99'),   # Sample A, Sample B
    'choice': ('#377eb8', '#4daf4a'),   # No lick, Lick
    'test':   ('#e41a1c', '#ff7f00'),   # Test C, Test D
}

# ── Loop over train epochs ────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    fig_dir = os.path.join(FIG_BASE, train_tag, 'correct')
    os.makedirs(os.path.join(fig_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(fig_dir, 'svg'), exist_ok=True)
    print(f'\n=== {train_tag} (bins {bins_train[0]}–{bins_train[-1]}) ===')

    # Decision-value trajectory: average over selected train-time bins, dfs channel
    X_epoch = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)

    # Per-mouse baseline normalisation
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_epoch[m][:, BINS_BL].std()
        if sd > 0:
            X_epoch[m] /= sd

    for cond in CONDITIONS:
        for stage in STAGES:
            fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), sharey=False)

            for ax, (target, name0, name1) in zip(axes, PANELS):
                colors = PANEL_COLORS[target]
                for label_val, label_name, color in [
                    (0, name0, colors[0]),
                    (1, name1, colors[1]),
                ]:
                    mouse_means = []
                    for mouse in ALL_MICE:
                        mask = (
                            (y_single.mouse  == mouse)      &
                            (y_single.tasks  == cond)        &
                            (y_single.stage  == stage)       &
                            (y_single.target == target)      &
                            (y_single.labels == label_val)   &
                            idx_correct
                        ).values
                        if mask.sum() == 0:
                            continue
                        mouse_means.append(X_epoch[mask].mean(0))

                    if not mouse_means:
                        continue

                    arr = np.stack(mouse_means, axis=0)
                    mu  = arr.mean(0)
                    sem = arr.std(0) / np.sqrt(arr.shape[0])
                    plot_mean_sem(ax, xtime, mu, sem, color,
                                  lw=2.5, label=label_name)

                ax.axhline(0, ls=':', color='k', lw=0.8)
                add_vlines(ax, if_dpa=(cond == 'DPA'))
                ax.set_xlim([0, 14])
                ax.set_xticks(np.linspace(0, 14, 8)[::2])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'{target} overlap')
                ax.legend(fontsize=12, frameon=False)
            fig.tight_layout()
            stem = f'{DUM}_{train_tag}_{cond}_{stage}'
            fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
            fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
            path = os.path.join(fig_dir, 'png', f'{stem}.png')
            plt.close(fig)
            print(f'saved {path}')

print(f'\nCodes → {FIG_BASE}/')
