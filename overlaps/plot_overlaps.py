"""
Plot CCGD overlap figures from run_overlaps.py output.

Figures
-------
1. genmat_<dum>_<stage>_<target>.png
   Generalisation matrices (train × test) per task, for every stage/target combo.

2. traj_<dum>_<target>.png
   Decision-value time-courses per task, Naive vs Expert panels, per target.

3. traj_<dum>_<task>.png
   Decision-value time-courses per target, Naive vs Expert overlay, per task.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.options import set_options
from src.common.plot_utils import add_vlines, add_vdashed
from src.pca.io import pkl_load

# ── Style (mirrors notebooks/setup.py) ───────────────────────────────────────

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5 ** 0.5 - 1) / 2
width  = 6
height = width * golden_ratio          # ≈ 3.7 inches per panel row

matplotlib.rcParams.update({
    'figure.figsize':   (width, height),
    'lines.markersize': 5,
    'axes.titlesize':   24,
    'axes.labelsize':   19,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'axes.titlepad':    24,
    'axes.labelpad':    10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        14,
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM        = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN    = '../data/overlaps'
FIG_GENMAT = './figures/overlaps/genmat'
FIG_TRAJ   = './figures/overlaps/traj'
for d in (FIG_GENMAT, FIG_TRAJ):
    os.makedirs(d, exist_ok=True)

TRAIN_TAG = 'trainTEST'   # train-time slice used for trajectory plots

TASKS   = ['DPA', 'DualGo', 'DualNoGo']
STAGES  = ['Naive', 'Expert']
TARGETS = ['sample', 'choice', 'test']

# ── Epoch bins from set_options ───────────────────────────────────────────────

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL',
    avg_noise=False, unit_var_BL=False, random_state=None, T_WINDOW=0.0,
    l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_TEST = options['bins_TEST']
BINS_BL   = options['bins_BL']

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)   # (N, 2, T_train, T_test)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)   # (N, ncols)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

# Correct, non-laser trials
idx_correct = (
    (y_single.laser == 0) & (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)

# Decision-value trajectory: average over TEST-epoch train times, dfs channel
# Shape: (N, T_test)
X_epoch = X_single[..., BINS_TEST, :].mean(-2)[:, 1]
xtime   = np.linspace(0, 14, X_single.shape[-1])

# Per-mouse baseline normalisation: divide by each mouse's baseline-period std
# so all mice contribute on the same scale before averaging.
X_epoch_norm = X_epoch.copy().astype(float)
for mouse in y_single.mouse.unique():
    m  = (y_single.mouse == mouse).values
    sd = X_epoch[m][:, BINS_BL].std()
    if sd > 0:
        X_epoch_norm[m] /= sd
X_epoch = X_epoch_norm

# ── Colour scheme ─────────────────────────────────────────────────────────────

pal = sns.color_palette('muted')
c_task  = {'DPA': pal[3], 'DualGo': pal[0], 'DualNoGo': pal[2]}
c_stage = {'Naive': pal[4], 'Expert': pal[5]}

# ── Figure 1 – generalisation matrices ───────────────────────────────────────

for stage in STAGES:
    for target in TARGETS:
        fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), sharey=True)

        for ax, task in zip(axes, TASKS):
            mask = (
                (y_single.tasks  == task)   &
                (y_single.stage  == stage)  &
                (y_single.target == target) &
                (y_single.labels == 1)      &
                idx_correct
            )
            if mask.sum() == 0:
                ax.set_visible(False)
                continue

            mat = X_single[mask].mean(0)[1]          # mean over trials, dfs channel
            vmin, vmax = np.percentile(mat, [5, 95])
            im = ax.imshow(mat, interpolation=None, origin='lower',
                           cmap='jet', extent=[0, 14, 0, 14],
                           vmin=vmin, vmax=vmax)
            add_vdashed(ax, 1)
            ax.set_xlim([0, 14]); ax.set_ylim([0, 14])
            ax.set_xticks(np.linspace(0, 14, 8)[::2])
            ax.set_xlabel('Testing Time (s)')
            ax.set_ylabel('Training Time (s)')
            ax.set_title(task)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f'{stage}  —  {target}', fontsize=11)
        fig.tight_layout()
        path = f'{FIG_GENMAT}/{DUM}_{stage}_{target}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f'saved {path}')

# ── Figure 2 – trajectories per target (Naive | Expert panels) ───────────────

for target in TARGETS:
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, height), sharey=True)

    for ax, stage in zip(axes, STAGES):
        for task in TASKS:
            mask = (
                (y_single.tasks  == task)   &
                (y_single.stage  == stage)  &
                (y_single.target == target) &
                (y_single.labels == 1)      &
                idx_correct
            )
            if mask.sum() == 0:
                continue
            mu  = X_epoch[mask].mean(0)
            sem = X_epoch[mask].std(0) / np.sqrt(mask.sum())
            ax.plot(xtime, mu, color=c_task[task], label=task)
            ax.fill_between(xtime, mu - sem, mu + sem,
                            color=c_task[task], alpha=0.2)

        ax.axhline(0, ls='--', color='k', lw=0.8)
        add_vlines(ax, if_dpa=0)
        ax.set_xlim([0, 14])
        ax.set_xticks(np.linspace(0, 14, 8)[::2])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{target} overlap')
        ax.set_title(stage)
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    path = f'{FIG_TRAJ}/{DUM}_{TRAIN_TAG}_{target}.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'saved {path}')

# ── Figure 3 – trajectories per task (targets × Naive vs Expert overlay) ─────

for task in TASKS:
    fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), sharey=True)

    for ax, target in zip(axes, TARGETS):
        for stage in STAGES:
            mask = (
                (y_single.tasks  == task)   &
                (y_single.stage  == stage)  &
                (y_single.target == target) &
                (y_single.labels == 1)      &
                idx_correct
            )
            if mask.sum() == 0:
                continue
            mu  = X_epoch[mask].mean(0)
            sem = X_epoch[mask].std(0) / np.sqrt(mask.sum())
            ax.plot(xtime, mu, color=c_stage[stage], label=stage)
            ax.fill_between(xtime, mu - sem, mu + sem,
                            color=c_stage[stage], alpha=0.2)

        ax.axhline(0, ls='--', color='k', lw=0.8)
        add_vlines(ax, if_dpa=(task == 'DPA'))
        ax.set_xlim([0, 14])
        ax.set_xticks(np.linspace(0, 14, 8)[::2])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{target} overlap')
        ax.set_title(target)
        ax.legend(fontsize=9, frameon=False)

    fig.suptitle(task, fontsize=11)
    fig.tight_layout()
    path = f'{FIG_TRAJ}/{DUM}_{TRAIN_TAG}_{task}.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'saved {path}')

print(f'\nGenmat → {FIG_GENMAT}\nTraj   → {FIG_TRAJ}')
