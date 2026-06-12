"""
Plot decision-value trajectories broken down by odor pair, task, and stage.

Layout per target: rows = tasks (DPA, DualGo, DualNoGo)
                   cols = stages (Naive, Expert)
                   lines = odor pair (AC, AD, BD, BC)

Pair labelling (sample_odor × test_odor, both coded 0/1):
  odor_pair 0 → sample=0, test=0 → AC  (match)
  odor_pair 1 → sample=0, test=1 → AD  (non-match)
  odor_pair 2 → sample=1, test=1 → BD  (match)
  odor_pair 3 → sample=1, test=0 → BC  (non-match)
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
FIG_PAIRS = './figures/overlaps/pairs'
os.makedirs(FIG_PAIRS, exist_ok=True)

TRAIN_TAG = 'trainTEST'

TASKS   = ['DPA', 'DualGo', 'DualNoGo']
STAGES  = ['Naive', 'Expert']
TARGETS = ['sample', 'choice', 'test']

PAIR_LABELS = {0: 'AC', 1: 'AD', 2: 'BD', 3: 'BC'}
# match pairs (same odor): AC (0), BD (2) → solid
# non-match (different):   AD (1), BC (3) → dashed
PAIR_STYLE = {0: '-', 1: '--', 2: '-', 3: '--'}

# ── Epoch bins ────────────────────────────────────────────────────────────────

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

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_correct = (
    (y_single.laser == 0) & (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)

# Decision-value trajectory: mean over TEST-epoch train times, dfs channel
X_epoch = X_single[..., BINS_TEST, :].mean(-2)[:, 1]   # (N, T_test)
xtime   = np.linspace(0, 14, X_single.shape[-1])

# Per-mouse baseline normalisation: divide by each mouse's baseline-period std
X_epoch_norm = X_epoch.copy().astype(float)
for mouse in y_single.mouse.unique():
    m  = (y_single.mouse == mouse).values
    sd = X_epoch[m][:, BINS_BL].std()
    if sd > 0:
        X_epoch_norm[m] /= sd
X_epoch = X_epoch_norm

# Pair colours: one hue per pair, match=solid, non-match=dashed
pal = sns.color_palette('tab10')
PAIR_COLOR = {0: pal[0], 1: pal[1], 2: pal[2], 3: pal[3]}

# ── Plot ──────────────────────────────────────────────────────────────────────

for target in TARGETS:
    n_rows, n_cols = len(TASKS), len(STAGES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * width, n_rows * height),
        sharey='row', sharex=True,
    )

    for ri, task in enumerate(TASKS):
        for ci, stage in enumerate(STAGES):
            ax = axes[ri, ci]

            for pair_id, pair_name in PAIR_LABELS.items():
                mask = (
                    (y_single.tasks     == task)     &
                    (y_single.stage     == stage)    &
                    (y_single.target    == target)   &
                    (y_single.odor_pair == pair_id)  &
                    idx_correct
                )
                if mask.sum() == 0:
                    continue
                mu  = X_epoch[mask].mean(0)
                sem = X_epoch[mask].std(0) / np.sqrt(mask.sum())
                plot_mean_sem(ax, xtime, mu, sem, PAIR_COLOR[pair_id],
                              alpha=0.15, ls=PAIR_STYLE[pair_id],
                              lw=2, label=pair_name)

            ax.axhline(0, ls=':', color='k', lw=0.8)
            add_vlines(ax, if_dpa=(task == 'DPA'))
            ax.set_xlim([0, 14])
            ax.set_xticks(np.linspace(0, 14, 8)[::2])
            if ri == n_rows - 1:
                ax.set_xlabel('Time (s)')
            if ci == 0:
                ax.set_ylabel(f'{target} overlap')
            ax.set_title(f'{task} — {stage}', fontsize=16)
            if ri == 0 and ci == n_cols - 1:
                ax.legend(fontsize=11, frameon=False,
                          title='pair', title_fontsize=11)

    fig.suptitle(f'Target: {target}', fontsize=18, y=1.01)
    fig.tight_layout()
    path = f'{FIG_PAIRS}/{DUM}_{TRAIN_TAG}_{target}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')

print(f'\nPairs → {FIG_PAIRS}')
