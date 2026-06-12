"""
Flow field of delay-period dynamics in (sample code × choice code) space.

Two sets of figures:
  1. All-mice pooled  →  figures/overlaps/flowfield/
  2. Per-animal       →  figures/overlaps/flowfield/per_mouse/
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.overlaps.flowfield import compute_axis_limits, draw_flow_figure

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)
matplotlib.rcParams.update({
    'axes.titlesize': 16, 'axes.labelsize': 12,
    'xtick.labelsize': 9,  'ytick.labelsize': 9,
    'font.size': 9,
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/flowfield'
os.makedirs(os.path.join(FIG_BASE, 'png'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'svg'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'per_mouse', 'png'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'per_mouse', 'svg'), exist_ok=True)

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']

# All-mice parameters   (9 mice × 4 odor_pairs × 35 bins = 1260 pts)
N_BINS_ALL   = 12
SIGMA_ALL    = 2.0
DENSITY_ALL  = 0.04

# Per-mouse parameters  (4 odor_pairs × 35 bins = 140 pts)
N_BINS_MOUSE  = 5
SIGMA_MOUSE   = 2.5
DENSITY_MOUSE = 0.03

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

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
]

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_laser = (y_single.laser == 0)
xtime     = np.linspace(0, 14, X_single.shape[-1])

# ── Main loop ─────────────────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    print(f'\n=== {train_tag} ===')

    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd

    xlim, ylim = compute_axis_limits(
        X_ep, y_single, ALL_MICE, BINS_DELAY, idx_laser=idx_laser)

    # All-mice figure
    fig, speed_vmax = draw_flow_figure(
        X_ep, y_single, ALL_MICE, xlim, ylim, train_tag, BINS_DELAY,
        N_BINS_ALL, SIGMA_ALL, DENSITY_ALL,
        title_prefix='All mice  —  ',
        idx_laser=idx_laser, xtime=xtime,
    )
    stem = f'{DUM}_{train_tag}'
    fig.savefig(os.path.join(FIG_BASE, 'png', f'{stem}.png'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(FIG_BASE, 'svg', f'{stem}.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}.png/.svg')

    # Per-mouse figures
    for mouse in ALL_MICE:
        fig, _ = draw_flow_figure(
            X_ep, y_single, [mouse], xlim, ylim, train_tag, BINS_DELAY,
            N_BINS_MOUSE, SIGMA_MOUSE, DENSITY_MOUSE,
            title_prefix=f'{mouse}  —  ',
            speed_vmax=speed_vmax,
            idx_laser=idx_laser, xtime=xtime,
        )
        stem = f'{DUM}_{train_tag}_{mouse}'
        fig.savefig(os.path.join(FIG_BASE, 'per_mouse', 'png', f'{stem}.png'),
                    bbox_inches='tight', dpi=150)
        fig.savefig(os.path.join(FIG_BASE, 'per_mouse', 'svg', f'{stem}.svg'),
                    bbox_inches='tight')
        plt.close(fig)
    print(f'  saved per_mouse/{train_tag}_*.png')

print(f'\nFlow field → {FIG_BASE}/')
