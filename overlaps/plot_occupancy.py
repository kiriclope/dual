"""
Option 3 — 2D occupancy heatmap: where delay-period states cluster in the
(sample code, choice code) plane, Naive vs Expert comparison.

Unit: per-(mouse, odor_pair) mean trajectory at each BINS_DELAY bin.
Layout: 2 rows (Naive / Expert) × 3 cols (DPA / DualGo / DualNoGo).
One figure per train epoch.
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.ndimage import gaussian_filter

from src.common.options import set_options
from src.pca.io import pkl_load

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
FIG_BASE = './figures/overlaps/occupancy'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

N_BINS   = 30
SIGMA    = 0.8
AXIS_LIM = (-4, 4)
CMAP     = matplotlib.colormaps['viridis'].copy()
CMAP.set_under('white')

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
BINS_LATE  = np.arange(27, 54)   # last ~40 % of delay

X_single = pkl_load(f'X_{DUM}', path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_correct = (y_single.laser == 0)


SAMPLE_SPLITS = [
    ('A', [0, 1], '#332288'),
    ('B', [2, 3], '#44AA99'),
]


def collect_delay_points(X_ep, cond, stage):
    """Per-(mouse, odor_pair) mean trajectory at each BINS_DELAY bin.

    Returns pooled (x, y) and per-sample dict {'A': (x,y), 'B': (x,y)}.
    """
    pts_by_sample = {label: ([], []) for label, _, _ in SAMPLE_SPLITS}
    for label, pairs, _ in SAMPLE_SPLITS:
        for mouse in ALL_MICE:
            for op in pairs:
                base = (
                    (y_single.mouse     == mouse) &
                    (y_single.tasks     == cond)  &
                    (y_single.stage     == stage) &
                    (y_single.odor_pair == op)    &
                    idx_correct
                )
                mask_s = (base & (y_single.target == 'sample')).values
                mask_c = (base & (y_single.target == 'choice')).values
                if mask_s.sum() == 0 or mask_c.sum() == 0:
                    continue
                xs = X_ep[mask_s].mean(0)[BINS_LATE]
                ys = X_ep[mask_c].mean(0)[BINS_LATE]
                pts_by_sample[label][0].append(xs)
                pts_by_sample[label][1].append(ys)
    sample_pts = {
        label: (np.concatenate(xlist), np.concatenate(ylist))
        for label, (xlist, ylist) in pts_by_sample.items()
        if xlist
    }
    all_x = np.concatenate([v[0] for v in sample_pts.values()])
    all_y = np.concatenate([v[1] for v in sample_pts.values()])
    return all_x, all_y, sample_pts


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

    # Pre-compute all histograms to get a shared vmax
    hists = {}
    for stage in STAGES:
        for cond in CONDITIONS:
            xp, yp, sample_pts = collect_delay_points(X_ep, cond, stage)
            H, xe, ye = np.histogram2d(xp, yp, bins=N_BINS,
                                       range=[AXIS_LIM, AXIS_LIM])
            H = gaussian_filter(H.astype(float), SIGMA)
            hists[(stage, cond)] = (H, xe, ye, sample_pts)

    vmax = max(H.max() for H, *_ in hists.values())

    panel_w = 2.6
    fig, axes = plt.subplots(2, 3, figsize=(3 * panel_w, 2 * panel_w),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    for ri, stage in enumerate(STAGES):
        for ci, cond in enumerate(CONDITIONS):
            ax = axes[ri, ci]
            H, xe, ye, sample_pts = hists[(stage, cond)]

            im = ax.pcolormesh(xe, ye, H.T, cmap=CMAP, vmin=1e-6, vmax=vmax,
                               shading='auto', rasterized=True)

            # Centre of mass per sample identity (A / B)
            for label, _, color in SAMPLE_SPLITS:
                if label not in sample_pts:
                    continue
                xp, yp = sample_pts[label]
                cx, cy = xp.mean(), yp.mean()
                ax.scatter(cx, cy, marker='o', s=90, color='white', lw=0,
                           zorder=10)
                ax.scatter(cx, cy, marker='o', s=60, color=color, lw=1.5,
                           edgecolors='white', zorder=11)

            ax.axhline(0, color='white', lw=0.7, alpha=0.6, zorder=5)
            ax.axvline(0, color='white', lw=0.7, alpha=0.6, zorder=5)
            ax.set_xlim(AXIS_LIM)
            ax.set_ylim(AXIS_LIM)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.tick_params(length=3, width=0.9)

            if ri == 0:
                ax.set_title(cond, pad=5)
            if ri == len(STAGES) - 1:
                ax.set_xlabel('Sample code (BL σ)')
            if ci == 0:
                ax.set_ylabel(f'{stage}\nChoice code (BL σ)')

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
                        label='Density (a.u.)')
    cbar.ax.tick_params(labelsize=8)

    stem = f'{DUM}_{train_tag}'
    fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}')

print(f'\nOccupancy → {FIG_BASE}/')
