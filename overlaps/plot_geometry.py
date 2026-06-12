"""
Sample × Choice geometry: where delay-period states cluster in the
(sample code, choice code) plane, comparing Naive vs Expert.

Unit of observation: per-(mouse, odor_pair) mean position at BINS_LATE.
Layout: 1 row × 3 cols (DPA / DualGo / DualNoGo), Naive and Expert overlaid.
One figure per train epoch.
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from src.common.options import set_options
from src.pca.io import pkl_load

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_style("ticks")

panel_w = 3.0

matplotlib.rcParams.update({
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize':    13,
    'axes.titlesize':    13,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9,
    'axes.labelpad':     4,
    'axes.linewidth':    0.9,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'svg.fonttype':      'none',
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/geometry'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

SAMPLE_A_PAIRS = [0, 1]
SAMPLE_B_PAIRS = [2, 3]
SAMPLE_SPLITS = [
    ('A', SAMPLE_A_PAIRS, '#332288'),
    ('B', SAMPLE_B_PAIRS, '#44AA99'),
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
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
    ('trainTEST',   options['bins_TEST']),
]

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_correct = (y_single.laser == 0)

# ── Main loop ─────────────────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    fig_dir = os.path.join(FIG_BASE, train_tag, 'correct')
    os.makedirs(os.path.join(fig_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(fig_dir, 'svg'), exist_ok=True)
    print(f'\n=== {train_tag} ===')

    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd

    fig = plt.figure(figsize=(3 * panel_w, panel_w * 2.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], hspace=0.08)
    axes       = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_strip = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax in axes[1:]:
        ax.sharex(axes[0]); ax.sharey(axes[0])
    for ax in axes_strip[1:]:
        ax.sharey(axes_strip[0])

    rng = np.random.default_rng(42)

    for ci, (ax, ax_s, cond) in enumerate(zip(axes, axes_strip, CONDITIONS)):
        # Collect per-(mouse, odor_pair) positions for each stage
        # pts[stage][label] = list of (x, y) tuples
        pts = {s: {'A': [], 'B': []} for s in STAGES}

        for stage in STAGES:
            for label, pairs, _ in SAMPLE_SPLITS:
                for mouse in ALL_MICE:
                    for op in pairs:
                        base = (
                            (y_single.mouse    == mouse) &
                            (y_single.tasks    == cond)  &
                            (y_single.stage    == stage) &
                            (y_single.odor_pair == op)   &
                            idx_correct
                        )
                        mask_s = (base & (y_single.target == 'sample')).values
                        mask_c = (base & (y_single.target == 'choice')).values
                        if mask_s.sum() == 0 or mask_c.sum() == 0:
                            continue
                        x = X_ep[mask_s].mean(0)[BINS_LATE].mean()
                        y = X_ep[mask_c].mean(0)[BINS_LATE].mean()
                        pts[stage][label].append((x, y))

        # Draw lines connecting Naive → Expert per (mouse, odor_pair)
        for label, _, color in SAMPLE_SPLITS:
            n_pts = min(len(pts['Naive'][label]), len(pts['Expert'][label]))
            for i in range(n_pts):
                xn, yn = pts['Naive'][label][i]
                xe, ye = pts['Expert'][label][i]
                ax.plot([xn, xe], [yn, ye], color=color, lw=0.8, alpha=0.3, zorder=1)

        # Draw individual points (Naive = open, Expert = filled)
        for label, _, color in SAMPLE_SPLITS:
            for stage, mfc, zorder in [('Naive', 'white', 3), ('Expert', color, 4)]:
                if not pts[stage][label]:
                    continue
                xs, ys = zip(*pts[stage][label])
                ax.scatter(xs, ys, color=color, facecolors=mfc,
                           s=55, lw=1.4, zorder=zorder)

        # Grand mean centroids (larger markers)
        for label, _, color in SAMPLE_SPLITS:
            for stage, mfc, zorder in [('Naive', 'white', 5), ('Expert', color, 6)]:
                if not pts[stage][label]:
                    continue
                xs, ys = zip(*pts[stage][label])
                ax.scatter(np.mean(xs), np.mean(ys), color=color, facecolors=mfc,
                           s=180, lw=2.0, edgecolors='k', zorder=zorder)

        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(length=3, width=0.9)
        ax.set_xlabel('Sample code')
        ax.set_title(cond, pad=6)

        # ── Strip plot: 1D choice-code distribution in BINS_LATE ────────────
        STAGE_X = {'Naive': 0, 'Expert': 1}
        for label, _, color in SAMPLE_SPLITS:
            offset = -0.12 if label == 'A' else 0.12
            for stage in STAGES:
                y_vals = [y for _, y in pts[stage][label]]
                if not y_vals:
                    continue
                x0   = STAGE_X[stage] + offset
                jit  = rng.uniform(-0.04, 0.04, len(y_vals))
                mfc  = 'white' if stage == 'Naive' else color
                ax_s.scatter(x0 + jit, y_vals, color=color, facecolors=mfc,
                             edgecolors=color, s=22, lw=0.9, alpha=0.75, zorder=3)
                ym = np.mean(y_vals)
                ye = np.std(y_vals, ddof=1) / np.sqrt(len(y_vals)) if len(y_vals) > 1 else 0
                ax_s.plot([x0 - 0.08, x0 + 0.08], [ym, ym],
                          color=color, lw=2.2, zorder=5)
                ax_s.errorbar(x0, ym, yerr=ye, color=color, lw=1.4,
                              capsize=3, fmt='none', zorder=5)

        ax_s.axhline(0, color='0.75', lw=0.6, zorder=0)
        ax_s.set_xticks([0, 1])
        ax_s.set_xticklabels(['Naive', 'Expert'])
        ax_s.tick_params(length=3, width=0.9)
        ax_s.set_xlim(-0.5, 1.5)
        if ci == 0:
            ax_s.set_ylabel('Choice code (BL σ)')

    axes[0].set_ylabel('Choice code')

    # Legend: sample identity + stage
    handles = [
        Line2D([0], [0], marker='o', color='w', mfc='#332288', mec='#332288',
               ms=7, label='Sample A'),
        Line2D([0], [0], marker='o', color='w', mfc='#44AA99', mec='#44AA99',
               ms=7, label='Sample B'),
        Line2D([0], [0], marker='o', color='w', mfc='white',   mec='0.3',
               ms=7, lw=1.0, label='Naive'),
        Line2D([0], [0], marker='o', color='w', mfc='0.3',     mec='0.3',
               ms=7, label='Expert'),
    ]
    axes[-1].legend(handles=handles, frameon=False,
                    bbox_to_anchor=(1.03, 1), loc='upper left')

    stem = f'{DUM}_{train_tag}'
    fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}.png/.svg')

print(f'\nGeometry → {FIG_BASE}/')
