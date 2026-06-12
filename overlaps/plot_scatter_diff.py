"""
Scatter: Expert − Naive change in sample-code strength vs choice-code strength.

For each mouse × condition, two points:
  Dot A (circle):   Δx = sample-A strength (Expert − Naive)
                    Δy = choice strength on A-sample trials (Expert − Naive)
  Dot B (triangle): Δx = sample-B strength (Expert − Naive)
                    Δy = choice strength on B-sample trials (Expert − Naive)

Strength = mean X_epoch over LATE_DELAY bins (27–53, ~4.3–9 s).
Per-mouse baseline normalisation applied per stage before differencing.

One figure per train epoch (3 panels = DPA / DualGo / DualNoGo).
Saved in figures/overlaps/scatter_diff/{train_tag}/{DUM}_{train_tag}.png
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load

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

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/scatter_diff'

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

SAMPLE_A_PAIRS = [0, 1]
SAMPLE_B_PAIRS = [2, 3]

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
BINS_BL   = options['bins_BL']
BINS_LATE = np.arange(27, 54)   # late delay: ~4.3–9 s

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

idx_correct = (
    (y_single.laser == 0) & (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)

# ── Colours per mouse ─────────────────────────────────────────────────────────

pal_mice = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}


def _strength(X_ep, mask_s, mask_c):
    """Return (sample_strength, choice_strength) scalar pair, or (nan, nan)."""
    if mask_s.sum() == 0 or mask_c.sum() == 0:
        return np.nan, np.nan
    return (X_ep[mask_s][:, BINS_LATE].mean(),
            X_ep[mask_c][:, BINS_LATE].mean())


# ── Loop over train epochs ────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    fig_dir = os.path.join(FIG_BASE, train_tag)
    os.makedirs(os.path.join(fig_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(fig_dir, 'svg'), exist_ok=True)
    print(f'\n=== {train_tag} (train bins {bins_train[0]}–{bins_train[-1]}) ===')

    # Compute per-stage X_epoch with per-mouse BL normalisation
    X_by_stage = {}
    for stage in STAGES:
        X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
        for mouse in ALL_MICE:
            m  = (y_single.mouse == mouse).values
            sd = X_ep[m][:, BINS_BL].std()
            if sd > 0:
                X_ep[m] /= sd
        X_by_stage[stage] = X_ep

    fig, axes = plt.subplots(1, 3, figsize=(3 * width, height))

    # Collect raw deltas for all conditions up front
    cond_raw = {}
    cond_stats = {}

    for cond in CONDITIONS:
        raw = {}
        all_dx_flat, all_dy_flat = [], []
        for mouse in ALL_MICE:
            delta_x, delta_y = [], []
            for cls_label, odor_pairs in [(0, SAMPLE_A_PAIRS), (1, SAMPLE_B_PAIRS)]:
                pts = {}
                for stage in STAGES:
                    mask_s = (
                        (y_single.mouse  == mouse)     &
                        (y_single.tasks  == cond)      &
                        (y_single.stage  == stage)     &
                        (y_single.target == 'sample')  &
                        (y_single.labels == cls_label) &
                        idx_correct
                    ).values
                    mask_c = (
                        (y_single.mouse  == mouse)              &
                        (y_single.tasks  == cond)               &
                        (y_single.stage  == stage)              &
                        (y_single.target == 'choice')           &
                        y_single.odor_pair.isin(odor_pairs)     &
                        idx_correct
                    ).values
                    pts[stage] = _strength(X_by_stage[stage], mask_s, mask_c)
                xs_n, yc_n = pts['Naive']
                xs_e, yc_e = pts['Expert']
                delta_x.append(xs_e - xs_n)
                delta_y.append(yc_e - yc_n)
            raw[mouse] = (delta_x, delta_y)
            all_dx_flat.extend(delta_x)
            all_dy_flat.extend(delta_y)
        cond_raw[cond] = raw
        ax_arr = np.array(all_dx_flat, dtype=float)
        ay_arr = np.array(all_dy_flat, dtype=float)
        xm, xs = np.nanmean(ax_arr), np.nanstd(ax_arr)
        ym, ys = np.nanmean(ay_arr), np.nanstd(ay_arr)
        cond_stats[cond] = (xm, xs or 1.0, ym, ys or 1.0)

    def _draw(axes_list, use_zscore):
        for ax, cond in zip(axes_list, CONDITIONS):
            xm, xs, ym, ys = cond_stats[cond]
            for mouse in ALL_MICE:
                color = MOUSE_COLOR[mouse]
                delta_x, delta_y = cond_raw[cond][mouse]
                if use_zscore:
                    px = [(v - xm) / xs for v in delta_x]
                    py = delta_y
                    xlabel = 'Δ sample code (z-score)'
                    ylabel = 'Δ choice code (BL σ)'
                else:
                    px, py = delta_x, delta_y
                    xlabel = 'Δ sample code (BL σ)'
                    ylabel = 'Δ choice code (BL σ)'
                ax.plot(px, py, '-', color=color, lw=0.8, alpha=0.5, zorder=3)
                for xv, yv, marker in zip(px, py, ['o', '^']):
                    if not (np.isnan(xv) or np.isnan(yv)):
                        ax.scatter(xv, yv, color=color, marker=marker,
                                   s=70, zorder=5, linewidths=0.5, edgecolors='w')
            ax.axhline(0, ls=':', color='k', lw=0.8)
            ax.axvline(0, ls=':', color='k', lw=0.8)
            ax.set_xlabel(xlabel, labelpad=8)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_title(cond)

    h_A = mlines.Line2D([], [], color='k', marker='o', ls='none',
                         ms=8, label='odor A (no-lick)')
    h_B = mlines.Line2D([], [], color='k', marker='^', ls='none',
                         ms=8, label='odor B (lick)')
    suptitle_base = (f'Expert − Naive  [{train_tag}]  —  '
                     f'late-delay (bins {BINS_LATE[0]}–{BINS_LATE[-1]})')

    for use_zscore, suffix, extra in [
        (False, '_raw', ''),
        (True,  '',     '  z-scored'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(3 * width, height))
        _draw(axes, use_zscore)
        axes[-1].legend(handles=[h_A, h_B], fontsize=12, frameon=False)
        fig.suptitle(suptitle_base + extra, fontsize=16, y=1.02)
        fig.tight_layout()
        stem = f'{DUM}_{train_tag}{suffix}'
        fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
        path = os.path.join(fig_dir, 'png', f'{stem}.png')
        plt.close(fig)
    print(f'saved {path}')

print(f'\nScatter diff → {FIG_BASE}/')
