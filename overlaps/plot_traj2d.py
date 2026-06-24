"""
2D trajectory: sample code (x) × choice code (y) over time.

Each panel traces the mean population state through (sample, choice) space
as time progresses from 0→14 s. Path is colored by time.
Individual mouse paths shown as thin translucent lines.
Epoch boundaries marked with symbols.

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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import make_time_cmap, plot_gradient_line, add_arrows, sem_band

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_style("ticks")

width  = 2.8
height = width   # square panels for 2D trajectory

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
    'svg.fonttype':      'none',   # keep text editable in SVG
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'

# Trial set: 'correct' (default) = correct trials only; 'all' = all laser-off
# (correct + incorrect). Pass 'all' (or --all) on the command line.
TRIALS   = 'all' if {'all', '--all'} & set(sys.argv[1:]) else 'correct'
FIG_BASE = f'./figures/overlaps/traj2d/{TRIALS}'
os.makedirs(os.path.join(FIG_BASE, 'png'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'svg'), exist_ok=True)

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

# Odor-pair colour code — matches plot_pairs.py
#   0→AC, 1→AD, 2→BD, 3→BC ; match pairs (AC, BD) solid, non-match dashed
PAIR_LABELS = {0: 'AC', 1: 'AD', 2: 'BD', 3: 'BC'}
PAIR_STYLE  = {0: '-', 1: '--', 2: '-', 3: '--'}
_pal_pairs  = sns.color_palette('tab10')
PAIR_COLOR  = {0: _pal_pairs[0], 1: _pal_pairs[1],
               2: _pal_pairs[2], 3: _pal_pairs[3]}

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
BINS_LATE  = BINS_DELAY[int(0.6 * len(BINS_DELAY)):]

# Sample identity splits for histograms
SAMPLE_SPLITS_HIST = [
    ('A', [0, 1], '#332288'),
    ('B', [2, 3], '#44AA99'),
]

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
]

# Trajectories are drawn up to the test offset (end of the test epoch).
TRAJ_END = options['bins_TEST'][-1] + 1

# Stimulus/event markers placed on each trajectory at the epoch-onset bin.
# Only events at or before the test offset are shown (choice is dropped).
EVENT_MARKERS = [
    (name, b, mk) for name, b, mk in [
        ('sample', options['bins_STIM'][0],   's'),
        ('delay',  options['bins_ED'][0],     '^'),
        ('test',   options['bins_TEST'][0],   'D'),
        ('choice', options['bins_CHOICE'][0], '*'),
    ] if b < TRAJ_END
]

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_laser = (y_single.laser == 0)
# Correct trials only: DPA correct, and GNG correct (odr_perf) on Dual trials.
idx_correct = (
    idx_laser & (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)
# Trial mask used to build trajectories: all laser-off, or correct only.
idx_trials = idx_laser if TRIALS == 'all' else idx_correct
print(f'TRIALS={TRIALS}  n_trials={int(idx_trials.sum())}')
xtime     = np.linspace(0, 14, X_single.shape[-1])

# ── Helpers ───────────────────────────────────────────────────────────────────


def mouse_trajectories(X_ep, cond, stage, target, odor_pairs=None):
    """Return list of per-mouse mean trajectories (T_test,).

    odor_pairs: if given, restrict to trials whose odor_pair is in this list.
    """
    trajs = []
    for mouse in ALL_MICE:
        base = (
            (y_single.mouse  == mouse) &
            (y_single.tasks  == cond)  &
            (y_single.stage  == stage) &
            (y_single.target == target) &
            idx_trials
        )
        if odor_pairs is not None:
            base = base & y_single.odor_pair.isin(odor_pairs)
        mask = base.values
        if mask.sum() == 0:
            continue
        trajs.append(X_ep[mask].mean(0))   # (T_test,)
    return trajs

# ── Main loop ─────────────────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    print(f'\n=== {train_tag} ===')

    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd

    # ── Collect per-mouse trajectories for each odor pair ────────────────────
    # traj[stage][cond][pair_id] = (x_mice, y_mice)
    traj = {s: {c: {} for c in CONDITIONS} for s in STAGES}
    for stage in STAGES:
        for cond in CONDITIONS:
            for pair_id in PAIR_LABELS:
                xs = mouse_trajectories(X_ep, cond, stage, 'sample',
                                        odor_pairs=[pair_id])
                ys = mouse_trajectories(X_ep, cond, stage, 'choice',
                                        odor_pairs=[pair_id])
                traj[stage][cond][pair_id] = (xs, ys)

    # Fixed limits
    xlim = (-4, 4)
    ylim = (-2, 6)

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows, n_cols = len(STAGES), len(CONDITIONS)
    # Each condition gets a wide 2D panel + a narrow histogram strip.
    hist_w = 0.9          # histogram strip width (inches)
    fig_w  = n_cols * width + n_cols * hist_w
    fig_h  = n_rows * height
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs  = gridspec.GridSpec(n_rows, n_cols * 2, figure=fig,
                            width_ratios=[width, hist_w] * n_cols)

    axes_traj = np.empty((n_rows, n_cols), dtype=object)
    axes_hist = np.empty((n_rows, n_cols), dtype=object)
    for ri in range(n_rows):
        for ci in range(n_cols):
            at = fig.add_subplot(gs[ri, ci * 2])
            ah = fig.add_subplot(gs[ri, ci * 2 + 1], sharey=at)
            if ri > 0 or ci > 0:
                at.sharex(axes_traj[0, 0])
                at.sharey(axes_traj[0, 0])
            axes_traj[ri, ci] = at
            axes_hist[ri, ci] = ah

    for ri, stage in enumerate(STAGES):
        for ci, cond in enumerate(CONDITIONS):
            ax   = axes_traj[ri, ci]
            ax_h = axes_hist[ri, ci]

            for pair_id in PAIR_LABELS:
                xs, ys = traj[stage][cond][pair_id]
                if not xs or not ys:
                    continue
                color = PAIR_COLOR[pair_id]

                # Truncate at the test offset
                arr_x = np.stack(xs, 0)[:, :TRAJ_END]   # (n_mice, T')
                arr_y = np.stack(ys, 0)[:, :TRAJ_END]
                n_mice = arr_x.shape[0]
                x_mean, y_mean = arr_x.mean(0), arr_y.mean(0)
                x_sem = arr_x.std(0, ddof=1) / np.sqrt(n_mice)
                y_sem = arr_y.std(0, ddof=1) / np.sqrt(n_mice)

                # SEM-over-mice band around the mean path
                sem_band(ax, x_mean, y_mean, x_sem, y_sem, color)

                # Group-mean path coloured by time (light → full pair colour)
                plot_gradient_line(ax, x_mean, y_mean, color)
                add_arrows(ax, x_mean, y_mean, color, n_arrows=3)

            ax.axhline(0, color='0.85', lw=0.6, zorder=0)
            ax.axvline(0, color='0.85', lw=0.6, zorder=0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-2, 0, 2, 4, 6])
            ax.tick_params(length=3, width=0.9)
            for artist in ax.collections:
                artist.set_clip_on(True)

            if ri == 0:
                ax.set_title(cond, pad=6)
            if ri == n_rows - 1:
                ax.set_xlabel('Sample code')
            if ci == 0:
                ax.set_ylabel('Choice code')
            if ci == n_cols - 1:
                ax.annotate(stage, xy=(1.04, 0.5), xycoords='axes fraction',
                            fontsize=11, rotation=-90, va='center', ha='left',
                            color='0.3')

            # ── Histogram: choice code over BINS_DELAY ───────────────────────
            y_grid   = np.linspace(ylim[0], ylim[1], 300)
            hist_handles = []
            for label, pairs, color in SAMPLE_SPLITS_HIST:
                vals = []
                for pair_id in pairs:
                    ys_list = traj[stage][cond][pair_id][1]
                    for y_traj in ys_list:
                        vals.extend(y_traj[BINS_DELAY].tolist())
                if len(vals) < 2:
                    continue
                kde  = gaussian_kde(vals, bw_method=0.4)
                dens = kde(y_grid)
                ax_h.fill_betweenx(y_grid, 0, dens,
                                   color=color, alpha=0.35, lw=0)
                ax_h.plot(dens, y_grid, color=color, lw=1.2)
                mean_val = np.mean(vals)
                ax_h.axhline(mean_val, color=color, lw=1.4,
                             ls='--', alpha=0.9, zorder=5)
                from matplotlib.patches import Patch
                hist_handles.append(Patch(facecolor=color, alpha=0.6,
                                          label=f'Sample {label}'))

            ax_h.axhline(0, color='0.85', lw=0.6, zorder=0)
            ax_h.set_xlim(left=0)
            ax_h.tick_params(left=False, labelleft=False,
                             bottom=False, labelbottom=False)
            ax_h.spines['left'].set_visible(False)
            ax_h.spines['bottom'].set_visible(False)
            ax_h.spines['top'].set_visible(False)

            if ri == 0:
                ax_h.set_title('choice\ndist.', fontsize=8, pad=4,
                               color='0.4', linespacing=1.2)
            if ri == 0 and ci == 0 and hist_handles:
                ax_h.legend(handles=hist_handles, frameon=False,
                            fontsize=7, loc='upper right',
                            handlelength=0.8, handletextpad=0.3,
                            labelspacing=0.25, borderaxespad=0.2)

    # Legend — odor pairs (colour swatch = base/end colour of gradient)
    pair_handles = [
        Line2D([0], [0], color=PAIR_COLOR[p], lw=2.0, label=PAIR_LABELS[p])
        for p in PAIR_LABELS
    ]
    axes_traj[0, -1].legend(handles=pair_handles, frameon=False,
                            loc='upper right', handletextpad=0.5,
                            borderaxespad=0.2, labelspacing=0.3)

    stem = f'{DUM}_{train_tag}'
    fig.savefig(os.path.join(FIG_BASE, 'png', f'{stem}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_BASE, 'svg', f'{stem}.svg'), bbox_inches='tight')
    path = os.path.join(FIG_BASE, 'png', f'{stem}.png')
    plt.close(fig)
    print(f'saved {path}')

print(f'\nTraj2D → {FIG_BASE}/')
