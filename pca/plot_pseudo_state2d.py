"""
2D sample-PC x choice-PC state-space trajectories — pseudo PCA analog of the
overlaps `plot_traj2d.py` figure.

Layout: 2 rows (Naive / Expert) x 3 cols (DPA / DualGo / DualNoGo).  Each cell is
a square 2D trajectory panel (x = sample PC, y = choice PC) plus a narrow KDE
histogram strip of the choice-PC location over the delay.

Per odor pair (AC/AD/BD/BC) we take per-mouse mean paths, then the group mean +
SEM-over-mice band, drawn as a light->dark time-gradient line with arrows.

Loads pseudo_{traj,labels}_<dum>.pkl from ../data/pca/.
Figures -> figures/pseudo/state2d/<scale>/{png,svg}/<dum>_pc<sx>x<cy>.{png,svg}
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.pca.identify import identify_pcs, pc_label
from src.plot.traj import plot_gradient_line, add_arrows, sem_band

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='2D sample-PC x choice-PC state-space trajectory grid (pseudo PCA).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dum', default='pseudo_DELAY_Expert_zscore_5x1',
                    help='Run-id tag of the pseudo_{traj,labels}_<dum>.pkl files')
parser.add_argument('--data-dir', default='../data/pca', dest='data_dir')
parser.add_argument('--sample-pc', type=int, default=None, dest='sample_pc',
                    help='1-indexed PC on the x-axis. Default: the identified Sample PC.')
parser.add_argument('--choice-pc', type=int, default=None, dest='choice_pc',
                    help='1-indexed PC on the y-axis (histogram axis). '
                         'Default: the identified Choice PC.')
parser.add_argument('--no-bl-correct', dest='bl_correct', action='store_false',
                    help='Disable per-trial baseline subtraction')
parser.set_defaults(bl_correct=True)
args = parser.parse_args()

DATA = args.data_dir
DUM  = args.dum

SCALE = DUM.split('_scale_')[1].split('_')[0] if '_scale_' in DUM else 'center'
FACTOR = DUM.split('_f-')[1] if '_f-' in DUM else 'odor_pair-tasks'
EPOCH = DUM.split('_')[1].lower()
CI = 'ci' + DUM.split('_ci')[1].split('_')[0] if '_ci' in DUM else 'ci0'
OUT = os.path.join('figures/pseudo/state2d', EPOCH, SCALE, CI, FACTOR)
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

PAIR_LABELS = {0: 'AC', 1: 'AD', 2: 'BD', 3: 'BC'}
_pal = sns.color_palette('tab10')
PAIR_COLOR = {0: _pal[0], 1: _pal[1], 2: _pal[2], 3: _pal[3]}
SAMPLE_SPLITS_HIST = [('A', [0, 1], '#332288'), ('B', [2, 3], '#44AA99')]

# time bins
options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0, trials='',
    data_type='dF', prescreen=None, pval=0.05, preprocess=None,
    scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
    scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL    = options['bins_BL']
BINS_DELAY = options['bins_DELAY']
TRAJ_END   = options['bins_TEST'][-1] + 1     # truncate paths at test offset

# ── load ──────────────────────────────────────────────────────────────────────

X = pkl_load(f'pseudo_traj_{DUM}',   path=DATA)   # (trials, n_comp, 84)
y = pkl_load(f'pseudo_labels_{DUM}', path=DATA)
print('traj', X.shape, ' labels', y.shape)

# identify which PC carries which task variable; default the axes to the
# Sample / Choice PCs (basis property → Expert trials)
PC_ID = identify_pcs(X, y, stage='Expert')
SX = (args.sample_pc - 1) if args.sample_pc else (
    PC_ID.index('Sample') if 'Sample' in PC_ID else 0)
CY = (args.choice_pc - 1) if args.choice_pc else (
    PC_ID.index('Choice') if 'Choice' in PC_ID else 1)
print('PC identity:', [pc_label(k, PC_ID) for k in range(min(X.shape[1], 3))],
      f'→ x=PC{SX + 1}, y=PC{CY + 1}')

if args.bl_correct:
    X = X - X[:, :, BINS_BL].mean(axis=2, keepdims=True)

idx_correct = (
    (y.laser == 0) & (y.performance == 1) &
    ((y.tasks == 'DPA') | (y.odr_perf == 1))
)

# ── per-mouse mean paths ──────────────────────────────────────────────────────

def mouse_paths(cond, stage, pair_id):
    """Per-mouse mean (x=PC_SX, y=PC_CY) paths for one (cond, stage, odor_pair)."""
    xs, ys = [], []
    for mouse in ALL_MICE:
        m = (idx_correct & (y.mouse == mouse) & (y.tasks == cond) &
             (y.learning == stage) & (y.odor_pair == pair_id)).to_numpy()
        if m.sum() == 0:
            continue
        mu = X[m].mean(0)            # (n_comp, time)
        xs.append(mu[SX]); ys.append(mu[CY])
    return xs, ys


traj = {s: {c: {p: mouse_paths(c, s, p) for p in PAIR_LABELS}
            for c in CONDITIONS} for s in STAGES}

# shared limits across all panels, from the GROUP-MEAN paths (per-mouse paths
# have large outliers that would stretch the view)
allx, ally = [], []
for s in STAGES:
    for c in CONDITIONS:
        for p in PAIR_LABELS:
            xs, ys = traj[s][c][p]
            if not xs:
                continue
            allx.append(np.stack(xs, 0)[:, :TRAJ_END].mean(0))
            ally.append(np.stack(ys, 0)[:, :TRAJ_END].mean(0))
allx = np.concatenate(allx); ally = np.concatenate(ally)
mx = 0.12 * (allx.max() - allx.min()); my = 0.12 * (ally.max() - ally.min())
xlim = (allx.min() - mx, allx.max() + mx)
ylim = (ally.min() - my, ally.max() + my)

# ── figure ────────────────────────────────────────────────────────────────────

width, hist_w = 2.8, 0.9
n_rows, n_cols = len(STAGES), len(CONDITIONS)
fig = plt.figure(figsize=(n_cols * (width + hist_w), n_rows * width),
                 constrained_layout=True)
gs = gridspec.GridSpec(n_rows, n_cols * 2, figure=fig,
                       width_ratios=[width, hist_w] * n_cols)

axes_traj = np.empty((n_rows, n_cols), dtype=object)
for ri in range(n_rows):
    for ci in range(n_cols):
        at = fig.add_subplot(gs[ri, ci * 2])
        ah = fig.add_subplot(gs[ri, ci * 2 + 1], sharey=at)
        axes_traj[ri, ci] = (at, ah)

for ri, stage in enumerate(STAGES):
    for ci, cond in enumerate(CONDITIONS):
        ax, ax_h = axes_traj[ri, ci]

        for pair_id in PAIR_LABELS:
            xs, ys = traj[stage][cond][pair_id]
            if not xs:
                continue
            color = PAIR_COLOR[pair_id]
            ax_arr = np.stack(xs, 0)[:, :TRAJ_END]
            ay_arr = np.stack(ys, 0)[:, :TRAJ_END]
            n_m = ax_arr.shape[0]
            x_mean, y_mean = ax_arr.mean(0), ay_arr.mean(0)
            if n_m > 1:
                x_sem = ax_arr.std(0, ddof=1) / np.sqrt(n_m)
                y_sem = ay_arr.std(0, ddof=1) / np.sqrt(n_m)
                sem_band(ax, x_mean, y_mean, x_sem, y_sem, color)
            plot_gradient_line(ax, x_mean, y_mean, color)
            add_arrows(ax, x_mean, y_mean, color, n_arrows=3)

        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.tick_params(length=3, width=0.9, labelsize=9)
        if ri == 0:
            ax.set_title(cond.replace('Dual', ''), pad=6, fontsize=12)
        if ri == n_rows - 1:
            ax.set_xlabel(pc_label(SX, PC_ID), fontsize=11)
        if ci == 0:
            ax.set_ylabel(pc_label(CY, PC_ID), fontsize=11)
        if ci == n_cols - 1:
            ax.annotate(stage, xy=(1.04, 0.5), xycoords='axes fraction',
                        fontsize=11, rotation=-90, va='center', ha='left',
                        color='0.3')

        # ── histogram: choice-PC location over the delay, split by sample ─────
        y_grid = np.linspace(ylim[0], ylim[1], 300)
        for label, pairs, color in SAMPLE_SPLITS_HIST:
            vals = []
            for pid in pairs:
                for y_traj in traj[stage][cond][pid][1]:
                    vals.extend(y_traj[BINS_DELAY].tolist())
            if len(vals) < 2:
                continue
            dens = gaussian_kde(vals, bw_method=0.4)(y_grid)
            ax_h.fill_betweenx(y_grid, 0, dens, color=color, alpha=0.35, lw=0)
            ax_h.plot(dens, y_grid, color=color, lw=1.2)
            ax_h.axhline(np.mean(vals), color=color, lw=1.4, ls='--',
                         alpha=0.9, zorder=5)
        ax_h.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax_h.set_xlim(left=0)
        ax_h.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for sp in ('left', 'bottom', 'top'):
            ax_h.spines[sp].set_visible(False)
        if ri == 0:
            ax_h.set_title('choice\ndist.', fontsize=8, pad=4, color='0.4', linespacing=1.2)
        if ri == 0 and ci == 0:
            ax_h.legend(handles=[Patch(facecolor=c, alpha=0.6, label=f'Sample {l}')
                                 for l, _, c in SAMPLE_SPLITS_HIST],
                        frameon=False, fontsize=7, loc='upper right',
                        handlelength=0.8, handletextpad=0.3, labelspacing=0.25)

axes_traj[0, -1][0].legend(
    handles=[Line2D([0], [0], color=PAIR_COLOR[p], lw=2, label=PAIR_LABELS[p])
             for p in PAIR_LABELS],
    frameon=False, loc='upper right', handletextpad=0.5, labelspacing=0.3)

stem = f'{DUM}_pc{SX + 1}x{CY + 1}'
fig.savefig(os.path.join(OUT, 'png', f'{stem}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'svg', f'{stem}.svg'), bbox_inches='tight')
plt.close(fig)
print('saved', os.path.join(OUT, 'png', f'{stem}.png'))
