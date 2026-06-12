"""
1D PC trajectories for the pseudo-population PCA.

For each split (odor_pair / tasks / sample / choice / test) draws PC1-n_show
mean +/- SEM time traces and saves PNG (dpi=300) + SVG.

Loads pseudo_{traj,labels}_<DUM>.pkl from ../data/pca/.
Figures -> figures/pseudo/traj/{png,svg}/{DUM}_<split>.{png,svg}
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys, os
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.plot_utils import add_vlines
from src.pca.io import pkl_load
from src.pca.identify import identify_pcs, pc_label
from src.plot.traj import plot_mean_sem

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook')
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

# ── config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Plot 1D PC trajectories from a pseudo-population PCA run.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dum', default='pseudo_DELAY_Expert_zscore_5x1_raw',
                    help='Run-id tag of the pseudo_{traj,labels}_<dum>.pkl files')
parser.add_argument('--data-dir', default='../data/pca', dest='data_dir',
                    help='Directory holding the pseudo result pkls')
parser.add_argument('--stage', default='Expert', choices=['Expert', 'Naive'],
                    help='Learning stage of trials to plot')
parser.add_argument('--n-show', type=int, default=3, dest='n_show',
                    help='Number of PCs to show')
parser.add_argument('--no-bl-correct', dest='bl_correct', action='store_false',
                    help='Disable per-trial baseline subtraction')
parser.add_argument('--correct', dest='correct', action='store_true',
                    help='Correct trials only (default)')
parser.add_argument('--no-correct', dest='correct', action='store_false',
                    help='Include all trials regardless of performance')
parser.add_argument('--laser', default='0', choices=['0', '1', 'all'],
                    help='Laser state: 0=off (default), 1=on, all=both')
parser.add_argument('--relevant', dest='relevant', action='store_true',
                    help='Plot the identified task PCs (Sample/Choice/Test) by '
                         'role instead of the first --n-show PCs by index.')
parser.set_defaults(bl_correct=True, correct=True)
args = parser.parse_args()

DATA = args.data_dir
DUM  = args.dum

n_show     = args.n_show     # PCs shown
W, H       = 3.5, 2.6       # panel size (inches)
xtime      = np.linspace(0, 14, 84)
BL         = slice(0, 12)   # pre-stim baseline (t 0-2 s)
bl_correct = args.bl_correct
STAGE      = args.stage

# scale lives in the DUM ('_scale_<x>'); untagged means the default 'center'
SCALE = DUM.split('_scale_')[1].split('_')[0] if '_scale_' in DUM else 'center'
FACTOR = DUM.split('_f-')[1] if '_f-' in DUM else 'odor_pair-tasks'
EPOCH = DUM.split('_')[1].lower()   # fit window (delay/test/...)
CI = 'ci' + DUM.split('_ci')[1].split('_')[0] if '_ci' in DUM else 'ci0'

# figures grouped by scale, factor set, then plotted stage
OUT = os.path.join('figures/pseudo/traj', EPOCH, SCALE, CI, FACTOR, STAGE)
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

pal = sns.color_palette('muted')

# ── load ──────────────────────────────────────────────────────────────────────

X = pkl_load(f'pseudo_traj_{DUM}',   path=DATA)   # (trials, n_comp, 84)
y = pkl_load(f'pseudo_labels_{DUM}', path=DATA)
print('traj', X.shape, ' labels', y.shape)

# identify which PC carries which task variable (basis property → Expert trials)
PC_ID = identify_pcs(X, y, stage='Expert')

# which PCs to show: the identified task PCs (by role), or the first n_show
if args.relevant:
    PCS = [PC_ID.index(r) for r in ['Sample', 'Choice', 'Test'] if r in PC_ID]
else:
    PCS = list(range(n_show))
print('PC identity:', [pc_label(k, PC_ID) for k in range(len(PC_ID))])
print('showing PCs:', [pc_label(k, PC_ID) for k in PCS])

# trial selection from flags (the 'Expert' in the DUM is the BASIS-fit stage;
# STAGE here is which trials get projected/plotted)
base_mask = (y.learning == STAGE)
if args.laser != 'all':
    base_mask &= (y.laser == int(args.laser))
if args.correct:
    base_mask &= (y.performance == 1)

# selection tag kept in the filename (also reflected in the <stage> subfolder)
SEL = STAGE
if args.laser != '0':
    SEL += f'_laser{args.laser}'
if not args.correct:
    SEL += '_all'

# ── figure builder ────────────────────────────────────────────────────────────

def traj_fig(mask, factor, levels, labels, colors, title):
    fig, axes = plt.subplots(1, len(PCS), figsize=(len(PCS) * W, H))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, PCS):
        add_vlines(ax, if_dpa=0)
        ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(pc_label(k, PC_ID), fontsize=10)
        ax.set_xlim([0, 14])
        ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
        ax.tick_params(labelsize=8)
    for lv, lab, col in zip(levels, labels, colors):
        sel = mask & (y[factor] == lv)
        Xs = X[sel.to_numpy()]
        if Xs.shape[0] == 0:
            continue
        if bl_correct:
            Xs = Xs - Xs[:, :, BL].mean(axis=2, keepdims=True)
        mu  = Xs.mean(0)
        sem = Xs.std(0) / np.sqrt(Xs.shape[0])
        for ax, k in zip(axes, PCS):
            plot_mean_sem(ax, xtime, mu[k], sem[k], col, lw=1.6, label=lab, zorder=2)
    # shared ylim across the three PC panels
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(fontsize=8, frameon=False, loc='best')
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


PCS_TAG = '_relPCs' if args.relevant else ''


def save(fig, tag):
    png = os.path.join(OUT, 'png', f'{DUM}_{SEL}_{tag}{PCS_TAG}.png')
    svg = os.path.join(OUT, 'svg', f'{DUM}_{SEL}_{tag}{PCS_TAG}.svg')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print('saved', png)


# ── panels ────────────────────────────────────────────────────────────────────

save(traj_fig(base_mask, 'odor_pair', [0, 1, 2, 3], ['AC', 'AD', 'BD', 'BC'],
              ['#332288', '#88CCEE', '#117733', '#44AA99'],
              'Pseudo PCA — by odor pair'), 'odor_pair')

task_mask = base_mask & ((y.tasks == 'DPA') | (y.odr_perf == 1)) if args.correct else base_mask
save(traj_fig(task_mask, 'tasks', ['DPA', 'DualGo', 'DualNoGo'],
              ['DPA', 'Go', 'NoGo'], [pal[3], pal[0], pal[2]],
              'Pseudo PCA — by task'), 'tasks')

save(traj_fig(base_mask, 'sample_odor', [0, 1], ['Odor A', 'Odor B'],
              ['#332288', '#44AA99'], 'Pseudo PCA — by sample'), 'sample')

save(traj_fig(base_mask, 'choice', [0, 1], ['No lick', 'Lick'],
              ['#377eb8', '#4daf4a'], 'Pseudo PCA — by choice'), 'choice')

save(traj_fig(base_mask, 'test_odor', [0, 1], ['Odor C', 'Odor D'],
              ['#377eb8', '#4daf4a'], 'Pseudo PCA — by test'), 'test')

print('\nAll done.')
