"""
2D PC-plane trajectories for a pseudo-population PCA run.

For each split (odor_pair / tasks / sample / choice / test) draws the three PC
planes (PC1-2, PC1-3, PC2-3) as time-gradient paths with direction arrows
(shared primitives from src/plot/traj.py), and saves PNG (dpi=300) + SVG.

Loads pseudo_{traj,labels}_<dum>.pkl from ../data/pca/.
Figures -> figures/pseudo/traj2d/<scale>/<stage>/{png,svg}/<dum>_<SEL>_<split>.*
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

from src.pca.io import pkl_load
from src.pca.identify import identify_pcs, pc_label
from src.pca.plot import plot_trajectories_2d

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook')
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

# ── config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Plot 2D PC-plane trajectories from a pseudo-population PCA run.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dum', default='pseudo_DELAY_Expert_zscore_5x1',
                    help='Run-id tag of the pseudo_{traj,labels}_<dum>.pkl files')
parser.add_argument('--data-dir', default='../data/pca', dest='data_dir',
                    help='Directory holding the pseudo result pkls')
parser.add_argument('--stage', default='Expert', choices=['Expert', 'Naive'],
                    help='Learning stage of trials to plot')
parser.add_argument('--correct', dest='correct', action='store_true',
                    help='Correct trials only (default)')
parser.add_argument('--no-correct', dest='correct', action='store_false',
                    help='Include all trials regardless of performance')
parser.add_argument('--laser', default='0', choices=['0', '1', 'all'],
                    help='Laser state: 0=off (default), 1=on, all=both')
parser.add_argument('--no-bl-correct', dest='bl_correct', action='store_false',
                    help='Disable per-trial baseline subtraction')
parser.add_argument('--t-start', type=float, default=0.0, dest='t_start',
                    help='Start time (s) of the plotted path')
parser.add_argument('--t-end', type=float, default=14.0, dest='t_end',
                    help='End time (s) of the plotted path')
parser.set_defaults(bl_correct=True, correct=True)
args = parser.parse_args()

DATA  = args.data_dir
DUM   = args.dum
STAGE = args.stage
BL    = slice(0, 12)        # pre-stim baseline (t 0-2 s)
xtime = np.linspace(0, 14, 84)
i0    = int(np.searchsorted(xtime, args.t_start))
i1    = int(np.searchsorted(xtime, args.t_end, side='right'))

# scale lives in the DUM ('_scale_<x>'); untagged means the default 'center'
SCALE = DUM.split('_scale_')[1].split('_')[0] if '_scale_' in DUM else 'center'
FACTOR = DUM.split('_f-')[1] if '_f-' in DUM else 'odor_pair-tasks'
EPOCH = DUM.split('_')[1].lower()
CI = 'ci' + DUM.split('_ci')[1].split('_')[0] if '_ci' in DUM else 'ci0'
OUT = os.path.join('figures/pseudo/traj2d', EPOCH, SCALE, CI, FACTOR, STAGE)
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# selection tag kept in the filename (also reflected in the <stage> subfolder)
SEL = STAGE
if args.laser != '0':
    SEL += f'_laser{args.laser}'
if not args.correct:
    SEL += '_all'

pal = sns.color_palette('muted')

# ── load ──────────────────────────────────────────────────────────────────────

X = pkl_load(f'pseudo_traj_{DUM}',   path=DATA)   # (trials, n_comp, 84)
y = pkl_load(f'pseudo_labels_{DUM}', path=DATA)
print('traj', X.shape, ' labels', y.shape)

# identify which PC carries which task variable (basis property → Expert trials)
PC_ID = identify_pcs(X, y, stage='Expert')
PC_LABELS = [pc_label(k, PC_ID) for k in range(X.shape[1])]
print('PC identity:', PC_LABELS[:3])

if args.bl_correct:
    X = X - X[:, :, BL].mean(axis=2, keepdims=True)
# restrict to the requested time window (paths are time-ordered)
X = X[:, :, i0:i1]

base_mask = (y.learning == STAGE)
if args.laser != 'all':
    base_mask &= (y.laser == int(args.laser))
if args.correct:
    base_mask &= (y.performance == 1)


def save(fig, tag):
    png = os.path.join(OUT, 'png', f'{DUM}_{SEL}_{tag}.png')
    svg = os.path.join(OUT, 'svg', f'{DUM}_{SEL}_{tag}.svg')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print('saved', png)


def panel(mask, factor, labels, colors, tag, title):
    fig, axes = plot_trajectories_2d(X, y, mask.to_numpy(), factor, labels, colors,
                                     pc_labels=PC_LABELS)
    fig.suptitle(title, fontsize=11, y=1.03)
    save(fig, tag)


# ── panels (labels/colors ordered by sorted factor level) ─────────────────────

panel(base_mask, 'odor_pair', ['AC', 'AD', 'BD', 'BC'],
      ['#332288', '#88CCEE', '#117733', '#44AA99'],
      'odor_pair', 'Pseudo PCA 2D — by odor pair')

task_mask = base_mask & ((y.tasks == 'DPA') | (y.odr_perf == 1)) if args.correct else base_mask
panel(task_mask, 'tasks', ['DPA', 'Go', 'NoGo'], [pal[3], pal[0], pal[2]],
      'tasks', 'Pseudo PCA 2D — by task')

panel(base_mask, 'sample_odor', ['Odor A', 'Odor B'], ['#332288', '#44AA99'],
      'sample', 'Pseudo PCA 2D — by sample')

panel(base_mask, 'choice', ['No lick', 'Lick'], ['#377eb8', '#4daf4a'],
      'choice', 'Pseudo PCA 2D — by choice')

panel(base_mask, 'test_odor', ['Odor C', 'Odor D'], ['#377eb8', '#4daf4a'],
      'test', 'Pseudo PCA 2D — by test')

print('\nAll done.')
