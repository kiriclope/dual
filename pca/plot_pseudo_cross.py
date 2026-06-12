"""
Cross-projection: plot one run's trials on another run's PCA basis.

Default use: project the *none* (raw, un-centered) trials onto the *center*
run's PCs.  The basis (loadings + per-neuron mean/scale) is re-fit on the
basis-scale clean data, then the data-scale trials are projected through it.

Loads X_all_<scale> + y_all_<scale> + mouse_slices from ../data/pca/.
Figures -> figures/pseudo/cross/<data>_on_<basis>/<epoch>/<stage>/{png,svg}/
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
from sklearn.decomposition import PCA

from src.common.plot_utils import add_vlines
from src.pca.io import pkl_load
from src.pca.pseudo import pseudo_population_pca, project_trials
from src.pca.identify import identify_pcs, pc_label

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook')
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

BINS = {'DELAY': np.arange(18, 54), 'TEST': np.arange(54, 60),
        'ED': np.arange(18, 36), 'CHOICE': np.arange(60, 72)}

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Plot data-scale trials projected onto a basis-scale PCA basis.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--basis-scale', default='center', dest='basis_scale',
                    help="Scale of the run whose PCs define the basis (e.g. center)")
parser.add_argument('--data-scale', default='none', dest='data_scale',
                    help="Scale of the trials to project (e.g. none)")
parser.add_argument('--data-dir', default='../data/pca', dest='data_dir')
parser.add_argument('--epoch', default='DELAY', choices=list(BINS))
parser.add_argument('--norm', default='zscore', choices=['zscore', 'mad', 'none'])
parser.add_argument('--factors', nargs='+', default=['odor_pair', 'tasks'])
parser.add_argument('--n-comp', type=int, default=6, dest='n_comp')
parser.add_argument('--stage', default='Expert', choices=['Expert', 'Naive'])
parser.add_argument('--n-show', type=int, default=3, dest='n_show')
parser.add_argument('--relevant', action='store_true',
                    help='Show the identified Sample/Choice/Test PCs by role')
parser.add_argument('--no-bl-correct', dest='bl_correct', action='store_false')
parser.set_defaults(bl_correct=True)
args = parser.parse_args()

DATA = args.data_dir
norm = None if args.norm == 'none' else args.norm
epoch_bins = BINS[args.epoch]
bl = slice(0, 12)
xtime = np.linspace(0, 14, 84)


def scale_tag(s):
    return '' if s == 'none' else s


FACTOR = '-'.join(args.factors)
OUT = os.path.join('figures/pseudo/cross',
                   f'{args.data_scale}_on_{args.basis_scale}',
                   args.epoch.lower(), FACTOR, args.stage)
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
pal = sns.color_palette('muted')

# ── load basis-scale + data-scale ───────────────────────────────────────────────

ms = pkl_load('mouse_slices', path=DATA)


def load_xy(scale):
    Xs = pkl_load(f'X_all_{scale_tag(scale)}', path=DATA)
    ys = pkl_load(f'y_all_{scale_tag(scale)}', path=DATA)
    ys['sample'] = ys.sample_odor; ys['test'] = ys.test_odor
    return Xs, ys


Xb, yb = load_xy(args.basis_scale)
Xd, yd = load_xy(args.data_scale)

# ── fit basis on basis-scale clean trials ───────────────────────────────────────

mb = (yb.laser == 0) & (yb.learning == 'Expert') & (yb.performance == 1)
W, mean_, scale_, evr, _, _ = pseudo_population_pca(
    Xb[mb.to_numpy()], yb.loc[mb].reset_index(drop=True),
    PCA(n_components=args.n_comp, svd_solver='auto'), args.factors, ms,
    epoch=epoch_bins, bl_bins=bl, norm=norm,
)
print(f'basis fit on {args.basis_scale} clean  EVR(%):', np.round(evr * 100, 1))

# ── PC identity from the basis run's saved (held-out, cross-validated) traj ─────
# (an in-sample re-projection of the basis data overfits Choice onto near-
#  degenerate high PCs; the saved held-out traj gives the reliable assignment)
def basis_dum():
    d = f'pseudo_{args.epoch}_Expert_{args.norm}_5x1'
    if args.basis_scale != 'center':
        d += '_scale_' + args.basis_scale
    if args.basis_scale == 'none':
        d += '_raw'
    if args.factors != ['odor_pair', 'tasks']:
        d += '_f-' + '-'.join(args.factors)
    return d

bdum = basis_dum()
PC_ID = identify_pcs(pkl_load(f'pseudo_traj_{bdum}', path=DATA),
                     pkl_load(f'pseudo_labels_{bdum}', path=DATA), stage='Expert')
print(f'{args.basis_scale}-basis PC identity (from {bdum}):',
      [pc_label(k, PC_ID) for k in range(args.n_comp)])

Zd, y = project_trials(Xd, yd, W, mean_, scale_, ms)
X = np.swapaxes(Zd, 1, 2).astype(float)        # (trials, n_comp, 84)
print(f'projected {args.data_scale} trials:', X.shape)

if args.relevant:
    PCS = [PC_ID.index(r) for r in ['Sample', 'Choice', 'Test'] if r in PC_ID]
else:
    PCS = list(range(args.n_show))

STAGE = args.stage
base_mask = (y.learning == STAGE) & (y.laser == 0) & (y.performance == 1)
SEL = STAGE + ('_relPCs' if args.relevant else '')
W_, H_ = 3.5, 2.6

# ── figure builder (mirrors plot_pseudo_traj) ───────────────────────────────────

def traj_fig(mask, factor, levels, labels, colors, title):
    fig, axes = plt.subplots(1, len(PCS), figsize=(len(PCS) * W_, H_))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, PCS):
        add_vlines(ax, if_dpa=0)
        ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(pc_label(k, PC_ID), fontsize=10)
        ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
        ax.tick_params(labelsize=8)
    for lv, lab, col in zip(levels, labels, colors):
        sel = mask & (y[factor] == lv)
        Xs = X[sel.to_numpy()]
        if Xs.shape[0] == 0:
            continue
        if args.bl_correct:
            Xs = Xs - Xs[:, :, bl].mean(axis=2, keepdims=True)
        mu, sem = Xs.mean(0), Xs.std(0) / np.sqrt(Xs.shape[0])
        for ax, k in zip(axes, PCS):
            from src.plot.traj import plot_mean_sem
            plot_mean_sem(ax, xtime, mu[k], sem[k], col, lw=1.6, label=lab, zorder=2)
    lo = min(ax.get_ylim()[0] for ax in axes); hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(fontsize=8, frameon=False, loc='best')
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def save(fig, tag):
    p = os.path.join(OUT, 'png', f'{args.data_scale}_on_{args.basis_scale}_{SEL}_{tag}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print('saved', p)


save(traj_fig(base_mask, 'odor_pair', [0, 1, 2, 3], ['AC', 'AD', 'BD', 'BC'],
              ['#332288', '#88CCEE', '#117733', '#44AA99'],
              f'{args.data_scale} trials on {args.basis_scale} PCs — odor pair'), 'odor_pair')
save(traj_fig(base_mask, 'sample_odor', [0, 1], ['Odor A', 'Odor B'],
              ['#332288', '#44AA99'],
              f'{args.data_scale} on {args.basis_scale} PCs — sample'), 'sample')
save(traj_fig(base_mask, 'choice', [0, 1], ['No lick', 'Lick'], ['#377eb8', '#4daf4a'],
              f'{args.data_scale} on {args.basis_scale} PCs — choice'), 'choice')
save(traj_fig(base_mask, 'test_odor', [0, 1], ['Odor C', 'Odor D'], ['#377eb8', '#4daf4a'],
              f'{args.data_scale} on {args.basis_scale} PCs — test'), 'test')
print('\nAll done.')
