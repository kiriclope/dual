"""
EVR and loadings figures for a pseudo-population PCA run.

These are properties of the fitted basis (independent of which trials are
plotted), so they are grouped by scale only:

  figures/pseudo/evr/<scale>/{png,svg}/<dum>_evr.{png,svg}
  figures/pseudo/loadings/<scale>/{png,svg}/<dum>_{theta,2d,energy}.{png,svg}

Loads pseudo_{evr,weights}_<dum>.pkl and mouse_slices.pkl from ../data/pca/.
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
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from src.pca.io import pkl_load
from src.pca.identify import identify_pcs, pc_label
from src.pca.meta import pc_mouse_energy

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook')
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

# ── config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Plot EVR and loadings from a pseudo-population PCA run.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dum', default='pseudo_DELAY_Expert_zscore_5x1',
                    help='Run-id tag of the pseudo_{evr,weights}_<dum>.pkl files')
parser.add_argument('--data-dir', default='../data/pca', dest='data_dir',
                    help='Directory holding the pseudo result pkls + mouse_slices')
parser.add_argument('--n-show', type=int, default=3, dest='n_show',
                    help='Number of PCs shown in the loadings panels')
args = parser.parse_args()

DATA   = args.data_dir
DUM    = args.dum
n_show = args.n_show
W, H   = 3.5, 2.6

# scale lives in the DUM ('_scale_<x>'); untagged means the default 'center'
SCALE = DUM.split('_scale_')[1].split('_')[0] if '_scale_' in DUM else 'center'
FACTOR = DUM.split('_f-')[1] if '_f-' in DUM else 'odor_pair-tasks'
EPOCH = DUM.split('_')[1].lower()
CI = 'ci' + DUM.split('_ci')[1].split('_')[0] if '_ci' in DUM else 'ci0'

EVR_OUT  = os.path.join('figures/pseudo/evr', EPOCH, SCALE, CI, FACTOR)
LOAD_OUT = os.path.join('figures/pseudo/loadings', EPOCH, SCALE, CI, FACTOR)
for base in (EVR_OUT, LOAD_OUT):
    for sub in ('png', 'svg'):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

try:
    import cmocean
    cmap = cmocean.cm.phase
except ImportError:
    cmap = plt.cm.hsv

# ── load ──────────────────────────────────────────────────────────────────────

evr = np.asarray(pkl_load(f'pseudo_evr_{DUM}',     path=DATA))   # (n_folds, n_comp)
w   = pkl_load(f'pseudo_weights_{DUM}', path=DATA)                # (n_comp, n_neurons)
ms  = pkl_load('mouse_slices',          path=DATA)
mice = list(ms.keys())
n_comp = w.shape[0]
print('evr', evr.shape, ' weights', w.shape, ' n_mice', len(mice))

# identify which PC carries which task variable (needs the projected traj)
PC_ID = identify_pcs(pkl_load(f'pseudo_traj_{DUM}',   path=DATA),
                     pkl_load(f'pseudo_labels_{DUM}', path=DATA), stage='Expert')
print('PC identity:', [pc_label(k, PC_ID) for k in range(min(n_comp, 3))])


def save(fig, out, tag):
    png = os.path.join(out, 'png', f'{DUM}_{tag}.png')
    svg = os.path.join(out, 'svg', f'{DUM}_{tag}.svg')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print('saved', png)


# ── 1. EVR ────────────────────────────────────────────────────────────────────

evr_mean = evr.mean(0)
evr_sem  = evr.std(0, ddof=1) / np.sqrt(evr.shape[0]) if evr.shape[0] > 1 else np.zeros(n_comp)
n_pcs    = np.arange(1, n_comp + 1)

fig, ax = plt.subplots(figsize=(W, W))
ax.plot(n_pcs, evr_mean, '-o', ms=5)
ax.fill_between(n_pcs, evr_mean - evr_sem, evr_mean + evr_sem, alpha=0.25)
ax.set_xlabel('PC #', fontsize=11)
ax.set_ylabel('Explained variance ratio', fontsize=11)
ax.set_xticks(n_pcs)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
fig.tight_layout()
save(fig, EVR_OUT, 'evr')

# ── 2. Loadings vs preferred direction (theta) ────────────────────────────────

theta      = np.arctan2(w[1], w[0]) * 180 / np.pi
idx_sorted = np.argsort(theta)
theta_norm = (theta + 360) % 360
smooth_w   = max(1, int(0.05 * w.shape[1]))
z_lim      = float(np.percentile(np.abs(w[:n_show]), 99) * 1.1)

fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, 2 * H))
for k, ax in enumerate(axes):
    sc = ax.scatter(theta[idx_sorted], w[k, idx_sorted],
                    c=theta_norm[idx_sorted], cmap=cmap,
                    alpha=0.4, s=3, rasterized=True)
    ax.plot(theta[idx_sorted],
            gaussian_filter1d(w[k, idx_sorted], smooth_w, mode='wrap'),
            'k', lw=1.5)
    ax.axhline(0, ls='--', color='k', lw=0.6)
    ax.set_ylabel(f'Weights {pc_label(k, PC_ID)}', fontsize=10)
    ax.set_xlabel('Neuron loc (°)', fontsize=10)
    ax.set_ylim([-z_lim, z_lim])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=8)
plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
fig.suptitle('Pseudo-PC loadings vs preferred direction', fontsize=11, y=1.01)
fig.tight_layout()
save(fig, LOAD_OUT, 'theta')

# ── 3. Loadings in 2-D weight planes ──────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(3 * W, 2 * H))
for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
    sc = ax.scatter(w[a, idx_sorted], w[b, idx_sorted],
                    c=theta_norm[idx_sorted], cmap=cmap,
                    alpha=0.4, s=3, rasterized=True)
    ax.plot(
        gaussian_filter1d(w[a, idx_sorted], smooth_w, mode='wrap'),
        gaussian_filter1d(w[b, idx_sorted], smooth_w, mode='wrap'),
        'k', lw=1.5,
    )
    ax.set_xlabel(f'{pc_label(a, PC_ID)} weight', fontsize=10)
    ax.set_ylabel(f'{pc_label(b, PC_ID)} weight', fontsize=10)
    ax.set_xlim(-z_lim, z_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.tick_params(labelsize=8)
plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
fig.suptitle('Pseudo-PC loadings in weight space', fontsize=11, y=1.01)
fig.tight_layout()
save(fig, LOAD_OUT, '2d')

# ── 4. Loading energy per mouse ───────────────────────────────────────────────

energy_df   = pc_mouse_energy(w / 100.0, ms)            # (n_comp, n_mice)
energy_frac = energy_df.div(energy_df.sum(axis=1), axis=0)
mouse_cols  = sns.color_palette('tab10', n_colors=len(mice))

fig, axes = plt.subplots(1, 2, figsize=(2 * W * 1.5, 2 * H))
for ax, df, ylab, pct in (
    (axes[0], energy_df,   'Loading energy (a.u.)', False),
    (axes[1], energy_frac, 'Fraction of loading energy', True),
):
    bottom = np.zeros(n_comp)
    for mouse, col in zip(mice, mouse_cols):
        vals = df[mouse].values
        ax.bar(range(1, n_comp + 1), vals, bottom=bottom, color=col,
               label=mouse, width=0.7)
        bottom += vals
    ax.set_xlabel('PC #', fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xticks(range(1, n_comp + 1))
    ax.tick_params(labelsize=8)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
axes[0].legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
fig.suptitle('Pseudo-PC loading energy per mouse', fontsize=11, y=1.01)
fig.tight_layout()
save(fig, LOAD_OUT, 'energy')

print('\nAll done.')
