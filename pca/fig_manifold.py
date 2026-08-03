"""Defensible WM-manifold figure (no fitted flow, no extrapolation):
 A. single-trial delay occupancy — what the data resolves at the single-trial level
    (one contractive 2-D blob; A/B wells are sub-noise, shown as the condition-mean dots).
 B. condition-mean state geometry over the full trial, mean ± SEM — the clean 2-D set of
    states the system visits: sample-coded in the delay (choice≈0) → test-resolved corners.
Pooled dPCA, sample×choice plane, Expert correct trials."""
import matplotlib; matplotlib.use('Agg')
import sys, warnings, os
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from itertools import product
from scipy.ndimage import gaussian_filter
from src.pca.io import pkl_load
from src.plot.traj import add_arrows

# ── Style (Nature Neuroscience house style — matches the main figures) ──────────
sns.set_context('notebook')          # MUST come after importing src.common.plot_utils (sets "poster")
sns.set_style('ticks')
plt.rcParams.update({                 # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
DUM = 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca'
Z = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca')
y = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')
lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca')
i, j = lab.index('sample'), lab.index('sample:test')
m = ((y.laser == 0) & (y.learning == 'Expert') & (y.performance == 1)).to_numpy()
Z2 = Z[m][:, [i, j], :].astype(float); yc = y[m].reset_index(drop=True)
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}; LS = {0: '-', 1: '--'}
NAME = {(0, 0): 'AC', (0, 1): 'AD', (1, 0): 'BC', (1, 1): 'BD'}
Wd = np.arange(21, 54); L = 3.0
fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.0))

# ── A: single-trial delay occupancy ──────────────────────────────────────────────
px = Z2[:, 0, Wd].ravel(); py = Z2[:, 1, Wd].ravel()
e = np.linspace(-L, L, 41); c = 0.5 * (e[:-1] + e[1:])
H, _, _ = np.histogram2d(px, py, bins=[e, e]); H = gaussian_filter(H, 1.2)
GX, GY = np.meshgrid(c, c)
hm = ax[0].pcolormesh(GX, GY, H.T / H.max(), cmap='mako' if 'mako' in plt.colormaps() else 'magma',
                      shading='auto')
for s in (0, 1):                                                  # condition-mean A/B delay states
    w = Z2[(yc['sample'] == s).to_numpy()][:, :, Wd].mean(0).mean(1)
    ax[0].scatter(*w, s=130, marker='o', color=SAMPLE_COL[s], edgecolor='w', lw=1.5, zorder=5)
    ax[0].annotate(f'  sample {"AB"[s]} mean', w, color='w', fontsize=6.5, va='center')
ax[0].set_xlim(-L, L); ax[0].set_ylim(-L, L); ax[0].set_aspect('equal')
ax[0].set_xlabel('sample axis'); ax[0].set_ylabel('choice axis')
ax[0].set_title('A  single-trial delay occupancy\n(one 2-D blob — A/B wells are sub-noise)', loc='left', fontsize=TITLE_FS)
fig.colorbar(hm, ax=ax[0], shrink=0.8, label='state density (norm.)')

# ── B: condition-mean geometry over the full trial, ± SEM ────────────────────────
halo = [pe.withStroke(linewidth=3.0, foreground='white')]
EPO = {5: 'base', 50: 'delay', 64: 'test'}
for s, t in product([0, 1], [0, 1]):
    sel = ((yc['sample'] == s) & (yc['test'] == t)).to_numpy(); n = sel.sum()
    mu = Z2[sel].mean(0); sem = Z2[sel].std(0) / np.sqrt(n)
    ax[1].plot(mu[0], mu[1], color=SAMPLE_COL[s], ls=LS[t], lw=2.2, zorder=5,
               path_effects=halo, solid_capstyle='round')
    add_arrows(ax[1], mu[0], mu[1], SAMPLE_COL[s], n_arrows=3)
    for b in EPO:                                                 # ±SEM ellipses at key epochs
        ax[1].add_patch(Ellipse((mu[0, b], mu[1, b]), 4 * sem[0, b], 4 * sem[1, b],
                                color=SAMPLE_COL[s], alpha=0.5, lw=0, zorder=6))
    ax[1].scatter(mu[0, 5], mu[1, 5], s=40, color='k', zorder=7)
ax[1].axhline(0, color='0.85', lw=.6); ax[1].axvline(0, color='0.85', lw=.6)
ax[1].set_xlim(-L, L); ax[1].set_ylim(-L, L); ax[1].set_aspect('equal')
ax[1].set_xlabel('sample axis'); ax[1].set_ylabel('choice axis')
ax[1].set_title('B  condition-mean geometry, full trial (mean ± 2 SEM)\n'
                'delay: sample-coded @ choice≈0  →  test: choice-resolved corners', loc='left', fontsize=TITLE_FS)
leg = [Line2D([], [], color=SAMPLE_COL[0], lw=2.4, label='sample A'),
       Line2D([], [], color=SAMPLE_COL[1], lw=2.4, label='sample B'),
       Line2D([], [], color='.3', lw=2, ls='-', label='test C'),
       Line2D([], [], color='.3', lw=2, ls='--', label='test D')]
ax[1].legend(handles=leg, fontsize=6.5, loc='upper left', framealpha=.9)
fig.suptitle('Working-memory manifold — measured (no fitted flow): single-trial blob vs condition-mean 2-D geometry', y=1.0, fontsize=9)
fig.tight_layout()
out = 'figures/pseudo/flow/wm_manifold_defensible.png'
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
