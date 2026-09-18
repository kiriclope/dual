"""exp_embed_preview7.py — trajectories/snapshots in a METRIC unsupervised map (the nonlinear embedding of trajectories
starbursts: each condition's own temporal chain is the strongest local structure, so t-SNE/UMAP give one filament per
condition and destroy the between-condition geometry).

Design: at each of a sequence of windows through the trial, compute the 12x12 condition RDM (correlation distance of the
condition means, neurons z-scored), embed it in 2-D by metric MDS, and PROCRUSTES-align each window's map to the previous
one (rotation/reflection/scale only, which MDS leaves free). The result is ONE common geometry space in which
  - a SNAPSHOT is the 12 conditions at one window (Fig. 3b's display, but with no axes chosen), and
  - a TRAJECTORY is one condition's path across windows.
Everything is unsupervised: the map is built from the distances between conditions, never from a decoder axis.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'svg.fonttype': 'none'})
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
TASKC = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
SAMPC = {0: '#332288', 1: '#44AA99'}; SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); CMBIN = RES['CMBIN']
SM = 5; STEP = 3; WIDTH = 6
WSTART = list(range(3, 67, STEP))                                   # rolling 1-s windows through the trial
NAMED = [(6, 'baseline'), (12, 'sample'), (21, 'early delay'), (33, 'mid-delay'), (45, 'late delay'), (54, 'decision')]

def cond_means(stage):
    CM = np.asarray(CMBIN[stage], float); k = np.ones(SM) / SM
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)      # (12, N, 84)

def rdm_at(CM, a, b):
    M = CM[:, :, a:b].mean(2); M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    return squareform(pdist(np.nan_to_num(M), 'correlation'))

def aligned_maps(stage):
    CM = cond_means(stage); out = []; prev = None
    for a in WSTART:
        D = rdm_at(CM, a, a + WIDTH)
        E = MDS(2, dissimilarity='precomputed', random_state=0, n_init=12, normalized_stress='auto').fit_transform(D)
        E = E - E.mean(0)
        if prev is not None:
            R, sc = orthogonal_procrustes(E, prev); E = E @ R                        # rotation/reflection only
        prev = E; out.append(E)
    return np.array(out)                                                             # (nwin, 12, 2)

fig = plt.figure(figsize=(10.5, 7.2))
gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1.0], hspace=0.3, wspace=0.25)
for r, stage in enumerate(['Naive', 'Expert']):
    M = aligned_maps(stage)
    ax = fig.add_subplot(gs[0, 3 * r:3 * r + 3])
    for ci, cd in enumerate(ALL12):
        p = M[:, ci, :]
        ax.plot(p[:, 0], p[:, 1], '-', color=TASKC[cd[0]], lw=1.1, alpha=0.8, zorder=2)
        ax.scatter(*p[0], s=18, color=TASKC[cd[0]], marker='o', lw=0, zorder=4)
        ax.scatter(*p[-1], s=36, color=TASKC[cd[0]], marker='*', lw=0, zorder=5)
        for wi, (ws, nm) in enumerate(NAMED):
            k = int(np.argmin(np.abs(np.array(WSTART) - ws)))
            ax.scatter(*p[k], s=14, facecolor='w', edgecolor=TASKC[cd[0]], lw=0.7, zorder=3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_title(f'{"naïve" if stage == "Naive" else "expert"} · condition paths through the aligned geometry (o = trial start, ★ = trial end)', loc='left', fontsize=8)
    print(f'{stage}: {len(M)} windows aligned', flush=True)
    for c, (ws, nm) in enumerate(NAMED):
        if r == 1: continue
        k = int(np.argmin(np.abs(np.array(WSTART) - ws))); ax2 = fig.add_subplot(gs[1, c])
        Mn = aligned_maps('Naive') if False else M
        for ci, cd in enumerate(ALL12):
            ax2.scatter(*Mn[k, ci], s=42, color=TASKC[cd[0]], marker='o' if cd[1] == 0 else 's', edgecolors='k' if cd[1] == cd[2] else 'none', linewidths=0.8)
        ax2.set_aspect('equal'); ax2.set_xticks([]); ax2.set_yticks([]); [sp.set_visible(False) for sp in ax2.spines.values()]
        ax2.set_title(nm, loc='left', fontsize=7.5)
        lim = np.abs(M).max() * 1.1; ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
hs = [mlines.Line2D([0], [0], color=TASKC[t], label=SHORT[t]) for t in TASKC] + \
     [mlines.Line2D([0], [0], marker='o', ls='none', color='0.4', label='sample A'), mlines.Line2D([0], [0], marker='s', ls='none', color='0.4', label='sample B'),
      mlines.Line2D([0], [0], marker='o', ls='none', color='w', mec='k', label='match (lick)')]
fig.legend(handles=hs, loc='lower center', ncol=6, frameon=False, fontsize=6.5)
fig.suptitle('Metric unsupervised geometry: per-window MDS of the 12-condition RDM, Procrustes-aligned across windows (bottom row = naïve snapshots)', fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig('figures/pseudo/dimensionality/png/embed_preview7_mds.png', bbox_inches='tight'); print('saved mds paths')
