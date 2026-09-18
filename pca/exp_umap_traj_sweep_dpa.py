"""exp_umap_traj_sweep_dpa.py — the dual-trial parameter sweep, repeated on DPA trials (Leon 2026-09-18).
Same grid as umap_traj_sweep_dual.png: n_neighbors x min_dist, expert, condition-independent component removed.
DPA trials carry no Go/NoGo odor, so only the sample, cue and test onsets are marked.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
import umap, seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'axes.titlesize': 8, 'svg.fonttype': 'none'})
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
DPA = [c for c in ALL12 if c[0] == 'DPA']
SAMPC = {0: '#332288', 1: '#44AA99'}
EVENTS = [(12, 'sample', '#332288'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]   # no GNG odor on DPA trials
NB, SM = 84, 5
CMBIN = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))['CMBIN']
sel = [ALL12.index(c) for c in DPA]
CM = np.asarray(CMBIN['Expert'], float)[sel]; k = np.ones(SM) / SM
CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM); CM = CM - CM.mean(0, keepdims=True)
Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0
P = PCA(50, random_state=0).fit_transform((Z - Z.mean(0)) / sd)
NN, MD = [15, 50, 120, 250], [0.1, 0.6]
fig, axs = plt.subplots(len(MD), len(NN), figsize=(3.0 * len(NN), 3.0 * len(MD)))
for r, md in enumerate(MD):
    for c, nn in enumerate(NN):
        E = umap.UMAP(n_components=2, n_neighbors=min(nn, len(P) - 1), min_dist=md, metric='euclidean', random_state=0).fit_transform(P)
        ax = axs[r, c]
        for ci, cd in enumerate(DPA):
            tr = E[ci * NB:(ci + 1) * NB]; match = cd[1] == cd[2]
            ax.plot(tr[:, 0], tr[:, 1], '-' if match else '--', color=SAMPC[cd[1]], lw=1.1, alpha=0.9, zorder=2)
            ax.scatter(tr[0, 0], tr[0, 1], s=15, color=SAMPC[cd[1]], marker='o', lw=0, zorder=4)
            for b, _, col in EVENTS: ax.scatter(tr[b, 0], tr[b, 1], s=12, color=col, marker='s', lw=0, zorder=5)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(f'n_neighbors {nn}, min_dist {md}', loc='left', fontsize=7)
        print(f'DPA expert nn={nn} md={md}: done', flush=True)
hs = [mlines.Line2D([0], [0], color=SAMPC[s], label=f'sample {"A" if s == 0 else "B"}') for s in SAMPC] + \
     [mlines.Line2D([0], [0], color='0.4', ls='-', label='match'), mlines.Line2D([0], [0], color='0.4', ls='--', label='nonmatch')] + \
     [mlines.Line2D([0], [0], marker='s', ls='none', color=c, label=n) for _, n, c in EVENTS]
fig.legend(handles=hs, loc='lower center', ncol=7, frameon=False, fontsize=6.5)
fig.suptitle(f'UMAP trajectories, DPA trials, expert, CI removed (4 conditions x 84 bins = {len(P)} points; colour = sample, line = match/nonmatch)', fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig('figures/pseudo/dimensionality/png/umap_traj_sweep_DPA.png', bbox_inches='tight'); print('saved')
