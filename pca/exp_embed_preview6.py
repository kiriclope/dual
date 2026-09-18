"""exp_embed_preview6.py — TRAJECTORIES and SNAPSHOTS from ONE embedding (Leon 2026-09-18: "can we get trajectories? or
snapshots like in fig 3"). Substrate = CMBIN (12 condition means x N neurons x 84 bins), so time is fully resolved.

One embedding per (stage, variant) holds ALL 12 x 84 states, so trajectories are lines through it and snapshots are time
slices of the SAME map (positions comparable across windows — separate per-window embeddings would not be).
Variants: raw states (the condition-independent ramp dominates) vs CI-REMOVED (cross-condition mean subtracted per bin,
so only condition-specific geometry remains). Methods: UMAP, t-SNE, Isomap.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap, seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'svg.fonttype': 'none'})
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
TASKC = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
SAMPC = {0: '#332288', 1: '#44AA99'}; MATCHC = {True: '#4daf4a', False: '#377eb8'}
SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
NB = 84; SM = 5
EVENTS = [(12, 18, 'sample', '#332288'), (27, 33, 'GNG', '#cc3311'), (39, 42, 'cue', '#ee7733'), (54, 60, 'test', '#377eb8')]
WINS = [(0, 12, 'baseline'), (12, 18, 'sample'), (18, 33, 'early delay'), (33, 39, 'mid-delay'), (45, 54, 'late delay'), (54, 63, 'decision')]
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); CMBIN = RES['CMBIN']

def states(stage, ci_removed):
    CM = np.asarray(CMBIN[stage], float); k = np.ones(SM) / SM
    CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)
    if ci_removed:
        CM = CM - CM.mean(0, keepdims=True)                      # drop the condition-independent time ramp
    Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1])           # (12*84, N), condition-major
    sd = Z.std(0); sd[sd < 1e-9] = 1.0
    return (Z - Z.mean(0)) / sd

def embed(Z, method):
    P = PCA(50, random_state=0).fit_transform(Z)
    if method == 'umap': return umap.UMAP(n_components=2, n_neighbors=40, min_dist=0.35, metric='euclidean', random_state=0).fit_transform(P)
    if method == 'tsne': return TSNE(2, perplexity=60, init='pca', random_state=0, learning_rate='auto').fit_transform(P)
    return Isomap(n_neighbors=12, n_components=2).fit_transform(P)

# ── A. trajectories: methods x CI-removal, expert, coloured by task ──
fig, axs = plt.subplots(2, 3, figsize=(9.6, 6.4)); EM = {}
for r, cir in enumerate([False, True]):
    Z = states('Expert', cir)
    for c, meth in enumerate(['umap', 'tsne', 'isomap']):
        E = embed(Z, meth); EM[(cir, meth)] = E; ax = axs[r, c]
        for ci, cd in enumerate(ALL12):
            tr = E[ci * NB:(ci + 1) * NB]
            ax.plot(tr[:, 0], tr[:, 1], '-', color=TASKC[cd[0]], lw=1.0, alpha=0.75, zorder=2)
            ax.scatter(*tr[0], s=16, color=TASKC[cd[0]], marker='o', zorder=4, lw=0)
            for a, b, nm, col in EVENTS:
                ax.scatter(*tr[a], s=13, color=col, marker='s', zorder=5, lw=0)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(f'{meth.upper()} · {"CI removed" if cir else "raw states"}', loc='left', fontsize=8)
        print(f'expert {meth} ci_removed={cir}: embedded', flush=True)
hs = [mlines.Line2D([0], [0], color=TASKC[t], label=SHORT[t]) for t in TASKC] + [mlines.Line2D([0], [0], marker='s', ls='none', color=c, label=n) for _, _, n, c in EVENTS]
fig.legend(handles=hs, loc='lower center', ncol=7, frameon=False, fontsize=6.5)
fig.suptitle('Expert condition-mean trajectories (12 conditions x 84 bins, 5-bin smoothed); dot = trial start, squares = event onsets', fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig('figures/pseudo/dimensionality/png/embed_preview6_traj.png', bbox_inches='tight'); plt.close(fig)

# ── B. snapshots: time slices of ONE map (the CI-removed t-SNE), expert ──
E = EM[(True, 'tsne')]
fig, axs = plt.subplots(2, len(WINS), figsize=(2.0 * len(WINS), 4.4))
for r, (nm, cfun) in enumerate([('task', lambda cd: TASKC[cd[0]]), ('sample / choice', lambda cd: SAMPC[cd[1]])]):
    for c, (a, b, wl) in enumerate(WINS):
        ax = axs[r, c]; ax.scatter(E[:, 0], E[:, 1], s=2, color='0.9', lw=0, zorder=1)
        for ci, cd in enumerate(ALL12):
            seg = E[ci * NB + a:ci * NB + b]
            ax.plot(seg[:, 0], seg[:, 1], '-', color=cfun(cd), lw=1.6, alpha=0.9, zorder=3)
            ax.scatter(*seg.mean(0), s=26, color=cfun(cd), marker='o' if cd[1] == cd[2] else 'X', zorder=4, lw=0)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        if r == 0: ax.set_title(wl, loc='left', fontsize=7.5)
        if c == 0: ax.set_ylabel(f'by {nm}', fontsize=7.5)
fig.suptitle('Snapshots = time slices of ONE embedding (CI-removed t-SNE, expert); grey = the whole trial, coloured = the window; o = match, X = nonmatch', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('figures/pseudo/dimensionality/png/embed_preview6_snap.png', bbox_inches='tight'); print('saved traj + snap')
