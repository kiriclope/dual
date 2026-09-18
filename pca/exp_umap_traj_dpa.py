"""exp_umap_traj_dpa.py — UMAP trajectories on DPA trials, refined settings + the naive/expert panel."""
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
SAMPC = {0: '#332288', 1: '#44AA99'}; MATCHC = {True: '#4daf4a', False: '#377eb8'}
EVENTS = [(12, 'sample', '#332288'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]
NB = 84; SM = 5
CMBIN = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))['CMBIN']

def prep(stage):
    sel = [ALL12.index(c) for c in DPA]
    CM = np.asarray(CMBIN[stage], float)[sel]; k = np.ones(SM) / SM
    CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM); CM = CM - CM.mean(0, keepdims=True)
    Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0
    return PCA(50, random_state=0).fit_transform((Z - Z.mean(0)) / sd)

def draw(ax, E, cf, ttl, lw=1.3):
    for ci, cd in enumerate(DPA):
        tr = E[ci * NB:(ci + 1) * NB]
        ax.plot(tr[:, 0], tr[:, 1], '-', color=cf(cd), lw=lw, alpha=0.9, zorder=2)
        ax.scatter(tr[0, 0], tr[0, 1], s=18, color=cf(cd), marker='o', lw=0, zorder=4)
        for b, _, col in EVENTS: ax.scatter(tr[b, 0], tr[b, 1], s=14, color=col, marker='s', lw=0, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]; ax.set_title(ttl, loc='left', fontsize=7)

P = prep('Expert'); NN = [80, 120, 160]; MD = [0.0, 0.05, 0.15]
fig, axs = plt.subplots(len(MD), len(NN), figsize=(3.0 * len(NN), 3.0 * len(MD)))
for r, md in enumerate(MD):
    for c, nn in enumerate(NN):
        E = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=md, metric='euclidean', random_state=0).fit_transform(P)
        draw(axs[r, c], E, lambda cd: SAMPC[cd[1]], f'nn {nn}, min_dist {md}')
        print(f'expert nn={nn} md={md}', flush=True)
fig.suptitle('UMAP trajectories, DPA trials, expert, CI removed — refined settings (colour = sample)', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('figures/pseudo/dimensionality/png/umap_traj_dpa_refine.png', bbox_inches='tight'); plt.close(fig)

NNF, MDF = 120, 0.05
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.8))
for r, stage in enumerate(['Naive', 'Expert']):
    E = umap.UMAP(n_components=2, n_neighbors=NNF, min_dist=MDF, metric='euclidean', random_state=0).fit_transform(prep(stage))
    for c, (nm, cf) in enumerate([('sample', lambda cd: SAMPC[cd[1]]), ('match (choice)', lambda cd: MATCHC[cd[1] == cd[2]])]):
        draw(axs[r, c], E, cf, f'{"naïve" if stage == "Naive" else "expert"} · by {nm}')
hs = [mlines.Line2D([0], [0], color=SAMPC[s], label=f'sample {"A" if s == 0 else "B"}') for s in SAMPC] + \
     [mlines.Line2D([0], [0], color=MATCHC[m], label='match (lick)' if m else 'nonmatch') for m in MATCHC] + \
     [mlines.Line2D([0], [0], marker='s', ls='none', color=c, label=n) for _, n, c in EVENTS]
fig.legend(handles=hs, loc='lower center', ncol=4, frameon=False, fontsize=6.5)
fig.suptitle(f'UMAP trajectories, DPA trials (4 condition means x 84 bins, CI removed; n_neighbors {NNF}, min_dist {MDF})', fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig('figures/pseudo/dimensionality/png/umap_traj_dpa_final.png', bbox_inches='tight'); print('saved')
