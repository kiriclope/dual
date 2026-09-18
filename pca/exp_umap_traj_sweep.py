"""exp_umap_traj_sweep.py — UMAP trajectories, parameter sweep (Leon 2026-09-18: "I would like the UMAP trajectories").
The earlier starburst has a cause: with 84 time points per condition, a point's nearest neighbours are its own temporal
neighbours, so the UMAP graph splits into one chain per condition. The fix to test is n_neighbors LARGER than the
within-condition temporal neighbourhood, so the graph must connect conditions; also CI removal (the shared ramp) and
metric. Run per trial set: DPA (4 conditions x 84 bins) and dual (8 x 84).
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
import umap, seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'axes.titlesize': 8, 'svg.fonttype': 'none'})
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
SAMPC = {0: '#332288', 1: '#44AA99'}; GNGC = {'DualGo': '#023eff', 'DualNoGo': '#1ac938', 'DPA': '#e8000b'}
EVENTS = [(12, 'sample', '#332288'), (27, 'GNG', '#cc3311'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]
NB = 84; SM = 5
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); CMBIN = RES['CMBIN']

def prep(stage, conds, ci_removed=True):
    sel = [ALL12.index(c) for c in conds]
    CM = np.asarray(CMBIN[stage], float)[sel]; k = np.ones(SM) / SM
    CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)
    if ci_removed: CM = CM - CM.mean(0, keepdims=True)
    Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0
    return PCA(50, random_state=0).fit_transform((Z - Z.mean(0)) / sd)

def draw(ax, E, conds, cf, ttl):
    for ci, cd in enumerate(conds):
        tr = E[ci * NB:(ci + 1) * NB]
        ax.plot(tr[:, 0], tr[:, 1], '-', color=cf(cd), lw=1.1, alpha=0.85, zorder=2)
        ax.scatter(tr[0, 0], tr[0, 1], s=15, color=cf(cd), marker='o', lw=0, zorder=4)
        for b, _, col in EVENTS: ax.scatter(tr[b, 0], tr[b, 1], s=12, color=col, marker='s', lw=0, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_title(ttl, loc='left', fontsize=7)

NN = [15, 50, 120, 250]; MD = [0.1, 0.6]
for setname, conds in SETS.items():
    P = prep('Expert', conds)
    fig, axs = plt.subplots(len(MD), len(NN), figsize=(3.0 * len(NN), 3.0 * len(MD)))
    for r, md in enumerate(MD):
        for c, nn in enumerate(NN):
            nn_use = min(nn, len(P) - 1)
            E = umap.UMAP(n_components=2, n_neighbors=nn_use, min_dist=md, metric='euclidean', random_state=0).fit_transform(P)
            cf = (lambda cd: SAMPC[cd[1]]) if setname == 'DPA' else (lambda cd: GNGC[cd[0]])
            draw(axs[r, c], E, conds, cf, f'n_neighbors {nn_use}, min_dist {md}')
            print(f'{setname} expert nn={nn_use} md={md}: done', flush=True)
    fig.suptitle(f'UMAP trajectories, {setname} trials, expert, CI removed ({len(conds)} conditions x 84 bins = {len(P)} points; colour = ' + ('sample' if setname == 'DPA' else 'Go/NoGo') + ')', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f'figures/pseudo/dimensionality/png/umap_traj_sweep_{setname}.png', bbox_inches='tight'); plt.close(fig)
print('saved sweeps')
