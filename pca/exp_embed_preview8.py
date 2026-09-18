"""exp_embed_preview8.py — unsupervised METRIC trajectories: PCA of the condition-mean state space (12 conds x 84 bins),
raw and with the condition-independent ramp removed; expert and naive. No decoder axes, distances meaningful, curves smooth.
Also the variance explained, and the same map used for snapshots (states at named windows).
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'svg.fonttype': 'none'})
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
TASKC = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}; SAMPC = {0: '#332288', 1: '#44AA99'}; SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
EVENTS = [(12, 'sample', '#332288'), (27, 'GNG', '#cc3311'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]
NB = 84; SM = 5
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); CMBIN = RES['CMBIN']

def proj(stage, ci_removed, pcs=(0, 1)):
    CM = np.asarray(CMBIN[stage], float); k = np.ones(SM) / SM
    CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)
    if ci_removed: CM = CM - CM.mean(0, keepdims=True)
    Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0; Z = (Z - Z.mean(0)) / sd
    p = PCA(6, random_state=0); Y = p.fit_transform(Z)
    return Y.reshape(12, NB, 6), p.explained_variance_ratio_

fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.8))
for r, stage in enumerate(['Naive', 'Expert']):
    for c, (cir, pcs) in enumerate([(False, (0, 1)), (True, (0, 1)), (True, (1, 2)), (True, (0, 2))]):
        Y, ev = proj(stage, cir); ax = axs[r, c]; i, j = pcs
        for ci, cd in enumerate(ALL12):
            tr = Y[ci]; ax.plot(tr[:, i], tr[:, j], '-', color=TASKC[cd[0]], lw=1.1, alpha=0.8, zorder=2)
            ax.scatter(tr[0, i], tr[0, j], s=16, color=TASKC[cd[0]], marker='o', lw=0, zorder=4)
            for b, nm, col in EVENTS: ax.scatter(tr[b, i], tr[b, j], s=13, color=col, marker='s', lw=0, zorder=5)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(f'{"naïve" if stage == "Naive" else "expert"} · {"CI removed" if cir else "raw"} · PC{i+1} vs PC{j+1} ({100*ev[i]:.0f}%, {100*ev[j]:.0f}%)', loc='left', fontsize=7.5)
        if c == 0: print(f'{stage} raw EV: ' + ' '.join(f'{100*v:.0f}%' for v in ev), flush=True)
        if c == 1: print(f'{stage} CI-removed EV: ' + ' '.join(f'{100*v:.0f}%' for v in ev), flush=True)
hs = [mlines.Line2D([0], [0], color=TASKC[t], label=SHORT[t]) for t in TASKC] + [mlines.Line2D([0], [0], marker='s', ls='none', color=c, label=n) for _, n, c in EVENTS]
fig.legend(handles=hs, loc='lower center', ncol=7, frameon=False, fontsize=6.5)
fig.suptitle('Unsupervised metric trajectories: PCA of the condition-mean state space (no decoder axes)', fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig('figures/pseudo/dimensionality/png/embed_preview8_pca.png', bbox_inches='tight'); print('saved pca traj')
