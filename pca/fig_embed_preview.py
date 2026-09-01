"""fig_embed_preview.py — Bernardi-style 3-D embedding of the 12 condition means (PREVIEW; user
routing 2026-09-01: "let's see how that looks and decide if main or supp").

Per stage × window (md | decision): per-neuron z-scaled condition means (correct laser-off,
per-mouse neuron fill), PCA to 3-D, conditions drawn as labelled vertices — colour = task
(DPA grey / Go blue / NoGo green outline), fill = sample A (indigo) / B (teal), edges join the
A–B pair within each task (the sample coding vector, per task: parallel edges = parallel codes).
Variance-explained printed per axis. NOT a stats panel — a geometry-at-a-glance display.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_embed_preview.py
Output: figures/pseudo/dimensionality/png/fig_embed_preview.png
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'svg.fonttype': 'none',
})
TITLE_FS = 8
SAMPC = {0: '#332288', 1: '#44AA99'}
TASKE = {'DPA': '0.35', 'DualGo': '#1f77b4', 'DualNoGo': '#2ca02c'}

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def cond_means(stage, wn):
    M = AW[wn]; R = np.zeros((12, N))
    for ci, (t, s, te) in enumerate(ALL12):
        for m in MICE:
            val = VALIDIX[(m, stage)]
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], 0)
    R = R - R.mean(0, keepdims=True)
    sd = R.std(0)
    return R / np.where(sd > 1e-9, sd, 1.0)


fig = plt.figure(figsize=(10.5, 9.5))
for r, stage in enumerate(['Naive', 'Expert']):
    for k, wn in enumerate(['md', 'decision']):
        R = cond_means(stage, wn)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        P = R @ Vt[:3].T
        ve = (S ** 2 / (S ** 2).sum())[:3]
        ax = fig.add_subplot(2, 2, r * 2 + k + 1, projection='3d')
        for t in ['DPA', 'DualGo', 'DualNoGo']:
            for te in (0, 1):
                i0 = ALL12.index((t, 0, te)); i1 = ALL12.index((t, 1, te))
                ax.plot(*zip(P[i0], P[i1]), color=TASKE[t], lw=1.2, alpha=0.8)
        for ci, (t, s, te) in enumerate(ALL12):
            ax.scatter(*P[ci], s=55, c=SAMPC[s], edgecolors=TASKE[t], linewidths=1.6,
                       marker='o' if t == 'DPA' else ('^' if t == 'DualGo' else 's'),
                       depthshade=False)
        ax.set_title(f'{stage} · {"mid-delay" if wn == "md" else "decision"}   '
                     f'(var {100 * ve[0]:.0f}/{100 * ve[1]:.0f}/{100 * ve[2]:.0f}%)',
                     loc='left', fontsize=TITLE_FS)
        ax.set_xlabel('PC1', fontsize=7, labelpad=-4); ax.set_ylabel('PC2', fontsize=7, labelpad=-4)
        ax.set_zlabel('PC3', fontsize=7, labelpad=-4)
        ax.tick_params(labelsize=5, pad=-2)
        ax.view_init(elev=18, azim=-60)
        print(f'{stage:6s} {wn:9s} var3 {np.round(ve, 2)}')
fig.suptitle('12 condition means, per-neuron z-scaled, PCA-3D — colour=sample, edge/marker=task; '
             'edges join A–B within task (parallel edges = parallel sample codes)', fontsize=8, y=0.99)
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/fig_embed_preview.png', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_embed_preview.png'))
