"""exp_embed_preview.py — PREVIEW for the new geometry figure (between Figs 2 and 3, 2026-09-18): 2-D embeddings
of the population states, pure geometry (no axes drawn), UMAP and t-SNE side by side on the same points.

Points = window-averaged pseudo-trials (KP per condition; one random correct laser-off trial per mouse per pseudo-trial,
the decoder's construction; NaNs imputed from that mouse's condition mean), z-scored by the stage-level trial std,
PCA-30, then UMAP (n_neighbors 30, min_dist 0.3, cosine) or t-SNE (perplexity 30, PCA init). One embedding per
(stage x window), shown three times: coloured by task, by sample, by match (= future choice).
Trajectories = the 12 condition-mean trajectories (CMBIN cache, 5-bin smoothed), embedded with UMAP (n_neighbors 60).

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_embed_preview.py
Output: figures/pseudo/dimensionality/png/embed_preview_{umap,tsne,traj}.png
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
})
TITLE_FS = 8
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
KP = 40
TASKC = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
SAMPC = {0: '#332288', 1: '#44AA99'}
MATCHC = {True: '#4daf4a', False: '#377eb8'}
SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def pseudo_cloud(stage, wn, rng):
    M = AW[wn]; sd = neuron_scale(stage, M)
    X = np.zeros((len(ALL12) * KP, N))
    for ci, (t, s, te) in enumerate(ALL12):
        for m in MICE:
            val = VALIDIX[(m, stage)]
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if not len(idx):
                continue
            cm = np.nanmean(M[np.ix_(idx, val)], 0); block = M[np.ix_(rng.choice(idx, KP, replace=True), val)]
            bad = ~np.isfinite(block)
            if bad.any():
                block[bad] = np.broadcast_to(cm, block.shape)[bad]
            X[ci * KP:(ci + 1) * KP, val] = block
    return X / sd[None, :]


crow = np.repeat(np.arange(12), KP)
EMB = {}
for stage in STAGES:
    for wn in ['md', 'decision']:
        X = pseudo_cloud(stage, wn, np.random.RandomState(1)); P = PCA(30, random_state=0).fit_transform(X)
        EMB[('umap', stage, wn)] = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='cosine', random_state=0).fit_transform(P)
        EMB[('tsne', stage, wn)] = TSNE(2, perplexity=30, init='pca', random_state=0, learning_rate='auto').fit_transform(P)
        print(f'embedded {stage} {wn}', flush=True)

COLS = [('task', lambda cd: TASKC[cd[0]]), ('sample', lambda cd: SAMPC[cd[1]]), ('match (future choice)', lambda cd: MATCHC[cd[1] == cd[2]])]
ROWS = [(st, wn) for st in STAGES for wn in ['md', 'decision']]
for method in ['umap', 'tsne']:
    fig, axs = plt.subplots(4, 3, figsize=(7.5, 9.6))
    for r, (stage, wn) in enumerate(ROWS):
        E = EMB[(method, stage, wn)]
        for c_i, (ttl, cfun) in enumerate(COLS):
            ax = axs[r, c_i]; cols = [cfun(ALL12[c]) for c in crow]
            ax.scatter(E[:, 0], E[:, 1], s=5, c=cols, alpha=0.55, lw=0)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 0: ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
            if c_i == 0: ax.set_ylabel(f'{"naïve" if stage == "Naive" else "expert"} · {"mid-delay" if wn == "md" else "decision"}', fontsize=7.5)
    hs = [mlines.Line2D([0], [0], marker='o', ls='none', color=TASKC[t], label=SHORT[t]) for t in TASKC] + \
         [mlines.Line2D([0], [0], marker='o', ls='none', color=SAMPC[s], label=f'sample {"A" if s == 0 else "B"}') for s in SAMPC] + \
         [mlines.Line2D([0], [0], marker='o', ls='none', color=MATCHC[m], label='match (lick)' if m else 'nonmatch (no lick)') for m in MATCHC]
    fig.legend(handles=hs, loc='lower center', ncol=7, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f'{method.upper()} of window-averaged pseudo-trials ({KP} per condition; PCA-30 input; one embedding per row, three colourings)', fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    os.makedirs('figures/pseudo/dimensionality/png', exist_ok=True)
    fig.savefig(f'figures/pseudo/dimensionality/png/embed_preview_{method}.png', bbox_inches='tight'); plt.close(fig)
    print('saved', method)

# ── trajectories (condition means over time), UMAP ──
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); CMBIN = RES['CMBIN']; NB = 84; SM = 5
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.4))
for r, stage in enumerate(STAGES):
    CM = np.asarray(CMBIN[stage], dtype=np.float64); k = np.ones(SM) / SM
    CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)
    Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0; Z = (Z - Z.mean(0)) / sd
    P = PCA(50, random_state=0).fit_transform(Z)
    E = umap.UMAP(n_components=2, n_neighbors=60, min_dist=0.5, metric='cosine', random_state=0).fit_transform(P)
    for c_i, (ttl, cfun) in enumerate([('task', lambda cd: TASKC[cd[0]]), ('sample', lambda cd: SAMPC[cd[1]])]):
        ax = axs[r, c_i]
        for ci, cd in enumerate(ALL12):
            tr = E[ci * NB:(ci + 1) * NB]; ax.plot(tr[:, 0], tr[:, 1], '-', color=cfun(cd), lw=1.0, alpha=0.8)
            ax.scatter(tr[0, 0], tr[0, 1], s=14, color=cfun(cd), marker='o', zorder=3)          # trial start
            for b in (12, 27, 39, 54):                                                              # sample, GNG, cue, test onsets
                ax.scatter(tr[b, 0], tr[b, 1], s=10, color=cfun(cd), marker='s', zorder=3, lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f'{"naïve" if stage == "Naive" else "expert"} · trajectories by {ttl}', loc='left', fontsize=TITLE_FS)
    print(f'{stage}: trajectories embedded', flush=True)
fig.suptitle('UMAP of the 12 condition-mean trajectories (5-bin smoothed; dot = trial start, squares = sample / GNG / cue / test onsets)', fontsize=8, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig('figures/pseudo/dimensionality/png/embed_preview_traj.png', bbox_inches='tight'); print('saved traj')
