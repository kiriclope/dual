"""exp_embed_preview2.py — second preview for the geometry figure: tuned UMAP vs t-SNE (expert), JOINT naive+expert maps
(same neurons, one embedding, coloured by stage / sample / match), and kNN label purity in the 2-D maps with shuffle nulls.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import umap, seaborn as sns, matplotlib.pyplot as plt
exec(open('exp_embed_preview.py').read().split('crow = np.repeat')[0].split('import matplotlib.lines as mlines')[1])   # reuse loaders/constants
KP = 40; crow = np.repeat(np.arange(12), KP); lab_task = np.array([ALL12[c][0] for c in crow]); lab_samp = np.array([ALL12[c][1] for c in crow]); lab_match = np.array([ALL12[c][1] == ALL12[c][2] for c in crow])

def knn_purity(E, lab, k=15, nshuf=200, rng=np.random.RandomState(0)):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E); idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    obs = np.mean(lab[idx] == lab[:, None]); null = [np.mean(lab[rng.permutation(len(lab))][idx] == lab[rng.permutation(len(lab))][:, None]) for _ in range(nshuf)]
    return obs, float(np.mean(null)), float(np.percentile(null, 97.5))

# ── 1. expert, tuned UMAP vs t-SNE, kNN purity ──
fig, axs = plt.subplots(2, 4, figsize=(10.5, 5.4))
for r, wn in enumerate(['md', 'decision']):
    X = pseudo_cloud('Expert', wn, np.random.RandomState(1)); P = PCA(30, random_state=0).fit_transform(X)
    Eu = umap.UMAP(n_components=2, n_neighbors=60, min_dist=0.9, metric='euclidean', random_state=0).fit_transform(P)
    Et = TSNE(2, perplexity=40, init='pca', random_state=0, learning_rate='auto').fit_transform(P)
    for c, (E, nm) in enumerate([(Eu, 'UMAP (nn 60, min_dist 0.9)'), (Et, 't-SNE (perp 40)')]):
        for k, (lab, cmap, ttl) in enumerate([(lab_samp, SAMPC, 'sample'), (lab_match, MATCHC, 'match')]):
            ax = axs[r, 2 * c + k]; ax.scatter(E[:, 0], E[:, 1], s=5, c=[cmap[v] for v in lab], alpha=0.55, lw=0)
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            pu = {n: knn_purity(E, l) for n, l in [('task', lab_task), ('sample', lab_samp), ('match', lab_match)]}
            ax.set_title(f'{nm} · {ttl}' if r == 0 else ttl, loc='left', fontsize=7.5)
            if k == 0: ax.set_ylabel(f'expert · {"mid-delay" if wn == "md" else "decision"}', fontsize=7.5)
            ax.text(0.01, 0.01, 'kNN purity ' + '  '.join(f'{n} {v[0]:.2f} (null {v[1]:.2f})' for n, v in pu.items()), transform=ax.transAxes, fontsize=5.2, color='0.3', va='bottom')
        print(f'{wn} {nm}: ' + '  '.join(f'{n} {v[0]:.2f} vs null {v[1]:.2f} (97.5% {v[2]:.2f})' for n, v in pu.items()), flush=True)
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/embed_preview2_tuned.png', bbox_inches='tight'); plt.close(fig); print('saved tuned')

# ── 2. joint naive + expert maps (same neuron scale per stage; one t-SNE) ──
fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.6))
for r, wn in enumerate(['md', 'decision']):
    Xn = pseudo_cloud('Naive', wn, np.random.RandomState(1)); Xe = pseudo_cloud('Expert', wn, np.random.RandomState(2))
    X = np.vstack([Xn, Xe]); stage = np.r_[np.zeros(len(Xn), int), np.ones(len(Xe), int)]
    P = PCA(30, random_state=0).fit_transform(X); E = TSNE(2, perplexity=40, init='pca', random_state=0, learning_rate='auto').fit_transform(P)
    L2 = {'stage': stage, 'sample': np.r_[lab_samp, lab_samp], 'match': np.r_[lab_match, lab_match]}
    C2 = {'stage': {0: '0.6', 1: '#332288'}, 'sample': SAMPC, 'match': MATCHC}
    for c, nm in enumerate(['stage', 'sample', 'match']):
        ax = axs[r, c]; ax.scatter(E[:, 0], E[:, 1], s=5, c=[C2[nm][v] for v in L2[nm]], alpha=0.55, lw=0)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        pu = knn_purity(E, L2[nm]); ax.set_title(f'joint map · {nm}  (kNN purity {pu[0]:.2f}, null {pu[1]:.2f})', loc='left', fontsize=7)
        if c == 0: ax.set_ylabel('mid-delay' if wn == 'md' else 'decision', fontsize=7.5)
        print(f'joint {wn} {nm}: purity {pu[0]:.2f} null {pu[1]:.2f} (97.5% {pu[2]:.2f})', flush=True)
fig.suptitle('One t-SNE per window on naïve + expert pseudo-trials together (grey = naïve, indigo = expert)', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig('figures/pseudo/dimensionality/png/embed_preview2_joint.png', bbox_inches='tight'); print('saved joint')
