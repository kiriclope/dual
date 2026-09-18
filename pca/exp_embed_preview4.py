"""exp_embed_preview4.py — pooled pseudo-trials WITHOUT trial reuse (each real trial of each mouse enters at most one
pseudo-trial per map; KP = smallest pool), t-SNE per (stage x window), kNN purity vs shuffle; plus a joint naive+expert
map with per-stage per-neuron centering. Also: k-trial AVERAGED pseudo-trials (disjoint groups of 3) as the middle ground.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns, matplotlib.pyplot as plt
exec(open('exp_embed_preview.py').read().split('crow = np.repeat')[0].split('import matplotlib.lines as mlines')[1])

def knn_purity(E, lab, k=7, nshuf=200, rng=np.random.RandomState(0)):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E); idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    obs = np.mean(lab[idx] == lab[:, None]); null = []
    for _ in range(nshuf):
        pl = lab[rng.permutation(len(lab))]; null.append(np.mean(pl[idx] == pl[:, None]))
    return float(obs), float(np.mean(null))

def pools(stage, wn):
    out = {}
    for ci, (t, s, te) in enumerate(ALL12):
        for m in MICE:
            out[(ci, m)] = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
    return out

def cloud_disjoint(stage, wn, rng, group=1):
    """KP pseudo-trials per condition; pseudo-trial j of mouse m uses its j-th (group of) permuted trials — no reuse."""
    M = AW[wn]; sd = neuron_scale(stage, M); P = pools(stage, wn)
    kp = min(len(P[k]) // group for k in P if len(P[k]) > 0)
    X = np.zeros((12 * kp, N)); crow = np.repeat(np.arange(12), kp)
    for ci in range(12):
        for m in MICE:
            val = VALIDIX[(m, stage)]; idx = rng.permutation(P[(ci, m)])
            if not len(idx): continue
            cm = np.nanmean(M[np.ix_(idx, val)], 0)
            for j in range(kp):
                pick = idx[j * group:(j + 1) * group]; blk = np.nanmean(M[np.ix_(pick, val)], 0); blk = np.where(np.isfinite(blk), blk, cm)
                X[ci * kp + j, val] = blk
    return X / sd[None, :], crow, kp

fig, axs = plt.subplots(4, 3, figsize=(7.5, 9.6)); rows = []
for r, (stage, wn) in enumerate([(s, w) for s in STAGES for w in ['md', 'decision']]):
    for group, tag in [(1, 'single'), (3, 'avg3')]:
        X, crow, kp = cloud_disjoint(stage, wn, np.random.RandomState(1), group)
        P = PCA(min(30, X.shape[0] - 1), random_state=0).fit_transform(X)
        E = TSNE(2, perplexity=max(5, min(30, kp * 12 // 8)), init='pca', random_state=0, learning_rate='auto').fit_transform(P)
        labs = {'task': np.array([ALL12[c][0] for c in crow]), 'sample': np.array([ALL12[c][1] for c in crow]), 'match': np.array([ALL12[c][1] == ALL12[c][2] for c in crow])}
        pu = {n: knn_purity(E, l) for n, l in labs.items()}
        print(f'{stage:6s} {wn:8s} {tag:6s} KP={kp:2d} n={len(E):3d}: ' + '  '.join(f'{n} {v[0]:.2f}/{v[1]:.2f}' for n, v in pu.items()), flush=True)
        if group == 1:
            for c, (nm, cmap) in enumerate([('task', TASKC), ('sample', SAMPC), ('match', MATCHC)]):
                ax = axs[r, c]; ax.scatter(E[:, 0], E[:, 1], s=9, c=[cmap[v] for v in labs[nm]], alpha=0.7, lw=0)
                ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
                ax.set_title(f'{"naïve" if stage=="Naive" else "expert"} · {"mid-delay" if wn=="md" else "decision"} · {nm}  purity {pu[nm][0]:.2f} (null {pu[nm][1]:.2f}; KP {kp})', loc='left', fontsize=6.2)
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/embed_preview4_disjoint.png', bbox_inches='tight'); print('saved disjoint')
# joint naive+expert, no reuse, per-stage centering
fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9))
Xn, cn, kn = cloud_disjoint('Naive', 'md', np.random.RandomState(1)); Xe, ce, ke = cloud_disjoint('Expert', 'md', np.random.RandomState(2))
Xn = Xn - Xn.mean(0); Xe = Xe - Xe.mean(0); X = np.vstack([Xn, Xe]); stage = np.r_[np.zeros(len(Xn), int), np.ones(len(Xe), int)]
P = PCA(30, random_state=0).fit_transform(X); E = TSNE(2, perplexity=30, init='pca', random_state=0, learning_rate='auto').fit_transform(P)
L = {'stage': stage, 'sample': np.r_[[ALL12[c][1] for c in cn], [ALL12[c][1] for c in ce]], 'task': np.r_[[ALL12[c][0] for c in cn], [ALL12[c][0] for c in ce]]}
C = {'stage': {0: '0.6', 1: '#332288'}, 'sample': SAMPC, 'task': TASKC}
for c, nm in enumerate(['stage', 'task', 'sample']):
    ax = axs[c]; ax.scatter(E[:, 0], E[:, 1], s=9, c=[C[nm][v] for v in L[nm]], alpha=0.7, lw=0); pu = knn_purity(E, np.asarray(L[nm]))
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_title(f'joint mid-delay (centered per stage) · {nm}  purity {pu[0]:.2f} (null {pu[1]:.2f})', loc='left', fontsize=6.2); print(f'joint md {nm}: {pu[0]:.2f} / {pu[1]:.2f}')
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/embed_preview4_joint.png', bbox_inches='tight'); print('saved joint')
