"""exp_embed_preview5.py — preview of panel c for the geometry figure: CONDITION-MEAN geometry.
 (1) pooled split-half condition means: 30 random halvings x 2 halves x 12 conditions -> t-SNE clouds per (stage x window);
 (2) per-MOUSE representational geometry: each mouse's 12-condition RDM (correlation distance of z-scored condition means),
     its split-half reliability (noise ceiling), the across-mouse consistency (correlation of each mouse's RDM with the
     leave-one-out mean RDM), and a 2-D MDS of the mean RDM (a metric map of the condition-mean geometry, shared across mice).
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns, matplotlib.pyplot as plt
exec(open('exp_embed_preview.py').read().split('crow = np.repeat')[0].split('import matplotlib.lines as mlines')[1])
SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
def cname(cd): return f'{SHORT[cd[0]]} {"A" if cd[1]==0 else "B"}{"m" if cd[1]==cd[2] else "n"}'

def pools(stage):
    return {(ci, m): np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            for ci, (t, s, te) in enumerate(ALL12) for m in MICE}

# ── (1) pooled split-half condition-mean clouds ──
NS = 30
fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.2))
for r, stage in enumerate(STAGES):
    P = pools(stage)
    for c, wn in enumerate(['md', 'decision']):
        M = AW[wn]; sd = neuron_scale(stage, M); pts = []; lab = []
        rng = np.random.RandomState(0)
        for sp in range(NS):
            halves = {k: (lambda p: (p[:len(p)//2], p[len(p)//2:]))(rng.permutation(v)) for k, v in P.items()}
            for h in (0, 1):
                for ci in range(12):
                    x = np.zeros(N)
                    for m in MICE:
                        idx = halves[(ci, m)][h]; val = VALIDIX[(m, stage)]
                        if len(idx): x[val] = np.nanmean(M[np.ix_(idx, val)], 0)
                    pts.append(x / sd); lab.append(ci)
        X = np.nan_to_num(np.array(pts)); lab = np.array(lab)
        Z = PCA(30, random_state=0).fit_transform(X); E = TSNE(2, perplexity=30, init='pca', random_state=0, learning_rate='auto').fit_transform(Z)
        ax = axs[r, c]
        for ci, cd in enumerate(ALL12):
            sel = lab == ci; ax.scatter(E[sel, 0], E[sel, 1], s=7, color=TASKC[cd[0]], marker='o' if cd[1] == 0 else 's', alpha=0.6, lw=0)
            ax.text(E[sel, 0].mean(), E[sel, 1].mean(), cname(cd), fontsize=5, ha='center', va='center', color='k')
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        ax.set_title(f'{"naïve" if stage=="Naive" else "expert"} · {"mid-delay" if wn=="md" else "decision"} · split-half condition means (o = A, □ = B; m/n = match/nonmatch)', loc='left', fontsize=6)
        print(f'{stage} {wn}: cloud embedded', flush=True)
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/embed_preview5_means.png', bbox_inches='tight'); plt.close(fig)

# ── (2) per-mouse RDMs, reliability, consistency, MDS of the mean RDM ──
def rdm_of(m, stage, wn, idxsel=None, rng=None):
    val = VALIDIX[(m, stage)]; M = AW[wn]; P = pools(stage); means = []
    for ci in range(12):
        idx = P[(ci, m)]
        if idxsel is not None: idx = idxsel(idx, rng)
        means.append(np.nanmean(M[np.ix_(idx, val)], 0) if len(idx) else np.full(len(val), np.nan))
    Mm = np.array(means); Mm = (Mm - np.nanmean(Mm, 0)) / (np.nanstd(Mm, 0) + 1e-9); Mm = np.nan_to_num(Mm)
    return squareform(pdist(Mm, 'correlation'))
out = {}
fig, axs = plt.subplots(2, 4, figsize=(11, 5.6))
for r, stage in enumerate(STAGES):
    for c, wn in enumerate(['md', 'decision']):
        R = {m: rdm_of(m, stage, wn) for m in MICE}
        rel = []; cons = []
        for m in MICE:
            rng = np.random.RandomState(0)
            ra = rdm_of(m, stage, wn, lambda idx, g: g.permutation(idx)[:len(idx)//2], rng); rng = np.random.RandomState(0)
            rb = rdm_of(m, stage, wn, lambda idx, g: g.permutation(idx)[len(idx)//2:], rng)
            iu = np.triu_indices(12, 1); rel.append(spearmanr(ra[iu], rb[iu])[0])
            loo = np.mean([R[o] for o in MICE if o != m], 0); cons.append(spearmanr(R[m][iu], loo[iu])[0])
        meanR = np.mean([R[m] for m in MICE], 0); out[(stage, wn)] = dict(mean=meanR, rel=rel, cons=cons)
        print(f'{stage:6s} {wn:8s}: RDM split-half reliability median {np.median(rel):.2f} (range {min(rel):.2f}-{max(rel):.2f}); across-mouse consistency median {np.median(cons):.2f} ({min(cons):.2f}-{max(cons):.2f})', flush=True)
        ax = axs[r, 2 * c]; im = ax.imshow(meanR, cmap='viridis'); ax.set_xticks(range(12)); ax.set_yticks(range(12))
        ax.set_xticklabels([cname(cd) for cd in ALL12], rotation=90, fontsize=5); ax.set_yticklabels([cname(cd) for cd in ALL12], fontsize=5)
        ax.set_title(f'{"naïve" if stage=="Naive" else "expert"} · {"mid-delay" if wn=="md" else "decision"} · mean RDM over 9 mice', loc='left', fontsize=6.5)
        ax = axs[r, 2 * c + 1]; E = MDS(2, dissimilarity='precomputed', random_state=0, n_init=8).fit_transform(meanR)
        for ci, cd in enumerate(ALL12):
            ax.scatter(E[ci, 0], E[ci, 1], s=60, color=TASKC[cd[0]], marker='o' if cd[1] == 0 else 's', edgecolors='k' if cd[1] == cd[2] else 'none', linewidths=0.8)
            ax.text(E[ci, 0], E[ci, 1] + 0.02, cname(cd), fontsize=5.5, ha='center', va='bottom')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        ax.set_title(f'MDS of the mean RDM  (per-mouse reliability {np.median(rel):.2f}, consistency {np.median(cons):.2f})', loc='left', fontsize=6.5)
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/embed_preview5_rdm.png', bbox_inches='tight')
pickle.dump(out, open('figures/pseudo/dimensionality/embed_rdm_preview.pkl', 'wb')); print('saved rdm')
