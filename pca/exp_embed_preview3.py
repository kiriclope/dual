"""exp_embed_preview3.py — per-MOUSE real-trial embeddings (no pseudo-trials): t-SNE of each mouse's own single trials
(all laser-off trials of a stage; neurons valid for that mouse) at mid-delay and decision, coloured by task / sample / match.
kNN purity (k = 15) with shuffle nulls per mouse x stage x window x variable — the honest version of the geometry display.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns, matplotlib.pyplot as plt
exec(open('exp_embed_preview.py').read().split('crow = np.repeat')[0].split('import matplotlib.lines as mlines')[1])

def knn_purity(E, lab, k=15, nshuf=200, rng=np.random.RandomState(0)):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E); idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    obs = np.mean(lab[idx] == lab[:, None]); null = []
    for _ in range(nshuf):
        pl = lab[rng.permutation(len(lab))]; null.append(np.mean(pl[idx] == pl[:, None]))
    return float(obs), float(np.mean(null)), float(np.percentile(null, 97.5))

def mouse_trials(m, stage, wn):
    val = VALIDIX[(m, stage)]; idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0))[0]
    X = AW[wn][np.ix_(idx, val)]; X = np.where(np.isfinite(X), X, np.nan); X = X[:, np.isfinite(X).all(0)]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, TSK[idx], SAMP[idx], (SAMP[idx] == TESTO[idx]), PERF[idx]

rows = []; EMBS = {}
for m in MICE:
    for stage in STAGES:
        for wn in ['md', 'decision']:
            X, t, s_, mt, pf = mouse_trials(m, stage, wn)
            P = PCA(min(30, X.shape[1] - 1), random_state=0).fit_transform(X)
            E = TSNE(2, perplexity=30, init='pca', random_state=0, learning_rate='auto').fit_transform(P); EMBS[(m, stage, wn)] = (E, t, s_, mt)
            for nm, lab in [('task', t), ('sample', s_), ('match', mt)]:
                o, nu, hi = knn_purity(E, lab); rows.append(dict(mouse=m, stage=stage, win=wn, var=nm, purity=o, null=nu, null975=hi, n=len(E), N=X.shape[1]))
    print(m, 'done', flush=True)
R = pd.DataFrame(rows); pickle.dump(dict(R=R, EMBS=EMBS), open('figures/pseudo/dimensionality/embed_permouse_preview.pkl', 'wb'))
print(R.groupby(['stage', 'win', 'var'])[['purity', 'null']].mean().round(2).to_string())
# figure: three largest mice, expert, md + decision, three colourings
SHOW = ['JawsM15', 'ChRM04', 'JawsM18']
fig, axs = plt.subplots(len(SHOW) * 2, 3, figsize=(7.2, 4.6 * len(SHOW)))
for i, m in enumerate(SHOW):
    for r, wn in enumerate(['md', 'decision']):
        E, t, s_, mt = EMBS[(m, 'Expert', wn)]
        for c, (nm, lab, cmap) in enumerate([('task', t, TASKC), ('sample', s_, SAMPC), ('match', mt, MATCHC)]):
            ax = axs[2 * i + r, c]; ax.scatter(E[:, 0], E[:, 1], s=6, c=[cmap[v] for v in lab], alpha=0.6, lw=0)
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            pu = R[(R.mouse == m) & (R.stage == 'Expert') & (R.win == wn) & (R['var'] == nm)].iloc[0]
            ax.set_title(f'{m} expert · {"mid-delay" if wn == "md" else "decision"} · {nm}  purity {pu.purity:.2f} (null {pu.null:.2f}, n = {pu.n} trials, {pu.N} neurons)', loc='left', fontsize=6.2)
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/embed_preview3_permouse.png', bbox_inches='tight'); print('saved permouse')
