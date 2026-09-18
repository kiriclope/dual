"""exp_manifold_robust.py — is the sample structure in the DPA delay manifold robust to the embedding method, k, and the
pseudo-trial draw? (t-SNE with k=7 gave p ~ .05-.09; UMAP with k=10 gave p < .001.)"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import umap
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
WINS = [('md', 'mid-delay'), ('delay', 'late delay'), ('decision', 'decision')]
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

def pools(stage, conds):
    return {(ci, m): np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            for ci, (t, s, te) in enumerate(conds) for m in MICE}

def nscale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s_ = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s_) & (s_ > 1e-6), s_, 1.0)
    return sd

def cloud(stage, wn, conds, rng):
    M = AW[wn]; sd = nscale(stage, M); P = pools(stage, conds); kp = min(len(v) for v in P.values() if len(v))
    X = np.zeros((len(conds) * kp, N)); crow = np.repeat(np.arange(len(conds)), kp)
    for ci in range(len(conds)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; ix = rng.permutation(P[(ci, m)])
            if not len(ix): continue
            cm = np.nanmean(M[np.ix_(ix, val)], 0)
            for j in range(kp):
                v = M[ix[j], val]; X[ci * kp + j, val] = np.where(np.isfinite(v), v, cm)
    X = X / sd[None, :]
    return X - X.mean(0), crow

def purity(E, lab, k, nshuf=500, seed=0):
    k = min(k, len(E) - 1); rng = np.random.RandomState(seed)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    obs = float(np.mean(lab[idx] == lab[:, None])); null = np.array([np.mean(lab[rng.permutation(len(lab))][idx] == lab[rng.permutation(len(lab))][:, None]) for _ in range(nshuf)])
    return obs, float(null.mean()), float((null >= obs).mean())

DPA4 = SETS['DPA']
print('DPA delay/decision manifolds — sample purity across method x k x pseudo-trial draw (naive+expert joint map)')
for wn, wlab in WINS:
    for seed in [1, 7, 23]:
        Xn, cn = cloud('Naive', wn, DPA4, np.random.RandomState(seed)); Xe, ce = cloud('Expert', wn, DPA4, np.random.RandomState(seed + 100))
        X = np.vstack([Xn, Xe]); crow = np.r_[cn, ce]
        lab = np.array([DPA4[c][1] for c in crow]); labm = np.array([DPA4[c][1] == DPA4[c][2] for c in crow])
        Z = PCA(min(30, len(X) - 1), random_state=0).fit_transform(X)
        for meth in ['umap', 'tsne']:
            E = (umap.UMAP(n_components=2, n_neighbors=14, min_dist=0.3, random_state=0).fit_transform(Z) if meth == 'umap'
                 else TSNE(2, perplexity=15, init='pca', random_state=0, learning_rate='auto').fit_transform(Z))
            out = []
            for k in [5, 7, 10, 15]:
                o, nu, p = purity(E, lab, k); om, num, pm = purity(E, labm, k)
                out.append(f'k{k}: sample {o:.2f} p={p:.3f} | match {om:.2f} p={pm:.3f}')
            print(f'  {wlab:10s} draw {seed:2d} {meth:5s}  ' + '   '.join(out), flush=True)
