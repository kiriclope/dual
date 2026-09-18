"""exp_manifold_purity_draws.py — kNN purity averaged over pseudo-trial DRAWS (the honest statistic).
A single embedding of ~56-96 pseudo-trials is itself a random variable: the same data gives purity 0.48 (n.s.) on one
draw and 0.61 (p < .001) on another. Statistic = mean purity over NDRAW independent draws; null = the same average over
label-shuffled purities, paired draw by draw. Run for both trial sets, three windows, UMAP and t-SNE.
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import umap
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA'], 'all': ALL12}
WINS = [('md', 'mid-delay'), ('delay', 'late delay'), ('decision', 'decision')]
NDRAW, NSHUF, KNN = 20, 200, 7
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
SC = {(st, wn): nscale(st, AW[wn]) for st in ['Naive', 'Expert'] for wn, _ in WINS}
def cloud(stage, wn, conds, rng):
    M = AW[wn]; sd = SC[(stage, wn)]; P = pools(stage, conds); kp = min(len(v) for v in P.values() if len(v))
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

rows = []
for setname, conds in SETS.items():
    for wn, wlab in WINS:
        obs = {}; nul = {}
        for d in range(NDRAW):
            rng = np.random.RandomState(1000 + d)
            Xn, cn = cloud('Naive', wn, conds, rng); Xe, ce = cloud('Expert', wn, conds, np.random.RandomState(5000 + d))
            X = np.vstack([Xn, Xe]); crow = np.r_[cn, ce]; stg = np.r_[np.zeros(len(Xn), int), np.ones(len(Xe), int)]
            labs = {'sample': np.array([conds[c][1] for c in crow]), 'match': np.array([conds[c][1] == conds[c][2] for c in crow]), 'stage': stg}
            if setname == 'dual': labs['gng'] = np.array([conds[c][0] for c in crow])
            if setname == 'all': labs['task'] = np.array([conds[c][0] for c in crow])
            Z = PCA(min(30, len(X) - 1), random_state=0).fit_transform(X)
            for meth in ['umap', 'tsne']:
                E = (umap.UMAP(n_components=2, n_neighbors=min(30, len(X) // 4), min_dist=0.3, random_state=0).fit_transform(Z) if meth == 'umap'
                     else TSNE(2, perplexity=min(30, max(5, len(X) // 5)), init='pca', random_state=0, learning_rate='auto').fit_transform(Z))
                idx = NearestNeighbors(n_neighbors=KNN + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
                srng = np.random.RandomState(d)
                for v, lab in labs.items():
                    obs.setdefault((meth, v), []).append(np.mean(lab[idx] == lab[:, None]))
                    nul.setdefault((meth, v), []).append([np.mean(lab[srng.permutation(len(lab))][idx] == lab[srng.permutation(len(lab))][:, None]) for _ in range(NSHUF)])
        for (meth, v), o in obs.items():
            O = float(np.mean(o)); Ndist = np.mean(np.array(nul[(meth, v)]), axis=0)      # mean across draws, per shuffle index
            p = float((Ndist >= O).mean()); sd_draw = float(np.std(o, ddof=1))
            rows.append(dict(set=setname, win=wn, method=meth, var=v, purity=O, sd_draws=sd_draw, null=float(Ndist.mean()), p=p,
                             frac_sig=float(np.mean([x > np.percentile(np.array(nul[(meth, v)])[i], 97.5) for i, x in enumerate(o)]))))
            print(f'{setname:4s} {wlab:10s} {meth:5s} {v:7s}  purity {O:.3f} ± {sd_draw:.3f} (draw s.d.)  null {Ndist.mean():.3f}  p = {p:.4f}  [{100*rows[-1]["frac_sig"]:.0f}% of draws individually ∗]', flush=True)
D = pd.DataFrame(rows); pickle.dump(D, open('figures/pseudo/dimensionality/manifold_purity_draws.pkl', 'wb')); print('\nsaved')
