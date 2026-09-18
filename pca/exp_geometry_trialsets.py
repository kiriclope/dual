"""exp_geometry_trialsets.py — the geometry analysis of fig_geometry_main.py run WITHIN each trial set (Leon 2026-09-18):
DPA trials only (4 conditions = sample x test) and dual trials only (8 conditions = GNG x sample x test).
Within DPA the task-context variable is gone, so an unsupervised map can only be structured by the memory or the choice.
Same construction as the main figure: pseudo-trials with NO trial reuse, naive+expert in one t-SNE per window, kNN purity
(k = 7) against label-shuffle nulls, and PCA trajectories of the condition means (condition-independent ramp removed).
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'svg.fonttype': 'none'})
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
SAMPC = {0: '#332288', 1: '#44AA99'}; MATCHC = {True: '#4daf4a', False: '#377eb8'}; STAGEC = {0: '0.62', 1: '#332288'}
GNGC = {'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
K_NN = 7; NSHUF = 500
SETS = {'DPA': [('DPA', s, te) for s in (0, 1) for te in (0, 1)],
        'dual': [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]}
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr): s = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd

def cloud(stage, wn, conds, rng):
    M = AW[wn]; sd = neuron_scale(stage, M)
    P = {(ci, m): np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
         for ci, (t, s, te) in enumerate(conds) for m in MICE}
    kp = min(len(v) for v in P.values() if len(v))
    X = np.zeros((len(conds) * kp, N)); crow = np.repeat(np.arange(len(conds)), kp)
    for ci in range(len(conds)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; idx = rng.permutation(P[(ci, m)])
            if not len(idx): continue
            cm = np.nanmean(M[np.ix_(idx, val)], 0)
            for j in range(kp):
                v = M[idx[j], val]; X[ci * kp + j, val] = np.where(np.isfinite(v), v, cm)
    X = X / sd[None, :]
    return X - X.mean(0), crow, kp

def knn_purity(E, lab, k=K_NN, nshuf=NSHUF, seed=0):
    k = min(k, len(E) - 1); rng = np.random.RandomState(seed)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    obs = float(np.mean(lab[idx] == lab[:, None])); null = np.empty(nshuf)
    for i in range(nshuf):
        pl = lab[rng.permutation(len(lab))]; null[i] = np.mean(pl[idx] == pl[:, None])
    return obs, float(null.mean()), float(np.percentile(null, 97.5)), float((null >= obs).mean())

OUT = {}
fig, axs = plt.subplots(4, 4, figsize=(11.0, 11.0))
for R, (setname, conds) in enumerate(SETS.items()):
    for r, wn in enumerate(['md', 'decision']):
        Xn, cn, kn = cloud('Naive', wn, conds, np.random.RandomState(1)); Xe, ce, ke = cloud('Expert', wn, conds, np.random.RandomState(2))
        X = np.vstack([Xn, Xe]); crow = np.r_[cn, ce]; stage = np.r_[np.zeros(len(Xn), int), np.ones(len(Xe), int)]
        Z = PCA(min(30, X.shape[0] - 1), random_state=0).fit_transform(X)
        E = TSNE(2, perplexity=min(30, max(5, len(X) // 5)), init='pca', random_state=0, learning_rate='auto').fit_transform(Z)
        labs = {'sample': np.array([conds[c][1] for c in crow]), 'match': np.array([conds[c][1] == conds[c][2] for c in crow]), 'stage': stage}
        if setname == 'dual': labs['gng'] = np.array([conds[c][0] for c in crow])
        pu = {nm: knn_purity(E, lab) for nm, lab in labs.items()}
        OUT[(setname, wn)] = dict(purity=pu, n=len(E), kp=(kn, ke))
        print(f'{setname:4s} {wn:8s}: n={len(E):3d} (kp {kn}/{ke})  ' + '  '.join(f'{nm} {v[0]:.2f} (null {v[1]:.2f}, p {v[3]:.3f})' for nm, v in pu.items()), flush=True)
        order = ['gng', 'sample', 'match', 'stage'] if setname == 'dual' else ['sample', 'match', 'stage']
        for c, nm in enumerate(order):
            ax = axs[2 * R + r, c]; cmap = {'sample': SAMPC, 'match': MATCHC, 'stage': STAGEC, 'gng': GNGC}[nm]
            ax.scatter(E[:, 0], E[:, 1], s=16, c=[cmap[v] for v in labs[nm]], alpha=0.8, lw=0)
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            ax.set_title(f'{setname} · {"mid-delay" if wn == "md" else "decision"} · {nm}  purity {pu[nm][0]:.2f} (null {pu[nm][1]:.2f})', loc='left', fontsize=6.8)
        if setname == 'DPA': axs[2 * R + r, 3].axis('off')
fig.tight_layout(); fig.savefig('figures/pseudo/dimensionality/png/geometry_trialsets_maps.png', bbox_inches='tight'); plt.close(fig)

# ── trajectories within each trial set (PCA of the set's condition means, CI removed) ──
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb')); CMBIN = RES['CMBIN']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
NB = 84; SM = 5; EVENTS = [(12, 'sample', '#332288'), (27, 'GNG', '#cc3311'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]
fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.8))
for R, (setname, conds) in enumerate(SETS.items()):
    sel = [ALL12.index(cd) for cd in conds]
    for c, stage in enumerate(STAGES):
        CM = np.asarray(CMBIN[stage], float)[sel]; k = np.ones(SM) / SM
        CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM); CM = CM - CM.mean(0, keepdims=True)
        Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0; Z = (Z - Z.mean(0)) / sd
        pc = PCA(4, random_state=0); Y = pc.fit_transform(Z).reshape(len(conds), NB, 4); ev = pc.explained_variance_ratio_
        for cc, (nm, cf) in enumerate([('sample', lambda cd: SAMPC[cd[1]]), ('match', lambda cd: MATCHC[cd[1] == cd[2]])]):
            ax = axs[R, 2 * c + cc]
            for ci, cd in enumerate(conds):
                tr = Y[ci]; ax.plot(tr[:, 0], tr[:, 1], '-', color=cf(cd), lw=1.2, alpha=0.85, zorder=2)
                ax.scatter(tr[0, 0], tr[0, 1], s=16, color=cf(cd), marker='o', lw=0, zorder=4)
                for b, _, col in EVENTS: ax.scatter(tr[b, 0], tr[b, 1], s=13, color=col, marker='s', lw=0, zorder=5)
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            ax.set_title(f'{setname} · {"naïve" if stage == "Naive" else "expert"} · by {nm}  (PC1 {100*ev[0]:.0f}%, PC2 {100*ev[1]:.0f}%)', loc='left', fontsize=6.8)
        print(f'{setname:4s} {stage:6s} trajectories: EV ' + ' '.join(f'{100*v:.0f}%' for v in ev), flush=True)
fig.suptitle('Trajectories within each trial set (PCA of that set\'s condition means, CI removed); squares = sample/GNG/cue/test onsets', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig('figures/pseudo/dimensionality/png/geometry_trialsets_traj.png', bbox_inches='tight')
pickle.dump(OUT, open('figures/pseudo/dimensionality/geometry_trialsets.pkl', 'wb')); print('saved')
