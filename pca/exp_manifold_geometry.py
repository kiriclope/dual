"""exp_manifold_geometry.py — UMAP characterization of the DELAY and DECISION manifolds, naive vs expert, DPA vs dual
(Leon 2026-09-18: "use UMAP to characterize in detail the geometry of the delay manifold, and the geometry of the decision
manifold, and how these change between naive and expert. Then compare with the same manifolds on dual trials only").

Two levels, because a 2-D embedding is a display and the statistics have to be animal-level:

  DISPLAY  — one UMAP per (trial set x window) holding naive AND expert pseudo-trials, so the two stages live in the same
             map and can be compared directly (separate embeddings have arbitrary axes and are not comparable).
             Pseudo-trials use each real trial at most once (no reuse), as in fig_geometry_main.py.

  GEOMETRY — computed per MOUSE on its own simultaneously recorded neurons and its own real trials, then compared across
             the nine animals (paired Wilcoxon, naive vs expert and DPA vs dual):
               ID      intrinsic dimension of the trial cloud (Two-NN estimator, Facco et al. 2017, 10% trim)
               PR      participation ratio of the trial covariance (linear dimensionality)
               R       manifold radius: mean distance to the cloud centroid (per-neuron z units)
               SEP(v)  separation of a variable v: distance between its two condition centroids / mean within-condition
                       spread — a decoder-free, unit-free index of how far apart the variable's states sit
               PUR(v)  kNN label purity (k = 10) in that mouse's own 2-D UMAP, against a label-shuffle null
             v = sample and match (both sets) and gng (dual only).

Windows: mid-delay (bins 33-38, the memory window of Figs 2-3), late delay (45-53) and decision (54-62).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_manifold_geometry.py
Output: figures/pseudo/dimensionality/{png}/manifold_geometry_{maps,stats}.png; cache manifold_geometry.pkl
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon
import umap
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
WINS = [('md', 'mid-delay'), ('delay', 'late delay'), ('decision', 'decision')]
SAMPC = {0: '#332288', 1: '#44AA99'}; MATCHC = {True: '#4daf4a', False: '#377eb8'}; STAGEC = {0: '0.62', 1: '#332288'}
GNGC = {'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
UM = dict(n_components=2, min_dist=0.3, metric='euclidean', random_state=0)
KP_NN = 10; NSHUF = 200

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


# ── estimators ────────────────────────────────────────────────────────────────────────────────
def two_nn_id(X, trim=0.1):
    """Intrinsic dimension, Facco et al. 2017: slope of log(mu) vs -log(1 - F), mu = r2/r1."""
    D = squareform(pdist(X)); np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    ok = r[:, 0] > 1e-12
    mu = r[ok, 1] / r[ok, 0]
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    n = len(mu)
    if n < 10:
        return np.nan
    keep = int(n * (1 - trim))
    x = np.log(mu[:keep]); F = np.arange(1, n + 1) / n
    y = -np.log(1 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


def part_ratio(X):
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    ev = ev[ev > 1e-12]
    return float(ev.sum() ** 2 / np.sum(ev ** 2)) if len(ev) else np.nan


def separation(X, lab):
    """|centroid difference| / mean within-class RMS spread (unit-free, decoder-free)."""
    vals = np.unique(lab)
    if len(vals) != 2:
        return np.nan
    A, B = X[lab == vals[0]], X[lab == vals[1]]
    if len(A) < 3 or len(B) < 3:
        return np.nan
    d = np.linalg.norm(A.mean(0) - B.mean(0))
    w = np.mean([np.sqrt(np.mean(np.sum((A - A.mean(0)) ** 2, 1))), np.sqrt(np.mean(np.sum((B - B.mean(0)) ** 2, 1)))])
    return float(d / w) if w > 0 else np.nan


def knn_purity(E, lab, k=KP_NN, nshuf=NSHUF, seed=0):
    k = min(k, len(E) - 1); rng = np.random.RandomState(seed)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    obs = float(np.mean(lab[idx] == lab[:, None])); null = np.empty(nshuf)
    for i in range(nshuf):
        pl = lab[rng.permutation(len(lab))]; null[i] = np.mean(pl[idx] == pl[:, None])
    return obs, float(null.mean()), float((null >= obs).mean())


def mouse_cloud(m, stage, wn, conds):
    val = VALIDIX[(m, stage)]
    keep = np.zeros(len(MOUSE), bool)
    for (t, s, te) in conds:
        keep |= (TSK == t) & (SAMP == s) & (TESTO == te)
    idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & keep)[0]
    X = AW[wn][np.ix_(idx, val)]
    good = np.isfinite(X).all(0)
    X = X[:, good]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, idx


# ── per-mouse geometry ────────────────────────────────────────────────────────────────────────
rows = []
for setname, conds in SETS.items():
    for wn, wlab in WINS:
        for m in MICE:
            for stage in STAGES:
                X, idx = mouse_cloud(m, stage, wn, conds)
                if len(X) < 20:
                    continue
                labs = {'sample': SAMP[idx], 'match': (SAMP[idx] == TESTO[idx])}
                if setname == 'dual':
                    labs['gng'] = TSK[idx]
                nn = min(30, max(5, len(X) // 4))
                E = umap.UMAP(n_neighbors=nn, **UM).fit_transform(PCA(min(30, X.shape[1], len(X) - 1), random_state=0).fit_transform(X))
                r = dict(set=setname, win=wn, mouse=m, stage=stage, n=len(X), nneu=X.shape[1],
                         ID=two_nn_id(X), PR=part_ratio(X), R=float(np.mean(np.linalg.norm(X - X.mean(0), axis=1))))
                for v, lab in labs.items():
                    r[f'SEP_{v}'] = separation(X, np.asarray(lab))
                    o, nu, p = knn_purity(E, np.asarray(lab)); r[f'PUR_{v}'] = o; r[f'PURnull_{v}'] = nu; r[f'PURp_{v}'] = p
                rows.append(r)
        print(f'{setname} {wn}: per-mouse geometry done', flush=True)
R = pd.DataFrame(rows)

# ── stage and trial-set comparisons (paired over the nine mice) ────────────────────────────────
def paired(df, col, by='stage', a='Naive', b='Expert'):
    p = df.pivot_table(index='mouse', columns=by, values=col).dropna()
    if len(p) < 5 or a not in p or b not in p:
        return np.nan, np.nan, np.nan, 0
    return float(p[a].median()), float(p[b].median()), float(wilcoxon(p[a], p[b]).pvalue), len(p)

stats = []
METRICS = ['ID', 'PR', 'R', 'SEP_sample', 'SEP_match', 'SEP_gng', 'PUR_sample', 'PUR_match', 'PUR_gng']
print('\n══ naive → expert, per window and trial set (medians over mice, paired Wilcoxon) ══')
for setname in SETS:
    for wn, wlab in WINS:
        sub = R[(R.set == setname) & (R.win == wn)]
        for mt in METRICS:
            if mt not in sub or sub[mt].isna().all():
                continue
            na, ex, p, n = paired(sub, mt)
            stats.append(dict(set=setname, win=wn, metric=mt, naive=na, expert=ex, p=p, n=n))
            if not np.isnan(p):
                print(f'  {setname:4s} {wlab:10s} {mt:11s} {na:6.2f} → {ex:6.2f}   p = {p:.3f}  (n = {n})')
S = pd.DataFrame(stats)
print('\n══ DPA vs dual, same window and stage (medians over mice, paired Wilcoxon) ══')
cross = []
for wn, wlab in WINS:
    for stage in STAGES:
        for mt in ['ID', 'PR', 'R', 'SEP_sample', 'SEP_match', 'PUR_sample', 'PUR_match']:
            sub = R[(R.win == wn) & (R.stage == stage)]
            p_ = sub.pivot_table(index='mouse', columns='set', values=mt).dropna()
            if len(p_) < 5:
                continue
            pv = float(wilcoxon(p_['DPA'], p_['dual']).pvalue)
            cross.append(dict(win=wn, stage=stage, metric=mt, DPA=float(p_['DPA'].median()), dual=float(p_['dual'].median()), p=pv, n=len(p_)))
            print(f'  {wlab:10s} {stage:6s} {mt:11s} DPA {p_["DPA"].median():6.2f} vs dual {p_["dual"].median():6.2f}   p = {pv:.3f}')
C = pd.DataFrame(cross)

# ── pooled display maps: one UMAP per (set x window), naive + expert together ──────────────────
def pools(stage, conds):
    return {(ci, m): np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            for ci, (t, s, te) in enumerate(conds) for m in MICE}

def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s_ = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s_) & (s_ > 1e-6), s_, 1.0)
    return sd

def cloud_no_reuse(stage, wn, conds, rng):
    M = AW[wn]; sd = neuron_scale(stage, M); P = pools(stage, conds)
    kp = min(len(v) for v in P.values() if len(v))
    X = np.zeros((len(conds) * kp, N)); crow = np.repeat(np.arange(len(conds)), kp)
    for ci in range(len(conds)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; ix = rng.permutation(P[(ci, m)])
            if not len(ix):
                continue
            cm = np.nanmean(M[np.ix_(ix, val)], 0)
            for j in range(kp):
                v = M[ix[j], val]; X[ci * kp + j, val] = np.where(np.isfinite(v), v, cm)
    X = X / sd[None, :]
    return X - X.mean(0), crow, kp

MAPS = {}
fig, axs = plt.subplots(len(WINS) * 2, 4, figsize=(12.0, 4.6 * len(WINS)))
for R_, (setname, conds) in enumerate(SETS.items()):
    for r_, (wn, wlab) in enumerate(WINS):
        Xn, cn, kn = cloud_no_reuse('Naive', wn, conds, np.random.RandomState(1))
        Xe, ce, ke = cloud_no_reuse('Expert', wn, conds, np.random.RandomState(2))
        X = np.vstack([Xn, Xe]); crow = np.r_[cn, ce]; stage = np.r_[np.zeros(len(Xn), int), np.ones(len(Xe), int)]
        Z = PCA(min(30, len(X) - 1), random_state=0).fit_transform(X)
        E = umap.UMAP(n_neighbors=min(30, len(X) // 4), **UM).fit_transform(Z)
        labs = {'stage': stage, 'sample': np.array([conds[c][1] for c in crow]), 'match': np.array([conds[c][1] == conds[c][2] for c in crow])}
        if setname == 'dual':
            labs['gng'] = np.array([conds[c][0] for c in crow])
        pu = {v: knn_purity(E, np.asarray(l)) for v, l in labs.items()}
        MAPS[(setname, wn)] = dict(E=E, labs=labs, purity=pu, kp=(kn, ke))
        print(f'{setname:4s} {wlab:10s} pooled map: n={len(E)} ' + '  '.join(f'{v} {x[0]:.2f}/{x[1]:.2f} p={x[2]:.3f}' for v, x in pu.items()), flush=True)
        order = ['stage', 'sample', 'match'] + (['gng'] if setname == 'dual' else [])
        row = R_ * len(WINS) + r_
        for c_, v in enumerate(order):
            ax = axs[row, c_]; cmap = {'stage': STAGEC, 'sample': SAMPC, 'match': MATCHC, 'gng': GNGC}[v]
            ax.scatter(E[:, 0], E[:, 1], s=14, c=[cmap[x] for x in labs[v]], alpha=0.8, lw=0)
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            ax.set_title(f'{setname} · {wlab} · {v}   purity {pu[v][0]:.2f} (null {pu[v][1]:.2f})', loc='left', fontsize=6.6)
        if setname == 'DPA':
            axs[row, 3].axis('off')
fig.suptitle('Delay and decision manifolds: one UMAP per trial set and window, naive + expert together (no trial reuse)', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig('figures/pseudo/dimensionality/png/manifold_geometry_maps.png', bbox_inches='tight'); plt.close(fig)

# ── summary figure of the per-mouse geometry ──────────────────────────────────────────────────
PLOT = [('ID', 'intrinsic dimension'), ('PR', 'participation ratio'), ('R', 'manifold radius'),
        ('SEP_sample', 'sample separation'), ('SEP_match', 'choice separation'), ('PUR_sample', 'sample purity (UMAP)')]
fig, axs = plt.subplots(len(PLOT), 1, figsize=(7.6, 2.3 * len(PLOT)), sharex=True)
xpos = {}; k = 0
for setname in SETS:
    for wn, wlab in WINS:
        xpos[(setname, wn)] = k; k += 1
for a_, (mt, lab) in enumerate(PLOT):
    ax = axs[a_]
    for (setname, wn), x in xpos.items():
        for si, stage in enumerate(STAGES):
            v = R[(R.set == setname) & (R.win == wn) & (R.stage == stage)][mt].dropna().to_numpy()
            if not len(v):
                continue
            xx = x + (si - 0.5) * 0.34
            ax.scatter(np.full(len(v), xx) + np.random.RandomState(a_).uniform(-0.06, 0.06, len(v)), v, s=11,
                       color=STAGEC[si], alpha=0.75, lw=0, zorder=3)
            ax.plot([xx - 0.13, xx + 0.13], [np.median(v)] * 2, color='k', lw=1.3, zorder=4)
        st = S[(S.set == setname) & (S.win == wn) & (S.metric == mt)]
        if len(st) and not np.isnan(st.iloc[0].p) and st.iloc[0].p < 0.05:
            ax.text(x, ax.get_ylim()[1], '∗', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.set_ylabel(lab, fontsize=7.5); ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f'{s}\n{dict(WINS)[w]}' for (s, w) in xpos], fontsize=6.5)
axs[0].legend(handles=[mlines.Line2D([0], [0], marker='o', ls='none', color=STAGEC[i], label=s.lower()) for i, s in enumerate(['naïve', 'expert'])],
              frameon=False, loc='upper right', ncol=2)
fig.suptitle('Per-mouse manifold geometry (one point per mouse; ∗ = paired Wilcoxon p < .05 naive vs expert)', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig('figures/pseudo/dimensionality/png/manifold_geometry_stats.png', bbox_inches='tight')
pickle.dump(dict(R=R, S=S, C=C, MAPS=MAPS), open('figures/pseudo/dimensionality/manifold_geometry.pkl', 'wb'))
print('\nsaved manifold_geometry.pkl + two figures')
