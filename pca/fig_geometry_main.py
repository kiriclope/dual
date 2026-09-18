"""fig_geometry_main.py — NEW main figure between Figs 2 and 3 (2026-09-18, Leon: "characterize the geometry of the data"
with a 2-D embedding, "purely geometry"): the population state space seen whole, with no axes chosen.

  a  mid-delay: one t-SNE of naïve + expert pooled pseudo-trials (no trial reuse: each real trial of each mouse enters at
     most one pseudo-trial; per-stage neuron scaling and centering), coloured by task, by sample, by stage
  b  decision: the same, coloured by task, by match (the choice) and by stage
  c  k-nearest-neighbour label purity in the maps (k = 7) against label-shuffle nulls: task structures the maps; sample and
     match are at chance in 2-D at both windows; naïve and expert interleave (stage purity at chance)
  d  unsupervised METRIC trajectories: PCA of the condition-mean state space over the whole trial, condition-independent
     ramp removed (nonlinear embeddings of trajectories starburst — one filament per condition — so they are not used here)
  e  representational geometry: 2-D MDS of the mean 12-condition RDM (expert, both windows), and per-mouse RDM split-half
     reliability, across-mouse consistency and naïve-vs-expert correlation

Substrate: fits_inputs.pkl (AW window matrices, correct laser-off trials for the condition means; the single-trial maps use
correct laser-off trials of the twelve odor-defined conditions). Windows: mid-delay bins 33-38, decision 54-62.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_geometry_main.py [--nocap]
Output: figures/pseudo/dimensionality/{png,svg}/fig_geometry_main.{png,svg}; cache figures/pseudo/dimensionality/geometry_cache.pkl
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.25
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7, 'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = PS*8
NOCAP = '--nocap' in sys.argv[1:]
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
TASKC = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
SAMPC = {0: '#332288', 1: '#44AA99'}
MATCHC = {True: '#4daf4a', False: '#377eb8'}
STAGEC = {0: '0.62', 1: '#332288'}
SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
WLAB = {'md': 'mid-delay', 'decision': 'decision'}
K_NN = 7; NSHUF = 500
NB = 84; SM = 5                                            # trajectory bins and smoothing
EVENTS = [(12, 'sample', '#332288'), (27, 'GNG', '#cc3311'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def pools(stage):
    return {(ci, m): np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            for ci, (t, s, te) in enumerate(ALL12) for m in MICE}


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def cloud_no_reuse(stage, wn, rng):
    """kp pseudo-trials per condition, kp = the smallest (mouse x condition) pool: pseudo-trial j takes each mouse's j-th
    permuted trial, so no real trial enters two pseudo-trials of a map."""
    M = AW[wn]; sd = neuron_scale(stage, M); P = pools(stage)
    kp = min(len(v) for v in P.values() if len(v))
    X = np.zeros((12 * kp, N)); crow = np.repeat(np.arange(12), kp)
    for ci in range(12):
        for m in MICE:
            val = VALIDIX[(m, stage)]; idx = rng.permutation(P[(ci, m)])
            if not len(idx):
                continue
            cm = np.nanmean(M[np.ix_(idx, val)], 0)
            for j in range(kp):
                v = M[idx[j], val]; X[ci * kp + j, val] = np.where(np.isfinite(v), v, cm)
    X = X / sd[None, :]
    return X - X.mean(0), crow, kp


def knn_purity(E, lab, k=K_NN, nshuf=NSHUF, seed=0):
    rng = np.random.RandomState(seed)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    obs = float(np.mean(lab[idx] == lab[:, None])); null = np.empty(nshuf)
    for i in range(nshuf):
        pl = lab[rng.permutation(len(lab))]; null[i] = np.mean(pl[idx] == pl[:, None])
    return obs, float(null.mean()), float(np.percentile(null, 97.5)), float((null >= obs).mean())


def rdm_of(m, stage, wn, half=None, seed=0):
    val = VALIDIX[(m, stage)]; M = AW[wn]; P = pools(stage); rng = np.random.RandomState(seed); means = []
    for ci in range(12):
        idx = P[(ci, m)]
        if half is not None:
            p = rng.permutation(idx); idx = p[:len(p) // 2] if half == 0 else p[len(p) // 2:]
        means.append(np.nanmean(M[np.ix_(idx, val)], 0) if len(idx) else np.full(len(val), np.nan))
    Mm = np.array(means); Mm = (Mm - np.nanmean(Mm, 0)) / (np.nanstd(Mm, 0) + 1e-9)
    return squareform(pdist(np.nan_to_num(Mm), 'correlation'))


# ══ compute ═══════════════════════════════════════════════════════════════════════════════════
CACHE = 'figures/pseudo/dimensionality/geometry_cache.pkl'
if os.path.exists(CACHE) and '--recompute' not in sys.argv[1:]:
    G = pickle.load(open(CACHE, 'rb'))
else:
    G = {'maps': {}, 'purity': {}, 'rdm': {}}
    IU = np.triu_indices(12, 1)
    for wn in ['md', 'decision']:
        Xn, cn, kn = cloud_no_reuse('Naive', wn, np.random.RandomState(1)); Xe, ce, ke = cloud_no_reuse('Expert', wn, np.random.RandomState(2))
        X = np.vstack([Xn, Xe]); crow = np.r_[cn, ce]; stage = np.r_[np.zeros(len(Xn), int), np.ones(len(Xe), int)]
        Z = PCA(30, random_state=0).fit_transform(X)
        E = TSNE(2, perplexity=30, init='pca', random_state=0, learning_rate='auto').fit_transform(Z)
        labs = {'task': np.array([ALL12[c][0] for c in crow]), 'sample': np.array([ALL12[c][1] for c in crow]),
                'match': np.array([ALL12[c][1] == ALL12[c][2] for c in crow]), 'stage': stage}
        G['maps'][wn] = dict(E=E, crow=crow, stage=stage, kp=(kn, ke), labs=labs)
        G['purity'][wn] = {nm: knn_purity(E, lab) for nm, lab in labs.items()}
        print(f'{wn}: {len(E)} points (kp naive {kn}, expert {ke}); purity ' + '  '.join(f'{nm} {v[0]:.2f} (null {v[1]:.2f}, p {v[3]:.3f})' for nm, v in G['purity'][wn].items()), flush=True)
        R = {(m, st): rdm_of(m, st, wn) for m in MICE for st in STAGES}
        rel = {st: [spearmanr(rdm_of(m, st, wn, 0)[IU], rdm_of(m, st, wn, 1)[IU])[0] for m in MICE] for st in STAGES}
        cons = {st: [spearmanr(R[(m, st)][IU], np.mean([R[(o, st)] for o in MICE if o != m], 0)[IU])[0] for m in MICE] for st in STAGES}
        xstage = [spearmanr(R[(m, 'Naive')][IU], R[(m, 'Expert')][IU])[0] for m in MICE]
        meanR = {st: np.mean([R[(m, st)] for m in MICE], 0) for st in STAGES}
        xmean = spearmanr(meanR['Naive'][IU], meanR['Expert'][IU])[0]
        G['rdm'][wn] = dict(mean=meanR, rel=rel, cons=cons, xstage=xstage, xmean=float(xmean),
                            mds={st: MDS(2, dissimilarity='precomputed', random_state=0, n_init=16).fit_transform(meanR[st]) for st in STAGES})
        for st in STAGES:
            print(f'{wn} {st}: RDM reliability median {np.median(rel[st]):.2f}, consistency median {np.median(cons[st]):.2f}', flush=True)
        print(f'{wn}: naive-vs-expert RDM ρ per mouse median {np.median(xstage):.2f} (range {min(xstage):.2f}-{max(xstage):.2f}); mean RDMs ρ {xmean:.2f}', flush=True)
    CMBIN = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))['CMBIN']
    G['traj'] = {}
    for st in STAGES:
        CM = np.asarray(CMBIN[st], float); k = np.ones(SM) / SM
        CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)
        CM = CM - CM.mean(0, keepdims=True)                     # remove the condition-independent ramp
        Z = CM.transpose(0, 2, 1).reshape(-1, CM.shape[1]); sd = Z.std(0); sd[sd < 1e-9] = 1.0; Z = (Z - Z.mean(0)) / sd
        pc = PCA(6, random_state=0); Y = pc.fit_transform(Z)
        G['traj'][st] = dict(Y=Y.reshape(12, NB, 6), ev=pc.explained_variance_ratio_)
        print(f'{st}: trajectory PCs ' + ' '.join(f'{100*v:.0f}%' for v in pc.explained_variance_ratio_[:3]), flush=True)
    pickle.dump(G, open(CACHE, 'wb'))


def cname(cd): return f'{SHORT[cd[0]]}·{"A" if cd[1] == 0 else "B"}{"m" if cd[1] == cd[2] else "n"}'


def plabel(ax, s, dx=-0.08, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


# ══ draw ══════════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 10.4))
outer = fig.add_gridspec(3, 24, height_ratios=[1.0, 1.0, 1.05], hspace=0.42, wspace=0.9, left=0.05, right=0.985, top=0.955, bottom=0.055)
firsts = {}
for r, wn in enumerate(['md', 'decision']):
    Mp = G['maps'][wn]; E = Mp['E']; labs = Mp['labs']
    cols = [('task', labs['task'], TASKC), ('sample', labs['sample'], SAMPC) if wn == 'md' else ('match', labs['match'], MATCHC), ('stage', labs['stage'], STAGEC)]
    gsm = outer[r, 0:17].subgridspec(1, 3, wspace=0.12)
    for c, (nm, lab, cmap) in enumerate(cols):
        ax = fig.add_subplot(gsm[0, c])
        if c == 0: firsts[wn] = ax
        ax.scatter(E[:, 0], E[:, 1], s=13, c=[cmap[v] for v in lab], alpha=0.75, lw=0)
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        pu = G['purity'][wn][nm]
        ax.set_title(f'{WLAB[wn]} · by {nm if nm != "match" else "choice (match)"}', loc='left', fontsize=TITLE_FS)
        ax.text(0.5, -0.02, f'kNN purity {pu[0]:.2f} (null {pu[1]:.2f})', transform=ax.transAxes, ha='center', va='top', fontsize=PS*6.5, color='0.3')
        hs = ([mlines.Line2D([0], [0], marker='o', ls='none', ms=4, color=TASKC[t], label=SHORT[t]) for t in TASKC] if nm == 'task' else
              [mlines.Line2D([0], [0], marker='o', ls='none', ms=4, color=SAMPC[x], label=f'sample {"A" if x == 0 else "B"}') for x in SAMPC] if nm == 'sample' else
              [mlines.Line2D([0], [0], marker='o', ls='none', ms=4, color=MATCHC[x], label='match (lick)' if x else 'nonmatch') for x in MATCHC] if nm == 'match' else
              [mlines.Line2D([0], [0], marker='o', ls='none', ms=4, color=STAGEC[x], label='naïve' if x == 0 else 'expert') for x in STAGEC])
        ax.legend(handles=hs, frameon=False, loc='upper left', handletextpad=0.2, borderaxespad=0.1)
# c: purity bars, both windows
ax = fig.add_subplot(outer[0, 18:24]); firsts['c'] = ax
names = ['task', 'sample', 'match', 'stage']
for i_, nm in enumerate(names):
    for k_, wn in enumerate(['md', 'decision']):
        o, nu, hi, pv = G['purity'][wn][nm]; x = i_ * 2.4 + k_ * 0.9
        ax.bar(x, o, color=('0.45' if k_ == 0 else '#332288') if pv < .05 else ('0.8' if k_ == 0 else '#b9b2d6'), width=0.82)
        ax.plot([x - 0.41, x + 0.41], [hi, hi], color='#cc3311', lw=1.0, zorder=4)
        ax.text(x, o + 0.03, '∗' if pv < .05 else 'n.s.', ha='center', va='bottom', fontsize=PS*(10 if pv < .05 else 5.8), fontweight='bold', color='k' if pv < .05 else '0.55')
ax.set_xticks([i_ * 2.4 + 0.45 for i_ in range(4)]); ax.set_xticklabels(names, fontsize=PS*6.8)
ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel('kNN label purity')
ax.set_title('purity vs shuffle (grey mid-delay, indigo decision)', loc='left', fontsize=TITLE_FS)
# e1: MDS of the mean RDM, expert, both windows
gse = outer[2, 15:24].subgridspec(1, 2, wspace=0.10)
for c, wn in enumerate(['md', 'decision']):
    ax = fig.add_subplot(gse[0, c]); E = G['rdm'][wn]['mds']['Expert']
    if c == 0: firsts['e1'] = ax
    for ci, cd in enumerate(ALL12):
        ax.scatter(E[ci, 0], E[ci, 1], s=42, color=TASKC[cd[0]], marker='o' if cd[1] == 0 else 's', edgecolors='k' if cd[1] == cd[2] else 'none', linewidths=0.8, zorder=3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_title(f'{WLAB[wn]}', loc='left', fontsize=TITLE_FS)
    ax.margins(0.18)
# d: unsupervised metric trajectories (PCA of the condition-mean state space, CI removed)
gst = outer[2, 0:14].subgridspec(1, 2, wspace=0.12)
for c, st in enumerate(STAGES):
    ax = fig.add_subplot(gst[0, c]); Y = G['traj'][st]['Y']; ev = G['traj'][st]['ev']
    if c == 0: firsts['d'] = ax
    for ci, cd in enumerate(ALL12):
        tr = Y[ci]
        ax.plot(tr[:, 0], tr[:, 1], '-', color=TASKC[cd[0]], lw=1.0, alpha=0.8, zorder=2)
        ax.scatter(tr[0, 0], tr[0, 1], s=14, color=TASKC[cd[0]], marker='o', lw=0, zorder=4)
        for b, nm, col in EVENTS:
            ax.scatter(tr[b, 0], tr[b, 1], s=12, color=col, marker='s', lw=0, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xlabel(f'PC1 ({100*ev[0]:.0f}%)', fontsize=PS*6.8); ax.set_ylabel(f'PC2 ({100*ev[1]:.0f}%)', fontsize=PS*6.8)
    ax.set_title(f'{"naïve" if st == "Naive" else "expert"} · condition-mean trajectories', loc='left', fontsize=TITLE_FS)
    if c == 0:
        ax.legend(handles=[mlines.Line2D([0], [0], marker='s', ls='none', ms=4, color=col, label=nm) for _, nm, col in EVENTS],
                  frameon=False, loc='upper left', ncol=2, handletextpad=0.2, columnspacing=0.6, fontsize=PS*5.8)
# e2: per-mouse RDM reliability / consistency / naive-vs-expert
ax = fig.add_subplot(outer[1, 18:24]); firsts['e2'] = ax
X0 = 0
for c, wn in enumerate(['md', 'decision']):
    Rd = G['rdm'][wn]
    for i_, (lab, vals) in enumerate([('split-half\nreliability', Rd['rel']['Expert']), ('across-mouse\nconsistency', Rd['cons']['Expert']), ('naïve vs\nexpert', Rd['xstage'])]):
        x = X0 + i_; v = np.asarray(vals)
        ax.scatter(np.full(len(v), x) + np.random.RandomState(i_).uniform(-0.14, 0.14, len(v)), v, s=12, color='0.5' if c == 0 else '#332288', alpha=0.75, lw=0, zorder=3)
        ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color='k', lw=1.3, zorder=4)
        pass
    X0 += 3.4
ax.axhline(0, ls=':', color='0.6', lw=0.8); ax.set_ylim(-0.05, 1.05); ax.set_xlim(-0.7, 6.1)
ax.set_xticks([0, 1, 2, 3.4, 4.4, 5.4])
ax.set_xticklabels(['split-half', 'across mice', 'naïve vs\nexpert'] * 2, fontsize=PS*6.0, rotation=35, ha='right')
ax.set_ylabel('Spearman ρ between RDMs'); ax.set_title('per mouse (grey mid-delay, indigo decision)', loc='left', fontsize=TITLE_FS)
plabel(firsts['md'], 'a', dx=-0.05); plabel(firsts['decision'], 'b', dx=-0.05); plabel(firsts['c'], 'c', dx=-0.26)
plabel(firsts['e2'], 'd', dx=-0.26); plabel(firsts['d'], 'e', dx=-0.07); plabel(firsts['e1'], 'f', dx=-0.16)

pm = G['purity']; rd = G['rdm']
CAP = [
    'Figure 3 | The population geometry seen whole: an unsupervised map of the states is organized by the task and by '
    'nothing else, is the same map before and after learning, and the condition geometry it summarizes is shared across '
    'animals. Pseudo-population of 3,319 neurons; mid-delay (5.5–6.5 s) and decision (9.0–10.5 s) states; correct laser-off '
    'trials of the twelve odor-defined conditions.',
    'a, One t-SNE of naïve and expert pseudo-trials together at mid-delay, coloured by task, by sample and by stage. Each '
    'pseudo-trial takes one real trial per mouse and no trial enters two pseudo-trials, so neighbourhoods reflect trial-to-trial '
    'population states, not resampled duplicates (' + f'{G["maps"]["md"]["kp"][0]} naïve and {G["maps"]["md"]["kp"][1]} expert pseudo-trials per condition' + '); '
    'neurons scaled and centered per stage. Axes are arbitrary and distances between clusters are not metric.',
    'b, The same at the decision, coloured by task, by choice (match, the lick) and by stage.',
    'c, k-nearest-neighbour label purity (k = 7) in the maps of a and b against label-shuffle nulls (red, 97.5th percentile; '
    f'∗ p < .05). Mid-delay: task {pm["md"]["task"][0]:.2f}, sample {pm["md"]["sample"][0]:.2f}, match {pm["md"]["match"][0]:.2f}, stage {pm["md"]["stage"][0]:.2f}; '
    f'decision: task {pm["decision"]["task"][0]:.2f}, sample {pm["decision"]["sample"][0]:.2f}, match {pm["decision"]["match"][0]:.2f}, stage {pm["decision"]["stage"][0]:.2f} '
    '(nulls 0.33 for task, 0.50 otherwise). The task context is the only variable that structures the map; the memory and '
    'the choice, decodable from the same states (Figs 2c, 4c), occupy too small a share of the variance to organize '
    'neighbourhoods in two dimensions, and naïve and expert states interleave.',
    'd, The condition geometry is reliable within, shared across and preserved between animals: split-half reliability of each mouse\'s '
    f'RDM (median {np.median(rd["md"]["rel"]["Expert"]):.2f} mid-delay, {np.median(rd["decision"]["rel"]["Expert"]):.2f} decision), its correlation with the leave-one-out mean of the other eight '
    f'(median {np.median(rd["md"]["cons"]["Expert"]):.2f}, {np.median(rd["decision"]["cons"]["Expert"]):.2f}) and the correlation of each mouse\'s naïve and expert RDMs '
    f'(median {np.median(rd["md"]["xstage"]):.2f}, {np.median(rd["decision"]["xstage"]):.2f}; mean RDMs, ρ = {rd["md"]["xmean"]:.2f} and {rd["decision"]["xmean"]:.2f}). '
    'Spearman ρ over the 66 condition pairs; one point per mouse, line = median.',
    'e, The same state space with time, and with a metric: principal components of the twelve condition-mean trajectories '
    'over the whole trial, after removing the condition-independent component that all conditions share (5-bin smoothing; '
    f'PC1 and PC2 carry {100*G["traj"]["Expert"]["ev"][0]:.0f}% and {100*G["traj"]["Expert"]["ev"][1]:.0f}% of the condition-mean variance in expert mice). Dots, trial start; squares, event onsets. '
    'The conditions leave a common state at the sample and travel along three task-specific arms; the arrangement is the '
    'same before and after learning. Nonlinear embeddings cannot show this: t-SNE and UMAP preserve local neighbourhoods, '
    'and a condition\'s own temporal sequence is its strongest neighbourhood, so each condition becomes an isolated filament '
    'and the geometry between conditions is lost (Methods).',
    'f, Two-dimensional multidimensional scaling of the twelve-condition dissimilarity matrix averaged over the nine mice '
    '(expert, both windows; colour, task; circle, sample A; square, sample B; black edge, match): the conditions group by '
    'task, and split by sample and by choice within every task.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = 'figures/pseudo/dimensionality'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_geometry_main.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/fig_geometry_main.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/fig_geometry_main.png')
