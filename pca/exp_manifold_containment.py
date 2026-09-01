"""exp_manifold_containment.py — is the learning EDIT within the naive intrinsic manifold?
(BCI-field standard analysis, Sadtler 2014 / Golub 2018 / Oby 2019, applied to natural learning.)

Per mouse (neurons registered across stages): the naive intrinsic manifold = top-d PCs of
naive single-trial activity (correct laser-off trials, mean-centred, naive-referenced per-neuron
scaling). Decompose the two learning edits into within- vs outside-manifold components:
  - PUSH vector    : expert - naive mean DPA delay state (per neuron), late-delay window;
  - ROTATION vector: expert dist axis - naive dist axis (unit decoder axes @ md, Go vs NoGo).
Containment f = ||P v||^2 / ||v||^2 (P = naive top-d projector). References:
  - CEILING: containment of held-out naive activity itself (split-half: manifold from half 1,
    f of held-out half-2 single-trial deviations) — what "fully within-manifold" looks like;
  - NULL: random directions in the same neuron space (E[f] = d/N; 1000 draws, 95% band).
NOT the retired Elsayed alignment-index (that compared condition-mean subspaces at a covariance
null — dead-end logged); here the manifold is the SINGLE-TRIAL covariance space and the tested
objects are the specific learning-edit vectors. Caveat to carry: the stage difference spans
days, so slow drift contributes to the push vector; naive-referenced scaling + the xstage
scaling-sensitivity check (delta<=0.02) bound but do not eliminate this.

d = 10 main (5/20 sweep printed). Cache MANIFOLD_CONTAIN+SUF into results.pkl; preview PNG.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_manifold_containment.py --nopca
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon
from decoders import fit_axis, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
DS = [5, 10, 20]
D_MAIN = 10
NNULL = 1000

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

def sel(mo, st, task=None, correct=True):
    m = (MOUSE == mo) & (LEARN == st) & (LAS == 0)
    if correct:
        m &= (PERF == 1)
    if task == 'DPA':
        m &= (TSK == 'DPA')
    elif task == 'Go':
        m &= (TSK == 'DualGo')
    elif task == 'NoGo':
        m &= (TSK == 'DualNoGo')
    return np.where(m)[0]

def contain(P, v):
    n = np.linalg.norm(v)
    return float(np.linalg.norm(P @ v) ** 2 / n ** 2) if n > 0 else np.nan

rng = np.random.RandomState(42)
RES_C = {}
print(f'══ MANIFOLD_CONTAIN (SUF="{SUF}", d={D_MAIN}; sweep {DS}) ══')
for mo in MICE:
    val = np.intersect1d(VALIDIX[(mo, 'Naive')], VALIDIX[(mo, 'Expert')])
    if len(val) < 30:
        continue
    ent = {}
    for win, tag in [('delay', 'push'), ('md', 'rot')]:
        M = AW[win]
        nv_all = sel(mo, 'Naive')
        Xn = np.nan_to_num(M[np.ix_(nv_all, val)])
        sd = Xn.std(0); sd = np.where(sd > 1e-6, sd, 1.0)      # naive-referenced scaling
        Xn = Xn / sd
        # the edit vector
        if tag == 'push':
            a = np.nan_to_num(M[np.ix_(sel(mo, 'Expert', 'DPA'), val)]) / sd
            b = np.nan_to_num(M[np.ix_(sel(mo, 'Naive', 'DPA'), val)]) / sd
            v = a.mean(0) - b.mean(0)
        else:
            def _axis(st):
                g = np.nan_to_num(M[np.ix_(sel(mo, st, 'Go', correct=False), val)]) / sd
                n_ = np.nan_to_num(M[np.ix_(sel(mo, st, 'NoGo', correct=False), val)]) / sd
                if min(len(g), len(n_)) < 6:
                    return None
                w, _ = fit_axis(np.vstack([g, n_]),
                                np.r_[np.ones(len(g), int), np.zeros(len(n_), int)])
                return w
            wn, we = _axis('Naive'), _axis('Expert')
            if wn is None or we is None:
                continue
            v = we - wn
        # manifold + ceiling from split halves of naive trials
        p = rng.permutation(len(Xn)); h = len(p) // 2
        X1, X2 = Xn[p[:h]], Xn[p[h:]]
        for d in DS:
            U = np.linalg.svd((X1 - X1.mean(0)), full_matrices=False)[2][:d].T   # (N, d)
            P = U @ U.T
            f_edit = contain(P, v)
            dev = X2 - X1.mean(0)
            f_ceil = float(np.mean([contain(P, x) for x in dev[:200]]))
            f_null = np.array([contain(P, rng.randn(len(val))) for _ in range(NNULL if d == D_MAIN else 50)])
            ent[(tag, d)] = (f_edit, f_ceil, float(np.median(f_null)),
                             float(np.percentile(f_null, 97.5)))
    RES_C[mo] = ent
    if ('push', D_MAIN) in ent:
        e = ent[('push', D_MAIN)]; r = ent.get(('rot', D_MAIN), (np.nan,) * 4)
        print(f'  {mo:8s} N={len(val):4d}  push f={e[0]:.2f} (ceil {e[1]:.2f}, null {e[2]:.2f}'
              f'[{e[3]:.2f}])  rot f={r[0]:.2f} (ceil {r[1]:.2f}, null {r[2]:.2f}[{r[3]:.2f}])')

for tag in ['push', 'rot']:
    for d in DS:
        fs = np.array([RES_C[m][(tag, d)][0] for m in RES_C if (tag, d) in RES_C[m]])
        ns = np.array([RES_C[m][(tag, d)][2] for m in RES_C if (tag, d) in RES_C[m]])
        cs = np.array([RES_C[m][(tag, d)][1] for m in RES_C if (tag, d) in RES_C[m]])
        if len(fs) >= 6:
            p_null = wilcoxon(fs, ns).pvalue
            p_ceil = wilcoxon(fs, cs).pvalue
            print(f'{tag:4s} d={d:2d}: edit f md={np.median(fs):.2f}  null md={np.median(ns):.2f} '
                  f'(vs null p={p_null:.4f})  ceiling md={np.median(cs):.2f} (vs ceiling p={p_ceil:.4f}) n={len(fs)}')

RESF = 'figures/pseudo/dimensionality/results.pkl'
d_ = pickle.load(open(RESF, 'rb'))
d_['MANIFOLD_CONTAIN' + SUF] = RES_C
pickle.dump(d_, open(RESF, 'wb'))
print('merged MANIFOLD_CONTAIN' + SUF)

import seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
_pal = sns.color_palette('tab10', n_colors=len(MICE))
MC = {m: _pal[i] for i, m in enumerate(MICE)}
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
for ax, tag, ttl in [(axs[0], 'push', 'push displacement'), (axs[1], 'rot', 'distractor-axis rotation')]:
    for i, m in enumerate([m for m in MICE if m in RES_C and (tag, D_MAIN) in RES_C[m]]):
        f, c, nmd, n95 = RES_C[m][(tag, D_MAIN)]
        ax.plot([0, 1], [f, c], '-', color=MC[m], lw=0.8, alpha=0.45, zorder=2)
        ax.scatter([0, 1], [f, c], s=34, color=MC[m], edgecolors='w', linewidths=0.6, zorder=3)
    n95s = [RES_C[m][(tag, D_MAIN)][3] for m in RES_C if (tag, D_MAIN) in RES_C[m]]
    ax.axhspan(0, float(np.median(n95s)), color='0.85', zorder=1)
    ax.text(0.02, float(np.median(n95s)) + 0.02, 'random-direction null (95%)',
            fontsize=6.0, color='0.4')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['learning\nedit', 'held-out naive\nactivity (ceiling)'])
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1.02)
    ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
axs[0].set_ylabel(f'fraction within the naive\nintrinsic manifold (top-{D_MAIN} PCs)')
fig.suptitle('The learning edit lies within the naive intrinsic manifold', x=0.02, ha='left',
             fontsize=TITLE_FS)
fig.tight_layout(rect=(0, 0, 1, 0.92))
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/exp_manifold_containment{SUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/exp_manifold_containment{SUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/exp_manifold_containment{SUF}.png'))
