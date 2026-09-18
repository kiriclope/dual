"""fig_unsup_axes_cv.py — CROSS-VALIDATED unsupervised axes of the manifold (Leon 2026-09-18).

Same idea as fig_unsup_axes.py, with the basis and the trajectories taken from disjoint trials:

  for each of NSPLIT random half-splits, and in BOTH directions,
      the principal axes are fitted on the per-bin condition means of one half (labels never used),
      and the OTHER half's condition means are projected on them.

So the drawn trajectory is a held-out projection and the variance numbers are honest:
  var_heldout[j]  variance of the held-out scores along axis j, as a fraction of the held-out states' total variance
  reliable[j]     the cvPCA cross-term <fit-half score x held-out score> / total — unbiased for signal variance, and
                  the one to quote: a component that only fits noise has a held-out variance but no cross-term.
Axes from different splits are matched to a fixed reference (the full-data axes) by Hungarian assignment on |cos| and
sign-aligned before averaging; the reference only labels the components (exp_pceta_cv.py uses the same device).
Identity is read off afterwards by eta^2 of each design factor on the held-out scores.

Needs figures/pseudo/dimensionality/cmbin_splits.pkl (exp_cmbin_splits.py).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_unsup_axes_cv.py
Output: figures/pseudo/dimensionality/{png,svg}/unsup_axes_cv.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'axes.linewidth': 0.7,
})
TITLE_FS = 8
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
NB, SM, NPC = 84, 5, 3
SAMPC = {0: '#332288', 1: '#44AA99'}; GNGC = {'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
EPOCHS = [(2.0, 3.0, '#332288', 'sample'), (4.5, 5.5, '#cc3311', 'GNG'), (6.5, 7.0, '#ee7733', 'cue'), (9.0, 10.0, '#377eb8', 'test')]
T = np.arange(NB) / 6.0
FACTORS = {'sample': lambda c: 1.0 if c[1] == 0 else -1.0, 'gng': lambda c: 1.0 if c[0] == 'DualGo' else -1.0,
           'choice': lambda c: 1.0 if c[1] == c[2] else -1.0, 'test': lambda c: 1.0 if c[2] == 0 else -1.0}
FWIN = {'sample': slice(33, 39), 'gng': slice(33, 39), 'choice': slice(54, 63), 'test': slice(54, 63)}

SPL = pickle.load(open('figures/pseudo/dimensionality/cmbin_splits.pkl', 'rb'))
SPLITS, NSPLIT = SPL['splits'], SPL['nsplit']
R = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
CMFULL = R['CMBIN']
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
KSM = np.ones(SM) / SM


def neuron_scale(stage):
    """Stage-level, condition-agnostic per-neuron scale (the cvPCA convention; carries no condition information)."""
    M = AW['md']; sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def states(CM, sel, sd):
    """(ncond, N, 84) -> smoothed, condition-independent component removed, scaled, flattened (ncond*84, N)."""
    A = np.asarray(CM, float)[sel]
    A = np.apply_along_axis(lambda v: np.convolve(v, KSM, mode='same'), 2, A)
    A = A - A.mean(0, keepdims=True)
    A = A / sd[None, :, None]
    return np.nan_to_num(A.transpose(0, 2, 1).reshape(-1, N))


print(f'cross-validated unsupervised axes: {NSPLIT} splits x 2 directions, basis and trajectories from disjoint trials\n')
RES = {}
for setname, conds in SETS.items():
    sel = [ALL12.index(c) for c in conds]
    for stage in STAGES:
        sd = neuron_scale(stage)
        Vref = PCA(NPC, random_state=0).fit(states(CMFULL[stage], sel, sd)).components_      # labels the components only
        acc, vh, rel = [], [], []
        for sp in range(NSPLIT):
            for fit_half in (0, 1):
                Zf = states(SPLITS[(stage, sp, fit_half)], sel, sd)
                Zp = states(SPLITS[(stage, sp, 1 - fit_half)], sel, sd)
                p = PCA(NPC, random_state=0).fit(Zf)
                V = p.components_
                ri, ci = linear_sum_assignment(-np.abs(Vref @ V.T))                           # match to the reference
                V = V[ci]
                V = V * np.sign(np.sum(Vref * V, axis=1))[:, None]                            # and sign-align
                Yf = (Zf - Zf.mean(0)) @ V.T
                Yp = (Zp - Zp.mean(0)) @ V.T
                tot = float(np.sum((Zp - Zp.mean(0)) ** 2))
                vh.append([float(np.sum(Yp[:, j] ** 2) / tot) for j in range(NPC)])
                rel.append([float(np.sum(Yf[:, j] * Yp[:, j]) / tot) for j in range(NPC)])
                acc.append(Yp.reshape(len(conds), NB, NPC))
        Y = np.mean(acc, axis=0); Ysd = np.std(acc, axis=0)
        vh = np.mean(vh, axis=0); rel = np.mean(rel, axis=0)
        ident, eta_t = {}, {}
        for nm, sgn in FACTORS.items():
            if setname == 'DPA' and nm == 'gng':
                continue
            g = np.array([sgn(cd) for cd in conds]); g = g - g.mean(); g = g / (np.linalg.norm(g) + 1e-12)
            et = np.zeros((NB, NPC))
            for b in range(NB):
                for j in range(NPC):
                    y = Y[:, b, j] - Y[:, b, j].mean(); tot_ = float(y @ y)
                    et[b, j] = float((g @ y) ** 2 / tot_) if tot_ > 1e-12 else 0.0
            eta_t[nm] = et
            ident[nm] = [float(et[FWIN[nm], j].mean()) for j in range(NPC)]
        RES[(setname, stage)] = dict(Y=Y, Ysd=Ysd, vh=vh, rel=rel, ident=ident, eta_t=eta_t, conds=conds)
        print(f'{setname:4s} {stage:6s}  held-out variance PC1-3: ' + ' '.join(f'{100*v:.0f}%' for v in vh) +
              '   |  reliable (cross-term): ' + ' '.join(f'{100*v:.0f}%' for v in rel))
        for nm, v in ident.items():
            print(f'      eta^2 of {nm:6s} at its window: ' + '  '.join(f'PC{j+1} {v[j]:.2f}' for j in range(NPC)))

rows = [(s, st) for s in SETS for st in STAGES]
fig, axs = plt.subplots(len(rows), 4, figsize=(13.0, 2.9 * len(rows)))
for r, (setname, stage) in enumerate(rows):
    D = RES[(setname, stage)]; Y, Ysd, conds, ident = D['Y'], D['Ysd'], D['conds'], D['ident']
    for j in range(NPC):
        ax = axs[r, j]
        for t0, t1, col, _ in EPOCHS:
            ax.axvspan(t0, t1, color=col, alpha=0.10, lw=0)
        for ci, cd in enumerate(conds):
            ax.plot(T, Y[ci, :, j], '-' if cd[1] == cd[2] else '--', color=SAMPC[cd[1]], lw=1.2, alpha=0.95, zorder=3)
            ax.fill_between(T, Y[ci, :, j] - Ysd[ci, :, j], Y[ci, :, j] + Ysd[ci, :, j], color=SAMPC[cd[1]], alpha=0.10, lw=0, zorder=2)
            if setname == 'dual':
                ax.scatter(T[-1], Y[ci, -1, j], s=46, color=GNGC[cd[0]], marker='*', edgecolors='k', linewidths=0.4, zorder=6)
        ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.set_xlim(0, 13.5)
        best = max(ident, key=lambda nm: ident[nm][j])
        ax.set_title(f'{setname} · {"naïve" if stage == "Naive" else "expert"} · PC{j+1} '
                     f'({100*D["rel"][j]:.0f}% reliable)  codes {best} (eta2 {ident[best][j]:.2f})', loc='left', fontsize=TITLE_FS)
        if j == 0:
            ax.set_ylabel('held-out projection (z)')
        if r == len(rows) - 1:
            ax.set_xlabel('time (s)')
    ax = axs[r, 3]
    for ci, cd in enumerate(conds):
        ax.plot(Y[ci, :, 0], Y[ci, :, 1], '-' if cd[1] == cd[2] else '--', color=SAMPC[cd[1]], lw=1.2, alpha=0.9, zorder=3)
        ax.scatter(Y[ci, 0, 0], Y[ci, 0, 1], s=16, color=SAMPC[cd[1]], marker='o', lw=0, zorder=5)
        for t0, _, col, _2 in EPOCHS:
            b = int(t0 * 6); ax.scatter(Y[ci, b, 0], Y[ci, b, 1], s=13, color=col, marker='s', lw=0, zorder=6)
        if setname == 'dual':
            ax.scatter(Y[ci, -1, 0], Y[ci, -1, 1], s=52, color=GNGC[cd[0]], marker='*', edgecolors='k', linewidths=0.4, zorder=7)
    ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.axvline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('PC1 (z)'); ax.set_ylabel('PC2 (z)')
    ax.set_title(f'{setname} · {"naïve" if stage == "Naive" else "expert"} · PC1 × PC2', loc='left', fontsize=TITLE_FS)
hs = [mlines.Line2D([0], [0], color=SAMPC[s], label=f'sample {"A" if s == 0 else "B"}') for s in SAMPC] + \
     [mlines.Line2D([0], [0], color='0.4', ls='-', label='match'), mlines.Line2D([0], [0], color='0.4', ls='--', label='nonmatch')] + \
     [mlines.Line2D([0], [0], marker='*', ls='none', ms=9, color=GNGC[g], mec='k', mew=0.4, label=('Go' if g == 'DualGo' else 'NoGo') + ' (trial end)') for g in GNGC] + \
     [mlines.Line2D([0], [0], marker='s', ls='none', color=col, label=nm) for _, _, col, nm in EPOCHS]
fig.legend(handles=hs, loc='lower center', ncol=6, frameon=False, fontsize=6.5)
fig.suptitle(f'Cross-validated unsupervised axes: basis fitted on one half of the trials, trajectories projected from the other '
             f'({NSPLIT} splits x 2 directions; band = s.d. over splits)', fontsize=9)
fig.tight_layout(rect=(0, 0.035, 1, 0.975))
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/unsup_axes_cv.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/unsup_axes_cv.svg', bbox_inches='tight')
pickle.dump(RES, open(f'{OUT}/unsup_axes_cv.pkl', 'wb'))
print('\nsaved', f'{OUT}/png/unsup_axes_cv.png')
