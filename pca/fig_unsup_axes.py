"""fig_unsup_axes.py — UNSUPERVISED axes of the manifold, and what they turn out to code (Leon 2026-09-18).

The axes here never see a condition label. They are the principal axes of the condition-mean state cloud itself:
per trial set and stage, the per-bin condition means (half 1 of the trials, results.pkl['CMBIN_H1']) with the
condition-independent component removed, per-neuron stage-level scaling, then PCA over the (condition x time) states.
PC1, PC2, PC3 are directions in neuron space found with no knowledge of which condition is which.

Trajectories are those states projected back on PC1-PC3. The identity of each axis is then asked AFTERWARDS, and
cross-validated: each axis's score across conditions is decomposed by eta^2 onto the design factors (sample, GNG, choice, test) —
the measure of Fig. 2d. Raw cosine between independently fitted high-dimensional directions is attenuated to near
zero and was rejected for this. The axes themselves are found without labels; only their naming uses them.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_unsup_axes.py
Output: figures/pseudo/dimensionality/{png,svg}/unsup_axes.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.decomposition import PCA
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
# contrast axes for the post-hoc labelling only, each at its own window
CONTRASTS = {'sample': (lambda c: 1.0 if c[1] == 0 else -1.0, slice(33, 39)),
             'gng': (lambda c: 1.0 if c[0] == 'DualGo' else -1.0, slice(33, 39)),
             'choice': (lambda c: 1.0 if c[1] == c[2] else -1.0, slice(54, 63)),
             'test': (lambda c: 1.0 if c[2] == 0 else -1.0, slice(54, 63))}

R = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
H1 = R['CMBIN_H1']; FIT = R['TRAJ_FIT']
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]; tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0); sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def half0_means(stage, wn):
    """Condition means of the DISJOINT half 0 (the half the trajectories are NOT taken from)."""
    M = AW[wn]; out = np.zeros((12, N))
    for ci in range(12):
        acc = np.zeros(N); cnt = np.zeros(N)
        for m in MICE:
            idx = FIT.get((m, stage, ci), [])
            if len(idx) == 0:
                continue
            val = VALIDIX[(m, stage)]; v = np.nanmean(M[np.ix_(np.asarray(idx), val)], 0)
            acc[val] += np.nan_to_num(v); cnt[val] += 1
        out[ci] = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return out


print('unsupervised axes: PCA of the half-1 condition-mean states; identity checked against half-0 contrasts\n')
RES = {}
for setname, conds in SETS.items():
    sel = [ALL12.index(c) for c in conds]
    for stage in STAGES:
        CM = np.asarray(H1[stage], float)[sel]                       # (ncond, N, 84), half 1
        k = np.ones(SM) / SM
        CM = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 2, CM)
        CM = CM - CM.mean(0, keepdims=True)                          # drop the condition-independent component
        sd = neuron_scale(stage, AW['md'])
        CM = CM / sd[None, :, None]
        Z = np.nan_to_num(CM.transpose(0, 2, 1).reshape(-1, N))      # (ncond*84, N)
        pc = PCA(NPC, random_state=0).fit(Z)
        Y = pc.transform(Z).reshape(len(conds), NB, NPC)
        # post-hoc identity of each unsupervised axis: eta^2 of its score across conditions, per factor
        # (raw cosine between independently fitted high-D directions is attenuated to ~0 and says nothing;
        #  eta^2 asks what fraction of the axis's across-condition variance each design factor explains)
        FAC = {'sample': lambda c: 1.0 if c[1] == 0 else -1.0, 'gng': lambda c: 1.0 if c[0] == 'DualGo' else -1.0,
               'choice': lambda c: 1.0 if c[1] == c[2] else -1.0, 'test': lambda c: 1.0 if c[2] == 0 else -1.0}
        if setname == 'DPA':
            FAC.pop('gng')
        ident = {}; eta_t = {}
        for nm, sgn in FAC.items():
            g = np.array([sgn(cd) for cd in conds]); g = g - g.mean(); g = g / (np.linalg.norm(g) + 1e-12)
            et = np.zeros((NB, NPC))
            for b in range(NB):
                for j in range(NPC):
                    y = Y[:, b, j]; y = y - y.mean()
                    tot = float(y @ y)
                    et[b, j] = float((g @ y) ** 2 / tot) if tot > 1e-12 else 0.0
            eta_t[nm] = et
            win = slice(33, 39) if nm in ('sample', 'gng') else slice(54, 63)
            ident[nm] = [float(et[win, j].mean()) for j in range(NPC)]
        RES[(setname, stage)] = dict(Y=Y, ev=pc.explained_variance_ratio_, ident=ident, eta_t=eta_t, conds=conds)
        print(f'{setname:4s} {stage:6s}  variance PC1-3: ' + ' '.join(f'{100*v:.0f}%' for v in pc.explained_variance_ratio_[:NPC]))
        for nm, v in ident.items():
            print(f'      eta^2 of the {nm:6s} factor at its window: ' + '  '.join(f'PC{j+1} {v[j]:.2f}' for j in range(NPC)))

# ── figure ────────────────────────────────────────────────────────────────────────────────────
rows = [(s, st) for s in SETS for st in STAGES]
fig, axs = plt.subplots(len(rows), 4, figsize=(13.0, 2.9 * len(rows)))
for r, (setname, stage) in enumerate(rows):
    D = RES[(setname, stage)]; Y = D['Y']; conds = D['conds']; ev = D['ev']; ident = D['ident']
    for j in range(NPC):
        ax = axs[r, j]
        for t0, t1, col, _ in EPOCHS:
            ax.axvspan(t0, t1, color=col, alpha=0.10, lw=0)
        for ci, cd in enumerate(conds):
            ax.plot(T, Y[ci, :, j], '-' if cd[1] == cd[2] else '--', color=SAMPC[cd[1]], lw=1.2, alpha=0.95, zorder=3)
            if setname == 'dual':
                ax.scatter(T[-1], Y[ci, -1, j], s=46, color=GNGC[cd[0]], marker='*', edgecolors='k', linewidths=0.4, zorder=6)
        ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.set_xlim(0, 13.5)
        best = max(ident, key=lambda nm: ident[nm][j])
        ax.set_title(f'{setname} · {"naïve" if stage == "Naive" else "expert"} · PC{j+1} ({100*ev[j]:.0f}%)  codes {best} (η² {ident[best][j]:.2f})', loc='left', fontsize=TITLE_FS)
        if j == 0:
            ax.set_ylabel('projection (z)')
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
fig.suptitle("Unsupervised axes: principal axes of the condition-mean manifold (no labels used), with their identity read off afterwards (eta-squared of each design factor)", fontsize=9)
fig.tight_layout(rect=(0, 0.035, 1, 0.975))
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/unsup_axes.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/unsup_axes.svg', bbox_inches='tight')
pickle.dump(RES, open(f'{OUT}/unsup_axes.pkl', 'wb'))
print('\nsaved', f'{OUT}/png/unsup_axes.png')
