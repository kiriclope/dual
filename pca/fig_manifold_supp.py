"""fig_manifold_supp.py — ED companion to Figs 2-4 (created 2026-08-31, the "redistribute"
restructure): the per-mouse panels that left the main figures.

  A  per-mouse CCGP, Naive vs Expert (sample / dist / test / choice) — abstraction is present from
     the start and NOT built by learning. NO title verdicts: the test-code comparison flips with
     the PCA knob (p=.04 nopca / .73 pca20) — exact p stays in the annotations only.
  B  per-mouse cross-task generalisation companions to Fig 2's panel E (within- vs cross-task
     accuracy, Expert). Raw accuracies on purpose — the chance-corrected ratio is unusable per
     animal (within-task sits near chance for sample/test, so the denominator ~0 and the ratio
     explodes). Distance below the unity line IS the generalisation loss.

Canonical no-PCA caches (overlaps permouse_ccgp_cache.pkl + pca results.pkl PM_GEN_nopca).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_manifold_supp.py
Output: figures/pseudo/dimensionality/{png,svg}/fig_manifold_supp.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import os, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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
CODE_NAME = {'sample': 'sample', 'GNG': 'dist', 'test': 'test', 'choice': 'choice'}
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
GROUP = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
         **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
_pal = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal[i] for i, m in enumerate(ALL_MICE)}

CCGP_CACHE = '/home/leon/dual/overlaps/figures/overlaps/ccgp/permouse_ccgp_cache.pkl'   # no-PCA build
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
PG = RES['PM_GEN_nopca']                                                # canonical no-PCA


def plabel(ax, s, dx=-0.10):
    ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')


# ══ A — per-mouse CCGP, Naive vs Expert ═══════════════════════════════════════
def panel_ccgp(fig, gsA):
    R = pd.read_pickle(CCGP_CACHE)
    axes = []
    for j, v in enumerate(['sample', 'GNG', 'test', 'choice']):         # cache keys
        ax = fig.add_subplot(gsA[0, j]); axes.append(ax)
        piv = (R[R['variable'] == v].pivot_table(index='mouse', columns='stage', values='ccgp')
               .dropna(subset=['Naive', 'Expert']))
        ax.plot([0.42, 1.0], [0.42, 1.0], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        for m, rr in piv.iterrows():
            ax.scatter(rr['Naive'], rr['Expert'], s=42, color=MOUSE_COLOR.get(m, '0.5'),
                       marker=GMARKER[GROUP.get(m, 'Jaws')], edgecolors='w', linewidths=0.5, zorder=3)
        ax.scatter(piv['Naive'].mean(), piv['Expert'].mean(), s=95, color='k', marker='D',
                   edgecolors='w', linewidths=0.6, zorder=5)
        p = float(wilcoxon(piv['Expert'], piv['Naive']).pvalue)
        ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_aspect('equal', adjustable='box')
        # no title verdicts: the test-code comparison is knob-dependent (see header)
        ax.set_title(CODE_NAME[v], loc='left', fontsize=TITLE_FS)
        ax.set_xlabel('CCGP — Naive', fontsize=7.5)
        ax.text(0.05, 0.96, f'Δ={piv.Expert.mean() - piv.Naive.mean():+.2f}\np={p:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=6, color='0.3')
        print(f'A: {v:7s} N {piv["Naive"].mean():.3f} -> E {piv["Expert"].mean():.3f}  p={p:.3f}')
    axes[0].set_ylabel('CCGP — Expert', fontsize=7.5)
    return axes[0]


# ══ B — per-mouse within- vs cross-task accuracy (Fig 2 panel-E companion) ════
def panel_gen_mouse(fig, gsB):
    axes = []
    for j, v in enumerate(['sample', 'choice', 'test']):
        ax = fig.add_subplot(gsB[0, j]); axes.append(ax)
        E3 = np.eye(3, dtype=bool)
        ax.plot([0.45, 0.95], [0.45, 0.95], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6); ax.axvline(0.5, ls=':', color='0.85', lw=0.6)
        wi, cr = [], []
        for m in ALL_MICE:
            if (m, 'Expert', v) not in PG:
                continue
            M = PG[(m, 'Expert', v)]
            x = float(np.diag(M).mean()); y = float(M[~E3].mean())
            wi.append(x); cr.append(y)
            ax.scatter(x, y, s=34, color=MOUSE_COLOR[m], marker=GMARKER[GROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        ax.scatter(np.mean(wi), np.mean(cr), s=80, color='k', marker='D', edgecolors='w',
                   linewidths=0.6, zorder=5)
        ax.set_xlim(0.45, 0.95); ax.set_ylim(0.45, 0.95); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
        ax.set_title(v, loc='left', fontsize=7)
        ax.set_xlabel('within-task', fontsize=7)
        if j == 0:
            ax.set_ylabel('cross-task', fontsize=7)
        print(f'B: {v:7s} per-mouse within {np.mean(wi):.3f}  cross {np.mean(cr):.3f}')
    return axes[0]


# (the per-mouse raw-|cos| scatters that briefly lived here as panel C moved BACK into main
#  Fig 3 panel E, 2026-08-31 user request — see fig_manifold_main.panel_e_pm)

fig = plt.figure(figsize=(9.4, 5.2))
gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.0], hspace=0.55,
                      left=0.08, right=0.975, top=0.935, bottom=0.09)
gsA = gs[0, 0:12].subgridspec(1, 4, wspace=0.45)
axA = panel_ccgp(fig, gsA)
gsB = gs[1, 0:9].subgridspec(1, 3, wspace=0.45)
axB = panel_gen_mouse(fig, gsB)
plabel(axA, 'A'); plabel(axB, 'B')

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_manifold_supp.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_supp.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_supp.png'))
