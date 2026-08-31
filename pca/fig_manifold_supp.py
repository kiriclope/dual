"""fig_manifold_supp.py — ED companion to Figs 2-4 (created 2026-08-31, the "redistribute"
restructure): the per-mouse panels that left the main figures.

  A  the FOUR CODES over time (2x4: sample / dist / test / choice on their own CCGD axes) — the
     original Fig-3 panel A, moved here 2026-08-31 when the main adopted the task-split trace row
     (DPA|Go|NoGo x sample/choice). This is the definitional reference for the dist and test
     codes used by Fig 3 C-F.
  B  per-mouse CCGP, Naive vs Expert (sample / dist / test / choice) — abstraction is present from
     the start and NOT built by learning. NO title verdicts: the test-code comparison flips with
     the PCA knob (p=.04 nopca / .73 pca20) — exact p stays in the annotations only.
  C  per-mouse cross-task generalisation companions to Fig 2's panel E (within- vs cross-task
     accuracy, Expert). Raw accuracies on purpose — the chance-corrected ratio is unusable per
     animal (within-task sits near chance for sample/test, so the denominator ~0 and the ratio
     explodes). Distance below the unity line IS the generalisation loss.

Canonical no-PCA caches (overlaps permouse_ccgp_cache.pkl + pca results.pkl PM_GEN_nopca +
ORIG_TRACES/ORIG_SPECS/ORIG_XTIME).
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


# ══ A — the four codes over time (moved from main Fig 3 panel A, 2026-08-31) ══════════════════
SAMPC = {0: '#332288', 1: '#44AA99'}
CODE_ORDER = ['sample', 'dist', 'test', 'choice']
CODE_NAME = {'sample': 'sample', 'GNG': 'dist', 'gng': 'dist', 'distractor': 'dist',
             'test': 'test', 'lick': 'choice', 'action': 'choice', 'choice': 'choice'}
EVENTS = [('sample', 2.0, 3.0, SAMPC[0]), ('distractor', 4.5, 5.5, '#cc3311'),
          ('GNG cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]


def panel_traj(fig, gsT):
    """2 rows (Naive | Expert) x 4 codes — the original Fig-3 panel A, replayed from ORIG_TRACES
    (exp_traj_orig.py; per-mouse CCGD projections, baseline zero, shared per-mouse unit).
    Y-limits shared per code column across stages."""
    TR = RES['ORIG_TRACES']; xt = np.asarray(RES['ORIG_XTIME'])
    SP = sorted(RES['ORIG_SPECS'], key=lambda sp: CODE_ORDER.index(CODE_NAME[sp['code']]))
    YLK = {}
    for k, spec in enumerate(SP):
        lo, hi = 0.0, 0.0
        for stage in ['Naive', 'Expert']:
            for lv in spec['levels']:
                M = np.asarray(TR[(stage, spec['code'], int(lv))], dtype=float)
                if not len(M):
                    continue
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                lo = min(lo, (mu - se).min()); hi = max(hi, (mu + se).max())
        pad = 0.05 * (hi - lo)
        YLK[k] = (lo - pad, hi + pad)
    axes = []
    for r, stage in enumerate(['Naive', 'Expert']):
        for k, spec in enumerate(SP):
            ax = fig.add_subplot(gsT[r, k]); axes.append(ax)
            for nm, lo, hi, col in EVENTS:
                ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0)
                if r == 0 and k == 0:
                    yl = 0.905 if nm == 'distractor' else 0.98
                    ax.text((lo + hi) / 2, yl, nm, transform=ax.get_xaxis_transform(),
                            ha='center', va='top', fontsize=5.8, color=col)
            for lv, lab, col in zip(spec['levels'], spec['labels'], spec['colors']):
                M = np.asarray(TR[(stage, spec['code'], int(lv))], dtype=float)
                if not len(M):
                    continue
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                ax.plot(xt, mu, color=col, lw=1.5, label=f'{lab} (n={len(M)})', zorder=3)
                ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.20, lw=0, zorder=2)
            ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
            ax.set_ylim(*YLK[k])
            ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12])
            if r == 1:
                ax.set_xlabel('time (s)', fontsize=7)
                if k == 0:
                    ax.legend(frameon=False, fontsize=6.0, handlelength=1.2, loc='lower right')
            else:
                ax.tick_params(labelbottom=False)
                ax.set_title(f"{CODE_NAME[spec['code']]} code", loc='left', fontsize=TITLE_FS)
                if k > 0:
                    ax.legend(frameon=False, fontsize=6.0, handlelength=1.2, loc='upper left')
            ax.set_ylabel(f'{stage}\ncode depth' if k == 0 else 'code depth', fontsize=7)
    return axes[0]


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

fig = plt.figure(figsize=(9.4, 9.8))
gs = fig.add_gridspec(3, 12, height_ratios=[1.9, 1.0, 1.0], hspace=0.50,
                      left=0.08, right=0.975, top=0.955, bottom=0.055)
gsT = gs[0, 0:12].subgridspec(2, 4, wspace=0.36, hspace=0.16)
axT = panel_traj(fig, gsT)
gsA = gs[1, 0:12].subgridspec(1, 4, wspace=0.45)
axA = panel_ccgp(fig, gsA)
gsB = gs[2, 0:9].subgridspec(1, 3, wspace=0.45)
axB = panel_gen_mouse(fig, gsB)
plabel(axT, 'A', dx=-0.16); plabel(axA, 'B'); plabel(axB, 'C')

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_manifold_supp.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_supp.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_supp.png'))
