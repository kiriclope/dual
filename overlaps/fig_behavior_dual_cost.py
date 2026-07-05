"""
fig_behavior_dual_cost.py — is DPA↔GNG a capacity trade-off?  Recorded cohort,
laser-OFF trials, main-figure scatter convention.

Two diagonal (y=x) scatters, one dot per mouse per stage (Naive open, Expert filled,
joined by a thin line = the learning shift):

  A  Dual-task cost.   x = DPA accuracy on PURE DPA trials,
                       y = DPA accuracy on DUAL trials.
                       Below the diagonal = adding GNG costs DPA.

  B  Trial coupling.   x = DPA accuracy | GNG-error trial,
                       y = DPA accuracy | GNG-correct trial   (dual trials only).
                       Above the diagonal = DPA better on trials where GNG is also
                       correct → a shared good/bad-trial engagement axis, i.e. the
                       OPPOSITE sign of a capacity trade-off.

Colours per mouse (tab10); marker = opto group (● Jaws / ▲ ChR / ■ ACC); white edge.
Corner stats: per-stage mean effect + paired-t p across mice (n=9).

Output: figures/overlaps/behavior/{png,svg}/behavior_dual_cost.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_dual_cost.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import ttest_rel

sns.set_context("poster")
sns.set_style("ticks")
E_RC = {
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'lines.linewidth': 1.5,
}

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
pal_mice = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}
GROUP = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
         **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[(y.target == 'choice') & (y.laser == 0)].copy()      # OFF trials, one row/trial

UNPAIRED = '--unpaired' in sys.argv[1:]
SUFFIX = '_unpaired' if UNPAIRED else ''
DLAB = 'DPA(unp)' if UNPAIRED else 'DPA'

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)


def compute():
    """Return {test: {stage: {mouse: (x, y)}}} for the two tests."""
    A, B = {s: {} for s in STAGES}, {s: {} for s in STAGES}
    for stage in STAGES:
        ds = d[d.stage == stage]
        for m in ALL_MICE:
            dm = ds[ds.mouse == m]
            puredf = dm[dm.tasks == 'DPA']
            dual = dm[dm.tasks.isin(['DualGo', 'DualNoGo'])]
            if UNPAIRED:                                    # DPA-unpaired trials only
                puredf = puredf[puredf.pair == 0]
                dual = dual[dual.pair == 0]
            pure = puredf.performance.dropna()
            dualp = dual.performance.dropna()
            if len(pure) and len(dualp):
                A[stage][m] = (pure.mean(), dualp.mean())
            a = dual.performance.values.astype(float)
            b = dual.odr_perf.values.astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if (b == 1).any() and (b == 0).any():
                B[stage][m] = (a[b == 0].mean(), a[b == 1].mean())
    return {'A': A, 'B': B}


def paired_stats(pts):
    """mean(y-x) and paired-t p across mice for one stage's {mouse:(x,y)}."""
    xs = np.array([v[0] for v in pts.values()])
    ys = np.array([v[1] for v in pts.values()])
    if len(xs) < 3:
        return np.nan, np.nan, len(xs)
    return float(np.mean(ys - xs)), float(ttest_rel(ys, xs).pvalue), len(xs)


data = compute()
PANELS = [
    ('A', f'Dual-task cost ({DLAB})', f'{DLAB} accuracy · pure', f'{DLAB} accuracy · dual',
     'below diagonal = dual-task cost'),
    ('B', f'Trial coupling ({DLAB} × GNG)', f'{DLAB} acc | GNG-error', f'{DLAB} acc | GNG-correct',
     'above diagonal = shared engagement,\nnot a trade-off'),
]

with plt.rc_context(E_RC):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7))
    for ax, (key, title, xlab, ylab, note) in zip(axes, PANELS):
        allv = [c for stage in STAGES for v in data[key][stage].values() for c in v]
        lo, hi = (min(allv), max(allv)) if allv else (0.5, 1.0)
        pad = (hi - lo) * 0.10 or 0.05
        lim = (max(0.0, lo - pad), min(1.02, hi + pad))
        ax.plot(lim, lim, ls='--', color='k', lw=0.8, zorder=1)

        for m in ALL_MICE:
            xy = [data[key][s].get(m) for s in STAGES]
            if all(p is not None for p in xy):
                ax.plot([xy[0][0], xy[1][0]], [xy[0][1], xy[1][1]], '-',
                        color=MOUSE_COLOR[m], lw=0.8, alpha=0.5, zorder=3)
            for stage, p in zip(STAGES, xy):
                if p is None:
                    continue
                face = 'w' if stage == 'Naive' else MOUSE_COLOR[m]
                ax.scatter(p[0], p[1], marker=GMARKER[GROUP[m]], s=90,
                           facecolors=face, edgecolors=MOUSE_COLOR[m],
                           linewidths=1.3, zorder=5)
        # per-stage summary stats
        lines = []
        for stage in STAGES:
            md, pv, n = paired_stats(data[key][stage])
            eff = 'Δ' if key == 'A' else 'Δ'
            lines.append(f'{stage}: {eff}={md:+.3f} p={pv:.3f} (n={n})')
        ax.text(0.97, 0.05, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color='0.3')
        ax.text(0.03, 0.97, note, transform=ax.transAxes, ha='left', va='top',
                fontsize=8.5, color='0.45', style='italic')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f'{key}  {title}', loc='left', fontweight='bold')

    # legend: stage fill + group marker + mouse colour
    stage_h = [mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none',
                             ms=8, label='Naive (open)'),
               mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none',
                             ms=8, label='Expert (filled)')]
    mouse_h = [mlines.Line2D([0], [0], marker='o', color=MOUSE_COLOR[m], ls='none',
                             ms=8, label=m) for m in ALL_MICE]
    axes[1].legend(handles=stage_h + mouse_h, frameon=False, fontsize=8,
                   loc='upper left', bbox_to_anchor=(1.01, 1),
                   title='stage / mouse (● Jaws / ▲ ChR / ■ ACC)', title_fontsize=8)
    fig.suptitle(f'{DLAB}↔GNG is not a capacity trade-off (recorded cohort, laser OFF'
                 f'{", unpaired only" if UNPAIRED else ""})', fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/behavior_dual_cost{SUFFIX}.{ext}'
        fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
    plt.close(fig)
