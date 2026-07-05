"""
fig_behavior_pareto.py — is the DPA×GNG population on a Pareto frontier?  Recorded
cohort, laser OFF, per-animal DPA vs GNG performance with the Pareto front made
explicit (non-dominated animals highlighted + the frontier staircase drawn).

Two panels Naive | Expert.  Default = unpaired-only both axes (the subset that
"looks" like a trade-off); pass --all for all trials.

A point is non-dominated (on the front) if no other animal beats it on BOTH DPA and
GNG.  A genuinely Pareto-optimal population would be MOSTLY on the front along a
descending envelope; a dominated interior means the opposite.

Convention: tab10 per-mouse colours, ● Jaws / ▲ ChR / ■ ACC markers.  Front animals
= full size + black ring; dominated = faded + small.  Grey staircase = the frontier;
grey shading = the region dominated by the front.

Output: figures/overlaps/behavior/{png,svg}/behavior_pareto[_all].{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_pareto.py [--all]
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

sns.set_context("poster")
sns.set_style("ticks")
E_RC = {
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
}

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
pal_mice = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}
GROUP = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
         **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}

ALLTR = '--all' in sys.argv[1:]
ON = '--on' in sys.argv[1:]
LASVAL = 1.0 if ON else 0.0
LASLAB = 'ON' if ON else 'OFF'
LASSFX = '_on' if ON else ''
SUFFIX = '' if ALLTR else '_unpaired'
NOTE = 'all trials' if ALLTR else 'unpaired trials only'

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[(y.target == 'choice') & (y.laser == LASVAL)].copy()
DPA_M = (d.tasks == 'DPA') if ALLTR else (d.tasks == 'DPA') & (d.pair == 0)
DUAL_M = (d.tasks.isin(['DualGo', 'DualNoGo']) if ALLTR
          else d.tasks.isin(['DualGo', 'DualNoGo']) & (d.pair == 0))


def per_mouse(mask, col, stage):
    sub = d[mask & (d.stage == stage)]
    out = {}
    for m in ALL_MICE:
        v = sub.loc[sub.mouse == m, col].dropna()
        if len(v):
            out[m] = v.mean()
    return out


def pareto_front(pts):
    """Non-dominated set for max-max (higher DPA & higher GNG both better)."""
    front = []
    for m, (x, yy) in pts.items():
        if not any(n != m and a >= x and b >= yy and (a > x or b > yy)
                   for n, (a, b) in pts.items()):
            front.append(m)
    return front


OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)

xlab = ('DPA' if ALLTR else 'DPA unpaired') + ' performance'
ylab = ('GNG' if ALLTR else 'GNG unpaired') + ' performance'

with plt.rc_context(E_RC):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), sharex=True, sharey=True)
    # global limits
    allv = []
    STG = {}
    for stage in STAGES:
        xd = per_mouse(DPA_M, 'performance', stage)
        yd = per_mouse(DUAL_M, 'odr_perf', stage)
        pts = {m: (xd[m], yd[m]) for m in ALL_MICE if m in xd and m in yd}
        STG[stage] = pts
        allv += [c for v in pts.values() for c in v]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08
    lim = (max(0.0, lo - pad), min(1.02, hi + pad))

    for ax, stage in zip(axes, STAGES):
        pts = STG[stage]
        front = pareto_front(pts)
        ax.plot(lim, lim, ls='--', color='0.8', lw=0.8, zorder=0)

        # frontier staircase (front points sorted by x asc → y desc) + dominated shading
        fp = sorted((pts[m] for m in front), key=lambda p: p[0])
        stair_x, stair_y = [lim[0]], [fp[0][1]]
        for (x, yy) in fp:
            stair_x += [x, x]; stair_y += [stair_y[-1], yy]
        stair_x += [fp[-1][0]]; stair_y += [lim[0]]
        ax.plot(stair_x, stair_y, color='0.45', lw=1.6, zorder=2)
        ax.fill_between(stair_x, lim[0], stair_y, color='0.5', alpha=0.07, zorder=0)

        for m, (x, yy) in pts.items():
            on = m in front
            ax.scatter(x, yy, marker=GMARKER[GROUP[m]],
                       s=150 if on else 55,
                       facecolors=MOUSE_COLOR[m],
                       edgecolors='k' if on else 'w',
                       linewidths=1.6 if on else 0.6,
                       alpha=1.0 if on else 0.4,
                       zorder=6 if on else 4,
                       label=m if ax is axes[1] else None)
        ax.text(0.03, 0.03, f'Pareto front: {len(front)}/{len(pts)}\n'
                f'dominated: {len(pts) - len(front)}/{len(pts)}',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color='0.25')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(xlab); ax.set_title(stage, fontweight='bold')
    axes[0].set_ylabel(ylab)

    handles = [mlines.Line2D([0], [0], marker='o', color='k', mfc='w', mec='k',
                             ls='none', ms=10, label='on front (ringed)'),
               mlines.Line2D([0], [0], marker='o', color='0.6', ls='none', ms=6,
                             alpha=0.5, label='dominated (faded)')]
    handles += [mlines.Line2D([0], [0], marker='o', color=MOUSE_COLOR[m], ls='none',
                              ms=8, label=m) for m in ALL_MICE]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, loc='upper left',
                   bbox_to_anchor=(1.01, 1),
                   title='mouse (● Jaws / ▲ ChR / ■ ACC)', title_fontsize=8)
    fig.suptitle(f'Pareto frontier of DPA vs GNG per animal — laser {LASLAB}, {NOTE}'
                 + (f'  (n={len(STG["Expert"])} mice — ACC has no laser)' if ON else ''),
                 fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/behavior_pareto{SUFFIX}{LASSFX}.{ext}'
        fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
    plt.close(fig)
