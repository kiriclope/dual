"""
fig_behavior_dpa_vs_gng.py — per-animal DPA performance vs GNG performance,
recorded cohort (9 mice), in the main-figure scatter convention.

Two panels = Naive | Expert.  In each:
  x = DPA performance   (fraction correct over all DPA trials)
  y = GNG performance   (odr_perf over all dual Go/NoGo trials)
One dot per mouse — per-mouse tab10 colour, group marker (● Jaws / ▲ ChR / ■ ACC),
white edge — with the y=x diagonal, regression + 95% CI band, and across-animal
Pearson r / Spearman ρ (+ star) exactly as the panel-E / scatter_perf figures.

Emitted twice: laser OFF (`_off`, laser==0, the convention default) and laser ON
(`_on`, laser==1).

Output: figures/overlaps/behavior/{png,svg}/behavior_dpa_vs_gng_{off,on}.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_dpa_vs_gng.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, linregress

# ── Style — matches panel E (plot_scatter_laser.py) / scatter_perf E_RC ─────────
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

# ── Config ────────────────────────────────────────────────────────────────────
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
# per-animal colour + group marker — IDENTICAL convention to scatter_perf / panel E
pal_mice = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}
GROUP = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
         **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}     # ● Jaws / ▲ ChR / ■ ACC

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'choice'].copy()      # one row per trial

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)


def fmt_p(p):
    return f'p={p:.3f}' if p >= 0.001 else 'p<0.001'


def regression_line(ax, xs, ys, color='0.35'):
    """Thin across-animal regression line (no CI band — this is a diagonal x-vs-y
    plot, so the wide n=9 band of the Δ-vs-Δ figures would just clutter it)."""
    valid = ~(np.isnan(xs) | np.isnan(ys))
    if valid.sum() < 3:
        return
    xv, yv = xs[valid], ys[valid]
    slope, intercept, *_ = linregress(xv, yv)
    x_line = np.array([xv.min(), xv.max()])
    ax.plot(x_line, slope * x_line + intercept, color=color, lw=1.3, ls='-', zorder=4)


def per_mouse(mask, col, stage, laser_val):
    """Mean of `col` over the mask trial subset for one mouse-stage, given laser."""
    sub = d[mask & (d.stage == stage) & (d.laser == laser_val)]
    out = {}
    for mouse in ALL_MICE:
        v = sub.loc[sub.mouse == mouse, col].dropna()
        out[mouse] = v.mean() if len(v) else np.nan
    return out


IS_DPA = d.tasks == 'DPA'
IS_DUAL = d.tasks.isin(['DualGo', 'DualNoGo'])


def make(laser_val, tag, laser_label):
    print(f'\n=== laser {laser_label} ===')
    with plt.rc_context(E_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
        # global limits across both panels
        allv = []
        data = {}
        for stage in STAGES:
            xd = per_mouse(IS_DPA, 'performance', stage, laser_val)
            yd = per_mouse(IS_DUAL, 'odr_perf', stage, laser_val)
            data[stage] = (xd, yd)
            allv += [v for v in list(xd.values()) + list(yd.values()) if np.isfinite(v)]
        lo, hi = (min(allv), max(allv)) if allv else (0.45, 1.0)
        pad = (hi - lo) * 0.08 or 0.05
        lim = (max(0.0, lo - pad), min(1.02, hi + pad))

        for ax, stage in zip(axes, STAGES):
            xd, yd = data[stage]
            xs = np.array([xd[m] for m in ALL_MICE], float)
            ys = np.array([yd[m] for m in ALL_MICE], float)
            # y=x diagonal + chance lines
            ax.plot(lim, lim, ls='--', color='k', lw=0.8, zorder=1)
            if lim[0] < 0.5 < lim[1]:
                ax.axhline(0.5, ls=':', color='0.7', lw=0.8, zorder=0)
                ax.axvline(0.5, ls=':', color='0.7', lw=0.8, zorder=0)
            regression_line(ax, xs, ys)
            for xx, yy, m in zip(xs, ys, ALL_MICE):
                if np.isfinite(xx) and np.isfinite(yy):
                    ax.scatter(xx, yy, color=MOUSE_COLOR[m], marker=GMARKER[GROUP[m]],
                               s=90, edgecolors='w', linewidths=0.6, zorder=5,
                               label=m if ax is axes[1] else None)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() >= 3:
                r_p, p_p = pearsonr(xs[ok], ys[ok])
                r_s, p_s = spearmanr(xs[ok], ys[ok])
                # stats + star in the empty lower-right corner (points sit above the diagonal)
                ax.text(0.97, 0.05, f'n={ok.sum()}   r={r_p:+.2f} {fmt_p(p_p)}\n'
                        f'ρ={r_s:+.2f} {fmt_p(p_s)}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8.5, color='0.3')
                star = '*' if p_s < 0.05 else 'n.s.'
                ax.text(0.97, 0.22, star, transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=18, fontweight='bold',
                        color='k' if p_s < 0.05 else '0.55')
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.set_xlabel('DPA performance')
            ax.set_title(stage, fontweight='bold')
        axes[0].set_ylabel('GNG performance')
        axes[1].legend(frameon=False, fontsize=8, loc='upper left',
                       bbox_to_anchor=(1.01, 1),
                       title='mouse (● Jaws / ▲ ChR / ■ ACC)', title_fontsize=8)
        fig.suptitle(f'DPA vs GNG performance per animal — laser {laser_label}',
                     fontsize=12, y=1.02)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            p = f'{OUT}/{ext}/behavior_dpa_vs_gng_{tag}.{ext}'
            fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
        plt.close(fig)


make(0.0, 'off', 'OFF')
make(1.0, 'on', 'ON')
