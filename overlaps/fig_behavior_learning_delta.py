"""
fig_behavior_learning_delta.py — laser EFFECT (ON−OFF) version of the behavioural
learning-curve figure.

Same 5 panels / same condition definitions as fig_behavior_learning.py, but the
y-axis is Δ performance = perf(laser ON) − perf(laser OFF), computed WITHIN each
mouse×day (interleaved laser trials), then averaged across mice.  A dashed line at
0 marks "no laser effect".

  A  Δ DPA & Δ GNG vs day            (DPA red / GNG blue)
  B  Δ Go vs Δ NoGo                   (Go blue / NoGo green)
  C  Δ paired vs Δ unpaired            (paired solid / unpaired dashed, red)
  D  Δ DPA-unpaired by task            (DPA-only red / Go blue / NoGo green, dashed)
  E  per-condition mean laser effect ± 95% CI, one-sample LMM  Δ ~ 1 + (1|mouse)
     (is the ON−OFF effect different from zero?), β on the y-axis.

7 laser mice.  --jaws (5, INHIBITION) / --chr (2, EXCITATION) restrict to one opto
group — recommended, since the two are OPPOSITE manipulations and averaging them
in the pooled (default) figure cancels real effects.

Top stars = per-day one-sample t-test of Δ vs 0 (exploratory, uncorrected).

Output: figures/overlaps/behavior/{png,svg}/behavior_learning_delta[_jaws|_chr].{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_learning_delta.py [--jaws|--chr]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

LASER_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23']
RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'
N_MIN = 4

GRP = 'Jaws' if '--jaws' in sys.argv[1:] else 'ChR' if '--chr' in sys.argv[1:] else None
if GRP:
    MICE = [m for m in LASER_MICE if m.startswith(GRP)]
    verb = 'inhibit' if GRP == 'Jaws' else 'excite'
    OUT_NAME = f'behavior_learning_delta_{GRP.lower()}'
    TITLE = f'Laser effect on behaviour Δ(ON−OFF) vs day — {GRP} {verb} (n={len(MICE)})'
else:
    MICE, OUT_NAME = LASER_MICE, 'behavior_learning_delta'
    TITLE = 'Laser effect on behaviour Δ(ON−OFF) vs day — 7 laser mice (5 Jaws + 2 ChR pooled)'

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'choice'].copy()
d = d[d.mouse.isin(MICE)]
DAYS = list(range(1, int(d.day.max()) + 1))
print(f'{OUT_NAME}: {len(d)} trials, {d.mouse.nunique()} mice, days {DAYS[0]}–{DAYS[-1]}')


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def delta_pmd(col, mask):
    """{day: {mouse: perf(ON) − perf(OFF)}} within-mouse over the `mask` trial subset."""
    m = mask.values
    df = pd.DataFrame({'v': d.loc[m, col].values, 'mouse': d.loc[m, 'mouse'].values,
                       'day': d.loc[m, 'day'].values, 'laser': d.loc[m, 'laser'].values}).dropna()
    out = {day: {} for day in DAYS}
    for (mouse, day), sub in df.groupby(['mouse', 'day']):
        off, on = sub[sub.laser == 0].v, sub[sub.laser == 1].v
        if len(off) and len(on):
            out[day][mouse] = float(on.mean() - off.mean())
    return out


def agg(pmd):
    mean, sem, n = [], [], []
    for day in DAYS:
        pm = np.array(list(pmd[day].values()))
        if len(pm):
            mean.append(pm.mean())
            sem.append(pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else 0.0)
            n.append(len(pm))
        else:
            mean.append(np.nan); sem.append(np.nan); n.append(0)
    return np.array(DAYS, float), np.array(mean), np.array(sem), np.array(n)


def plot_line(ax, pmd, color, label, ls='-'):
    x, m, s, _ = agg(pmd)
    ok = ~np.isnan(m)
    ax.plot(x[ok], m[ok], ls=ls, color=color, lw=2, marker='o', ms=5,
            mfc=color, mec=color, label=label, zorder=3)
    ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)


def perday_stars(ax, pmd, color, y_star):
    """One-sample t-test of this line's per-mouse Δ vs 0, per day (exploratory).
       Stars drawn in the line colour and staggered per line so panels with
       several lines stay unambiguous."""
    for day in DAYS:
        v = np.array(list(pmd[day].values()))
        if len(v) < N_MIN or np.allclose(v, v[0]):
            continue
        p = float(ttest_1samp(v, 0.0).pvalue)
        if star(p):
            ax.text(day, y_star, star(p), ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color)


def mean_effect(pmd):
    """One-sample LMM Δ ~ 1 + (1|mouse) over per-mouse/day deltas → (mean, lo, hi, p)."""
    rows = [(mo, day, v) for day in DAYS for mo, v in pmd[day].items()]
    g = pd.DataFrame(rows, columns=['mouse', 'day', 'delta'])
    mean = g.delta.mean()
    if g.mouse.nunique() < 3:
        return mean, np.nan, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.mixedlm('delta ~ 1', g, groups=g['mouse']).fit()
    ci = res.conf_int()
    return mean, float(ci.loc['Intercept', 0]), float(ci.loc['Intercept', 1]), float(res.pvalues['Intercept'])


IS_DPA = d.tasks == 'DPA'
IS_GO  = d.tasks == 'DualGo'
IS_NOGO = d.tasks == 'DualNoGo'
IS_DUAL = IS_GO | IS_NOGO
UNP = d.pair == 0

# (panel, label, color, linestyle, col, mask)
LINES = [
    ('A', 'DPA',  RED,   '-',  'performance', IS_DPA),
    ('A', 'GNG',  BLUE,  '-',  'odr_perf',    IS_DUAL),
    ('B', 'Go',   BLUE,  '-',  'odr_perf',    IS_GO),
    ('B', 'NoGo', GREEN, '-',  'odr_perf',    IS_NOGO),
    ('C', 'paired',   RED, '-',  'performance', IS_DPA & (d.pair == 1)),
    ('C', 'unpaired', RED, '--', 'performance', IS_DPA & UNP),
    ('D', 'DPA only', RED,   '--', 'performance', UNP & IS_DPA),
    ('D', 'Go',       BLUE,  '--', 'performance', UNP & IS_GO),
    ('D', 'NoGo',     GREEN, '--', 'performance', UNP & IS_NOGO),
]

fig, (axA, axB, axC, axD, axE) = plt.subplots(1, 5, figsize=(22, 4.3))
AXES = {'A': axA, 'B': axB, 'C': axC, 'D': axD}
TITLES = {'A': 'A  Δ DPA vs GNG', 'B': 'B  Δ Go vs NoGo',
          'C': 'C  Δ paired vs unpaired', 'D': 'D  Δ unpaired, by task'}

effects = []   # (panel+label, mean, lo, hi, p) for panel E
pcount = {}    # per-panel line index, to stagger the per-day stars
for panel, label, color, ls, col, mask in LINES:
    ax = AXES[panel]
    pmd = delta_pmd(col, mask)
    plot_line(ax, pmd, color, label, ls=ls)
    idx = pcount.get(panel, 0); pcount[panel] = idx + 1
    perday_stars(ax, pmd, color, y_star=0.225 - idx * 0.02)
    m, lo, hi, p = mean_effect(pmd)
    effects.append((f'{panel} {label}', m, lo, hi, p))
    ptxt = f'p={p:.3f}{star(p) or " ns"}' if np.isfinite(p) else 'n<3'
    print(f'  {panel} {label:9s} mean Δ={m:+.3f}  {ptxt}')

for panel, ax in AXES.items():
    ax.axhline(0, ls='--', color='0.4', lw=1)
    ax.set_ylim(-0.22, 0.24)
    ax.set_xticks(DAYS)
    ax.set_xlabel('training day')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(TITLES[panel], loc='left', fontweight='bold', fontsize=11)
axA.set_ylabel('Δ performance  (ON − OFF)')

# ── Panel E — per-condition mean laser effect ± 95% CI (β on y) ────────────────
for i, (lab, m, lo, hi, p) in enumerate(effects):
    sig = np.isfinite(p) and p < 0.05
    col = 'k' if sig else '0.6'
    yerr = [[m - lo], [hi - m]] if np.isfinite(lo) else None
    axE.errorbar(i, m, yerr=yerr, fmt='o', color=col, ms=6, capsize=3, lw=1.5, zorder=3)
    if sig:
        axE.text(i, hi + 0.005, star(p), ha='center', va='bottom', fontsize=10, fontweight='bold')
axE.axhline(0, ls='--', color='0.4', lw=1)
axE.set_xticks(range(len(effects)))
axE.set_xticklabels([lab for lab, *_ in effects], rotation=45, ha='right', fontsize=8)
axE.set_xlim(-0.6, len(effects) - 0.4)
axE.set_ylabel('mean Δ performance  (ON − OFF)')
axE.set_title('E  Laser effect per condition (95% CI)', loc='left', fontweight='bold', fontsize=11)
axE.spines[['top', 'right']].set_visible(False)

fig.suptitle(TITLE, fontsize=13, y=0.99)
fig.text(0.5, 0.005,
         f'Within-mouse Δ(ON−OFF), interleaved laser trials, mean ± SEM across mice ({d.mouse.nunique()}).  '
         'Top stars: per-day one-sample t-test of Δ vs 0 (exploratory, uncorrected).  '
         'Panel E: one-sample LMM Δ ~ 1 + (1|mouse) per condition.  * p<0.05  ** p<0.01  *** p<0.001',
         ha='center', va='bottom', fontsize=8, color='0.35')
fig.tight_layout(rect=(0, 0.05, 1, 0.94))

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/{OUT_NAME}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
