"""
fig_behavior_dual_cost_trials.py — the dual-task cost / trade-off analysis at the
TRIAL level (each trial a data point; mouse as cluster), rather than per-mouse means.

GEE logistic regression, clustered by mouse (exchangeable), recorded cohort, laser OFF:
  A  Dual-task cost.   DPA-correct ~ dual        (pure=0 vs dual=1).  OR<1 = cost.
  B  Trial coupling.   DPA-correct ~ GNG-correct (dual trials).  OR>1 = shared
                       good/bad-trial engagement; OR<1 = a within-trial trade-off.

Forest of odds ratios (95% CI, log-x), 4 rows each = {all, unpaired} × {Naive, Expert}.
Filled = p<0.05.  This is the powered within-animal test the n=9 between-animal
scatter can't give.

Output: figures/overlaps/behavior/{png,svg}/behavior_dual_cost_trials.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_dual_cost_trials.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set_context("poster")
sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.8,
})

STAGE_C = {'Naive': '#888888', 'Expert': '#332288'}

ON = '--on' in sys.argv[1:]
LASVAL = 1.0 if ON else 0.0
LASLAB = 'ON' if ON else 'OFF'
LASSFX = '_on' if ON else ''

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[(y.target == 'choice') & (y.laser == LASVAL)].copy()
d['dual'] = d.tasks.isin(['DualGo', 'DualNoGo']).astype(int)


def gee_or(df, formula, term):
    df = df.dropna(subset=['performance', term] if term != 'dual' else ['performance'])
    m = smf.gee(formula, groups=df['mouse'], data=df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p, int(df.shape[0])


# rows top→bottom: Expert-unpaired, Naive-unpaired, Expert-all, Naive-all
ROWS = [('Expert', 'unpaired'), ('Naive', 'unpaired'), ('Expert', 'all'), ('Naive', 'all')]


def subset(stage, scope):
    ds = d[d.stage == stage]
    return ds if scope == 'all' else ds[ds.pair == 0]


PANELS = [
    ('A  Dual-task cost', 'performance ~ dual', 'dual', lambda ds: ds,
     'OR (dual vs pure DPA)', 'OR < 1  =  dual-task cost'),
    ('B  Trial coupling', 'performance ~ odr_perf', 'odr_perf', lambda ds: ds[ds.dual == 1],
     'OR (GNG-correct)', 'OR > 1 = shared engagement   ·   OR < 1 = trade-off'),
]

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (title, formula, term, prep, xlab, note) in zip(axes, PANELS):
    for i, (stage, scope) in enumerate(ROWS):
        ds = prep(subset(stage, scope))
        orr, lo, hi, p, n = gee_or(ds, formula, term)
        sig = p < 0.05
        c = STAGE_C[stage]
        ax.plot([lo, hi], [i, i], '-', color=c, lw=2, zorder=2)
        ax.scatter(orr, i, s=110, facecolor=c if sig else 'w', edgecolor=c,
                   linewidths=1.8, zorder=3)
        ax.text(hi * 1.04, i, f'OR={orr:.2f}  p={p:.3f}{"*" if sig else ""}  (n={n})',
                va='center', ha='left', fontsize=8.5, color='0.25')
    ax.axvline(1.0, ls='--', color='k', lw=0.9, zorder=1)
    ax.set_xscale('log')
    ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0.45, 6.0)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f'{st}\n{sc}' for st, sc in ROWS], fontsize=9)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlabel(f'{xlab}\n{note}', fontsize=11)
    ax.set_title(title, loc='left', fontweight='bold')

fig.suptitle(f'Dual-task cost & trial coupling at the trial level '
             f'(GEE logistic, clustered by mouse, laser {LASLAB})', fontsize=12, y=1.02)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_dual_cost_trials{LASSFX}.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
