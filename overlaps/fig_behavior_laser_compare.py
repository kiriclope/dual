"""
fig_behavior_laser_compare.py — laser ON vs OFF behavioural comparison.

Companion to fig_behavior_learning.py.  Where that figure shows the learning
curves within one laser condition, this one asks the causal question directly:
does turning the laser ON change behaviour?  Because the laser is interleaved
trial-by-trial within the same sessions, ON−OFF is a within-mouse contrast.

7 laser mice (ACCM03/04 have no laser trials):
  - 5 Jaws  = optogenetic INHIBITION   (JawsM01/06/12/15/18)
  - 2 ChR   = optogenetic EXCITATION    (ChRM04/23)
These are OPPOSITE manipulations, so ON−OFF is shown per mouse with the two
groups distinguished (never silently pooled into a single mean claim).

Expert stage only (trained behaviour; matches the causal depth analysis in
plot_scatter_laser.py).  Six behavioural metrics, same definitions as the
learning-curve figure:
  DPA (perf|DPA)  GNG (odr_perf|Dual)  Go/NoGo (odr_perf|DualGo/NoGo)
  paired / unpaired (perf|DPA, pair==1/0)

Two panels:
  A  grouped mean OFF vs ON per metric (mean ± SEM across 7 mice) — do the
     absolute levels move?
  B  within-mouse ON−OFF Δ per metric: per-mouse dots (● Jaws / ▲ ChR),
     mean ± 95% CI, and a within-mouse LMM laser test  perf ~ laser + (1|mouse)
     (Wald p; pooled 7-mouse, printed + starred).  Jaws-only laser effect also
     printed for the interpretable inhibition contrast.

Output: figures/overlaps/behavior/{png,svg}/behavior_laser_offon.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_laser_compare.py
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

matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

LASER_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23']
GROUP  = {m: ('Jaws' if m.startswith('Jaws') else 'ChR') for m in LASER_MICE}
GCOL   = {'Jaws': '#332288', 'ChR': '#CC6677'}      # inhibition indigo / excitation rose
GMARK  = {'Jaws': 'o', 'ChR': '^'}
STAGE  = 'Expert'

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'choice'].copy()
d = d[d.mouse.isin(LASER_MICE) & (d.stage == STAGE)]

IS_DPA = d.tasks == 'DPA'
IS_GO  = d.tasks == 'DualGo'
IS_NOGO = d.tasks == 'DualNoGo'
IS_DUAL = IS_GO | IS_NOGO
METRICS = [                                   # (label, value col, trial mask)
    ('DPA',      'performance', IS_DPA),
    ('GNG',      'odr_perf',    IS_DUAL),
    ('Go',       'odr_perf',    IS_GO),
    ('NoGo',     'odr_perf',    IS_NOGO),
    ('paired',   'performance', IS_DPA & (d.pair == 1)),
    ('unpaired', 'performance', IS_DPA & (d.pair == 0)),
]


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def proportions(col, mask):
    """Per-(mouse,day,laser) accuracy proportions for the mixed model / means."""
    m = mask.values
    df = pd.DataFrame({'v': d.loc[m, col].values, 'mouse': d.loc[m, 'mouse'].values,
                       'day': d.loc[m, 'day'].values, 'laser': d.loc[m, 'laser'].values}).dropna()
    return df.groupby(['mouse', 'day', 'laser'], observed=True).v.mean().reset_index(name='perf')


def laser_lmm(g, mice):
    """Within-mouse LMM laser effect (ON−OFF): perf ~ laser + (1|mouse).  Wald p.
       Needs ≥3 mice for a meaningful random effect; returns (β, nan) otherwise."""
    gg = g[g.mouse.isin(mice)]
    if gg.mouse.nunique() < 3:
        b = (gg[gg.laser == 1].perf.mean() - gg[gg.laser == 0].perf.mean())
        return float(b), float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.mixedlm('perf ~ laser', gg, groups=gg['mouse']).fit()
    return float(res.params['laser']), float(res.pvalues['laser'])


labels = [m[0] for m in METRICS]
xs = np.arange(len(labels))

# groups to render: --jaws / --chr select one; default = both groups + all-7
SEL = ('Jaws',) if '--jaws' in sys.argv[1:] else ('ChR',) if '--chr' in sys.argv[1:] else \
      ('all', 'Jaws', 'ChR')
GROUP_SPEC = {
    'all':  (LASER_MICE, 'behavior_laser_offon',      '7 laser mice (5 Jaws + 2 ChR)'),
    'Jaws': ([m for m in LASER_MICE if GROUP[m] == 'Jaws'], 'behavior_laser_offon_jaws',
             'Jaws — optogenetic INHIBITION (n=5)'),
    'ChR':  ([m for m in LASER_MICE if GROUP[m] == 'ChR'],  'behavior_laser_offon_chr',
             'ChR — optogenetic EXCITATION (n=2)'),
}

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)


def make(mice, out_name, subtitle):
    n = len(mice)
    print(f'\n=== {out_name}  ({subtitle}) ===')
    records, stats = {}, {}
    for name, col, mask in METRICS:
        g = proportions(col, mask)
        rows = []
        for m in mice:
            off = g[(g.mouse == m) & (g.laser == 0)].perf
            on  = g[(g.mouse == m) & (g.laser == 1)].perf
            if len(off) and len(on):
                rows.append(dict(mouse=m, group=GROUP[m], off=off.mean(), on=on.mean(),
                                 delta=on.mean() - off.mean()))
        records[name] = pd.DataFrame(rows)
        b, p = laser_lmm(g, mice)
        stats[name] = (b, p)
        ptxt = f'p={p:.3f}{star(p) or " ns"}' if np.isfinite(p) else 'no test (n<3)'
        print(f'  {name:9s} ON-OFF β={b:+.3f}  {ptxt}')

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # A — grouped mean OFF vs ON per metric (mean ± SEM across mice)
    for i, name in enumerate(labels):
        r = records[name]
        for cond, cc, mk in [('off', '0.55', 'o'), ('on', '#d62728', 'D')]:
            mean = r[cond].mean()
            sem = r[cond].std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
            axA.errorbar(i + (-0.12 if cond == 'off' else 0.12), mean, yerr=sem, fmt=mk,
                         color=cc, ms=7, capsize=3, lw=1.5, zorder=3)
        axA.plot([i - 0.12, i + 0.12], [r['off'].mean(), r['on'].mean()], '-', color='0.7', lw=1, zorder=2)
    axA.axhline(0.5, ls=':', color='0.5', lw=1)
    axA.set_xticks(xs); axA.set_xticklabels(labels, rotation=30, ha='right')
    axA.set_ylabel('performance'); axA.set_ylim(0.45, 1.0)
    axA.set_title(f'A  OFF vs ON — absolute performance (Expert, n={n})', loc='left',
                  fontweight='bold', fontsize=11)
    axA.spines[['top', 'right']].set_visible(False)
    axA.legend(handles=[mlines.Line2D([0], [0], marker='o', color='0.55', ls='none', ms=7, label='laser OFF'),
                        mlines.Line2D([0], [0], marker='D', color='#d62728', ls='none', ms=7, label='laser ON')],
               frameon=False, fontsize=9, loc='lower right')

    # B — within-mouse ON−OFF Δ per metric
    rng = np.random.default_rng(0)
    for i, name in enumerate(labels):
        r = records[name]
        for _, row in r.iterrows():
            jit = (rng.random() - 0.5) * 0.28
            axB.scatter(i + jit, row['delta'], marker=GMARK[row['group']], s=42,
                        facecolor=GCOL[row['group']], edgecolor='w', linewidths=0.5, zorder=3)
        mean = r['delta'].mean()
        ci = 1.96 * r['delta'].std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
        axB.errorbar(i, mean, yerr=ci, fmt='_', color='k', ms=18, capsize=4, lw=2, zorder=4)
        p = stats[name][1]
        if np.isfinite(p) and star(p):
            axB.text(i, r['delta'].max() + 0.02, star(p), ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
    axB.axhline(0, ls='--', color='0.4', lw=1)
    axB.set_xticks(xs); axB.set_xticklabels(labels, rotation=30, ha='right')
    axB.set_ylabel('Δ performance  (ON − OFF)')
    axB.set_title('B  Within-mouse laser effect (ON − OFF)', loc='left', fontweight='bold', fontsize=11)
    axB.spines[['top', 'right']].set_visible(False)
    grp = GROUP[mice[0]] if len({GROUP[m] for m in mice}) == 1 else None
    if grp:
        gl = 'Jaws (inhibit)' if grp == 'Jaws' else 'ChR (excite)'
        hb = [mlines.Line2D([0], [0], marker=GMARK[grp], color=GCOL[grp], ls='none', ms=7, label=f'{gl}, n={n}')]
    else:
        hb = [mlines.Line2D([0], [0], marker='o', color=GCOL['Jaws'], ls='none', ms=7, label='Jaws (inhibit, n=5)'),
              mlines.Line2D([0], [0], marker='^', color=GCOL['ChR'], ls='none', ms=7, label='ChR (excite, n=2)')]
    hb.append(mlines.Line2D([0], [0], marker='_', color='k', ls='none', ms=11, label='mean ± 95% CI'))
    axB.legend(handles=hb, frameon=False, fontsize=8.5, loc='lower right')

    test = ('within-mouse LMM laser effect (perf ~ laser + (1|mouse), Wald)'
            if n >= 3 else 'n=2 — no formal test, per-mouse Δ shown')
    fig.suptitle(f'Laser ON vs OFF — behavioural comparison · {subtitle}', fontsize=13, y=0.99)
    fig.text(0.5, 0.005,
             f'Expert stage, within-mouse ON−OFF (interleaved laser trials).  Panel B mean ± 95% CI; stars = {test}.  '
             '* p<0.05  ** p<0.01  *** p<0.001',
             ha='center', va='bottom', fontsize=7.8, color='0.35')
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/{out_name}.{ext}'
        fig.savefig(p, bbox_inches='tight')
        print('saved', os.path.abspath(p))
    plt.close(fig)


for key in SEL:
    mice, out_name, subtitle = GROUP_SPEC[key]
    make(mice, out_name, subtitle)
