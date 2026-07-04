"""
fig_behavior_learning.py — behavioural learning curves for the opto cohort.

Characterises the task behaviour of the 9 opto mice (laser-OFF trials only; the
overlaps labels file contains no laser-on trials) as accuracy vs training day.

Four panels (2×2):
  A  DPA & GNG performance vs day               (DPA red / GNG blue)
  B  GNG performance, Go vs NoGo distractor      (Go blue / NoGo green)
  C  DPA performance, paired vs unpaired          (paired red solid / unpaired red dashed)
  D  DPA UNPAIRED performance by task context     (DPA-only red / Go blue / NoGo green, all dashed)

Definitions (mirroring plot_scatter_perf.py):
  - DPA performance = `performance` on tasks=='DPA' trials
  - GNG performance = `odr_perf`   on Dual (DualGo/DualNoGo) trials
  - pair==1 → paired (match, lick expected);  pair==0 → unpaired (nonmatch, no-lick)
  - one row per trial: filter target=='choice' (each trial appears 3× — one per decoder row)

Each point = across-mouse mean ± SEM of the per-mouse daily accuracy.
Day counts vary per mouse (4–6) so n drops on later days (annotated on panel A).

Statistics — LINEAR MIXED MODEL (LMM) on per-mouse/day accuracy proportions,
      random intercept for mouse:  perf ~ C(condition)*day_centred + (1|mouse).
      Chosen as the less-conservative alternative to a cluster-robust GEE: the
      random intercept models within-mouse pairing (like a repeated-measures
      test), recovering power the population-average GEE loses with only 9
      clusters. One condition effect + condition×day interaction per panel (Wald;
      panel D 3-level → omnibus df=2), boxed in each panel + coefficients in E.
  CAVEAT: the mouse random-effect variance sits near the boundary (≈0.004–0.01,
      convergence warnings), so the LMM behaves close to OLS on the proportions
      and is somewhat ANTI-CONSERVATIVE (under-penalises the 9-mouse limit). The
      honest range is bracketed by GEE (conservative) and this LMM (liberal).
  Top stars = per-day LMM condition effect (random intercept mouse), UNCORRECTED
      (exploratory); days with <N_MIN (=4) mice untested (day 6 has n=4).
  * p<0.05  ** p<0.01  *** p<0.001.

Output: figures/overlaps/behavior/{png,svg}/behavior_learning.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_learning.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.formula.api as smf

matplotlib.rcParams.update({
    'figure.dpi':   150,
    'savefig.dpi':  300,
    'font.family':  'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
LASER_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23']            # 5 Jaws inhibit + 2 ChR excite (ACC has no laser)

RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'
N_MIN = 4                         # min mice/day for a per-day test (day 6 has n=4)

LASER_FILE = ('../data/overlaps/'
              'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')

MAIN_FILE = '../data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl'

# ── Mode selection ────────────────────────────────────────────────────────────
#   default            : laser-OFF, all 9 mice (the main figure)
#   --on               : laser-ON, 7 laser mice pooled
#   --jaws / --chr     : one opto group (from the laser file); + --on for laser-ON
#   --acc              : ACC control mice (no laser → laser-OFF only)
GRP = ('Jaws' if '--jaws' in sys.argv[1:] else 'ChR' if '--chr' in sys.argv[1:]
       else 'ACC' if '--acc' in sys.argv[1:] else None)
ON  = '--on' in sys.argv[1:]
if GRP == 'ACC':                              # ACC has no laser trials → OFF only
    LAB = MAIN_FILE
    MICE = [m for m in ALL_MICE if m.startswith('ACC')]
    LASER_VAL, OUT_NAME = 0, 'behavior_learning_acc'
    TITLE = f'Behavioural learning curves (ACC control, no laser, n={len(MICE)})'
elif GRP:
    LAB = LASER_FILE
    MICE = [m for m in LASER_MICE if m.startswith(GRP)]
    LASER_VAL = 1 if ON else 0
    lname = 'ON' if ON else 'OFF'
    OUT_NAME = f'behavior_learning_{GRP.lower()}{"_on" if ON else ""}'
    verb = 'inhibit' if GRP == 'Jaws' else 'excite'
    TITLE = f'Behavioural learning curves ({GRP} {verb}, laser {lname}, n={len(MICE)})'
elif ON:
    LAB = LASER_FILE
    MICE, LASER_VAL, OUT_NAME = LASER_MICE, 1, 'behavior_learning_laseron'
    TITLE = ('Behavioural learning curves (laser ON — causal; '
             '5 Jaws inhibit + 2 ChR excite pooled)')
else:
    LAB = '../data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl'
    MICE, LASER_VAL, OUT_NAME = ALL_MICE, 0, 'behavior_learning'
    TITLE = 'Behavioural learning curves (opto cohort, laser-off)'

# ── Load labels, keep one row per trial ───────────────────────────────────────
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'choice'].copy() if 'target' in y.columns else y.copy()
d = d[(d.laser == LASER_VAL) & (d.mouse.isin(MICE))]
DAYS = list(range(1, int(d.day.max()) + 1))
print(f'{OUT_NAME}: {len(d)} trials, {d.mouse.nunique()} mice, days {DAYS[0]}–{DAYS[-1]}')


def per_mouse_day(col, mask):
    """{day: {mouse: mean accuracy that day}} over the `mask` trial subset."""
    sel = d[mask]
    out = {day: {} for day in DAYS}
    for day in DAYS:
        for m in MICE:
            v = sel[(sel.mouse == m) & (sel.day == day)][col].dropna()
            if len(v):
                out[day][m] = float(v.mean())
    return out


def agg(pmd):
    """Across-mouse mean ± SEM and n per day from a per_mouse_day dict."""
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
    x, m, s, n = agg(pmd)
    ok = ~np.isnan(m)
    ax.plot(x[ok], m[ok], ls=ls, color=color, lw=2, marker='o', ms=5,
            mfc=color, mec=color, label=label, zorder=3)
    ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18,
                    lw=0, zorder=1)
    return n


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def _prop(mask, cond, correct, by_day=True):
    """Per-mouse (and per-day) accuracy proportions for the mixed model."""
    df = pd.DataFrame({
        'correct': np.asarray(correct)[mask.values],
        'cond':    np.asarray(cond)[mask.values],
        'mouse':   d.loc[mask, 'mouse'].values,
        'day':     d.loc[mask, 'day'].values,
    }).dropna()
    keys = ['mouse', 'day', 'cond'] if by_day else ['mouse', 'cond']
    return df.groupby(keys, observed=True).correct.mean().reset_index(name='perf')


def lmm(ax, mask, cond, correct, ref, tag):
    """Trajectory LMM: perf ~ condition*day_centred + (1|mouse), on per-mouse/day
       proportions.  Boxes the condition + condition×day Wald tests (bottom-left)
       and returns the fixed-effect β records (Δ performance) for panel E."""
    g = _prop(mask, cond, correct, by_day=True)
    g['dayc'] = g['day'] - g['day'].mean()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.mixedlm(f"perf ~ C(cond, Treatment('{ref}'))*dayc",
                          g, groups=g['mouse']).fit()
        wt = res.wald_test_terms(scalar=True).table
    ct = [i for i in wt.index if i.startswith('C(cond') and ':' not in i][0]
    it = [i for i in wt.index if i.startswith('C(cond') and ':' in i][0]
    cp, cdf = float(wt.loc[ct, 'pvalue']), int(wt.loc[ct, 'df_constraint'])
    ip = float(wt.loc[it, 'pvalue'])
    dfs = f' (df{cdf})' if cdf > 1 else ''
    print(f'  [{tag}] LMM  condition p={cp:.4f}{dfs} {star(cp) or "ns"} | '
          f'cond×day p={ip:.4f} {star(ip) or "ns"}  (RE var={res.cov_re.iloc[0, 0]:.4f})')
    # trajectory p-values are shown as coefficients in panel E (no in-panel box).
    # per-coefficient betas + 95% CI (Δ performance) for the forest panel
    ci, pv, pr = res.conf_int(), res.pvalues, res.params
    recs = []
    for name in pr.index:
        if not name.startswith('C(cond'):
            continue
        lvl = name.split('[T.')[1].split(']')[0]
        recs.append(dict(model=tag.split()[0], contrast=f'{lvl}−{ref}',
                         kind='cond×day' if ':dayc' in name else 'condition',
                         beta=float(pr[name]), lo=float(ci.loc[name, 0]),
                         hi=float(ci.loc[name, 1]), p=float(pv[name])))
    return recs


def perday_stars(ax, mask, cond, correct, ref, tag, y_star=1.03):
    """Day-by-day stars via per-day LMM condition effect: perf ~ condition + (1|mouse)
       on per-mouse proportions (within-mouse paired). UNCORRECTED (exploratory).
       Days with <N_MIN mice are not tested."""
    ps = {}
    for day in DAYS:
        sub = mask & (d.day == day)
        if d.loc[sub, 'mouse'].nunique() < N_MIN:
            continue
        g = _prop(sub, cond, correct, by_day=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = smf.mixedlm(f"perf ~ C(cond, Treatment('{ref}'))",
                                g, groups=g['mouse']).fit()
                wt = r.wald_test_terms(scalar=True).table
            ct = [i for i in wt.index if i.startswith('C(cond')][0]
            ps[day] = float(wt.loc[ct, 'pvalue'])
        except Exception:
            pass
    if not ps:
        return
    days = sorted(ps)
    print(f'  [{tag}] per-day LMM (uncorrected): '
          + '  '.join(f'd{k} p={ps[k]:.3f}{star(ps[k]) or "ns"}' for k in days))
    for k in days:
        if star(ps[k]):
            ax.text(k, y_star, star(ps[k]), ha='center', va='top',
                    fontsize=11, fontweight='bold', color='k')


IS_DPA   = d.tasks == 'DPA'
IS_GO    = d.tasks == 'DualGo'
IS_NOGO  = d.tasks == 'DualNoGo'
IS_DUAL  = IS_GO | IS_NOGO
UNP      = d.pair == 0

# trial-level correctness + condition label series (index-aligned to d) for the GEE
CORR_DPA = d.performance                                  # DPA correctness
CORR_GNG = d.odr_perf                                     # GNG correctness (NaN on DPA)
CORR_A   = np.where(IS_DPA, d.performance, d.odr_perf)    # panel A: task-appropriate
COND_A   = np.where(IS_DPA, 'DPA', 'GNG')
COND_PAIR = np.where(d.pair == 1, 'paired', 'unpaired')
COND_TASK = d.tasks.map({'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'})

# per-mouse/day tables for every curve we draw
pmd_dpa      = per_mouse_day('performance', IS_DPA)
pmd_gng      = per_mouse_day('odr_perf',    IS_DUAL)
pmd_go       = per_mouse_day('odr_perf',    IS_GO)
pmd_nogo     = per_mouse_day('odr_perf',    IS_NOGO)
pmd_pair     = per_mouse_day('performance', IS_DPA & (d.pair == 1))
pmd_unpair   = per_mouse_day('performance', IS_DPA & UNP)
pmd_u_dpa    = per_mouse_day('performance', UNP & IS_DPA)
pmd_u_go     = per_mouse_day('performance', UNP & IS_GO)
pmd_u_nogo   = per_mouse_day('performance', UNP & IS_NOGO)

fig, (axA, axB, axC, axD, axE) = plt.subplots(1, 5, figsize=(22, 4.3))
betas = []

# A — DPA & GNG vs day
plot_line(axA, pmd_dpa, RED,  'DPA')
plot_line(axA, pmd_gng, BLUE, 'GNG')
betas += lmm(axA, IS_DPA | IS_DUAL, COND_A, CORR_A, 'DPA', 'A DPA vs GNG')
perday_stars(axA, IS_DPA | IS_DUAL, COND_A, CORR_A, 'DPA', 'A DPA vs GNG')
axA.set_title('A  DPA vs GNG performance', loc='left', fontweight='bold', fontsize=11)

# B — GNG Go vs NoGo distractor
plot_line(axB, pmd_go,   BLUE,  'Go')
plot_line(axB, pmd_nogo, GREEN, 'NoGo')
betas += lmm(axB, IS_DUAL, COND_TASK, CORR_GNG, 'Go', 'B Go vs NoGo')
perday_stars(axB, IS_DUAL, COND_TASK, CORR_GNG, 'Go', 'B Go vs NoGo')
axB.set_title('B  GNG: Go vs NoGo', loc='left', fontweight='bold', fontsize=11)

# C — DPA paired vs unpaired
plot_line(axC, pmd_pair,   RED, 'paired',   ls='-')
plot_line(axC, pmd_unpair, RED, 'unpaired', ls='--')
betas += lmm(axC, IS_DPA, COND_PAIR, CORR_DPA, 'paired', 'C paired vs unpaired')
perday_stars(axC, IS_DPA, COND_PAIR, CORR_DPA, 'paired', 'C paired vs unpaired')
axC.set_title('C  DPA: paired vs unpaired', loc='left', fontweight='bold', fontsize=11)

# D — DPA UNPAIRED performance by task context (all dashed)
plot_line(axD, pmd_u_dpa,  RED,   'DPA only', ls='--')
plot_line(axD, pmd_u_go,   BLUE,  'Go',       ls='--')
plot_line(axD, pmd_u_nogo, GREEN, 'NoGo',     ls='--')
betas += lmm(axD, UNP, COND_TASK, CORR_DPA, 'DPA', 'D unpaired by task')
perday_stars(axD, UNP, COND_TASK, CORR_DPA, 'DPA', 'D unpaired by task')
axD.set_title('D  DPA unpaired, by task', loc='left', fontweight='bold', fontsize=11)

for ax in (axA, axB, axC, axD):
    ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_ylim(0.18, 1.07)
    ax.set_xticks(DAYS)
    ax.set_xlabel('training day')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
axA.set_ylabel('performance')

# ── Panel E — GLMM coefficients (condition + condition×day, 95% CI), β on y ────
cond_recs = [r for r in betas if r['kind'] == 'condition']
int_map = {(r['model'], r['contrast']): r for r in betas if r['kind'] == 'cond×day'}
xlabels = []
for i, c in enumerate(cond_recs):
    it = int_map[(c['model'], c['contrast'])]
    cc = 'k'   if c['p'] < 0.05 else '0.65'
    ci = BLUE  if it['p'] < 0.05 else '0.65'
    axE.errorbar(i - 0.16, c['beta'], yerr=[[c['beta'] - c['lo']], [c['hi'] - c['beta']]],
                 fmt='o', color=cc, ms=7, capsize=3, lw=1.6, zorder=3)
    axE.errorbar(i + 0.16, it['beta'], yerr=[[it['beta'] - it['lo']], [it['hi'] - it['beta']]],
                 fmt='s', mfc='white', mec=ci, color=ci, ms=6, capsize=3, lw=1.4, zorder=3)
    for r, dx in ((c, -0.16), (it, 0.16)):
        if star(r['p']):
            axE.text(i + dx, r['hi'] + 0.08, star(r['p']), ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    xlabels.append((i, f"{c['model']}  {c['contrast']}"))

axE.axhline(0, ls='--', color='0.4', lw=1)
axE.set_xticks([x for x, _ in xlabels])
axE.set_xticklabels([lab for _, lab in xlabels], rotation=40, ha='right', fontsize=8.5)
axE.set_xlim(-0.6, len(cond_recs) - 0.4)
axE.set_ylabel('LMM coefficient  β  (Δ performance)')
axE.set_title('E  LMM fixed-effect coefficients (95% CI)', loc='left', fontweight='bold', fontsize=11)
axE.spines[['top', 'right']].set_visible(False)
leg_h = [mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=7, label='condition (β at mean day)'),
         mlines.Line2D([0], [0], marker='s', color=BLUE, mfc='white', ls='none', ms=6, label='condition × day (slope)')]
axE.legend(handles=leg_h, frameon=False, fontsize=8, loc='upper right')

fig.suptitle(TITLE, fontsize=13, y=0.99)
fig.text(0.5, 0.005,
         f'Curves: mean ± SEM across mice ({d.mouse.nunique()} mice).  '
         'Top stars: per-day LMM condition effect, random intercept mouse, UNCORRECTED (exploratory; day 6 n=4).  '
         'Panel E: trajectory LMM, perf ~ condition×day + (1|mouse) — less conservative than GEE, '
         'RE variance near boundary (mildly anti-conservative).  * p<0.05  ** p<0.01  *** p<0.001',
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
