"""
fig_behavior_history.py — SUPPLEMENTARY to the behavioural main figure.

Sequential (trial-history) effects: does the CURRENT trial's performance depend on
the PREVIOUS trial's task type (DPA / Go / NoGo)? Two outcomes, one row each:
  row 1  DPA memory  = `performance` (hit/CR), defined on every trial.
  row 2  GNG distractor = `odr_perf` (Go/NoGo correct), defined on dual trials only.

Previous task is taken within each session (first trial of every session dropped, no
cross-session carry-over). One row per trial.

  A/E  current × previous task grid (mean performance, pooled trials).
  B/F  marginal by PREVIOUS task (collapsing current), per-mouse.
  C/G  Go-current zoom by previous task, per-mouse (paired) + GEE.
  D/H  GEE effect forest: prev-dual effect (odds ratio) for each current task.

Stats: trial-level logistic GEE clustered by mouse (Exchangeable), controlling for the
previous trial's own DPA outcome (prev_perf) and, for the recorded cohort, stage.
prev_dual = 1 if the previous task was Go or NoGo.

Data source: recorded cohort (pickle, 9 mice, laser OFF) — genuinely interleaved,
so previous-task is a clean trial-lag contrast (transition rate 0.70, P(prev|cur)≈⅓).
The training BATCHES are NOT analysed here: they use a blocked design (4 pure-DPA /
~8 dual, repeating) that confounds previous-task with pure↔dual block position — see
fig_behavior_history_batch.py, which analyses the batch as an (honest) switch-cost figure.

Output: figures/overlaps/behavior/{png,svg}/behavior_history.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_history.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.9, 'lines.linewidth': 1.8,
})

RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'   # DPA / Go / NoGo
TASK_COL = {'DPA': RED, 'Go': BLUE, 'NoGo': GREEN}
ORDER = ['DPA', 'Go', 'NoGo']
HAS_STAGE = True
SESS = ['mouse', 'stage', 'day']
TAG = ''
SRC = 'Recorded cohort, 9 mice, laser OFF'

# ── data source: recorded cohort (interleaved → clean trial-lag) ───────────────
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
LAB = '../data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl'
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'sample'].copy()
d = d[(d.laser == 0) & (d.mouse.isin(ALL_MICE))]

MICE = sorted(d.mouse.unique())
MANY = False
pal = sns.color_palette('tab10', n_colors=max(len(MICE), 3))
MOUSE_COLOR = {m: pal[i] for i, m in enumerate(MICE)}

# ── previous-trial task & outcome within each session ──────────────────────────
d = d.sort_index()
gb = d.groupby(SESS)
d['prev_task'] = gb['tasks'].shift(1)
d['prev_perf'] = gb['performance'].shift(1)
d = d.dropna(subset=['prev_task']).copy()
lab = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
d['cur'] = d['tasks'].map(lab)
d['prev'] = d['prev_task'].map(lab)
d['perf'] = d['performance'].astype(float)
d['op'] = d['odr_perf'].astype(float)
d['prev_dual'] = (d['prev'] != 'DPA').astype(int)


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def gee_prevdual(sub, outcol):
    sub = sub.dropna(subset=[outcol, 'prev_dual', 'prev_perf']).copy()
    sub['prev_perf'] = sub['prev_perf'].astype(float)
    f = f'{outcol} ~ prev_dual + prev_perf'
    if HAS_STAGE and sub['stage'].nunique() > 1:
        f += ' + C(stage)'
    m = smf.gee(f, 'mouse', data=sub, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se = m.params['prev_dual'], m.bse['prev_dual']
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues['prev_dual']


# ── generic panel builders ─────────────────────────────────────────────────────
def grid_panel(ax, outcol, curset, hi_row, title):
    piv = d.pivot_table(index='cur', columns='prev', values=outcol,
                        aggfunc='mean').reindex(curset)[ORDER] * 100
    gm = d[outcol].dropna().mean() * 100
    im = ax.imshow(piv.values, cmap='RdBu_r', vmin=gm - 6, vmax=gm + 6, aspect='auto')
    for i, rc in enumerate(curset):
        for j in range(3):
            ax.text(j, i, f'{piv.values[i, j]:.1f}', ha='center', va='center', fontsize=10,
                    fontweight='bold' if rc == hi_row else 'normal', color='k')
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDER)
    ax.set_yticks(range(len(curset))); ax.set_yticklabels(curset)
    for t, c in zip(ax.get_xticklabels(), ORDER):
        t.set_color(TASK_COL[c]); t.set_fontweight('bold')
    for t, c in zip(ax.get_yticklabels(), curset):
        t.set_color(TASK_COL[c]); t.set_fontweight('bold')
    ax.set_xlabel('previous task'); ax.set_ylabel('current task')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('perf (%)', fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=TITLE_FS)


def bytask_panel(ax, sub, outcol, title, ylabel, bracket=False):
    """Per-mouse mean performance by previous task (x = DPA/Go/NoGo)."""
    pm = sub.groupby(['mouse', 'prev'])[outcol].mean().unstack().reindex(columns=ORDER) * 100
    xs = np.arange(3)
    for m in MICE:
        if m in pm.index and pm.loc[m].notna().any():
            ax.plot(xs, pm.loc[m].values, '-', color=MOUSE_COLOR[m], lw=0.7,
                    alpha=0.2 if MANY else 0.4, zorder=2)
    mn = pm.mean().values
    se = pm.std(ddof=1).values / np.sqrt(pm.notna().sum().values.clip(1))
    for j, c in enumerate(ORDER):
        ax.errorbar(j, mn[j], yerr=se[j], fmt='o', color=TASK_COL[c], ms=11, capsize=4,
                    lw=2, mec='k', mew=0.8, zorder=5)
    ax.plot(xs, mn, '-', color='0.3', lw=1.6, zorder=4)
    orr, lo, hi, p = gee_prevdual(sub, outcol)
    txt = f'prev-dual OR={orr:.2f} p={p:.3f} {star(p)}'
    if bracket:
        pd_vs, pdual = pm['DPA'], pm[['Go', 'NoGo']].mean(axis=1)
        ok = pd_vs.notna() & pdual.notna()
        if ok.sum() > 2:
            _, pw = wilcoxon(pd_vs[ok], pdual[ok])
            txt += f'\npaired Wilcoxon p={pw:.2f} (n={int(ok.sum())})'
        ybr = np.nanmax(pm.values) + 2
        ax.plot([0, 0, 1.5, 1.5], [ybr - 1, ybr, ybr, ybr - 1], color='k', lw=1.2, zorder=6)
        ax.text(0.75, ybr + 0.4, star(p), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.035, txt, transform=ax.transAxes, ha='center', va='bottom', fontsize=8, color='0.3')
    ax.set_xticks(xs); ax.set_xticklabels(ORDER)
    for t, c in zip(ax.get_xticklabels(), ORDER):
        t.set_color(TASK_COL[c]); t.set_fontweight('bold')
    ax.set_xlim(-0.4, 2.4); ax.set_xlabel('previous task'); ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=TITLE_FS)


def forest_panel(ax, outcol, curset, title):
    rows = [(c, *gee_prevdual(d[d.cur == c], outcol)) for c in curset]
    ys = np.arange(len(rows))[::-1]
    for yy, (c, orr, lo, hi, p) in zip(ys, rows):
        sig = p < 0.05
        col = TASK_COL[c] if sig else '0.6'
        ax.errorbar(orr, yy, xerr=[[orr - lo], [hi - orr]], fmt='o', color=col,
                    mfc=col if sig else 'white', mec=col, ms=9, capsize=4, lw=1.6, zorder=3)
        ax.text(hi + 0.015, yy, star(p), va='center', ha='left', fontsize=11, fontweight='bold')
    ax.axvline(1.0, ls='--', color='0.4', lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([f'{c}-current' for c, *_ in rows])
    for t, (c, *_) in zip(ax.get_yticklabels(), rows):
        t.set_color(TASK_COL[c]); t.set_fontweight('bold')
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel('prev-dual effect\n(odds ratio, 95% CI)')
    ax.set_title(title, loc='left', fontweight='bold', fontsize=TITLE_FS)


# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.2, 10.6))
gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.05], height_ratios=[1, 1],
                      wspace=0.5, hspace=0.42, left=0.06, right=0.985, top=0.9, bottom=0.09)
TITLE_FS = 10.5


def panel_letter(ax, L, dx=0.030, dy=0.028):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, L, fontsize=15, fontweight='bold', va='top', ha='left')


# row 1 — DPA memory performance
axA = fig.add_subplot(gs[0, 0]); grid_panel(axA, 'perf', ORDER, 'Go', 'DPA memory by trial history')
axB = fig.add_subplot(gs[0, 1]); bytask_panel(axB, d, 'perf', 'No overall history effect', 'DPA performance (%)')
axC = fig.add_subplot(gs[0, 2]); bytask_panel(axC, d[d.cur == 'Go'], 'perf', 'Go trials carry a dual-task cost',
                                              'DPA performance (%) · Go-current', bracket=True)
axD = fig.add_subplot(gs[0, 3]); forest_panel(axD, 'perf', ORDER, 'DPA cost is specific to Go')

# row 2 — GNG distractor accuracy
axE = fig.add_subplot(gs[1, 0]); grid_panel(axE, 'op', ['Go', 'NoGo'], None, 'GNG accuracy by trial history')
axF = fig.add_subplot(gs[1, 1]); bytask_panel(axF, d[d.op.notna()], 'op', 'GNG has no history effect', 'GNG accuracy (%)')
axG = fig.add_subplot(gs[1, 2]); bytask_panel(axG, d[d.cur == 'Go'], 'op', 'Go-current GNG (null)',
                                              'GNG accuracy (%) · Go-current')
axH = fig.add_subplot(gs[1, 3]); forest_panel(axH, 'op', ['Go', 'NoGo'], 'GNG is history-independent')

for _ax, _L in [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D'),
                (axE, 'E'), (axF, 'F'), (axG, 'G'), (axH, 'H')]:
    panel_letter(_ax, _L)

fig.text(0.5, 0.5, 'DPA memory (performance)   ·   row below: GNG distractor (odr_perf)',
         ha='center', va='center', fontsize=9, color='0.5', style='italic')
fig.suptitle(f'Trial-history effects — {SRC}', fontsize=12.5, fontweight='bold', y=0.965)
fig.text(0.5, 0.018,
         f'{SRC} · previous task within session (first trial dropped) · '
         'GEE: correct ~ prev-dual + prev-outcome' + (' + stage' if HAS_STAGE else '') +
         ', clustered by mouse · prev-dual = previous trial was Go or NoGo · '
         'C/G show Go-current trials · * p<0.05 ** p<0.01 *** p<0.001',
         ha='center', va='bottom', fontsize=7.3, color='0.45')

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_history{TAG}.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
