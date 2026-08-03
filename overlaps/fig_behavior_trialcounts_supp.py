"""
fig_behavior_trialcounts_supp.py — per-mouse behavioural trial counts / inclusion.

A Nature-Neuroscience reporting supplement (figure G3): how many behavioural
trials from each recorded mouse enter the overlaps analysis, broken down by
task (DPA / Go / NoGo) and by stage (Naive / Expert), laser-OFF trials only.

Data source: the same per-trial labels DataFrame the behaviour figures load
(`../data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl`).
Each trial appears in three target rows (sample / choice / test); we keep one
row per trial (`target=='choice'`), restrict to `laser==0`, and count trials
per (mouse × task × stage).

NOTE: within each mouse×stage the counts are matched across the three tasks by
the pseudo-population balancing (e.g. 128/128/128) — the three task bars are
therefore equal length by design; the figure documents the *inclusion*, not a
behavioural task imbalance.

Two panels = Naive | Expert.  One horizontal bar group per mouse, three bars
per group (DPA / Go / NoGo).  Count printed at each bar tip.

Output: figures/overlaps/behavior/{png,svg}/behavior_trialcounts.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_trialcounts_supp.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style (Nature Neuroscience house style — matches the main figures) ──────────
sns.set_context('notebook')          # MUST come after importing src.common.plot_utils (sets "poster")
sns.set_style('ticks')
plt.rcParams.update({                 # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']

# task display order + consistent task colours (DPA red / Go blue / NoGo green,
# matching fig_behavior_learning.py)
TASK_KEYS  = ['DPA', 'DualGo', 'DualNoGo']
TASK_LABEL = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
TASK_COLOR = {'DPA': '#d62728', 'DualGo': '#1f77b4', 'DualNoGo': '#2ca02c'}

LAB = '../data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl'

# ── Load labels, keep one row per trial, laser-OFF ─────────────────────────────
if not os.path.exists(LAB):
    sys.exit(f'ERROR: labels pickle not found: {os.path.abspath(LAB)}')
y = pickle.load(open(LAB, 'rb'))
need = {'target', 'tasks', 'laser', 'mouse', 'stage'}
missing = need - set(y.columns)
if missing:
    sys.exit(f'ERROR: labels DataFrame missing column(s): {sorted(missing)}; '
             f'has {sorted(y.columns)}')

d = y[y.target == 'choice'].copy()
d = d[d.laser == 0]

# ── Count trials per (mouse × stage × task) ────────────────────────────────────
counts = (d.groupby(['mouse', 'stage', 'tasks'], observed=True)
            .size().rename('n').reset_index())
# pivot to a lookup: cnt[(mouse, stage, task)] -> n
cnt = {(r.mouse, r.stage, r.tasks): int(r.n) for r in counts.itertuples()}

# ── Print per-mouse tables to stdout ───────────────────────────────────────────
print('=' * 70)
print('Behavioural trial counts (laser-OFF, one row per trial) — recorded cohort')
print('=' * 70)
for stage in STAGES:
    print(f'\n[{stage}]  trials per mouse × task')
    tab = (d[d.stage == stage]
           .groupby(['mouse', 'tasks'], observed=True).size()
           .unstack('tasks').reindex(index=ALL_MICE, columns=TASK_KEYS)
           .fillna(0).astype(int))
    tab.columns = [TASK_LABEL[c] for c in tab.columns]
    tab['total'] = tab.sum(axis=1)
    print(tab.to_string())

print('\n[per-mouse totals across both stages]')
tot = (d.groupby('mouse').size().reindex(ALL_MICE).fillna(0).astype(int)
       .rename('total_trials'))
tot_by_stage = (d.groupby(['mouse', 'stage'], observed=True).size()
                  .unstack('stage').reindex(index=ALL_MICE, columns=STAGES)
                  .fillna(0).astype(int))
summary = pd.concat([tot_by_stage, tot], axis=1)
print(summary.to_string())
print(f'\nGRAND TOTAL trials (laser-OFF): {int(tot.sum())} '
      f'across {d.mouse.nunique()} mice, '
      f"{d.stage.value_counts().to_dict()}")

# ── Figure: two panels (Naive | Expert), grouped horizontal bars per mouse ─────
n_mice = len(ALL_MICE)
y_pos = np.arange(n_mice)[::-1]              # JawsM01 at top
grp_h = 0.78                                  # total height of a mouse's bar group
bar_h = grp_h / len(TASK_KEYS)
offs = (np.arange(len(TASK_KEYS)) - (len(TASK_KEYS) - 1) / 2) * bar_h

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), sharey=True)

xmax = max([cnt.get((m, s, t), 0)
            for m in ALL_MICE for s in STAGES for t in TASK_KEYS] + [1])

for ax, stage in zip(axes, STAGES):
    for mi, m in enumerate(ALL_MICE):
        for ti, t in enumerate(TASK_KEYS):
            n = cnt.get((m, stage, t), 0)
            yb = y_pos[mi] + offs[ti]
            ax.barh(yb, n, height=bar_h * 0.92, color=TASK_COLOR[t],
                    edgecolor='none', zorder=3)
            if n > 0:
                ax.text(n + xmax * 0.012, yb, str(n), va='center', ha='left',
                        fontsize=6.5, color='0.25')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ALL_MICE, fontsize=7)
    ax.set_xlim(0, xmax * 1.14)
    ax.set_xlabel('trials (laser-OFF)')
    ax.set_title(stage, loc='left', fontsize=TITLE_FS)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.margins(y=0.01)

# task legend (colours) on the right panel
handles = [mpatches.Patch(color=TASK_COLOR[t], label=TASK_LABEL[t]) for t in TASK_KEYS]
axes[1].legend(handles=handles, title='task', frameon=False, fontsize=6.5,
               title_fontsize=6.5, loc='lower right')

fig.suptitle('Behavioural trial counts per mouse, task and stage '
             '(recorded cohort, laser-OFF; n=9 mice)', fontsize=9, y=0.99)
fig.text(0.5, 0.005,
         'One row per trial (target=choice), laser==0.  Within each mouse×stage '
         'the three task counts are matched by the pseudo-population balancing — '
         'equal bar length is by design, not a behavioural imbalance.',
         ha='center', va='bottom', fontsize=6.5, color='0.3')
fig.tight_layout(rect=(0, 0.045, 1, 0.94))

# Panel letters — the only bold text (after layout so positions are final)
for ax, L in zip(axes, 'AB'):
    p = ax.get_position()
    fig.text(p.x0 - 0.03, p.y1 + 0.02, L, fontsize=11, fontweight='bold',
             va='top', ha='left')

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_trialcounts.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
