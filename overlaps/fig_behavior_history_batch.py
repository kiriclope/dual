"""
fig_behavior_history_batch.py — SUPPLEMENTARY: trial-history in the training batches.

The training-batch sessions use a BLOCKED design — trials run in mini-blocks of
~4 pure-DPA followed by ~8 dual (Go/NoGo interleaved), repeating. This confounds
"previous task" with pure↔dual block position (a Go-current trial preceded by DPA is
*always* the first dual trial after a pure block), so the recorded-cohort trial-lag
analysis (fig_behavior_history.py) cannot be replicated cleanly here. Instead the batch
is analysed for what its design supports: a task-SWITCH-cost figure.

  A  Block structure — an example session's task sequence (4 pure-DPA / ~8 dual).
  B  Switch cost on DPA memory: first trial after a context switch vs mid-block, for
     →dual and →pure switches (per-mouse; GEE). Switching INTO dual costs memory;
     switching back to pure does not (asymmetric).
  C  Within-block warm-up: GNG accuracy vs position in the dual block (monotonic rise;
     the first dual trial — the switch — is worst).
  D  The one CLEAN trial-lag contrast the block design allows: prev-Go vs prev-NoGo on
     dual-current trials (matched block position), controlling for position + current
     task + previous outcome. ~Null on DPA memory; weak on GNG.

Data: CONTROL mice pooled across the 3 silencing batches (light-only, normal behaviour;
30 mice). --batch <name> --group <g> analyses a single batch/group instead.
Stats: trial-level logistic GEE clustered by mouse (Exchangeable).

Output: figures/overlaps/behavior/{png,svg}/behavior_history_batch[_<tag>].{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_history_batch.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, glob, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
import scipy.io as sio
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'    # DPA / Go / NoGo
SWITCH_C, REPEAT_C = '#332288', '#999999'             # switch / mid-block
TASK_COL = {'DPA': RED, 'Go': BLUE, 'NoGo': GREEN}
DATA_ROOT = '/storage/leon/dual_task/data/behavior'
SHORT = {'DualTask-Silencing-ACC': 'ACC', 'DualTask-Silencing-ACC-Prl': 'ACCPrl',
         'DualTask-Silencing-Prl-ACC': 'PrlACC'}
SILENCING = list(SHORT.keys())
MAXPOS = 8   # dual-block positions to show in the warm-up curve


def arg(flag, default=None):
    return next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == flag), default)


# ── loader (copied from fig_behavior_learning_batch.py; chronological order) ────
def load_session(path, mouse, day):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    Tr = np.atleast_2d(m['Trials'])
    out = Tr[:, 2].astype(int)
    n = len(out)
    perf = np.isin(out, [1, 4]).astype(float)
    has_gng = 'Trials1' in m and np.size(m['Trials1']) > 0
    if has_gng and 'SampleP' in m and np.size(m['SampleP']) > 0:
        S = np.atleast_2d(m['Sample'])[:, 0].astype('int64')
        SP = np.atleast_2d(m['SampleP'])[:, 0].astype('int64')
        isP = np.isin(S, SP)
    else:
        isP = np.ones(n, bool) if not has_gng else np.zeros(n, bool)
    tasks = np.where(isP, 'DPA', '').astype(object)
    odr = np.full(n, np.nan)
    if has_gng:
        Tr1 = np.atleast_2d(m['Trials1'])
        gout = Tr1[:, 2].astype(int)
        di = np.where(~isP)[0]
        k = min(len(di), len(gout))
        di, gout = di[:k], gout[:k]
        tasks[di] = np.where(np.isin(gout, [1, 2]), 'DualGo', 'DualNoGo')
        odr[di] = np.isin(gout, [1, 4]).astype(float)
    return pd.DataFrame({'mouse': mouse, 'day': day, 'tasks': tasks,
                         'perf': perf, 'op': odr})


def load_batch(batch, group, prefix=''):
    rows = []
    for fol in sorted(glob.glob(f'{DATA_ROOT}/{batch}/{group}_mouse_*'),
                      key=lambda p: int(p.rsplit('_', 1)[1])):
        mouse = prefix + os.path.basename(fol)
        for f in glob.glob(f'{fol}/session_*.mat'):
            day = int(os.path.basename(f).split('_')[1].split('.')[0]) + 1
            try:
                rows.append(load_session(f, mouse, day))
            except Exception as e:
                print(f'  !! {mouse} {os.path.basename(f)}: {e}')
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── data source ────────────────────────────────────────────────────────────────
BATCH, GROUP = arg('--batch'), arg('--group', 'control')
if BATCH is not None:
    d = load_batch(BATCH, GROUP)
    TAG = f'_{SHORT.get(BATCH, BATCH)}_{GROUP}'
    SRC = f'{SHORT.get(BATCH, BATCH)} batch, {GROUP} group ({d.mouse.nunique()} mice)'
else:
    d = pd.concat([load_batch(b, 'control', prefix=SHORT[b] + ':') for b in SILENCING],
                  ignore_index=True)
    TAG = ''
    SRC = f'Training batches, control mice pooled ({d.mouse.nunique()} mice, light-only)'
MICE = sorted(d.mouse.unique())

# ── per-trial context, block position, switch, clean prev-dual lag ────────────
d = d.sort_index()
d['ctx'] = np.where(d.tasks == 'DPA', 'pure', 'dual')


def pos_in_run(s):
    """1-indexed position within the current same-context run."""
    chg = np.concatenate([[True], s.values[1:] != s.values[:-1]])
    out = np.empty(len(s), int); c = 0
    for i, new in enumerate(chg):
        c = 1 if new else c + 1
        out[i] = c
    return pd.Series(out, index=s.index)


d['pos'] = d.groupby(['mouse', 'day'], group_keys=False)['ctx'].apply(pos_in_run)
d['switch'] = (d['pos'] == 1).astype(int)
gb = d.groupby(['mouse', 'day'])
d['prev_task'] = gb['tasks'].shift(1)
d['prev_perf'] = gb['perf'].shift(1)
lab = {'DualGo': 'Go', 'DualNoGo': 'NoGo', 'DPA': 'DPA'}
d['cur'] = d['tasks'].map(lab)
d['prevL'] = d['prev_task'].map(lab)


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def gee_or(sub, formula, term):
    sub = sub.dropna().copy()
    m = smf.gee(formula, 'mouse', data=sub, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se = m.params[term], m.bse[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[term]


# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.5, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 0.95], wspace=0.42,
                      left=0.045, right=0.99, top=0.82, bottom=0.2)


def panel_letter(ax, L, dx=0.028, dy=0.05):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, L, fontsize=11, fontweight='bold', va='top', ha='left')


# ── A: block-structure strip from a representative session ─────────────────────
axA = fig.add_subplot(gs[0, 0])
ex = load_session(sorted(glob.glob(
    f'{DATA_ROOT}/DualTask-Silencing-ACC/control_mouse_0/session_*.mat'))[5], 'ex', 1)
seq = ex['tasks'].map(lab).values[:24]
for i, tk in enumerate(seq):
    axA.add_patch(Rectangle((i, 0), 1, 1, facecolor=TASK_COL[tk], edgecolor='white', lw=0.4))
axA.set_xlim(0, len(seq)); axA.set_ylim(-1.7, 1.05)
# bracket the first pure and dual runs
runs = []; s = 0
for i in range(1, len(seq) + 1):
    if i == len(seq) or (seq[i] == 'DPA') != (seq[i - 1] == 'DPA'):
        runs.append((s, i, 'pure' if seq[s] == 'DPA' else 'dual')); s = i
for a, b, kind in runs[:3]:
    axA.plot([a + 0.1, b - 0.1], [-0.22, -0.22], color='k', lw=1.4)
    axA.text((a + b) / 2, -0.42, f'{kind} ({b - a})', ha='center', va='top', fontsize=7.5, color='k')
axA.text(len(seq) / 2, -0.95, 'trial within session →', ha='center', va='top', fontsize=8, color='0.4')
axA.axis('off')
axA.legend(handles=[mlines.Line2D([0], [0], marker='s', color=TASK_COL[t], ls='none', ms=8, label=t)
                    for t in ['DPA', 'Go', 'NoGo']], frameon=False, fontsize=8,
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), handletextpad=0.1, columnspacing=0.9)
axA.set_title('Blocked design (example session)', loc='left', fontsize=TITLE_FS)

# ── B: switch cost on DPA memory (→dual and →pure) ─────────────────────────────
axB = fig.add_subplot(gs[0, 1])
PAIRS = [('dual', 'into dual', 0.0, 1.0), ('pure', 'into pure', 2.4, 3.4)]
for ctx, name, x0, x1 in PAIRS:
    sub = d[d.ctx == ctx]
    sw, rp = [], []
    for m in MICE:
        a = sub[(sub.mouse == m) & (sub.switch == 1)]['perf']
        b = sub[(sub.mouse == m) & (sub.switch == 0)]['perf']
        sw.append(a.mean() * 100 if len(a) else np.nan)
        rp.append(b.mean() * 100 if len(b) else np.nan)
        if np.isfinite(sw[-1]) and np.isfinite(rp[-1]):
            axB.plot([x0, x1], [sw[-1], rp[-1]], '-', color='0.7', lw=0.5, alpha=0.3, zorder=2)
    for xx, vals, col in [(x0, sw, SWITCH_C), (x1, rp, REPEAT_C)]:
        vals = np.array(vals); mn = np.nanmean(vals)
        se = np.nanstd(vals, ddof=1) / np.sqrt(np.isfinite(vals).sum())
        axB.errorbar(xx, mn, yerr=se, fmt='o', color=col, ms=11, capsize=4, lw=2,
                     mec='k', mew=0.8, zorder=5)
    orr, lo, hi, p = gee_or(d[d.ctx == ctx][['perf', 'switch', 'mouse']], 'perf ~ switch', 'switch')
    ybr = 74
    axB.plot([x0, x0, x1, x1], [ybr - 0.6, ybr, ybr, ybr - 0.6], color='k', lw=1.2)
    axB.text((x0 + x1) / 2, ybr + 0.3, star(p), ha='center', va='bottom',
             fontsize=12 if p < 0.05 else 8, fontweight='bold', color='k' if p < 0.05 else '0.55')
    axB.text((x0 + x1) / 2, 62.5, f'OR={orr:.2f}\np={p:.3f}', ha='center', va='bottom',
             fontsize=6.5, color='0.3')
    axB.text((x0 + x1) / 2, 76.2, name, ha='center', va='bottom', fontsize=8)
axB.set_xticks([0, 1, 2.4, 3.4])
axB.set_xticklabels(['1st\nafter\nswitch', 'mid\nblock', '1st\nafter\nswitch', 'mid\nblock'], fontsize=8)
for t, c in zip(axB.get_xticklabels(), [SWITCH_C, REPEAT_C, SWITCH_C, REPEAT_C]):
    t.set_color(c); t.set_fontweight('bold')
axB.set_xlim(-0.5, 3.9); axB.set_ylim(62, 78)
axB.set_ylabel('DPA performance (%)')
axB.set_title('Switching into dual costs memory', loc='left', fontsize=TITLE_FS)

# ── C: warm-up within the dual block (GNG accuracy vs position) ────────────────
axC = fig.add_subplot(gs[0, 2])
du = d[d.ctx == 'dual']
pos = np.arange(1, MAXPOS + 1)
mn, se = [], []
for pp in pos:
    pmv = du[du.pos == pp].groupby('mouse')['op'].mean() * 100
    mn.append(pmv.mean()); se.append(pmv.std(ddof=1) / np.sqrt(pmv.notna().sum()))
mn, se = np.array(mn), np.array(se)
axC.errorbar(pos, mn, yerr=se, fmt='-o', color=BLUE, ms=6, capsize=3, lw=1.8, zorder=3)
axC.scatter(pos[0], mn[0], s=140, facecolor='none', edgecolor=SWITCH_C, linewidths=2, zorder=4)
axC.annotate('switch\n(1st dual)', (pos[0], mn[0]), textcoords='offset points',
             xytext=(14, -2), fontsize=6.5, color=SWITCH_C, va='center')
orr, lo, hi, p = gee_or(du[['op', 'switch', 'mouse']], 'op ~ switch', 'switch')
axC.text(0.96, 0.06, f'switch OR={orr:.2f}\np={p:.3f} {star(p)}', transform=axC.transAxes,
         ha='right', va='bottom', fontsize=6.5, color='0.3')
axC.set_xticks(pos); axC.set_xlabel('position within dual block')
axC.set_ylabel('GNG accuracy (%)')
axC.set_title('Distractor accuracy warms up', loc='left', fontsize=TITLE_FS)

# ── D: clean trial-lag — prev-Go vs prev-NoGo on dual trials (matched position) ─
axD = fig.add_subplot(gs[0, 3])
lagsub = d[(d.ctx == 'dual') & (d.prevL.isin(['Go', 'NoGo']))].copy()
lagsub['prev_nogo'] = (lagsub['prevL'] == 'NoGo').astype(int)
rows = []
for oc, name, col in [('perf', 'DPA memory', RED), ('op', 'GNG accuracy', BLUE)]:
    sub = lagsub.dropna(subset=[oc, 'prev_perf'])[[oc, 'prev_nogo', 'pos', 'cur', 'prev_perf', 'mouse']]
    orr, lo, hi, p = gee_or(sub, f'{oc} ~ prev_nogo + pos + C(cur) + prev_perf', 'prev_nogo')
    rows.append((name, col, orr, lo, hi, p))
ys = np.arange(len(rows))[::-1]
for yy, (name, col, orr, lo, hi, p) in zip(ys, rows):
    sig = p < 0.05
    axD.errorbar(orr, yy, xerr=[[orr - lo], [hi - orr]], fmt='o', color=col,
                 mfc=col if sig else 'white', mec=col, ms=9, capsize=4, lw=1.6, zorder=3)
    axD.text(hi + 0.01, yy, star(p), va='center', ha='left',
             fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
axD.axvline(1.0, ls='--', color='0.4', lw=1)
axD.set_yticks(ys); axD.set_yticklabels([n for n, *_ in rows])
for t, (_, col, *_) in zip(axD.get_yticklabels(), rows):
    t.set_color(col); t.set_fontweight('bold')
axD.set_ylim(-0.6, len(rows) - 0.4)
axD.set_xlabel('prev NoGo vs Go\n(odds ratio, 95% CI)')
axD.set_title('Clean lag: null memory, weak GNG', loc='left', fontsize=TITLE_FS)

for _ax, _L in [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')]:
    panel_letter(_ax, _L)

fig.suptitle(f'Trial-history in the blocked training batches — {SRC}',
             fontsize=11, y=0.99)
fig.text(0.5, 0.015,
         f'{SRC} · blocked design (~4 pure-DPA / ~8 dual, repeating) confounds previous-task with '
         'block position → only the switch-cost & matched within-dual (prev Go/NoGo) contrasts are '
         'clean · GEE clustered by mouse · D controls for position + current task + previous outcome · '
         '* p<0.05 ** p<0.01 *** p<0.001', ha='center', va='bottom', fontsize=6.5, color='0.45')

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_history_batch{TAG}.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
