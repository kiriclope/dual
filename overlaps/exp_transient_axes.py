"""exp_transient_axes.py — where does the intruding action's activity GO? (idea #5,
2026-09-01 analysis menu). On correct DualGo trials the animal licks the mid-delay cue —
the intruding action executes. Decompose the cue-evoked transient by code axis: per mouse,
Delta = mean(projection over CUE+RWD bins 39-44) - mean(over MD bins 33-38), computed within
class then class-averaged (common-mode shift, in each code's own per-mouse evoked-SD unit —
main_panels' uniform normalisation, so the four axes are comparable).

Prediction (the geometric twin of Fig 1g's lick chain): the transient rides the ACTION axes
(choice/lick, distractor) and barely touches the memory (sample) or test axes — the intruding
action travels along dimensions orthogonal to the memory, which is why it can execute without
overwriting the memoranda. Control: correct DualNoGo trials (withheld — no lick) should show a
much smaller choice-axis transient.

Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_transient_axes.py
Output: figures/overlaps/transient/png/exp_transient_axes.png (+svg) + printed stats + cache.
NB imports main_panels (loads the 1.9 GB bundled tensor, ~2-4 min).
"""
import matplotlib; matplotlib.use('Agg')
import os, sys, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import main_panels as mp
import seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

MD = np.arange(33, 39)          # pre-cue mid-delay (33-38)
CUE = np.arange(39, 45)         # GNG cue + reward window (39-44)
STAGES = ['Naive', 'Expert']

# (name, projection matrix, label frame, class column) — class-averaged common-mode shift
CODES = [('sample', mp.SAMPLE_D, mp.Y_SAM, 'sample_odor'),
         ('dist',   mp.GNG_D,    mp.Y_GNG, 'gng'),
         ('choice', mp.LICK_D,   mp.Lm,    'choice'),
         ('test',   mp.TEST_D,   mp.Y_TST, 'test_odor')]

def _transient(D, yy, ccol, mouse, stage, task, need_lick):
    base = ((yy.mouse == mouse).to_numpy() & (yy.laser == 0).to_numpy()
            & (yy.stage == stage).to_numpy() & (yy.tasks == task).to_numpy())
    if need_lick is True:                      # correct Go = licked the cue
        base &= (yy.odr_perf == 1).to_numpy()
    elif need_lick is False:                   # correct NoGo = withheld
        base &= (yy.odr_perf == 1).to_numpy()
    ds = []
    for cv in pd.unique(yy.loc[base, ccol].dropna()):
        rows = base & (yy[ccol] == cv).to_numpy()
        if rows.sum() < 3:
            continue
        M = D[rows]
        ds.append(np.nanmean(M[:, CUE]) - np.nanmean(M[:, MD]))
    return np.mean(ds) if ds else np.nan

rows = []
for task, lick, tag in [('DualGo', True, 'Go'), ('DualNoGo', False, 'NoGo')]:
    for st in STAGES:
        for m in mp.ALL_MICE:
            r = dict(task=tag, stage=st, mouse=m)
            for name, D, yy, ccol in CODES:
                r[name] = _transient(D, yy, ccol, m, st, task, lick)
            rows.append(r)
DF = pd.DataFrame(rows)
pd.set_option('display.width', 150)
for tag in ['Go', 'NoGo']:
    for st in STAGES:
        sub = DF[(DF.task == tag) & (DF.stage == st)]
        med = sub[['sample', 'dist', 'choice', 'test']].abs().median()
        print(f'{tag:4s} {st:6s} |transient| medians: ' +
              '  '.join(f'{k}={med[k]:.2f}' for k in ['sample', 'dist', 'choice', 'test']))
        for ax_ in ['dist', 'choice']:
            a = sub[ax_].abs().values; b = sub['sample'].abs().values
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 6:
                w = wilcoxon(a[ok], b[ok])
                print(f'    |{ax_}| vs |sample|: p={w.pvalue:.4f} (n={ok.sum()})')

# stage change of the Go-trial transient, per axis (per-mouse |Delta|, Expert vs Naive)
for ax_ in ['sample', 'dist', 'choice']:
    nv = DF[(DF.task == 'Go') & (DF.stage == 'Naive')].set_index('mouse')[ax_].abs()
    ev = DF[(DF.task == 'Go') & (DF.stage == 'Expert')].set_index('mouse')[ax_].abs()
    both = pd.concat([nv, ev], axis=1, keys=['n', 'e']).dropna()
    if len(both) >= 6:
        w = wilcoxon(both.e, both.n)
        print(f'Go stage change |{ax_}|: {both.n.median():.2f} -> {both.e.median():.2f} '
              f'p={w.pvalue:.3f} ({int((both.e > both.n).sum())}/{len(both)} up)')

OUT = 'figures/overlaps/transient'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
pickle.dump(DF, open(f'{OUT}/transient_axes.pkl', 'wb'))

fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
XPOS = {'sample': 0, 'test': 1, 'dist': 2, 'choice': 3}
for ax, st in zip(axs, STAGES):
    sub = DF[(DF.task == 'Go') & (DF.stage == st)]
    for _, r in sub.iterrows():
        vals = [abs(r[k]) if np.isfinite(r[k]) else np.nan for k in XPOS]
        ax.plot(list(XPOS.values()), vals, '-', color=mp.MOUSE_COLOR[r.mouse],
                lw=0.8, alpha=0.45, zorder=2)
        ax.scatter(list(XPOS.values()), vals, s=34, color=mp.MOUSE_COLOR[r.mouse],
                   edgecolors='w', linewidths=0.6, zorder=3)
    med = sub[list(XPOS)].abs().median()
    ax.plot(list(XPOS.values()), [med[k] for k in XPOS], '-', color='0.2', lw=1.6, zorder=4)
    for ax_ in ['dist', 'choice']:
        a, b = sub[ax_].abs().values, sub['sample'].abs().values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 6:
            p = wilcoxon(a[ok], b[ok]).pvalue
            sig = p < .05
            ax.text(XPOS[ax_], 0.97, '*' if sig else 'n.s.', transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=12 if sig else 8, fontweight='bold',
                    color='k' if sig else '0.55')
    ax.set_xticks(list(XPOS.values()))
    ax.set_xticklabels(['sample\n(memory)', 'test', 'dist', 'choice\n(lick)'])
    ax.set_title(st, loc='left', fontsize=TITLE_FS)
axs[0].set_ylabel('|cue-evoked shift| (evoked-SD)')
fig.suptitle('The intruding action rides the action axes and spares the memory axis (correct Go trials)',
             x=0.02, ha='left', fontsize=TITLE_FS)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f'{OUT}/png/exp_transient_axes.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/exp_transient_axes.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/exp_transient_axes.png'))
