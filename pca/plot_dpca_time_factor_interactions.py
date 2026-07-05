"""time × factor interactions of the f-sample-test-tasks dPCA.

This dPCA (src/pca/dpca.py) marginalises over factors [sample, test, tasks]; the only
condition-INDEPENDENT marginal is `time` (the () key).  There is therefore no separate
`time:sample` component — each *factor* marginal is already a full time-course, i.e. it
lumps the time-CONSTANT part of that factor's code with its time-VARYING part
(Kobak's {s} + {s·t}).  The genuine "time:factor" interaction is the time-varying part:

    inter_g(t) = comp_g(t) - mean_t comp_g(t)          (per condition group g)

This script splits every non-time marginal into that static offset (dotted flat line)
and its time:factor interaction (solid), and reports the time-varying variance fraction
frac_t = var_t(inter) / var_t(comp) — how much of the factor's demixed code is dynamic.

Expert by default; --stage Naive; --both overlays Naive (dashed) on Expert (solid).
Saved to figures/pseudo/time_inter/.
"""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'
sns.set_context('paper'); sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='Expert', choices=['Expert', 'Naive'])
ap.add_argument('--both', action='store_true', help='overlay Naive (dashed) on Expert')
args = ap.parse_args()

FS = 6.0
DATA = '../data/pca'
DUM = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}
TEST_COL = {0: '#377eb8', 1: '#4daf4a'}
CHOICE_COL = {0: '#377eb8', 1: '#4daf4a'}
_bright = sns.color_palette('bright')
TASK_COL = {'DPA': _bright[3], 'DualGo': _bright[0], 'DualNoGo': _bright[2]}
TASK_LS = {'DPA': '-', 'DualGo': '--', 'DualNoGo': ':'}
EPOCHS = {'STIM': (15, 18), 'DIST': (30, 33), 'TEST': (57, 60)}
OUT = 'figures/pseudo/time_inter'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def load(stage):
    """Native-unit traj (no per-comp z-score, no centring), clean trials, IDX, sign-oriented."""
    dum = DUM.format(stage)
    X = pkl_load(f'pseudo_traj_{dum}', path=DATA).astype(float)
    y = pkl_load(f'pseudo_labels_{dum}', path=DATA)
    labels = pkl_load(f'pseudo_marglabels_{dum}', path=DATA)
    IDX = {nm: labels.index(nm) for nm in dict.fromkeys(labels)}
    m = ((y.laser == 0) & (y.learning == stage) & (y.performance == 1)).to_numpy()
    Z = X[m]; yc = y[m].reset_index(drop=True)
    DLY, TST = np.arange(42, 54), np.arange(57, 66)
    B = (yc['sample'] == 1).to_numpy(); D = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    orient = {'sample': (B, ~B, DLY), 'test': (D, ~D, TST),
              'sample:test': (lick, ~lick, TST), 'tasks': (go, nogo, TST),
              'sample:tasks': (B & go, B & nogo, DLY), 'test:tasks': (D & go, D & nogo, TST),
              'sample:test:tasks': (lick & go, lick & nogo, TST)}
    for nm, (pos, neg, w) in orient.items():
        c = IDX[nm]
        if Z[pos][:, c][:, w].mean() < Z[neg][:, c][:, w].mean():
            Z[:, c, :] *= -1
    return Z, yc, IDX


# marginal -> list of (group-label, boolean-mask-builder, colour, linestyle)
def group_specs(yc):
    s = yc['sample'].to_numpy(); t = yc['test'].to_numpy(); tk = yc['tasks'].to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    specs = {
        'sample': [('A', s == 0, SAMPLE_COL[0], '-'), ('B', s == 1, SAMPLE_COL[1], '-')],
        'test': [('C', t == 0, TEST_COL[0], '-'), ('D', t == 1, TEST_COL[1], '-')],
        'tasks': [(k[4:] or 'DPA' if k.startswith('Dual') else k, tk == k, TASK_COL[k], '-')
                  for k in TASK_COL],
        'sample:test': [('match', lick, CHOICE_COL[1], '-'),
                        ('mismatch', ~lick, CHOICE_COL[0], '-')],
        'sample:tasks': [(f'{"AB"[sv]}/{TASK_LS_NM[k]}', (s == sv) & (tk == k),
                          SAMPLE_COL[sv], TASK_LS[k]) for sv in (0, 1)
                         for k in ('DualGo', 'DualNoGo')],
        'test:tasks': [(f'{"CD"[tv]}/{TASK_LS_NM[k]}', (t == tv) & (tk == k),
                        TEST_COL[tv], TASK_LS[k]) for tv in (0, 1)
                       for k in ('DualGo', 'DualNoGo')],
        'sample:test:tasks': [(f'{"nl/lk"[lv*3:lv*3+2]}/{TASK_LS_NM[k]}',
                               (lick == bool(lv)) & (tk == k), CHOICE_COL[lv], TASK_LS[k])
                              for lv in (0, 1) for k in ('DualGo', 'DualNoGo')],
    }
    return specs


TASK_LS_NM = {'DualGo': 'Go', 'DualNoGo': 'NoGo'}
MARGS = ['sample', 'test', 'tasks', 'sample:test', 'sample:tasks', 'test:tasks',
         'sample:test:tasks']

STAGES = ['Expert', 'Naive'] if args.both else [args.stage]
STAGE_LS = {'Expert': '-', 'Naive': (0, (3, 2))}
D = {st: load(st) for st in STAGES}

fig, AX = plt.subplots(2, 4, figsize=(13.5, 6.4))
AX = AX.ravel()

for st in STAGES:
    Z, yc, IDX = D[st]
    specs = group_specs(yc)
    stage_ls = STAGE_LS[st]
    for pi, nm in enumerate(MARGS):
        ax = AX[pi]
        c = IDX[nm]
        fracs = []
        for glabel, mask, color, ls in specs[nm]:
            a = Z[mask][:, c, :]
            mu = a.mean(0)
            inter = mu - mu.mean()                       # time:factor part (demean over time)
            fracs.append(inter.var() / (mu.mean()**2 + inter.var() + 1e-12))  # dynamic power frac
            se = a.std(0) / np.sqrt(max(len(a), 1))
            tt = np.arange(len(mu)) / FS
            use_ls = stage_ls if args.both else ls
            ax.plot(tt, inter, color=color, ls=use_ls, lw=1.3,
                    label=glabel if (st == STAGES[0] and not args.both) else None)
            if not args.both:
                ax.fill_between(tt, inter - se, inter + se, color=color, alpha=0.13, lw=0)
                ax.axhline(mu.mean(), color=color, ls=':', lw=0.7, alpha=0.6)  # static offset
        if st == STAGES[0]:
            ax.set_title(f'time:{nm}\n(dynamic frac {np.mean(fracs):.2f})', fontsize=7.5)

for ax in AX[:len(MARGS)]:
    for lo, hi in EPOCHS.values():
        ax.axvspan(lo / FS, hi / FS, color='0.9', zorder=0)
    ax.axhline(0, color='0.5', lw=0.6, zorder=1)
    ax.set_xlabel('time (s)', fontsize=7); ax.tick_params(labelsize=6)
    if not args.both and len(ax.get_legend_handles_labels()[0]):
        ax.legend(fontsize=5.5, frameon=False, ncol=2)
AX[0].set_ylabel('time-varying part (native)', fontsize=7)
AX[4].set_ylabel('time-varying part (native)', fontsize=7)

# legend / notes cell
AX[7].axis('off')
notes = ('dotted line = static (time-constant) offset\n'
         'solid = time:factor interaction (demeaned)\n'
         'dynamic frac = var_t(interaction)/var_t(component)')
handles = [Line2D([], [], color=SAMPLE_COL[0], lw=1.3, label='sample A'),
           Line2D([], [], color=SAMPLE_COL[1], lw=1.3, label='sample B'),
           Line2D([], [], color=TEST_COL[0], lw=1.3, label='test C'),
           Line2D([], [], color=TEST_COL[1], lw=1.3, label='test D'),
           Line2D([], [], color=CHOICE_COL[1], lw=1.3, label='lick/match'),
           Line2D([], [], color=CHOICE_COL[0], lw=1.3, label='no-lick/mismatch'),
           Line2D([], [], color=TASK_COL['DPA'], lw=1.3, label='DPA'),
           Line2D([], [], color='0.3', ls='--', lw=1.3, label='Go (ls)'),
           Line2D([], [], color='0.3', ls=':', lw=1.3, label='NoGo (ls)')]
if args.both:
    handles += [Line2D([], [], color='0.3', ls=STAGE_LS['Expert'], lw=1.3, label='Expert'),
                Line2D([], [], color='0.3', ls=STAGE_LS['Naive'], lw=1.3, label='Naive')]
AX[7].legend(handles=handles, fontsize=6.5, frameon=False, loc='upper center', ncol=2,
             title='groups')
AX[7].text(0.5, 0.08, notes, ha='center', va='bottom', fontsize=6.5,
           transform=AX[7].transAxes)

sfx = 'both' if args.both else args.stage
fig.suptitle(f'time × factor interactions (time-varying part of each dPCA marginal) — {sfx}',
             fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ('png', 'svg'):
    fig.savefig(f'{OUT}/{ext}/dpca_time_factor_interactions_{sfx}.{ext}',
                dpi=300, bbox_inches='tight')
print(f'saved figures/pseudo/time_inter/*/dpca_time_factor_interactions_{sfx}.*')

for st in STAGES:
    Z, yc, IDX = D[st]
    specs = group_specs(yc)
    print(f'\n{st}  dynamic fraction (var_t interaction / var_t component):')
    for nm in MARGS:
        c = IDX[nm]
        fr = []
        for _, mask, _, _ in specs[nm]:
            mu = Z[mask][:, c, :].mean(0)
            inter = mu - mu.mean()
            fr.append(inter.var() / (mu.mean()**2 + inter.var() + 1e-12))
        print(f'   time:{nm:20s} {np.mean(fr):.2f}')
