"""dPCA time-marginal and interaction-marginal trajectories.

The story figure (section 2) plots the four MAIN-effect / low-order marginals of the
f-sample-test-tasks dPCA (sample, test, sample:test=choice, tasks).  This diagnostic
draws the pieces that figure leaves out:

  - time                (condition-independent ramp; q0 and q1 components)
  - sample:tasks        (how sample coding is modulated by task)
  - test:tasks          (how test coding is modulated by task)
  - sample:test:tasks   (how the match/choice code is modulated by task)

sample:test (choice) is included too for reference (it IS an interaction).  Each panel
is the top component of that marginal vs time, condition-mean ± SEM, grouped by the
interacting variables.  Expert by default; --naive for the Naive DUM; --both overlays.

Palette matches the population convention (sample A/B #332288/#44AA99, test C/D
#377eb8/#4daf4a, task DPA/Go/NoGo canonical bright).  Saved to figures/pseudo/marg_inter/.
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
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}          # A indigo / B teal
TEST_COL = {0: '#377eb8', 1: '#4daf4a'}            # C blue / D green
CHOICE_COL = {0: '#377eb8', 1: '#4daf4a'}          # no-lick / lick
_bright = sns.color_palette('bright')
TASK_COL = {'DPA': _bright[3], 'DualGo': _bright[0], 'DualNoGo': _bright[2]}
TASK_LS = {'DPA': '-', 'DualGo': '--', 'DualNoGo': ':'}
EPOCHS = {'STIM': (15, 18), 'DIST': (30, 33), 'TEST': (57, 60)}
OUT = 'figures/pseudo/marg_inter'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def load(stage):
    """Z-scored, sign-oriented latents Z (trials, comp, time), clean trials, IDX map.
    Orientation: main effects to canonical convention; interactions oriented on their
    leading contrast; time late>early — so sign is comparable Naive vs Expert."""
    dum = DUM.format(stage)
    X = pkl_load(f'pseudo_traj_{dum}', path=DATA)
    y = pkl_load(f'pseudo_labels_{dum}', path=DATA)
    labels = pkl_load(f'pseudo_marglabels_{dum}', path=DATA)
    IDX = {nm: labels.index(nm) for nm in dict.fromkeys(labels)}
    m = ((y.laser == 0) & (y.learning == stage) & (y.performance == 1)).to_numpy()
    Z = X[m].astype(float)
    Z = (Z - Z.mean((0, 2), keepdims=True)) / Z.std((0, 2), keepdims=True)
    yc = y[m].reset_index(drop=True)
    DLY, TST = np.arange(42, 54), np.arange(57, 66)
    B = (yc['sample'] == 1).to_numpy(); D = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    orient = {                                     # (pos, neg, window)
        'sample': (B, ~B, DLY), 'test': (D, ~D, TST),
        'sample:test': (lick, ~lick, TST), 'tasks': (go, nogo, TST),
        'sample:tasks': (B & go, B & nogo, DLY),   # sample-B code: Go>NoGo
        'test:tasks': (D & go, D & nogo, TST),     # test-D code:   Go>NoGo
        'sample:test:tasks': (lick & go, lick & nogo, TST),  # match code: Go>NoGo
    }
    for nm, (pos, neg, w) in orient.items():
        if nm in IDX:
            c = IDX[nm]
            if Z[pos][:, c][:, w].mean() < Z[neg][:, c][:, w].mean():
                Z[:, c, :] *= -1
    tc = IDX['time']
    if Z[:, tc, 60:].mean() < Z[:, tc, :12].mean():
        Z[:, tc, :] *= -1
    return Z, yc, IDX


def band(ax, Z, mask, comp, color, ls='-', label=None, alpha=0.18):
    a = Z[mask][:, comp, :]
    n = max(a.shape[0], 1)
    mu, se = a.mean(0), a.std(0) / np.sqrt(n)
    t = np.arange(a.shape[1]) / FS
    ax.plot(t, mu, color=color, ls=ls, lw=1.4, label=label)
    ax.fill_between(t, mu - se, mu + se, color=color, alpha=alpha, lw=0)


def decorate(ax, title):
    for lo, hi in EPOCHS.values():
        ax.axvspan(lo / FS, hi / FS, color='0.9', zorder=0)
    ax.axhline(0, color='0.5', lw=0.6, zorder=1)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('time (s)', fontsize=7)
    ax.tick_params(labelsize=6)


STAGES = ['Expert', 'Naive'] if args.both else [args.stage]
STAGE_LS = {'Expert': '-', 'Naive': (0, (3, 2))}
D = {st: load(st) for st in STAGES}

fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2))
ax = ax.ravel()

for st in STAGES:
    Z, yc, IDX = D[st]
    lsq = STAGE_LS[st]
    B = (yc['sample'] == 1).to_numpy(); D_ = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    task = yc['tasks'].to_numpy()
    all_ = np.ones(len(yc), bool)

    # 0. time — condition-independent, q0 and q1
    tc = IDX['time']
    band(ax[0], Z, all_, tc, '#333333', ls=lsq, label=f'{st} q0')
    band(ax[0], Z, all_, tc + 1, '#cc6677', ls=lsq, label=f'{st} q1')

    # 1. sample:test  (= choice / match)
    c = IDX['sample:test']
    band(ax[1], Z, lick, c, CHOICE_COL[1], ls=lsq, label='match (lick)' if st == STAGES[0] else None)
    band(ax[1], Z, ~lick, c, CHOICE_COL[0], ls=lsq, label='mismatch (no-lick)' if st == STAGES[0] else None)

    # 2. sample:tasks — sample colour, task line-style
    c = IDX['sample:tasks']
    for s in (0, 1):
        for tk in ('DualGo', 'DualNoGo'):
            band(ax[2], Z, (yc['sample'] == s).to_numpy() & (task == tk),
                 c, SAMPLE_COL[s], ls=TASK_LS[tk], alpha=0.12)

    # 3. test:tasks — test colour, task line-style
    c = IDX['test:tasks']
    for tv in (0, 1):
        for tk in ('DualGo', 'DualNoGo'):
            band(ax[3], Z, (yc['test'] == tv).to_numpy() & (task == tk),
                 c, TEST_COL[tv], ls=TASK_LS[tk], alpha=0.12)

    # 4. sample:test:tasks — choice colour, task line-style
    c = IDX['sample:test:tasks']
    for lk in (0, 1):
        for tk in ('DualGo', 'DualNoGo'):
            band(ax[4], Z, (lick == bool(lk)) & (task == tk),
                 c, CHOICE_COL[lk], ls=TASK_LS[tk], alpha=0.12)

titles = ['time (condition-independent)', 'sample:test  (choice / match)',
          'sample:tasks', 'test:tasks', 'sample:test:tasks  (choice × task)']
for i, ttl in enumerate(titles):
    decorate(ax[i], ttl)
ax[0].legend(fontsize=6, frameon=False, ncol=len(STAGES))
ax[1].legend(fontsize=6, frameon=False)

# legend panel (last cell) for interaction groupings
ax[5].axis('off')
handles = [Line2D([], [], color=SAMPLE_COL[0], lw=1.4, label='sample A'),
           Line2D([], [], color=SAMPLE_COL[1], lw=1.4, label='sample B'),
           Line2D([], [], color=TEST_COL[0], lw=1.4, label='test C'),
           Line2D([], [], color=TEST_COL[1], lw=1.4, label='test D'),
           Line2D([], [], color=CHOICE_COL[1], lw=1.4, label='lick / match'),
           Line2D([], [], color=CHOICE_COL[0], lw=1.4, label='no-lick / mismatch'),
           Line2D([], [], color='0.3', ls=TASK_LS['DualGo'], lw=1.4, label='DualGo'),
           Line2D([], [], color='0.3', ls=TASK_LS['DualNoGo'], lw=1.4, label='DualNoGo')]
if args.both:
    handles += [Line2D([], [], color='0.3', ls=STAGE_LS['Expert'], lw=1.4, label='Expert'),
                Line2D([], [], color='0.3', ls=STAGE_LS['Naive'], lw=1.4, label='Naive')]
ax[5].legend(handles=handles, fontsize=7, frameon=False, loc='center', ncol=2,
             title='interaction groupings')

sfx = 'both' if args.both else args.stage
fig.suptitle(f'dPCA time & interaction marginals — {sfx}  '
             '(f-sample-test-tasks, top component per marginal)', fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ('png', 'svg'):
    fig.savefig(f'{OUT}/{ext}/dpca_time_interactions_{sfx}.{ext}',
                dpi=300, bbox_inches='tight')
print(f'saved figures/pseudo/marg_inter/*/dpca_time_interactions_{sfx}.*')

# quick numeric summary
for st in STAGES:
    Z, yc, IDX = D[st]
    tc = IDX['time']
    ramp = Z[:, tc, 60:72].mean() - Z[:, tc, :12].mean()
    print(f'{st}: time q0 ramp (late-early) = {ramp:+.2f}')
    for nm in ('sample:tasks', 'test:tasks', 'sample:test:tasks'):
        c = IDX[nm]
        amp = np.abs(Z[:, c, 45:66].mean(0)).max()
        print(f'   {nm:20s} peak |comp| (delay→test) = {amp:.2f}')
