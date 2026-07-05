"""Tasks-marginal dPCA trajectory, with the condition-independent time component added back.

Section 4's no-lick push lives on the `tasks` axis, which is ~0.55 aligned with the
`time` (condition-independent) axis.  dPCA *demixes* them into separate components; this
script shows what the tasks trajectory looks like when the shared time ramp is put back:

    z_recon_cond(t) = z_tasks_cond(t) + z_time(t)          (time is the same for all conds)

Because time is condition-independent, adding it shifts all three task lines (DPA / Go /
NoGo) by the SAME time-varying amount — so the *between-condition* structure (the DPA dip)
is untouched; only the common ramp is restored.  Components are in NATIVE dPCA units
(baseline-centred, NOT per-component z-scored) so the sum is meaningful.  Top component (q0).

Palette: task DPA/Go/NoGo canonical bright.  Saved to figures/pseudo/tasks_time/.
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
import seaborn as sns

from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'
sns.set_context('paper'); sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='Expert', choices=['Expert', 'Naive'])
ap.add_argument('--both', action='store_true', help='row per stage (Naive + Expert)')
args = ap.parse_args()

FS = 6.0
DATA = '../data/pca'
DUM = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
_bright = sns.color_palette('bright')
TASK_COL = {'DPA': _bright[3], 'DualGo': _bright[0], 'DualNoGo': _bright[2]}
TASK_NAME = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
EPOCHS = {'STIM': (15, 18), 'DIST': (30, 33), 'TEST': (57, 60)}
BL = np.arange(0, 12)          # pre-stim baseline for native centring
OUT = 'figures/pseudo/tasks_time'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def load(stage):
    """Native-unit (baseline-centred) tasks-q0 and time-q0 trajectories, per task.
    Sign-oriented: tasks Go>NoGo @ test, time late>early — comparable across stages."""
    dum = DUM.format(stage)
    X = pkl_load(f'pseudo_traj_{dum}', path=DATA).astype(float)
    y = pkl_load(f'pseudo_labels_{dum}', path=DATA)
    labels = pkl_load(f'pseudo_marglabels_{dum}', path=DATA)
    IDX = {nm: labels.index(nm) for nm in dict.fromkeys(labels)}
    m = ((y.laser == 0) & (y.learning == stage) & (y.performance == 1)).to_numpy()
    Z = X[m]
    yc = y[m].reset_index(drop=True)
    ct, cT = IDX['tasks'], IDX['time']
    Z = Z - Z[:, :, BL].mean(2, keepdims=True)         # native baseline-centre (no z-score)
    TST = np.arange(57, 66)
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    if Z[go][:, ct][:, TST].mean() < Z[nogo][:, ct][:, TST].mean():
        Z[:, ct, :] *= -1
    if Z[:, cT, 60:].mean() < Z[:, cT, :12].mean():
        Z[:, cT, :] *= -1

    def cond_mean(comp, mask=None):
        a = Z[:, comp, :] if mask is None else Z[mask][:, comp, :]
        n = max(len(a), 1)
        return a.mean(0), a.std(0) / np.sqrt(n)

    tasks = {tk: cond_mean(ct, (yc['tasks'] == tk).to_numpy()) for tk in TASK_COL}
    time0 = cond_mean(cT)                               # condition-independent
    return tasks, time0


def _band(ax, mu, se, color, ls='-', lw=1.5, label=None, alpha=0.15):
    t = np.arange(len(mu)) / FS
    ax.plot(t, mu, color=color, ls=ls, lw=lw, label=label)
    if se is not None:
        ax.fill_between(t, mu - se, mu + se, color=color, alpha=alpha, lw=0)


def _decorate(ax, title, ylab=None):
    for lo, hi in EPOCHS.values():
        ax.axvspan(lo / FS, hi / FS, color='0.9', zorder=0)
    ax.axhline(0, color='0.5', lw=0.6, zorder=1)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('time (s)', fontsize=7)
    if ylab:
        ax.set_ylabel(ylab, fontsize=7)
    ax.tick_params(labelsize=6)


STAGES = ['Naive', 'Expert'] if args.both else [args.stage]
fig, AX = plt.subplots(len(STAGES), 3, figsize=(11, 3.1 * len(STAGES)), squeeze=False)

for r, st in enumerate(STAGES):
    tasks, (tmu, tse) = load(st)
    ax0, ax1, ax2 = AX[r]

    # panel 0: tasks component alone
    for tk in TASK_COL:
        mu, se = tasks[tk]
        _band(ax0, mu, se, TASK_COL[tk], label=TASK_NAME[tk] if r == 0 else None)
    _decorate(ax0, f'{st}: tasks component (native)', ylab='dPCA (native units)')

    # panel 1: time component alone (condition-independent)
    _band(ax1, tmu, tse, '#333333', label='time (q0)')
    _decorate(ax1, f'{st}: time component (condition-indep.)')

    # panel 2: tasks + time, per task
    for tk in TASK_COL:
        mu, se = tasks[tk]
        _band(ax2, mu + tmu, np.sqrt(se**2 + tse**2), TASK_COL[tk],
              label=TASK_NAME[tk] if r == 0 else None)
    # faint tasks-alone for reference (dashed)
    for tk in TASK_COL:
        mu, _ = tasks[tk]
        _band(ax2, mu, None, TASK_COL[tk], ls=(0, (2, 2)), lw=0.9)
    _decorate(ax2, f'{st}: tasks + time (reconstructed)')

    if r == 0:
        ax0.legend(fontsize=6, frameon=False, ncol=3)
        ax2.legend(fontsize=6, frameon=False, ncol=3, title='solid=tasks+time, dashed=tasks')

    # numeric: DPA dip depth at delay (42-54) tasks-alone vs reconstructed
    dpa_t = tasks['DPA'][0]
    dip_alone = dpa_t[np.arange(42, 54)].mean()
    dip_recon = (dpa_t + tmu)[np.arange(42, 54)].mean()
    print(f'{st}: DPA tasks-alone delay = {dip_alone:+.2f} | +time = {dip_recon:+.2f} '
          f'| time delay ramp = {tmu[np.arange(42,54)].mean():+.2f}')

# share y within the tasks-alone col and within the recon col
for col in (0, 2):
    lo = min(AX[r][col].get_ylim()[0] for r in range(len(STAGES)))
    hi = max(AX[r][col].get_ylim()[1] for r in range(len(STAGES)))
    for r in range(len(STAGES)):
        AX[r][col].set_ylim(lo, hi)

sfx = 'both' if args.both else args.stage
fig.suptitle(f'tasks dPCA + condition-independent time — {sfx}  (top component)', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ('png', 'svg'):
    fig.savefig(f'{OUT}/{ext}/dpca_tasks_plus_time_{sfx}.{ext}', dpi=300, bbox_inches='tight')
print(f'saved figures/pseudo/tasks_time/*/dpca_tasks_plus_time_{sfx}.*')
