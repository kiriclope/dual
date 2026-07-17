"""Go vs NoGo (Dual) trajectories projected on the DPA ACTION axis — per stage × normalisation.

The DPA action axis = the DPA choice/lick decoder trained at the DPA lick moment (bins 57–63), read
across test-time. We project DUAL Go (DualGo) and NoGo (DualNoGo) trials onto it: if the GNG lick (Go)
shares the DPA lick direction, Go should ride toward the lick pole (+) and NoGo toward no-lick (−).
Shown under the three per-mouse normalisations we compared for the push (all denominators from DPA
trials, so it is the SAME per-mouse unit as the main figure's action axis):
  baseline-std      = std over the baseline window (SNR-weighted)
  signal-std/eqnorm = std over all trials × all bins (democratic-ish, noise-inflated)
  pooled-evoked     = temporal std of the class-signed (DPA lick) pooled mean trajectory (evoked scale)

Rows = normalisation, cols = stage (Naive | Expert). Go = blue, NoGo = green; mean±SEM over 9 mice.

Output: figures/overlaps/action/{png,svg}/overlaps_gng_on_dpaaxis.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DATA = '../data/overlaps'
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BL = np.arange(0, 12); ACT = np.arange(57, 63); xt = np.linspace(0, 14, 84)

y = pkl_load(f'labels_{BDUM}', path=DATA); X = np.asarray(pkl_load(f'X_{BDUM}', path=DATA))
ch = (y.target == 'choice').to_numpy()                                 # DPA choice/action axis rows (all tasks)
D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True)
mo = yc.mouse.to_numpy(); tsk = yc.tasks.to_numpy(); lz = (yc.laser == 0).to_numpy()
lrn = yc.learning.to_numpy(); chc = yc.choice.to_numpy()

NORMS = [('baseline-std', 'bl'), ('signal-std (eqnorm)', 'sig'), ('pooled-evoked', 'pe')]
GROUPS = [('Go (DualGo)', 'DualGo', '#1f77b4', '-'), ('NoGo (DualNoGo)', 'DualNoGo', '#2ca02c', '--')]


def sd_mu(m, kind):
    dpa = lz & (tsk == 'DPA') & (mo == m)                              # denominator from DPA trials (defines the axis unit)
    mu = D[dpa][:, BL].mean()
    if kind == 'bl':
        return D[dpa][:, BL].std(), mu
    if kind == 'sig':
        return D[dpa].std(), mu
    s = np.where(chc[dpa] == 1, 1.0, -1.0)                             # class-signed by DPA lick/no-lick
    vbar = (s[:, None] * D[dpa]).mean(0)
    return (vbar - vbar[BL].mean()).std(), mu


def curve(kind, stage, task):
    per = []
    for m in MICE:
        sd, mu = sd_mu(m, kind)
        sel = lz & (lrn == stage) & (tsk == task) & (mo == m)
        if sel.sum() < 5:
            continue
        v = (D[sel].mean(0) - mu) / (sd + 1e-9)
        per.append(v)
    A = np.stack(per)
    return A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))


fig, axs = plt.subplots(len(NORMS), 2, figsize=(9.5, 9.5), sharex=True)
for ri, (nlab, nkey) in enumerate(NORMS):
    for ci, stage in enumerate(('Naive', 'Expert')):
        ax = axs[ri, ci]
        for glab, gtask, col, ls in GROUPS:
            mu, se = curve(nkey, stage, gtask)
            ax.plot(xt, mu, color=col, lw=2.0, ls=ls, label=glab, zorder=3)
            ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color='0.75', lw=0.7)
        ax.axvspan(xt[42], xt[45], color='0.85', alpha=0.4, lw=0)      # GNG reward (Go lick moment)
        ax.axvspan(xt[57], xt[63], color='gold', alpha=0.18, lw=0)     # DPA action window (axis train)
        if ri == 0:
            ax.set_title(stage, fontweight='bold', fontsize=11)
        if ri == len(NORMS) - 1:
            ax.set_xlabel('time (s)')
        if ci == 0:
            ax.set_ylabel(f'proj. on DPA action axis\n[{nlab}]', fontsize=8)
        if ri == 0 and ci == 0:
            ax.legend(frameon=False, fontsize=7.5, loc='upper left')
            ax.text(xt[43], ax.get_ylim()[1], 'GNG\nrwd', fontsize=6, va='top', color='0.4')
            ax.text(xt[58], ax.get_ylim()[1], 'DPA\naction', fontsize=6, va='top', color='0.4')
fig.suptitle('Go vs NoGo (Dual) projected on the DPA action axis (choice decoder @57–63) — per stage × normalisation.\n'
             'If Go rides toward the lick (+) pole and NoGo toward no-lick (−), the GNG lick shares the DPA lick direction.',
             fontsize=9.5, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT = 'figures/overlaps/action'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
p = f'{OUT}/png/overlaps_gng_on_dpaaxis.png'
fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
