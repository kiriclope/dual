"""DIAGNOSTIC — why the A-third-row action trajectories look different from a position-preserving view.

Same DPA lick/no-lick trajectories on each action axis, under two normalisations:
  'traj'  = EXACTLY the main A third row (`_traj_curves`): each group's per-mouse MEAN trajectory is
            BL-centred and divided by ITS OWN temporal std (v.std()). Per-group, per-mouse → unit
            "wiggle". This is a SHAPE view: it discards absolute position AND relative amplitude between
            groups, so you CANNOT read depth/push off it.
  'bl'    = shared per-mouse scale (divide every group by the SAME baseline std, BL-centred). Preserves
            position → this is what the push panel needs.
Rows = action axis (DPA @57–63, GNG @39–45). Cols = normalisation. Each cell: DPA lick (solid) &
no-lick (dashed), Naive (blue) & Expert (red). Raw decoder sign (lick/Go = +), no circular flip.

Output: figures/overlaps/action/{png,svg}/overlaps_action_norm_diag.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DATA = '../data/overlaps'
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BL = np.arange(0, 12); LD = np.arange(45, 53); xt = np.linspace(0, 14, 84)

y = pkl_load(f'labels_{BDUM}', path=DATA)
X = np.asarray(pkl_load(f'X_{BDUM}', path=DATA))

AXES = [('DPA action axis\n(choice decoder @57–63)', 'choice', np.arange(57, 63)),
        ('GNG action axis\n(gng decoder @39–45)',    'gng',    np.arange(39, 45))]
NORMS = [('traj', 'ARTIFACT — per-GROUP unit-std (A 3rd row):\neach curve ÷ its OWN temporal std'),
         ('bl',   'NO ARTIFACT — shared scale:\nall curves ÷ the SAME baseline std')]


def curves(target, act, norm, stage):
    m = (y.target == target).to_numpy()
    D = X[m][:, 1, act, :].mean(1).astype(float)
    yy = y[m].reset_index(drop=True)
    mo = yy.mouse.to_numpy()
    base = ((yy.laser == 0) & (yy.learning == stage) & (yy.performance == 1) & (yy.tasks == 'DPA')).to_numpy()
    groups = {'lick': base & (yy.choice.to_numpy() == 1), 'nolick': base & (yy.choice.to_numpy() == 0)}
    res = {}
    for key, msk in groups.items():
        permouse = []
        for mm in MICE:
            sm = mo == mm; sel = msk & sm
            if sel.sum() < 3:
                continue
            v = D[sel].mean(0)                                   # per-mouse per-GROUP mean trajectory (raw sign)
            v = v - v[BL].mean()
            sd = (v.std() if norm == 'traj' else D[sm][:, BL].std())   # 'traj' = own temporal std; 'bl' = shared
            permouse.append(v / (sd + 1e-9))
        if permouse:
            A = np.stack(permouse); res[key] = (A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A)))
    return res


fig, axs = plt.subplots(len(AXES), len(NORMS), figsize=(11.5, 8.5), sharex=True)
STCOL = {'Naive': '#4477AA', 'Expert': '#CC3311'}
_xLD = xt[LD].mean()
for ri, (albl, tgt, act) in enumerate(AXES):
    for ci, (norm, ndesc) in enumerate(NORMS):
        ax = axs[ri, ci]; gaps = []
        for si, stage in enumerate(('Naive', 'Expert')):
            c = curves(tgt, act, norm, stage); col = STCOL[stage]
            for key, ls, lw in (('lick', '-', 2.2), ('nolick', '--', 1.6)):
                if key in c:
                    mu, se = c[key]
                    ax.plot(xt, mu, color=col, lw=lw, ls=ls, zorder=3)
                    ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.12, lw=0)
            if 'lick' in c and 'nolick' in c:
                vl = c['lick'][0][LD].mean(); vn = c['nolick'][0][LD].mean()
                xm = _xLD + (-0.5 + si)                                  # Naive left of LD, Expert right
                ax.plot([xm, xm], [vn, vl], color=col, lw=1.2, zorder=6)                 # the lick↔no-lick GAP
                ax.scatter([xm], [vl], s=42, facecolor=col, edgecolor='k', lw=0.6, zorder=7)   # lick = filled
                ax.scatter([xm], [vn], s=42, facecolor='w', edgecolor=col, lw=1.3, zorder=7)   # no-lick = open
                ax.annotate(f'{stage}\ngap={vl - vn:+.2f}', (xm, max(vl, vn)), xytext=(0, 4),
                            textcoords='offset points', ha='center', va='bottom', fontsize=6.5, color=col)
                gaps.append(vl - vn)
        ax.axhline(0, color='0.7', lw=0.7)
        ax.axvspan(xt[LD[0]], xt[LD[-1]], color='0.85', alpha=0.5, lw=0)
        ax.axvspan(xt[act[0]], xt[act[-1]], color='gold', alpha=0.2, lw=0)
        if ri == len(AXES) - 1:
            ax.set_xlabel('time (s)')
        if ci == 0:
            ax.set_ylabel(albl, fontsize=8)
        ax.set_title(f"{ndesc}\nLD lick−nolick gap:  Naive {gaps[0]:+.2f}   Expert {gaps[1]:+.2f}", fontsize=7.5)
fig.legend(handles=[Line2D([0], [0], color=STCOL['Naive'], lw=2.2, label='Naive'),
                    Line2D([0], [0], color=STCOL['Expert'], lw=2.2, label='Expert'),
                    Line2D([0], [0], color='0.3', lw=2.2, ls='-', label='lick (choice=1)'),
                    Line2D([0], [0], color='0.3', lw=1.6, ls='--', label='no-lick (choice=0)')],
           frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.99))
fig.suptitle("The A-3rd-row normalisation MANUFACTURES an LD lick↔no-lick gap.  LEFT (per-group unit-std): "
             "flat no-lick ÷ small std is inflated → big apparent gap.\nRIGHT (shared scale, the honest depth): "
             "the same LD gap nearly vanishes.  Markers = LD lick (filled) vs no-lick (open); bar = the gap.",
             fontsize=8.5, y=1.05)
fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT = 'figures/overlaps/action'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
p = f'{OUT}/png/overlaps_action_norm_diag.png'
fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
