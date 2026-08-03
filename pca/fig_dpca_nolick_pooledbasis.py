"""No-lick push measured on the POOLED (pseudo-population) tasks axis, stage-specific:
Naive trials read through the POOLED NAIVE-fit axis, Expert trials through the POOLED EXPERT-fit axis.

Motivation: the per-mouse own-basis refit (fig_dpca_nolick_ownbasis.py) was axis-noise-limited — for
weak-lick mice the per-mouse tasks axis had no usable Go/NoGo separation, so its sign was random and the
cross-stage comparison washed out (p=0.73).  The pooled axis is estimated from ALL 9 mice's neurons at
once → one well-conditioned direction with ONE sign convention (oriented by DualGo, external lick anchor).
We keep per-mouse statistics by projecting each mouse's trials onto that pooled stage-axis and measuring
its DPA-delay displacement with the mouse's OWN baseline/scale (so cross-mouse offsets don't leak in).

So: pooled stage-specific DIRECTION (stable, Naive read through a Naive-defined axis) + per-mouse
displacement (paired Wilcoxon + mouse-bootstrap CI).

Usage: /home/leon/mambaforge/envs/dual/bin/python fig_dpca_nolick_pooledbasis.py [--correct]
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, argparse
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument('--correct', action='store_true', help='correct trials only (default: ALL trials)')
_ap.add_argument('--axis', default='tasks', choices=['tasks', 'choice'],
                 help='action axis to measure: tasks (lick/no-lick main effect, default) or '
                      'choice (sample:test interaction).')
_args = _ap.parse_args()
CORRECT, AX = _args.correct, _args.axis
TAG = ('_choice' if AX == 'choice' else '') + ('_correct' if CORRECT else '')
TRIALSET = 'correct trials' if CORRECT else 'all trials'
AXLAB = 'sample:test' if AX == 'choice' else 'tasks'
ANCHNAME = 'match-nonmatch' if AX == 'choice' else 'Go-NoGo'
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from src.pca.io import pkl_load

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

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
LATE = np.arange(39, 54)        # late-delay window for DPA depth
ANCHOR_TASKS = np.arange(30, 66)   # distractor+delay+test: Go vs NoGo
ANCHOR_CHOICE = np.arange(57, 72)  # test+response: match(lick) vs nonmatch — choice is expressed here
rng = np.random.default_rng(0)


def load_oriented(stg):
    """Load the pooled stage-DUM, return (Zr, y) with the action axis sign-oriented by an EXTERNAL
    lick anchor (tasks: DualGo>DualNoGo over the distractor/delay; choice: match>nonmatch at test)."""
    base = f'pseudo_ALL_{stg}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
    Z = pkl_load(f'pseudo_traj_{base}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{base}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{base}', path='../data/pca')
    isam, iax = lab.index('sample'), lab.index(AXLAB)
    Zr = Z[:, [isam, iax], :].astype(float)
    if AX == 'choice':                                          # match(lick)=choice 1 vs nonmatch=choice 0 at test
        pos = ((y.laser == 0) & (y.choice == 1)).to_numpy(); neg = ((y.laser == 0) & (y.choice == 0)).to_numpy()
        anchor = float(Zr[pos][:, 1, ANCHOR_CHOICE].mean() - Zr[neg][:, 1, ANCHOR_CHOICE].mean())
    else:                                                       # DualGo(lick) vs DualNoGo
        pos = ((y.laser == 0) & (y.tasks == 'DualGo')).to_numpy(); neg = ((y.laser == 0) & (y.tasks == 'DualNoGo')).to_numpy()
        anchor = float(Zr[pos][:, 1, ANCHOR_TASKS].mean() - Zr[neg][:, 1, ANCHOR_TASKS].mean())
    if anchor < 0:
        Zr[:, 1, :] *= -1; anchor = -anchor
    print(f'  pooled {stg} basis: {ANCHNAME} lick anchor = {anchor:+.2f}')
    return Zr, y


def mouse_depth(Zr, y, mm, stg):
    """DPA-delay tasks-axis depth for one mouse on this pooled stage-axis (own baseline/scale, std 2.8)."""
    refm = ((y.laser == 0) & (y.tasks == 'DPA') & (y.mouse == mm)).to_numpy()
    mu0 = Zr[refm][:, :, 0:12].mean((0, 2), keepdims=True); sd0 = Zr[refm].std((0, 2), keepdims=True)
    Zn = (Zr - mu0) / sd0 * 2.8
    m = (y.laser == 0) & (y.learning == stg) & (y.tasks == 'DPA') & (y.mouse == mm)
    if CORRECT: m = m & (y.performance == 1)
    m = m.to_numpy(); yc = y[m].reset_index(drop=True); Zs = Zn[m]
    means = [Zs[(yc['sample'] == s).to_numpy()].mean(0) for s in (0, 1)]
    return float(np.mean([means[s][1, LATE].mean() for s in (0, 1)]))


ZN, yN = load_oriented('Naive')
ZE, yE = load_oriented('Expert')
dep = np.array([[mouse_depth(ZN, yN, mm, 'Naive'), mouse_depth(ZE, yE, mm, 'Expert')] for mm in MICE])
for mm, (dN, dE) in zip(MICE, dep):
    print(f'{mm:8s}  depth  Naive {dN:+.2f} -> Expert {dE:+.2f}   ({"deeper" if dE < dN else "shallower"})')

p = wilcoxon(dep[:, 0], dep[:, 1]).pvalue
diff = dep[:, 1] - dep[:, 0]                                   # Expert - Naive (negative = deeper no-lick)
boot = np.array([diff[rng.integers(0, 9, 9)].mean() for _ in range(5000)])
ci = np.percentile(boot, [2.5, 97.5])
print(f'\nPOOLED-BASIS depth  Naive {dep[:,0].mean():+.2f} -> Expert {dep[:,1].mean():+.2f}')
print(f'  Wilcoxon p={p:.3f}; deepen {(diff<0).sum()}/9; both<0 {(dep<0).all(1).sum()}/9')
print(f'  Naive->Expert change {diff.mean():+.2f}  mouse-bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]')

fig, ax = plt.subplots(figsize=(3.6, 3.6))
for row in dep: ax.plot([0, 1], row, '-', color='0.6', lw=1.0, marker='o', ms=4, mfc='0.4', mec='none')
ax.plot([0, 1], dep.mean(0), '-', color='k', lw=1.6, marker='o', ms=6, zorder=5)
ax.axhline(0, color='r', lw=0.8, ls='--')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive\n(pooled Naive axis)', 'Expert\n(pooled Expert axis)'])
ax.set_xlim(-0.3, 1.3)
ax.set_title(f'{AX}-axis DPA-delay depth, POOLED stage-axis\nWilcoxon p={p:.3f}, ΔCI [{ci[0]:+.2f},{ci[1]:+.2f}]  [{TRIALSET}]', loc='left', fontsize=TITLE_FS)
ax.set_ylabel(f'{AX}-axis depth u_y  (− = no-lick)'); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
out = f'figures/pseudo/flow/lowrank/png/dpca_nolick_pooledbasis{TAG}.png'
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
