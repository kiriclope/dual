"""No-lick push vs CI removal (q0/q1/q2). Tasks-axis (condition-dependent) DPA late-delay depth, oriented
no-lick<0, Naive vs Expert. Left = pooled (all mice, one dPCA); right = per-mouse mean±SEM. Shows: the
LEARNING deepening (Expert below Naive) survives CI removal at every q (both panels); pooled, the Naive
push collapses to ~0 with CI removal (the shared ramp), while per-mouse it persists (CI subspace under-
estimated from ~370 neurons/mouse). Saves PNG(300)+SVG."""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
DELAY = np.arange(39, 54); BL = np.arange(0, 12); QS = [0, 1, 2]
CN, CE = '#888888', '#cc3311'                                  # Naive grey, Expert red


def depth(dum):
    Z = pkl_load(f'pseudo_traj_{dum}', path='../data/pca').astype(float)
    y = pkl_load(f'pseudo_labels_{dum}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{dum}', path='../data/pca'); t = Z[:, lab.index('tasks'), :]
    be = ((y.laser == 0) & (y.tasks == 'DPA')).to_numpy(); sd = t[be].std(); cbl = t[be][:, BL].mean()
    def v(stg):
        m = ((y.laser == 0) & (y.learning == stg) & (y.tasks == 'DPA') & (y.performance == 1)).to_numpy()
        return (t[m].mean(0)[DELAY].mean() - cbl) / sd * 2.8
    e = v('Expert'); s = -1 if e > 0 else 1
    return s * v('Naive'), s * e


def pooled_dum(q): return f'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test-tasks{"" if q==0 else f"_ci{q}"}_dpca'
def mouse_dum(mm, q): return f'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca_{mm}{"" if q==0 else f"_ci{q}"}'

pooled = {q: depth(pooled_dum(q)) for q in QS}
perm = {q: np.array([depth(mouse_dum(mm, q)) for mm in MICE]) for q in QS}   # (9,2) per q

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
x = np.array(QS)
# pooled
a1.plot(x, [pooled[q][0] for q in QS], '-o', color=CN, lw=1.6, ms=5, label='Naive')
a1.plot(x, [pooled[q][1] for q in QS], '-o', color=CE, lw=1.6, ms=5, label='Expert')
a1.axhline(0, color='0.6', lw=0.8, ls='--')
a1.set_title('Pooled (one dPCA, 3319 neurons)', loc='left', fontsize=TITLE_FS)
a1.legend(frameon=False, loc='upper right')
# per-mouse mean±SEM
for j, (c, lbl) in enumerate([(CN, 'Naive'), (CE, 'Expert')]):
    mu = np.array([perm[q][:, j].mean() for q in QS]); se = np.array([perm[q][:, j].std(ddof=1) / 3 for q in QS])
    a2.errorbar(x + (j - 0.5) * 0.04, mu, yerr=se, fmt='-o', color=c, lw=1.6, ms=5, capsize=2.5, label=lbl)
a2.axhline(0, color='0.6', lw=0.8, ls='--')
a2.set_title('Per-mouse (mean ± SEM, n=9)', loc='left', fontsize=TITLE_FS)
for ax in (a1, a2):
    ax.set_xticks(QS); ax.set_xticklabels([f'q{q}' for q in QS]); ax.set_xlabel('CI dimensions removed')
    ax.spines[['top', 'right']].set_visible(False)
a1.set_ylabel('tasks-axis DPA delay depth  (− = no-lick)')
fig.suptitle('No-lick push vs CI removal: learning deepening survives; pooled Naive push collapses to ~0', y=1.0, fontsize=TITLE_FS)
fig.tight_layout()
out = 'figures/pseudo/flow/lowrank/png/dpca_nolick_ci_qsweep.png'
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
for q in QS:
    print(f'q{q}: pooled N {pooled[q][0]:+.2f} E {pooled[q][1]:+.2f} | per-mouse N {perm[q][:,0].mean():+.2f} E {perm[q][:,1].mean():+.2f}')
