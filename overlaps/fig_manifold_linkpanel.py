"""Fig-2 → manifold LINKING PANEL (scratch/dev).

Recap of Fig 2's low-dimensional dPCA subspace, shown as the plane (sample axis × action[tasks] axis), with the
DPA / Go / NoGo delay states projected onto it. The point: the SAMPLE (memory) axis separates odour A vs B the
SAME way in all three task contexts (a reused code), orthogonal to the pre-existing action axis. This is the
visual bridge from Fig 2 ("low-dimensional, factorised") to the manifold figure ("abstract & reused across tasks").

Standalone dev script → later folded into fig_overlaps_manifold.py.
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from src.pca.io import pkl_load

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}                 # A indigo, B teal
TASK_LS = {'DPA': '-', 'Go': '--', 'NoGo': ':'}
TASKDUM = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
FS = 6.0


def load_marg(dum, stage='Expert'):                        # copied from pca/fig_dpca_story_main.py
    X = pkl_load(f'pseudo_traj_{dum}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{dum}', path='../data/pca')
    labels = pkl_load(f'pseudo_marglabels_{dum}', path='../data/pca')
    IDX = {nm: labels.index(nm) for nm in dict.fromkeys(labels)}
    m = ((y.laser == 0) & (y.learning == stage) & (y.performance == 1)).to_numpy()
    Z = X[m].astype(float)
    Z = (Z - Z.mean((0, 2), keepdims=True)) / Z.std((0, 2), keepdims=True)
    yc = y[m].reset_index(drop=True)
    DLYw, TST = np.arange(42, 54), np.arange(57, 66)
    B = (yc['sample'] == 1).to_numpy(); Dd = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    for nm, (pos, neg, w) in {'sample': (B, ~B, DLYw), 'test': (Dd, ~Dd, TST),
                              'sample:test': (lick, ~lick, TST), 'tasks': (go, nogo, TST)}.items():
        if nm in IDX and Z[pos][:, IDX[nm]][:, w].mean() < Z[neg][:, IDX[nm]][:, w].mean():
            Z[:, IDX[nm], :] *= -1
    return Z, yc, IDX


def link_panel(ax, stage='Expert', win=np.arange(30, 54)):
    """sample axis (x) × action/tasks axis (y): per (task × sample) mean trajectory over `win`, dot at window end."""
    Z, yc, IDX = load_marg(TASKDUM.format(stage), stage)
    sx, ay = IDX['sample'], IDX['tasks']
    TASKS = {'DPA': (yc['tasks'] == 'DPA').to_numpy(),
             'Go': (yc['tasks'] == 'DualGo').to_numpy(),
             'NoGo': (yc['tasks'] == 'DualNoGo').to_numpy()}
    for tname, tmask in TASKS.items():
        for s in (0, 1):                                   # sample A=0, B=1
            m = tmask & (yc['sample'] == s).to_numpy()
            if m.sum() < 2:
                continue
            traj = Z[m][:, [sx, ay], :][:, :, win].mean(0)  # (2, len(win))
            ax.plot(traj[0], traj[1], TASK_LS[tname], color=SAMPLE_COL[s], lw=1.4, alpha=0.9, zorder=2)
            ax.scatter(traj[0, -1], traj[1, -1], s=42, color=SAMPLE_COL[s], edgecolor='k',
                       linewidths=0.6, zorder=4)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0); ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlabel('sample (memory) axis\n← odor A          odor B →', fontsize=7.5)
    ax.set_ylabel('action axis (tasks)\n← no-lick        lick →', fontsize=7.5)
    ax.set_title('Reused on the low-D dPCA manifold', loc='left', fontsize=TITLE_FS)
    # legends: sample colour + task linestyle
    from matplotlib.lines import Line2D
    h1 = [Line2D([0], [0], color=SAMPLE_COL[0], lw=2, label='odor A'),
          Line2D([0], [0], color=SAMPLE_COL[1], lw=2, label='odor B')]
    h2 = [Line2D([0], [0], color='0.4', ls=TASK_LS[t], lw=1.4, label=t) for t in TASK_LS]
    leg1 = ax.legend(handles=h1, frameon=False, fontsize=6.5, loc='upper left', handlelength=1.3)
    ax.add_artist(leg1)
    ax.legend(handles=h2, frameon=False, fontsize=6.5, loc='lower right', handlelength=1.8)


fig, ax = plt.subplots(figsize=(3.6, 3.4))
link_panel(ax, 'Expert')
OUT = 'figures/overlaps/manifold'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
fig.savefig(f'{OUT}/png/link_panel.png', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/link_panel.png'))
