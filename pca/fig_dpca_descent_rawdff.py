"""Belt-and-suspenders: confirm the DPA-delay 'descent into no-lick' is in RAW ΔF/F, not a z-score artifact.
The no-lick component(t) = W @ x(t). z-scoring only reweights neurons by 1/s (a per-neuron constant in
time), so it CANNOT change a trajectory's time course. We show this directly: project RAW (un-z-scored,
baseline-centred only) ΔF/F onto the dPCA no-lick decoder and overlay the z-scored version — identical
descent. no-lick axis = 0.53·tasks + 0.85·time (the combined direction). DPA condition mean, Naive vs
Expert. Saves PNG(300)+SVG."""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
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

DUM = 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
SRC = '/home/leon/dual_task/dual_data/data/pca/'
BL = np.arange(0, 12); LATE = np.arange(39, 54); T = np.arange(84)
WT = [0.53, 0.85]                                              # combined no-lick weights [tasks, time]

W = np.asarray(pkl_load(f'pseudo_weights_{DUM}', path='../data/pca'))     # (16, 3319) decoders (z-scored space)
lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca')
itask, itime = lab.index('tasks'), lab.index('time')
Zz = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca')                   # z-scored component traj (cond,16,84)
yz = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')

# RAW (un-z-scored) padded ΔF/F + matching labels; baseline-centre per neuron (scalar) — NO std division
Xr = np.asarray(pickle.load(open(SRC + 'X_all_no_scale.pkl', 'rb')))
yr = pd.read_pickle(SRC + 'y_all_no_scale.pkl')
Xr = Xr - np.nanmean(Xr[:, :, BL], axis=(0, 2), keepdims=True)


def comp_norm(c):                                             # baseline-centre + unit-std over the trace
    c = c - c[BL].mean(); return c / (c.std() + 1e-12)


def combined(tasks_t, time_t, orient):                       # 0.53·tasks + 0.85·time, oriented no-lick<0
    v = WT[0] * comp_norm(tasks_t) + WT[1] * comp_norm(time_t)
    return v * orient


def raw_proj(stg):                                           # RAW dF projected onto the dPCA decoders
    m = ((yr.laser == 0) & (yr.learning == stg) & (yr.tasks == 'DPA') & (yr.performance == 1)).to_numpy()
    cm = np.nanmean(Xr[m], axis=0)                           # (3319, 84) raw condition mean
    return W[itask] @ cm, W[itime] @ cm


def z_proj(stg):                                            # z-scored component traj (the normal pipeline)
    m = ((yz.laser == 0) & (yz.learning == stg) & (yz.tasks == 'DPA') & (yz.performance == 1)).to_numpy()
    cm = Zz[m].mean(0)                                       # (16, 84)
    return cm[itask], cm[itime]


# orient each stream so the late-delay DPA value is NEGATIVE (no-lick down)
def orient_of(t, ti):
    v = WT[0] * comp_norm(t) + WT[1] * comp_norm(ti)
    return -1.0 if v[LATE].mean() > 0 else 1.0


fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
print('DPA no-lick projection (late delay), baseline=0:')
for ax, stg in zip(axes, ['Naive', 'Expert']):
    rt, rti = raw_proj(stg); zt, zti = z_proj(stg)
    o_r, o_z = orient_of(rt, rti), orient_of(zt, zti)
    raw = combined(rt, rti, o_r); zc = combined(zt, zti, o_z)
    r = np.corrcoef(raw, zc)[0, 1]
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.axvspan(21, 54, color='0.92', zorder=0)               # delay
    ax.plot(T, raw, '-', color='#d1495b', lw=1.6, label='RAW ΔF/F (no z-score)')
    ax.plot(T, zc, '--', color='#30638e', lw=1.4, label='z-scored (pipeline)')
    ax.set_title(f'{stg}   (raw vs z-scored r = {r:.3f})', loc='left', fontsize=TITLE_FS)
    ax.set_xlabel('time bin'); ax.set_xlim(0, 83)
    print(f'  {stg:7s} RAW {raw[LATE].mean():+.2f}   z-scored {zc[LATE].mean():+.2f}   shape corr r={r:.3f}')
axes[0].set_ylabel('no-lick projection (norm., − = no-lick)')
axes[0].legend(frameon=False, loc='lower left')
fig.suptitle('The DPA-delay descent into no-lick is present in RAW ΔF/F (z-scoring only reweights neurons)', y=1.0, fontsize=TITLE_FS)
fig.tight_layout()
out = 'figures/pseudo/flow/lowrank/png/dpca_descent_rawdff.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
