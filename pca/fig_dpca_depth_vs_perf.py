"""Correlate the per-mouse no-lick-push DEPTH with individual behavioural performance.
  (levels)  Expert |tasks-axis depth|           vs Expert accuracy in DPA / Go / NoGo
  (changes) Naive->Expert depth deepening Δ|d|  vs Naive->Expert accuracy change in DPA / Go / NoGo

Depth = headline quantity (per-mouse Expert dPCA basis, tasks axis, late-delay DPA condition mean, sample
A/B avg, baseline-ref, std 2.8), oriented Expert-DPA-delay<0 (applied to BOTH stages), sign-free |depth|.
DPA perf = mean(performance) on DPA trials; Go/NoGo perf = mean(odr_perf) on DualGo / DualNoGo trials.
Laser off, ALL trials. n=9 mice.

Usage: /home/leon/mambaforge/envs/dual/bin/python fig_dpca_depth_vs_perf.py
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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
BASE = 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
LATE = np.arange(39, 54)


def stage_depth(Zr, y, stg):
    m = ((y.laser == 0) & (y.learning == stg) & (y.tasks == 'DPA')).to_numpy()
    yc = y[m].reset_index(drop=True); Zs = Zr[m]
    means = [Zs[(yc['sample'] == s).to_numpy()].mean(0) for s in (0, 1)]
    return abs(float(np.mean([means[s][1, LATE].mean() for s in (0, 1)])))


def stage_perf(y, stg):
    e = y[(y.laser == 0) & (y.learning == stg)]
    return (float(e[e.tasks == 'DPA'].performance.mean()),
            float(e[e.tasks == 'DualGo'].odr_perf.mean()),
            float(e[e.tasks == 'DualNoGo'].odr_perf.mean()))


def mouse_row(mm):
    Z = pkl_load(f'pseudo_traj_{BASE}_{mm}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{BASE}_{mm}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{BASE}_{mm}', path='../data/pca')
    Zr = Z[:, [lab.index('sample'), lab.index('tasks')], :].astype(float)
    ref = ((y.laser == 0) & (y.tasks == 'DPA')).to_numpy()
    mu0 = Zr[ref][:, :, 0:12].mean((0, 2), keepdims=True); sd0 = Zr[ref].std((0, 2), keepdims=True)
    Zr = (Zr - mu0) / sd0 * 2.8
    em = ((y.laser == 0) & (y.learning == 'Expert') & (y.tasks == 'DPA') & (y.performance == 1)).to_numpy()
    if Zr[em][:, 1, LATE].mean() > 0:
        Zr[:, 1, :] *= -1
    dN, dE = stage_depth(Zr, y, 'Naive'), stage_depth(Zr, y, 'Expert')
    pN, pE = stage_perf(y, 'Naive'), stage_perf(y, 'Expert')
    return dN, dE, pN, pE


R = [mouse_row(mm) for mm in MICE]
dN = np.array([r[0] for r in R]); dE = np.array([r[1] for r in R])
pN = np.array([r[2] for r in R]); pE = np.array([r[3] for r in R])     # (9,3): DPA,Go,NoGo
ddep = dE - dN                                                          # depth deepening (Δ|depth|)
dperf = pE - pN                                                         # (9,3) perf change
TASKS = ['DPA', 'Go', 'NoGo']

print('per-mouse |depth| Naive->Expert  and  accuracy Naive->Expert (DPA / Go / NoGo):')
for i, mm in enumerate(MICE):
    print(f'  {mm:8s} d {dN[i]:.2f}->{dE[i]:.2f} (Δ{ddep[i]:+.2f}) | '
          f'DPA {pN[i,0]:.2f}->{pE[i,0]:.2f}  Go {pN[i,1]:.2f}->{pE[i,1]:.2f}  NoGo {pN[i,2]:.2f}->{pE[i,2]:.2f}')

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
for j, tk in enumerate(TASKS):
    # row 0: Expert level   |depthE| vs perfE ;  row 1: change  Δ|depth| vs Δperf
    for row, (x, yv, xl, yl, ttl) in enumerate([
        (dE, pE[:, j], '|tasks depth| Expert', f'{tk} accuracy (Expert)', f'LEVEL · depth vs {tk}'),
        (ddep, dperf[:, j], 'Δ|depth| (deepening)', f'Δ {tk} accuracy', f'CHANGE · Δdepth vs Δ{tk}')]):
        ax = axes[row, j]; rP, pP = pearsonr(x, yv); rS, pS = spearmanr(x, yv)
        ax.scatter(x, yv, s=42, color='#332288' if row == 0 else '#44AA99', zorder=3)
        for mm, xx, yy in zip(MICE, x, yv): ax.annotate(mm, (xx, yy), fontsize=6, xytext=(3, 2), textcoords='offset points')
        b, a0 = np.polyfit(x, yv, 1); xs = np.linspace(x.min(), x.max(), 2); ax.plot(xs, a0 + b * xs, '0.6', lw=1.1, zorder=2)
        if row == 1: ax.axhline(0, color='0.8', lw=0.7); ax.axvline(0, color='0.8', lw=0.7)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'{ttl}\nr={rP:+.2f} p={pP:.3f} | ρ={rS:+.2f} p={pS:.3f}', loc='left', fontsize=TITLE_FS)
        ax.spines[['top', 'right']].set_visible(False)
        print(f'{ttl:26s}  Pearson r={rP:+.2f} p={pP:.3f}   Spearman rho={rS:+.2f} p={pS:.3f}')
fig.suptitle('No-lick push depth vs performance — levels (top) & learning changes (bottom), n=9', y=1.0, fontsize=TITLE_FS)
fig.tight_layout()
out = 'figures/pseudo/flow/lowrank/png/dpca_depth_vs_perf.png'
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
