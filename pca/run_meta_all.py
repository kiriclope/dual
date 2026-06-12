"""
Meta-mouse cv PCA: run analysis + summary figures.

Results → pca/results/meta_*.pkl
Figures → pca/figures/summary/meta_*.svg

Figures
-------
meta_evr.svg              — EVR mean ± SEM across folds
meta_traj_odor_pair.svg   — PC1-3 traces by odor pair
meta_traj_tasks.svg       — PC1-3 traces by task
meta_traj_sample.svg      — PC1-3 traces by sample odor
meta_traj_choice.svg      — PC1-3 traces by choice
meta_traj_test.svg        — PC1-3 traces by test odor
meta_loadings_theta.svg   — weights vs preferred direction
meta_loadings_2d.svg      — PC weight planes
meta_mouse_energy.svg     — per-PC loading energy per mouse
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, '/home/leon/dual_task/dual_data/')
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from time import perf_counter
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.ndimage import gaussian_filter1d

from src.common.options import set_options
from src.common.plot_utils import add_vlines
from src.pca.io import pkl_save, pkl_load
from src.pca.meta import cv_pca_meta, pc_mouse_energy

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)
GOLDEN = (5 ** 0.5 - 1) / 2

RESULTS = 'results'
FIGURES = 'figures/meta/summary'
DATA    = '/home/leon/dual_task/dual_data/data/pca'

# ── parameters ────────────────────────────────────────────────────────────────

mice = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
        'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']

kwargs = {
    'mice': mice, 'tasks': ['Dual'],
    'mouse': mice[0], 'laser': 0,
    'trials': '', 'reload': 0, 'data_type': 'dF',
    'prescreen': None, 'pval': 0.05,
    'preprocess': None, 'scaler_BL': 'center_BL',
    'avg_noise': False, 'unit_var_BL': False,
    'random_state': None, 'T_WINDOW': 0.0,
    'l1_ratio': 0.95, 'n_comp': 3, 'pca': 'pca', 'scaler': None,
    'bootstrap': 1, 'n_boots': 128,
    'n_splits': 5, 'n_repeats': 10,
    'class_weight': 0, 'multilabel': 0,
    'mne_estimator': 'generalizing', 'n_jobs': 64,
}
kwargs['days'] = ['first', 'last']
options = set_options(**kwargs)

epoch   = options['bins_TEST']
stage   = 'Expert'
factors = ['odor_pair']
n_comp  = 10
n_splits, n_repeats = 5, 10
DUM     = f'meta_TEST_{stage}_center_{n_splits}x{n_repeats}'

folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

# ── load data ─────────────────────────────────────────────────────────────────

print('Loading data ...')
X_all        = pkl_load('X_all_center',  path=DATA)
y_all        = pkl_load('y_all_center',  path=DATA)
mouse_slices = pkl_load('mouse_slices',  path=DATA)

y_all['sample'] = y_all.sample_odor
y_all['test']   = y_all.test_odor
print(X_all.shape, y_all.shape)

# ── run cv_pca_meta ───────────────────────────────────────────────────────────

pca_est = PCA(n_components=n_comp, svd_solver='randomized')

t0 = perf_counter()
Z_all, y_meta, W_mean, W_ref, evr_folds = cv_pca_meta(
    X_all, y_all, pca_est, folds, factors,
    epoch=epoch,
    learning=stage,
    mouse_slices=mouse_slices,
    mouse_gain_mode=None,     # equal-neuron contribution
    scale=0,                  # X_all_center already centered
    if_scale=0,
    scale_test=0,
)
print(f'cv_pca_meta done in {(perf_counter()-t0)/60:.1f} min')

# swap to (trials, n_comp, time)
X_meta = np.swapaxes(Z_all, 1, 2).astype(float)
print('X_meta:', X_meta.shape, '  y_meta:', y_meta.shape)
print('W_ref:', W_ref.shape, '  evr_folds:', evr_folds.shape)

# ── scale weights to percent ──────────────────────────────────────────────────

w_meta   = W_ref  * 100          # reference loadings
w_mean   = W_mean * 100

# ── fix PC2 sign (lick-positive in CHOICE epoch) ──────────────────────────────

bins_choice = options['bins_CHOICE']
m_lick   = (y_meta.laser == 0) & (y_meta.learning == stage) & (y_meta.performance == 1) & (y_meta.choice == 1)
m_nolick = (y_meta.laser == 0) & (y_meta.learning == stage) & (y_meta.performance == 1) & (y_meta.choice == 0)

if np.nanmean(X_meta[m_lick][:, 1, bins_choice]) < np.nanmean(X_meta[m_nolick][:, 1, bins_choice]):
    print('Flipping PC2')
    X_meta[:, 1, :]  *= -1
    w_meta[1, :]     *= -1
    w_mean[1, :]     *= -1
else:
    print('PC2 orientation OK')

# ── save results ──────────────────────────────────────────────────────────────

pkl_save(X_meta,    f'meta_traj_{DUM}',    path=RESULTS)
pkl_save(y_meta,    f'meta_labels_{DUM}',  path=RESULTS)
pkl_save(w_meta,    f'meta_weights_{DUM}', path=RESULTS)
pkl_save(evr_folds, f'meta_evr_{DUM}',     path=RESULTS)
print('Results saved to', RESULTS)

# ══ FIGURES ══════════════════════════════════════════════════════════════════

xtime   = np.linspace(0, 14, 84)
n_show  = 3
W       = 3.5
H       = W * GOLDEN   # ≈ 2.16

pal = sns.color_palette('muted')

try:
    import cmocean
    cmap = cmocean.cm.phase
except ImportError:
    cmap = plt.cm.hsv


def _save(name):
    path = os.path.join(FIGURES, f'meta_{name}.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('saved', path)


BL = slice(0, 12)   # pre-stimulus baseline: t = 0–2 s (bins 0-11)


def _traj_fig(X, y, base_mask, factor, levels_vals, labels, colors,
              bl_correct=True):
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, 2 * H))
    for k, ax in enumerate(axes):
        add_vlines(ax, if_dpa=0)
        ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'PC {k+1}', fontsize=10)
        ax.set_xlim([0, 14])
        ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
        ax.tick_params(labelsize=8)
    for lv, lab, col in zip(levels_vals, labels, colors):
        mask = base_mask & (y[factor] == lv)
        X_sel = X[mask]
        if X_sel.shape[0] == 0:
            continue
        if bl_correct:
            X_sel = X_sel - X_sel[:, :, BL].mean(axis=2, keepdims=True)
        mu  = X_sel.mean(0)
        sem = X_sel.std(0) / np.sqrt(X_sel.shape[0])
        for k, ax in enumerate(axes):
            ax.plot(xtime, mu[k], color=col, label=lab, lw=1.5, zorder=2)
            ax.fill_between(xtime, mu[k]-sem[k], mu[k]+sem[k],
                            color=col, alpha=0.2, zorder=2)
    axes[0].legend(fontsize=8, frameon=False, loc='best')
    fig.tight_layout()
    return fig


base_mask = (
    (y_meta.laser == 0) &
    (y_meta.learning == stage) &
    (y_meta.performance == 1)
)

# ── 1. EVR ────────────────────────────────────────────────────────────────────

evr_arr  = np.array(evr_folds)        # (n_folds, n_comp)
evr_mean = evr_arr.mean(0)
evr_sem  = evr_arr.std(0, ddof=1) / np.sqrt(evr_arr.shape[0])
n_pcs    = np.arange(1, evr_arr.shape[1] + 1)

fig, ax = plt.subplots(figsize=(W, W))
ax.plot(n_pcs, evr_mean, '-o', ms=5)
ax.fill_between(n_pcs, evr_mean - evr_sem, evr_mean + evr_sem, alpha=0.25)
ax.set_xlabel('PC #', fontsize=11)
ax.set_ylabel('Explained variance ratio', fontsize=11)
ax.set_xticks(n_pcs)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
fig.tight_layout()
_save('evr')

# ── 2-6. Trajectories ─────────────────────────────────────────────────────────

color_pair = ['#332288', '#88CCEE', '#117733', '#44AA99']
fig = _traj_fig(X_meta, y_meta, base_mask,
                'odor_pair', [0, 1, 2, 3], ['AC', 'AD', 'BD', 'BC'], color_pair)
_save('traj_odor_pair')

task_mask   = base_mask & ((y_meta.tasks == 'DPA') | (y_meta.odr_perf == 1))
task_colors = [pal[3], pal[0], pal[2]]
fig = _traj_fig(X_meta, y_meta, task_mask,
                'tasks', ['DPA', 'DualGo', 'DualNoGo'],
                ['DPA', 'DualGo', 'DualNoGo'], task_colors)
_save('traj_tasks')

fig = _traj_fig(X_meta, y_meta, base_mask,
                'sample_odor', [0, 1], ['Odor A', 'Odor B'],
                ['#332288', '#44AA99'])
_save('traj_sample')

fig = _traj_fig(X_meta, y_meta, base_mask,
                'choice', [0, 1], ['No lick', 'Lick'],
                ['#377eb8', '#4daf4a'])
_save('traj_choice')

fig = _traj_fig(X_meta, y_meta, base_mask,
                'test_odor', [0, 1], ['Odor C', 'Odor D'],
                ['#377eb8', '#4daf4a'])
_save('traj_test')

# ── 7. Loadings: theta scatter ────────────────────────────────────────────────

theta      = np.arctan2(w_meta[1], w_meta[0]) * 180 / np.pi
idx_sorted = np.argsort(theta)
theta_norm = (theta + 360) % 360
smooth_w   = max(1, int(0.05 * w_meta.shape[1]))
z_lim      = 5

fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, 2 * H))
for k, ax in enumerate(axes):
    sc = ax.scatter(theta[idx_sorted], w_meta[k, idx_sorted],
                    c=theta_norm[idx_sorted], cmap=cmap,
                    alpha=0.4, s=3, rasterized=True)
    ax.plot(theta[idx_sorted],
            gaussian_filter1d(w_meta[k, idx_sorted], smooth_w, mode='wrap'),
            'k', lw=1.5)
    ax.axhline(0, ls='--', color='k', lw=0.6)
    ax.set_ylabel(f'Weights PC {k+1}', fontsize=10)
    ax.set_xlabel('Neuron loc (°)', fontsize=10)
    ax.set_ylim([-z_lim, z_lim])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=8)
plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
fig.suptitle('Meta-PC loadings vs preferred direction', fontsize=11, y=1.01)
fig.tight_layout()
_save('loadings_theta')

# ── 8. Loadings: 2-D planes ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(3 * W, 2 * H))
for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
    sc = ax.scatter(w_meta[a, idx_sorted], w_meta[b, idx_sorted],
                    c=theta_norm[idx_sorted], cmap=cmap,
                    alpha=0.4, s=3, rasterized=True)
    ax.plot(
        gaussian_filter1d(w_meta[a, idx_sorted], smooth_w, mode='wrap'),
        gaussian_filter1d(w_meta[b, idx_sorted], smooth_w, mode='wrap'),
        'k', lw=1.5,
    )
    ax.set_xlabel(f'PC {a+1} weight', fontsize=10)
    ax.set_ylabel(f'PC {b+1} weight', fontsize=10)
    ax.set_xlim(-z_lim, z_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.tick_params(labelsize=8)
plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
fig.suptitle('Meta-PC loadings in weight space', fontsize=11, y=1.01)
fig.tight_layout()
_save('loadings_2d')

# ── 9. Mouse energy per PC ────────────────────────────────────────────────────

energy_df = pc_mouse_energy(W_ref, mouse_slices)   # (n_comp, n_mice)
# normalise to fraction per PC
energy_frac = energy_df.div(energy_df.sum(axis=1), axis=0)
energy_frac.index = [f'PC{i+1}' for i in range(len(energy_frac))]

mouse_colors = sns.color_palette('tab10', n_colors=len(mice))

fig, axes = plt.subplots(1, 2, figsize=(2 * W * 1.5, 2 * H))

# absolute energy
ax = axes[0]
bottom = np.zeros(n_comp)
for i, (mouse, col) in enumerate(zip(mice, mouse_colors)):
    vals = energy_df[mouse].values
    ax.bar(range(1, n_comp + 1), vals, bottom=bottom,
           color=col, label=mouse, width=0.7)
    bottom += vals
ax.set_xlabel('PC #', fontsize=10)
ax.set_ylabel('Loading energy (a.u.)', fontsize=10)
ax.set_xticks(range(1, n_comp + 1))
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

# fractional energy
ax = axes[1]
bottom = np.zeros(n_comp)
for mouse, col in zip(mice, mouse_colors):
    vals = energy_frac[mouse].values
    ax.bar(range(1, n_comp + 1), vals, bottom=bottom,
           color=col, label=mouse, width=0.7)
    bottom += vals
ax.set_xlabel('PC #', fontsize=10)
ax.set_ylabel('Fraction of loading energy', fontsize=10)
ax.set_xticks(range(1, n_comp + 1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.tick_params(labelsize=8)

fig.suptitle('Meta-PC loading energy per mouse', fontsize=11, y=1.01)
fig.tight_layout()
_save('mouse_energy')

print('\nAll done.')
