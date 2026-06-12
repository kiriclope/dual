"""
Run single-mouse cv PCA for all mice, save results and figures.

Results → pca/results/single_*.pkl
Figures → pca/figures/single_*.svg

Figures produced
----------------
evr.svg              — explained variance ratio, mean ± SEM across mice
traj_odor_pair.svg   — PC1-3 time traces by odor pair (AC/AD/BD/BC)
traj_tasks.svg       — PC1-3 time traces by task (DPA / DualGo / DualNoGo)
traj_sample.svg      — PC1-3 time traces by sample odor (A / B)
traj_choice.svg      — PC1-3 time traces by choice (No Lick / Lick)
traj_test.svg        — PC1-3 time traces by test odor (C / D)
loadings_theta.svg   — per-PC weight vs preferred direction scatter + smooth
loadings_2d.svg      — PC-weight 2D planes coloured by preferred direction
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
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
from sklearn.model_selection import LeaveOneOut

from src.common.options import set_options
from src.common.plot_utils import add_vlines
from src.pca.io import pkl_save, pkl_load
from src.pca.single import cv_pca_single, align_mice
from src.plot.traj import plot_mean_sem

sns.set_style('ticks')
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

RESULTS = '../data/pca'
FIGURES = 'figures/single/summary'

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
scale   = 'standard'
factors = ['odor_pair']
n_comp  = 10
DUM     = 'pca_TEST_Expert_standard_loo_correct_odor_pair'

folds = LeaveOneOut()

# ── load raw data ─────────────────────────────────────────────────────────────

print('Loading X_all_nan ...')
X_all = pkl_load('X_all_nan_', path=RESULTS)
y_all = pkl_load('y_all_nan_', path=RESULTS)
y_all['sample']     = y_all.sample_odor
y_all['test']       = y_all.test_odor
y_all['distractor'] = y_all.dist_odor
print(X_all.shape, y_all.shape)

# ── run cv_pca_single per mouse ───────────────────────────────────────────────

t0 = perf_counter()
X_mice, y_mice, w_mice, evr_mice = [], [], [], []

pca_est = PCA(n_components=n_comp, svd_solver='randomized')

for mouse in mice:
    idx = y_all['mouse'] == mouse
    X_m = X_all[idx]
    y_m = y_all.loc[idx].reset_index(drop=True)

    valid = ~np.all(np.isnan(X_m), axis=(0, 2))
    X_m = X_m[:, valid, :]

    Z, y_z, w, evr = cv_pca_single(
        X_m, y_m, pca_est, folds, factors,
        epoch=epoch, scale=scale, stage=stage, correct=True,
    )
    X_mice.append(Z)
    y_mice.append(y_z)
    w_mice.append(w)
    evr_mice.append(evr)

elapsed = perf_counter() - t0
print(f'cv_pca_single done in {elapsed/60:.1f} min')

# ── align across mice (Procrustes on score-space anchors) ─────────────────────

X_aligned, y_aligned = align_mice(X_mice, y_mice, factors, ref_idx=7)  # ACCM03

X_single = np.swapaxes(X_aligned, 1, 2)   # (trials, n_comp, time)
y_single = y_aligned
print('X_single:', X_single.shape)

# ── fix PC2 sign (lick-positive) ─────────────────────────────────────────────

bins_choice = options['bins_CHOICE']
m_lick   = (y_single.laser == 0) & (y_single.performance == 1) & (y_single.choice == 1)
m_nolick = (y_single.laser == 0) & (y_single.performance == 1) & (y_single.choice == 0)
if np.nanmean(X_single[m_lick][:, 1, bins_choice]) < np.nanmean(X_single[m_nolick][:, 1, bins_choice]):
    print('Flipping PC2')
    X_single[:, 1, :] *= -1
else:
    print('PC2 orientation OK')

# ── collapse fold weights and EVR ─────────────────────────────────────────────

evr_single = np.vstack([evr.mean(0) for evr in evr_mice])          # (n_mice, n_comp)
w_single   = np.hstack([w.mean(0) for w in w_mice]) * 100          # (n_comp, n_neurons)
print('w_single:', w_single.shape, '  evr_single:', evr_single.shape)

# ── save results ──────────────────────────────────────────────────────────────

pkl_save(X_single,   f'single_traj_{DUM}',    path=RESULTS)
pkl_save(y_single,   f'single_labels_{DUM}',  path=RESULTS)
pkl_save(w_single,   f'single_weights_{DUM}', path=RESULTS)
pkl_save(evr_single, f'single_evr_{DUM}',     path=RESULTS)
print('Results saved to', RESULTS)

# ══ FIGURES ══════════════════════════════════════════════════════════════════

xtime   = np.linspace(0, 14, 84)
n_show  = 3
W       = 3.5   # panel width  (inches)
H       = 2.8   # panel height (inches)


def _save(name):
    path = os.path.join(FIGURES, f'single_{name}.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('saved', path)


BL = slice(0, 12)   # pre-stimulus baseline: t = 0–2 s (bins 0-11)


def _traj_fig(X, y, base_mask, levels_vals, labels, colors, factor_col,
              title=None, sharey=False, bl_correct=True):
    """Draw 3-panel time-trace figure and save."""
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, H), sharey=sharey)
    for k, ax in enumerate(axes):
        add_vlines(ax, if_dpa=0)
        ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'PC {k+1}', fontsize=10)
        ax.set_xlim([0, 14])
        ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
        ax.tick_params(labelsize=8)
    for lv, lab, col in zip(levels_vals, labels, colors):
        mask = base_mask & (y[factor_col] == lv)
        X_sel = X[mask]
        if X_sel.shape[0] == 0:
            continue
        if bl_correct:
            X_sel = X_sel - X_sel[:, :, BL].mean(axis=2, keepdims=True)
        mu  = X_sel.mean(0)                           # (n_comp, n_time)
        sem = X_sel.std(0) / np.sqrt(X_sel.shape[0])
        for k, ax in enumerate(axes):
            plot_mean_sem(ax, xtime, mu[k], sem[k], col, lw=1.5, label=lab, zorder=2)
    axes[0].legend(fontsize=8, frameon=False, loc='best')
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ── base mask: expert, laser-off, correct ─────────────────────────────────────

base_mask = (
    (y_single.laser == 0) &
    (y_single.learning == 'Expert') &
    (y_single.performance == 1)
)

# ── 1. Explained variance ratio ───────────────────────────────────────────────

evr_arr  = np.array(evr_single, dtype=float)       # (n_mice, n_comp)
evr_mean = evr_arr.mean(0)
evr_sem  = evr_arr.std(0, ddof=1) / np.sqrt(evr_arr.shape[0])
n_pcs    = np.arange(1, evr_arr.shape[1] + 1)

fig, ax = plt.subplots(figsize=(W, H))
ax.plot(n_pcs, evr_mean, '-o', ms=5)
ax.fill_between(n_pcs, evr_mean - evr_sem, evr_mean + evr_sem, alpha=0.25)
for i, (m, s) in enumerate(zip(evr_mean, evr_sem)):
    ax.errorbar(n_pcs[i], m, yerr=s, color='C0', capsize=3, lw=1)
ax.set_xlabel('PC #', fontsize=11)
ax.set_ylabel('Explained variance ratio', fontsize=11)
ax.set_xticks(n_pcs)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
fig.tight_layout()
_save('evr')

# ── 2. Trajectories by odor pair ─────────────────────────────────────────────

color_pair = ['#332288', '#88CCEE', '#117733', '#44AA99']
pair_labels = ['AC', 'AD', 'BD', 'BC']
# odor_pair column has values 0,1,2,3
fig = _traj_fig(X_single, y_single, base_mask,
                levels_vals=[0, 1, 2, 3],
                labels=pair_labels,
                colors=color_pair,
                factor_col='odor_pair',
                title='Trajectories by odor pair')
_save('traj_odor_pair')

# ── 3. Trajectories by task ───────────────────────────────────────────────────

palette = sns.color_palette('muted')
task_colors = [palette[3], palette[0], palette[2]]
task_list   = ['DPA', 'DualGo', 'DualNoGo']

# for tasks: correct means (performance==1 for DPA) or (odr_perf==1 for GNG)
task_mask = base_mask & ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
fig = _traj_fig(X_single, y_single, task_mask,
                levels_vals=task_list,
                labels=task_list,
                colors=task_colors,
                factor_col='tasks',
                title='Trajectories by task')
_save('traj_tasks')

# ── 4. Trajectories by sample odor ───────────────────────────────────────────

color_sample = ['#332288', '#44AA99']
fig = _traj_fig(X_single, y_single, base_mask,
                levels_vals=[0, 1],
                labels=['Odor A', 'Odor B'],
                colors=color_sample,
                factor_col='sample_odor',
                title='Trajectories by sample odor')
_save('traj_sample')

# ── 5. Trajectories by choice (lick) ─────────────────────────────────────────

color_choice = ['#377eb8', '#4daf4a']
fig = _traj_fig(X_single, y_single, base_mask,
                levels_vals=[0, 1],
                labels=['No lick', 'Lick'],
                colors=color_choice,
                factor_col='choice',
                title='Trajectories by choice')
_save('traj_choice')

# ── 6. Trajectories by test odor ─────────────────────────────────────────────

color_test = ['#377eb8', '#4daf4a']
fig = _traj_fig(X_single, y_single, base_mask,
                levels_vals=[0, 1],
                labels=['Odor C', 'Odor D'],
                colors=color_test,
                factor_col='test_odor',
                title='Trajectories by test odor')
_save('traj_test')

# ── 7. Loadings: weights vs preferred direction (theta) ──────────────────────

try:
    import cmocean
    cmap = cmocean.cm.phase
except ImportError:
    cmap = plt.cm.hsv

# theta = angle of each neuron's (PC1-weight, PC2-weight) vector
theta      = np.arctan2(w_single[1], w_single[0]) * 180 / np.pi
idx_sorted = np.argsort(theta)
theta_norm = (theta + 360) % 360

smooth_width = max(1, int(0.05 * w_single.shape[1]))  # 5% of neurons

fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, H))
z_lim = 5
for k, ax in enumerate(axes):
    sc = ax.scatter(theta[idx_sorted], w_single[k, idx_sorted],
                    c=theta_norm[idx_sorted], cmap=cmap,
                    alpha=0.4, s=3, rasterized=True)
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(w_single[k, idx_sorted], smooth_width, mode='wrap')
    ax.plot(theta[idx_sorted], smoothed, 'k', lw=1.5)
    ax.axhline(0, ls='--', color='k', lw=0.6)
    ax.set_ylabel(f'Weights PC {k+1}', fontsize=10)
    ax.set_xlabel('Neuron loc (°)', fontsize=10)
    ax.set_ylim([-z_lim, z_lim])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=8)

plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
fig.suptitle('PC loadings vs preferred direction', fontsize=11, y=1.01)
fig.tight_layout()
_save('loadings_theta')

# ── 8. Loadings: 2-D weight planes ───────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(3 * W, H))
pairs_2d = [(0, 1), (0, 2), (1, 2)]

for ax, (a, b) in zip(axes, pairs_2d):
    sc = ax.scatter(w_single[a, idx_sorted], w_single[b, idx_sorted],
                    c=theta_norm[idx_sorted], cmap=cmap,
                    alpha=0.4, s=3, rasterized=True)
    from scipy.ndimage import gaussian_filter1d
    ax.plot(
        gaussian_filter1d(w_single[a, idx_sorted], smooth_width, mode='wrap'),
        gaussian_filter1d(w_single[b, idx_sorted], smooth_width, mode='wrap'),
        'k', lw=1.5,
    )
    ax.set_xlabel(f'PC {a+1} weight', fontsize=10)
    ax.set_ylabel(f'PC {b+1} weight', fontsize=10)
    ax.set_xlim(-z_lim, z_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.tick_params(labelsize=8)

plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
fig.suptitle('PC loadings in weight space', fontsize=11, y=1.01)
fig.tight_layout()
_save('loadings_2d')

print('\nAll done.')
