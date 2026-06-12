"""
Per-mouse figures for metaPCA → pca/figures/meta/individual/{mouse}_{panel}.svg

For each mouse
--------------
  {mouse}_traj_odor_pair.svg   — PC1-3 traces by odor pair
  {mouse}_traj_tasks.svg       — PC1-3 traces by task
  {mouse}_traj_sample.svg      — PC1-3 traces by sample odor
  {mouse}_traj_choice.svg      — PC1-3 traces by choice
  {mouse}_traj_test.svg        — PC1-3 traces by test odor
  {mouse}_loadings_theta.svg   — weights vs preferred direction (this mouse's neurons)
  {mouse}_loadings_2d.svg      — PC weight planes
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from src.common.plot_utils import add_vlines
from src.pca.io import pkl_load
from src.plot.traj import plot_mean_sem

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)
GOLDEN = (5 ** 0.5 - 1) / 2

RESULTS = 'results'
OUT     = 'figures/meta/individual'
DATA    = '/home/leon/dual_task/dual_data/data/pca'
DUM     = 'meta_TEST_Expert_center_5x10'

mice = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
        'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']

# ── load results ──────────────────────────────────────────────────────────────

X_meta   = pkl_load(f'meta_traj_{DUM}',    path=RESULTS)   # (trials, n_comp, time)
y_meta   = pkl_load(f'meta_labels_{DUM}',  path=RESULTS)
w_meta   = pkl_load(f'meta_weights_{DUM}', path=RESULTS)   # (n_comp, n_neurons)

mouse_slices = pkl_load('mouse_slices', path=DATA)

# ── shared settings ───────────────────────────────────────────────────────────

xtime  = np.linspace(0, 14, 84)
n_show = 3
W      = 3.5
H      = W * GOLDEN

pal = sns.color_palette('muted')

try:
    import cmocean
    cmap = cmocean.cm.phase
except ImportError:
    cmap = plt.cm.hsv

colors = {
    'odor_pair': ['#332288', '#88CCEE', '#117733', '#44AA99'],
    'tasks':     [pal[3], pal[0], pal[2]],
    'sample':    ['#332288', '#44AA99'],
    'choice':    ['#377eb8', '#4daf4a'],
    'test':      ['#377eb8', '#4daf4a'],
}


def savefig(mouse, tag):
    path = os.path.join(OUT, f'{mouse}_{tag}.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('  saved', path)


BL = slice(0, 12)   # pre-stimulus baseline: t = 0–2 s (bins 0-11)


def traj_panels(X_m, y_m, base_mask, factor, levels_vals, labels, cols,
                bl_correct=True):
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, 2 * H))
    for k, ax in enumerate(axes):
        add_vlines(ax, if_dpa=0)
        ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel(f'PC {k+1}', fontsize=9)
        ax.set_xlim([0, 14])
        ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
        ax.tick_params(labelsize=8)
    for lv, lab, col in zip(levels_vals, labels, cols):
        mask = base_mask & (y_m[factor] == lv)
        X_sel = X_m[mask]
        if X_sel.shape[0] == 0:
            continue
        if bl_correct:
            X_sel = X_sel - X_sel[:, :, BL].mean(axis=2, keepdims=True)
        mu  = X_sel.mean(0)
        sem = X_sel.std(0) / np.sqrt(X_sel.shape[0])
        for k, ax in enumerate(axes):
            plot_mean_sem(ax, xtime, mu[k], sem[k], col,
                          lw=1.5, label=lab, zorder=2)
    axes[0].legend(fontsize=8, frameon=False, loc='best')
    fig.tight_layout()
    return fig


# ── per-mouse loop ────────────────────────────────────────────────────────────

for mouse in mice:
    print(f'\n{mouse}')
    m_idx = y_meta.mouse == mouse
    X_m   = X_meta[m_idx]
    y_m   = y_meta[m_idx].reset_index(drop=True)

    base = (
        (y_m.laser == 0) &
        (y_m.learning == 'Expert') &
        (y_m.performance == 1)
    )

    # ── trajectories ──────────────────────────────────────────────────────────

    fig = traj_panels(X_m, y_m, base,
                      'odor_pair', [0, 1, 2, 3], ['AC', 'AD', 'BD', 'BC'],
                      colors['odor_pair'])
    fig.suptitle(f'{mouse} — odor pair', fontsize=10, y=1.01)
    savefig(mouse, 'traj_odor_pair')

    task_base = base & ((y_m.tasks == 'DPA') | (y_m.odr_perf == 1))
    fig = traj_panels(X_m, y_m, task_base,
                      'tasks', ['DPA', 'DualGo', 'DualNoGo'],
                      ['DPA', 'DualGo', 'DualNoGo'], colors['tasks'])
    fig.suptitle(f'{mouse} — task', fontsize=10, y=1.01)
    savefig(mouse, 'traj_tasks')

    fig = traj_panels(X_m, y_m, base,
                      'sample_odor', [0, 1], ['Odor A', 'Odor B'],
                      colors['sample'])
    fig.suptitle(f'{mouse} — sample odor', fontsize=10, y=1.01)
    savefig(mouse, 'traj_sample')

    fig = traj_panels(X_m, y_m, base,
                      'choice', [0, 1], ['No lick', 'Lick'],
                      colors['choice'])
    fig.suptitle(f'{mouse} — choice', fontsize=10, y=1.01)
    savefig(mouse, 'traj_choice')

    fig = traj_panels(X_m, y_m, base,
                      'test_odor', [0, 1], ['Odor C', 'Odor D'],
                      colors['test'])
    fig.suptitle(f'{mouse} — test odor', fontsize=10, y=1.01)
    savefig(mouse, 'traj_test')

    # ── loadings (this mouse's neurons only) ─────────────────────────────────

    sl   = mouse_slices[mouse]
    w_m  = w_meta[:, sl]                          # (n_comp, n_neurons_mouse)

    theta      = np.arctan2(w_m[1], w_m[0]) * 180 / np.pi
    idx_sorted = np.argsort(theta)
    theta_norm = (theta + 360) % 360
    smooth_w   = max(1, int(0.05 * w_m.shape[1]))
    z_lim      = np.percentile(np.abs(w_m[:n_show]), 99) * 1.1

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * W, 2 * H))
    for k, ax in enumerate(axes):
        sc = ax.scatter(theta[idx_sorted], w_m[k, idx_sorted],
                        c=theta_norm[idx_sorted], cmap=cmap,
                        alpha=0.4, s=4, rasterized=True)
        ax.plot(theta[idx_sorted],
                gaussian_filter1d(w_m[k, idx_sorted], smooth_w, mode='wrap'),
                'k', lw=1.5)
        ax.axhline(0, ls='--', color='k', lw=0.6)
        ax.set_ylabel(f'Weights PC {k+1}', fontsize=9)
        ax.set_xlabel('Neuron loc (°)', fontsize=9)
        ax.set_ylim([-z_lim, z_lim])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=8)
    plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
    fig.suptitle(f'{mouse} — loadings', fontsize=10, y=1.01)
    fig.tight_layout()
    savefig(mouse, 'loadings_theta')

    fig, axes = plt.subplots(1, 3, figsize=(3 * W, 2 * H))
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        sc = ax.scatter(w_m[a, idx_sorted], w_m[b, idx_sorted],
                        c=theta_norm[idx_sorted], cmap=cmap,
                        alpha=0.4, s=4, rasterized=True)
        ax.plot(
            gaussian_filter1d(w_m[a, idx_sorted], smooth_w, mode='wrap'),
            gaussian_filter1d(w_m[b, idx_sorted], smooth_w, mode='wrap'),
            'k', lw=1.5,
        )
        ax.set_xlabel(f'PC {a+1} weight', fontsize=9)
        ax.set_ylabel(f'PC {b+1} weight', fontsize=9)
        ax.set_xlim(-z_lim, z_lim)
        ax.set_ylim(-z_lim, z_lim)
        ax.tick_params(labelsize=8)
    plt.colorbar(sc, ax=axes[-1], label='Angle (°)', shrink=0.8)
    fig.suptitle(f'{mouse} — loading planes', fontsize=10, y=1.01)
    fig.tight_layout()
    savefig(mouse, 'loadings_2d')

print('\nAll done.')
