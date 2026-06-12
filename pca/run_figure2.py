import sys
sys.path.insert(0, '/home/leon/dual_task/dual_data/')
sys.path.insert(0, '/home/leon/dual/')

import os
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.plot_utils import add_vlines

# ── parameters ────────────────────────────────────────────────────────────────
dum       = 'pca_TEST_Expert_standard_loo_correct_odor_pair'
DATA_PATH = '/home/leon/dual/data/pca'
OUT_DIR   = '/home/leon/dual/pca/figures/figure2'
os.makedirs(OUT_DIR, exist_ok=True)

n_show    = 3
pc_labels = ['PC1\n(sample)', 'PC2\n(lick)', 'PC3\n(test)']
xtime     = np.linspace(0, 14, 84)
width, height = 6, 6 * 1.618

# ── load ──────────────────────────────────────────────────────────────────────
def pkl_load(name, path='.'):
    src = f'{path}/{name}.pkl'
    print('loading', src)
    return pkl.load(open(src, 'rb'))

X_single   = pkl_load(f'single_traj_{dum}',    DATA_PATH)
y_single   = pkl_load(f'single_labels_{dum}',  DATA_PATH)
w_single   = pkl_load(f'single_weights_{dum}', DATA_PATH)
evr_single = pkl_load(f'single_evr_{dum}',     DATA_PATH)

print('X_single :', X_single.shape)
print('w_single :', w_single.shape)
print('evr_single:', evr_single.shape)

mask_base = (
    (y_single.laser == 0) &
    (y_single.learning == 'Expert') &
    (y_single.performance == 1)
)

# ── panel B prep ──────────────────────────────────────────────────────────────
W        = w_single[:n_show].T
dominant = np.argmax(np.abs(W), axis=1)
sort_idx = np.argsort(dominant, kind='stable')
W_sorted = W[sort_idx]
W_z      = (W_sorted - W_sorted.mean(0)) / (W_sorted.std(0) + 1e-8)
counts   = [np.sum(dominant == k) for k in range(n_show)]

# ── panel B: heatmap ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(width * 0.9, height),
                         gridspec_kw={'width_ratios': [6, 1], 'wspace': 0.05})

ax = axes[0]
im = ax.imshow(W_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2,
               interpolation='nearest')
ax.set_xticks(range(n_show))
ax.set_xticklabels(pc_labels, fontsize=11)
ax.set_ylabel(f'Neurons (pooled, n={W.shape[0]})')
ax.set_yticks([])
plt.colorbar(im, ax=ax, label='z-scored loading', fraction=0.03, pad=0.02)

ax2 = axes[1]
ax2.barh(range(n_show), counts, color='0.4')
ax2.set_yticks(range(n_show))
ax2.set_yticklabels(pc_labels, fontsize=11)
ax2.set_xlabel('# neurons')
ax2.invert_yaxis()

plt.savefig(f'{OUT_DIR}/panel_B_heatmap_{dum}.svg')
plt.savefig(f'{OUT_DIR}/panel_B_heatmap_{dum}.png', dpi=150)
plt.close()
print('saved panel_B_heatmap')

# ── panel B: EVR inset ────────────────────────────────────────────────────────
evr_arr  = np.array(evr_single, dtype=float)
evr_mean = evr_arr.mean(0)
evr_sem  = evr_arr.std(0, ddof=1) / np.sqrt(evr_arr.shape[0])

fig, ax = plt.subplots(figsize=(width * 0.5, height * 0.4))
x = np.arange(1, n_show + 1)
ax.bar(x, evr_mean[:n_show] * 100, yerr=evr_sem[:n_show] * 100,
       color='0.4', capsize=4, width=0.6)
ax.set_xlabel('PC')
ax.set_ylabel('Variance\nexplained (%)')
ax.set_xticks(x)
ax.set_xticklabels([f'PC{i}' for i in x])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/panel_B_evr_{dum}.svg')
plt.savefig(f'{OUT_DIR}/panel_B_evr_{dum}.png', dpi=150)
plt.close()
print('saved panel_B_evr')

# ── panel C: trajectories ─────────────────────────────────────────────────────
conditions = [
    (0, 'sample_odor', ['Sample A', 'Sample B'], ['#332288', '#44AA99']),
    (1, 'choice',      ['No lick',  'Lick'],     ['#377eb8', '#4daf4a']),
    (2, 'test_odor',   ['Test C',   'Test D'],   ['#e41a1c', '#ff7f00']),
]
ylabels = ['PC1 projection (s.d.)', 'PC2 projection (s.d.)', 'PC3 projection (s.d.)']

fig, axes = plt.subplots(3, 1, figsize=(width, height * 1.4), sharex=True)

for row, ((pc_idx, col, labels, colors), ylabel) in enumerate(zip(conditions, ylabels)):
    ax = axes[row]
    for i in range(2):
        mask = mask_base & (y_single[col] == i)
        X_sel = X_single[mask][:, pc_idx, :]
        mu  = X_sel.mean(0)
        sem = X_sel.std(0) / np.sqrt(X_sel.shape[0])
        ax.plot(xtime, mu, color=colors[i], label=labels[i], lw=1.5)
        ax.fill_between(xtime, mu - sem, mu + sem, color=colors[i], alpha=0.2)
    ax.axhline(0, ls='--', color='k', lw=0.8)
    add_vlines(ax, if_dpa=0)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim([0, 14])
    ax.legend(frameon=False, fontsize=10)

axes[-1].set_xlabel('Time from sample onset (s)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/panel_C_{dum}.svg')
plt.savefig(f'{OUT_DIR}/panel_C_{dum}.png', dpi=150)
plt.close()
print('saved panel_C')

# ── assemble figure 2BC ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(width * 2.4, height * 1.8))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        left=0.08, right=0.97, top=0.95, bottom=0.08,
                        hspace=0.5, wspace=0.4,
                        width_ratios=[5, 1, 3])

ax_heat = fig.add_subplot(gs[:, 0])
im = ax_heat.imshow(W_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2,
                    interpolation='nearest')
ax_heat.set_xticks(range(n_show))
ax_heat.set_xticklabels(pc_labels, fontsize=10)
ax_heat.set_ylabel(f'Neurons (n={W.shape[0]})')
ax_heat.set_yticks([])
plt.colorbar(im, ax=ax_heat, label='z-score', fraction=0.04, pad=0.02)
ax_heat.set_title('B', loc='left', fontweight='bold', fontsize=13)

ax_bar = fig.add_subplot(gs[:, 1])
ax_bar.barh(range(n_show), counts, color='0.4')
ax_bar.set_yticks(range(n_show))
ax_bar.set_yticklabels(pc_labels, fontsize=9)
ax_bar.set_xlabel('# neurons', fontsize=9)
ax_bar.invert_yaxis()

for row, ((pc_idx, col, labels, colors), ylabel) in enumerate(zip(conditions, ylabels)):
    ax = fig.add_subplot(gs[row, 2])
    for i in range(2):
        mask = mask_base & (y_single[col] == i)
        X_sel = X_single[mask][:, pc_idx, :]
        mu  = X_sel.mean(0)
        sem = X_sel.std(0) / np.sqrt(X_sel.shape[0])
        ax.plot(xtime, mu, color=colors[i], label=labels[i], lw=1.5)
        ax.fill_between(xtime, mu - sem, mu + sem, color=colors[i], alpha=0.2)
    ax.axhline(0, ls='--', color='k', lw=0.8)
    add_vlines(ax, if_dpa=0)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim([0, 14])
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    if row == 0:
        ax.set_title('C', loc='left', fontweight='bold', fontsize=13)
    if row == 2:
        ax.set_xlabel('Time from sample onset (s)', fontsize=9)

plt.savefig(f'{OUT_DIR}/figure2BC_{dum}.svg')
plt.savefig(f'{OUT_DIR}/figure2BC_{dum}.png', dpi=150)
plt.close()
print('saved figure2BC')
print('done')
