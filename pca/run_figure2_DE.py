import sys
sys.path.insert(0, '/home/leon/dual_task/dual_data/')
sys.path.insert(0, '/home/leon/dual/')

import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from utils.plot_utils import add_vlines
from utils.options import set_options

OUT_DIR   = '/home/leon/dual/pca/figures/figure2'
DATA_PCA  = '/home/leon/dual/data/pca'
DATA_CCGD = '/home/leon/dual_task/dual_data/data/ccgd'
os.makedirs(OUT_DIR, exist_ok=True)

dum        = 'pca_TEST_Expert_standard_loo_correct_odor_pair'
xtime_pca  = np.linspace(0, 14, 84)
xtime_ccgd = np.linspace(0, 14, 42)

# ── helpers ───────────────────────────────────────────────────────────────────
def pkl_load(name, path='.'):
    src = f'{path}/{name}.pkl'
    print('loading', src)
    return pkl.load(open(src, 'rb'))

options = set_options(mice=['JawsM15'], tasks=['Dual'], mouse='JawsM15', laser=0,
                      trials='', reload=0, data_type='dF', prescreen=None, pval=0.05,
                      preprocess=True, scaler_BL='standard_BL', avg_noise=False,
                      unit_var_BL=False, random_state=None, T_WINDOW=0.0,
                      l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
                      bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
                      class_weight=0, multilabel=0, mne_estimator='generalizing',
                      n_jobs=64)

# Delay bins in 84-bin space
bins_DELAY = options['bins_DELAY']

# ── load PCA data ─────────────────────────────────────────────────────────────
X_single   = pkl_load(f'single_traj_{dum}',    DATA_PCA)
y_single   = pkl_load(f'single_labels_{dum}',  DATA_PCA)

# ── load CCGD data ────────────────────────────────────────────────────────────
X_ccgd = pkl_load('X_CCGD_sample_generalizing_accuracy',    DATA_CCGD)
y_ccgd = pkl_load('labels_CCGD_sample_generalizing_accuracy', DATA_CCGD)

# Delay bins in 42-bin space (after smooth_and_bin2 which halved T)
bins_delay_42 = bins_DELAY // 2

print('X_single:', X_single.shape)
print('X_ccgd:  ', X_ccgd.shape)

# ──────────────────────────────────────────────────────────────────────────────
# Panel D: variance decomposition of PC projections by task variable
# ──────────────────────────────────────────────────────────────────────────────
# For each mouse × day, regress PC1-3 delay-period projections on sample, choice, test
# Report η² (SS_effect / SS_total) per variable, averaged over PCs and delay bins.

n_pcs = 3
mice_order = sorted(y_single.mouse.unique())

days_to_plot = [1.0, 6.0]  # first and last training day

def variance_decomp(X, y, pc_indices, delay_bins, day_val):
    """For a single mouse/day, compute η² for sample, choice, test."""
    m = (y.day == day_val) & (y.laser == 0) & (y.performance == 1)
    if m.sum() < 20:
        return dict(sample=np.nan, choice=np.nan, test=np.nan)

    Xd = X[m][:, :, delay_bins].mean(-1)  # (n_trials, n_comp)
    yd = y[m].reset_index(drop=True)

    labels = {'sample': yd.sample_odor.values,
              'choice': yd.choice.values,
              'test':   yd.test_odor.values}

    result = {}
    for var_name, var_vals in labels.items():
        ss_total = 0.0
        ss_effect = 0.0
        for pc in pc_indices:
            proj = Xd[:, pc]
            grand_mean = proj.mean()
            ss_t = ((proj - grand_mean)**2).sum()
            groups = [proj[var_vals == g] for g in np.unique(var_vals)]
            ss_b = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups if len(g) > 0)
            ss_total  += ss_t
            ss_effect += ss_b
        result[var_name] = ss_effect / (ss_total + 1e-12)

    return result

# Per-mouse day1 and day6 variance decomposition
vd_records = []
for mouse in mice_order:
    m_mouse = y_single.mouse == mouse
    X_m = X_single[m_mouse]
    y_m = y_single[m_mouse].reset_index(drop=True)
    for day in days_to_plot:
        vd = variance_decomp(X_m, y_m, range(n_pcs), bins_DELAY, day)
        vd['mouse'] = mouse
        vd['day']   = day
        vd_records.append(vd)

vd_df = pd.DataFrame(vd_records)
print('\nVariance decomposition (mean across mice):')
print(vd_df.groupby('day')[['sample','choice','test']].mean())

# ── panel D plot ──────────────────────────────────────────────────────────────
vars_order  = ['sample', 'choice', 'test']
var_colors  = ['#332288', '#4daf4a', '#e41a1c']
var_labels  = ['Sample', 'Choice', 'Test']

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

for ax, var, col, lbl in zip(axes, vars_order, var_colors, var_labels):
    day1_vals = vd_df[vd_df.day == 1.0][var].values
    day6_vals = vd_df[vd_df.day == 6.0][var].values

    for d1, d6 in zip(day1_vals, day6_vals):
        if not (np.isnan(d1) or np.isnan(d6)):
            ax.plot([0, 1], [d1, d6], color=col, alpha=0.4, lw=1)
            ax.scatter([0], [d1], s=40, facecolors='none', edgecolors=col, lw=1.5, zorder=3)
            ax.scatter([1], [d6], s=40, facecolors=col, edgecolors=col, zorder=3)

    valid = [(d1, d6) for d1, d6 in zip(day1_vals, day6_vals)
             if not (np.isnan(d1) or np.isnan(d6))]
    if valid:
        m1 = np.mean([v[0] for v in valid])
        m6 = np.mean([v[1] for v in valid])
        s1 = np.std([v[0] for v in valid]) / np.sqrt(len(valid))
        s6 = np.std([v[1] for v in valid]) / np.sqrt(len(valid))
        ax.errorbar([0, 1], [m1, m6], yerr=[s1, s6], color='k', lw=2,
                    capsize=4, zorder=4)

    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Day 1', 'Day 6'])
    ax.set_title(lbl, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Variance explained (η²)', fontsize=10)
fig.suptitle('D', x=0.02, y=1.0, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/panel_D_{dum}.svg')
plt.savefig(f'{OUT_DIR}/panel_D_{dum}.png', dpi=150)
plt.close()
print('saved panel_D')

# ──────────────────────────────────────────────────────────────────────────────
# Panel E: CCGD – within vs cross-context sample decoding
# ──────────────────────────────────────────────────────────────────────────────

def ccgd_acc(X, y, mouse, stage=None, day=None):
    """Return (within_acc, cross_acc) for a mouse/stage or mouse/day."""
    m_base = y.mouse == mouse
    if stage is not None:
        m_base = m_base & (y.stage == stage)
    if day is not None:
        m_base = m_base & (y.day == day)

    # DualGo vs DualNoGo cross-context only
    m_w_go   = m_base & (y.context == 'DualGo')   & (y.tasks == 'DualGo')
    m_w_nogo = m_base & (y.context == 'DualNoGo') & (y.tasks == 'DualNoGo')
    m_c_g2n  = m_base & (y.context == 'DualGo')   & (y.tasks == 'DualNoGo')
    m_c_n2g  = m_base & (y.context == 'DualNoGo') & (y.tasks == 'DualGo')

    def diag_delay(mask):
        if mask.sum() == 0:
            return np.nan
        d = np.diagonal(X[mask], axis1=-2, axis2=-1)
        return d[:, bins_delay_42].mean()

    w_vals = [diag_delay(m) for m in [m_w_go, m_w_nogo]]
    c_vals = [diag_delay(m) for m in [m_c_g2n, m_c_n2g]]

    w_vals = [v for v in w_vals if not np.isnan(v)]
    c_vals = [v for v in c_vals if not np.isnan(v)]

    return (np.mean(w_vals) if w_vals else np.nan,
            np.mean(c_vals) if c_vals else np.nan)


# Per-mouse, per-stage
stage_records = []
for mouse in mice_order:
    for stage in ['Naive', 'Expert']:
        w, c = ccgd_acc(X_ccgd, y_ccgd, mouse, stage=stage)
        stage_records.append(dict(mouse=mouse, stage=stage, within=w, cross=c))

stage_df = pd.DataFrame(stage_records)

# Per-mouse, per-day (for gap trajectory)
day_records = []
for mouse in mice_order:
    for day in sorted(y_ccgd.day.unique()):
        w, c = ccgd_acc(X_ccgd, y_ccgd, mouse, day=day)
        day_records.append(dict(mouse=mouse, day=day, within=w, cross=c, gap=w - c))

day_df = pd.DataFrame(day_records)

# ── panel E plot ──────────────────────────────────────────────────────────────
mouse_colors = dict(zip(mice_order,
                        plt.cm.tab10(np.linspace(0, 1, len(mice_order)))))

fig = plt.figure(figsize=(12, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig, left=0.08, right=0.97,
                        top=0.90, bottom=0.15, wspace=0.4)

stage_labels = ['Naive', 'Expert']
for col_idx, stage in enumerate(stage_labels):
    ax = fig.add_subplot(gs[0, col_idx])
    df_s = stage_df[stage_df.stage == stage]

    for _, row in df_s.iterrows():
        col = mouse_colors[row.mouse]
        ax.plot([0, 1], [row.within, row.cross], color=col, lw=1, alpha=0.7)
        ax.scatter([0], [row.within], s=45, color=col, zorder=3)
        ax.scatter([1], [row.cross],  s=45, color=col, zorder=3)

    # group mean ± SEM
    w_vals = df_s.within.dropna().values
    c_vals = df_s.cross.dropna().values
    ax.errorbar([0, 1],
                [w_vals.mean(), c_vals.mean()],
                yerr=[w_vals.std()/np.sqrt(len(w_vals)),
                      c_vals.std()/np.sqrt(len(c_vals))],
                color='k', lw=2, capsize=5, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Within', 'Cross'], fontsize=10)
    ax.set_ylabel('Decoding accuracy' if col_idx == 0 else '')
    ax.set_title(stage, fontsize=11)
    ax.axhline(0.5, ls='--', color='k', lw=0.8, alpha=0.5)
    ax.set_ylim(0.4, 0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if col_idx == 0:
        ax.set_title('E', loc='left', fontweight='bold', fontsize=13)
        ax.set_title('Naive', fontsize=11)

# Right panel: generalization gap across days
ax_gap = fig.add_subplot(gs[0, 2])
all_days = sorted(day_df.day.unique())

for mouse in mice_order:
    df_m = day_df[day_df.mouse == mouse].sort_values('day')
    ax_gap.plot(df_m.day, df_m.gap, color=mouse_colors[mouse],
                lw=1, alpha=0.6, marker='o', markersize=3)

# group mean ± SEM across mice per day
gap_mean = day_df.groupby('day')['gap'].mean()
gap_sem  = day_df.groupby('day')['gap'].sem()
ax_gap.errorbar(gap_mean.index, gap_mean.values, yerr=gap_sem.values,
                color='k', lw=2, capsize=4, zorder=4)
ax_gap.axhline(0, ls='--', color='k', lw=0.8)
ax_gap.set_xlabel('Day', fontsize=10)
ax_gap.set_ylabel('Generalization gap\n(within − cross)', fontsize=10)
ax_gap.set_xticks(all_days)
ax_gap.spines['top'].set_visible(False)
ax_gap.spines['right'].set_visible(False)

plt.savefig(f'{OUT_DIR}/panel_E_{dum}.svg')
plt.savefig(f'{OUT_DIR}/panel_E_{dum}.png', dpi=150)
plt.close()
print('saved panel_E')

# ── assemble figure 2DE ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))
gs2 = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.98,
                        top=0.92, bottom=0.12, wspace=0.4,
                        width_ratios=[3, 4])

# Panel D (left: 3 var decomp subplots)
gs_d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs2[0], wspace=0.25)
for i, (var, col, lbl) in enumerate(zip(vars_order, var_colors, var_labels)):
    ax = fig.add_subplot(gs_d[i])
    day1_vals = vd_df[vd_df.day == 1.0][var].values
    day6_vals = vd_df[vd_df.day == 6.0][var].values
    for d1, d6 in zip(day1_vals, day6_vals):
        if not (np.isnan(d1) or np.isnan(d6)):
            ax.plot([0, 1], [d1, d6], color=col, alpha=0.4, lw=1)
            ax.scatter([0], [d1], s=30, facecolors='none', edgecolors=col, lw=1.5, zorder=3)
            ax.scatter([1], [d6], s=30, facecolors=col, edgecolors=col, zorder=3)
    valid = [(d1, d6) for d1, d6 in zip(day1_vals, day6_vals)
             if not (np.isnan(d1) or np.isnan(d6))]
    if valid:
        m1 = np.mean([v[0] for v in valid])
        m6 = np.mean([v[1] for v in valid])
        s1 = np.std([v[0] for v in valid]) / np.sqrt(len(valid))
        s6 = np.std([v[1] for v in valid]) / np.sqrt(len(valid))
        ax.errorbar([0, 1], [m1, m6], yerr=[s1, s6], color='k', lw=2, capsize=4, zorder=4)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['D1', 'D6'], fontsize=8)
    ax.set_title(lbl, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i == 0:
        ax.set_ylabel('η²', fontsize=9)
        ax.set_title('D', loc='left', fontweight='bold', fontsize=13)
        ax.set_title(lbl, fontsize=9)

# Panel E (right: 3 subplots)
gs_e = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs2[1], wspace=0.3)

for col_idx, stage in enumerate(stage_labels):
    ax = fig.add_subplot(gs_e[col_idx])
    df_s = stage_df[stage_df.stage == stage]
    for _, row in df_s.iterrows():
        col_m = mouse_colors[row.mouse]
        ax.plot([0, 1], [row.within, row.cross], color=col_m, lw=1, alpha=0.7)
        ax.scatter([0], [row.within], s=35, color=col_m, zorder=3)
        ax.scatter([1], [row.cross],  s=35, color=col_m, zorder=3)
    w_vals = df_s.within.dropna().values
    c_vals = df_s.cross.dropna().values
    if len(w_vals):
        ax.errorbar([0, 1],
                    [w_vals.mean(), c_vals.mean()],
                    yerr=[w_vals.std()/np.sqrt(len(w_vals)),
                          c_vals.std()/np.sqrt(len(c_vals))],
                    color='k', lw=2, capsize=5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Within', 'Cross'], fontsize=8)
    ax.set_title(stage, fontsize=9)
    ax.axhline(0.5, ls='--', color='k', lw=0.8, alpha=0.5)
    ax.set_ylim(0.4, 0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_idx == 0:
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_title('E', loc='left', fontweight='bold', fontsize=13)
        ax.set_title(stage, fontsize=9)

ax_gap = fig.add_subplot(gs_e[2])
for mouse in mice_order:
    df_m = day_df[day_df.mouse == mouse].sort_values('day')
    ax_gap.plot(df_m.day, df_m.gap, color=mouse_colors[mouse],
                lw=1, alpha=0.6, marker='o', markersize=3)
ax_gap.errorbar(gap_mean.index, gap_mean.values, yerr=gap_sem.values,
                color='k', lw=2, capsize=4, zorder=4)
ax_gap.axhline(0, ls='--', color='k', lw=0.8)
ax_gap.set_xlabel('Day', fontsize=9)
ax_gap.set_ylabel('Gap (W−C)', fontsize=9)
ax_gap.set_xticks(all_days)
ax_gap.spines['top'].set_visible(False)
ax_gap.spines['right'].set_visible(False)

plt.savefig(f'{OUT_DIR}/figure2DE_{dum}.svg')
plt.savefig(f'{OUT_DIR}/figure2DE_{dum}.png', dpi=150)
plt.close()
print('saved figure2DE')
print('done')
