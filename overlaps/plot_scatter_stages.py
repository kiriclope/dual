"""
Scatter: choice-code strength Naive (x) vs Expert (y), one dot per mouse.

3 panels = DPA / DualGo / DualNoGo.
One figure per train epoch.
Diagonal y = x line marks no change; points above = stronger in Expert.

Strength = mean X_epoch over LATE_DELAY bins (27–53, ~4.3–9 s).
Per-mouse baseline normalisation applied per stage before computing strength.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, mannwhitneyu

from src.common.options import set_options
from src.pca.io import pkl_load

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5 ** 0.5 - 1) / 2
width  = 6
height = width * golden_ratio

matplotlib.rcParams.update({
    'figure.figsize':    (width, height),
    'lines.markersize':  5,
    'axes.titlesize':    22,
    'axes.labelsize':    18,
    'xtick.labelsize':   14,
    'ytick.labelsize':   14,
    'axes.titlepad':     20,
    'axes.labelpad':     8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.size':         13,
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/scatter_stages'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']
STAGES     = ['Naive', 'Expert']

# ── Epoch bins ────────────────────────────────────────────────────────────────

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_BL   = options['bins_BL']
BINS_LATE = np.arange(27, 54)   # late delay: ~4.3–9 s

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
]

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

# All non-laser trials (no correct filter — Naive mice have too few correct
# trials per label class to get stable discriminability estimates)
idx_laser = (y_single.laser == 0)

# ── Colours per mouse ─────────────────────────────────────────────────────────

pal_mice = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}

os.makedirs(os.path.join(FIG_BASE, 'png'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'svg'), exist_ok=True)

# ── Loop over train epochs ────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    print(f'\n=== {train_tag} (train bins {bins_train[0]}–{bins_train[-1]}) ===')

    # Per-stage X_epoch with per-mouse BL normalisation
    X_by_stage = {}
    for stage in STAGES:
        X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
        for mouse in ALL_MICE:
            m  = (y_single.mouse == mouse).values
            sd = X_ep[m][:, BINS_BL].std()
            if sd > 0:
                X_ep[m] /= sd
        X_by_stage[stage] = X_ep

    # ── Collect data: pooled and by-sample-identity ───────────────────────────

    # pooled[cond] = (xs_naive, xs_expert, colors)  — 9 points
    # by_sample[cond] = list of (xn, xe, color, marker) — 18 points
    pooled    = {}
    by_sample = {}

    SAMPLE_CLASSES = [(0, [0, 1], 'o'), (1, [2, 3], '^')]  # (cls_label, odor_pairs, marker)

    def _strength(X_ep, mask_base):
        """Mean decision value over BINS_LATE — distance from the boundary."""
        m = mask_base.values
        if m.sum() == 0:
            return np.nan
        return X_ep[m][:, BINS_LATE].mean()

    for cond in CONDITIONS:
        xn_pool, xe_pool, col_pool = [], [], []
        pts_sample = []

        for mouse in ALL_MICE:
            color = MOUSE_COLOR[mouse]

            # Pooled discriminability: all laser=0 choice trials
            pool_n, pool_e = np.nan, np.nan
            for stage in STAGES:
                base = (
                    (y_single.mouse  == mouse)    &
                    (y_single.tasks  == cond)     &
                    (y_single.stage  == stage)    &
                    (y_single.target == 'choice') &
                    idx_laser
                )
                val = _strength(X_by_stage[stage], base)
                if stage == 'Naive': pool_n = val
                else:                pool_e = val
            xn_pool.append(pool_n); xe_pool.append(pool_e); col_pool.append(color)

            # Discriminability split by sample identity
            mouse_pts = []
            for cls_label, odor_pairs, marker in SAMPLE_CLASSES:
                xn, xe = np.nan, np.nan
                for stage in STAGES:
                    base = (
                        (y_single.mouse  == mouse)              &
                        (y_single.tasks  == cond)               &
                        (y_single.stage  == stage)              &
                        (y_single.target == 'choice')           &
                        y_single.odor_pair.isin(odor_pairs)     &
                        idx_laser
                    )
                    val = _strength(X_by_stage[stage], base)
                    if stage == 'Naive': xn = val
                    else:                xe = val
                mouse_pts.append((xn, xe, color, marker))
            pts_sample.append(mouse_pts)

        pooled[cond]    = (np.array(xn_pool, float), np.array(xe_pool, float), col_pool)
        by_sample[cond] = pts_sample

    # ── Trial-level Mann-Whitney U and paired t-test across mice ─────────────
    # Mann-Whitney: all trials as independent observations (anti-conservative,
    #   ignores mouse structure — n_trials >> 9 so p-values are too small).
    # Paired t-test: mouse means (n=9, df=8) — correct accounting for structure.

    trial_stats = {}   # cond -> dict with keys 'mwu', 'paired_t'
    for cond in CONDITIONS:
        naive_all, expert_all = [], []
        naive_means, expert_means = [], []
        for mouse in ALL_MICE:
            vals = {}
            for stage in STAGES:
                mask = (
                    (y_single.mouse  == mouse)    &
                    (y_single.tasks  == cond)     &
                    (y_single.stage  == stage)    &
                    (y_single.target == 'choice') &
                    idx_laser
                ).values
                if mask.sum() == 0:
                    vals[stage] = None
                    continue
                v = X_by_stage[stage][mask][:, BINS_LATE].mean(axis=1)
                vals[stage] = v
                if stage == 'Naive':
                    naive_all.extend(v.tolist())
                else:
                    expert_all.extend(v.tolist())
            if vals.get('Naive') is not None and vals.get('Expert') is not None:
                naive_means.append(vals['Naive'].mean())
                expert_means.append(vals['Expert'].mean())

        result = {}
        if len(naive_all) > 0 and len(expert_all) > 0:
            stat, p = mannwhitneyu(naive_all, expert_all, alternative='two-sided')
            result['mwu'] = p
        if len(naive_means) >= 3:
            _, p = ttest_rel(naive_means, expert_means)
            result['paired_t'] = p
        trial_stats[cond] = result

    # ── Helper: draw one panel set and save ───────────────────────────────────

    from matplotlib.lines import Line2D

    def _draw_and_save(fname, use_sample_split, title_suffix):
        fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), sharey=False)

        for ax, cond in zip(axes, CONDITIONS):
            if use_sample_split:
                # flatten 9×2 into arrays for diagonal + Pearson
                all_xn = [p[0] for mpts in by_sample[cond] for p in mpts]
                all_xe = [p[1] for mpts in by_sample[cond] for p in mpts]
                xn_arr = np.array(all_xn, float)
                xe_arr = np.array(all_xe, float)
            else:
                xn_arr, xe_arr, _ = pooled[cond]

            # Diagonal
            all_vals = np.concatenate([xn_arr, xe_arr])
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            margin = (vmax - vmin) * 0.1
            diag = [vmin - margin, vmax + margin]
            ax.plot(diag, diag, ls='--', color='k', lw=0.8, zorder=1)

            if use_sample_split:
                for mouse_pts in by_sample[cond]:
                    pts_xn = [p[0] for p in mouse_pts]
                    pts_xe = [p[1] for p in mouse_pts]
                    color  = mouse_pts[0][2]
                    ax.plot(pts_xn, pts_xe, '-', color=color, lw=0.8,
                            alpha=0.5, zorder=3)
                    for xn, xe, col, marker in mouse_pts:
                        if not (np.isnan(xn) or np.isnan(xe)):
                            ax.scatter(xn, xe, color=col, marker=marker,
                                       s=70, zorder=5,
                                       linewidths=0.5, edgecolors='w')
            else:
                _, _, colors = pooled[cond]
                for xv, yv, color in zip(xn_arr, xe_arr, colors):
                    if not (np.isnan(xv) or np.isnan(yv)):
                        ax.scatter(xv, yv, color=color, s=80, zorder=5,
                                   linewidths=0.5, edgecolors='w')

            def fmt_p(p): return f'p={p:.3f}' if p >= 0.001 else 'p<0.001'

            # Pearson + Spearman (cross-mouse correlation, n=9)
            valid = ~(np.isnan(xn_arr) | np.isnan(xe_arr))
            lines = []
            if valid.sum() >= 3:
                r_p, p_p = pearsonr(xn_arr[valid], xe_arr[valid])
                r_s, p_s = spearmanr(xn_arr[valid], xe_arr[valid])
                lines += [f'r={r_p:.2f} {fmt_p(p_p)}',
                           f'ρ={r_s:.2f} {fmt_p(p_s)}']

            # Paired t-test on mouse means (n=9, correct)
            ts = trial_stats[cond]
            if 'paired_t' in ts:
                lines.append(f't-test {fmt_p(ts["paired_t"])} (n=9)')
            # Mann-Whitney on all trials (anti-conservative, trials ≠ independent)
            if 'mwu' in ts:
                lines.append(f'MWU {fmt_p(ts["mwu"])}*')

            if lines:
                ax.text(0.05, 0.95, '\n'.join(lines),
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=10)

            ax.set_xlabel('Choice code — Naive (BL σ)', labelpad=8)
            ax.set_ylabel('Choice code — Expert (BL σ)', labelpad=8)
            ax.set_title(cond)

        # Legend
        if use_sample_split:
            handles = (
                [Line2D([0],[0], marker='o', color='k', ls='none', ms=8,
                         label='odor A sample'),
                 Line2D([0],[0], marker='^', color='k', ls='none', ms=8,
                         label='odor B sample')]
                + [Line2D([0],[0], marker='o', color=MOUSE_COLOR[m], ls='none',
                           ms=8, label=m) for m in ALL_MICE]
            )
        else:
            handles = [
                Line2D([0],[0], marker='o', color=MOUSE_COLOR[m], ls='none',
                       ms=9, label=m) for m in ALL_MICE
            ]
        axes[-1].legend(handles=handles, fontsize=9, frameon=False,
                        bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.suptitle(
            f'Choice code: Naive vs Expert  [{train_tag}]{title_suffix}  —  '
            f'late-delay (bins {BINS_LATE[0]}–{BINS_LATE[-1]})',
            fontsize=15, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {fname}')

    for stem, kwargs in [
        (f'{DUM}_{train_tag}',            dict(use_sample_split=False, title_suffix='')),
        (f'{DUM}_{train_tag}_by_sample',  dict(use_sample_split=True,  title_suffix='  [by sample identity]')),
    ]:
        _draw_and_save(os.path.join(FIG_BASE, 'png', f'{stem}.png'), **kwargs)
        _draw_and_save(os.path.join(FIG_BASE, 'svg', f'{stem}.svg'), **kwargs)

print(f'\nScatter stages → {FIG_BASE}/')
