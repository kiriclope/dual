"""
Scatter: Δ choice code (Expert − Naive) vs Δ performance (Expert − Naive).

Two figures per train epoch:
  _dpa_perf.png  : y = Δ mean DPA performance per mouse
  _gng_perf.png  : y = Δ mean GNG performance (odr_perf) per mouse, Dual trials only

3 panels per figure = DPA / DualGo / DualNoGo (x = Δ choice code in that context).
One dot per mouse. Stats: Pearson, Spearman, paired t-test (n=9).

Strength = mean X_epoch over BINS_LATE (27–53, ~4.3–9 s).
Performance computed from target=='choice' trials to avoid triple-counting.
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_1samp, linregress, t as t_dist

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
    'axes.titlesize':    22, 'axes.labelsize':  18,
    'xtick.labelsize':   14, 'ytick.labelsize': 14,
    'axes.titlepad':     20, 'axes.labelpad':   8,
    'axes.spines.top':   False, 'axes.spines.right': False,
    'font.size':         13,
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/scatter_perf'
DPA_PANEL = '--dpa-panel' in sys.argv[1:]   # focused 1×2: ΔDPA-perf & ΔGNG-perf vs DPA depth

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

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
BINS_LATE = np.arange(27, 54)

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
    ('trainLD',     options['bins_LD']),
    ('trainLD_TEST', np.concatenate([options['bins_LD'], options['bins_TEST']])),
    ('trainTEST_CHOICE', np.concatenate([options['bins_TEST'], options['bins_CHOICE']])),
    ('trainLD_TEST_CHOICE',
     np.concatenate([options['bins_LD'], options['bins_TEST'], options['bins_CHOICE']])),
    # narrow LD/TEST boundary: last 0.5 s of LD + first 0.5 s of TEST (bins 51-56)
    ('trainLDTEST05', np.concatenate([options['bins_LD'][-3:], options['bins_TEST'][:3]])),
]

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_laser  = (y_single.laser == 0)
idx_choice = (y_single.target == 'choice')   # one row per trial, avoids triple-counting
idx_correct = (
    idx_laser &
    (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)

# ── Per-mouse Δ performance (computed once, independent of train epoch) ───────

def perf_delta(perf_col, task_mask):
    """Δ performance = Expert − Naive mean for each mouse."""
    delta = {}
    for mouse in ALL_MICE:
        vals = {}
        for stage in STAGES:
            m = (
                (y_single.mouse == mouse) &
                (y_single.stage == stage) &
                idx_laser & idx_choice & task_mask
            )
            col = y_single.loc[m, perf_col].dropna()
            vals[stage] = col.mean() if len(col) > 0 else np.nan
        delta[mouse] = vals['Expert'] - vals['Naive']
    return delta   # {mouse: scalar}

delta_dpa_perf = perf_delta('performance', (y_single.tasks == 'DPA'))
delta_gng_perf = perf_delta('odr_perf',    (y_single.tasks != 'DPA'))

# ── Colours per mouse ─────────────────────────────────────────────────────────

pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}

# marker by opto group — matches panel E (plot_scatter_laser.py). ACC has no laser
# but appears in this learning scatter (n=9), so it gets its own marker.
GROUP   = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
           **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}     # ● Jaws / ▲ ChR / ■ ACC

# rcParams IDENTICAL to panel E (plot_scatter_laser.py) — applied locally to the
# DPA panel only (via rc_context) so it matches E's size/dpi/fonts exactly without
# changing this script's other (larger-font) figures.
E_RC = {
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none',
    # thin axis/tick lines (matplotlib defaults) — override sns "poster" bold spines
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
    'lines.linewidth': 1.5,
}

os.makedirs(FIG_BASE, exist_ok=True)

# ── Stats helper ──────────────────────────────────────────────────────────────

def fmt_p(p):
    return f'p={p:.3f}' if p >= 0.001 else 'p<0.001'

def annotate_stats(ax, xs, ys):
    valid = ~(np.isnan(xs) | np.isnan(ys))
    if valid.sum() < 3:
        return
    xv, yv = xs[valid], ys[valid]
    r_p, p_p = pearsonr(xv, yv)
    r_s, p_s = spearmanr(xv, yv)
    _, p_t = ttest_1samp(yv, 0)
    n = valid.sum()
    text = (f'r={r_p:.2f} {fmt_p(p_p)}   ρ={r_s:.2f} {fmt_p(p_s)}\n'
            f'1-samp t (Δperf≠0) {fmt_p(p_t)}  n={n}')
    ax.text(0.5, 1.01, text, transform=ax.transAxes,
            va='bottom', ha='center', fontsize=9, color='0.3')

def regression_band(ax, xs, ys, color='k', alpha=0.15):
    """Draw regression line + 95% confidence band."""
    valid = ~(np.isnan(xs) | np.isnan(ys))
    if valid.sum() < 3:
        return
    xv, yv = xs[valid], ys[valid]
    slope, intercept, _, _, se = linregress(xv, yv)
    n = len(xv)
    x_line = np.linspace(xv.min(), xv.max(), 200)
    y_line = slope * x_line + intercept
    # analytical 95% CI on the mean prediction
    x_mean = xv.mean()
    ss_x   = np.sum((xv - x_mean) ** 2)
    se_band = se * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)
    t_crit  = t_dist.ppf(0.975, df=n - 2)
    ax.plot(x_line, y_line, color=color, lw=1.5, ls='-', zorder=4)
    ax.fill_between(x_line,
                    y_line - t_crit * se_band,
                    y_line + t_crit * se_band,
                    color=color, alpha=alpha, zorder=2)

# ── Loop over train epochs ────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    print(f'\n=== {train_tag} ===')
    fig_dir = os.path.join(FIG_BASE, train_tag)
    os.makedirs(os.path.join(fig_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(fig_dir, 'svg'), exist_ok=True)

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

    SAMPLE_CLASSES = [(0, [0, 1], 'o', 'odor A'), (1, [2, 3], '^', 'odor B')]

    # Δ choice code per mouse × condition — pooled and by sample identity
    delta_choice        = {}   # (mouse, cond)             -> scalar
    delta_choice_sample = {}   # (mouse, cond, cls_label)  -> scalar

    for mouse in ALL_MICE:
        for cond in CONDITIONS:
            vals = {}
            vals_cls = {0: {}, 1: {}}
            for stage in STAGES:
                base = (
                    (y_single.mouse  == mouse)    &
                    (y_single.tasks  == cond)     &
                    (y_single.stage  == stage)    &
                    (y_single.target == 'choice') &
                    idx_correct
                )
                mask = base.values
                vals[stage] = (X_by_stage[stage][mask][:, BINS_LATE].mean()
                               if mask.sum() else np.nan)
                for cls_label, odor_pairs, _, _ in SAMPLE_CLASSES:
                    m = (base & y_single.odor_pair.isin(odor_pairs)).values
                    vals_cls[cls_label][stage] = (
                        X_by_stage[stage][m][:, BINS_LATE].mean()
                        if m.sum() else np.nan)

            delta_choice[(mouse, cond)] = vals['Expert'] - vals['Naive']
            for cls_label, _, _, _ in SAMPLE_CLASSES:
                vc = vals_cls[cls_label]
                delta_choice_sample[(mouse, cond, cls_label)] = (
                    vc['Expert'] - vc['Naive'])

    # Δ performance split by sample identity
    def perf_delta_by_sample(perf_col, task_mask):
        """Returns {(mouse, cls_label): scalar}."""
        out = {}
        for mouse in ALL_MICE:
            for cls_label, odor_pairs, _, _ in SAMPLE_CLASSES:
                vals = {}
                for stage in STAGES:
                    m = (
                        (y_single.mouse == mouse) &
                        (y_single.stage == stage) &
                        idx_laser & idx_choice & task_mask &
                        y_single.odor_pair.isin(odor_pairs)
                    )
                    col = y_single.loc[m, perf_col].dropna()
                    vals[stage] = col.mean() if len(col) > 0 else np.nan
                out[(mouse, cls_label)] = vals['Expert'] - vals['Naive']
        return out

    delta_dpa_perf_sample = perf_delta_by_sample('performance', y_single.tasks == 'DPA')
    delta_gng_perf_sample = perf_delta_by_sample('odr_perf',    y_single.tasks != 'DPA')

    # ── Draw helper ───────────────────────────────────────────────────────────

    def draw_perf_fig(perf_label, delta_perf, delta_perf_sample, ylabel,
                      by_sample, suffix):
        fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), sharey=True)

        # Collect x/y per panel; y shared, x per-panel centred on 0
        all_y_global = []
        panel_data   = {}
        panel_xlim   = {}
        panel_data   = {}

        def _lims(vals, pad=0.15):
            v = np.array(vals, float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return -1, 1
            lo, hi = v.min(), v.max()
            m = (hi - lo) * pad or 0.1
            return lo - m, hi + m

        for cond in CONDITIONS:
            if by_sample:
                all_dx, all_dy = [], []
                for mouse in ALL_MICE:
                    for cls_label, _, _, _ in SAMPLE_CLASSES:
                        all_dx.append(delta_choice_sample[(mouse, cond, cls_label)])
                        all_dy.append(delta_perf_sample.get((mouse, cls_label), np.nan))
            else:
                all_dx = [delta_choice[(m, cond)] for m in ALL_MICE]
                all_dy = [delta_perf[m]           for m in ALL_MICE]
            panel_data[cond] = (np.array(all_dx, float), np.array(all_dy, float))
            all_y_global.extend(all_dy)
            raw = _lims(all_dx)
            half = max(abs(raw[0]), abs(raw[1]))
            panel_xlim[cond] = (-half, half)

        ylim = _lims(all_y_global)

        # Second pass: plot
        for ax, cond in zip(axes, CONDITIONS):
            dx_arr, dy_arr = panel_data[cond]

            if by_sample:
                for mouse in ALL_MICE:
                    pts_x, pts_y = [], []
                    for cls_label, _, marker, _ in SAMPLE_CLASSES:
                        dx = delta_choice_sample[(mouse, cond, cls_label)]
                        dy = delta_perf_sample.get((mouse, cls_label), np.nan)
                        pts_x.append(dx); pts_y.append(dy)
                    ax.plot(pts_x, pts_y, '-', color=MOUSE_COLOR[mouse],
                            lw=0.8, alpha=0.5, zorder=3)
                    for xv, yv, (_, _, marker, _) in zip(
                            pts_x, pts_y, SAMPLE_CLASSES):
                        if not (np.isnan(xv) or np.isnan(yv)):
                            ax.scatter(xv, yv, color=MOUSE_COLOR[mouse],
                                       marker=marker, s=70, zorder=5,
                                       linewidths=0.5, edgecolors='w')
            else:
                for xv, yv, mouse in zip(dx_arr, dy_arr, ALL_MICE):
                    if not (np.isnan(xv) or np.isnan(yv)):
                        ax.scatter(xv, yv, color=MOUSE_COLOR[mouse], s=80,
                                   zorder=5, linewidths=0.5, edgecolors='w')

            regression_band(ax, dx_arr, dy_arr)
            ax.axhline(0, ls=':', color='k', lw=0.8)
            ax.axvline(0, ls=':', color='k', lw=0.8)
            ax.set_xlim(panel_xlim[cond])
            ax.set_ylim(ylim)
            annotate_stats(ax, dx_arr, dy_arr)
            ax.set_xlabel('Δ choice loc.', labelpad=8)
            if ax is axes[0]:
                ax.set_ylabel(ylabel, labelpad=8)
            ax.set_title(cond.replace('DualGo', 'Go').replace('DualNoGo', 'NoGo'))

        # Legend
        if by_sample:
            legend_handles = (
                [mlines.Line2D([0],[0], marker='o', color='k', ls='none',
                                ms=8, label='odor A sample'),
                 mlines.Line2D([0],[0], marker='^', color='k', ls='none',
                                ms=8, label='odor B sample')]
                + [mlines.Line2D([0],[0], marker='o', color=MOUSE_COLOR[m],
                                  ls='none', ms=8, label=m) for m in ALL_MICE]
            )
        else:
            legend_handles = [
                mlines.Line2D([0],[0], marker='o', color=MOUSE_COLOR[m],
                              ls='none', ms=9, label=m) for m in ALL_MICE
            ]
        axes[-1].legend(handles=legend_handles, fontsize=9, frameon=False,
                        bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.suptitle(
            f'Δ choice code vs Δ {perf_label.replace("_"," ")}  [{train_tag}]  —  '
            f'late-delay (bins {BINS_LATE[0]}–{BINS_LATE[-1]})',
            fontsize=14, y=1.02,
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.82)
        stem = f'{DUM}_{train_tag}_{perf_label}{suffix}'
        fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
        path = os.path.join(fig_dir, 'png', f'{stem}.png')
        plt.close(fig)
        print(f'saved {path}')

    # ── Save pooled and by-sample versions ───────────────────────────────────

    for perf_label, delta_perf, delta_perf_sample, ylabel in [
        ('dpa_perf', delta_dpa_perf, delta_dpa_perf_sample,
         'Δ DPA performance (Expert−Naive)'),
        ('gng_perf', delta_gng_perf, delta_gng_perf_sample,
         'Δ GNG performance (Expert−Naive)'),
    ]:
        draw_perf_fig(perf_label, delta_perf, delta_perf_sample, ylabel,
                      by_sample=False, suffix='')
        draw_perf_fig(perf_label, delta_perf, delta_perf_sample, ylabel,
                      by_sample=True,  suffix='_by_sample')

    # ── Focused DPA-only panel: Δdepth(DPA) vs ΔDPA-perf and ΔGNG-perf ─────────
    #   Styled to MATCH panel E (plot_scatter_laser.py) exactly: figsize (9,3.7),
    #   per-animal tab10 colors + group markers (● Jaws / ▲ ChR / ■ ACC), the same
    #   'all (n=N): r=.. p=..  ρ=.. p=..' stat line, star from Spearman, and a
    #   per-mouse legend on the right. This is the LEARNING (Expert−Naive) analog of
    #   E's laser (on−off) causal scatter.
    if DPA_PANEL:
      with plt.rc_context(E_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
        xdepth = np.array([delta_choice[(m, 'DPA')] for m in ALL_MICE], float)
        specs = [(delta_dpa_perf, 'Δ DPA accuracy (Exp−Naive)'),
                 (delta_gng_perf, 'Δ GNG accuracy (Exp−Naive)')]
        # shared y-limits across both panels
        ally = np.array([d[m] for d, _ in specs for m in ALL_MICE], float)
        ally = ally[~np.isnan(ally)]
        ypad = (ally.max() - ally.min()) * 0.15 or 0.05
        ylim = (ally.min() - ypad, ally.max() + ypad)
        for ax, (yv_dict, ylabel) in zip(axes, specs):
            yv = np.array([yv_dict[m] for m in ALL_MICE], float)
            for xx, yy, m in zip(xdepth, yv, ALL_MICE):
                if not (np.isnan(xx) or np.isnan(yy)):
                    ax.scatter(xx, yy, color=MOUSE_COLOR[m], marker=GMARKER[GROUP[m]],
                               s=90, edgecolors='w', linewidths=0.6, zorder=5,
                               label=m if ax is axes[1] else None)
            regression_band(ax, xdepth, yv, color='0.25')
            ax.axhline(0, ls=':', color='k', lw=0.8)
            ax.axvline(0, ls=':', color='k', lw=0.8)
            ax.set_ylim(ylim)
            ok = ~(np.isnan(xdepth) | np.isnan(yv))
            r_p, p_p = pearsonr(xdepth[ok], yv[ok])
            r_s, p_s = spearmanr(xdepth[ok], yv[ok])
            txt = (f'all (n={ok.sum()}): r={r_p:+.2f} p={p_p:.3f}  '
                   f'ρ={r_s:+.2f} p={p_s:.3f}')
            ax.text(0.5, 1.02, txt, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8.5, color='0.3')
            star = '*' if p_s < 0.05 else 'n.s.'
            ax.text(0.9, 0.93, star, transform=ax.transAxes, ha='center', va='top',
                    fontsize=22, fontweight='bold', color='k' if p_s < 0.05 else '0.55')
            ax.set_xlabel('Δ DPA choice-code depth')
            ax.set_ylabel(ylabel)
        axes[1].legend(frameon=False, fontsize=8, loc='upper left',
                       bbox_to_anchor=(1.01, 1), title='mouse (● Jaws / ▲ ChR / ■ ACC)',
                       title_fontsize=8)
        fig.suptitle(f'Learning (Expert−Naive): Δ depth vs Δ performance  '
                     f'({train_tag}, late delay {BINS_LATE[0]}–{BINS_LATE[-1]}, correct)',
                     fontsize=11, y=1.02)
        fig.tight_layout()
        stem = f'{DUM}_{train_tag}_dpa_panel'
        fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f'saved {os.path.join(fig_dir, "png", stem + ".png")}')

      # ── AB twin: treat odor-A and odor-B samples as INDEPENDENT points ─────────
      #   Same styling as _dpa_panel, but each mouse contributes TWO dots (sample A
      #   and sample B), doubling n (9→18). x = per-sample Δ DPA depth; y = per-sample
      #   Δ accuracy. Marker encodes sample (● A / ▲ B), color still encodes mouse; the
      #   two dots of a mouse are joined by a thin line. Stats are over all 2×9 points.
      with plt.rc_context(E_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
        specs = [(delta_dpa_perf_sample, 'Δ DPA accuracy (Exp−Naive)'),
                 (delta_gng_perf_sample, 'Δ GNG accuracy (Exp−Naive)')]
        ally = np.array([d[(m, c)] for d, _ in specs for m in ALL_MICE
                         for c, *_ in [(0,), (1,)]], float)
        ally = ally[~np.isnan(ally)]
        ypad = (ally.max() - ally.min()) * 0.15 or 0.05
        ylim = (ally.min() - ypad, ally.max() + ypad)
        for ax, (yv_dict, ylabel) in zip(axes, specs):
            xs, ys = [], []
            for mouse in ALL_MICE:
                pts_x, pts_y = [], []
                for cls_label, _, _, _ in SAMPLE_CLASSES:
                    xx = delta_choice_sample[(mouse, 'DPA', cls_label)]
                    yy = yv_dict.get((mouse, cls_label), np.nan)
                    pts_x.append(xx); pts_y.append(yy)
                    xs.append(xx); ys.append(yy)
                    if not (np.isnan(xx) or np.isnan(yy)):
                        # marker convention IDENTICAL to panel E: shape = opto-group
                        # (● Jaws / ▲ ChR / ■ ACC), fill = sample (A solid / B open)
                        face = MOUSE_COLOR[mouse] if cls_label == 0 else 'w'
                        ax.scatter(xx, yy, facecolors=face, edgecolors=MOUSE_COLOR[mouse],
                                   marker=GMARKER[GROUP[mouse]], s=90, linewidths=1.2,
                                   zorder=5,
                                   label=mouse if (ax is axes[1] and cls_label == 0)
                                   else None)
                ax.plot(pts_x, pts_y, '-', color=MOUSE_COLOR[mouse],
                        lw=0.8, alpha=0.5, zorder=3)
            xs = np.array(xs, float); ys = np.array(ys, float)
            regression_band(ax, xs, ys, color='0.25')
            ax.axhline(0, ls=':', color='k', lw=0.8)
            ax.axvline(0, ls=':', color='k', lw=0.8)
            ax.set_ylim(ylim)
            ok = ~(np.isnan(xs) | np.isnan(ys))
            r_p, p_p = pearsonr(xs[ok], ys[ok])
            r_s, p_s = spearmanr(xs[ok], ys[ok])
            txt = (f'A&B indep (n={ok.sum()}): r={r_p:+.2f} p={p_p:.3f}  '
                   f'ρ={r_s:+.2f} p={p_s:.3f}')
            ax.text(0.5, 1.02, txt, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8.5, color='0.3')
            star = '*' if p_s < 0.05 else 'n.s.'
            ax.text(0.9, 0.93, star, transform=ax.transAxes, ha='center', va='top',
                    fontsize=22, fontweight='bold', color='k' if p_s < 0.05 else '0.55')
            ax.set_xlabel('Δ DPA choice-code depth')
            ax.set_ylabel(ylabel)
        # sample fill (A solid / B open) + per-mouse colors — matches panel E
        sample_h = [mlines.Line2D([0],[0], marker='o', color='k', mfc='k', ls='none',
                                  ms=8, label='odor A (solid)'),
                    mlines.Line2D([0],[0], marker='o', color='k', mfc='w', ls='none',
                                  ms=8, label='odor B (open)')]
        mouse_h  = [mlines.Line2D([0],[0], marker='o', color=MOUSE_COLOR[m], ls='none',
                                  ms=8, label=m) for m in ALL_MICE]
        axes[1].legend(handles=sample_h + mouse_h, frameon=False, fontsize=8,
                       loc='upper left', bbox_to_anchor=(1.01, 1),
                       title='sample / mouse (● Jaws / ▲ ChR / ■ ACC)', title_fontsize=8)
        fig.suptitle(f'Learning (Expert−Naive), A&B independent: Δ depth vs Δ performance  '
                     f'({train_tag}, late delay {BINS_LATE[0]}–{BINS_LATE[-1]}, correct)',
                     fontsize=11, y=1.02)
        fig.tight_layout()
        stem = f'{DUM}_{train_tag}_dpa_panel_ab'
        fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f'saved {os.path.join(fig_dir, "png", stem + ".png")}')

print(f'\nScatter perf → {FIG_BASE}/')
