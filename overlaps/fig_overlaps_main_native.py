"""
fig_overlaps_main_native.py — the overlaps MAIN paper figure (A&B-independent "--ab"
variant), COMPOSED NATIVELY as one matplotlib gridspec figure.

Replaces the old `fig_overlaps_main.py --ab` "layout proof" (which glued five
pre-rendered PNG strips top-to-bottom, aspect ~0.46, non-editable). Every panel here
is DRAWN from the data as real matplotlib subplots, in the style of
`fig_behavior_opto_main.py` (message-titled panels, panel_letter() helper, row
banners, PNG @300 dpi + editable SVG).

Arc — three panels on the LOCKED trainLD_TEST read-out axis (bins 45–59), late-delay
readout window BINS_LATE = arange(27,54). Layout: A spans the top two rows; B and C share
one row below it (B left half, C right half). Styled like fig_behavior_opto_main.py.

  A  sample / choice / test / task 1-D codes over the trial, Naive (top) vs Expert
     (bottom).                                    (logic ← fig_overlaps_codes_1d.py)
  B  the no-lick push: DPA state Naive→Expert in the sample × choice plane, with the
     choice-code distribution strips.             (logic ← plot_traj2d.py --all --dpa-only)
  C  Δ depth vs Δ performance (Expert−Naive), A&B-independent: ΔDPA (sig `*`) & ΔGNG (null,
     DPA-specific).                               (logic ← plot_scatter_perf.py --dpa-panel, AB twin)

The old panel C (well deepening, exp_nolick_push_stats.py) and panel E (laser ON−OFF
causal analog, plot_scatter_laser.py) were removed at the user's request; the surviving
learning scatter was relabelled C. The laser causal analog lives in fig_behavior_opto_main.py.

All helper computation is copied inline (per repo convention); the source scripts are
untouched. Reusable plotting primitives (plot_mean_sem, sem_band, plot_gradient_line,
add_arrows, add_vlines) are imported from src — not modified.

Output: figures/overlaps/main/{png,svg}/fig_overlaps_main_ab.{png,svg}  (overwrites the
        deliverable named by the old assembler).

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main_native.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, pearsonr, spearmanr, linregress, t as t_dist
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import plot_mean_sem, plot_gradient_line, add_arrows, sem_band
from src.common.plot_utils import add_vlines

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_style('ticks')
plt.rcParams.update({          # same convention as fig_behavior_opto_main.py
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.9, 'lines.linewidth': 1.8,
})
_pal_muted = sns.color_palette('muted')
TITLE_FS = 10.5

# ── Config shared by every panel ───────────────────────────────────────────────
DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
GROUP   = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
           **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
_pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal_mice[i] for i, m in enumerate(ALL_MICE)}

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL      = options['bins_BL']
BINS_LATE    = np.arange(27, 54)                                        # late-delay readout window
TRAIN_LDTEST = np.concatenate([options['bins_LD'], options['bins_TEST']])  # 45–59 (locked axis)
BINS_DELAY   = options['bins_DELAY']
TRAJ_END     = options['bins_TEST'][-1] + 1
TEST_ONSET   = options['bins_TEST'][0]
xtime        = np.linspace(0, 14, 84)
BL_A         = slice(0, 12)                                             # codes_1d baseline slice


# ══════════════════════════════════════════════════════════════════════════════
# LOAD main (laser-off) tensor once; slice on the locked axis; free the 1.9 GB tensor
# ══════════════════════════════════════════════════════════════════════════════
print('loading main tensor …')
X = pkl_load(f'X_{DUM}',      path=DATA_IN)
y = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'  X {X.shape}  y {y.shape}')

raw = X[..., TRAIN_LDTEST, :].mean(-2)[:, 1].astype(float)             # (n,84) decision fn
del X                                                                  # free ~1.9 GB

# normalisation variant used by B/C/D (per-mouse BINS_BL std)
X_bl = raw.copy()
for mo in ALL_MICE:
    mm = (y.mouse == mo).values
    sd = X_bl[mm][:, BINS_BL].std()
    if sd > 0:
        X_bl[mm] /= sd

# normalisation variant used by A (per-mouse BL[0:12] std; per-code z applied at plot time)
df_A = raw.copy()
for mo in ALL_MICE:
    mm = (y.mouse == mo).values
    sd = df_A[mm][:, BL_A].std()
    if sd > 0:
        df_A[mm] /= sd

idx_laser   = (y.laser == 0)
idx_choice  = (y.target == 'choice')
idx_correct = idx_laser & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1))


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Δdepth ↔ Δperf, A&B-independent (plot_scatter_perf.py --dpa-panel AB twin)
#   depth deltas on idx_correct, per sample class; perf deltas per sample class.
# ══════════════════════════════════════════════════════════════════════════════
D_SAMPLE_CLASSES = [(0, [0, 1]), (1, [2, 3])]                           # (cls_label, odor_pairs)

delta_choice_sample = {}                                               # (mouse, cls) -> Δdepth (DPA)
for mouse in ALL_MICE:
    for cls, pairs in D_SAMPLE_CLASSES:
        vals = {}
        for stage in STAGES:
            m = ((y.mouse == mouse) & (y.tasks == 'DPA') & (y.stage == stage) &
                 (y.target == 'choice') & idx_correct & y.odor_pair.isin(pairs)).values
            vals[stage] = X_bl[m][:, BINS_LATE].mean() if m.sum() else np.nan
        delta_choice_sample[(mouse, cls)] = vals['Expert'] - vals['Naive']


def _perf_delta_by_sample(perf_col, task_mask):
    out = {}
    for mouse in ALL_MICE:
        for cls, pairs in D_SAMPLE_CLASSES:
            vals = {}
            for stage in STAGES:
                m = ((y.mouse == mouse) & (y.stage == stage) & idx_laser & idx_choice &
                     task_mask & y.odor_pair.isin(pairs))
                col = y.loc[m, perf_col].dropna()
                vals[stage] = col.mean() if len(col) else np.nan
            out[(mouse, cls)] = vals['Expert'] - vals['Naive']
    return out


delta_dpa_perf_sample = _perf_delta_by_sample('performance', y.tasks == 'DPA')
delta_gng_perf_sample = _perf_delta_by_sample('odr_perf',    y.tasks != 'DPA')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B data — per-mouse per-odor-pair sample(x)/choice(y) trajectories (traj2d)
# ══════════════════════════════════════════════════════════════════════════════
PAIR_LABELS = {0: 'AC', 1: 'AD', 2: 'BD', 3: 'BC'}
_pal_pairs  = sns.color_palette('tab10')
PAIR_COLOR  = {0: _pal_pairs[0], 1: _pal_pairs[1], 2: _pal_pairs[2], 3: _pal_pairs[3]}
SAMPLE_SPLITS_HIST = [('A', [0, 1], '#332288'), ('B', [2, 3], '#44AA99')]

idx_trials_B = idx_laser.values                                        # --all trial set


def _mouse_trajs_B(cond, stage, target, odor_pairs):
    trajs = []
    for mouse in ALL_MICE:
        base = ((y.mouse == mouse) & (y.tasks == cond) & (y.stage == stage) &
                (y.target == target) & y.odor_pair.isin(odor_pairs)).values & idx_trials_B
        if base.sum() == 0:
            continue
        trajs.append(X_bl[base].mean(0))                               # (84,)
    return trajs


trajB = {s: {} for s in STAGES}                                        # trajB[stage][pair] = (xs, ys)
for stage in STAGES:
    for pair_id in PAIR_LABELS:
        xs = _mouse_trajs_B('DPA', stage, 'sample', [pair_id])
        ys = _mouse_trajs_B('DPA', stage, 'choice', [pair_id])
        trajB[stage][pair_id] = (xs, ys)


def _draw_traj_B(ax, stage, xlim, ylim):
    for pair_id in PAIR_LABELS:
        xs, ys = trajB[stage][pair_id]
        if not xs or not ys:
            continue
        color = PAIR_COLOR[pair_id]
        arr_x = np.stack(xs, 0)[:, :TRAJ_END]
        arr_y = np.stack(ys, 0)[:, :TRAJ_END]
        n_mice = arr_x.shape[0]
        x_mean, y_mean = arr_x.mean(0), arr_y.mean(0)
        x_sem = arr_x.std(0, ddof=1) / np.sqrt(n_mice)
        y_sem = arr_y.std(0, ddof=1) / np.sqrt(n_mice)
        sem_band(ax, x_mean, y_mean, x_sem, y_sem, color)
        plot_gradient_line(ax, x_mean, y_mean, color)
        add_arrows(ax, x_mean, y_mean, color, n_arrows=3)
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-2, 0, 2, 4, 6])
    ax.tick_params(length=3, width=0.9)


def _draw_hist_B(ax_h, stage, ylim):
    y_grid = np.linspace(ylim[0], ylim[1], 300)
    handles = []
    for label, pairs, color in SAMPLE_SPLITS_HIST:
        vals = []
        for pair_id in pairs:
            for y_traj in trajB[stage][pair_id][1]:
                vals.extend(y_traj[BINS_DELAY].tolist())
        if len(vals) < 2:
            continue
        dens = gaussian_kde(vals, bw_method=0.4)(y_grid)
        ax_h.fill_betweenx(y_grid, 0, dens, color=color, alpha=0.35, lw=0)
        ax_h.plot(dens, y_grid, color=color, lw=1.2)
        ax_h.axhline(np.mean(vals), color=color, lw=1.4, ls='--', alpha=0.9, zorder=5)
        handles.append(Patch(facecolor=color, alpha=0.6, label=f'Sample {label}'))
    ax_h.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax_h.set_xlim(left=0)
    ax_h.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for sp in ('left', 'bottom', 'top'):
        ax_h.spines[sp].set_visible(False)
    return handles


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A data — 1-D codes; replicate fig_overlaps_codes_1d.grandmean_row
# ══════════════════════════════════════════════════════════════════════════════
# (title, target code, split column, levels, labels, colours, dpa_only)
VARS_A = [
    ('sample', 'sample', 'sample_odor', [0, 1], ['Odor A', 'Odor B'], ['#332288', '#44AA99'], True),
    ('choice', 'choice', 'choice',      [0, 1], ['No lick', 'Lick'],  ['#377eb8', '#4daf4a'], True),
    ('test',   'test',   'test_odor',   [0, 1], ['Odor C', 'Odor D'], ['#377eb8', '#4daf4a'], True),
    ('task',   'choice', 'tasks', ['DPA', 'DualGo', 'DualNoGo'], ['DPA', 'Go', 'NoGo'],
     [_pal_muted[3], _pal_muted[0], _pal_muted[2]], False),
]
TEST_VALID_A = np.max(TRAIN_LDTEST) >= TEST_ONSET                      # True on ld_test


def _setup_A(ax, ylab):
    add_vlines(ax, if_dpa=0)
    ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
    ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
    ax.set_ylabel(ylab, fontsize=9); ax.tick_params(labelsize=7)


def _draw_codes_row(axes_row, base, stage_label, show_titles, show_xlabel):
    for c, (ttl, code, col, levels, labs, cols, dpa_only) in enumerate(VARS_A):
        ax = axes_row[c]
        ylab = (f'{stage_label} — code' if stage_label else 'code') if c == 0 else ''
        _setup_A(ax, ylab)
        ax.set_xlabel('Time (s)' if show_xlabel else '', fontsize=9)
        pbase = base & (y.tasks == 'DPA').to_numpy() if dpa_only else base
        Zc = np.full_like(df_A, np.nan)
        for mo in ALL_MICE:                                            # per-mouse BL z of the code
            mm = (y.mouse == mo).to_numpy() & (y.target == code).to_numpy()
            z = df_A[mm]; z = z - z[:, BL_A].mean()
            Zc[mm] = z / (df_A[mm][:, BL_A].std() + 1e-9)
        for lv, lab, color in zip(levels, labs, cols):
            per_mouse = []
            for mo in ALL_MICE:
                s = (pbase & (y.target == code).to_numpy() &
                     (y.mouse == mo).to_numpy() & (y[col].to_numpy() == lv))
                if s.sum() >= 3:
                    per_mouse.append(np.nanmean(Zc[s], 0))
            if len(per_mouse) >= 2:
                M = np.stack(per_mouse, 0); n = M.shape[0]
                plot_mean_sem(ax, xtime, M.mean(0), M.std(0, ddof=1) / np.sqrt(n),
                              color, lw=1.6, label=f'{lab} (n={n})', zorder=2)
        if show_titles:
            t = f'{ttl} code' + (' (DPA)' if dpa_only else '')
            ax.set_title(t, fontsize=9.5)
        ax.legend(fontsize=6, frameon=False, loc='upper left')


# ── shared scatter helper ──────────────────────────────────────────────────────
def regression_band(ax, xs, ys, color='0.25', alpha=0.15):
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return
    xv, yv = xs[ok], ys[ok]
    slope, icpt, _, _, se = linregress(xv, yv)
    xl = np.linspace(xv.min(), xv.max(), 100)
    yl_ = slope * xl + icpt
    ssx = np.sum((xv - xv.mean()) ** 2)
    seb = se * np.sqrt(1 / len(xv) + (xl - xv.mean()) ** 2 / ssx)
    tc = t_dist.ppf(0.975, df=len(xv) - 2)
    ax.plot(xl, yl_, color=color, lw=1.5, zorder=4)
    ax.fill_between(xl, yl_ - tc * seb, yl_ + tc * seb, color=color, alpha=alpha, zorder=2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 8.4))
gs = fig.add_gridspec(3, 12, height_ratios=[1.0, 1.0, 2.05],
                      hspace=0.5, wspace=0.85,
                      left=0.05, right=0.985, top=0.9, bottom=0.065)


def panel_letter(ax, L, x=0.012, dy=0.020):
    # fixed left-margin x (box_aspect panels shrink/centre their axes, so ax.x0 is
    # unreliable); y tracks the panel top.
    p = ax.get_position()
    fig.text(x, p.y1 + dy, L, fontsize=15, fontweight='bold', va='top', ha='left')


# ── A: 2×4 code grid (Naive top, Expert bottom) ────────────────────────────────
axA = np.empty((2, 4), dtype=object)
for c in range(4):
    axA[0, c] = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
    axA[1, c] = fig.add_subplot(gs[1, 3 * c:3 * c + 3], sharex=axA[0, c])
for ri, STG in enumerate(STAGES):
    b = ((y.laser == 0) & (y.learning == STG) & (y.performance == 1)).to_numpy()
    _draw_codes_row(axA[ri], b, stage_label=STG, show_titles=(ri == 0), show_xlabel=(ri == 1))

# ── B: no-lick push planes (Naive | Expert) + choice-dist strips — left half ───
gsB = gs[2, 0:6].subgridspec(1, 4, width_ratios=[5, 1.1, 5, 1.1], wspace=0.06)
xlimB, ylimB = (-4, 4), (-2, 6)
axB_traj, axB_hist = [], []
ax0 = None
for ci, stage in enumerate(STAGES):
    at = fig.add_subplot(gsB[0, ci * 2])
    ah = fig.add_subplot(gsB[0, ci * 2 + 1], sharey=at)
    ax0 = at if ax0 is None else ax0
    _draw_traj_B(at, stage, xlimB, ylimB)
    at.set_title(stage, pad=5, fontsize=TITLE_FS)
    at.set_xlabel('Sample code')
    if ci == 0:
        at.set_ylabel('Choice code')
    hh = _draw_hist_B(ah, stage, ylimB)
    if ci == 0 and hh:
        ah.legend(handles=hh, frameon=False, fontsize=7, loc='upper right',
                  handlelength=0.8, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)
    axB_traj.append(at); axB_hist.append(ah)
pair_handles = [Line2D([0], [0], color=PAIR_COLOR[p], lw=2.0, label=PAIR_LABELS[p]) for p in PAIR_LABELS]
axB_traj[-1].legend(handles=pair_handles, frameon=False, loc='upper right',
                    handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3, fontsize=8)

# ── D: Δdepth ↔ Δperf (Expert−Naive), A&B independent (ΔDPA | ΔGNG) — right half ─
gsD = gs[2, 6:12].subgridspec(1, 2, wspace=0.5)
axD = [fig.add_subplot(gsD[0, 0]), fig.add_subplot(gsD[0, 1])]
D_specs = [(delta_dpa_perf_sample, 'Δ DPA accuracy (Exp−Naive)', 'Depth vs ΔDPA (learning)'),
           (delta_gng_perf_sample, 'Δ GNG accuracy (Exp−Naive)', 'DPA-specific (ΔGNG null)')]
_allyD = np.array([d[(m, c)] for d, _, _ in D_specs for m in ALL_MICE for c in (0, 1)], float)
_allyD = _allyD[~np.isnan(_allyD)]
_padD = (_allyD.max() - _allyD.min()) * 0.15 or 0.05
ylimD = (_allyD.min() - _padD, _allyD.max() + _padD)
for ax, (yv_dict, ylabel, msg) in zip(axD, D_specs):
    xs, ys = [], []
    for mouse in ALL_MICE:
        px, py = [], []
        for cls, pairs in D_SAMPLE_CLASSES:
            xx = delta_choice_sample[(mouse, cls)]
            yy = yv_dict.get((mouse, cls), np.nan)
            px.append(xx); py.append(yy); xs.append(xx); ys.append(yy)
            if not (np.isnan(xx) or np.isnan(yy)):
                face = MOUSE_COLOR[mouse] if cls == 0 else 'w'         # A solid / B open
                ax.scatter(xx, yy, facecolors=face, edgecolors=MOUSE_COLOR[mouse],
                           marker=GMARKER[GROUP[mouse]], s=80, linewidths=1.1, zorder=5)
        ax.plot(px, py, '-', color=MOUSE_COLOR[mouse], lw=0.8, alpha=0.5, zorder=3)
    xs = np.array(xs, float); ys = np.array(ys, float)
    regression_band(ax, xs, ys)
    ax.axhline(0, ls=':', color='k', lw=0.8); ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_ylim(ylimD)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    r_p, p_p = pearsonr(xs[ok], ys[ok]); r_s, p_s = spearmanr(xs[ok], ys[ok])
    ax.text(0.5, 0.02, f'A&B indep (n={ok.sum()}): r={r_p:+.2f} p={p_p:.3f}  ρ={r_s:+.2f} p={p_s:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7.5, color='0.3')
    ax.text(0.9, 0.93, '*' if p_s < 0.05 else 'n.s.', transform=ax.transAxes, ha='center',
            va='top', fontsize=18, fontweight='bold', color='k' if p_s < 0.05 else '0.55')
    ax.set_xlabel('Δ DPA choice-code depth'); ax.set_ylabel(ylabel)
    ax.set_title(msg, loc='left', fontweight='bold', fontsize=TITLE_FS)
    ax.set_box_aspect(1)
    print(f'D[{ylabel[:6]}] n={ok.sum()} r={r_p:+.2f} p={p_p:.3f}  rho={r_s:+.2f} p={p_s:.3f}')
_D_leg = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=7, label='odor A (solid)'),
          mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=7, label='odor B (open)')]
axD[0].legend(handles=_D_leg, frameon=False, fontsize=7, loc='upper left', handletextpad=0.3)

# ── panel letters + row banners ────────────────────────────────────────────────
panel_letter(axA[0, 0], 'A')
panel_letter(axB_traj[0], 'B')
panel_letter(axD[0], 'C', x=0.505)

fig.suptitle('Overlaps main figure (A&B-independent) — dual code · no-lick push · learning depth↔accuracy link '
             '(all trainLD_TEST, bins 45–59)', y=0.985, fontsize=12.5, fontweight='bold')

OUT = 'figures/overlaps/main'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_main_ab.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
