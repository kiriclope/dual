"""
fig_overlaps_main_native.py — the overlaps MAIN paper figure (A&B-independent "--ab"
variant), COMPOSED NATIVELY as one matplotlib gridspec, Nature-Neuroscience-styled.

Layout (4-row gridspec, print-scale typography ~7 pt):
  A  1-D codes over the trial, Naive (top) vs Expert (bottom) — sample / choice / test / task;
     y shared within each code column so learning shrinkage is visible. A 3rd row adds a per-mouse
     neural d′ scatter (Naive x vs Expert y, unity = unchanged) for each code, computed on the same
     trainLD_TEST readout over the code's window (sample/choice=bins_LD, test=bins_TEST, task=Go/NoGo
     at bins_MD), like the opto figure's d′ panels.                       (← fig_overlaps_codes_1d.py)
  B  the no-lick push: DPA state Naive→Expert in the sample × choice plane (its OWN full row),
     with choice-code distribution strips; axes carry pole labels (odor A/B, no-lick/lick).
     Trajectories and KDE both stop before test onset (bins 0–53), so B is a pure pre-test
     delay portrait matching C/D. A 5th sub-panel is a D-style paired plot of per-mouse late-delay
     choice-code depth, Naive→Expert (stage on x, panel-C per-mouse colours, sample A filled / B open,
     pairs joined, black mean±SEM bars). Stat = deepening mixed model depth ~ stage + sample +
     (1|mouse) (β=−0.68, p=0.023).                                     (← plot_traj2d.py --all --dpa-only)
  C  Δ depth vs Δ performance (Expert−Naive), A&B-independent: ΔDPA (sig `*`) & ΔGNG (null).
     Stat = mouse-respecting MIXED MODEL (Δperf ~ Δdepth + (1|mouse); ΔDPA β=−0.03 p=0.016) —
     NOT the pseudoreplicated n=18 correlation.                           (← plot_scatter_perf.py --dpa-panel)
  D  DPA choice-code depth on NONPAIRED trials, correct-rejection vs false-alarm, Naive, split
     by sample (AD=A, BC=B); per-mouse colours (panel-C palette), corr-rej filled / false-alarm open,
     black mean±SEM bars. Clean test of the well↔behaviour link: false alarms sit in shallower
     no-lick wells than correct rejections — significant & sample-A-specific (AD Δ≈−1.3 p≈.006,
     all 9 mice; BC null). Naive only (experts too rarely err).           (← fig_overlaps_depth_fa_cr.py)
C and D share the last row (so D is one C-panel wide). All helper computation is copied inline
(per repo convention); source scripts are untouched.

Output: figures/overlaps/main/{png,svg}/fig_overlaps_main_ab[_ldtest05].{png,svg}

Decoder training axis (--ldtest05): default trains on the full trainLD_TEST (bins 45–59);
--ldtest05 trains on the narrow LD/TEST boundary (last 0.5 s LD + first 0.5 s TEST = bins
51–56) and writes the _ldtest05 file. Depth readout stays at the broad late-delay (bins 27–53)
either way — during the delay, pre-test.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main_native.py [--ldtest05]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, linregress, ttest_rel, t as t_dist
import statsmodels.formula.api as smf
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import plot_mean_sem, plot_gradient_line, add_arrows, sem_band
from src.common.plot_utils import add_vlines

# ── Style ─────────────────────────────────────────────────────────────────────
# NB: importing src.common.plot_utils runs `sns.set_context("poster")` at module
# level, which inflates tick-mark size/width. Reset to 'notebook' (what the opto
# figure effectively uses) so ticks match — set_style/rcParams alone do NOT undo it.
sns.set_context('notebook')
sns.set_style('ticks')
plt.rcParams.update({          # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
_pal_muted = sns.color_palette('muted')
TITLE_FS = 8

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
BINS_LATE    = np.arange(27, 54)                                        # depth readout = broad late-delay (bins 27–53, source-script convention; ΔDPA sig)
# Decoder training axis. Default = full trainLD_TEST (bins 45–59, locked main figure).
# --ldtest05 = narrow LD/TEST boundary: last 0.5 s of LD + first 0.5 s of TEST (bins 51–56,
# 8.5–9.33 s), the convention in plot_scatter_perf/laser/traj2d/exp_nolick_push_stats.
LDTEST05 = '--ldtest05' in sys.argv[1:]
if LDTEST05:
    TRAIN_LDTEST = np.concatenate([options['bins_LD'][-3:], options['bins_TEST'][:3]])   # 51–59→51–56
    AXIS_LABEL, FILE_SUF = 'trainLDTEST05, bins 51–56', '_ldtest05'
else:
    TRAIN_LDTEST = np.concatenate([options['bins_LD'], options['bins_TEST']])            # 45–59
    AXIS_LABEL, FILE_SUF = 'trainLD_TEST, bins 45–59', ''
BINS_DELAY   = options['bins_DELAY']
TEST_ONSET   = options['bins_TEST'][0]
TRAJ_END     = TEST_ONSET                                               # stop B trajectories just before test onset (bins 0–53); KDE already uses bins_DELAY (18–53), pre-test
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

# Panel A uses the same per-mouse BL normalisation (bins_BL == BL_A == 0:11), then adds a
# per-code z-score at plot time — so it starts from X_bl directly (no separate copy).
df_A = X_bl

idx_laser   = (y.laser == 0)
idx_choice  = (y.target == 'choice')
idx_correct = idx_laser & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1))


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A d′ row — per-mouse neural d′ (Naive vs Expert) for each code, on the SAME
#   trainLD_TEST readout as the code traces (d′ is scale-invariant → BL-norm irrelevant).
#   d′ = (μ_pos − μ_neg)/σ_pooled of the decision function over the code's window, on
#   correct trials (matching the A traces). Windows: sample/choice = late delay (bins_LD),
#   test = test epoch (bins_TEST), task = Go vs NoGo at mid-delay (bins_MD, the cue).
# ══════════════════════════════════════════════════════════════════════════════
BINS_MD = options['bins_MD']
_vLD   = X_bl[:, options['bins_LD']].mean(1)
_vTEST = X_bl[:, options['bins_TEST']].mean(1)
_vMD   = X_bl[:, BINS_MD].mean(1)
#            (title,        target,  split col,     pos,       neg,        v,     dpa_only)
DPRIME_SPECS = [
    ('sample d′', 'sample', 'sample_odor', 1,        0,          _vLD,   True),
    ('choice d′', 'choice', 'choice',      1,        0,          _vLD,   True),
    ('test d′',   'test',   'test_odor',   1,        0,          _vTEST, True),
    ('task d′',   'choice', 'tasks',       'DualGo', 'DualNoGo', _vMD,   False),
]


def _code_dprime(v, target, col, pos, neg, mouse, stage, dpa_only):
    base = ((y.target == target) & (y.mouse == mouse) & (y.stage == stage) &
            (y.laser == 0) & (y.performance == 1)).values
    if dpa_only:
        base = base & (y.tasks == 'DPA').values
    a = v[base & (y[col].values == pos)]; b = v[base & (y[col].values == neg)]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


dpr = {}                                                               # title -> per-mouse Naive/Expert d′
for _title, _tgt, _col, _pos, _neg, _v, _dpaonly in DPRIME_SPECS:
    dN, dE, mice = [], [], []
    for mo in ALL_MICE:
        n = _code_dprime(_v, _tgt, _col, _pos, _neg, mo, 'Naive', _dpaonly)
        e = _code_dprime(_v, _tgt, _col, _pos, _neg, mo, 'Expert', _dpaonly)
        if np.isfinite(n) and np.isfinite(e):
            dN.append(n); dE.append(e); mice.append(mo)
    dpr[_title] = dict(naive=np.array(dN), expert=np.array(dE), mice=mice)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Δdepth ↔ Δperf, A&B-independent (plot_scatter_perf.py --dpa-panel AB twin)
#   depth deltas on idx_correct, per sample class; perf deltas per sample class.
#   Headline stat = mixed model Δperf ~ Δdepth + (1|mouse) (respects the 2-obs/mouse
#   clustering; the raw n=18 correlation is pseudoreplicated — do NOT report it).
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


def _panelC_lmm(perf_dict):
    # Δperf ~ Δdepth + (1|mouse) over the 18 (mouse × sampleclass) observations.
    rows = [dict(mouse=mo, dd=delta_choice_sample[(mo, cls)], dp=perf_dict[(mo, cls)])
            for mo in ALL_MICE for cls, _ in D_SAMPLE_CLASSES]
    d = pd.DataFrame(rows).dropna()
    fit = smf.mixedlm('dp ~ dd', d, groups=d['mouse']).fit()
    return float(fit.params['dd']), float(fit.pvalues['dd']), d['mouse'].nunique(), len(d)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — DPA choice-code depth on NONPAIRED trials: correct-rejection vs
#   false-alarm, Naive, sample-discriminated (AD = sample A, BC = sample B).
#   The clean test of the no-lick well ↔ behaviour link: on nonmatch (nonpaired)
#   trials the animal should WITHHOLD; a deep well → correct rejection, a shallow well
#   → the animal licks → false alarm. Depth read is identical to panel C (same
#   TRAIN_LDTEST axis, BINS_LATE window); trials split by the y.response signal-
#   detection label instead of collapsed to correct-only.
#   NAIVE ONLY: false alarms are plentiful when naive (all 9 mice clear the ≥MIN_TR
#   bar) whereas experts rarely err (4/9, uninterpretable). Sample split removes the
#   sample→depth bias (the reason the main figure keeps A/B apart). Unit = mouse
#   (paired-t). Effect is sample-A-specific and robust across training axes (AD p≈.006,
#   false alarms in shallower wells); sample B is null (documented A/B asymmetry).
# ══════════════════════════════════════════════════════════════════════════════
MIN_TR      = 3
depth_trial = X_bl[:, BINS_LATE].mean(1)                               # (n,) per-trial depth
base_dpa_ch = ((y.laser == 0) & (y.tasks == 'DPA') & (y.target == 'choice')).values
op_arr      = y.odor_pair.values
resp_arr    = y.response.values
FA_CR_SPEC  = [('AD', 'A', 1, '#332288'), ('BC', 'B', 3, '#44AA99')]  # (pair, sample, odor_pair, colour)


def _facr_cell(mouse, odor_pair, r):
    m = (base_dpa_ch & (y.mouse == mouse).values & (y.stage == 'Naive').values &
         (op_arr == odor_pair) & (resp_arr == r))
    return depth_trial[m]


facr = {}                                                             # pair -> per-mouse cr/fa arrays
for lab, samp, odor_pair, col in FA_CR_SPEC:
    va, vb, used = [], [], []
    for mouse in ALL_MICE:
        a, b = _facr_cell(mouse, odor_pair, 'correct_rej'), _facr_cell(mouse, odor_pair, 'incorrect_fa')
        if len(a) >= MIN_TR and len(b) >= MIN_TR:
            va.append(a.mean()); vb.append(b.mean()); used.append(mouse)
    facr[lab] = dict(cr=np.array(va), fa=np.array(vb), used=used)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B data — per-mouse per-odor-pair sample(x)/choice(y) trajectories (traj2d)
# ══════════════════════════════════════════════════════════════════════════════
PAIR_LABELS = {0: 'AC', 1: 'AD', 2: 'BD', 3: 'BC'}
# colour by SAMPLE family (consistent with A & D): sample A = indigo shades, B = teal shades;
# the two within-sample pairs differ by shade (dark = C-test, light = D-test).
PAIR_COLOR  = {0: '#2A1A70', 1: '#8478C4',      # AC, AD  (sample A)
               2: '#2E8B79', 3: '#8ACcbc'}      # BD, BC  (sample B)
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


# per-mouse late-delay choice-code depth (SAME BINS_LATE window as C/D → the same "depth"
# quantity), per stage & sample — quantifies the Naive→Expert push the KDE strips show.
def _mouse_depth_B(stage, odor_pairs):
    out = {}
    for mouse in ALL_MICE:
        base = ((y.mouse == mouse) & (y.tasks == 'DPA') & (y.stage == stage) &
                (y.target == 'choice') & y.odor_pair.isin(odor_pairs)).values & idx_trials_B
        out[mouse] = X_bl[base][:, BINS_LATE].mean() if base.sum() else np.nan
    return out


pushB = {}                                                             # sample -> per-mouse Naive/Expert depth
for _slab, _pairs in [('A', [0, 1]), ('B', [2, 3])]:
    dN, dE = _mouse_depth_B('Naive', _pairs), _mouse_depth_B('Expert', _pairs)
    _mice = [m for m in ALL_MICE if not (np.isnan(dN[m]) or np.isnan(dE[m]))]
    pushB[_slab] = dict(naive=np.array([dN[m] for m in _mice]),
                        expert=np.array([dE[m] for m in _mice]), mice=_mice)


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
    ('test',   'test',   'test_odor',   [0, 1], ['Odor C', 'Odor D'], ['#CC6677', '#999933'], True),
    ('task',   'choice', 'tasks', ['DPA', 'DualGo', 'DualNoGo'], ['DPA', 'Go', 'NoGo'],
     [_pal_muted[3], _pal_muted[0], _pal_muted[2]], False),
]


def _setup_A(ax, ylab):
    add_vlines(ax, if_dpa=0)
    ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
    ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
    ax.set_ylabel(ylab, fontsize=8); ax.tick_params(labelsize=7)


def _draw_codes_row(axes_row, base, stage_label, show_titles, show_xlabel):
    for c, (ttl, code, col, levels, labs, cols, dpa_only) in enumerate(VARS_A):
        ax = axes_row[c]
        ylab = (f'{stage_label}\ncode (z)' if stage_label else 'code (z)') if c == 0 else ''
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
        if show_titles:                                                # top (Naive) row only
            t = f'{ttl} code' + (' (DPA)' if dpa_only else '')
            ax.set_title(t, fontsize=8)
            ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.2)


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
fig = plt.figure(figsize=(10.0, 11.4))
gs = fig.add_gridspec(5, 12, height_ratios=[0.85, 0.85, 1.25, 1.7, 1.55],
                      hspace=0.6, wspace=0.9,
                      left=0.06, right=0.985, top=0.965, bottom=0.04)


def panel_letter(ax, L, x=0.008, dy=0.014):
    # fixed left-margin x (box_aspect panels shrink/centre their axes, so ax.x0 is
    # unreliable); y tracks the panel top.
    p = ax.get_position()
    fig.text(x, p.y1 + dy, L, fontsize=11, fontweight='bold', va='top', ha='left')


# ── A: 2×4 code grid (Naive top, Expert bottom) ────────────────────────────────
axA = np.empty((2, 4), dtype=object)
for c in range(4):
    axA[0, c] = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
    axA[1, c] = fig.add_subplot(gs[1, 3 * c:3 * c + 3], sharex=axA[0, c], sharey=axA[0, c])
for ri, STG in enumerate(STAGES):
    b = ((y.laser == 0) & (y.learning == STG) & (y.performance == 1)).to_numpy()
    _draw_codes_row(axA[ri], b, stage_label=STG, show_titles=(ri == 0), show_xlabel=(ri == 1))

# ── A d′ row: per-mouse d′ (Naive x vs Expert y) for each code — unity = unchanged ──
axA_dp = []
for c, (_title, *_) in enumerate(DPRIME_SPECS):
    axd = fig.add_subplot(gs[2, 3 * c:3 * c + 3])
    axA_dp.append(axd)
    P = dpr[_title]
    _av = np.concatenate([P['naive'], P['expert']]) if len(P['naive']) else np.array([0.0, 1.0])
    _lo, _hi = float(np.nanmin(_av)), float(np.nanmax(_av)); _pd = (_hi - _lo) * 0.12 or 0.2
    _lim = (min(_lo - _pd, -0.1), _hi + _pd)
    axd.plot(_lim, _lim, ls='--', color='0.6', lw=0.8, zorder=1)                   # unity = unchanged
    axd.axhline(0, ls=':', color='0.8', lw=0.6, zorder=0); axd.axvline(0, ls=':', color='0.8', lw=0.6, zorder=0)
    for mo, xn, ye in zip(P['mice'], P['naive'], P['expert']):
        axd.scatter(xn, ye, s=26, facecolors=MOUSE_COLOR[mo], edgecolors=MOUSE_COLOR[mo],
                    linewidths=0.6, zorder=4)
    _n = len(P['naive'])
    _tp = float(ttest_rel(P['expert'], P['naive']).pvalue) if _n >= 3 else np.nan
    _dm = float((P['expert'] - P['naive']).mean()) if _n else np.nan
    _sg = (_tp == _tp and _tp < 0.05)
    axd.set_xlim(_lim); axd.set_ylim(_lim); axd.set_box_aspect(1)
    axd.set_title(_title, fontsize=8)
    axd.set_xlabel('Naive d′', fontsize=7.5)
    if c == 0:
        axd.set_ylabel('Expert d′', fontsize=7.5)
    axd.text(0.06, 0.94, '*' if _sg else 'n.s.', transform=axd.transAxes, ha='left', va='top',
             fontsize=11 if _sg else 7, fontweight='bold', color='k' if _sg else '0.55')
    axd.text(0.5, 0.02, f'Δ={_dm:+.2f}\np={_tp:.3f}', transform=axd.transAxes, ha='center', va='bottom',
             fontsize=6, color='0.3')
    print(f"A d′[{_title.strip()}] Naive→Expert Δ={_dm:+.3f} paired-t p={_tp:.3f} n={_n}")

# ── B: no-lick push planes (Naive | Expert) + choice-dist strips — left half ───
gsB = gs[3, 0:12].subgridspec(1, 5, width_ratios=[5, 1.2, 5, 1.2, 4.4], wspace=0.3)   # B full row: Naive traj|kde, Expert traj|kde, push scatter
xlimB, ylimB = (-4, 4), (-2, 6)
axB_traj, axB_hist = [], []
ax0 = None
for ci, stage in enumerate(STAGES):
    at = fig.add_subplot(gsB[0, ci * 2])
    ah = fig.add_subplot(gsB[0, ci * 2 + 1], sharey=at)
    ax0 = at if ax0 is None else ax0
    _draw_traj_B(at, stage, xlimB, ylimB)
    at.set_title(stage, pad=4, fontsize=TITLE_FS)
    at.set_xlabel('Sample code\n← odor A            odor B →')
    if ci == 0:
        at.set_ylabel('Choice code\n← no lick            lick →')
    _draw_hist_B(ah, stage, ylimB)                                     # sample A/B colour = indigo/teal (per pair legend)
    axB_traj.append(at); axB_hist.append(ah)
pair_handles = [Line2D([0], [0], color=PAIR_COLOR[p], lw=2.0, label=PAIR_LABELS[p]) for p in PAIR_LABELS]
axB_traj[-1].legend(handles=pair_handles, frameon=False, loc='upper right',
                    handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3, fontsize=8)

# ── B depth panel: per-mouse late-delay choice-code depth, Naive → Expert (D-style paired plot) ──
#    Stage on x, per-mouse colour (panel-C palette), sample A filled / B open, each mouse's Naive→
#    Expert pair joined, black group mean±SEM bars. Stat = deepening mixed model
#    depth ~ stage + sample + (1|mouse) (same estimator family as C).
axB_sc = fig.add_subplot(gsB[0, 4])
GX_B = (0.0, 1.0)                                                                  # Naive, Expert x-positions
for _slab, _fill in (('A', True), ('B', False)):
    P = pushB[_slab]
    for mo, xn, ye in zip(P['mice'], P['naive'], P['expert']):
        _mc = MOUSE_COLOR[mo]
        axB_sc.plot(GX_B, [xn, ye], '-', color=_mc, lw=0.7, alpha=0.5, zorder=2)
        axB_sc.scatter(GX_B[0], xn, s=30, zorder=3, linewidths=1.0,
                       facecolors=_mc if _fill else 'w', edgecolors=_mc)
        axB_sc.scatter(GX_B[1], ye, s=30, zorder=3, linewidths=1.0,
                       facecolors=_mc if _fill else 'w', edgecolors=_mc)
_naive_all = np.concatenate([pushB[s]['naive'] for s in ('A', 'B')])              # 18 mouse×sample obs / stage
_expert_all = np.concatenate([pushB[s]['expert'] for s in ('A', 'B')])
for _xx, _vals in ((GX_B[0], _naive_all), (GX_B[1], _expert_all)):
    _mu = _vals.mean(); _se = _vals.std(ddof=1) / np.sqrt(len(_vals))
    axB_sc.plot([_xx - 0.14, _xx + 0.14], [_mu, _mu], color='k', lw=1.8, zorder=4)
    axB_sc.errorbar(_xx, _mu, yerr=_se, color='k', capsize=2.5, lw=1.2, zorder=4)
axB_sc.axhline(0, ls=':', color='0.6', lw=0.7)
# stat: deepening mixed model depth ~ stage + sample + (1|mouse)
_dfp = pd.DataFrame([dict(mouse=mo, sample=_s, st=_st, depth=_v)
                     for _s in ('A', 'B') for _st, _k in ((0, 'naive'), (1, 'expert'))
                     for mo, _v in zip(pushB[_s]['mice'], pushB[_s][_k])])
_pfit = smf.mixedlm('depth ~ st + C(sample)', _dfp, groups=_dfp['mouse']).fit()
_bpush, _ppush = float(_pfit.params['st']), float(_pfit.pvalues['st'])
_nmB, _noB = _dfp['mouse'].nunique(), len(_dfp)
_sigB = _ppush < 0.05
print(f'B depth [mixed model, {_noB} obs] β={_bpush:+.3f} p={_ppush:.3f} ({_nmB} mice)')
axB_sc.set_xlim(-0.5, 1.5); axB_sc.set_xticks(GX_B); axB_sc.set_xticklabels(['Naive', 'Expert'])
axB_sc.set_box_aspect(1)
axB_sc.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=7.5)
axB_sc.set_title('Choice-code depth', loc='left', fontsize=TITLE_FS)
axB_sc.text(0.03, 0.03, f'mixed model ({_nmB} mice, {_noB} obs)\nβ={_bpush:+.3f}, p={_ppush:.3f}',
            transform=axB_sc.transAxes, ha='left', va='bottom', fontsize=6.5, color='0.3')
axB_sc.text(0.06, 0.96, '*' if _sigB else 'n.s.', transform=axB_sc.transAxes, ha='left', va='top',
            fontsize=12 if _sigB else 8, fontweight='bold', color='k' if _sigB else '0.55')   # C-style marker
axB_sc.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
                       mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')],
              frameon=False, loc='upper right', fontsize=6.5, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.3)

# ── C: Δdepth ↔ Δperf (Expert−Naive), A&B independent (ΔDPA | ΔGNG) — right half ─
gsC = gs[4, 0:8].subgridspec(1, 2, wspace=0.55)                        # C + D share the last row
axC = [fig.add_subplot(gsC[0, 0]), fig.add_subplot(gsC[0, 1])]
C_specs = [(delta_dpa_perf_sample, 'Δ DPA accuracy (Exp−Naive)', 'Δ depth vs Δ DPA accuracy',
            _panelC_lmm(delta_dpa_perf_sample)),
           (delta_gng_perf_sample, 'Δ GNG accuracy (Exp−Naive)', 'Δ depth vs Δ GNG accuracy',
            _panelC_lmm(delta_gng_perf_sample))]
_allyC = np.array([d[(m, c)] for d, _, _, _ in C_specs for m in ALL_MICE for c in (0, 1)], float)
_allyC = _allyC[~np.isnan(_allyC)]
_padC = (_allyC.max() - _allyC.min()) * 0.15 or 0.05
ylimC = (_allyC.min() - _padC, _allyC.max() + _padC)
for ax, (yv_dict, ylabel, msg, (beta, pv, n_mice, n_obs)) in zip(axC, C_specs):
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
                           marker='o', s=42, linewidths=1.0, zorder=5)
        ax.plot(px, py, '-', color=MOUSE_COLOR[mouse], lw=0.7, alpha=0.5, zorder=3)
    xs = np.array(xs, float); ys = np.array(ys, float)
    regression_band(ax, xs, ys)
    ax.axhline(0, ls=':', color='k', lw=0.7); ax.axvline(0, ls=':', color='k', lw=0.7)
    ax.set_ylim(ylimC)
    sig = pv < 0.05
    ax.text(0.03, 0.03, f'mixed model ({n_mice} mice, {n_obs} obs)\nβ={beta:+.3f}, p={pv:.3f}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color='0.3')
    ax.text(0.92, 0.94, '*' if sig else 'n.s.', transform=ax.transAxes, ha='center', va='top',
            fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    ax.set_xlabel('Δ DPA choice-code depth'); ax.set_ylabel(ylabel)
    ax.set_title(msg, loc='left', fontsize=TITLE_FS)
    ax.set_box_aspect(1)
    print(f'C[{ylabel[:6]}] mixed model β={beta:+.3f} p={pv:.3f} ({n_mice} mice, {n_obs} obs)')
_C_leg = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
          mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')]
axC[0].legend(handles=_C_leg, frameon=False, loc='upper center', bbox_to_anchor=(0.42, 1.0),
              ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)

# ── D: Naive nonpaired corr-rej vs false-alarm depth, sample A | sample B ────────
axD = fig.add_subplot(gs[4, 8:12])
GX_FACR = {'AD': (0.0, 0.8), 'BC': (1.9, 2.7)}
for lab, samp, odor_pair, col in FA_CR_SPEC:
    xc, xe = GX_FACR[lab]; r = facr[lab]
    for ya, yb, mouse in zip(r['cr'], r['fa'], r['used']):
        mc = MOUSE_COLOR[mouse]                                                   # per-mouse colour (panel-C palette)
        axD.plot([xc, xe], [ya, yb], '-', color=mc, lw=0.7, alpha=0.5, zorder=2)
        axD.scatter(xc, ya, s=30, facecolors=mc, edgecolors=mc, linewidths=1.0, zorder=3)   # corr-rej = filled
        axD.scatter(xe, yb, s=30, facecolors='w', edgecolors=mc, linewidths=1.1, zorder=3)  # false-alarm = open
    for xx, vals in ((xc, r['cr']), (xe, r['fa'])):
        if len(vals):
            mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            axD.plot([xx - 0.18, xx + 0.18], [mu, mu], color='k', lw=1.8, zorder=4)
            axD.errorbar(xx, mu, yerr=se, color='k', capsize=2.5, lw=1.2, zorder=4)
    n = len(r['cr']); d_mean = float((r['cr'] - r['fa']).mean()) if n else np.nan
    tp = float(ttest_rel(r['cr'], r['fa']).pvalue) if n >= 3 else np.nan
    sig = (tp == tp and tp < 0.05)
    axD.text((xc + xe) / 2, 0.99, f'{lab} (sample {samp})', transform=axD.get_xaxis_transform(),
             ha='center', va='top', fontsize=7, fontweight='bold', color=col)
    axD.text((xc + xe) / 2, 0.88, '*' if sig else 'n.s.', transform=axD.get_xaxis_transform(),   # C-style marker
             ha='center', va='top', fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    axD.text((xc + xe) / 2, 0.02, f'p={tp:.3f}', transform=axD.get_xaxis_transform(),
             ha='center', va='bottom', fontsize=6.5, color='0.3')
    print(f'D(FA/CR)[Naive {lab} sample {samp}] Δ(cr−fa)={d_mean:+.3f} paired-t p={tp:.3f} n={n}')
axD.axhline(0, ls=':', color='0.6', lw=0.7)
axD.set_xticks([0.0, 0.8, 1.9, 2.7])
axD.set_xticklabels(['corr.\nrej.', 'false\nalarm', 'corr.\nrej.', 'false\nalarm'], fontsize=6.5)
axD.set_xlim(-0.5, 3.2)
axD.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=7.5)
axD.set_title('Naive nonpaired trials', loc='left', fontsize=TITLE_FS)
axD.set_box_aspect(1)

# ── panel letters ───────────────────────────────────────────────────────────────
panel_letter(axA[0, 0], 'A')
panel_letter(axB_traj[0], 'B')
panel_letter(axC[0], 'C')
panel_letter(axD, 'D', x=0.655)

OUT = 'figures/overlaps/main'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_main_ab{FILE_SUF}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
