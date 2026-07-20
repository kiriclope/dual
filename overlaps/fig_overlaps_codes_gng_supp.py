"""SUPPLEMENT — the four main-figure panel-A codes, read separately on GO vs NoGo (Dual) trials.

Panel A = DualGo trials, Panel B = DualNoGo trials; within each, the 2×4 code grid (Naive top / Expert
bottom) with columns sample / GNG / test / DPA-action, each code split by its OWN contrast on that panel's
trials. Because a panel fixes the distractor, the GNG column shows a single trace (Go in A, NoGo in B) —
the Go↑/NoGo↓ comparison is BETWEEN the panels. The DPA-action column reveals a distractor-lick peak
(~6.5 s) that appears on Go trials (A) but not NoGo trials (B), on top of the shared DPA test-lick peak.

Same projections / pooled-evoked normalisation / house style as the main figure.
Output: figures/overlaps/controls/{png,svg}/overlaps_codes_gng_trials.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from src.pca.io import pkl_load
from src.common.options import set_options
from src.plot.traj import plot_mean_sem
from src.common.plot_utils import add_vlines

# ── house style (matches fig_overlaps_main_native.py; see CLAUDE.md) ──
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
options = set_options()
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
_BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
xtime = np.linspace(0, 14, 84); BL_N = np.arange(0, 12)
_ACT_DPA = np.arange(57, 63); _GNG_WIN = np.asarray(options['bins_MD'])

# ── load tensor + build the four pooled-evoked-normalised codes ──
Xb = pkl_load(f'X_{_BDUM}', path='../data/overlaps'); yb = pkl_load(f'labels_{_BDUM}', path='../data/overlaps')
_sct = (yb.target != 'gng').to_numpy(); X = Xb[_sct]; y = yb[_sct].reset_index(drop=True)
_sam = (y.target == 'sample').to_numpy(); _tst = (y.target == 'test').to_numpy(); _cho = (y.target == 'choice').to_numpy()
SAMPLE_R = X[_sam][:, 1, np.arange(16, 48), :].mean(1).astype(float); Y_SAM = y[_sam].reset_index(drop=True)
TEST_R = X[_tst][:, 1, np.arange(58, 84), :].mean(1).astype(float);   Y_TST = y[_tst].reset_index(drop=True)
LICK_R = X[_cho][:, 1, _ACT_DPA, :].mean(1).astype(float);            Y_LCK = y[_cho].reset_index(drop=True)
del X
_gm = (yb.target == 'gng').to_numpy(); Xg = Xb[_gm]; Y_GNG = yb[_gm].reset_index(drop=True)
GNG_R = Xg[:, 1, _GNG_WIN, :].mean(1).astype(float); del Xg, Xb


def _norm_code(Draw, YY, class_col, cls1, task):
    Z = np.full_like(Draw, np.nan)
    tk = (YY.tasks == 'DPA').to_numpy() if task == 'DPA' else (YY.tasks != 'DPA').to_numpy()
    for mo in ALL_MICE:
        mm = (YY.mouse == mo).to_numpy(); pool = mm & (YY.laser == 0).to_numpy() & tk
        if pool.sum() == 0:
            continue
        s = np.where(YY[class_col].to_numpy()[pool] == cls1, 1.0, -1.0)
        vbar = (s[:, None] * Draw[pool]).mean(0); sd = (vbar - vbar[BL_N].mean()).std()
        Z[mm] = (Draw[mm] - Draw[pool][:, BL_N].mean()) / (sd + 1e-9)
    return Z


SAMPLE_D = _norm_code(SAMPLE_R, Y_SAM, 'sample_odor', 1, 'DPA')
TEST_D = _norm_code(TEST_R, Y_TST, 'test_odor', 1, 'DPA')
LICK_D = _norm_code(LICK_R, Y_LCK, 'choice', 1, 'DPA')
GNG_D = _norm_code(GNG_R, Y_GNG, 'gng', 1, 'Dual')

# column order: sample, GNG, test, DPA-action.  spec = (title, D, YY, split_col, levels, labs, colours)
SPECS = [
    ('sample',        SAMPLE_D, Y_SAM, 'sample_odor', [0, 1], ['Odor A', 'Odor B'], ['#332288', '#44AA99']),
    ('GNG\n(memory)', GNG_D,    Y_GNG, 'gng',         [0, 1], ['NoGo', 'Go'],       ['#2ca02c', '#1f77b4']),
    ('test',          TEST_D,   Y_TST, 'test_odor',   [0, 1], ['Odor C', 'Odor D'], ['#CC6677', '#999933']),
    ('DPA action',    LICK_D,   Y_LCK, 'choice',      [0, 1], ['No lick', 'Lick'],  ['#377eb8', '#4daf4a']),
]
PANELS = [('A', 'Go trials', 'DualGo'), ('B', 'NoGo trials', 'DualNoGo')]


def draw(ax, spec, stage, taskval, ylab, show_title, show_xlabel):
    ttl, D, YY, col, levels, labs, cols = spec
    add_vlines(ax, if_dpa=0)                                          # Dual-task epoch markers (incl. distractor)
    ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
    ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
    ax.set_ylabel(ylab); ax.set_xlabel('Time (s)' if show_xlabel else '')
    base = ((YY.laser == 0) & (YY.learning == stage) & (YY.performance == 1) & (YY.tasks == taskval)).to_numpy()
    for lv, lab, color in zip(levels, labs, cols):
        per_mouse = [np.nanmean(D[base & (YY.mouse == mo).to_numpy() & (YY[col].to_numpy() == lv)], 0)
                     for mo in ALL_MICE
                     if (base & (YY.mouse == mo).to_numpy() & (YY[col].to_numpy() == lv)).sum() >= 3]
        if len(per_mouse) >= 2:
            M = np.stack(per_mouse, 0); n = M.shape[0]
            plot_mean_sem(ax, xtime, M.mean(0), M.std(0, ddof=1) / np.sqrt(n), color, lw=1.6, label=f'{lab} (n={n})', zorder=2)
    if show_title:
        ax.set_title(f'{ttl} code', fontsize=TITLE_FS)
        ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.2)


fig = plt.figure(figsize=(10.0, 9.2))
outer = fig.add_gridspec(2, 1, hspace=0.32)
ref_stim = ref_gng = None
for pi, (letter, ptitle, taskval) in enumerate(PANELS):
    gsP = outer[pi].subgridspec(2, 4, wspace=0.55, hspace=0.35)
    for ri, stage in enumerate(['Naive', 'Expert']):
        for ci, spec in enumerate(SPECS):
            shy = ref_gng if ci == 1 else ref_stim
            ax = fig.add_subplot(gsP[ri, ci], sharey=shy)
            if ci == 1 and ref_gng is None:
                ref_gng = ax
            if ci != 1 and ref_stim is None:
                ref_stim = ax
            ylab = f'{stage}\ncode (z)' if ci == 0 else ''
            draw(ax, spec, stage, taskval, ylab, show_title=(ri == 0), show_xlabel=(ri == 1))
            if ri == 0 and ci == 0:
                ax.text(-0.42, 1.18, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
                ax.text(-0.42, 1.02, ptitle, transform=ax.transAxes, fontsize=9, va='bottom', ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.99))
OUT = 'figures/overlaps/controls'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
p = f'{OUT}/png/overlaps_codes_gng_trials.png'
fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
