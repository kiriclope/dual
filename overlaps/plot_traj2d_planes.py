"""
Overlaps 2D code-plane trajectories — analog of the PCA `plot_pseudo_traj2d.py`.

The CCGD decoder yields three decision-function codes (sample / choice / test).
For each split this draws the three code planes
    sample x choice,  sample x test,  choice x test
as time-gradient paths with direction arrows + SEM-over-mice band.

Splits (one set of figures each):
    pair    AC/AD/BD/BC        per (stage, condition)
    sample  Odor A / B         per (stage, condition)
    choice  No lick / Lick     per (stage, condition)
    test    Odor C / D         per (stage, condition)
    task    DPA / Go / NoGo    per stage (conditions pooled / coloured by task)

Figures -> figures/overlaps/traj2d_planes/<train_tag>/<split>/{png,svg}/<stem>.*
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import plot_gradient_line, add_arrows, sem_band

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)

parser = argparse.ArgumentParser(
    description='Overlaps 2D code-plane trajectories (3 panels per figure).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--no-fold', dest='fold', action='store_false',
                    help='Disable folding: pooled axes (sample/choice/test) are '
                         'raw means and may cancel to ~0. Default folds each '
                         'pooled code by its label to preserve coding magnitude.')
parser.set_defaults(fold=True)
args = parser.parse_args()
FOLD_MODE = 'fold' if args.fold else 'nofold'

# ── config ────────────────────────────────────────────────────────────────────

DUM     = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN = '../data/overlaps'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

CODES = ['sample', 'choice', 'test']
PLANES = [('sample', 'choice'), ('sample', 'test'), ('choice', 'test')]
CODE_LABEL = {'sample': 'Sample code', 'choice': 'Choice code', 'test': 'Test code'}
# binary label column each code discriminates (used for folding)
CODE_VAR = {'sample': 'sample_odor', 'choice': 'choice', 'test': 'test_odor'}

_pal, _mut = sns.color_palette('tab10'), sns.color_palette('muted')

# split name → (column, [(value, label, color)...], pool_conditions, fold_codes)
# fold_codes: codes whose discriminated variable is POOLED by this split, so the
# code is folded (trial sign = 2*label-1) to keep A/B (etc.) from cancelling.
SPLITS = [
    ('pair',   'odor_pair',
     [(0, 'AC', _pal[0]), (1, 'AD', _pal[1]), (2, 'BD', _pal[2]), (3, 'BC', _pal[3])],
     False, set()),
    ('sample', 'sample_odor',
     [(0, 'Odor A', '#332288'), (1, 'Odor B', '#44AA99')],
     False, {'choice', 'test'}),
    ('choice', 'choice',
     [(0, 'No lick', '#377eb8'), (1, 'Lick', '#4daf4a')],
     False, {'sample', 'test'}),
    ('test',   'test_odor',
     [(0, 'Odor C', '#377eb8'), (1, 'Odor D', '#4daf4a')],
     False, {'sample', 'choice'}),
    ('task',   'tasks',
     [('DPA', 'DPA', _mut[3]), ('DualGo', 'Go', _mut[0]), ('DualNoGo', 'NoGo', _mut[2])],
     True, {'sample', 'choice', 'test'}),
]

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0, trials='',
    data_type='dF', prescreen=None, pval=0.05, preprocess=None,
    scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
    scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL  = options['bins_BL']
TRAJ_END = options['bins_TEST'][-1] + 1

TRAIN_EPOCHS = [
    ('trainTEST',   options['bins_TEST']),
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
]

# ── load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)   # (trials, 2, T_train, T_test)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X {X_single.shape}  y {y_single.shape}')

idx_correct = (
    (y_single.laser == 0) & (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
).values

MOUSE = {m: (y_single.mouse == m).values for m in ALL_MICE}
# per-trial fold sign (±1) for each code, from its discriminated label
SIGNS = {c: (2.0 * y_single[CODE_VAR[c]].values - 1.0) for c in CODES}
ONES = np.ones(len(y_single))


def per_mouse(trial_mask, signs):
    """{mouse: mean (folded) decision-function trajectory (T,)} over masked trials."""
    out = {}
    for mouse in ALL_MICE:
        m = trial_mask & MOUSE[mouse]
        if m.sum() == 0:
            continue
        out[mouse] = (X_ep[m] * signs[m][:, None]).mean(0)
    return out


def draw_planes(data, levels, fold, title, out_dir, stem):
    """data: {level_value: {code: {mouse: traj}}}; levels: [(value,label,color)].
    fold: set of codes drawn folded (axis label annotated)."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    lims = [[np.inf, -np.inf, np.inf, -np.inf] for _ in PLANES]

    for val, _label, color in levels:
        d = data[val]
        for j, (ta, tb) in enumerate(PLANES):
            da, db = d[ta], d[tb]
            mice = [m for m in ALL_MICE if m in da and m in db]
            if not mice:
                continue
            ax_arr = np.stack([da[m][:TRAJ_END] for m in mice], 0)
            ay_arr = np.stack([db[m][:TRAJ_END] for m in mice], 0)
            xm, ym = ax_arr.mean(0), ay_arr.mean(0)
            if len(mice) > 1:
                xs = ax_arr.std(0, ddof=1) / np.sqrt(len(mice))
                ys = ay_arr.std(0, ddof=1) / np.sqrt(len(mice))
                sem_band(axes[j], xm, ym, xs, ys, color)
            plot_gradient_line(axes[j], xm, ym, color)
            add_arrows(axes[j], xm, ym, color, n_arrows=3)
            lims[j][0] = min(lims[j][0], float(xm.min()))
            lims[j][1] = max(lims[j][1], float(xm.max()))
            lims[j][2] = min(lims[j][2], float(ym.min()))
            lims[j][3] = max(lims[j][3], float(ym.max()))

    for j, (ax, (ta, tb)) in enumerate(zip(axes, PLANES)):
        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_xlabel(CODE_LABEL[ta] + (' (folded)' if ta in fold else ''))
        ax.set_ylabel(CODE_LABEL[tb] + (' (folded)' if tb in fold else ''))
        xmin, xmax, ymin, ymax = lims[j]
        if np.isfinite(xmin):
            mx = 0.1 * (xmax - xmin or 1.0); my = 0.1 * (ymax - ymin or 1.0)
            ax.set_xlim(xmin - mx, xmax + mx); ax.set_ylim(ymin - my, ymax + my)
        ax.tick_params(labelsize=8)

    axes[-1].legend(
        handles=[Line2D([0], [0], color=c, lw=2, label=lab) for _, lab, c in levels],
        frameon=False, fontsize=8, loc='best', labelspacing=0.3)
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    for sub in ('png', 'svg'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'png', f'{stem}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'svg', f'{stem}.svg'), bbox_inches='tight')
    plt.close(fig)


# ── main loop ─────────────────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    print(f'\n=== {train_tag} ===')

    # decision function over T_test for this train epoch; per-mouse BL-std norm
    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        sd = X_ep[MOUSE[mouse]][:, BINS_BL].std()
        if sd > 0:
            X_ep[MOUSE[mouse]] /= sd

    for split_name, col, levels, pool, fold in SPLITS:
        fold = fold if args.fold else set()
        col_vals = y_single[col].values
        out_dir = f'figures/overlaps/traj2d_planes/{FOLD_MODE}/{train_tag}/{split_name}'
        for stage in STAGES:
            stage_m = (y_single.stage == stage).values
            conds = [None] if pool else CONDITIONS
            for cond in conds:
                base = idx_correct & stage_m
                if cond is not None:
                    base = base & (y_single.tasks == cond).values
                data = {}
                for val, _lab, _c in levels:
                    lvl = base & (col_vals == val)
                    data[val] = {
                        code: per_mouse(lvl & (y_single.target == code).values,
                                        SIGNS[code] if code in fold else ONES)
                        for code in CODES}
                stem = stage if cond is None else f'{stage}_{cond}'
                title = (f'{split_name} — {stage}' if cond is None
                         else f'{cond.replace("Dual", "")} — {stage}')
                draw_planes(data, levels, fold, title, out_dir, stem)
        print(f'  {split_name} done')

print('\nAll done.')
