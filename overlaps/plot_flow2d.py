"""
Empirical flow field in the sample × choice code plane, built from the 2D
delay-period trajectories.

Why per-sample fields:
  Sample A and sample B trajectories both leave the origin at delay onset but
  move in OPPOSITE sample-code directions (A → negative, B → positive).  If
  their velocity vectors are pooled into one binned field they cancel near the
  centre, producing a spurious central fixed point and erasing the two real
  attractors at the A and B trajectory endpoints.  We therefore bin A and B
  separately and combine winner-take-all per cell (each cell keeps the velocity
  of whichever sample visits it more), then locate one attractor per sample.

For each (stage, condition) panel:
  - per-(mouse, odor_pair) mean trajectory over BINS_DELAY → (position, velocity)
  - bin velocities per sample identity (Nadaraya-Watson + Gaussian smoothing)
  - combine winner-take-all → single coherent field
  - pcolormesh: speed heatmap (magma), masked to visited cells
  - streamplot: white streamlines, zeroed outside visited cells
  - cyan circle per sample: speed minimum in that sample's late-delay region

Layout: 2 rows (Naive / Expert) × 3 cols (DPA / DualGo / DualNoGo).
One figure per train epoch.
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import (
    colored_path, truncate_cmap,
    velocity_points, transition_velocity_points,
    bin_velocity, raw_counts,
    panel_fields, draw_panel,
)

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

matplotlib.rcParams.update({
    'axes.titlesize':    18, 'axes.labelsize':  14,
    'xtick.labelsize':   11, 'ytick.labelsize': 11,
    'axes.titlepad':     10, 'axes.labelpad':   5,
    'font.size':         12,
})

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/flow2d'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']

# odor_pair → sample identity; colours/cmaps match plot_codes / figure2BC
SAMPLE_SPLITS = [
    ('A', [0, 1], '#332288', 'Purples'),
    ('B', [2, 3], '#44AA99', 'Greens'),
]

# Grid parameters
N_BINS        = 14       # bins per axis
SIGMA         = 1.2      # spatial Gaussian smoothing of the binned field (bins)
MIN_RAW_COUNT = 1        # a cell must be visited this many times to show flow

# Single-trial velocity de-noising
TRAJ_SMOOTH   = 0        # temporal Gaussian smoothing of each trajectory (bins); 0 = off
VEL_STEP      = 5        # finite-difference step k: vel = (x[t+k]-x[t]) / k

AXIS_PAD      = 0.4      # BL σ padding around the outermost fixed point

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_BL    = options['bins_BL']
BINS_DELAY = options['bins_DELAY']   # bins 18–53
# Late delay (last 40%) — restricts the attractor search to where the system
# settles, not the trajectory start (trivially slow before choice code ramps).
BINS_LATE  = BINS_DELAY[int(0.6 * len(BINS_DELAY)):]

CMAP_SPEED = matplotlib.colormaps['magma'].copy()   # slow = dark, fast = bright
CMAP_SPEED.set_bad('#f5f5f5')
CMAP_OCC = matplotlib.colormaps['Blues'].copy()
CMAP_OCC.set_bad('#f5f5f5')



TRAIN_EPOCHS = [
    ('trainDELAY',  options['bins_DELAY']),
    ('trainCHOICE', options['bins_CHOICE']),
    ('trainED',     options['bins_ED']),
    ('trainTEST',   options['bins_TEST']),
]

# Flow variants: (output subfolder, velocity-field bin segments).
FLOW_VARIANTS = [
    ('all_trials',    [BINS_DELAY]),
    ('all_trials_bl', [BINS_BL, BINS_DELAY]),
]

# Field modes: how the streamlines are computed.
#   velocity   — centred-difference drift E[v | x]  (current behaviour)
#   flux       — probability flux J = E[v|x] × ρ(x)
#   transition — forward-difference E[Δx | x], occupancy-normalised
FIELD_MODES = ['velocity', 'flux', 'transition']

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

idx_laser = (y_single.laser == 0)
xtime     = np.linspace(0, 14, X_single.shape[-1])

# ── Trajectory collection ─────────────────────────────────────────────────────

# Trial-invariant columns used to align sample- and choice-decoder rows so
# index-wise pairing matches the most similar trials.
_PAIR_KEY = ['response', 'performance', 'sample_odor', 'test_odor',
             'dist_odor', 'licks', 'day']


def single_trial_pairs(X_ep, cond, stage, pairs):
    """Per-TRIAL paired (sample_code, choice_code) trajectories.

    Within each (mouse, odor_pair) cell the sample- and choice-decoder rows are
    matched as a SET (not row-by-row) by sorting on trial-invariant columns.
    Uses all trials (correct AND incorrect — only laser-off filtering).
    Returns list of (sx, cy) tuples, each a (T,) single-trial array.
    """
    out = []
    for mouse in ALL_MICE:
        for op in pairs:
            base = (
                (y_single.mouse     == mouse) &
                (y_single.tasks     == cond)  &
                (y_single.stage     == stage) &
                (y_single.odor_pair == op)    &
                idx_laser
            )
            si = np.where((base & (y_single.target == 'sample')).values)[0]
            ci = np.where((base & (y_single.target == 'choice')).values)[0]
            if len(si) == 0 or len(ci) == 0:
                continue
            si = si[np.lexsort([y_single.iloc[si][c].values for c in _PAIR_KEY])]
            ci = ci[np.lexsort([y_single.iloc[ci][c].values for c in _PAIR_KEY])]
            n = min(len(si), len(ci))
            for a, b in zip(si[:n], ci[:n]):
                out.append((X_ep[a], X_ep[b]))
    return out


def group_mean_trajs(X_ep, cond, stage, pairs):
    """Per-(mouse, odor_pair) MEAN (sample_code, choice_code) trajectories.

    Averaging denoises positions so the binned velocity field is tangent to the
    average trajectory.  Uses all laser-off trials (correct and incorrect).
    """
    out = []
    for mouse in ALL_MICE:
        for op in pairs:
            base = (
                (y_single.mouse     == mouse) &
                (y_single.tasks     == cond)  &
                (y_single.stage     == stage) &
                (y_single.odor_pair == op)    &
                idx_laser
            )
            mask_s = (base & (y_single.target == 'sample')).values
            mask_c = (base & (y_single.target == 'choice')).values
            if mask_s.sum() == 0 or mask_c.sum() == 0:
                continue
            out.append((X_ep[mask_s].mean(0), X_ep[mask_c].mean(0)))
    return out


def grand_mean_traj(X_ep, cond, stage, pairs, target):
    """Equal-weight mean of per-(mouse, odor_pair) means.

    Same averaging as group_mean_trajs so the overlay is tangent to the
    velocity field by construction.
    """
    trajs = []
    for mouse in ALL_MICE:
        for op in pairs:
            mask = (
                (y_single.mouse     == mouse) &
                (y_single.tasks     == cond)  &
                (y_single.stage     == stage) &
                (y_single.target    == target) &
                (y_single.odor_pair == op)    &
                idx_laser
            ).values
            if mask.sum() == 0:
                continue
            trajs.append(X_ep[mask].mean(0))
    if not trajs:
        return None
    return np.mean(trajs, axis=0)


# ── Main loop ─────────────────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    print(f'\n=== {train_tag} ===')

    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd

    # Pooled all-trial mean overlay trajectory per (stage, cond, sample).
    # overlay_all[stage][cond] = list of (label, sx, cy, base_color, cmap_label)
    overlay_all = {s: {c: [] for c in CONDITIONS} for s in STAGES}
    for stage in STAGES:
        for cond in CONDITIONS:
            for label, pairs, base_color, cmap_label in SAMPLE_SPLITS:
                sx = grand_mean_traj(X_ep, cond, stage, pairs, 'sample')
                cy = grand_mean_traj(X_ep, cond, stage, pairs, 'choice')
                if sx is None or cy is None:
                    continue
                overlay_all[stage][cond].append(
                    (label, sx, cy, base_color, cmap_label))

    # Axis limits: cover all BINS_LATE fixed-point positions with padding,
    # symmetric around 0 and equal on both axes.
    fp_x, fp_y = [], []
    for stage in STAGES:
        for cond in CONDITIONS:
            for _, sx, cy, *_ in overlay_all[stage][cond]:
                fp_x.append(sx[BINS_LATE].mean())
                fp_y.append(cy[BINS_LATE].mean())
    if fp_x:
        lim = max(abs(v) for v in fp_x + fp_y) + AXIS_PAD
    else:
        lim = 3.0
    AXIS_LIM = (-lim, lim)

    # Cache per-(mouse, odor_pair) mean trajectories once per (stage, cond).
    # Mean (not single-trial) trajectories so the field stays tangent to the
    # average trajectories — see group_mean_trajs.
    trajs_cache = {}
    for stage in STAGES:
        for cond in CONDITIONS:
            trajs_cache[(stage, cond)] = {
                label: group_mean_trajs(X_ep, cond, stage, pairs)
                for label, pairs, _, _ in SAMPLE_SPLITS
            }

    # ── One figure per flow variant × field mode ──────────────────────────────
    for variant_name, segments in FLOW_VARIANTS:

        xlim, ylim = AXIS_LIM, AXIS_LIM
        x_edges = np.linspace(xlim[0], xlim[1], N_BINS + 1)
        y_edges = np.linspace(ylim[0], ylim[1], N_BINS + 1)
        xi = (x_edges[:-1] + x_edges[1:]) / 2
        yi = (y_edges[:-1] + y_edges[1:]) / 2

        # Build fields once per variant (shared across all field modes)
        panel_cache = {}
        all_speeds = []
        for stage in STAGES:
            for cond in CONDITIONS:
                f = panel_fields(trajs_cache[(stage, cond)],
                                 x_edges, y_edges, segments,
                                 bins_late=BINS_LATE)
                panel_cache[(stage, cond)] = f
                if f is not None:
                    speed_h = np.where(f['supported_late'],
                                       f['speed_late'], f['speed'])
                    all_speeds.extend(speed_h[f['supported']].ravel().tolist())
        occ_vmax = np.percentile(all_speeds, 98) if all_speeds else 1.0

        for field_mode in FIELD_MODES:
            fig_dir = os.path.join(FIG_BASE, train_tag, variant_name, field_mode)
            os.makedirs(os.path.join(fig_dir, 'png'), exist_ok=True)
            os.makedirs(os.path.join(fig_dir, 'svg'), exist_ok=True)

            n_rows, n_cols = len(STAGES), len(CONDITIONS)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 4.5, n_rows * 4.5),
                                     sharex=True, sharey=True)

            last_hm = None
            for ri, stage in enumerate(STAGES):
                for ci, cond in enumerate(CONDITIONS):
                    ax = axes[ri, ci]
                    hm = draw_panel(ax, panel_cache[(stage, cond)], xi, yi,
                                    xlim, ylim, occ_vmax, field_mode,
                                    overlay_all[stage][cond],
                                    bins_delay=BINS_DELAY, bins_late=BINS_LATE,
                                    xtime=xtime, cmap_speed=CMAP_SPEED)
                    if hm is not None:
                        last_hm = hm
                    if ri == 0:
                        ax.set_title(cond)
                    if ci == 0:
                        ax.set_ylabel(f'{stage}\nChoice code')
                    if ri == n_rows - 1:
                        ax.set_xlabel('Sample code')

            if last_hm is not None:
                cbar = fig.colorbar(last_hm, ax=axes, fraction=0.025, pad=0.03,
                                    shrink=0.6)
                cbar.set_label('Speed (BL σ / bin)', fontsize=11)

            handles = [
                Line2D([0], [0], color='#332288', lw=2.4, label='odor A mean'),
                Line2D([0], [0], color='#44AA99', lw=2.4, label='odor B mean'),
                Line2D([0], [0], marker='s', color='grey', ls='none', ms=7,
                       label='delay end'),
                Line2D([0], [0], marker='*', color='#332288', mec='white',
                       ls='none', ms=11, label='fixed pt A'),
                Line2D([0], [0], marker='*', color='#44AA99', mec='white',
                       ls='none', ms=11, label='fixed pt B'),
            ]
            axes[0, -1].legend(handles=handles, fontsize=8, frameon=False,
                               loc='upper right')

            stem = f'{DUM}_{train_tag}'
            fig.savefig(os.path.join(fig_dir, 'png', f'{stem}.png'),
                        bbox_inches='tight', dpi=150)
            fig.savefig(os.path.join(fig_dir, 'svg', f'{stem}.svg'),
                        bbox_inches='tight')
            plt.close(fig)
            print(f'  saved {variant_name}/{field_mode}/{stem}.png')

print(f'\nFlow2D → {FIG_BASE}/')
