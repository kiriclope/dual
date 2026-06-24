"""
Flow field computation and plotting for delay-period dynamics in
(sample code × choice code) space.

All functions are stateless — caller passes X_ep, y_df, and config
explicitly; no module-level data globals.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

_STAGES     = ['Naive', 'Expert']
_CONDITIONS = ['DPA', 'DualGo', 'DualNoGo']
_SAMPLE_SPLITS = [
    ('A', [0, 1], 'tab:blue', 'Blues'),
    ('B', [2, 3], 'tab:red',  'Reds'),
]
PANEL_W, PANEL_H = 4.5, 4.5


def colored_path(ax, x, y, t, cmap, lw=2.5, alpha=1.0, zorder=6):
    """Draw a time-colored path on ax via LineCollection."""
    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm     = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


def get_mean_traj(X_ep, y_df, cond, stage, target, pairs, mice,
                  idx_laser=None):
    """Per-mouse mean trajectories for a given (cond, stage, target, odor_pairs) slice.

    Parameters
    ----------
    X_ep      : (n_trials, T)
    y_df      : trial metadata DataFrame
    idx_laser : boolean array (n_trials,); None → all trials pass

    Returns
    -------
    list of (T,) arrays, one per mouse that has at least one matching trial
    """
    if idx_laser is None:
        idx_laser = np.ones(len(y_df), dtype=bool)
    trajs = []
    for mouse in mice:
        m = (
            (y_df.mouse  == mouse) &
            (y_df.tasks  == cond)  &
            (y_df.stage  == stage) &
            (y_df.target == target) &
            idx_laser &
            y_df.odor_pair.isin(pairs)
        ).values
        if m.sum() == 0:
            continue
        trajs.append(X_ep[m].mean(0))
    return trajs


def collect_flow_points(X_ep, y_df, cond, stage, mice, bins_delay,
                        idx_laser=None):
    """Collect per-(mouse, odor_pair) mean-trajectory velocity points.

    Individual-trial CCGD values are too noisy (σ ≈ 5 BL σ / bin, signal
    ≈ 0.1–0.5 BL σ / bin), so per-trial dx/dt is dominated by noise.  Mean
    trajectories cancel that noise and give the correct local velocity.

    Parameters
    ----------
    X_ep       : (n_trials, T) BL-normalised CCGD scores
    y_df       : trial metadata DataFrame
    bins_delay : int array spanning the delay epoch
    idx_laser  : boolean mask (n_trials,); None → all trials pass

    Returns
    -------
    pos_x, pos_y, vel_dx, vel_dy : (N,) arrays; all empty if no trials match
    """
    if idx_laser is None:
        idx_laser = np.ones(len(y_df), dtype=bool)

    pos_x, pos_y, vel_dx, vel_dy = [], [], [], []
    t_idx = bins_delay[:-1]

    for mouse in mice:
        for odor_pair in range(4):
            base = (
                (y_df.mouse     == mouse)     &
                (y_df.tasks     == cond)      &
                (y_df.stage     == stage)     &
                (y_df.odor_pair == odor_pair) &
                idx_laser
            )
            mask_s = (base & (y_df.target == 'sample')).values
            mask_c = (base & (y_df.target == 'choice')).values
            if mask_s.sum() == 0 or mask_c.sum() == 0:
                continue

            x_mean = X_ep[mask_s].mean(0)
            y_mean = X_ep[mask_c].mean(0)

            pos_x.extend(x_mean[t_idx].tolist())
            pos_y.extend(y_mean[t_idx].tolist())
            vel_dx.extend((x_mean[t_idx + 1] - x_mean[t_idx]).tolist())
            vel_dy.extend((y_mean[t_idx + 1] - y_mean[t_idx]).tolist())

    if not pos_x:
        return (np.empty(0),) * 4
    return (np.array(pos_x), np.array(pos_y),
            np.array(vel_dx), np.array(vel_dy))


def build_smooth_field(pos_x, pos_y, vel_dx, vel_dy, xlim, ylim,
                       n_bins, sigma, density_thresh):
    """Bin then Nadaraya–Watson Gaussian-smooth the velocity field.

    Returns
    -------
    xi, yi    : (n_bins,) bin centres
    U, V      : (n_bins, n_bins) smoothed velocity components  [ix, iy]
    speed     : (n_bins, n_bins) = sqrt(U² + V²)
    count_raw : (n_bins, n_bins) raw histogram counts
    count_s   : (n_bins, n_bins) smoothed counts (denominator)
    """
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    xi = (x_edges[:-1] + x_edges[1:]) / 2
    yi = (y_edges[:-1] + y_edges[1:]) / 2

    in_b = (
        (pos_x >= xlim[0]) & (pos_x <= xlim[1]) &
        (pos_y >= ylim[0]) & (pos_y <= ylim[1])
    )
    px, py = pos_x[in_b], pos_y[in_b]
    dx, dy = vel_dx[in_b], vel_dy[in_b]
    bins   = [x_edges, y_edges]

    count_raw, _, _ = np.histogram2d(px, py, bins=bins)
    sum_u,     _, _ = np.histogram2d(px, py, bins=bins, weights=dx)
    sum_v,     _, _ = np.histogram2d(px, py, bins=bins, weights=dy)

    count_s = gaussian_filter(count_raw.astype(float), sigma=sigma)
    U = gaussian_filter(sum_u, sigma=sigma) / (count_s + 1e-6)
    V = gaussian_filter(sum_v, sigma=sigma) / (count_s + 1e-6)
    speed = np.sqrt(U**2 + V**2)

    return xi, yi, U, V, speed, count_raw, count_s


def global_slow_point(xi, yi, speed, count_raw, late_count_raw=None,
                      min_raw_count=2):
    """Single global speed minimum in the late-delay supported region.

    Restricts search to cells that trajectories actually visit in the
    *late* delay (last ~40% of bins) so the result reflects where the
    system converges, not where it trivially starts with near-zero velocity.

    Parameters
    ----------
    count_raw      : (n_bins, n_bins) raw histogram counts from all delay bins
    late_count_raw : (n_bins, n_bins) raw counts from late-delay bins only;
                     if None, falls back to count_raw
    min_raw_count  : minimum raw count required to be "supported"

    Returns
    -------
    list with one (x, y) tuple, or [] when no supported cells exist
    """
    mask = late_count_raw if late_count_raw is not None else count_raw
    supported = mask >= min_raw_count
    speed_masked = np.where(supported, speed, np.inf)
    if np.isinf(speed_masked).all():
        return []
    idx = np.unravel_index(np.argmin(speed_masked), speed_masked.shape)
    return [(xi[idx[0]], yi[idx[1]])]


def compute_axis_limits(X_ep, y_df, mice, bins_delay, idx_laser=None,
                        stages=None, conditions=None, pad=0.12):
    """Axis limits from grand-mean (mean-across-mice) delay-period trajectories.

    Uses the same approach as plot_traj2d: compute the cross-mouse mean for
    each (stage, cond, sample_identity) combination, then take the 2nd–98th
    percentile of those grand-mean values.  This avoids single-mouse outliers
    inflating the grid far beyond the actual trajectory region.

    Parameters
    ----------
    bins_delay : int array — delay epoch bins (same as passed to draw_flow_figure)
    pad        : fractional padding added to each side of the range

    Returns
    -------
    xlim, ylim : each a (lo, hi) tuple
    """
    if stages is None:
        stages = _STAGES
    if conditions is None:
        conditions = _CONDITIONS
    if idx_laser is None:
        idx_laser = np.ones(len(y_df), dtype=bool)

    all_x, all_y = [], []
    for stage in stages:
        for cond in conditions:
            for _, pairs, _, _ in _SAMPLE_SPLITS:
                xs = get_mean_traj(X_ep, y_df, cond, stage, 'sample',
                                   pairs, mice, idx_laser)
                ys = get_mean_traj(X_ep, y_df, cond, stage, 'choice',
                                   pairs, mice, idx_laser)
                if xs:
                    all_x.extend(
                        np.stack(xs, 0).mean(0)[bins_delay].tolist())
                if ys:
                    all_y.extend(
                        np.stack(ys, 0).mean(0)[bins_delay].tolist())

    def _lim(vals):
        v = np.array(vals)
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        m = max((hi - lo) * pad, 0.2)
        return lo - m, hi + m

    return _lim(all_x), _lim(all_y)


def draw_flow_figure(X_ep, y_df, mice, xlim, ylim, train_tag, bins_delay,
                     n_bins, sigma, density_thresh,
                     title_prefix='', speed_vmax=None, idx_laser=None,
                     stages=None, conditions=None, xtime=None):
    """Draw a stages × conditions grid of flow-field panels.

    Parameters
    ----------
    X_ep        : (n_trials, T) BL-normalised CCGD scores
    y_df        : trial metadata DataFrame
    mice        : list of mouse names to include
    bins_delay  : int array of delay-epoch bin indices
    title_prefix: prepended to the figure suptitle
    speed_vmax  : if given, share this colour scale (for per-mouse figures)
    idx_laser   : boolean mask; None → all trials pass
    stages      : list of stage names (default ['Naive', 'Expert'])
    conditions  : list of condition names (default ['DPA', 'DualGo', 'DualNoGo'])
    xtime       : (T,) time axis for trajectory colouring; None → np.arange(T)

    Returns
    -------
    fig, speed_vmax_used
    """
    if stages is None:
        stages = _STAGES
    if conditions is None:
        conditions = _CONDITIONS
    if idx_laser is None:
        idx_laser = np.ones(len(y_df), dtype=bool)
    if xtime is None:
        xtime = np.arange(X_ep.shape[1])

    # Late-delay bins used only for the slow-point support mask.
    # Last 40% of the delay; avoids picking the trivially-slow trajectory
    # *start* (where choice code hasn't yet ramped) as the "attractor".
    bins_late = bins_delay[int(0.6 * len(bins_delay)):]

    # Build and cache all smooth fields
    cache = {}
    late_cnts = {}   # late-delay raw histogram per panel
    all_speeds = []
    x_edges_g = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges_g = np.linspace(ylim[0], ylim[1], n_bins + 1)

    for stage in stages:
        for cond in conditions:
            px, py, vx, vy = collect_flow_points(
                X_ep, y_df, cond, stage, mice, bins_delay, idx_laser)
            if len(px) < 4:
                continue
            xi, yi, U, V, spd, cnt_raw, cnt_s = build_smooth_field(
                px, py, vx, vy, xlim, ylim, n_bins, sigma, density_thresh)
            cache[(stage, cond)] = (xi, yi, U, V, spd, cnt_raw, cnt_s)

            # Late-delay support: histogram of positions visited in bins_late
            px_l, py_l, _, _ = collect_flow_points(
                X_ep, y_df, cond, stage, mice, bins_late, idx_laser)
            if len(px_l) > 0:
                in_b = ((px_l >= xlim[0]) & (px_l <= xlim[1]) &
                        (py_l >= ylim[0]) & (py_l <= ylim[1]))
                late_cnt, _, _ = np.histogram2d(
                    px_l[in_b], py_l[in_b],
                    bins=[x_edges_g, y_edges_g])
            else:
                late_cnt = np.zeros((n_bins, n_bins))
            late_cnts[(stage, cond)] = late_cnt

            supported = cnt_s >= density_thresh * cnt_s.max()
            all_speeds.append(spd[supported].ravel())

    if speed_vmax is None:
        speed_vmax = (np.percentile(np.concatenate(all_speeds), 98)
                      if all_speeds else 1.0)

    cmap_speed = matplotlib.colormaps['magma'].copy()
    cmap_speed.set_bad('#e8e8e8')

    n_rows, n_cols = len(stages), len(conditions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * PANEL_W + 0.6, n_rows * PANEL_H),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    last_hm = None
    for ri, stage in enumerate(stages):
        for ci, cond in enumerate(conditions):
            ax = axes[ri, ci]

            if (stage, cond) in cache:
                xi, yi, U, V, spd, cnt_raw, cnt_s = cache[(stage, cond)]
                supported = cnt_s >= density_thresh * cnt_s.max()

                speed_ma = np.ma.masked_where(~supported.T, spd.T)
                last_hm = ax.pcolormesh(
                    xi, yi, speed_ma,
                    shading='auto', cmap=cmap_speed,
                    vmin=0, vmax=speed_vmax,
                    rasterized=True, zorder=0,
                )

                U_plot = np.where(supported, U, 0.0).T
                V_plot = np.where(supported, V, 0.0).T
                try:
                    ax.streamplot(
                        xi, yi, U_plot, V_plot,
                        color='white', density=1.2,
                        linewidth=0.7, arrowsize=0.9, zorder=2,
                    )
                except Exception as e:
                    print(f'  streamplot skipped ({stage}/{cond}): {e}')

                late_cnt = late_cnts.get((stage, cond),
                                         np.zeros_like(cnt_raw))
                for sx, sy in global_slow_point(
                        xi, yi, spd, cnt_raw,
                        late_count_raw=late_cnt, min_raw_count=2):
                    ax.scatter(sx, sy, marker='o', s=100,
                               facecolors='cyan', edgecolors='white',
                               linewidths=1.2, zorder=9)

            for label, pairs, base_color, cmap_label in _SAMPLE_SPLITS:
                xs = get_mean_traj(X_ep, y_df, cond, stage, 'sample',
                                   pairs, mice, idx_laser)
                ys = get_mean_traj(X_ep, y_df, cond, stage, 'choice',
                                   pairs, mice, idx_laser)
                if not xs or not ys:
                    continue
                x_mean = np.stack(xs, 0).mean(0)
                y_mean = np.stack(ys, 0).mean(0)
                xd = x_mean[bins_delay]
                yd = y_mean[bins_delay]
                colored_path(ax, xd, yd, xtime[bins_delay],
                             cmap=cmap_label, lw=2.5, zorder=5)
                ax.scatter(xd[0],  yd[0],  marker='o', s=40,
                           color=base_color, zorder=7, linewidths=0)
                ax.scatter(xd[-1], yd[-1], marker='s', s=40,
                           color=base_color, zorder=7, linewidths=0)

            ax.axhline(0, ls=':', color='grey', lw=0.5, zorder=1)
            ax.axvline(0, ls=':', color='grey', lw=0.5, zorder=1)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
            for artist in ax.collections:
                artist.set_clip_on(True)

            if ri == n_rows - 1:
                ax.set_xlabel('Sample code (BL σ)')
            if ri == 0:
                ax.set_title(cond)
            if ci == 0:
                ax.set_ylabel(f'{stage}\nChoice code (BL σ)')

    if last_hm is not None:
        cb = fig.colorbar(last_hm, ax=axes, fraction=0.015, pad=0.02)
        cb.set_label('Speed (BL σ / bin)', fontsize=10)

    handles = [
        Line2D([0],[0], color='tab:blue', lw=2.5, label='odor A mean'),
        Line2D([0],[0], color='tab:red',  lw=2.5, label='odor B mean'),
        Line2D([0],[0], marker='o', color='grey', ls='none', ms=6,
               label='delay start'),
        Line2D([0],[0], marker='s', color='grey', ls='none', ms=6,
               label='delay end'),
        Line2D([0],[0], marker='o', color='cyan', ls='none', ms=9,
               label='slow point'),
    ]
    axes[0, -1].legend(handles=handles, fontsize=7, frameon=False,
                       loc='upper right')

    fig.suptitle(
        f'{title_prefix}Delay flow field  [{train_tag}]  '
        f'{n_bins}×{n_bins} bins  σ={sigma}',
        fontsize=12, y=1.01,
    )
    return fig, speed_vmax
