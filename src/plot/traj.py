"""
Shared trajectory drawing and flow-field primitives.

Used by both the overlaps (sample×choice CCGD plane) and PCA (sample×lick
axis projection plane) plotting scripts.  All functions are pure: they accept
numpy arrays and matplotlib axes; they never read project data files.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, binary_closing, gaussian_filter1d


# ── 1-D time-trace helper ─────────────────────────────────────────────────────

def plot_mean_sem(ax, xtime, mu, sem, color, alpha=0.2, **plot_kwargs):
    """Draw a mean line with a ± SEM shaded band.

    Parameters
    ----------
    ax         : matplotlib Axes
    xtime      : (n_time,) time axis
    mu, sem    : (n_time,) pre-computed mean and standard error
    color      : line and band colour
    alpha      : band transparency
    **plot_kwargs : forwarded to ax.plot (e.g. lw, label, ls, zorder)
    """
    ax.plot(xtime, mu, color=color, **plot_kwargs)
    ax.fill_between(xtime, mu - sem, mu + sem, color=color, alpha=alpha, lw=0)


# ── Drawing primitives ─────────────────────────────────────────────────────────

def make_time_cmap(base_color, lo=0.22):
    """Sequential colormap from a diluted tint of base_color (start) to base_color (end)."""
    rgb = np.array(mcolors.to_rgb(base_color))
    light = rgb * lo + (1 - lo)
    return mcolors.LinearSegmentedColormap.from_list('', [tuple(light), tuple(rgb)])


def plot_gradient_line(ax, x, y, color, lw=2.1, zorder=5):
    """Draw path (x, y) with a light→dark time gradient using base color."""
    cmap = make_time_cmap(color)
    t = np.linspace(0, 1, len(x))
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1),
                        linewidth=lw, zorder=zorder,
                        capstyle='round', joinstyle='round')
    lc.set_array(t[:-1])
    ax.add_collection(lc)


def add_arrows(ax, x, y, color, n_arrows=3, zorder=6):
    """Add evenly-spaced direction arrowheads along path (x, y)."""
    T = len(x)
    for frac in np.linspace(0.3, 0.92, n_arrows):
        i = int(frac * (T - 2))
        ax.annotate(
            '', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle='-|>', mutation_scale=11,
                            color=color, lw=0),
            zorder=zorder,
        )


def sem_band(ax, mx, my, ex, ey, color, alpha=0.18, zorder=3):
    """Shaded tube: half-width = per-time SEM projected onto the path normal."""
    dx, dy = np.gradient(mx), np.gradient(my)
    L = np.hypot(dx, dy)
    L[L == 0] = 1.0
    nx, ny = -dy / L, dx / L
    w = np.sqrt((nx * ex) ** 2 + (ny * ey) ** 2)
    upper = np.column_stack([mx + nx * w, my + ny * w])
    lower = np.column_stack([mx - nx * w, my - ny * w])
    poly = np.vstack([upper, lower[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, lw=0, zorder=zorder)


def colored_path(ax, x, y, t, cmap, lw=2.4, alpha=0.95, zorder=6):
    """Draw path (x, y) coloured by scalar array t (e.g. time)."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(t.min(), t.max()),
                        linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


def truncate_cmap(name, lo=0.35, hi=1.0, n=256):
    """Return a named colormap truncated to the [lo, hi] intensity range."""
    base = matplotlib.colormaps[name]
    return mcolors.LinearSegmentedColormap.from_list(
        f'{name}_tr', base(np.linspace(lo, hi, n)))


# ── Velocity / flow-field primitives ──────────────────────────────────────────

def velocity_points(trajs, segments, step=5, smooth=0):
    """Centred finite-difference velocity samples from (sx, cy) trajectories.

    Parameters
    ----------
    trajs    : list of (sx, cy) — each a 1-D array of length n_time
    segments : list of contiguous int arrays (e.g. [BINS_DELAY])
    step     : finite-difference step; h = step//2; vel = (x[t+h]-x[t-h])/(2h)
    smooth   : Gaussian temporal smooth sigma applied before differencing (0=off)

    Returns
    -------
    px, py, du, dv : flat arrays of positions and velocity components
    """
    h = max(1, step // 2)
    px, py, du, dv = [], [], [], []
    for sx, cy in trajs:
        xs = gaussian_filter1d(sx, smooth) if smooth > 0 else np.asarray(sx)
        ys = gaussian_filter1d(cy, smooth) if smooth > 0 else np.asarray(cy)
        for seg in segments:
            seg = np.asarray(seg)
            if len(seg) <= 2 * h:
                continue
            c = seg[h:len(seg) - h]
            px.extend(xs[c].tolist())
            py.extend(ys[c].tolist())
            du.extend(((xs[c + h] - xs[c - h]) / (2 * h)).tolist())
            dv.extend(((ys[c + h] - ys[c - h]) / (2 * h)).tolist())
    return np.array(px), np.array(py), np.array(du), np.array(dv)


def transition_velocity_points(trajs, segments, step=5):
    """Forward-difference displacement E[Δx | x] from (sx, cy) trajectories.

    Each (src, src+step) pair contributes a displacement vector attributed to
    the source position.  After binning this gives the expected next-step
    displacement regardless of occupancy density.
    """
    px, py, du, dv = [], [], [], []
    for sx, cy in trajs:
        xs, ys = np.asarray(sx), np.asarray(cy)
        for seg in segments:
            seg = np.asarray(seg)
            if len(seg) <= step:
                continue
            src = seg[:-step]
            tgt = seg[step:]
            px.extend(xs[src].tolist())
            py.extend(ys[src].tolist())
            du.extend((xs[tgt] - xs[src]).tolist())
            dv.extend((ys[tgt] - ys[src]).tolist())
    return np.array(px), np.array(py), np.array(du), np.array(dv)


def bin_velocity(px, py, du, dv, x_edges, y_edges, sigma=1.2):
    """Nadaraya-Watson velocity field on a grid defined by x_edges, y_edges.

    Returns U, V (smoothed velocity components) and count_raw (raw histogram).
    """
    n_x, n_y = len(x_edges) - 1, len(y_edges) - 1
    if len(px) == 0:
        z = np.zeros((n_x, n_y))
        return z, z.copy(), z.copy()
    in_b = (
        (px >= x_edges[0]) & (px <= x_edges[-1]) &
        (py >= y_edges[0]) & (py <= y_edges[-1])
    )
    px, py, du, dv = px[in_b], py[in_b], du[in_b], dv[in_b]
    bins = [x_edges, y_edges]
    count_raw, _, _ = np.histogram2d(px, py, bins=bins)
    sum_u,     _, _ = np.histogram2d(px, py, bins=bins, weights=du)
    sum_v,     _, _ = np.histogram2d(px, py, bins=bins, weights=dv)
    count_s = gaussian_filter(count_raw.astype(float), sigma=sigma)
    U = gaussian_filter(sum_u, sigma=sigma) / (count_s + 1e-6)
    V = gaussian_filter(sum_v, sigma=sigma) / (count_s + 1e-6)
    return U, V, count_raw


def raw_counts(px, py, x_edges, y_edges):
    """Raw 2-D histogram of positions."""
    n_x, n_y = len(x_edges) - 1, len(y_edges) - 1
    if len(px) == 0:
        return np.zeros((n_x, n_y))
    in_b = (
        (px >= x_edges[0]) & (px <= x_edges[-1]) &
        (py >= y_edges[0]) & (py <= y_edges[-1])
    )
    cnt, _, _ = np.histogram2d(px[in_b], py[in_b], bins=[x_edges, y_edges])
    return cnt


# ── Flow-field panels ──────────────────────────────────────────────────────────

def panel_fields(trajs_by_label, x_edges, y_edges, segments, bins_late,
                 sigma=1.2, min_raw_count=1):
    """Build per-label velocity fields and a winner-take-all combined field.

    Parameters
    ----------
    trajs_by_label : dict {label: list of (sx, cy)} — one entry per condition label
    x_edges, y_edges : grid bin edges
    segments    : list of contiguous bin arrays for the velocity field
    bins_late   : bin indices of the late-delay epoch (for attractor support)
    sigma       : spatial Gaussian smoothing of binned fields (bins)
    min_raw_count : minimum raw visits for a cell to be marked supported

    Returns dict with combined U, V, speed, support arrays and per-label data,
    or None if no data.
    """
    per_sample = {}
    for label, trajs in trajs_by_label.items():
        if not trajs:
            continue
        px, py, du, dv = velocity_points(trajs, segments)
        U, V, cnt = bin_velocity(px, py, du, dv, x_edges, y_edges, sigma=sigma)
        lx, ly, lu, lv = velocity_points(trajs, [bins_late])
        late_cnt = raw_counts(lx, ly, x_edges, y_edges)
        U_late, V_late, _ = bin_velocity(lx, ly, lu, lv, x_edges, y_edges, sigma=sigma)
        per_sample[label] = dict(U=U, V=V, U_late=U_late, V_late=V_late,
                                 count=cnt, late=late_cnt, trajs=trajs)

    if not per_sample:
        return None

    labels = list(per_sample.keys())
    cnt_stack = np.stack([per_sample[l]['count'] for l in labels], axis=0)
    U_stack   = np.stack([per_sample[l]['U']     for l in labels], axis=0)
    V_stack   = np.stack([per_sample[l]['V']     for l in labels], axis=0)
    winner    = np.argmax(cnt_stack, axis=0)
    ii, jj    = np.indices(winner.shape)
    U     = U_stack[winner, ii, jj]
    V     = V_stack[winner, ii, jj]
    count = cnt_stack.sum(axis=0)
    supported    = binary_closing(count >= min_raw_count,
                                  structure=np.ones((3, 3)), iterations=1)
    speed        = np.sqrt(U**2 + V**2)
    count_smooth = gaussian_filter(count.astype(float), sigma=sigma)

    # Transition field (forward-difference, WTA)
    per_sample_tr = {}
    for label in per_sample:
        trajs = per_sample[label]['trajs']
        px_t, py_t, du_t, dv_t = transition_velocity_points(trajs, segments)
        U_t, V_t, cnt_t = bin_velocity(px_t, py_t, du_t, dv_t,
                                       x_edges, y_edges, sigma=sigma)
        per_sample_tr[label] = dict(U=U_t, V=V_t, count=cnt_t)

    labels_tr = list(per_sample_tr.keys())
    if labels_tr:
        cnt_tr = np.stack([per_sample_tr[l]['count'] for l in labels_tr])
        U_tr   = np.stack([per_sample_tr[l]['U']     for l in labels_tr])
        V_tr   = np.stack([per_sample_tr[l]['V']     for l in labels_tr])
        win_tr = np.argmax(cnt_tr, axis=0)
        ii_t, jj_t = np.indices(win_tr.shape)
        U_trans = U_tr[win_tr, ii_t, jj_t]
        V_trans = V_tr[win_tr, ii_t, jj_t]
    else:
        U_trans = V_trans = np.zeros_like(U)

    # WTA late-delay speed for hybrid heatmap
    U_late_stack   = np.stack([per_sample[l]['U_late'] for l in labels], axis=0)
    V_late_stack   = np.stack([per_sample[l]['V_late'] for l in labels], axis=0)
    late_cnt_stack = np.stack([per_sample[l]['late']   for l in labels], axis=0)
    winner_late    = np.argmax(late_cnt_stack, axis=0)
    ii_l, jj_l    = np.indices(winner_late.shape)
    U_late_wta     = U_late_stack[winner_late, ii_l, jj_l]
    V_late_wta     = V_late_stack[winner_late, ii_l, jj_l]
    speed_late     = np.sqrt(U_late_wta**2 + V_late_wta**2)
    count_late     = late_cnt_stack.sum(axis=0)
    supported_late = binary_closing(count_late >= min_raw_count,
                                    structure=np.ones((3, 3)), iterations=1)

    return dict(U=U, V=V, U_trans=U_trans, V_trans=V_trans,
                count=count, count_smooth=count_smooth,
                supported=supported, speed=speed,
                speed_late=speed_late, supported_late=supported_late,
                per_sample=per_sample)


def draw_panel(ax, fields, xi, yi, xlim, ylim, occ_vmax, field_mode,
               traj_overlay, bins_delay, bins_late, xtime, cmap_speed):
    """Draw a flow-field panel: speed heatmap + streamlines + trajectory overlay.

    Parameters
    ----------
    fields       : dict from panel_fields(), or None
    xi, yi       : 1-D grid cell-centre arrays
    xlim, ylim   : axis limits
    occ_vmax     : colormap max for speed heatmap
    field_mode   : 'velocity' | 'flux' | 'transition'
    traj_overlay : list of (label, sx_full, cy_full, base_color, cmap_name)
                   sx_full and cy_full are full-time trajectories (n_time,)
    bins_delay   : delay-period bin indices (used for the trajectory overlay)
    bins_late    : late-delay bin indices (used for fixed-point markers)
    xtime        : (n_time,) time axis
    cmap_speed   : matplotlib colormap for the speed heatmap
    """
    if fields is None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return None

    supported    = fields['supported']
    count_smooth = fields['count_smooth']

    # Hybrid speed heatmap: late-delay speed at attractor cells (fixed pts dark),
    # all-delay speed elsewhere (approach path visible).
    speed_hybrid = np.where(fields['supported_late'],
                            fields['speed_late'], fields['speed'])
    speed_ma = np.ma.masked_where(~fields['supported'].T, speed_hybrid.T)
    hm = ax.pcolormesh(xi, yi, speed_ma, cmap=cmap_speed, shading='auto',
                       vmin=0, vmax=occ_vmax, rasterized=True, zorder=0)

    if field_mode == 'flux':
        rho = count_smooth / (count_smooth.max() + 1e-9)
        U_stream = fields['U'] * rho
        V_stream = fields['V'] * rho
    elif field_mode == 'transition':
        U_stream = fields['U_trans']
        V_stream = fields['V_trans']
    else:
        U_stream = fields['U']
        V_stream = fields['V']

    U_plot = np.where(supported, U_stream, 0.0).T
    V_plot = np.where(supported, V_stream, 0.0).T
    try:
        ax.streamplot(xi, yi, U_plot, V_plot, color='white', density=0.8,
                      linewidth=0.7, arrowsize=0.9, zorder=2)
    except Exception as e:
        print(f'  streamplot skipped: {e}')

    for label, sx_full, cy_full, base_color, cmap_name in traj_overlay:
        cx    = sx_full[bins_late].mean()
        cy_fp = cy_full[bins_late].mean()
        ax.scatter(cx, cy_fp, marker='*', s=220, facecolors=base_color,
                   edgecolors='white', linewidths=0.8, zorder=10)

    for label, sx_full, cy_full, base_color, cmap_name in traj_overlay:
        sx_m = sx_full[bins_delay]
        cy_m = cy_full[bins_delay]
        ax.plot(sx_m, cy_m, color='white', lw=5.5, alpha=0.9,
                solid_capstyle='round', zorder=6)
        colored_path(ax, sx_m, cy_m, xtime[bins_delay],
                     cmap=truncate_cmap(cmap_name),
                     lw=3.2, alpha=1.0, zorder=7)
        ax.scatter(sx_m[0], cy_m[0], marker='o', s=70, color=base_color,
                   edgecolors='white', linewidths=1.2, zorder=8)
        ax.scatter(sx_m[-1], cy_m[-1], marker='s', s=95, color=base_color,
                   edgecolors='white', linewidths=1.2, zorder=8)

    ax.axhline(0, ls=':', color='0.6', lw=0.7, zorder=1)
    ax.axvline(0, ls=':', color='0.6', lw=0.7, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return hm
