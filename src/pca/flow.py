"""Flow-field estimation from PC-space trajectories (used in metaPCA)."""

import numpy as np


def flow_field_from_trajectories(
    x, y, dt=1.0, bins=25, xrange=None, yrange=None, min_count=1
):
    """
    Estimate a 2-D velocity field by binning step velocities at their
    start positions.

    Parameters
    ----------
    x, y : (n_trials, n_time) — PC coordinates
    dt : float — timestep between frames
    bins : int or (nx, ny)
    xrange, yrange : (min, max) or None (inferred)
    min_count : int — bins with fewer samples get NaN velocity

    Returns
    -------
    xedges, yedges : bin edges
    U, V : (nx, ny) mean velocity components
    count : (nx, ny) samples per bin
    """
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape

    u = (x[:, 1:] - x[:, :-1]) / dt
    v = (y[:, 1:] - y[:, :-1]) / dt
    xs, ys = x[:, :-1], y[:, :-1]

    return _bin_velocities(xs, ys, u, v, bins, xrange, yrange, min_count)


def flow_field_midpoint(
    x, y, dt=1.0, bins=25, xrange=None, yrange=None, min_count=1
):
    """
    Same as flow_field_from_trajectories but assigns each step's velocity to
    the segment midpoint rather than the start position.  Produces smoother
    fields when trajectories are curved.
    """
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape

    u = (x[:, 1:] - x[:, :-1]) / dt
    v = (y[:, 1:] - y[:, :-1]) / dt
    xm = 0.5 * (x[:, 1:] + x[:, :-1])
    ym = 0.5 * (y[:, 1:] + y[:, :-1])

    return _bin_velocities(xm, ym, u, v, bins, xrange, yrange, min_count)


# ── internal ──────────────────────────────────────────────────────────────────

def _bin_velocities(xs, ys, u, v, bins, xrange, yrange, min_count):
    if xrange is None:
        xrange = (xs.min(), xs.max())
    if yrange is None:
        yrange = (ys.min(), ys.max())

    nx, ny = (bins, bins) if isinstance(bins, int) else bins
    xedges = np.linspace(xrange[0], xrange[1], nx + 1)
    yedges = np.linspace(yrange[0], yrange[1], ny + 1)

    xsf, ysf = xs.ravel(), ys.ravel()
    uf, vf = u.ravel(), v.ravel()

    ix = np.searchsorted(xedges, xsf, side='right') - 1
    iy = np.searchsorted(yedges, ysf, side='right') - 1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, uf, vf = ix[valid], iy[valid], uf[valid], vf[valid]

    count = np.zeros((nx, ny), dtype=int)
    usum = np.zeros((nx, ny), dtype=float)
    vsum = np.zeros((nx, ny), dtype=float)
    np.add.at(count, (ix, iy), 1)
    np.add.at(usum, (ix, iy), uf)
    np.add.at(vsum, (ix, iy), vf)

    U = np.full((nx, ny), np.nan)
    V = np.full((nx, ny), np.nan)
    ok = count >= min_count
    U[ok] = usum[ok] / count[ok]
    V[ok] = vsum[ok] / count[ok]

    return xedges, yedges, U, V, count
