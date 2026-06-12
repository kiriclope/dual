"""
Plotting helpers shared by singlePCA and metaPCA notebooks.

All functions accept X : (n_trials, n_comp, n_time) and y : DataFrame.
They return the figure so the caller can save or further decorate it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, t as t_dist


# ── trajectories ──────────────────────────────────────────────────────────────

def plot_trajectories_1d(
    X, y, mask, factor, factor_labels, colors,
    xtime, n_comp=3, add_vlines_fn=None,
    figsize=None, sharey=False,
):
    """
    Time-trace plot: one panel per PC, one line per condition level.

    Parameters
    ----------
    X : (n_trials, n_comp, n_time)
    y : DataFrame aligned with X
    mask : boolean array — base trial filter (laser, performance, …)
    factor : str — column in y to split by (e.g. 'odor_pair', 'tasks')
    factor_labels : list[str] — display names for each level
    colors : list of colour specs, one per level
    xtime : (n_time,) array — time axis for plotting
    add_vlines_fn : callable(ax, **kwargs) or None
    """
    levels = sorted(y.loc[mask, factor].unique())
    width = figsize[0] / n_comp if figsize else 3
    height = figsize[1] if figsize else 3
    fig, axes = plt.subplots(1, n_comp, figsize=(n_comp * width, height), sharey=sharey)
    if n_comp == 1:
        axes = [axes]

    for i, lv in enumerate(levels):
        sel = mask & (y[factor] == lv)
        X_sel = X[sel]
        mu = X_sel.mean(0)
        sem = X_sel.std(0) / np.sqrt(X_sel.shape[0])
        label = factor_labels[i] if i < len(factor_labels) else str(lv)
        for k, ax in enumerate(axes):
            ax.plot(xtime, mu[k], color=colors[i], label=label)
            ax.fill_between(xtime, mu[k] - sem[k], mu[k] + sem[k],
                            color=colors[i], alpha=0.2)
            ax.axhline(0, ls='--', color='k', lw=0.8)
            ax.set_xlabel('Time')
            ax.set_ylabel(f'PC {k + 1}')
            ax.set_xlim([xtime[0], xtime[-1]])
            if add_vlines_fn is not None:
                add_vlines_fn(ax)

    for ax in axes:
        ax.legend(fontsize=10, frameon=False, loc='best')

    fig.tight_layout()
    return fig, axes


def plot_trajectories_2d(X, y, mask, factor, factor_labels, colors, t_end_idx=None,
                         pc_labels=None):
    """
    PC-plane trajectories (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3).
    Paths are drawn with a light→dark time gradient and direction arrows.

    pc_labels : optional list of axis labels per PC (e.g. 'PC 1 (Choice)').
                Defaults to 'PC {k+1}'.
    """
    from src.plot.traj import plot_gradient_line, add_arrows

    levels = sorted(y.loc[mask, factor].unique())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    pairs = [(0, 1), (0, 2), (1, 2)]
    # track data extent per panel: LineCollection doesn't drive autoscale, and
    # ax.relim() ignores collections — so set limits from the trajectories
    lims = [[np.inf, -np.inf, np.inf, -np.inf] for _ in pairs]  # xmin,xmax,ymin,ymax
    for i, lv in enumerate(levels):
        sel = mask & (y[factor] == lv)
        mu = X[sel].mean(0)  # (n_comp, n_time)
        if t_end_idx is not None:
            mu = mu[:, :t_end_idx]
        label = factor_labels[i] if i < len(factor_labels) else str(lv)
        color = colors[i] if i < len(colors) else f'C{i}'
        for j, (ax, (a, b)) in enumerate(zip(axes, pairs)):
            plot_gradient_line(ax, mu[a], mu[b], color)
            add_arrows(ax, mu[a], mu[b], color)
            ax.plot([], [], color=color, label=label, lw=2)  # legend proxy
            lims[j][0] = min(lims[j][0], float(mu[a].min()))
            lims[j][1] = max(lims[j][1], float(mu[a].max()))
            lims[j][2] = min(lims[j][2], float(mu[b].min()))
            lims[j][3] = max(lims[j][3], float(mu[b].max()))

    if pc_labels is None:
        pc_labels = [f'PC {k + 1}' for k in range(X.shape[1])]
    for j, (ax, (a, b)) in enumerate(zip(axes, pairs)):
        ax.set_xlabel(pc_labels[a])
        ax.set_ylabel(pc_labels[b])
        ax.axhline(0, ls='--', lw=0.8, color='0.6')
        ax.axvline(0, ls='--', lw=0.8, color='0.6')
        xmin, xmax, ymin, ymax = lims[j]
        mx = 0.08 * (xmax - xmin or 1.0)
        my = 0.08 * (ymax - ymin or 1.0)
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    return fig, axes


# ── explained variance ────────────────────────────────────────────────────────

def plot_evr(evr, ylim=(0, 0.5)):
    """
    EVR curve with SEM across mice / folds.

    evr : (n_mice_or_folds, n_comp)
    """
    evr = np.asarray(evr, dtype=float)
    mu = evr.mean(0)
    sem = 1.96 * evr.std(0, ddof=1) / np.sqrt(evr.shape[0])
    n_pcs = np.arange(1, len(mu) + 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(n_pcs, mu, '-o')
    ax.fill_between(n_pcs, mu - sem, mu + sem, alpha=0.2)
    ax.set_xlabel('PC #')
    ax.set_ylabel('Explained variance ratio')
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    return fig, ax


# ── PC weights in theta space ─────────────────────────────────────────────────

def plot_weights_theta(w, theta, theta_norm, n_comp=3, cmap=None, sigma=0.1, z_lim=5):
    """
    Scatter of PC weights vs preferred direction theta, with smoothed mean.

    w : (n_comp, n_neurons)
    theta : (n_neurons,) preferred direction in degrees (–180 to 180)
    theta_norm : (n_neurons,) same but 0–360
    """
    import cmocean
    if cmap is None:
        cmap = cmocean.cm.phase

    idx = np.argsort(theta)
    smooth_width = int(sigma * w.shape[1])

    fig, axes = plt.subplots(1, n_comp, figsize=(n_comp * 3, 3))
    if n_comp == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        sc = ax.scatter(theta[idx], w[k, idx], alpha=0.4,
                        c=theta_norm[idx], cmap=cmap, rasterized=True, s=4)
        ax.plot(theta[idx],
                gaussian_filter1d(w[k, idx], smooth_width, mode='wrap'),
                'k', lw=1.5)
        ax.axhline(0, ls='--', color='k', lw=0.8)
        ax.set_ylabel(f'Weights PC {k + 1}')
        ax.set_xlabel('Neuron loc (°)')
        ax.set_ylim([-z_lim, z_lim])

    plt.colorbar(sc, ax=axes[-1], label='Angle (°)')
    fig.tight_layout()
    return fig, axes


def plot_weights_theta_binned(w_mean_s, ci_lo_s, ci_hi_s, theta_bins, n_comp=3):
    """
    Binned mean (± CI) of PC weights across theta bins.

    w_mean_s, ci_lo_s, ci_hi_s : (n_comp, n_bins) — pre-smoothed
    """
    bin_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])

    fig, axes = plt.subplots(1, n_comp, figsize=(n_comp * 3, 3))
    if n_comp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(bin_centers, w_mean_s[i], lw=2)
        ax.fill_between(bin_centers, ci_lo_s[i], ci_hi_s[i], alpha=0.25)
        ax.axhline(0, ls='--', color='k', lw=0.8)
        ax.set_ylabel(f'Weights PC {i + 1}')
        ax.set_xlabel('Neuron loc (°)')
        ax.set_xticks([0, 90, 180, 270, 360])

    fig.tight_layout()
    return fig, axes


def compute_theta_bins(w, theta_norm, n_comp=3, nbins=16, alpha=0.05, sigma=1):
    """
    Bin PC weights by theta, compute mean ± CI, return smoothed arrays.

    Returns w_mean_s, ci_lo_s, ci_hi_s : each (n_comp, nbins)
    """
    theta_bins = np.linspace(0, 360, nbins + 1)
    theta_digitized = np.digitize(theta_norm, theta_bins) - 1

    w_mean = np.full((n_comp, nbins), np.nan)
    w_sem = np.full((n_comp, nbins), np.nan)
    w_n = np.zeros((n_comp, nbins), dtype=int)

    for i in range(nbins):
        mask = theta_digitized == i
        for j in range(n_comp):
            x = w[j, mask]
            n = x.size
            w_n[j, i] = n
            if n >= 2:
                w_mean[j, i] = np.mean(x)
                w_sem[j, i] = np.std(x, ddof=1) / np.sqrt(n)
            elif n == 1:
                w_mean[j, i] = x[0]

    tcrit = np.full((n_comp, nbins), np.nan)
    for j in range(n_comp):
        for i in range(nbins):
            if w_n[j, i] >= 2:
                tcrit[j, i] = t_dist.ppf(1 - alpha / 2, df=w_n[j, i] - 1)

    ci_lo = w_mean - tcrit * w_sem
    ci_hi = w_mean + tcrit * w_sem

    w_mean_s = gaussian_filter1d(w_mean, sigma=sigma, axis=1, mode='wrap')
    ci_lo_s = gaussian_filter1d(ci_lo, sigma=sigma, axis=1, mode='wrap')
    ci_hi_s = gaussian_filter1d(ci_hi, sigma=sigma, axis=1, mode='wrap')

    return theta_bins, w_mean_s, ci_lo_s, ci_hi_s


# ── optogenetic perturbation ──────────────────────────────────────────────────

def plot_opto_scatter(df, delta_pc_col, delta_perf_cols, palette, label_map=None):
    """
    Regression scatter: delta PC vs delta performance for laser mice.

    df : DataFrame with columns delta_pc_col, *delta_perf_cols, 'mouse'
    delta_perf_cols : list of two column names (e.g. ['delta_dpa', 'delta_odr'])
    label_map : dict mapping delta_perf_col -> axis label, or None
    """
    fig, axes = plt.subplots(1, len(delta_perf_cols),
                             figsize=(len(delta_perf_cols) * 3, 3), sharey=True)

    for ax, col in zip(axes, delta_perf_cols):
        valid = df[[delta_pc_col, col]].dropna()
        sns.regplot(data=df, x=delta_pc_col, y=col,
                    scatter=False, fit_reg=True, ci=95, ax=ax,
                    line_kws={'color': 'k', 'lw': 1.5, 'ls': '--'})
        sns.scatterplot(data=df, x=delta_pc_col, y=col,
                        hue='mouse', palette=palette,
                        s=70, alpha=0.8, ax=ax, legend=False)
        r, p = pearsonr(valid[delta_pc_col], valid[col])
        ax.annotate(f'r = {r:.2f}\np = {p:.3f}',
                    xy=(0.65, 0.95), xycoords='axes fraction', fontsize=10,
                    va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
        ax.set_xlabel(f'Δ {delta_pc_col.upper()}')
        ylabel = (label_map or {}).get(col, f'Δ {col}')
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, axes


# ── Naive vs Expert ───────────────────────────────────────────────────────────

def compute_coding_strength(X, y, epoch_bins, groupby_cols, pc_indices=(0, 1, 2)):
    """
    Aggregate normalised trajectory amplitude per condition group and epoch.

    Returns plot_df with columns groupby_cols + ['pc', 'value'].
    """
    pc_names = [f'PC{p + 1}' for p in pc_indices]
    df = y.reset_index(drop=True).copy()
    df['_traj'] = list(X[:, list(pc_indices), :])

    rows = []
    for keys, g in df.groupby(groupby_cols, sort=False):
        T = np.stack(g['_traj'].to_numpy(), axis=0)       # (n_trials, n_pc, n_time)
        mean_traj = (T / (np.abs(T).max(axis=0) + 1e-12)).mean(axis=0)
        val = np.abs(mean_traj[:, epoch_bins]).mean(axis=1)
        row = dict(zip(groupby_cols, keys if isinstance(keys, tuple) else (keys,)))
        for i, pc in enumerate(pc_names):
            rows.append({**row, 'pc': pc, 'value': float(val[i])})

    plot_df = pd.DataFrame(rows)
    if 'mouse' in plot_df.columns:
        order = sorted(plot_df['mouse'].unique())
        plot_df['mouse'] = pd.Categorical(plot_df['mouse'], categories=order, ordered=True)
    return plot_df


def plot_naive_expert_bar(plot_df, pc_names, value_label='Value'):
    """Bar chart: mean value per mouse × learning, faceted by PC."""
    g = sns.catplot(
        data=plot_df, kind='bar',
        x='mouse', y='value', col='pc', col_order=pc_names,
        hue='learning', height=3.4, aspect=1.15, errorbar=None,
    )
    g.set_titles('{col_name}')
    g.set_axis_labels('Mouse', value_label)
    for ax in g.axes.flat:
        ax.grid(False)
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
            lab.set_ha('right')
        ax.tick_params(direction='out', length=4, width=1)
    g._legend.set_title('Learning')
    g.fig.subplots_adjust(right=0.86, wspace=0.15, top=0.90)
    g._legend.set_bbox_to_anchor((0.98, 0.5))
    return g.fig


def plot_naive_expert_scatter(plot_df, pc_names):
    """Scatter Naive vs Expert amplitude, one dot per mouse, faceted by PC."""
    df = plot_df.copy()
    if not np.issubdtype(df['learning'].dtype, np.number):
        stage_map = {'naive': 'Naive', 'expert': 'Expert', '0': 'Naive', '1': 'Expert'}
        df['stage'] = df['learning'].astype(str).str.strip().str.lower().map(stage_map)
    else:
        df['stage'] = df['learning'].map({0: 'Naive', 1: 'Expert'})

    df = df[df['stage'].isin(['Naive', 'Expert'])]
    agg = df.groupby(['mouse', 'stage', 'pc'], as_index=False)['value'].mean()
    wide = (agg.pivot(index=['mouse', 'pc'], columns='stage', values='value')
               .reset_index().dropna(subset=['Naive', 'Expert']))

    mice = pd.Index(wide['mouse'].unique()).sort_values()
    palette = dict(zip(mice, sns.color_palette('tab20', n_colors=len(mice))))

    g = sns.FacetGrid(wide, col='pc', col_order=pc_names, height=3, aspect=1.0)
    g.map_dataframe(sns.scatterplot, x='Naive', y='Expert',
                    hue='mouse', palette=palette, s=70,
                    edgecolor='black', linewidth=0.3, legend=False)
    for ax in g.axes.flat:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color='0.4', lw=1, ls='--')
        ax.set_xlabel('Naive')
        ax.set_ylabel('Expert')
        ax.tick_params(direction='out', length=4, width=1)
    g.set_titles('{col_name}')
    g.fig.tight_layout()
    return g.fig
