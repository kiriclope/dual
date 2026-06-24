import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from src.common.plot_utils import add_vdashed


def plot_mat(X, ax, vmin=-1, vmax=1, palette="bwr"):
    """Plot a train×test generalisation matrix with standard axis formatting."""
    im = ax.imshow(
        X,
        interpolation=None,
        origin="lower",
        cmap=palette,
        extent=[0, 14, 0, 14],
        vmin=vmin,
        vmax=vmax,
    )
    add_vdashed(ax, 1)
    ax.set_xlim([2, 12])
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_ylim([2, 12])
    ax.set_yticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    return im


def corr_annotate(ax, x, y):
    """Overlay regression line and Pearson/Spearman annotation on ax."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return
    r, p = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, "k", lw=1)
    ax.set_title(
        ax.get_title() + f"\nr={r:.2f} p={p:.2g} | ρ={rs:.2f} p={ps:.2g}",
        fontsize=10,
    )


def style_axes(ax, xlabel, ylabel, identity=False):
    """Add reference lines and labels; optionally draw identity line."""
    ax.axhline(0, ls="--", color="0.4", lw=1)
    ax.axvline(0, ls="--", color="0.4", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if identity:
        xl, yl = ax.get_xlim(), ax.get_ylim()
        lo, hi = min(xl[0], yl[0]), max(xl[1], yl[1])
        ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)


def fig_naive_vs_expert(wide, col, col_order, fname, dum="", fig_dir=".",
                        height=4):
    """Scatter Naive vs Expert overlap, one FacetGrid panel per `col` level."""
    g = sns.FacetGrid(wide, col=col, col_order=col_order,
                      height=height, aspect=1.0, sharex=True, sharey=True)

    def _plot(data, **kw):
        ax = plt.gca()
        sns.scatterplot(
            data=data, x="Naive", y="Expert",
            hue="mouse", style="task",
            s=80, edgecolor="black", linewidth=0.4,
            legend=False, ax=ax,
        )

    g.map_dataframe(_plot)

    for ax, key in zip(g.axes.flat, col_order):
        sub = wide[wide[col] == key]
        style_axes(ax, "Naive", "Expert", identity=True)
        ax.set_title(str(key))
        corr_annotate(ax, sub["Naive"].values, sub["Expert"].values)

    g.fig.tight_layout()
    g.fig.savefig(f"{fig_dir}/{fname}_{dum}.svg")
    plt.show()
