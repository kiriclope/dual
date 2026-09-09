"""mouse_cartoon.py — a cute, cartoony head-fixed-mouse portrait, drawn in matplotlib (pure vector).

Replaces the traced `mouse_lineart.svg` raster in Fig. 1a (Leon 2026-09-09: "draw a cute portrait of a
head fixed mouse instead (cartoony)"). Front-facing portrait so it fits the tall, narrow first slot of
panel a: head-plate across the skull with its two clamps, big ears, big eyes, whiskers, front paws on the
edge of the body tube, and the lick spout coming in under the snout — the setup the task runs in.

    from mouse_cartoon import draw_headfixed_mouse
    draw_headfixed_mouse(ax, ps=PS)            # ax is left in data coords 0..1, aspect equal, no frame

Preview on its own:
    cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python mouse_cartoon.py
"""
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

INK = '#2E2A28'          # outline (warm near-black, softer than pure k)
FUR = '#FBF7F4'          # fur fill
FUR_SHADE = '#EFE7E1'    # muzzle / inner shading
PINK = '#F0BFC6'         # inner ear
PINK_DK = '#DE93A0'      # nose
METAL = '#C9CDD2'        # head-plate
METAL_DK = '#8D949B'     # clamps
SPOUT = '#B9BEC4'
WATER = '#7FB8D8'


def _bez(verts, codes, **kw):
    return PathPatch(Path(verts, codes), **kw)


def draw_headfixed_mouse(ax, ps=1.0, ink=INK):
    """Draw the portrait into `ax` (axes are cleared, set to 0..1 data coords, equal aspect, no frame)."""
    lw = 1.15 * ps
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')
    common = dict(ec=ink, lw=lw, joinstyle='round', capstyle='round', zorder=3)

    # ── head-plate: bar across the skull + a clamp at each end (behind the head) ──
    ax.add_patch(FancyBboxPatch((0.10, 0.845), 0.80, 0.052, boxstyle='round,pad=0,rounding_size=0.026',
                                fc=METAL, ec=ink, lw=lw * 0.9, zorder=1))
    for cx in (0.055, 0.855):
        ax.add_patch(FancyBboxPatch((cx, 0.805), 0.09, 0.135, boxstyle='round,pad=0,rounding_size=0.022',
                                    fc=METAL_DK, ec=ink, lw=lw * 0.9, zorder=1))
        ax.plot([cx + 0.045, cx + 0.045], [0.93, 1.0], color=METAL_DK, lw=lw * 3.2, solid_capstyle='round', zorder=0)

    # ── ears (behind the head) ──
    for ex in (0.215, 0.785):
        ax.add_patch(Circle((ex, 0.735), 0.145, fc=FUR, **{**common, 'zorder': 1}))
        ax.add_patch(Circle((ex, 0.728), 0.085, fc=PINK, ec=ink, lw=lw * 0.75, zorder=2))

    # ── body hint: shoulders rising out of the tube, and the tube edge ──
    ax.add_patch(_bez([(0.13, 0.02), (0.20, 0.30), (0.80, 0.30), (0.87, 0.02)],
                      [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4], fc=FUR, **{**common, 'zorder': 2}))

    # ── head: wide cranium, full cheeks, soft chin ──
    ax.add_patch(_bez(
        [(0.50, 0.865),                                     # crown
         (0.755, 0.865), (0.855, 0.685), (0.845, 0.505),    # right temple → cheek
         (0.835, 0.315), (0.690, 0.185), (0.500, 0.185),    # right cheek → chin
         (0.310, 0.185), (0.165, 0.315), (0.155, 0.505),    # chin → left cheek
         (0.145, 0.685), (0.245, 0.865), (0.500, 0.865)],   # left temple → crown
        [Path.MOVETO] + [Path.CURVE4] * 12, fc=FUR, **common))

    # ── muzzle ──
    ax.add_patch(Ellipse((0.50, 0.335), 0.40, 0.235, fc=FUR_SHADE, ec=ink, lw=lw * 0.8, zorder=4))

    # ── eyes (big, with a highlight) ──
    for ex in (0.363, 0.637):
        ax.add_patch(Ellipse((ex, 0.560), 0.125, 0.145, fc=ink, ec=ink, lw=lw * 0.6, zorder=5))
        ax.add_patch(Circle((ex - 0.030, 0.600), 0.030, fc='white', ec='none', zorder=6))
        ax.add_patch(Circle((ex + 0.028, 0.523), 0.014, fc='white', ec='none', alpha=0.85, zorder=6))

    # ── blush ──
    for ex in (0.245, 0.755):
        ax.add_patch(Ellipse((ex, 0.415), 0.105, 0.058, fc=PINK, ec='none', alpha=0.55, zorder=5))

    # ── nose + philtrum + smile ──
    ax.add_patch(Polygon([(0.50, 0.345), (0.452, 0.408), (0.548, 0.408)], closed=True,
                         fc=PINK_DK, ec=ink, lw=lw * 0.7, zorder=6, joinstyle='round'))
    ax.plot([0.50, 0.50], [0.345, 0.300], color=ink, lw=lw * 0.8, solid_capstyle='round', zorder=6)
    t = np.linspace(0, np.pi, 40)
    for sgn in (-1, 1):
        ax.plot(0.50 + sgn * 0.052 * (1 - np.cos(t)) / 2 * 1.6, 0.300 - 0.042 * np.sin(t),
                color=ink, lw=lw * 0.8, solid_capstyle='round', zorder=6)

    # ── whiskers: start at the muzzle edge, fan outward ──
    for sgn in (-1, 1):
        for dy, span, curve in [(0.062, 0.245, 0.070), (0.012, 0.275, 0.012), (-0.038, 0.250, -0.055)]:
            x0 = 0.50 + sgn * 0.205
            u = np.linspace(0, 1, 30)
            ax.plot(x0 + sgn * span * u, 0.335 + dy + curve * u ** 1.7,
                    color=ink, lw=lw * 0.5, alpha=0.8, solid_capstyle='round', zorder=4)

    # ── front paws on the tube edge ──
    for px in (0.385, 0.615):
        ax.add_patch(Ellipse((px, 0.145), 0.115, 0.075, fc=FUR, ec=ink, lw=lw * 0.85, zorder=5))
        for dx in (-0.028, 0.0, 0.028):
            ax.plot([px + dx, px + dx], [0.115, 0.145], color=ink, lw=lw * 0.5, solid_capstyle='round', zorder=6)
    _u = np.linspace(0, 1, 60)
    ax.plot(0.055 + 0.89 * _u, 0.085 + 0.030 * np.sin(np.pi * _u),
            color=ink, lw=lw * 1.2, solid_capstyle='round', zorder=4)

    # ── lick spout: tip in front of the mouth, tube out to the bottom-right, clear of the paws ──
    sp = FancyBboxPatch((0.590, 0.220), 0.455, 0.028, boxstyle='round,pad=0,rounding_size=0.014',
                        fc=SPOUT, ec=ink, lw=lw * 0.8, zorder=7)
    sp.set_transform(Affine2D().rotate_deg_around(0.590, 0.234, -25) + ax.transData)
    ax.add_patch(sp)
    ax.add_patch(Circle((0.587, 0.232), 0.026, fc=WATER, ec=ink, lw=lw * 0.7, zorder=8))
    return ax


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.6, 3.0), dpi=200)
    draw_headfixed_mouse(ax, ps=1.45)
    fig.savefig('/home/leon/dual/overlaps/figures/overlaps/behavior/assets/mouse_cartoon_preview.png',
                bbox_inches='tight', dpi=300)
    print('saved preview → figures/overlaps/behavior/assets/mouse_cartoon_preview.png')
