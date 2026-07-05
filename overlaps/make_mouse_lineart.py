#!/usr/bin/env python
"""Build `mouse_lineart.svg` — a bold black-and-white *continuous-line* cartoon
derived from the original ~/dual/mouse.svg illustration.

Pipeline: render the original -> darkness-threshold to its own bold outlines ->
morphological close (connect broken strokes) -> erase baked-in labels + stray
apparatus fragments -> flip horizontal so the mouse faces the task -> vectorise
with potrace (potracer, pure-python) into smooth continuous Bézier curves ->
emit an SVG (traced path + crisp vector labels).

Run once; the figure just renders the resulting SVG. Re-run to regenerate.
    /home/leon/mambaforge/envs/dual/bin/python make_mouse_lineart.py
"""
import subprocess, numpy as np
from PIL import Image, ImageFilter
import potrace

SRC = '/home/leon/dual/mouse.svg'
OUT = '/home/leon/dual/mouse_lineart.svg'
RENDER_W = 1800                      # working resolution
s = RENDER_W / 1400.0                # boxes below were estimated at 1400 px wide

raw = subprocess.run(['rsvg-convert', '-w', str(RENDER_W), SRC, '-o', '/tmp/_ml_raw.png'],
                     check=True)
g = Image.open('/tmp/_ml_raw.png').convert('RGBA')
bg = Image.new('RGBA', g.size, (255, 255, 255, 255)); bg.alpha_composite(g)
a = np.array(bg.convert('L'))
bw = Image.fromarray(np.where(a < 135, 0, 255).astype('uint8'))
bw = bw.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))   # close gaps
arr = np.array(bw)

def erase(boxes):
    for x0, y0, x1, y1 in boxes:
        arr[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = 255

erase([(30, 70, 250, 170), (470, 10, 965, 120), (80, 780, 350, 876)])   # baked-in labels
arr = arr[:, ::-1].copy()                                               # flip -> face task
erase([(1258, 230, 1320, 430),      # stray floating vertical tick
       (682, 600, 802, 765),        # stray platform bar (centre)
       (0, 826, 1400, 876)])        # full-width baseline / table edge

# potracer's concatenated even-odd output fills the COMPLEMENT of the traced mask,
# so trace the paper (light) and the filled path lands on the ink -> black on white.
paper = arr >= 128
path = potrace.Bitmap(paper).trace(turdsize=30, alphamax=1.0, opticurve=1, opttolerance=0.2)
H, W = arr.shape

def xy(p):
    return (p.x, p.y) if hasattr(p, 'x') else (p[0], p[1])

P = []
for curve in path:
    x0, y0 = xy(curve.start_point); P.append(f"M{x0:.1f},{y0:.1f}")
    for seg in curve.segments:
        ex, ey = xy(seg.end_point)
        if seg.is_corner:
            cx, cy = xy(seg.c); P.append(f"L{cx:.1f},{cy:.1f}L{ex:.1f},{ey:.1f}")
        else:
            a1, b1 = xy(seg.c1); a2, b2 = xy(seg.c2)
            P.append(f"C{a1:.1f},{b1:.1f} {a2:.1f},{b2:.1f} {ex:.1f},{ey:.1f}")
    P.append("z")

labels = "".join(
    f'<text x="{x}" y="{y}" font-size="{fs}" font-family="DejaVu Sans" '
    f'font-weight="bold" text-anchor="middle">{t}</text>'
    for x, y, t, fs in [(830, 72, 'Head-fixed', 60), (1360, 96, 'Odor', 62),
                        (1505, 1015, 'Water', 62)])
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="#fff"/>'
       f'<path d="{"".join(P)}" fill="#000" fill-rule="evenodd"/>{labels}</svg>')
open(OUT, 'w').write(svg)
print('wrote', OUT, f'({W}x{H}, {len(P)} path ops)')
