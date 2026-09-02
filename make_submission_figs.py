"""make_submission_figs.py — export the five mains as Nature-final artwork (2026-09-03).

Per the Nature 'Guide to preparing final artwork' (nature.com/documents/nature-final-artwork.pdf):
183 mm double-column width, supplied AT print size, editable-vector PDF, Arial/Helvetica text
5-7 pt at final size, 8 pt bold lowercase panel letters, legend NOT inside the figure.

For each main this script (1) renders the --nocap build (no in-figure caption; the caption stays
in the gallery/artifact builds and the manuscript carries the legend), (2) converts the SVG to a
PDF scaled to exactly 183 mm width (rsvg-convert; text preserved as glyphs), (3) re-runs the
normal captioned build so the repo PNG/SVG keep their caption, and (4) prints the final page
size in mm so the <=247 mm depth rule is auditable.

Run:  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python make_submission_figs.py
Output: figures/paper_share/submission/Fig*.pdf
"""
import os, re, subprocess, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/leon/mambaforge/envs/dual/bin/python'
OUT = 'figures/paper_share/submission'
os.makedirs(OUT, exist_ok=True)

FIGS = [
    ('Fig1_behaviour', 'overlaps', 'fig_behavior_main.py', [],
     'overlaps/figures/overlaps/behavior/svg/behavior_main.svg'),
    ('Fig2_geometry', 'pca', 'fig_dimensionality_main.py', [],
     'pca/figures/pseudo/dimensionality/svg/fig_dimensionality_main.svg'),
    ('Fig3_one_manifold', 'pca', 'fig_manifold_main.py', ['--nopca'],
     'pca/figures/pseudo/dimensionality/svg/fig_manifold_main.svg'),
    ('Fig4_learning', 'overlaps', 'fig_overlaps_main_native.py', [],
     'overlaps/figures/overlaps/main/svg/fig_overlaps_main_ab_dpaact.svg'),
    ('Fig6_opto', 'overlaps', 'fig_behavior_opto_main.py', [],
     'overlaps/figures/overlaps/behavior/svg/behavior_opto_main.svg'),
]

MM = 183.0
only = sys.argv[1] if len(sys.argv) > 1 else None
for name, d, script, flags, svg in FIGS:
    if only and only not in name:
        continue
    subprocess.run([PY, script, '--nocap', *flags], cwd=d, check=True,
                   capture_output=True, text=True)
    head = open(svg).read(2000)
    w = float(re.search(r'width="([\d.]+)pt"', head).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', head).group(1))
    px = MM / 25.4 * 96                          # rsvg -w is 96-dpi px
    pdf = f'{OUT}/{name}.pdf'
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-w', f'{px:.0f}', '--keep-aspect-ratio',
                    '-o', pdf, svg], check=True)
    hmm = h / w * MM
    flag = '' if hmm <= 247 else '  ** EXCEEDS 247 mm PAGE DEPTH **'
    print(f'{name}: {MM:.0f} x {hmm:.0f} mm{flag}')
    subprocess.run([PY, script, *flags], cwd=d, check=True,
                   capture_output=True, text=True)   # restore the captioned build
print('done — submission PDFs in', os.path.abspath(OUT))
