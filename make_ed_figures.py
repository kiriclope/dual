"""make_ed_figures.py — build the Extended Data set and export it for submission (rewritten 2026-09-15).

Until 2026-09-15 this script pasted whole standalone figures into nine 20–60-inch mosaics whose text
printed at ~1 pt on a Nature page (Leon: "half of them are unreadable"). The ED figures are now native
composites, one script each, drawn from the caches in the house style like the mains, and TRIMMED TO WHAT
THE MANUSCRIPT CITES (Leon: "keep only what is essential for the paper's argumentation") — six figures
instead of nine; the uncited behaviour-cohort, DPA↔GNG-balance and dPCA-push pages are gone.

  ED 1  overlaps/fig_ed_behavior.py         licks (rasters, PSTHs), delay-lick rate, per-animal learning, Fig 1g + trial history
  ED 2  pca/fig_ed_imaging.py               neurons per mouse; per-session held-out decodability (drift check)
  ED 3  pca/fig_ed1_dimensionality.py       dimensionality: spectra, PR, shattering, per-mouse cvPCA, selection effect
  ED 4  pca/fig_ed2_plane.py                the plane per animal and out of context; GNG/test codes; CCGP
  ED 5  pca/fig_ed6_dpca.py                 demixed-PCA trajectories and axis alignment
  ED 6  overlaps/fig_ed3_coupling.py        push/coupling under units, decoders, a fixed axis, a trial-level lick control
  ED 7  overlaps/fig_ed_opto_validation.py  PLACEHOLDER: histology / placements / laser parameters (author-supplied)
  ED 8  overlaps/fig_ed4_chronic.py         chronic silencing of the two control projections
  ED 9  overlaps/fig_ed5_laser7.py          acute ON−OFF coupling over all seven laser mice
  (numbered by first citation, nine figures since 2026-09-16; older scripts keep their names. SI trial counts = tables.)

For each figure: (1) the --nocap build → SVG → 183 mm-wide vector PDF in figures/paper_share/submission/
(rsvg-convert, text kept as glyphs; the ≤247 mm depth rule is printed), (2) the captioned build for the
gallery/artifact, copied to figures/paper_share/ED_Fig{N}.png. Each script writes figures/ed/{png,svg}/ed_fig{N}.

Run:  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python make_ed_figures.py [1 2 ...]
"""
import os, re, shutil, subprocess, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/leon/mambaforge/envs/dual/bin/python'
SUB = 'figures/paper_share/submission'; SHARE = 'figures/paper_share'
os.makedirs(SUB, exist_ok=True)
FIGS = [(1, 'overlaps', 'fig_ed_behavior.py'), (2, 'pca', 'fig_ed_imaging.py'),                 # citation order (2026-09-16):
        (3, 'pca', 'fig_ed1_dimensionality.py'), (4, 'pca', 'fig_ed2_plane.py'),               # older scripts keep their names
        (5, 'pca', 'fig_ed6_dpca.py'), (6, 'overlaps', 'fig_ed3_coupling.py'),
        (7, 'overlaps', 'fig_ed_opto_validation.py'), (8, 'overlaps', 'fig_ed4_chronic.py'), (9, 'overlaps', 'fig_ed5_laser7.py')]
MM = 183.0
only = [int(a) for a in sys.argv[1:] if a.isdigit()]
for n, d, script in FIGS:
    if only and n not in only:
        continue
    t0 = time.time()
    subprocess.run([PY, script, '--nocap'], cwd=d, check=True, capture_output=True, text=True)
    svg = f'figures/ed/svg/ed_fig{n}.svg'
    head = open(svg).read(2000)
    w = float(re.search(r'width="([\d.]+)pt"', head).group(1)); h = float(re.search(r'height="([\d.]+)pt"', head).group(1))
    px = MM / 25.4 * 96
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-w', f'{px:.0f}', '--keep-aspect-ratio', '-o', f'{SUB}/ED_Fig{n}.pdf', svg], check=True)
    hmm = h / w * MM
    subprocess.run([PY, script], cwd=d, check=True, capture_output=True, text=True)       # captioned build
    shutil.copy(f'figures/ed/png/ed_fig{n}.png', f'{SHARE}/ED_Fig{n}.png')
    print(f'ED {n}: {MM:.0f} x {hmm:.0f} mm{"" if hmm <= 247 else "  ** EXCEEDS 247 mm PAGE DEPTH **"}   ({time.time() - t0:.0f} s, two builds)')
print('done —', os.path.abspath(SUB))
