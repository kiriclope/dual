"""build_ed_artifact.py — compose the shared EXTENDED DATA artifact page from the rendered ED pages.

  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python make_ed_figures.py     # renders figures/ed/png/*
  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python build_ed_artifact.py

then publish the result with the Artifact tool, passing the URL so it updates in place:
  file_path = figures/paper_share/artifact_build/mpfc_dual_ed_v1.html
  url       = https://claude.ai/code/artifact/95df947e-a8d6-452f-8d6f-431ea9cf7dcb

Inputs  figures/ed/png/ed_fig{1..6}.png   (captions and panel letters are drawn INTO these
        pages by make_ed_figures.py — the HTML only adds an eyebrow, a title and a one-line claim)
Style   reuses the <style> block of figures/paper_share/artifact_build/mpfc_dual_figures_v1.html, so the two
        shared galleries stay visually identical
Output  figures/paper_share/artifact_build/mpfc_dual_ed_v1.html

Web copies are max-dim 2800 + quantize(256) (~0.1-0.6 MB each) to stay under the 16 MB artifact cap; the
full-resolution PNG/PDF of every page travel in dual_paper_figures.zip.

Rebuilt 2026-09-09 after the original lived in a Claude job tmp dir that was cleaned up. Keep the titles and
claims below in step with the captions in make_ed_figures.py.
"""
import base64
import io
import os
import re

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = '/home/leon/dual'
BUILD = f'{ROOT}/figures/paper_share/artifact_build'
FIGPAGE = f'{BUILD}/mpfc_dual_figures_v1.html'
OUT = f'{BUILD}/mpfc_dual_ed_v1.html'
SRC = f'{ROOT}/figures/ed/png'

# (file stem, eyebrow, title, one-line claim) — titles/claims mirror the captions in make_ed_figures.py
SPECS = [
    ('ed_fig1', 'Extended Data Fig. 1', 'Licking and per-animal learning in the imaged cohort',
     'Lick rasters and rates per trial type, the fall of the delay lick with training, per-animal learning curves, and Fig. 1g with trial-history covariates.'),
    ('ed_fig2', 'Extended Data Fig. 2', 'The imaged population and the session-by-session stability of its codes',
     'Neurons per mouse (3,319) and held-out sample/choice decodability in every session: no session-to-session drift behind the cross-stage transfer.'),
    ('ed_fig3', 'Extended Data Fig. 3', 'Dimensionality: provenance and robustness',
     'Twelve-condition spectra and participation ratios at Fig. 2 windows, the 462-dichotomy shattering dimension with across-animal intervals, the per-mouse cvPCA companion, and the correct-trial selection effect behind Fig. 2c.'),
    ('ed_fig4', 'Extended Data Fig. 4', 'The sample x choice plane, per animal and out of context',
     'GNG and test codes over time, per-mouse CCGP on the canonical windows, the out-of-context plane test, and the plane-ablation grid in every animal.'),
    ('ed_fig5', 'Extended Data Fig. 5', 'The demixed-PCA decomposition gives the same picture',
     'Single-axis time courses sharpen without reorganizing; no change in the angles between axes is resolved across animals (refit mouse bootstrap).'),
    ('ed_fig6', 'Extended Data Fig. 6', 'The push and the learning coupling under other units, other decoders, a fixed axis and a trial-level lick control',
     'All on held-out CCGD decision functions: the coupling holds under every unit and decoder except LDA, reverses on one axis fitted to both stages, and holds on no-lick trials.'),
    ('ed_fig7', 'Extended Data Fig. 7', 'Optogenetic and imaging validation (PLACEHOLDER)',
     'Author-supplied histology, expression, placements, fields of view and laser parameters - slots reserved.'),
    ('ed_fig8', 'Extended Data Fig. 8', 'Chronic silencing of the two control projections',
     'ACC cell bodies: no deficit detected; prelimbic-to-ACC terminals: GNG impaired, DPA spared - bounded nulls that do not reproduce the ACC-to-mPFC DPA deficit.'),
    ('ed_fig9', 'Extended Data Fig. 9', 'The acute laser ON-OFF coupling over all seven laser mice',
     'With the Fig. 6 estimator: GNG arm rho = -0.94, p = 0.002; the DPA trend is carried by the two ChR2 mice (n = 7).'),
]

style = re.search(r'<style>.*?</style>', open(FIGPAGE, encoding='utf-8').read(), re.S)
assert style, f'no <style> block in {FIGPAGE}'

parts = ['<title>mPFC Dual-Task Extended Data</title>', style.group(0), '''<main>
<header>
<p class="kicker">Extended Data &middot; composed pages</p>
<h1>Extended Data figures &mdash; mPFC population geometry, dual task</h1>
<p class="sub">The nine Extended Data figures plus the Supplementary trial-count figure, composed at
native panel resolution with full justified captions and verified statistics in-page. Companion to the
<b>mPFC Dual-Task Figures</b> and <b>mPFC Dual-Task Draft</b> artifacts.</p>
<div class="formats"><span class="tag">PNG</span><span class="tag">PDF</span>
This page shows review-quality rasters &mdash; click any figure to view at actual size. Full-resolution
PNG and PDF of every page travel in the companion <b>dual_paper_figures.zip</b>.</div>
</header>''']

total = 0
for stem, eyebrow, title, claim in SPECS:
    im = Image.open(f'{SRC}/{stem}.png').convert('RGB')
    im.thumbnail((2800, 2800), Image.Resampling.LANCZOS)
    im = im.quantize(256, method=Image.Quantize.MEDIANCUT)
    buf = io.BytesIO(); im.save(buf, 'PNG', optimize=True)
    total += buf.tell()
    print(f'{stem}: {buf.tell() / 1e6:.2f} MB')
    b64 = base64.b64encode(buf.getvalue()).decode()
    parts.append(f'''<section class="fig">
<div class="eyebrow">{eyebrow}</div>
<h2>{title}</h2>
<p class="claim">{claim}</p>
<div class="imgwrap"><img src="data:image/png;base64,{b64}" alt="{eyebrow} — {title}"
     tabindex="0" title="Click to toggle actual size"></div>
</section>''')

parts.append('''<footer>Composed from the component renders (make_ed_figures.py). Captions and
panel letters in-page are the canon and match the manuscript's Extended Data section.</footer>
</main>
<script>
document.querySelectorAll('.imgwrap img').forEach(function (im) {
  function flip() { im.classList.toggle('full'); }
  im.addEventListener('click', flip);
  im.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip(); }
  });
});
</script>''')

page = '\n'.join(parts) + '\n'
os.makedirs(BUILD, exist_ok=True)
open(OUT, 'w', encoding='utf-8').write(page)
print(f'total embedded: {total / 1e6:.2f} MB; html: {len(page) / 1e6:.2f} MB -> {OUT}')
assert len(page) < 15.5e6, 'page is near the 16 MB artifact cap — shrink the web copies'
