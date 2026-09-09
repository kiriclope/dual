"""swap_figure_embeds.py — replace one or more embedded main-figure PNGs in the shared FIGURES artifact page.

Durable home (2026-09-09): this script and the page used to live in a Claude job tmp dir, which was cleaned up
between sessions and took them with it. The script now lives in the repo (tracked, beside make_submission_figs.py)
and the 4 MB page beside the figures it embeds, in `figures/paper_share/artifact_build/` (gitignored).

  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python swap_figure_embeds.py Fig4_learning [Fig2_geometry ...]

then publish the page with the Artifact tool, passing the artifact URL so it updates in place:
  file_path = figures/paper_share/artifact_build/mpfc_dual_figures_v1.html
  url       = https://claude.ai/code/artifact/324c8888-f67a-4a74-b581-d51ef0c73abd

Web copies are max-dim 2800 + quantize(256) (~0.6-0.9 MB each) so the page stays under the 16 MB artifact cap;
full-resolution PNG/SVG/PDF travel in dual_paper_figures.zip. If the page is ever lost again, recover it with
the Artifact tool's `action: "read"` on that URL and strip the `<!doctype …><body>` skeleton (it is re-added at
publish time) — that is exactly how this copy was made.
"""
import re, base64, io, sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = '/home/leon/dual/figures/paper_share'
BUILD = f'{ROOT}/artifact_build'
PAGE = f'{BUILD}/mpfc_dual_figures_v1.html'
ORDER = ['Fig1_behaviour', 'Fig2_geometry', 'Fig3_one_manifold', 'Fig4_learning', 'Fig6_opto']

targets = sys.argv[1:]
if not targets:
    sys.exit(f'usage: swap_figure_embeds.py <{" | ".join(ORDER)}> [...]')
unknown = [t for t in targets if t not in ORDER]
assert not unknown, f'unknown figure(s): {unknown}; expected any of {ORDER}'

page = open(PAGE, encoding='utf-8').read()
ms = list(re.finditer(r'<img src="data:image/png;base64,[A-Za-z0-9+/=]+"', page))
assert len(ms) == len(ORDER), f'expected {len(ORDER)} embedded images, found {len(ms)}'


def enc(name):
    im = Image.open(f'{ROOT}/{name}.png').convert('RGB')
    im.thumbnail((2800, 2800), Image.Resampling.LANCZOS)
    im = im.quantize(256, method=Image.Quantize.MEDIANCUT)
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
    return '<img src="data:image/png;base64,' + base64.b64encode(b.getvalue()).decode() + '"', im.size, len(b.getvalue())


out, last = [], 0
for i, m in enumerate(ms):
    out.append(page[last:m.start()])
    if ORDER[i] in targets:
        tag, size, nb = enc(ORDER[i]); out.append(tag)
        print(f'swapped {ORDER[i]}: {size} {nb / 1e6:.2f} MB')
    else:
        out.append(m.group(0))
    last = m.end()
out.append(page[last:])
page = ''.join(out)
open(PAGE, 'w', encoding='utf-8').write(page)
print(f'figures html {len(page) / 1e6:.2f} MB -> {PAGE}')
