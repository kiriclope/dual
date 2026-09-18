"""(RENUMBERED 2026-09-18 to citation order, ten figures after the unsupervised-geometry page was inserted as Extended Data Fig. 4: this script draws Extended Data Fig. 8.)
fig_ed_opto_validation.py — Extended Data Fig. 8: PLACEHOLDER for the optogenetic and imaging validation page
(companion to Fig. 6a). Built 2026-09-16 after the supplementary review: histology, expression, fibre placement and
laser parameters are author-supplied and not in the pipeline; this page reserves the slot, the panel letters and
the legend so the manuscript can cite it. Replace each box with the real image (same letters) when the material
arrives.
Run:  cd /home/leon/dual/overlaps && python fig_ed_opto_validation.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig8.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os
sys.path.insert(0, '/home/leon/dual/'); sys.path.insert(0, '/home/leon/dual/pca')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import seaborn as sns, matplotlib.pyplot as plt
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.2
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'svg.fonttype': 'none',
})
TITLE_FS = PS*8
NOCAP = '--nocap' in sys.argv[1:]
ED_N = 8
PANELS = [
    ('a', 'ACC injection site', 'Jaws–GFP expression in ACC\n(coronal section, DAPI)\n[AUTHOR: image, scale bar, coordinates]'),
    ('b', 'ACC terminals in mPFC', 'Jaws–GFP axons in prelimbic mPFC\nwith the GRIN lens / imaging FOV\n[AUTHOR: image, lens track]'),
    ('c', 'Fibre and lens placements', 'Reconstructed placements, all mice\n(Jaws n = 5, ChR2 n = 2, ACC-implant n = 2)\n[AUTHOR: atlas overlay]'),
    ('d', 'Chronic cohorts', 'Expression and fibre placement:\nACC→mPFC, ACC somata, PrL→ACC,\ncontrol-illumination mice\n[AUTHOR: images]'),
    ('e', 'Imaging fields of view', 'Example FOV per mouse with ROIs;\nnaïve and expert sessions registered\n[AUTHOR: images, registration metric]'),
    ('f', 'Laser and imaging parameters', 'Wavelength, power at fibre tip, duty cycle,\ndelay-period timing; light-artefact control\n(ON vs OFF fluorescence, no-opsin mice)\n[AUTHOR: table / traces]'),
]
fig = plt.figure(figsize=(10.0, 6.0))
gs = fig.add_gridspec(2, 3, wspace=0.18, hspace=0.35, left=0.04, right=0.98, top=0.89, bottom=0.06)
for i, (L, ttl, txt) in enumerate(PANELS):
    ax = fig.add_subplot(gs[i // 3, i % 3]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linestyle('--'); sp.set_color('0.6'); sp.set_linewidth(0.8)
    ax.set_facecolor('0.96')
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha='center', va='center', fontsize=PS*7.0, color='0.35', linespacing=1.5)
    ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
    ax.text(-0.04, 1.04, L, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')
fig.text(0.5, 0.985, 'PLACEHOLDER — author-supplied validation material to be inserted', ha='center', va='top', fontsize=PS*8, color='#cc3311', fontweight='bold')

CAP = [
    f'Extended Data Fig. {ED_N} | Optogenetic and imaging validation (companion to Fig. 6a). a, Jaws–GFP expression at the '
    'ACC injection site. b, Jaws–GFP-expressing ACC axons in prelimbic mPFC beneath the imaging lens. c, Reconstructed '
    'lens and fibre placements for every imaged mouse. d, Expression and fibre placement in the chronic silencing '
    'cohorts (ACC→mPFC terminals, ACC cell bodies, prelimbic→ACC terminals) and their control-illumination groups. '
    'e, Example fields of view with the extracted ROIs, naïve and expert sessions registered. f, Laser and imaging '
    'parameters, and the light-artefact control (fluorescence on laser-ON against laser-OFF trials in mice without '
    'opsin). [AUTHOR: all panels to be supplied; the layout and letters are fixed so that the text can cite them.]',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
fig.savefig(f'{OUT}/png/ed_fig{ED_N}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig{ED_N}.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig{ED_N}.png')
