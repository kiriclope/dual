"""
fig_overlaps_main.py — assemble the overlaps MAIN paper figure from the
individual panel PNG/SVGs produced by the per-analysis scripts.

This is a LAYOUT PROOF: it stacks the already-rendered panels (each of which is
itself a wide multi-subplot strip) into one labelled figure so the arc reads at a
glance. Final publication assembly is vector editing from the individual SVGs
listed in PANELS below — this script just fixes the order, letters, and the
panel→source mapping so it is reproducible.

LOCKED MAIN FIGURE = the combined late-delay + test readout axis (trainLD_TEST,
bins 45-59). The sample code is invariant across LD→test, so the combined axis is
clean; because the window spans the test epoch the test code (A) is valid (no
pre-test confound, unlike the all-delay / all-ld variants).

Arc (overlaps-only, sample × choice CCGD frame) — the four main results, all trainLD_TEST:
  A  sample / choice / test / task codes, 1-D over the trial (test code valid;
     task code lives on the choice axis)
  B  the no-lick push: DPA state Naive→Expert in the sample × choice plane
  C  well deepening: per-mouse late-delay depth Naive→Expert + maximal-LMM stars
     (pooled dz=-0.53, 8/9 mice, LMM p=0.098 — a strong TREND; the deepening
     reaches p=0.024 only on the full trainDELAY window: run --delay)
  D  Δ DPA accuracy vs Δ depth (ρ=-0.67, p=0.050, *) and Δ GNG accuracy vs Δ depth
     (null → the depth link is DPA-specific)
  E  laser ON-OFF causal analog of D: Δ(on-off) depth vs Δ(on-off) accuracy, Expert,
     7 laser mice (● Jaws inhibit / ▲ ChR excite). GNG a robust between-animal
     correlation (ρ≈-0.9); DPA null. See docs/overlaps/laser_onoff.md.
     Only added on axes with a laser twin (ld_test / ld / delay).

Axis variants for robustness (all panels on one axis unless noted):
  (default) ld_test  — locked main figure (A valid, D sig, C strong trend)
  --mixed            — C on trainDELAY (p=0.024, sig) + D on trainTEST (ρ=-0.67)
  --test / --delay / --ld — single-axis robustness (see docs/overlaps/overview.md)

Output: figures/overlaps/main/{png,svg}/fig_overlaps_main[_withF].{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main.py [--with-scatter]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

matplotlib.rcParams.update({
    'figure.dpi':   150,
    'savefig.dpi':  300,
    'font.family':  'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

FIG_ROOT = 'figures/overlaps'
DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'

# Axis variants (all panels on ONE axis): --test / --delay / --ld.
# Default = 'mixed': codes on trainTEST (test read DURING test), push+deepening on the
# principled trainDELAY axis (where the deepening is significant, p=0.024), scatter trainTEST.
AXES = {  # AXIS: (codes_epoch, push_ax, deepset, scatter_ax)
    'mixed':   ('test',    'trainDELAY',  'delay',   'trainTEST'),
    'test':    ('test',    'trainTEST',   'test',    'trainTEST'),
    'delay':   ('delay',   'trainDELAY',  'delay',   'trainDELAY'),
    'ld':      ('ld',      'trainLD',     'ld',      'trainLD'),
    'ld_test': ('ld_test', 'trainLD_TEST','ld_test', 'trainLD_TEST'),
    'choice':          ('choice',          'trainCHOICE',          'choice',          'trainCHOICE'),
    'test_choice':     ('test_choice',     'trainTEST_CHOICE',     'test_choice',     'trainTEST_CHOICE'),
    'ld_test_choice':  ('ld_test_choice',  'trainLD_TEST_CHOICE',  'ld_test_choice',  'trainLD_TEST_CHOICE'),
    # narrow LD/TEST boundary: last 0.5 s of LD + first 0.5 s of TEST (bins 51-56)
    'ldtest05':        ('ldtest05',        'trainLDTEST05',        'ldtest05',        'trainLDTEST05'),
}
AXIS = next((a for a in ('ld_test_choice', 'test_choice', 'choice', 'ldtest05', 'ld_test',
                         'mixed', 'test', 'delay', 'ld')
             if f'--{a}' in sys.argv[1:]), 'ld_test')   # default = locked main figure
CODES_EP, PUSH_AX, DEEPSET, SCAT_AX = AXES[AXIS]

# --ab : swap panels D and E to the A&B-independent twins (each mouse contributes
# two dots, sample A and sample B — doubles n on both scatters). Same axis/stage.
AB     = '--ab' in sys.argv[1:]
AB_SUF = '_ab' if AB else ''

# (letter, source PNG, one-line description) — SVG twins live beside each PNG.
panels = [
    ('A', f'{FIG_ROOT}/codes1d/{CODES_EP}/png/overlaps_codes1d_grandmean_naive_expert.png',
     f'sample/choice/test/task codes ({CODES_EP} epoch, Naive+Expert)'),
    ('B', f'{FIG_ROOT}/traj2d/all/png/{DUM}_{PUSH_AX}_dpaonly.png',
     f'no-lick push: DPA state Naive->Expert (sample x choice, {PUSH_AX})'),
    ('C', f'{FIG_ROOT}/nolick_push/png/{DUM}_nolick_push_paired_{DEEPSET}_all.png',
     f'well deepening: late-delay depth Naive->Expert + maximal-LMM ({PUSH_AX})'),
    ('D', f'{FIG_ROOT}/scatter_perf/{SCAT_AX}/png/{DUM}_{SCAT_AX}_dpa_panel{AB_SUF}.png',
     f'Delta perf vs Delta DPA-depth: DPA & GNG specificity ({SCAT_AX}{AB_SUF})'),
]

# E — laser ON-OFF causal analog of D. Only the ld_test/ld/delay axes have a laser
# twin (plot_scatter_laser.py); Expert-stage panel to match the figure's Expert focus.
LASER_AX   = {'ld_test': 'ld_test', 'ld': 'ld', 'delay': 'delay',
              'test': 'test', 'ldtest05': 'ldtest05'}
LASER_MODE = 'expert'
_lax = LASER_AX.get(AXIS)
if _lax:
    panels.append(
        ('E', f'{FIG_ROOT}/scatter_laser/png/{DUM}_laser_targets_choice_onoff_{_lax}_{LASER_MODE}{AB_SUF}.png',
         f'laser ON-OFF causal: Delta depth vs Delta perf, DPA & GNG ({_lax}, {LASER_MODE}{AB_SUF})'))

# ── Load images, drop any missing ─────────────────────────────────────────────
loaded = []
for letter, path, desc in panels:
    if not os.path.exists(path):
        print(f'  !! MISSING {letter}: {path} — skipped')
        continue
    img = mpimg.imread(path)
    loaded.append((letter, path, desc, img))
    print(f'  {letter}  {img.shape[1]}x{img.shape[0]}  {path}')

if not loaded:
    sys.exit('no panels found — run the per-analysis scripts first')

# ── Stack panels as full-width rows, height ∝ each panel's aspect ──────────────
FIG_W  = 8.5                                   # inches (double-column proof)

aspects = [img.shape[0] / img.shape[1] for *_, img in loaded]      # h/w per panel
row_h   = [FIG_W * a for a in aspects]
pad     = 0.12                                  # inches between rows
fig_h   = sum(row_h) + pad * (len(loaded) + 1)

fig = plt.figure(figsize=(FIG_W, fig_h))
y = 1.0
for (letter, path, desc, img), h in zip(loaded, row_h):
    frac = h / fig_h
    ypad = pad / fig_h
    y -= ypad
    ax = fig.add_axes((0.06, y - frac, 0.92, frac))
    ax.imshow(img)
    ax.axis('off')
    # bold panel letter just outside the top-left of the row
    fig.text(0.012, y, letter, fontsize=15, fontweight='bold',
             ha='left', va='top')
    y -= frac

OUT_DIR = 'figures/overlaps/main'          # inside the overlaps figures tree
os.makedirs(os.path.join(OUT_DIR, 'png'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'svg'), exist_ok=True)
TAG = {'ld_test': '', 'mixed': '_mixed', 'test': '_trainTEST',
       'delay': '_trainDELAY', 'ld': '_trainLD',   # ld_test = canonical (no tag)
       'choice': '_trainCHOICE', 'test_choice': '_trainTEST_CHOICE',
       'ld_test_choice': '_trainLD_TEST_CHOICE', 'ldtest05': '_trainLDTEST05'}
tag = TAG[AXIS] + AB_SUF
for ext in ('png', 'svg'):
    out = os.path.join(OUT_DIR, ext, f'fig_overlaps_main{tag}.{ext}')
    fig.savefig(out, bbox_inches='tight')
    print(f'saved {os.path.abspath(out)}')
plt.close(fig)

print('\nPanel -> source map:')
for letter, path, desc, _ in loaded:
    print(f'  {letter}: {desc}\n       {path}  (+ svg twin)')
print('\nProof only — final assembly = vector edit from the SVG twins above.')
