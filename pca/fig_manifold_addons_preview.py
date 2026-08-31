"""PREVIEW of the candidate 'one manifold' support panels for Fig 3 (2026-08-31) — render, decide,
then integrate the keepers into fig_manifold_main.py.

  E  SUFFICIENCY by decoding: accuracy from ONLY the 2 plane coordinates vs the full space, each
     variable at its own canonical window (PLANE_DEC). Shown as the in-plane fraction of
     above-chance accuracy, Naive -> Expert. sample/choice ~1 (fully in-plane; >1 = the 2-D
     projection denoises); dist@md RISES 0.53 -> 0.95 (learning pulls the dist code into the
     plane); test ~0 (out-of-plane — the third decision axis; the claim's honest boundary).
  F  the frame is the SAME across learning, by cross-stage decoding (XSTAGE_DEC): train sample/
     choice decoders in one stage, test in the other (registered neurons). 2x2 accuracy matrices.
  G  the frame does not rotate within the trial (AXIS_TIME): |cos(axis(t), axis(ref window))| of
     the per-mouse overlaps decoder weights, mean±SEM (n=9).

DEAD-ENDS already logged (do not revive): PLANE_VAR (axis-direction noise at rel~0.15 attenuates
the in-plane variance fraction to 0.04-0.21 — meaningless) and the corrected cross-stage cosine
(explodes >1 at those reliabilities). Decoding is the robust probe for both questions.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_manifold_addons_preview.py
"""
import matplotlib; matplotlib.use('Agg')
import os, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7,
    'ytick.major.width': 0.7})
TITLE_FS = 8
VAR_COL = {'sample': '#332288', 'test': '#377eb8', 'choice': '#4daf4a', 'dist': '#ee7733'}
SC = {'Naive': '0.55', 'Expert': '#332288'}

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
PD = RES['PLANE_DEC_nopca']; XD = RES['XSTAGE_DEC_nopca']
AT = RES['AXIS_TIME']; ATX = np.asarray(RES['AXIS_TIME_X'])
PM = RES['PM_PLANE_nopca']
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
GROUP = {**{m: 'Jaws' for m in MICE[:5]}, **{m: 'ChR' for m in MICE[5:7]},
         **{m: 'ACC' for m in MICE[7:]}}
GMARK = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
_pal = sns.color_palette('tab10', n_colors=len(MICE))
MCOL = {m: _pal[i] for i, m in enumerate(MICE)}

fig = plt.figure(figsize=(13.2, 10.2))
gs = fig.add_gridspec(3, 12, wspace=1.1, hspace=0.50, height_ratios=[1.0, 1.0, 1.55],
                      left=0.06, right=0.985, top=0.94, bottom=0.06)

# ── E: PER-MOUSE decoding, Naive (x) vs Expert (y) — rows: plane ONLY / OUT-OF-PLANE (plane
#      component removed) / full space (user design 2026-08-31 + out-of-plane arm). DOUBLE
#      DISSOCIATION per animal: sample & choice ≈ full in-plane and COLLAPSE out-of-plane;
#      test ≈ chance in-plane and keeps its full accuracy out-of-plane. PM_PLANE also carries
#      'dist' (undisplayed 4th column: out ≈ full — its own axis is mostly ⊥ the plane). ──
from scipy.stats import wilcoxon as _wilc
E_VARS = ['sample', 'test', 'choice']                       # timeline order (user spec)
lo, hi = 0.42, 0.95
gsE = gs[0:2, 0:6].subgridspec(3, 3, wspace=0.25, hspace=0.22)
axEs = []
for r, (key, rowlab) in enumerate([(0, 'plane only (2-D)'), (2, 'out-of-plane'),
                                   (1, 'full space')]):
    for c, vn in enumerate(E_VARS):
        ax = fig.add_subplot(gsE[r, c]); axEs.append(ax)   # display order = enumerate order
        ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        nv, ev = [], []
        for m in MICE:
            if (m, 'Naive') not in PM or (m, 'Expert') not in PM:
                continue
            if vn not in PM[(m, 'Naive')] or vn not in PM[(m, 'Expert')]:
                continue
            a = PM[(m, 'Naive')][vn][key]; b = PM[(m, 'Expert')][vn][key]
            nv.append(a); ev.append(b)
            ax.scatter(a, b, s=30, color=MCOL[m], marker=GMARK[GROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        nv, ev = np.array(nv), np.array(ev)
        ax.scatter(nv.mean(), ev.mean(), s=64, color='k', marker='D', edgecolors='w',
                   linewidths=0.6, zorder=5)
        p = float(_wilc(ev, nv).pvalue)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
        if c:
            ax.tick_params(labelleft=False)
        if r == 0:
            ax.set_title(vn, loc='left', fontsize=7)
        if r < 2:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel(f'{rowlab}\nExpert', fontsize=7)
        if r == 2 and c == 1:
            ax.set_xlabel('accuracy — Naive', fontsize=7)
        ax.text(0.05, 0.96, f'Δ={ev.mean() - nv.mean():+.02f}\np={p:.2f}', transform=ax.transAxes,
                va='top', ha='left', fontsize=5.6, color='0.3')
        print(f'E: {rowlab:16s} {vn:7s} {nv.mean():.2f} -> {ev.mean():.2f}  p={p:.2f}')
axE = axEs[0]

# ── E2: the E-block averages, mean ± SEM across mice (stage-averaged per mouse — the
#      Naive/Expert differences are n.s. throughout E). The double dissociation as bars:
#      sample/choice lose little from full→plane and collapse out-of-plane; test is at chance
#      in-plane and keeps its full accuracy out-of-plane. ──
axE2 = fig.add_subplot(gs[1, 6:8])
SPACES = [('plane', 0), ('out-of-plane', 2), ('full', 1)]
for gi, vn in enumerate(E_VARS):
    for si, (slab, key) in enumerate(SPACES):
        vals = []
        for m in MICE:
            if all((m, st) in PM and vn in PM[(m, st)] for st in ['Naive', 'Expert']):
                vals.append(np.mean([PM[(m, st)][vn][key] for st in ['Naive', 'Expert']]))
        vals = np.array(vals)
        x = gi + (si - 1) * 0.27
        col = VAR_COL[vn]
        if slab == 'plane':
            sty = dict(facecolor=col, edgecolor=col)
        elif slab == 'out-of-plane':
            sty = dict(facecolor='w', edgecolor=col, hatch='///')
        else:
            sty = dict(facecolor=col, alpha=0.35, edgecolor=col)
        axE2.bar(x, vals.mean(), 0.25, lw=0.9, zorder=2, **sty)
        axE2.errorbar(x, vals.mean(), yerr=vals.std(ddof=1) / np.sqrt(len(vals)),
                      color='k', capsize=2, lw=0.9, zorder=3)
    # paired tests for the record (printed, not drawn — star policy)
    trip = {slab: np.array([np.mean([PM[(m, st)][vn][key] for st in ['Naive', 'Expert']])
                            for m in MICE if all((m, st) in PM and vn in PM[(m, st)]
                                                 for st in ['Naive', 'Expert'])])
            for slab, key in SPACES}
    p_pf = float(_wilc(trip['plane'], trip['full']).pvalue)
    p_of = float(_wilc(trip['out-of-plane'], trip['full']).pvalue)
    print(f'E2: {vn:7s} plane {trip["plane"].mean():.2f} out {trip["out-of-plane"].mean():.2f} '
          f'full {trip["full"].mean():.2f}  plane-vs-full p={p_pf:.3f}  out-vs-full p={p_of:.3f}')
axE2.axhline(0.5, ls='--', color='0.6', lw=0.8, zorder=1)
axE2.set_xticks(range(len(E_VARS))); axE2.set_xticklabels(E_VARS, fontsize=6.6)
axE2.set_ylim(0.45, 0.80); axE2.set_yticks([0.5, 0.6, 0.7, 0.8])
axE2.set_ylabel('accuracy (mean ± SEM, n = 9)', fontsize=7)
axE2.set_title('spaces compared', loc='left', fontsize=TITLE_FS)
from matplotlib.patches import Patch as _Patch
axE2.legend(handles=[_Patch(fc='0.35', ec='0.35', label='plane (2-D)'),
                     _Patch(fc='w', ec='0.35', hatch='///', label='out-of-plane'),
                     _Patch(fc='0.35', alpha=0.35, ec='0.35', label='full')],
            frameon=False, fontsize=5.4, loc='upper left', handlelength=1.1,
            handletextpad=0.4, labelspacing=0.3)

# ── F: cross-stage decoding 2×2 per code ──
for k, vn in enumerate(['sample', 'choice']):
    ax = fig.add_subplot(gs[0, 6 + 3 * k:9 + 3 * k])
    M = np.array([[np.mean(XD[(vn, a, b)]) for b in ['Naive', 'Expert']]
                  for a in ['Naive', 'Expert']])
    ax.imshow(M, cmap='Reds', vmin=0.5, vmax=1.0, aspect='equal')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                    color='w' if M[i, j] > 0.82 else 'k')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=5.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Naive', 'Expert'] if k == 0 else [], fontsize=5.8)
    off = np.mean([M[0, 1], M[1, 0]]); dia = np.mean([M[0, 0], M[1, 1]])
    ax.set_title(f'{vn} axis   (transfer/within {(off - .5) / (dia - .5):.2f})',
                 loc='left', fontsize=7)
    if k == 0:
        ax.set_ylabel('train stage', fontsize=7)
    ax.set_xlabel('test stage', fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(True)
    print(f'F: {vn} within {dia:.2f} cross {off:.2f} ratio {(off - .5) / (dia - .5):.2f}')

# ── G: the axes do not rotate within the trial ──
axG = fig.add_subplot(gs[1, 8:12])
for vn in ['sample', 'choice', 'dist']:
    d = AT[(vn, 'Expert')]
    axG.plot(ATX, d['mean'], color=VAR_COL[vn], lw=1.4, label=vn)
    axG.fill_between(ATX, d['mean'] - d['sem'], d['mean'] + d['sem'],
                     color=VAR_COL[vn], alpha=0.2, lw=0)
for nm, lo, hi, col in [('sample', 2.0, 3.0, VAR_COL['sample']),
                        ('distractor', 4.5, 5.5, '#cc3311'),
                        ('cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]:
    axG.axvspan(lo, hi, color=col, alpha=0.08, lw=0)
axG.set_xlim(0, 12); axG.set_ylim(0, 1.0)
axG.set_xlabel('time (s)', fontsize=7)
axG.set_ylabel('|cos( axis(t), axis(ref) )|', fontsize=7)
axG.set_title('axes stable in time (Expert, n=9)', loc='left', fontsize=TITLE_FS)
axG.legend(frameon=False, fontsize=5.8, loc='upper left', handlelength=1.2)

# ── H: panel-A-style code traces read THROUGH the plane (solid) vs the full axis (dashed).
#      sample/choice coincide BY CONSTRUCTION (they span the plane); the content is dist (about
#      half the separation flows through the plane, more in Expert) and test (flattens). ──
PT = RES['PLANE_TRAJ_nopca']
XT = np.linspace(0, 14, 84)
H_CODES = [('sample', ['#332288', '#44AA99'], ['Odor A', 'Odor B']),
           ('dist', ['#2ca02c', '#1f77b4'], ['NoGo', 'Go']),
           ('test', ['#CC6677', '#999933'], ['Odor C', 'Odor D']),
           ('choice', ['#377eb8', '#4daf4a'], ['No lick', 'Lick'])]
gsH = gs[2, 0:12].subgridspec(2, 4, wspace=0.30, hspace=0.14)
# shared y-limits per code column (Naive vs Expert comparable — the panel-A lesson)
YLK = {}
for k, (cname, _, _) in enumerate(H_CODES):
    vals = np.concatenate([PT[(st, cname, cls, wh)] for st in ['Naive', 'Expert']
                           for cls in (0, 1) for wh in ('full', 'plane')])
    pad = 0.06 * (vals.max() - vals.min())
    YLK[k] = (vals.min() - pad, vals.max() + pad)
axH0 = None
for r, stage in enumerate(['Naive', 'Expert']):
    for k, (cname, cols, labs) in enumerate(H_CODES):
        ax = fig.add_subplot(gsH[r, k])
        axH0 = axH0 or ax
        ax.set_ylim(*YLK[k])
        for nm, l0, h0, col in [('sample', 2.0, 3.0, '#332288'), ('distractor', 4.5, 5.5, '#cc3311'),
                                ('cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]:
            ax.axvspan(l0, h0, color=col, alpha=0.08, lw=0)
        for cls in (0, 1):
            ax.plot(XT, PT[(stage, cname, cls, 'full')], ls='--', color=cols[cls], lw=1.0,
                    alpha=0.55)
            ax.plot(XT, PT[(stage, cname, cls, 'plane')], ls='-', color=cols[cls], lw=1.5,
                    label=labs[cls])
        ax.axhline(0, ls='--', color='k', lw=0.5, zorder=0)
        ax.set_xlim(0, 12)
        if r == 0:
            ax.set_title(f'{cname} code', loc='left', fontsize=TITLE_FS)
            ax.tick_params(labelbottom=False)
            ax.legend(frameon=False, fontsize=5.4, handlelength=1.2, loc='upper left')
        else:
            ax.set_xlabel('time (s)', fontsize=7)
        if k == 0:
            ax.set_ylabel(f'{stage}\nprojection (z)', fontsize=7)
        if r == 0 and k == 1:                       # in the dist panel: both styles visible there
            ax.text(0.98, 0.04, 'solid = plane\ndashed = full', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=5.4, color='0.35')

for ax, L, dx in [(axE, 'E', -0.22), (axE2, 'E2', -0.18), (fig.axes[10], 'F', -0.30),
                  (axG, 'G', -0.12), (axH0, 'H', -0.22)]:
    ax.text(dx, 1.08, L, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='bottom', ha='right')

OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/fig_manifold_addons_preview.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_addons_preview.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_addons_preview.png'))
