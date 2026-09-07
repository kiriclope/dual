"""fig_ooc_plane.py — the OUT-OF-CONTEXT plane test (candidate replacement for Fig 3c,d; 2026-09-07).

Reads OOC_PLANE_PSEUDO+SUF (pooled pseudo-population, exp_ooc_plane_pseudo.py) and OOC_PLANE+SUF
(per-mouse companion, exp_ooc_plane.py) from results.pkl. Drawn in the Fig 2e transfer-matrix idiom
(Reds, 0–1, equal cells, values in-cell, hatch = not interpretable).
  a  pooled: a sample x choice plane fitted in ONE reference context (naïve DPA | expert DPA) captures
     this fraction of the decodable signal in every other context, relative to a plane fitted IN that
     context, (fixed − 0.5)/(in-context − 0.5), with the 2-D readout refit in context. Columns, stage x
     trial type; rows, variable @ window. Boxed cell = the within check. Hatched = ratio above 1.2 (a
     denominator artefact; values slightly above 1 are within noise of 1); grey = in-context ceiling below CEIL_MIN (not resolvable; ceiling printed).
  b  the same with NO refit inside the plane (the reference axis decoder applied as is): a drop here
     with an unchanged cell in a means the code moved INSIDE the plane.
  c  per-mouse companion: fixed plane against in-context plane, every out-of-context cell with a
     resolvable ceiling, one colour per mouse; median per-mouse ratio and the paired test printed.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ooc_plane.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon
from decoders import SUF
import seaborn as sns, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PS = 1.0
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS * 8, 'axes.titlesize': PS * 8, 'xtick.labelsize': PS * 7, 'ytick.labelsize': PS * 7,
    'legend.fontsize': PS * 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = PS * 7
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MC = dict(zip(ALL_MICE, sns.color_palette('tab10', n_colors=len(ALL_MICE))))
STAGES, TASKS = ['Naive', 'Expert'], ['DPA', 'DualGo', 'DualNoGo']
COLS = [(st, tk) for st in STAGES for tk in TASKS]
TK = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
COL_LAB = [f"{'naïve' if st == 'Naive' else 'expert'}\n{TK[tk]}" for st, tk in COLS]
ROWS = [('sample', 'ed'), ('sample', 'md'), ('sample', 'decision'), ('choice', 'decision')]
ROW_LAB = ['sample, early delay', 'sample, mid-delay', 'sample, decision', 'choice, decision']
CEIL_MIN = 0.60                 # in-context plane ceiling below which a transfer ratio is not resolvable
REFS = [('Naive', 'DPA'), ('Expert', 'DPA')]
REF_LAB = {('Naive', 'DPA'): 'plane from naïve DPA trials', ('Expert', 'DPA'): 'plane from expert DPA trials'}
AXWIN = {'sample': 'md', 'choice': 'decision'}

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
PP = RES['OOC_PLANE_PSEUDO' + SUF]['cells']
PM = RES['OOC_PLANE' + SUF]['cells']


def matrix(ref, key):
    M = np.full((len(ROWS), len(COLS)), np.nan); C = np.full_like(M, np.nan)
    for i, (v, w) in enumerate(ROWS):
        for j, (st, tk) in enumerate(COLS):
            e = PP[ref][v].get((st, tk, w))
            if e is None:
                continue
            C[i, j] = e['iplane']; M[i, j] = e[key]
    return M, C


def draw_matrix(ax, M, C, ref, title, ylabels=True, xlabels=True):
    disp = np.where(C >= CEIL_MIN, np.clip(M, 0, 1), np.nan)
    ax.imshow(np.ma.masked_invalid(disp), cmap='Reds', vmin=0, vmax=1, aspect='equal')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not (C[i, j] >= CEIL_MIN):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc='0.93', ec='none'))
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=PS * 5.5, color='0.45')
                continue
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=PS * 6.0,
                    color='w' if disp[i, j] > 0.6 else 'k')
            if M[i, j] > 1.2:                       # a clear denominator artefact (low ceiling)
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='////',
                                       edgecolor='0.45', lw=0.0, zorder=3))
            v, w = ROWS[i]
            if (COLS[j] == ref) and (w == AXWIN[v]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='k', lw=0.9, zorder=4))
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COL_LAB if xlabels else [], fontsize=PS * 6.0)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROW_LAB if ylabels else [], fontsize=PS * 6.0)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc='left', fontsize=TITLE_FS)
    for sp in ax.spines.values():
        sp.set_visible(True)


fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.10, left=0.19, right=0.90, top=0.96, bottom=0.46)
gs2 = fig.add_gridspec(1, 2, wspace=0.10, left=0.19, right=0.90, top=0.33, bottom=0.07)
AX = {}
for r_i, ref in enumerate(REFS):
    ax = fig.add_subplot(gs[0, r_i]); AX[('a', r_i)] = ax
    M, C = matrix(ref, 'ratio_ic')
    draw_matrix(ax, M, C, ref, REF_LAB[ref], ylabels=(r_i == 0), xlabels=False)
    ax = fig.add_subplot(gs[1, r_i]); AX[('b', r_i)] = ax
    M, C = matrix(ref, 'tratio')
    draw_matrix(ax, M, C, ref, None, ylabels=(r_i == 0), xlabels=True)
# row-group labels on the right, and one thin colour key
for key, lab in [(('a', 1), '2-D readout\nrefit in context'), (('b', 1), 'reference decoder,\nno refit')]:
    bb = AX[key].get_position()
    fig.text(bb.x1 + 0.012, (bb.y0 + bb.y1) / 2, lab, rotation=90, va='center', ha='left', fontsize=PS * 6.5, color='0.25')
sm = matplotlib.cm.ScalarMappable(cmap='Reds', norm=matplotlib.colors.Normalize(0, 1))
bb = AX[('a', 1)].get_position(); bb2 = AX[('b', 1)].get_position()
cax = fig.add_axes((bb.x1 + 0.055, bb2.y0 + 0.25 * (bb.y1 - bb2.y0), 0.012, 0.5 * (bb.y1 - bb2.y0)))
cb = fig.colorbar(sm, cax=cax); cb.set_ticks([0, 0.5, 1])
cb.ax.tick_params(labelsize=PS * 6.0, length=2)
cb.set_label('captured fraction', fontsize=PS * 6.5)


def eligible(v, ref, ctx, e):
    st, tk, w = ctx
    if v == 'choice' and w != 'decision':
        return False
    if (st, tk) == ref and w == AXWIN[v]:
        return False                                  # the within check is not an out-of-context cell
    return np.isfinite(e.get('iplane', np.nan)) and e['iplane'] >= CEIL_MIN


stats_txt = []
for k, v in enumerate(['sample', 'choice']):
    ax = fig.add_subplot(gs2[0, k]); AX[('c', k)] = ax
    ncell = 0; per_mouse = []
    for mo in ALL_MICE:
        pl, ic, rat = [], [], []
        for ref in REFS:
            for ctx, e in PM[mo][ref][v].items():
                if not eligible(v, ref, ctx, e):
                    continue
                ax.scatter(e['iplane'], e['plane'], s=34, color=MC[mo], edgecolors='w', linewidths=0.5, zorder=3)
                pl.append(e['plane']); ic.append(e['iplane']); rat.append((e['plane'] - 0.5) / (e['iplane'] - 0.5)); ncell += 1
        if rat:
            per_mouse.append((np.mean(pl), np.mean(ic), np.mean(rat)))
    per_mouse = np.array(per_mouse)
    ax.plot([0.5, 1], [0.5, 1], ls='--', color='0.6', lw=0.8, zorder=0)
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0); ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0.5, 0.75, 1.0]); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xlabel('plane fitted in context', fontsize=PS * 7)
    if k == 0:
        ax.set_ylabel('fixed reference plane\n(balanced accuracy)', fontsize=PS * 7)
    else:
        ax.tick_params(labelleft=False)
    d = per_mouse[:, 0] - per_mouse[:, 1]
    p_pf = wilcoxon(per_mouse[:, 0], per_mouse[:, 1]).pvalue if len(per_mouse) > 5 else np.nan
    txt = (f'median ratio {np.median(per_mouse[:, 2]):.2f}\n'
           f'[IQR {np.percentile(per_mouse[:, 2], 25):.2f}–{np.percentile(per_mouse[:, 2], 75):.2f}]\n'
           f'n = {len(per_mouse)} mice, {ncell} cells\n'
           f'Δ = {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}, p = {p_pf:.2f}')
    stats_txt.append((v, txt))
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha='left', va='top', fontsize=PS * 6.0, color='0.3')
    ax.set_title(v, loc='left', fontsize=TITLE_FS)

for key, L in [(('a', 0), 'a'), (('b', 0), 'b'), (('c', 0), 'c')]:
    bb = AX[key].get_position()
    fig.text(0.02, bb.y1 + 0.005, L, fontsize=PS * 11, fontweight='bold', va='bottom')

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_ooc_plane{SUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_ooc_plane{SUF}.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/fig_ooc_plane{SUF}.png')
for v, t in stats_txt:
    print(v, '::', t.replace('\n', ' | '))
for ref in REFS:
    for v in ['sample', 'choice']:
        el = [(ctx, e) for ctx, e in PP[ref][v].items() if eligible(v, ref, ctx, e)]
        vals = [e['ratio_ic'] for _, e in el]; tv = [e['tratio'] for _, e in el]
        print(f'pooled ref {ref[0]}-{ref[1]} {v}: out-of-context cells n={len(vals)}  ratio_ic median {np.median(vals):.2f}  '
              f'no-refit median {np.median(tv):.2f}')
