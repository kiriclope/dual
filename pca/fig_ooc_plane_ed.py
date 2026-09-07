"""fig_ooc_plane_ed.py — the Extended Data component of the out-of-context plane test (ED 6e).

Compact version of fig_ooc_plane.py for the composed ED 6 page: the naïve-DPA reference plane only.
  left    captured fraction with the 2-D readout refit in context (pooled pseudo-population)
  middle  captured fraction with the reference decoder applied unchanged
  right   per-mouse ratio, mean over the mouse's out-of-context cells (median line)
No internal panel letter — make_ed_figures.py adds the outer 'e'.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ooc_plane_ed.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
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
CEIL_MIN = 0.60
REF = ('Naive', 'DPA')
REFS = [('Naive', 'DPA'), ('Expert', 'DPA')]
AXWIN = {'sample': 'md', 'choice': 'decision'}

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
PP = RES['OOC_PLANE_PSEUDO' + SUF]['cells']
PM = RES['OOC_PLANE' + SUF]['cells']


def matrix(ref, key):
    M = np.full((len(ROWS), len(COLS)), np.nan); C = np.full_like(M, np.nan)
    for i, (v, w) in enumerate(ROWS):
        for j, (st, tk) in enumerate(COLS):
            e = PP[ref][v].get((st, tk, w))
            if e is not None:
                C[i, j] = e['iplane']; M[i, j] = e[key]
    return M, C


def draw_matrix(ax, M, C, ref, title, ylabels):
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
            if M[i, j] > 1.2:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='////', edgecolor='0.45', lw=0.0, zorder=3))
            v, w = ROWS[i]
            if (COLS[j] == ref) and (w == AXWIN[v]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='k', lw=0.9, zorder=4))
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COL_LAB, fontsize=PS * 6.0)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROW_LAB if ylabels else [], fontsize=PS * 6.0)
    ax.tick_params(length=0)
    ax.set_title(title, loc='left', fontsize=TITLE_FS)
    for sp in ax.spines.values():
        sp.set_visible(True)


def eligible(v, ref, ctx, e):
    st, tk, w = ctx
    if v == 'choice' and w != 'decision':
        return False
    if (st, tk) == ref and w == AXWIN[v]:
        return False
    return np.isfinite(e.get('iplane', np.nan)) and e['iplane'] >= CEIL_MIN


fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.16, 0.40], wspace=0.10, left=0.155, right=0.99, top=0.86, bottom=0.17)
ax1 = fig.add_subplot(gs[0, 0]); M, C = matrix(REF, 'ratio_ic')
draw_matrix(ax1, M, C, REF, 'plane from naïve DPA trials, readout refit', True)
ax2 = fig.add_subplot(gs[0, 1]); M, C = matrix(REF, 'tratio')
draw_matrix(ax2, M, C, REF, 'reference decoder, no refit', False)
ax3 = fig.add_subplot(gs[0, 3])
for k, v in enumerate(['sample', 'choice']):
    vals = []
    for mo in ALL_MICE:
        rat = [(e['plane'] - 0.5) / (e['iplane'] - 0.5) for ref in REFS for ctx, e in PM[mo][ref][v].items()
               if eligible(v, ref, ctx, e)]
        if rat:
            r = float(np.mean(rat)); vals.append(r)
            ax3.scatter(k + np.random.RandomState(hash(mo) % 1000).uniform(-0.13, 0.13), r, s=34, color=MC[mo],
                        edgecolors='w', linewidths=0.5, zorder=3)
    ax3.plot([k - 0.24, k + 0.24], [np.median(vals)] * 2, color='k', lw=1.0, zorder=4)
    print(f'ED e per-mouse {v}: median {np.median(vals):.2f} n={len(vals)}')
ax3.axhline(1.0, ls='--', color='0.6', lw=0.8, zorder=0)
ax3.set_xlim(-0.6, 1.6); ax3.set_ylim(0, 1.3)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(['sample', 'choice'], fontsize=PS * 7)
ax3.set_yticks([0, 0.5, 1.0])
ax3.set_ylabel('captured fraction', fontsize=PS * 7)
ax3.set_title('per mouse', loc='left', fontsize=TITLE_FS)

fig.canvas.draw()
b1 = ax1.get_position(); b3 = ax3.get_position()
ax3.set_position((b3.x0, b1.y0, b3.width, b1.height))   # align the strip with the matrices
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/fig_ooc_plane_ed{SUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_ooc_plane_ed{SUF}.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/fig_ooc_plane_ed{SUF}.png')
