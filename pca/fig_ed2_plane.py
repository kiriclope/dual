"""(RENUMBERED 2026-09-18 to citation order, ten figures after the unsupervised-geometry page was inserted as Extended Data Fig. 4: this script draws Extended Data Fig. 5.)
fig_ed2_plane.py — Extended Data Fig. 5: the sample × choice plane, per animal and out of context
(companion to Fig. 3). Built 2026-09-15 (Leon: "keep only what is essential for the paper's
argumentation") — every panel is one the Results or the Fig. 3 legend cite; nothing else.

  a  the GNG and test codes over time on their own decoder axes (Fig. 3a legend: "GNG and test codes are
     shown in Extended Data")
  b  per-mouse cross-condition generalization (CCGP), naïve against expert (Results §3: "only the test
     code nudged upward, p = .04 ... without a verdict")
  c  the out-of-context plane test (Results §3, Methods)
  d  the plane ablation of Fig. 3c in every animal (Results §3: "the same pattern held in every animal")

Reads caches only (pca results.pkl ORIG_TRACES / OOC_PLANE* / PM_PLANE_nopca + the overlaps per-mouse CCGP
cache). Canonical no-PCA pipeline throughout.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ed2_plane.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig5.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.2      # 10-in canvas -> 183 mm is x0.72: 1.2 keeps every literal (5.5-8 pt) at >= 5 pt in print (review 2026-09-15)
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = PS*8
NOCAP = '--nocap' in sys.argv[1:]
SUF = '_nopca'
STAGES = ['Naive', 'Expert']
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
GROUP = {**{m: 'Jaws' for m in MICE[:5]}, **{m: 'ChR' for m in MICE[5:7]}, **{m: 'ACC' for m in MICE[7:]}}
GMARK = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
MC = dict(zip(MICE, sns.color_palette('tab10', n_colors=len(MICE))))
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
CCGP_CACHE = '/home/leon/dual/overlaps/figures/overlaps/ccgp/permouse_ccgp_cache_canon_all.pkl'   # fig_ccgp.py --canon --alltrials (canonical since 2026-09-15); _canon.pkl = correct trials; unsuffixed = the 2026-08-04 LD/MD/TEST build


def plabel(ax, s, dx=-0.10):
    ax.text(dx, 1.06, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


# ══ a — the GNG and test codes over time (per-mouse CCGD projections, replayed from ORIG_TRACES) ══
EVENTS = [('sample', 2.0, 3.0, '#332288'), ('GNG', 4.5, 5.5, '#cc3311'),
          ('cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]


def panel_a(fig, gs):
    TR = RES['ORIG_TRACES']; xt = np.asarray(RES['ORIG_XTIME'])
    SP = {sp['code']: sp for sp in RES['ORIG_SPECS']}
    codes = [('GNG', 'GNG code'), ('test', 'test code')]
    axes = []
    for k, (code, ttl) in enumerate(codes):
        spec = SP[code]; lo, hi = 0.0, 0.0
        for stage in STAGES:
            for lv in spec['levels']:
                M = np.asarray(TR[(stage, code, int(lv))], dtype=float)
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                lo = min(lo, (mu - se).min()); hi = max(hi, (mu + se).max())
        pad = 0.05 * (hi - lo)
        for r, stage in enumerate(STAGES):
            ax = fig.add_subplot(gs[r, k]); axes.append(ax)
            for nm, a, b, col in EVENTS:
                ax.axvspan(a, b, color=col, alpha=0.10, lw=0)
                if r == 0 and k == 0:
                    ax.text((a + b) / 2, 0.98, nm, transform=ax.get_xaxis_transform(), ha='center', va='top',
                            fontsize=PS*6.0, color=col)
            for lv, lab, col in zip(spec['levels'], spec['labels'], spec['colors']):
                M = np.asarray(TR[(stage, code, int(lv))], dtype=float)
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                ax.plot(xt, mu, color=col, lw=1.3, label=f'{lab} (n = {len(M)})', zorder=3)
                ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.20, lw=0, zorder=2)
            ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
            ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12])
            if r == 0:
                ax.set_title(ttl, loc='left', fontsize=TITLE_FS); ax.tick_params(labelbottom=False)
                ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.2, loc='upper left' if k else 'lower left')
            else:
                ax.set_xlabel('time (s)')
            ax.set_ylabel(f'{stage}\ncode depth' if k == 0 else '')
    return axes[0]


# ══ b — per-mouse CCGP, naïve against expert ═════════════════════════════════════════════════
def panel_b(fig, gs):
    R = pd.read_pickle(CCGP_CACHE)
    axes = []
    for j, v in enumerate(['sample', 'GNG', 'test', 'choice']):
        ax = fig.add_subplot(gs[j // 2, j % 2]); axes.append(ax)
        piv = (R[R['variable'] == v].pivot_table(index='mouse', columns='stage', values='ccgp')
               .dropna(subset=['Naive', 'Expert']))
        ax.plot([0.42, 1.0], [0.42, 1.0], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0); ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        for m, rr in piv.iterrows():
            ax.scatter(rr['Naive'], rr['Expert'], s=30, color=MC.get(m, '0.5'), marker=GMARK[GROUP.get(m, 'Jaws')],
                       edgecolors='w', linewidths=0.5, zorder=3)
        p = float(wilcoxon(piv['Expert'], piv['Naive']).pvalue)
        ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
        ax.set_title(v, loc='left', fontsize=TITLE_FS)
        ax.text(0.05, 0.96, f'Δ = {piv.Expert.mean() - piv.Naive.mean():+.2f}\np = {p:.3f}', transform=ax.transAxes,
                va='top', ha='left', fontsize=PS*6, color='0.3')
        if j // 2 == 1:
            ax.set_xlabel('CCGP, naïve')
        else:
            ax.tick_params(labelbottom=False)
        if j % 2 == 0:
            ax.set_ylabel('CCGP, expert')
        else:
            ax.tick_params(labelleft=False)
        print(f'b: {v:7s} N {piv["Naive"].mean():.3f} -> E {piv["Expert"].mean():.3f}  p={p:.3f} (n={len(piv)})')
    return axes[0]


# ══ c — the out-of-context plane test (fig_ooc_plane_ed.py, inlined) ═════════════════════════
TASKS = ['DPA', 'DualGo', 'DualNoGo']
COLS = [(st, tk) for st in STAGES for tk in TASKS]
TK = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
COL_LAB = [f"{'naïve' if st == 'Naive' else 'expert'}\n{TK[tk]}" for st, tk in COLS]
ROWS = [('sample', 'ed'), ('sample', 'md'), ('sample', 'decision'), ('choice', 'decision')]
ROW_LAB = ['sample, early delay', 'sample, mid-delay', 'sample, decision', 'choice, decision']
CEIL_MIN = 0.60
REF = ('Naive', 'DPA'); REFS = [('Naive', 'DPA'), ('Expert', 'DPA')]
AXWIN = {'sample': 'md', 'choice': 'decision'}
TRIALSET = '' if '--correctonly' in sys.argv[1:] else '_all'   # 2026-09-15: ALL laser-off trials are canonical (review: on correct trials
                                                              # lick == match); --correctonly draws the pre-2026-09-15 caches
PP = RES['OOC_PLANE_PSEUDO' + SUF + TRIALSET]['cells']; PMO = RES['OOC_PLANE' + SUF + TRIALSET]['cells']


def _matrix(ref, key):
    M = np.full((len(ROWS), len(COLS)), np.nan); C = np.full_like(M, np.nan)
    for i, (v, w) in enumerate(ROWS):
        for j, (st, tk) in enumerate(COLS):
            e = PP[ref][v].get((st, tk, w))
            if e is not None:
                C[i, j] = e['iplane']; M[i, j] = e[key]
    return M, C


def _draw_matrix(ax, M, C, ref, title, ylabels):
    disp = np.where(C >= CEIL_MIN, np.clip(M, 0, 1), np.nan)
    ax.imshow(np.ma.masked_invalid(disp), cmap='Reds', vmin=0, vmax=1, aspect='equal')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not (C[i, j] >= CEIL_MIN):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc='0.93', ec='none'))
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=PS*6.0, color='0.45')
                continue
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=PS*6.0, color='w' if disp[i, j] > 0.6 else 'k')
            if M[i, j] > 1.2:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='////', edgecolor='0.45', lw=0.0, zorder=3))
            v, w = ROWS[i]
            if (COLS[j] == ref) and (w == AXWIN[v]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='k', lw=0.9, zorder=4))
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COL_LAB, fontsize=PS*6.0)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROW_LAB if ylabels else [], fontsize=PS*6.0)
    ax.tick_params(length=0); ax.set_title(title, loc='left', fontsize=TITLE_FS)
    for sp in ax.spines.values():
        sp.set_visible(True)


def _eligible(v, ref, ctx, e):
    st, tk, w = ctx
    if v == 'choice' and w != 'decision':
        return False
    if (st, tk) == ref and w == AXWIN[v]:
        return False
    return np.isfinite(e.get('iplane', np.nan)) and e['iplane'] >= CEIL_MIN


def panel_c(fig, gs):
    ax1 = fig.add_subplot(gs[0, 0]); M, C = _matrix(REF, 'ratio_ic')
    _draw_matrix(ax1, M, C, REF, 'plane from naïve DPA trials, readout refit', True)
    ax2 = fig.add_subplot(gs[0, 1]); M, C = _matrix(REF, 'tratio')
    _draw_matrix(ax2, M, C, REF, 'reference decoder, no refit', False)
    ax3 = fig.add_subplot(gs[0, 3])
    for k, v in enumerate(['sample', 'choice']):
        vals = []
        for mo in MICE:
            rat = [(e['plane'] - 0.5) / (e['iplane'] - 0.5) for ref in REFS for ctx, e in PMO[mo][ref][v].items()
                   if _eligible(v, ref, ctx, e)]
            if rat:
                r = float(np.mean(rat)); vals.append(r)
                ax3.scatter(k + np.random.RandomState(MICE.index(mo)).uniform(-0.13, 0.13), r, s=30, color=MC[mo],   # deterministic jitter (str hash is salted per process)
                            marker=GMARK[GROUP[mo]], edgecolors='w', linewidths=0.5, zorder=3)
        ax3.plot([k - 0.24, k + 0.24], [np.median(vals)] * 2, color='k', lw=1.0, zorder=4)
        print(f'c: per-mouse {v}: median {np.median(vals):.2f} n={len(vals)}')
    ax3.axhline(1.0, ls='--', color='0.6', lw=0.8, zorder=0)
    ax3.set_xlim(-0.6, 1.6); ax3.set_ylim(0, 1.3); ax3.set_xticks([0, 1]); ax3.set_xticklabels(['sample', 'choice'])
    ax3.set_yticks([0, 0.5, 1.0]); ax3.set_ylabel('captured fraction'); ax3.set_title('per mouse', loc='left', fontsize=TITLE_FS)
    fig.canvas.draw()
    b1 = ax1.get_position(); b3 = ax3.get_position()
    ax3.set_position((b3.x0, b1.y0, b3.width, b1.height))
    return ax1


# ══ d — the plane ablation of Fig. 3c in every animal (fig_manifold_main.panel_e_plane, inlined) ═══
E_VARS = ['sample', 'dist', 'test', 'choice']        # PM_PLANE cache keys — do not rename
E_LABEL = {'dist': 'GNG'}
E_SPACES = [('plane only (2-D)', 0), ('out-of-plane', 2), ('full space', 1)]
PMPL = RES['PM_PLANE' + SUF + TRIALSET]


def panel_d(fig, gs):
    lo, hi = 0.42, 1.01
    axes = []
    for r, (rowlab, key) in enumerate(E_SPACES):
        for c, vn in enumerate(E_VARS):
            ax = fig.add_subplot(gs[r, c]); axes.append(ax)
            ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
            ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0); ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
            nv, ev = [], []
            for m in MICE:
                if any((m, st) not in PMPL or vn not in PMPL[(m, st)] for st in STAGES):
                    continue
                a = PMPL[(m, 'Naive')][vn][key]; b = PMPL[(m, 'Expert')][vn][key]
                nv.append(a); ev.append(b)
                ax.scatter(a, b, s=28, color=MC[m], marker=GMARK[GROUP[m]], edgecolors='w', linewidths=0.5, zorder=3)
            nv, ev = np.array(nv), np.array(ev)
            p = float(wilcoxon(ev, nv).pvalue)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
            if c:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(E_LABEL.get(vn, vn), loc='left', fontsize=TITLE_FS)
            if r < 2:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(f'{rowlab}\nexpert', fontsize=PS*7)
            if r == 2:
                ax.set_xlabel('naïve', fontsize=PS*7)
            star = p < 0.05                                       # no whitelist (review 2026-09-15); nothing reaches .05 on the canonical caches
            ax.text(0.05, 0.96, f'Δ = {ev.mean() - nv.mean():+.02f}\np = {p:.3f}' + ('  ∗' if star else ''),
                    transform=ax.transAxes, va='top', ha='left', fontsize=PS*6.0, color='k' if star else '0.3',
                    fontweight='bold' if star else 'normal')
            print(f'd: {rowlab:16s} {vn:7s} {nv.mean():.2f} -> {ev.mean():.2f}  p={p:.3f}{" *" if star else ""}')
    return axes[0]


# ══ ASSEMBLE ═══════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 11.6))
outer = fig.add_gridspec(3, 12, height_ratios=[1.05, 0.78, 1.32], hspace=0.42, wspace=1.0,
                         left=0.07, right=0.985, top=0.972, bottom=0.04)
gsA = outer[0, 0:6].subgridspec(2, 2, wspace=0.30, hspace=0.18)
axA = panel_a(fig, gsA)
gsB = outer[0, 7:12].subgridspec(2, 2, wspace=0.12, hspace=0.28)
axB = panel_b(fig, gsB)
gsC = outer[1, 0:12].subgridspec(1, 4, width_ratios=[1, 1, 0.12, 0.42], wspace=0.10)
axC = panel_c(fig, gsC)
gsD = outer[2, 0:9].subgridspec(3, 4, wspace=0.10, hspace=0.16)
axD = panel_d(fig, gsD)
plabel(axA, 'a', dx=-0.26); plabel(axB, 'b', dx=-0.34); plabel(axC, 'c', dx=-0.30); plabel(axD, 'd', dx=-0.42)

CAP = [
    'Extended Data Fig. 5 | The sample × choice plane, per animal and out of context (companion to Fig. 3). '
    'a, The GNG and test codes over time on their own cross-validated decoder axes (naïve | expert; mean ± SEM '
    'across mice; evoked-s.d. units, conventions as Fig. 3a), the definitional reference for the two codes Fig. 3c–e '
    'ablate and align. b, Per-mouse cross-condition generalization (CCGP) of each variable, naïve against expert '
    '(marker, opsin group; Δ, mean change; p, paired Wilcoxon, n = 9; canonical windows, sample and GNG at mid-delay, '
    'test and choice at the decision): abstraction is present from the first dual task sessions; the test code '
    'nudges upward as a trend (Δ = +0.02, p = .074), reported without a verdict. All laser-off trials.',
    'c, The out-of-context plane test. A sample × choice plane fitted on the naïve DPA trials is read in every other '
    'stage, trial type and moment against a plane fitted in that context (pooled pseudo-population): captured '
    'fraction (fixed − 0.5)/(in-context − 0.5) with the two-dimensional readout refit in context (left) and with '
    'the reference decoder applied unchanged (middle); boxed, the within-context check; grey, in-context ceiling '
    'below 0.60; hatched, ratio above 1.2. Cells carry no per-cell uncertainty, and ratios inflate as the in-context '
    'ceiling nears 0.60. Right, the per-mouse ratio over each mouse’s out-of-context cells (line, median; sample '
    'n = 9, choice n = 7 — two mice have no choice cell above the ceiling). The plane carries the codes everywhere; '
    'where the unchanged decoder fails and the refit does not (expert NoGo trials after the Go/NoGo odor), the code '
    'has moved within the plane.',
    'd, The plane ablation of Fig. 3c in every animal (naïve x against expert y; rows, spaces; columns, variables; '
    'Δ, mean change; p, paired Wilcoxon, n = 9; ∗ marks p < .05 on the canonical pipeline). The pattern of Fig. 3c '
    'holds mouse by mouse, and the grid carries the one change with learning: the GNG code’s plane-only accuracy rises '
    '(0.56 → 0.64, p = .027, 7/9 mice), while the test code’s plane-only accuracy stays at chance (p = .31). Learning '
    'pulls the GNG code toward the plane, which Fig. 4a quantifies. All laser-off trials.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig5.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/ed_fig5.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig5.png')
