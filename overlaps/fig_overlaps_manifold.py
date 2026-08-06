"""Fig 3 — "The low-dimensional manifold is abstract and reused across tasks".

Bridges Fig 2 (low-dimensional, factorised) to Fig 4 (learning repositions the state). Panels:
  A  code traces               — sample / GNG / test / choice codes, Naive vs Expert  (main_panels _draw_trace_col).
  B  within- vs cross-task      — balanced-accuracy generalization matrix, Naive+Expert; diagonals = within-task,
     generalization matrix        off-diagonals = cross-task; off-diag above chance ⇒ shared/abstract geometry.
  C  shared action axis         — cross-decode Go/NoGo ↔ DPA-lick (2×2, Naive|Expert). Off-diagonals above chance
                                  (different trials, tasks AND epochs → no leakage) ⇒ ONE action axis serves both
                                  the go/no-go decision and the DPA lick. Solidly established in the Expert
                                  (off/diag 0.56, 95% CI excludes 0); weaker/uncertain in Naive (CI incl. 0), a
                                  positive but non-significant learning trend (Δ off/diag +0.24, p≈0.30). Robust,
                                  generalization-based replacement for the weak (~0.18) axis cosine.
  D  generalization summary     — cross-context bal-acc per code (Naive vs Expert), chance line: every code reads
                                  out above chance across the context that challenges it (choice/GNG strongest).
  E  abstraction across learning— per-mouse CCGP Naive vs Expert per code; diamonds near the unity line ⇒
                                  abstraction already present in Naive, preserved.

Panel A reuses the code builders from main_panels.py; panels B/C/D read cached arrays
(figures/overlaps/ccgp/matrices_cache_acc.pkl, from fig_ccgp_matrices_pseudo.py --acc); panel E reads
permouse_ccgp_cache.pkl (fig_ccgp.py). Output figures/overlaps/manifold/{png,svg}/fig_overlaps_manifold.{png,svg}
DEFAULT axis (2026-08-06) = 48-62 (choice/action window), consistent with Fig 4 (fig_overlaps_main_native.py);
pass --legacy for the old 57-63 axis (→ fig_overlaps_manifold_legacy). Affects panel A's choice trace and panel
C's shared-action cosine (both read the DPA action axis).
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Fig 3 DEFAULTS to the 48-62 action axis (--antact) for consistency with Fig 4 (pooled-evoked: the traces /
# d′ / shared-action cosine don't need robust units). Pass --legacy for the old 57-63 axis. Injected before
# importing main_panels (parses sys.argv at import); scoped to THIS figure so main_panels' global default is off.
_LEGACY = '--legacy' in sys.argv[1:]
if not _LEGACY and '--antact' not in sys.argv:
    sys.argv.append('--antact')

# Pull main_panels' data + code builders (traces) into globals (renders nothing).
import main_panels as _MP
globals().update({k: v for k, v in vars(_MP).items() if not k.startswith('__')})

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
CACHE = 'figures/overlaps/ccgp/matrices_cache_acc.pkl'


# ── A: code traces (2×4, Naive/Expert × sample/GNG/test/choice), via main_panels helpers ──
def draw_traces(fig, cell):
    sub = cell.subgridspec(2, 4, wspace=0.55, hspace=0.32)
    axA = np.empty((2, 4), dtype=object)
    for c in range(4):
        axA[0, c] = fig.add_subplot(sub[0, c], sharey=(axA[0, 0] if c in (2, 3) else None))
        axA[1, c] = fig.add_subplot(sub[1, c], sharex=axA[0, c], sharey=(axA[1, 0] if c in (2, 3) else None))
    _A_SPECS = [VARS_A[0], VAR_GNG, VARS_A[2], VARS_A[1]]                   # sample, GNG, test, lick(action)
    for ri, STG in enumerate(STAGES):
        for c, spec in enumerate(_A_SPECS):
            ylab = (f'{STG}\ncode (z)' if c == 0 else ('GNG code (z)' if c == 1 else ''))
            _draw_trace_col(axA[ri, c], spec, STG, ylab, show_title=(ri == 0), show_xlabel=(ri == 1))
    return axA


# ── B: within- vs cross-task generalization matrix (balanced accuracy, Naive+Expert), 4 codes ──
#    sample/test/choice = across TASK (3×3, DPA/Go/NoGo); GNG = across SAMPLE (2×2, A/B) — GNG IS the task
#    distinction so it has no within-task diagonal; its abstraction is w.r.t. memory content.
def draw_matrix(fig, axes):
    c = pickle.load(open(CACHE, 'rb'))
    Mms, GNG_Mms, STG, TLAB, CH = c['Mms'], c['GNG_Mms'], c['STAGES'], c['TLAB'], c['CHANCE']
    ORDER = ['sample', 'GNG', 'test', 'choice']
    allM = [Mms[(s, l)] for s in STG for l in ('sample', 'test', 'choice')] + [GNG_Mms[s] for s in STG]
    DEV = max(np.nanmax(np.abs(M - CH)) for M in allM)
    im = None
    for r, stage in enumerate(STG):
        for cc, lab in enumerate(ORDER):
            ax = axes[r][cc]
            if lab == 'GNG':
                M = GNG_Mms[stage]; labs = ['A', 'B']; xlab = 'test sample'
            else:
                M = Mms[(stage, lab)]; labs = TLAB; xlab = 'test task'
            n = M.shape[0]
            im = ax.imshow(M, cmap='Reds', vmin=0.5, vmax=0.5 + DEV, aspect='equal')
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                            color='w' if M[i, j] > 0.5 + 0.62 * DEV else 'k',
                            fontweight='bold' if i == j else 'normal')
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labs, fontsize=6.3); ax.set_yticklabels(labs, fontsize=6.3)
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
            if r == 0:
                ax.set_title(lab + ('  (÷sample)' if lab == 'GNG' else ''), loc='center', fontsize=TITLE_FS)
            if cc == 0:
                ax.set_ylabel(f'{stage}\ntrain task', fontsize=7)
            if r == len(STG) - 1:
                ax.set_xlabel(xlab, fontsize=6.5)
    cb = fig.colorbar(im, ax=axes[0][-1], fraction=0.05, pad=0.08)
    cb.ax.tick_params(labelsize=6); cb.set_label('bal. acc.', fontsize=6.5)


# ── C: shared action axis — cross-decode Go/NoGo ↔ DPA-lick (2×2, Naive | Expert) ──
def draw_action(fig, axes):
    c = pickle.load(open(CACHE, 'rb'))
    ACT, SUM, DIF, STG = c['ACT_Mms'], c['ACT_SUMM'], c['ACT_DIFF'], c['STAGES']
    labs = ['Go/NoGo', 'DPA-lick']
    allv = np.concatenate([ACT[s].ravel() for s in STG])
    DEV = max(np.abs(allv - 0.5).max(), 0.05)
    im = None
    for k, stage in enumerate(STG):
        ax = axes[k]; M = ACT[stage]
        im = ax.imshow(M, cmap='Reds', vmin=0.5, vmax=0.5 + DEV, aspect='equal')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                        color='w' if M[i, j] > 0.5 + 0.62 * DEV else 'k',
                        fontweight='bold' if i == j else 'normal')
        # ring the cross-decode (off-diagonal) cells — the shared-axis evidence
        for (i, j) in [(0, 1), (1, 0)]:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='#117733', lw=1.4, zorder=5))
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labs, fontsize=6.0); ax.set_yticklabels(labs, fontsize=6.0, rotation=90, va='center')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
        ax.set_title(f'{stage}  (cross {SUM[stage]["off"]:.2f})', loc='left', fontsize=TITLE_FS)
        if k == 0:
            ax.set_ylabel('train code', fontsize=7)
        ax.set_xlabel('test code', fontsize=6.5)
    cb = fig.colorbar(im, ax=axes[-1], fraction=0.05, pad=0.10)
    cb.ax.tick_params(labelsize=6); cb.set_label('bal. acc.', fontsize=6.5)
    axes[0].text(0.0, -0.42,
                 f"green = cross-decode (shared axis). Expert off/diag {SUM['Expert']['offdiag']:.2f} "
                 f"[{SUM['Expert']['offdiag_lo']:.2f},{SUM['Expert']['offdiag_hi']:.2f}] ≫ chance; "
                 f"Naive {SUM['Naive']['offdiag']:.2f} (n.s.); Δ p={DIF['p']:.2f}",
                 transform=axes[0].transAxes, fontsize=6, color='0.3', ha='left', va='top')


# ── D: generalization summary — cross-context bal-acc per code, Naive vs Expert ──
def draw_summary(ax):
    c = pickle.load(open(CACHE, 'rb'))
    Mms, GNG, STG = c['Mms'], c['GNG_Mms'], c['STAGES']
    codes = ['sample', 'GNG', 'test', 'choice']

    def cross(stage, code):
        if code == 'GNG':
            M = GNG[stage]; e = np.eye(2, dtype=bool)
        else:
            M = Mms[(stage, code)]; e = np.eye(len(M), dtype=bool)
        return M[~e].mean()

    def within(stage, code):
        M = GNG[stage] if code == 'GNG' else Mms[(stage, code)]
        return np.diag(M).mean()

    x = np.arange(len(codes))
    for stage, dx, ec, fc in [('Naive', -0.14, '0.45', 'none'), ('Expert', 0.14, '#332288', '#332288')]:
        wy = [within(stage, cd) for cd in codes]
        cy = [cross(stage, cd) for cd in codes]
        ax.scatter(x + dx, wy, marker='_', s=150, color=ec, linewidths=1.3, zorder=2)   # within = tick (ceiling ref)
        ax.scatter(x + dx, cy, marker='o', s=44, facecolors=fc, edgecolors=ec, linewidths=1.1,
                   label=stage, zorder=3)
    ax.axhline(0.5, ls=':', color='0.6', lw=0.8)
    ax.text(len(codes) - 0.5, 0.505, 'chance', fontsize=5.5, color='0.6', va='bottom', ha='right')
    ax.set_xticks(x); ax.set_xticklabels(codes); ax.set_xlim(-0.5, len(codes) - 0.5); ax.set_ylim(0.47, 1.03)
    ax.set_ylabel('bal. acc.', fontsize=7.5)
    ax.set_title('Cross-context generalization (● cross-context, — within)', loc='left', fontsize=TITLE_FS)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left', ncol=2)


# ── E: per-mouse abstraction Naive vs Expert, one scatter per code (sample/GNG/test/choice).
#    Fig-1H conventions: colour = mouse (MOUSE_COLOR), marker = opsin group (GMARKER ●/▲/■), white edge. ──
def draw_scatters(axes):
    R = pd.read_pickle('figures/overlaps/ccgp/permouse_ccgp_cache.pkl')
    for ax, v in zip(axes, ['sample', 'GNG', 'test', 'choice']):
        piv = (R[R['variable'] == v].pivot_table(index='mouse', columns='stage', values='ccgp')
               .dropna(subset=['Naive', 'Expert']))
        ax.plot([0.42, 1.0], [0.42, 1.0], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0); ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        for m, rr in piv.iterrows():
            ax.scatter(rr['Naive'], rr['Expert'], s=42, color=MOUSE_COLOR.get(m, '0.5'),
                       marker=GMARKER[GROUP.get(m, 'Jaws')], edgecolors='w', linewidths=0.5, zorder=3)
        ax.scatter(piv['Naive'].mean(), piv['Expert'].mean(), s=95, color='k', marker='D',
                   edgecolors='w', linewidths=0.6, zorder=5)
        p = float(wilcoxon(piv['Expert'], piv['Naive']).pvalue)
        ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{v}  ({"∗" if p < .05 else "n.s."})', loc='left', fontsize=TITLE_FS)
        ax.set_xlabel('CCGP — Naive', fontsize=7.5)
        ax.text(0.05, 0.96, f'Δ={piv.Expert.mean() - piv.Naive.mean():+.2f}\np={p:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=6, color='0.3')
    axes[0].set_ylabel('CCGP — Expert', fontsize=7.5)


# ═══════════════════════════════════ assembly ═══════════════════════════════════
if __name__ == '__main__':
    fig = plt.figure(figsize=(11.0, 12.3))
    gs = fig.add_gridspec(4, 12, height_ratios=[1.02, 1.25, 1.02, 0.9],
                          hspace=0.6, wspace=0.9, left=0.075, right=0.965, top=0.945, bottom=0.05)

    def panel_letter(ax, L, x=0.008, dy=0.016):
        p = ax.get_position(); fig.text(x, p.y1 + dy, L, fontsize=11, fontweight='bold', va='top', ha='left')

    # Row 0: A code traces (2×4)
    axA = draw_traces(fig, gs[0, 0:12])
    # Row 1: B generalization matrix (2×4: sample, GNG, test, choice)
    gsB = gs[1, 0:11].subgridspec(2, 4, hspace=0.35, wspace=0.45)
    axB = [[fig.add_subplot(gsB[r, c]) for c in range(4)] for r in range(2)]
    draw_matrix(fig, axB)
    # Row 2: C shared action axis (2×2 Naive|Expert)  +  D generalization summary
    gsC = gs[2, 0:5].subgridspec(1, 2, wspace=0.55)
    axActs = [fig.add_subplot(gsC[0, k]) for k in range(2)]
    draw_action(fig, axActs)
    axSum = fig.add_subplot(gs[2, 6:12])
    draw_summary(axSum)
    # Row 3: E abstraction across learning (1×4 scatters)
    gsE = gs[3, 0:11].subgridspec(1, 4, wspace=0.5)
    axE = [fig.add_subplot(gsE[0, c]) for c in range(4)]
    draw_scatters(axE)

    panel_letter(axA[0, 0], 'A')
    panel_letter(axB[0][0], 'B')
    panel_letter(axActs[0], 'C')
    panel_letter(axSum, 'D', x=0.46)
    panel_letter(axE[0], 'E')
    fig.suptitle('The dual-task codes are abstract and reused across tasks', x=0.008, ha='left',
                 y=0.975, fontsize=10)
    OUT = 'figures/overlaps/manifold'
    _SUF = '_legacy' if _LEGACY else ''                                    # default → canonical fig_overlaps_manifold.png (48-62); --legacy → _legacy (57-63)
    for s in ('png', 'svg'):
        os.makedirs(f'{OUT}/{s}', exist_ok=True)
    fig.savefig(f'{OUT}/png/fig_overlaps_manifold{_SUF}.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/svg/fig_overlaps_manifold{_SUF}.svg', bbox_inches='tight')
    print('saved', os.path.abspath(f'{OUT}/png/fig_overlaps_manifold{_SUF}.png'))
