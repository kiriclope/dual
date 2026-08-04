"""Fig 3 — "The low-dimensional manifold is abstract and reused across tasks".

Bridges Fig 2 (low-dimensional, factorised) to Fig 4 (learning repositions the state). Panels:
  A  dPCA linking panel        — Fig-2 manifold (sample × action plane); the memory code is on ONE shared axis
                                  across DPA/Go/NoGo (reused), orthogonal to the pre-existing action axis.
  B  code traces               — sample / GNG / test / choice codes, Naive vs Expert  (main_panels _draw_trace_col).
  C  within- vs cross-task      — balanced-accuracy generalization matrix, Naive+Expert; diagonals = within-task,
     generalization matrix        off-diagonals = cross-task; off-diag above chance ⇒ shared/abstract geometry.
  D  abstraction across learning— per-mouse CCGP Naive vs Expert per code (sample/choice/test); diamonds on the
                                  unity line ⇒ abstraction already present in Naive, preserved.
  E  shared action axis + d'    — signed cos(DPA-lick · GNG-lick) Naive→Expert + within-task action-code d'
                                  unchanged  (main_panels _lick_dprime / _act_cos).

Panels B/E reuse the code/geometry builders from main_panels.py; panels C/D read cached arrays
(figures/overlaps/ccgp/*.pkl) from fig_ccgp_matrices_pseudo.py --acc and fig_ccgp.py.
Output figures/overlaps/manifold/{png,svg}/fig_overlaps_manifold.{png,svg}
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
from src.pca.io import pkl_load

# Pull main_panels' data + code/geometry builders (traces, d′, cosine) into globals (renders nothing).
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
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}                 # odor A indigo, B teal
TASK_LS = {'DPA': '-', 'Go': '--', 'NoGo': ':'}
TASKDUM = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
FS = 6.0


# ── A: dPCA linking panel (Fig-2 bridge) ──────────────────────────────────────
def _load_marg(dum, stage='Expert'):
    X = pkl_load(f'pseudo_traj_{dum}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{dum}', path='../data/pca')
    labels = pkl_load(f'pseudo_marglabels_{dum}', path='../data/pca')
    IDX = {nm: labels.index(nm) for nm in dict.fromkeys(labels)}
    m = ((y.laser == 0) & (y.learning == stage) & (y.performance == 1)).to_numpy()
    Z = X[m].astype(float); Z = (Z - Z.mean((0, 2), keepdims=True)) / Z.std((0, 2), keepdims=True)
    yc = y[m].reset_index(drop=True)
    DLYw, TST = np.arange(42, 54), np.arange(57, 66)
    B = (yc['sample'] == 1).to_numpy(); Dd = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    for nm, (pos, neg, w) in {'sample': (B, ~B, DLYw), 'test': (Dd, ~Dd, TST),
                              'sample:test': (lick, ~lick, TST), 'tasks': (go, nogo, TST)}.items():
        if nm in IDX and Z[pos][:, IDX[nm]][:, w].mean() < Z[neg][:, IDX[nm]][:, w].mean():
            Z[:, IDX[nm], :] *= -1
    return Z, yc, IDX


def draw_link(ax, stage='Expert', win=np.arange(30, 54)):
    Z, yc, IDX = _load_marg(TASKDUM.format(stage), stage)
    sx, ay = IDX['sample'], IDX['tasks']
    TASKS = {'DPA': (yc['tasks'] == 'DPA').to_numpy(), 'Go': (yc['tasks'] == 'DualGo').to_numpy(),
             'NoGo': (yc['tasks'] == 'DualNoGo').to_numpy()}
    for tname, tmask in TASKS.items():
        for s in (0, 1):
            m = tmask & (yc['sample'] == s).to_numpy()
            if m.sum() < 2:
                continue
            traj = Z[m][:, [sx, ay], :][:, :, win].mean(0)
            ax.plot(traj[0], traj[1], TASK_LS[tname], color=SAMPLE_COL[s], lw=1.4, alpha=0.9, zorder=2)
            ax.scatter(traj[0, -1], traj[1, -1], s=40, color=SAMPLE_COL[s], edgecolor='k', linewidths=0.6, zorder=4)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0); ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlabel('sample (memory) axis\n← odor A          odor B →', fontsize=7.5)
    ax.set_ylabel('action axis (tasks)\n← no-lick        lick →', fontsize=7.5)
    ax.set_title('Codes reused on the low-D dPCA manifold', loc='left', fontsize=TITLE_FS)
    h1 = [Line2D([0], [0], color=SAMPLE_COL[0], lw=2, label='odor A'),
          Line2D([0], [0], color=SAMPLE_COL[1], lw=2, label='odor B')]
    h2 = [Line2D([0], [0], color='0.4', ls=TASK_LS[t], lw=1.4, label=t) for t in TASK_LS]
    leg1 = ax.legend(handles=h1, frameon=False, fontsize=6.5, loc='upper left', handlelength=1.3)
    ax.add_artist(leg1)
    ax.legend(handles=h2, frameon=False, fontsize=6.5, loc='lower right', handlelength=1.8)


# ── B: code traces (2×4, Naive/Expert × sample/GNG/test/choice), via main_panels helpers ──
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


# ── E: within-task action-code d′ (unchanged) + shared action axis cosine ──
def draw_axis_dprime(ax):
    dN = np.array([_lick_dprime(m, 'Naive') for m in ALL_MICE]); dE = np.array([_lick_dprime(m, 'Expert') for m in ALL_MICE])
    ok = np.isfinite(dN) & np.isfinite(dE)
    av = np.concatenate([dN[ok], dE[ok]]); lim = (min(av.min(), -0.1), av.max() * 1.12)
    ax.plot(lim, lim, ls='--', color='0.6', lw=0.8, zorder=1)
    ax.axhline(0, ls=':', color='0.8', lw=0.6); ax.axvline(0, ls=':', color='0.8', lw=0.6)
    for m, xn, ye in zip(np.array(ALL_MICE)[ok], dN[ok], dE[ok]):
        ax.scatter(xn, ye, s=26, facecolors=MOUSE_COLOR[m], edgecolors=MOUSE_COLOR[m], linewidths=0.6, zorder=4)
    pt = float(ttest_rel(dE[ok], dN[ok]).pvalue); dd = float((dE[ok] - dN[ok]).mean()); sig = pt < 0.05
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_box_aspect(1)
    ax.set_title('action-code d′', fontsize=TITLE_FS, loc='left')
    ax.set_xlabel('Naive d′', fontsize=7.5); ax.set_ylabel('Expert d′', fontsize=7.5)
    ax.text(0.06, 0.95, '*' if sig else 'n.s.', transform=ax.transAxes, ha='left', va='top',
            fontsize=11 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    ax.text(0.5, 0.02, f'Δ={dd:+.2f}, p={pt:.3f}', transform=ax.transAxes, ha='center', va='bottom',
            fontsize=6, color='0.3')


def draw_shared_axis(ax):
    acN = np.array([_act_cos(m, 'Naive') for m in ALL_MICE]); acE = np.array([_act_cos(m, 'Expert') for m in ALL_MICE])
    for i, m in enumerate(ALL_MICE):
        ax.plot([0, 1], [acN[i], acE[i]], '-o', color=MOUSE_COLOR[m], lw=0.9, ms=4.5, mec='w', mew=0.5, zorder=3)
    for x, v in ((-0.16, acN), (1.16, acE)):
        mu = np.nanmean(v); se = np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())
        ax.errorbar(x, mu, yerr=se, fmt='s', color='k', ms=6, capsize=3.5, lw=1.3, zorder=5)
    ax.axhline(COS_CHANCE, ls=':', color='0.6', lw=0.8); ax.axhline(0, color='0.85', lw=0.6)
    ax.text(1.55, COS_CHANCE, 'chance', fontsize=5.5, color='0.6', va='bottom', ha='right')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=8); ax.set_xlim(-0.45, 1.65)
    ax.set_ylabel('cos(DPA-lick · GNG-lick axis)', fontsize=7.5)
    p0E = float(ttest_1samp(acE[np.isfinite(acE)], 0).pvalue); sig = p0E < 0.05
    ax.set_title('shared action code', loc='left', fontsize=TITLE_FS)
    ax.text(0.06, 0.96, '*' if sig else 'n.s.', transform=ax.transAxes, ha='left', va='top',
            fontsize=11 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    ax.set_box_aspect(1)


# ── C: within- vs cross-task generalization matrix (balanced accuracy, Naive+Expert), 4 codes ──
#    sample/test/choice = across TASK (3×3, DPA/Go/NoGo); GNG = across SAMPLE (2×2, A/B) — GNG IS the task
#    distinction so it has no within-task diagonal; its abstraction is w.r.t. memory content.
def draw_matrix(fig, axes):
    c = pickle.load(open('figures/overlaps/ccgp/matrices_cache_acc.pkl', 'rb'))
    Mms, GNG_Mms, STG, TLAB, CH = c['Mms'], c['GNG_Mms'], c['STAGES'], c['TLAB'], c['CHANCE']
    ORDER = ['sample', 'GNG', 'test', 'choice']
    allM = [Mms[(s, l)] for s in STG for l in ('sample', 'test', 'choice')] + [GNG_Mms[s] for s in STG]
    DEV = max(np.nanmax(np.abs(M - CH)) for M in allM)
    im = None
    for r, stage in enumerate(STG):
        for cc, lab in enumerate(ORDER):
            ax = axes[r][cc]
            if lab == 'GNG':
                M = GNG_Mms[stage]; labs = ['A', 'B']; xlab, ylab = 'test sample', 'train sample'
            else:
                M = Mms[(stage, lab)]; labs = TLAB; xlab, ylab = 'test task', 'train task'
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
            ax.set_title(lab + ('  (÷sample)' if lab == 'GNG' and r == 0 else ''), loc='center',
                         fontsize=TITLE_FS) if r == 0 else None
            if cc == 0:
                ax.set_ylabel(f'{stage}\ntrain task', fontsize=7)
            if r == len(STG) - 1:
                ax.set_xlabel(xlab, fontsize=6.5)
    cb = fig.colorbar(im, ax=axes[0][-1], fraction=0.05, pad=0.08)
    cb.ax.tick_params(labelsize=6); cb.set_label('bal. acc.', fontsize=6.5)


# ── D: per-mouse abstraction Naive vs Expert, one scatter per code (sample/GNG/test/choice).
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
    fig = plt.figure(figsize=(11.0, 9.4))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.15, 1.35, 0.95],
                          hspace=0.5, wspace=0.9, left=0.065, right=0.975, top=0.94, bottom=0.055)

    def panel_letter(ax, L, x=0.008, dy=0.016):
        p = ax.get_position(); fig.text(x, p.y1 + dy, L, fontsize=11, fontweight='bold', va='top', ha='left')

    # Row 0: A code traces (2×4)
    axA = draw_traces(fig, gs[0, 0:12])
    # Row 1: B generalization matrix (2×4: sample, GNG, test, choice)
    gsB = gs[1, 0:11].subgridspec(2, 4, hspace=0.35, wspace=0.45)
    axB = [[fig.add_subplot(gsB[r, c]) for c in range(4)] for r in range(2)]
    draw_matrix(fig, axB)
    # Row 2: C abstraction across learning (1×4 scatters)
    gsC = gs[2, 0:11].subgridspec(1, 4, wspace=0.5)
    axC = [fig.add_subplot(gsC[0, c]) for c in range(4)]
    draw_scatters(axC)

    panel_letter(axA[0, 0], 'A')
    panel_letter(axB[0][0], 'B')
    panel_letter(axC[0], 'C')
    fig.suptitle('The dual-task codes are abstract and reused across tasks', x=0.008, ha='left',
                 y=0.975, fontsize=10)
    OUT = 'figures/overlaps/manifold'
    for s in ('png', 'svg'):
        os.makedirs(f'{OUT}/{s}', exist_ok=True)
    fig.savefig(f'{OUT}/png/fig_overlaps_manifold.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/svg/fig_overlaps_manifold.svg', bbox_inches='tight')
    print('saved', os.path.abspath(f'{OUT}/png/fig_overlaps_manifold.png'))
