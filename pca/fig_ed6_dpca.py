"""fig_ed6_dpca.py — Extended Data Fig. 6: the demixed-PCA decomposition (companion to Fig. 2). Built
2026-09-15 (Leon: "keep only what is essential") — Results §2 cites it once: "An independent decomposition
of the same data by demixed PCA gives the same picture: time courses along single axes sharpened with
learning without reorganizing, the choice and action axes of that decomposition became more aligned
(|cos| 0.147 → 0.222, p < 0.001), and the sample and test axes separated (0.098 → 0.033, p = 0.008)".

  a  the leading demixed axis of each task variable over time, naïve | expert (per-condition means ± SEM of
     the withheld pseudo-trials)
  b  |cos| between the leading demixed axes of every pair of variables, naïve → expert (neuron bootstrap)

Glue copied from fig_dpca_story_main.py (sections 2 and 2-mixing), which is not import-safe.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ed6_dpca.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig6.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from src.pca.io import pkl_load
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.0
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
TASKDUM = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'
BASE, BASE_N = TASKDUM.format('Expert'), TASKDUM.format('Naive')
FS = 6.0
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}
TEST_COL = {0: '#CC6677', 1: '#999933'}
CHOICE_COL = {'nolick': '#377eb8', 'lick': '#4daf4a'}
TASK_COL = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
EP_SHADE = [('sample', 2.0, 3.0, '#332288'), ('GNG', 4.5, 5.5, '#cc3311'), ('cue', 6.5, 7.0, '#ee7733'),
            ('test', 9.0, 10.0, '#377eb8')]


def load_marg(dum, stage):
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


def stat(Z, mask, comp):
    a = Z[mask][:, comp, :]; n = max(a.shape[0], 1)
    return a.mean(0), a.std(0) / np.sqrt(n)


def panel_a(fig, gs):
    axes = []
    ylim = {}
    for r, (dum, stage) in enumerate([(BASE_N, 'Naive'), (BASE, 'Expert')]):
        Z, yc, IDX = load_marg(dum, stage); tt = np.arange(Z.shape[2]) / FS
        B = (yc['sample'] == 1).to_numpy(); Dd = (yc['test'] == 1).to_numpy()
        lick = (yc['sample'] == yc['test']).to_numpy(); dpa = (yc['tasks'] == 'DPA').to_numpy()
        go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
        panels = [('sample axis', IDX['sample'], [('odor A', ~B, SAMPLE_COL[0]), ('odor B', B, SAMPLE_COL[1])]),
                  ('test axis', IDX['test'], [('odor C', ~Dd, TEST_COL[0]), ('odor D', Dd, TEST_COL[1])]),
                  ('choice axis', IDX['sample:test'], [('lick', lick, CHOICE_COL['lick']), ('no lick', ~lick, CHOICE_COL['nolick'])]),
                  ('task axis', IDX['tasks'], [('DPA', dpa, TASK_COL['DPA']), ('Go', go, TASK_COL['DualGo']), ('NoGo', nogo, TASK_COL['DualNoGo'])])]
        for k, (nm, comp, lns) in enumerate(panels):
            ax = fig.add_subplot(gs[r, k]); axes.append(ax)
            for enm, lo, hi, col in EP_SHADE:
                ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0, zorder=0)
                if r == 0 and k == 0 and enm != 'cue':
                    ax.text((lo + hi) / 2, 0.90 if enm == 'GNG' else 0.98, enm, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=PS*6.0, color=col)
            for lab, sel, col in lns:
                mu, se = stat(Z, sel, comp)
                ax.fill_between(tt, mu - se, mu + se, color=col, alpha=0.25, lw=0)
                ax.plot(tt, mu, color=col, lw=1.3, label=lab)
            ax.axhline(0, color='0.75', lw=0.5); ax.set_xlim(0, 14); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12])
            lo_, hi_ = ax.get_ylim(); ylim[k] = (min(lo_, ylim.get(k, (0, 0))[0]), max(hi_, ylim.get(k, (0, 0))[1]))
            if r == 0:
                ax.set_title(nm, loc='left', fontsize=TITLE_FS); ax.tick_params(labelbottom=False)
                ax.legend(frameon=False, fontsize=PS*6.0, loc='upper left' if k else 'center right', handlelength=1.0)
            else:
                ax.set_xlabel('time (s)')
            if k == 0:
                ax.set_ylabel(f'{"naïve" if stage == "Naive" else "expert"}\ndemixed projection (z)')
    for i, ax in enumerate(axes):
        ax.set_ylim(*ylim[i % 4])
    return axes[0]


def panel_b(ax):
    MARGS = ['sample', 'test', 'sample:test', 'tasks']; SH = {'sample': 'sample', 'test': 'test', 'sample:test': 'choice', 'tasks': 'task'}

    def axes_of(st):
        W = np.asarray(pkl_load(f'pseudo_weights_{TASKDUM.format(st)}', path='../data/pca'), float)
        lab = pkl_load(f'pseudo_marglabels_{TASKDUM.format(st)}', path='../data/pca')
        return W, {m: [i for i, l in enumerate(lab) if l == m][0] for m in MARGS}

    def cos(W, i, j):
        return abs(float(W[i] / np.linalg.norm(W[i]) @ (W[j] / np.linalg.norm(W[j]))))
    WN, iN = axes_of('Naive'); WE, iE = axes_of('Expert')
    prs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    cN = {pr: cos(WN, iN[MARGS[pr[0]]], iN[MARGS[pr[1]]]) for pr in prs}
    cE = {pr: cos(WE, iE[MARGS[pr[0]]], iE[MARGS[pr[1]]]) for pr in prs}
    N = WN.shape[1]; rng = np.random.RandomState(0); Bn = 2000
    boot = {pr: np.empty(Bn) for pr in prs}
    for b in range(Bn):
        ix = rng.randint(0, N, N); wn, we = WN[:, ix], WE[:, ix]
        for pr in prs:
            boot[pr][b] = cos(we, iE[MARGS[pr[0]]], iE[MARGS[pr[1]]]) - cos(wn, iN[MARGS[pr[0]]], iN[MARGS[pr[1]]])
    pval = {pr: 2 * min((boot[pr] > 0).mean(), (boot[pr] < 0).mean()) for pr in prs}
    HL = {(2, 3): '#cc3311', (0, 1): '#377eb8'}
    for pr in prs:
        if pr in HL:
            continue
        ax.plot([0, 1], [cN[pr], cE[pr]], '-', color='0.75', lw=0.9, marker='o', ms=2.5, zorder=2)
    for pr, col in HL.items():
        ax.plot([0, 1], [cN[pr], cE[pr]], '-o', color=col, lw=1.8, ms=4.5, zorder=5)
        ax.annotate(f'{SH[MARGS[pr[0]]]}–{SH[MARGS[pr[1]]]}\np = {pval[pr]:.3f}' if pval[pr] >= 0.001 else f'{SH[MARGS[pr[0]]]}–{SH[MARGS[pr[1]]]}\np < 0.001',
                    (1, cE[pr]), xytext=(5, 0), textcoords='offset points', va='center', ha='left', color=col, fontsize=PS*6.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['naïve', 'expert']); ax.set_xlim(-0.3, 1.9)
    ax.set_ylim(bottom=-0.005); ax.set_ylabel('|cos| between demixed axes')
    ax.set_title('axis alignment', loc='left', fontsize=TITLE_FS)
    for pr in prs:
        print(f'b: {SH[MARGS[pr[0]]]:>6}-{SH[MARGS[pr[1]]]:<6} N {cN[pr]:.3f} -> E {cE[pr]:.3f}  Δ {cE[pr]-cN[pr]:+.3f}  p={pval[pr]:.3f}')


fig = plt.figure(figsize=(10.0, 4.6))
outer = fig.add_gridspec(1, 12, wspace=1.0, left=0.07, right=0.985, top=0.93, bottom=0.12)
gsA = outer[0, 0:9].subgridspec(2, 4, wspace=0.32, hspace=0.18)
axA = panel_a(fig, gsA)
axB = fig.add_subplot(outer[0, 10:12]); panel_b(axB)
axA.text(-0.36, 1.06, 'a', transform=axA.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')
axB.text(-0.36, 1.06, 'b', transform=axB.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')

CAP = [
    'Extended Data Fig. 6 | The demixed-PCA decomposition gives the same picture (companion to Fig. 2). '
    'a, Withheld pseudo-trials projected on the leading demixed axis of each task variable (sample, test, '
    'choice = sample × test, task), naïve (top) and expert (bottom), per condition (mean ± SEM; z-scored per axis). '
    'Time courses along single axes sharpen with learning without reorganizing. b, |cos| between the leading '
    'demixed axes of every pair of variables, naïve → expert (neuron bootstrap, 2,000 resamples of the 3,319 '
    'neurons, two-sided): the choice and task axes become more aligned and the sample and test axes separate; '
    'the other four pairs stay near-orthogonal (grey).',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig6.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig6.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig6.png')
