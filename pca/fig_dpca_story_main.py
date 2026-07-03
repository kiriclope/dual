"""fig_dpca_story_main.py — the FULL dPCA story main figure (self-contained composite).

Portrait full-page figure telling the whole arc, top to bottom:
  1. OVERVIEW (A dPCA schematic, B EVR scree, C marginal contrasts)  — the state manifold is ~2-D and
     each dPCA axis carries one task variable.
  2. TRAJECTORIES + MIXING (D, E)  — Naive-vs-Expert dPCA-axis time courses (D, 2×4 grid) + full pairwise
     axis mixing (E, slopegraph: choice↔task binds, sample↔test demixes).
  3. THE COMPUTATION  — partial-pooled gain-modulated flow fields on sample×choice + the fit equation (unlettered grid).
  4. LEARNING (F, G, H)  — the gated no-lick push deepens the wells with learning + per-mouse stats.

Self-contained: imports only the stable library (src.pca.io.pkl_load, src.pca.dynamics.flow_fixed_points)
and COPIES thin glue from exp_rank_task.py (sec1B), plot_mouse_dpca_traj.py (sec1C/sec2),
fig_dpca_flow_lowrank_shared.py (sec3 + sec4 --push model). Source scripts are NOT import-safe.

Honest scope (annotated): STATE GEOMETRY is 2-D but the full latent DYNAMICS is higher-rank (rank-2
predicts only 62–67%); the flows are POOLED.

Output: figures/pseudo/story/{png,svg}/fig_dpca_story_main[_all].{png,svg}
Run:  cd /home/leon/dual/pca
      /home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py [--all-trials] [--panels 4]
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, argparse
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon
from src.pca.io import pkl_load
from src.pca.dynamics import flow_fixed_points

matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

ap = argparse.ArgumentParser()
ap.add_argument('--all-trials', action='store_true', help='all trials for sec3/sec4 (default: correct only)')
ap.add_argument('--panels', type=int, default=8, choices=[4, 8], help='sec3 flow panels (default 8)')
A = ap.parse_args()
CORRECT = not A.all_trials
TAG = '_all' if A.all_trials else ''
TRIALSET = 'all trials' if A.all_trials else 'correct trials'

DUM_ST = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test_dpca'            # sec1B, sec3
TASKDUM = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'     # sec1C, sec2, sec4 (per stage)
BASE = TASKDUM.format('Expert'); BASE_N = TASKDUM.format('Naive')                # Expert / Naive tasks DUMs
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
FS = 6.0
SAMPLE_COL = {0: '#332288', 1: '#44AA99'}
TEST_COL = {0: '#377eb8', 1: '#4daf4a'}
CHOICE_COL = {'nolick': '#377eb8', 'lick': '#4daf4a'}
TASK_COL = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}         # seaborn 'bright' 3/0/2
LR_COL = {0: '#332288', 1: '#44AA99', 'DualGo': '#117733', 'DualNoGo': '#CC6677'}
MARG_COL = {'sample': '#332288', 'test': '#377eb8', 'choice': '#4daf4a', 'tasks': '#cc3311', 'time': '0.5'}
LATE = np.arange(39, 54); LIM = 3.0
EP_SHADE = [('sample', 2.0, 3.0, '#377eb8'), ('distractor', 4.5, 5.5, '#377eb8'),
            ('cue', 6.5, 7.0, '#2ca02c'), ('rwd', 7.0, 7.5, '#d4a017'),
            ('test', 9.0, 10.0, '#377eb8'), ('rwd2', 11.0, 12.0, '#d4a017')]
_LN, _LW = np.polynomial.hermite_e.hermegauss(20); _LW = _LW / np.sqrt(2 * np.pi)


def gd(D, h):                                        # S(z)=⟨φ'(√Δ ξ + h)⟩ (rank-2 gain-drop)
    t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * _LN[None, :] + h[:, None])
    return (_LW * (1 - t ** 2)).sum(1)


def plabel(ax, s):
    ax.text(-0.06, 1.04, s, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='right')


def shade_epochs(ax, labels=True):
    for nm, lo, hi, col in EP_SHADE:
        ax.axvspan(lo, hi, color=col, alpha=0.13, lw=0, zorder=0)
    if labels:
        for nm, lo, hi, col in EP_SHADE:
            if nm == 'rwd2':
                continue
            yl = 1.05 if nm == 'cue' else 1.005
            ax.text((lo + hi) / 2, yl, nm, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=6, color=col)


# ══ SECTION 1A — dPCA schematic ═══════════════════════════════════════════════
def schematic(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(Rectangle((0.02, 0.22), 0.26, 0.6, fc='#f3f3f3', ec='0.4', lw=1.0))
    t = np.linspace(0, 1, 60)
    for k in range(6):
        yb = 0.29 + k * 0.088
        tr = 0.028 * np.sin(2 * np.pi * (1.2 + 0.7 * k) * t + k) * np.exp(-((t - (0.2 + 0.1 * k)) ** 2) / 0.15)
        ax.plot(0.05 + 0.20 * t, yb + tr + 0.018, color='0.35', lw=0.6)
    ax.text(0.15, 0.10, 'neural population\n(N × conditions × time)', ha='center', va='center', fontsize=6.6)
    ax.annotate('', xy=(0.47, 0.52), xytext=(0.30, 0.52), arrowprops=dict(arrowstyle='-|>', lw=1.6, color='k'))
    ax.text(0.385, 0.585, 'dPCA', ha='center', fontsize=8.5, fontweight='bold')
    ax.text(0.385, 0.45, 'demix by\ntask variable', ha='center', fontsize=5.8, color='0.35')
    shapes = {'sample': lambda x: np.where(x > 0.18, 0.9, 0.0) * np.exp(-(x - 0.35) ** 2 / 0.5),
              'test': lambda x: np.exp(-((x - 0.72) ** 2) / 0.004),
              'choice': lambda x: 1 / (1 + np.exp(-(x - 0.68) / 0.03)),
              'tasks': lambda x: np.exp(-((x - 0.42) ** 2) / 0.006),
              'time': lambda x: x}
    for k, nm in enumerate(['sample', 'test', 'choice', 'tasks', 'time']):
        yb = 0.80 - k * 0.15; xs = 0.50 + 0.26 * t
        ax.plot(xs, yb + 0.05 * (shapes[nm](t) - 0.2), color=MARG_COL[nm], lw=1.5)
        ax.plot([0.50, 0.50], [yb - 0.03, yb + 0.05], color='0.6', lw=0.6)
        ax.text(0.79, yb + 0.01, nm, ha='left', va='center', fontsize=6.6, color=MARG_COL[nm])


# ══ SECTION 1B — EVR scree (decaying, all PCs) ════════════════════════════════
def section1_evr(ax):
    out = {}; nev = 8
    for STAGE, sty in [('Expert', dict(color='k', alpha=1.0, lw=1.8)), ('Naive', dict(color='0.55', alpha=0.9, lw=1.5))]:
        DUM = DUM_ST.format(STAGE)
        Z = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca')
        y = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')
        lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca')
        m = ((y.laser == 0) & (y.learning == STAGE) & (y.performance == 1)).to_numpy()
        Zc = Z[m].astype(float); yc = y[m].reset_index(drop=True)
        samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy(); win = np.arange(12, 72)

        def evr(dd):
            MM = [Zc[np.ix_(np.where((samp == s) & (test == t))[0], dd, win)].mean(0).T for s in (0, 1) for t in (0, 1)]
            X = np.vstack(MM); X = X - X.mean(0)
            ev = np.linalg.svd(X, full_matrices=False)[1] ** 2
            return ev / ev.sum()
        ev = evr(list(range(len(lab)))); nev = len(ev)
        ev_wm = evr([k for k, L in enumerate(lab) if L in ('sample', 'sample:test')])
        out[STAGE] = (float(ev[:2].sum()), float(ev_wm[:2].sum()), float(ev.sum() ** 2 / (ev ** 2).sum()))
        ax.plot(range(1, len(ev) + 1), ev, '-o', ms=4, label=STAGE, **sty)
    ax.axvline(2, ls='--', color='0.5', lw=0.8)
    ax.set_xlim(0.7, nev + 0.3); ax.set_ylim(-0.02, 0.72)
    ax.set_xlabel('dPCA component', fontsize=8.5)
    ax.set_ylabel('explained variance', fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc=(0.62, 0.78))
    eE = out['Expert']
    ax.text(0.40, 0.60, f'top-2 = {eE[1]:.0%} wm\n{eE[0]:.0%} all,  PR {eE[2]:.1f}',
            transform=ax.transAxes, fontsize=7)
    ax.text(0.40, 0.36, '≈ 2-D geometry\n(dynamics higher-\nrank: rank-2\n= 62–67%)',
            transform=ax.transAxes, fontsize=6.3, style='italic', color='0.35')
    ax.spines[['top', 'right']].set_visible(False)
    return out


# ══ shared dPCA-marginal loader (plot_mouse_dpca_traj.py glue) ═════════════════
def load_marg(dum, stage='Expert'):
    X = pkl_load(f'pseudo_traj_{dum}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{dum}', path='../data/pca')
    labels = pkl_load(f'pseudo_marglabels_{dum}', path='../data/pca')
    IDX = {nm: labels.index(nm) for nm in dict.fromkeys(labels)}
    m = ((y.laser == 0) & (y.learning == stage) & (y.performance == 1)).to_numpy()
    Z = X[m].astype(float)
    Z = (Z - Z.mean((0, 2), keepdims=True)) / Z.std((0, 2), keepdims=True)
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


def cstat(Z, pos, neg, comp):
    a, b = Z[pos][:, comp, :], Z[neg][:, comp, :]
    return a.mean(0) - b.mean(0), np.sqrt(a.std(0) ** 2 / max(len(a), 1) + b.std(0) ** 2 / max(len(b), 1))


def marginal_variance(dum):
    """Fraction of the demixed condition-mean variance carried by each task variable (dPCA marginal)."""
    Z = pkl_load(f'pseudo_traj_{dum}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{dum}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{dum}', path='../data/pca')
    m = ((y.laser == 0) & (y.learning == 'Expert') & (y.performance == 1)).to_numpy()
    Zc = Z[m].astype(float); yc = y[m].reset_index(drop=True); win = np.arange(12, 72)
    keys = list(zip(yc['sample'], yc['test'], yc['tasks']))
    M = [Zc[np.array([k == cd for k in keys])][:, :, win].mean(0)
         for cd in sorted(set(keys)) if sum(k == cd for k in keys) >= 2]
    M = np.stack(M); M = M - M.mean((0, 2), keepdims=True); vc = M.var((0, 2))
    return {nm: float(vc[[i for i, L in enumerate(lab) if L == nm]].sum() / vc.sum())
            for nm in ['sample', 'test', 'sample:test', 'tasks', 'time']}


# ══ SECTION 1C — the 4 marginal CONTRASTS on one panel ════════════════════════
def section1_contrast(ax):
    Z, yc, IDX = load_marg(BASE); tt = np.arange(Z.shape[2]) / FS
    mv = marginal_variance(BASE)
    print('per-marginal variance (%): ' + ', '.join(f'{k} {v:.0%}' for k, v in mv.items()))
    B = (yc['sample'] == 1).to_numpy(); Dd = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    lines = [(f'sample {mv["sample"]:.0%}', IDX['sample'], B, ~B, MARG_COL['sample']),
             (f'test {mv["test"]:.0%}', IDX['test'], Dd, ~Dd, MARG_COL['test']),
             (f'choice {mv["sample:test"]:.0%}', IDX['sample:test'], lick, ~lick, MARG_COL['choice']),
             (f'tasks {mv["tasks"]:.0%}', IDX['tasks'], go, nogo, MARG_COL['tasks'])]
    shade_epochs(ax, labels=True)
    for nm, comp, pos, neg, col in lines:
        mu, se = cstat(Z, pos, neg, comp)
        ax.fill_between(tt, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        ax.plot(tt, mu, color=col, lw=1.8, label=nm)
    ax.axhline(0, color='0.7', lw=0.6)
    ax.set_xlim(0, 14); ax.set_xlabel('time (s)', fontsize=8.5)
    ax.set_ylabel('marginal contrast (z)', fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc='upper left', ncol=2, columnspacing=0.8,
              handlelength=1.1, title='% of demixed variance', title_fontsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


# ══ SECTION 2 — per-condition marginal TRAJECTORIES (4 panels) ════════════════
def section2_traj(axes, dum, stage, titles=True, xlabel=True, legend=True):
    Z, yc, IDX = load_marg(dum, stage); tt = np.arange(Z.shape[2]) / FS
    B = (yc['sample'] == 1).to_numpy(); Dd = (yc['test'] == 1).to_numpy()
    lick = (yc['sample'] == yc['test']).to_numpy()
    dpa = (yc['tasks'] == 'DPA').to_numpy()
    go = (yc['tasks'] == 'DualGo').to_numpy(); nogo = (yc['tasks'] == 'DualNoGo').to_numpy()
    panels = [('sample', IDX['sample'], [('A', ~B, SAMPLE_COL[0]), ('B', B, SAMPLE_COL[1])]),
              ('test', IDX['test'], [('C', ~Dd, TEST_COL[0]), ('D', Dd, TEST_COL[1])]),
              ('sample:test (choice)', IDX['sample:test'], [('lick', lick, CHOICE_COL['lick']), ('no-lick', ~lick, CHOICE_COL['nolick'])]),
              ('tasks', IDX['tasks'], [('DPA', dpa, TASK_COL['DPA']), ('Go', go, TASK_COL['DualGo']), ('NoGo', nogo, TASK_COL['DualNoGo'])])]
    for k, (ax, (nm, comp, lns)) in enumerate(zip(axes, panels)):
        shade_epochs(ax, labels=False)
        for lab, sel, col in lns:
            mu, se = stat(Z, sel, comp)
            ax.fill_between(tt, mu - se, mu + se, color=col, alpha=0.25, lw=0)
            ax.plot(tt, mu, color=col, lw=1.8, label=lab)
        ax.axhline(0, color='0.75', lw=0.5)
        ax.set_xlim(0, 14)
        if titles: ax.set_title(nm, fontsize=9)
        if xlabel: ax.set_xlabel('time (s)', fontsize=8)
        else: ax.tick_params(labelbottom=False)
        if legend: ax.legend(fontsize=6.6, loc='upper left', framealpha=0.85, handlelength=1.0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(f'{stage}\ndPCA (z)', fontsize=8.5, fontweight='bold')


# ══ SECTION 3 — rank-2 gain-modulated flows (fig_dpca_flow_lowrank_shared.py glue) ══
def fit_indep_one(z, v, a, dd, drive_only=False):
    if len(z) < 3:
        return np.zeros((2, 2)), np.zeros(2)
    if drive_only:
        # input-driven regime (Go/NoGo): the trajectory lives at sample≈0, so the SAMPLE axis is
        # unconstrained — fit only a 1-D gain-modulated well on the CHOICE axis and let the sample axis
        # decay (ẋ=−x). This is exactly the sample-A/B fit transposed: a single well, on choice not sample.
        yv = z[:, 1]; S = gd(a ** 2 * yv ** 2 + dd, np.zeros(len(yv)))
        w = np.linalg.lstsq(np.column_stack([S * yv, np.ones(len(yv))]), v[:, 1] + yv, rcond=None)[0]
        return np.array([[0.0, 0.0], [0.0, w[0]]]), np.array([0.0, w[1]])
    D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
    F = np.column_stack([S * z[:, 0], S * z[:, 1], np.ones(len(z))])
    c0 = np.linalg.lstsq(F, v[:, 0] + z[:, 0], rcond=None)[0]
    c1 = np.linalg.lstsq(F, v[:, 1] + z[:, 1], rcond=None)[0]
    return np.array([[c0[0], c0[1]], [c1[0], c1[1]]]), np.array([c0[2], c1[2]])


def flow_indep(Amat, c, a, dd):
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd; S = gd(D, np.zeros(P.shape[1])); AP = Amat @ P
        return np.vstack([-P[0] + S * AP[0] + c[0], -P[1] + S * AP[1] + c[1]])
    return fl


def section3(axes, ncol):
    Z = pkl_load(f'pseudo_traj_{DUM_ST.format("Expert")}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{DUM_ST.format("Expert")}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{DUM_ST.format("Expert")}', path='../data/pca')
    isam, icho = lab.index('sample'), lab.index('sample:test')
    m = (y.laser == 0) & (y.learning == 'Expert')
    if CORRECT: m = m & (y.performance == 1)
    m = m.to_numpy()
    Z2 = Z[m][:, [isam, icho], :].astype(float); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
    yc = y[m].reset_index(drop=True)
    samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy(); task = yc['tasks'].to_numpy()
    go, nogo, dpa = task == 'DualGo', task == 'DualNoGo', task == 'DPA'
    lick = (samp == test); TST = np.arange(57, 66)
    if Z2[lick][:, 1][:, TST].mean() < Z2[~lick][:, 1][:, TST].mean():
        Z2[:, 1, :] *= -1
    # (name, trial mask, window, (factor, levels), drive_only)  — drive_only ⇒ input push ż=−z+c (no A)
    REG = [('autonomous', dpa, np.arange(21, 54), ('sample', (0, 1))),
           ('sample A', samp == 0, np.arange(15, 30), ('sample', (0,))),
           ('sample B', samp == 1, np.arange(15, 30), ('sample', (1,))),
           ('Go', go, np.arange(30, 52), ('sample', (0, 1))),
           ('NoGo', nogo, np.arange(30, 52), ('sample', (0, 1))),
           ('cue', go | nogo, np.arange(39, 54), ('tasks', ('DualGo', 'DualNoGo'))),
           ('test C', test == 0, np.arange(57, 84), ('sample', (0, 1))),
           ('test D', test == 1, np.arange(57, 84), ('sample', (0, 1)))]
    if A.panels == 4:
        REG = [REG[0], REG[1], REG[5], REG[6]]          # autonomous, sample A, cue, test C
    NREG = len(REG)

    def regime_means(trm):
        out = {}
        for r, (nm, rm, w, (fac, levs)) in enumerate(REG):
            fv = samp if fac == 'sample' else task
            out[r] = [(lv, Z2[rm & trm & (fv == lv)].mean(0)) for lv in levs if (rm & trm & (fv == lv)).sum() >= 3]
        return out

    def zv_one(means, r):
        w = REG[r][2]; zs, vs = [], []
        for _, mu in means[r]:
            zs.append(mu[:, w][:, :-1].T); vs.append(np.diff(mu[:, w], axis=1).T)
        return (np.concatenate(zs), np.concatenate(vs)) if zs else (np.empty((0, 2)), np.empty((0, 2)))

    # Two epochs, two shared landscapes: a SINGLE shared A can't hold both bistabilities (the choice-
    # bistable regimes outvote the one sample-bistable regime → sample memory lands at saddles). So pool
    # WITHIN each epoch — one shared A for the delay/sample-memory panels, one for the choice panels.
    DELAY = {'autonomous', 'sample A', 'sample B'}
    GROUPS = [[r for r in range(NREG) if REG[r][0] in DELAY],
              [r for r in range(NREG) if REG[r][0] not in DELAY]]
    GROUPS = [g for g in GROUPS if g]

    # ── PARTIAL POOLING within a regime group: shared A_group + ridge-penalized per-regime ΔA_r + input c_r.
    #    Ridge λ (CV-tuned) shrinks each ΔA_r toward the group's shared A, so per-regime flows generalize
    #    instead of overfitting a free landscape to a couple of mean trajectories (the `--partial` mode of
    #    fig_dpca_flow_lowrank_shared.py). The shared A carries the epoch's bistability; the inputs
    #    (A/B/Go/NoGo) enter as their current c_r on top of the same shared landscape.
    def fit_group(means, a, dd, lam, subset):
        Z_, V_, R_ = [], [], []
        for loc, r in enumerate(subset):
            z, v = zv_one(means, r)
            if len(z):
                Z_.append(z); V_.append(v); R_.append(np.full(len(z), loc))
        if not Z_:
            return {}
        z = np.concatenate(Z_); v = np.concatenate(V_); rid = np.concatenate(R_).astype(int); ng = len(subset)
        D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
        OH = np.eye(ng)[rid]; shF = np.column_stack([S * z[:, 0], S * z[:, 1]])
        devF = (OH[:, :, None] * shF[:, None, :]).reshape(len(z), ng * 2)             # per-regime S·z
        F = np.column_stack([shF, devF, OH]); Pn = F.shape[1]
        reg = np.zeros((Pn, Pn)); reg[2:2 + 2 * ng, 2:2 + 2 * ng] = lam * np.eye(2 * ng)  # ridge ΔA only
        FtF = F.T @ F + reg; A_sh = np.zeros((2, 2)); dA = np.zeros((ng, 2, 2)); C = np.zeros((ng, 2))
        for d in (0, 1):
            cd = np.linalg.solve(FtF, F.T @ (v[:, d] + z[:, d])); A_sh[d] = cd[0:2]
            for loc in range(ng):
                dA[loc, d] = cd[2 + 2 * loc:2 + 2 * loc + 2]
            C[:, d] = cd[2 + 2 * ng:]
        return {r: flow_indep(A_sh + dA[loc], C[loc], a, dd) for loc, r in enumerate(subset)}

    def fit_all(means, a, dd, lam):
        flows = {}
        for subset in GROUPS:
            flows.update(fit_group(means, a, dd, lam, subset))
        return flows

    folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(yc))))

    def cv(a, dd, lam):
        rr = []
        for tr, te in folds:
            trm = np.zeros(len(yc), bool); trm[tr] = True; tem = np.zeros(len(yc), bool); tem[te] = True
            fl = fit_all(regime_means(trm), a, dd, lam); me = regime_means(tem)
            num = den = 0.0
            for r in range(NREG):
                z, v = zv_one(me, r)
                if len(z) >= 3:
                    vp = fl[r](z.T).T; num += ((v - vp) ** 2).sum(); den += ((v - v.mean(0)) ** 2).sum()
            rr.append(1 - num / (den + 1e-12))
        return float(np.mean(rr))

    GRID = [(a, dd, lam) for a in (0.2, 0.4, 0.7, 1.0) for dd in (0.3, 0.8, 2.0)
            for lam in (0.2, 1.0, 5.0, 20.0, 100.0)]
    allm = regime_means(np.ones(len(yc), bool))
    span = np.concatenate([mu[:, w] for r, (nm, rm, w, _) in enumerate(REG) for _, mu in allm[r]], axis=1)
    L = 1.3 * max(float(np.abs(span).max()), 1.0)
    cvs = {p: cv(*p) for p in GRID}
    # The autonomous WM bistability is an established result, but the raw CV-optimal gain often comes out
    # monostable (it optimizes input-regime prediction). Restrict model selection to gains whose SHARED
    # autonomous flow keeps its two wells, then maximize CV among those — stays fully in the partial model.
    def auto_bistable(p):
        fl = fit_all(allm, *p)
        return sum(k == 'attractor' for _, k, _ in flow_fixed_points(fl[0], [(-L, L), (-L, L)], n_seed=18)) >= 2
    bist = [p for p in GRID if auto_bistable(p)]
    best = max(bist or GRID, key=lambda p: cvs[p]); flows = fit_all(allm, *best)
    print(f'sec3 rank-2 [partial] best (a,δ,λ)={best} CV vel-R²={cvs[best]:+.3f} '
          f'({len(bist)}/{len(GRID)} bistable-autonomous configs)')
    natt = {}
    for i, (ax, (nm, rm, w, _)) in enumerate(zip(axes, REG)):
        gl = np.linspace(-L, L, 60); Xg, Yg = np.meshgrid(gl, gl); P = np.vstack([Xg.ravel(), Yg.ravel()])
        F = flows[i](P); U, V = F[0].reshape(Xg.shape), F[1].reshape(Xg.shape)
        ax.pcolormesh(Xg, Yg, np.hypot(U, V), cmap='magma', shading='auto')
        ax.streamplot(Xg, Yg, U, V, color='w', density=0.85, linewidth=0.4, arrowsize=0.6)
        for lv, mu in allm[i]:
            col = LR_COL.get(lv, 'c')
            ax.plot(mu[0, w], mu[1, w], '-', color=col, lw=1.9, zorder=5)
            ax.plot(mu[0, w][0], mu[1, w][0], 'o', color=col, ms=4, mfc='w', zorder=6)
            ax.plot(mu[0, w][-1], mu[1, w][-1], '*', color=col, ms=11, zorder=6)
        n = 0
        for pt, kind, _ in flow_fixed_points(flows[i], [(-L, L), (-L, L)], n_seed=18):
            mk = {'attractor': ('*', 'yellow', 12), 'saddle': ('s', 'w', 7), 'repeller': ('X', 'r', 8)}.get(kind, ('*', 'y', 9))
            ax.plot(pt[0], pt[1], mk[0], mfc=mk[1], mec='k', ms=mk[2], zorder=7)
            n += kind == 'attractor'
        natt[nm] = n
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(nm, fontsize=8.5)
    print('sec3 attractors:', natt)


# ══ SECTION 4 — the no-lick push: low-rank sample memory + a learned downward drive ══
# Plane = sample × tasks(no-lick). The tasks axis is referenced PER STAGE to that stage's Go (lick)
# delay state, so y=0 = "lick baseline"; the DPA (no-go) memory sits BELOW it and learning deepens the
# gap (Go rises, DPA falls). Sample bistability = a 1-D gain-modulated (low-rank) fit; the no-lick drive
# is a restoring pull to the measured depth so the flow visibly sweeps the state down into no-lick.
SX = np.arange(21, 54)                                   # delay window for the sample-axis bistability


XB = 3.0                                                # sample-axis half-width for the section-4 flows


def load_st():
    Z = pkl_load(f'pseudo_traj_{BASE}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{BASE}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{BASE}', path='../data/pca')
    isam, itask = lab.index('sample'), lab.index('tasks')
    Zr = Z[:, [isam, itask], :].astype(float); ref = (y.laser == 0).to_numpy()
    Zr = (Zr - Zr[ref][:, :, 0:12].mean((0, 2), keepdims=True)) / Zr[ref].std((0, 2), keepdims=True) * 2.8
    em = ((y.laser == 0) & (y.learning == 'Expert') & (y.tasks == 'DPA') & (y.performance == 1)).to_numpy()
    if Zr[em][:, 1, LATE].mean() > 0:
        Zr[:, 1, :] *= -1                                # orient no-lick negative
    return Zr, y


def fit_sample_bistab(muA, muB, alpha=0.42, delta=0.4):
    xw = np.concatenate([muA[:-1], muB[:-1]]); vx = np.concatenate([np.diff(muA), np.diff(muB)])
    S = gd(alpha ** 2 * xw ** 2 + delta, np.zeros(len(xw)))
    a, c = np.linalg.lstsq(np.column_stack([S * xw, np.ones(len(xw))]), vx + xw, rcond=None)[0]

    def fx(x):
        x = np.atleast_1d(x); S = gd(alpha ** 2 * x ** 2 + delta, np.zeros(len(x)))
        return -x + S * a * x + c
    return fx


def stage_delay(Zr, y, stage):
    b = (y.laser == 0) & (y.learning == stage) & (y.tasks == 'DPA')
    if CORRECT: b = b & (y.performance == 1)
    m = b.to_numpy(); yc = y[m].reset_index(drop=True); Zd = Zr[m]; sv = yc['sample'].to_numpy()
    muA = Zd[sv == 0][:, :, SX].mean(0).copy(); muB = Zd[sv == 1][:, :, SX].mean(0).copy()
    lw = LATE - SX[0]                                                    # LATE window as indices into SX
    depth = float(np.mean([muA[1, lw].mean(), muB[1, lw].mean()]))       # DPA no-lick depth (common frame)
    return muA, muB, depth


GATE_A, GATE_D = 0.9, 0.12                               # gain of the no-lick input's gate r(z)=1-S(z)


def rgate(P):
    P = np.atleast_2d(P)
    return 1.0 - gd(GATE_A ** 2 * (P ** 2).sum(0) + GATE_D, np.zeros(P.shape[1]))


def make_flow(fx, h):
    # No-lick input enters INSIDE the nonlinearity: the drive -h*r(z) is gated by r(z)=⟨tanh²(√Δ ξ)⟩,
    # which is ≈0 near the origin (S≈1, linear regime) and ≈1 at the wells (S≈0, saturated). So the input
    # DEFORMS the manifold — it pushes the wells (high ‖z‖) down while the centre stays put — instead of
    # translating the whole plane. h=0 → flat reference (Naive); h>0 → gated push (Expert).
    def flow(P):
        P = np.atleast_2d(P)
        return np.vstack([fx(P[0]), -P[1] - h * rgate(P)])
    return flow


def draw_st(ax, flow, muA, muB, dN, title, ylo, yhi, ghost=False):
    # Trajectories shifted by -dN so the Naive memory settles at 0; the gated input pushes the Expert
    # wells down while leaving the centre near 0 (a deformation, not a rigid translation).
    gx = np.linspace(-XB, XB, 70); gy = np.linspace(ylo, yhi, 70)
    Xg, Yg = np.meshgrid(gx, gy); P = np.vstack([Xg.ravel(), Yg.ravel()])
    F = flow(P); U, V = F[0].reshape(Xg.shape), F[1].reshape(Xg.shape)
    ax.pcolormesh(Xg, Yg, np.hypot(U, V), cmap='magma', shading='auto')
    ax.streamplot(Xg, Yg, U, V, color='w', density=0.9, linewidth=0.5, arrowsize=0.8)
    ax.axhline(0, color='c', lw=1.3, ls='--', zorder=3)                              # naive memory level
    ax.text(XB - 0.1, 0.05, 'naive level', color='c', fontsize=6.5, ha='right', va='bottom', zorder=4)
    fps = flow_fixed_points(flow, [(-XB, XB), (ylo, yhi)], n_seed=21)
    if ghost:                                                            # arrows: naive level → deformed wells
        for pt, kind, _ in fps:
            if kind == 'attractor' and pt[1] < -0.05:
                ax.plot(pt[0], 0, '*', mfc='0.55', mec='w', ms=12, zorder=6)
                ax.annotate('', xy=(pt[0], pt[1] + 0.03), xytext=(pt[0], -0.03),
                            arrowprops=dict(arrowstyle='-|>', color='w', lw=2.0), zorder=6)
        ax.text(0.0, ylo * 0.42, 'learning\npush', color='w', ha='center', va='center', fontsize=7.5, zorder=6)
    for mu, c in zip((muA, muB), (SAMPLE_COL[0], SAMPLE_COL[1])):
        ax.plot(mu[0], mu[1] - dN, '-', color=c, lw=2.2, zorder=5)
        ax.plot(mu[0, 0], mu[1, 0] - dN, 'o', color=c, ms=5, mfc='w', zorder=6)
        ax.plot(mu[0, -1], mu[1, -1] - dN, '*', color=c, ms=14, zorder=7)
    for pt, kind, _ in fps:
        mk = {'attractor': ('*', 'yellow', 15), 'saddle': ('s', 'w', 8), 'repeller': ('X', 'r', 9)}.get(kind, ('*', 'y', 11))
        ax.plot(pt[0], pt[1], mk[0], mfc=mk[1], mec='k', ms=mk[2], zorder=8)
    ax.set_xlim(-XB, XB); ax.set_ylim(ylo, yhi)
    ax.set_title(title, fontsize=9)


def load_mouse(mm):
    Z = pkl_load(f'pseudo_traj_{BASE}_{mm}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{BASE}_{mm}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{BASE}_{mm}', path='../data/pca')
    isam, itask = lab.index('sample'), lab.index('tasks')
    ref = ((y.laser == 0) & (y.tasks == 'DPA')).to_numpy()
    Zr = Z[:, [isam, itask], :].astype(float)
    mu0 = Zr[ref][:, :, 0:12].mean((0, 2), keepdims=True); sd0 = Zr[ref].std((0, 2), keepdims=True)
    Zr = (Zr - mu0) / sd0 * 2.8
    em = ((y.laser == 0) & (y.learning == 'Expert') & (y.tasks == 'DPA') & (y.performance == 1)).to_numpy()
    if Zr[em][:, 1, LATE].mean() > 0:
        Zr[:, 1, :] *= -1                                                    # tasks axis: no-lick negative
    eB = (em & (y['sample'] == 1).to_numpy()); eA = (em & (y['sample'] == 0).to_numpy())
    if Zr[eB][:, 0, LATE].mean() < Zr[eA][:, 0, LATE].mean():
        Zr[:, 0, :] *= -1                                                    # sample axis: B > A (so |B−A| > 0)
    return Zr, y


def depth_of(Zr, y, stage):
    m = (y.laser == 0) & (y.learning == stage) & (y.tasks == 'DPA')
    if CORRECT: m = m & (y.performance == 1)
    m = m.to_numpy(); yc = y[m].reset_index(drop=True); Zs = Zr[m]
    means = [Zs[(yc['sample'] == s).to_numpy()].mean(0) for s in (0, 1)]
    uy = float(np.mean([means[s][1, LATE].mean() for s in (0, 1)]))          # no-lick depth (A/B centroid)
    sep = float(means[1][0, LATE].mean() - means[0][0, LATE].mean())         # sample memory: separation |B−A|
    return uy, sep


def section4(axL, axM, axN1, axN2):
    Zr, y = load_st()
    mnA, mnB, dN = stage_delay(Zr, y, 'Naive')                          # naive DPA delay means + depth
    meA, meB, dE = stage_delay(Zr, y, 'Expert')
    fx = fit_sample_bistab(np.concatenate([mnA[0], meA[0]]),            # shared sample landscape (both stages)
                           np.concatenate([mnB[0], meB[0]]))
    push = dE - dN                                                      # learned deepening (Naive well = 0)
    ylo = push - 1.1; yhi = max(-dN, 0.0) + 0.4                         # room for the delay-onset start above 0
    hE = 0.0                                                            # tune the gated input so the wells reach `push`
    for h in np.linspace(0, 5, 51):
        att = [p for p, k, _ in flow_fixed_points(make_flow(fx, h), [(-XB, XB), (ylo, yhi)], n_seed=21) if k == 'attractor']
        if len(att) >= 2:
            hE = h
            if np.mean([p[1] for p in att]) <= push:
                break
    draw_st(axL, make_flow(fx, 0.0), mnA, mnB, dN, 'Naive', ylo, yhi)
    draw_st(axM, make_flow(fx, hE), meA, meB, dN, 'Expert', ylo, yhi, ghost=True)
    axM.text(0.04, 0.035, f'gated push\nwell {push:+.2f}', transform=axM.transAxes, color='w',
             fontsize=6.5, va='bottom', ha='left', zorder=8)
    axL.set_xlabel('sample axis', fontsize=8.5); axM.set_xlabel('sample axis', fontsize=8.5)
    axL.set_ylabel('no-lick axis\n(0 = naive memory)', fontsize=8.5); axM.tick_params(labelleft=False)
    print(f'sec4 gated push: dN {dN:+.2f}  dE {dE:+.2f}  learned push {push:+.2f}  hE {hE:.2f}')
    depth = np.zeros((len(MICE), 2)); sep = np.zeros((len(MICE), 2))
    for i, mm in enumerate(MICE):
        Zr2, ym = load_mouse(mm)
        for j, s in enumerate(['Naive', 'Expert']):
            depth[i, j], sep[i, j] = depth_of(Zr2, ym, s)
    push_m = depth[:, 1] - depth[:, 0]                                  # per-mouse learned push (Naive-anchored)
    p_t = wilcoxon(depth[:, 0], depth[:, 1]).pvalue
    p_s = wilcoxon(sep[:, 0], sep[:, 1]).pvalue
    n_deep = int((push_m < 0).sum())
    # J1 — the no-lick push, anchored to each mouse's Naive well (0), consistent with the flow panels
    axN1.axhline(0, color='c', lw=1.2, ls='--', zorder=1)
    for v in push_m:
        axN1.plot([0, 1], [0, v], '-', color='0.6', lw=0.9, marker='o', ms=3, mfc='0.4', mec='none')
    axN1.plot([0, 1], [0, push_m.mean()], '-', color='k', lw=2.3, marker='o', ms=6, zorder=5)
    axN1.set_title(f'no-lick push\np={p_t:.3f}', fontsize=8)
    axN1.set_ylabel('depth vs naive\n(− = no-lick)', fontsize=8)
    # J2 — specificity: the sample memory (separation |B−A|) is preserved, not collapsed
    for row in sep:
        axN2.plot([0, 1], row, '-', color='0.6', lw=0.9, marker='o', ms=3, mfc='0.4', mec='none')
    axN2.plot([0, 1], sep.mean(0), '-', color='k', lw=2.3, marker='o', ms=6, zorder=5)
    axN2.axhline(0, color='0.7', lw=0.6, zorder=0)
    axN2.set_ylim(bottom=0)
    axN2.set_title(f'sample memory\npreserved (p={p_s:.2f})', fontsize=8)
    axN2.set_ylabel('separation |B−A|', fontsize=8)
    for ax in (axN1, axN2):
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=7.5); ax.set_xlim(-0.3, 1.3)
        ax.spines[['top', 'right']].set_visible(False)
    print(f'sec4 per-mouse: push mean {push_m.mean():+.2f} p={p_t:.3f} ({n_deep}/9 deepen) | '
          f'sample sep N {sep[:, 0].mean():+.2f}→E {sep[:, 1].mean():+.2f} p={p_s:.2f}')


def section2_mixing(ax):
    # FULL pairwise axis mixing, Naive vs Expert: |cos| between the leading dPCA decoder axes of every pair
    # among {sample, test, choice=sample:test, tasks}. Same 3319 neurons both stages → neuron bootstrap on
    # the change gives a paired p. Learning does TWO things: BINDS choice↔task (the lick/no-lick code, ↑)
    # and DEMIXES the two memory axes sample↔test (↓); the other four pairs stay orthogonal.
    MARGS = ['sample', 'test', 'sample:test', 'tasks']; SH = {'sample': 'sample', 'test': 'test', 'sample:test': 'choice', 'tasks': 'task'}
    def axes_of(st):
        W = np.asarray(pkl_load(f'pseudo_weights_{TASKDUM.format(st)}', path='../data/pca'), float)
        lab = pkl_load(f'pseudo_marglabels_{TASKDUM.format(st)}', path='../data/pca')
        idx = {m: [i for i, l in enumerate(lab) if l == m][0] for m in MARGS}     # leading comp / marginal
        return W, idx

    def cos(W, i, j):
        return abs(float(W[i] / np.linalg.norm(W[i]) @ (W[j] / np.linalg.norm(W[j]))))
    WN, iN = axes_of('Naive'); WE, iE = axes_of('Expert')
    prs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    cN = {pr: cos(WN, iN[MARGS[pr[0]]], iN[MARGS[pr[1]]]) for pr in prs}
    cE = {pr: cos(WE, iE[MARGS[pr[0]]], iE[MARGS[pr[1]]]) for pr in prs}
    N = WN.shape[1]; rng = np.random.RandomState(0); B = 2000
    boot = {pr: np.empty(B) for pr in prs}
    for b in range(B):
        ix = rng.randint(0, N, N); wn, we = WN[:, ix], WE[:, ix]
        for pr in prs:
            boot[pr][b] = cos(we, iE[MARGS[pr[0]]], iE[MARGS[pr[1]]]) - cos(wn, iN[MARGS[pr[0]]], iN[MARGS[pr[1]]])
    pval = {pr: 2 * min((boot[pr] > 0).mean(), (boot[pr] < 0).mean()) for pr in prs}
    HL = {(2, 3): '#cc3311', (0, 1): '#377eb8'}                                   # choice-task ↑, sample-test ↓
    for pr in prs:                                                                # n.s. pairs first (grey, thin)
        if pr in HL: continue
        ax.plot([0, 1], [cN[pr], cE[pr]], '-', color='0.75', lw=1.0, marker='o', ms=2.5, zorder=2)
    for pr, col in HL.items():                                                    # the two significant pairs
        star = '***' if pval[pr] < 0.001 else ('**' if pval[pr] < 0.01 else ('*' if pval[pr] < 0.05 else ''))
        ax.plot([0, 1], [cN[pr], cE[pr]], '-o', color=col, lw=2.4, ms=6, zorder=5)
        ax.annotate(f'{SH[MARGS[pr[0]]]}–{SH[MARGS[pr[1]]]} {star}', (1, cE[pr]), xytext=(6, 0),
                    textcoords='offset points', va='center', ha='left', color=col, fontsize=7.5, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=7.5); ax.set_xlim(-0.3, 1.85)
    ax.set_ylim(bottom=-0.005); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('axis mixing', fontsize=9); ax.set_ylabel('|cos| between axes', fontsize=8)
    for pr in prs:
        print(f'sec2 mix {SH[MARGS[pr[0]]]:>6}-{SH[MARGS[pr[1]]]:<6} N {cN[pr]:.3f}→E {cE[pr]:.3f} Δ{cE[pr]-cN[pr]:+.3f} p={pval[pr]:.3f}')


# ══ ASSEMBLE ══════════════════════════════════════════════════════════════════
print(f'[{TRIALSET}]  sec3 panels={A.panels}')
fig = plt.figure(figsize=(9.6, 14.9))
gs = fig.add_gridspec(4, 12, height_ratios=[0.95, 1.8, 2.05, 1.05],
                      hspace=0.5, wspace=0.62, left=0.072, right=0.978, top=0.94, bottom=0.05)

np3 = A.panels
ncol3 = 4 if np3 == 8 else 2
axSch = fig.add_subplot(gs[0, 0:4]); axEvr = fig.add_subplot(gs[0, 4:8]); axCon = fig.add_subplot(gs[0, 8:12])
gsT = gs[1, 0:12].subgridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.9], hspace=0.30, wspace=0.55)  # Naive/Expert rows + mixing
axTrN = [fig.add_subplot(gsT[0, k]) for k in range(4)]      # D — Naive row
axTrE = [fig.add_subplot(gsT[1, k]) for k in range(4)]      #     Expert row
axMix = fig.add_subplot(gsT[0:2, 4])                        # E — full pairwise mixing (spans both rows)
gsF = gs[2, 0:12].subgridspec(2, ncol3, hspace=0.14, wspace=0.05)
axF = [fig.add_subplot(gsF[i // ncol3, i % ncol3]) for i in range(np3)]
gsL = gs[3, 0:12].subgridspec(1, 4, width_ratios=[1.1, 1.1, 0.5, 0.5], wspace=0.5)
axLf = fig.add_subplot(gsL[0]); axMf = fig.add_subplot(gsL[1])
axN1 = fig.add_subplot(gsL[2]); axN2 = fig.add_subplot(gsL[3])

schematic(axSch)
s1 = section1_evr(axEvr)
print(f'sec1: Expert top2 all {s1["Expert"][0]:.2%} / wm {s1["Expert"][1]:.2%} PR {s1["Expert"][2]:.2f}')
section1_contrast(axCon)
section2_traj(axTrN, BASE_N, 'Naive', titles=True, xlabel=False, legend=True)
section2_traj(axTrE, BASE, 'Expert', titles=False, xlabel=True, legend=False)
section2_mixing(axMix)                                   # full pairwise axis mixing, Naive vs Expert
for k in range(4):                                       # share y-scale per marginal so N vs E is comparable
    ylo = min(axTrN[k].get_ylim()[0], axTrE[k].get_ylim()[0])
    yhi = max(axTrN[k].get_ylim()[1], axTrE[k].get_ylim()[1])
    axTrN[k].set_ylim(ylo, yhi); axTrE[k].set_ylim(ylo, yhi)
section3(axF, ncol3)
section4(axLf, axMf, axN1, axN2)

# panel letters (none on the section-3 flow grid, per request)
plabel(axSch, 'A'); plabel(axEvr, 'B'); plabel(axCon, 'C')
plabel(axTrN[0], 'D')                                    # sec-2 trajectory grid (Naive/Expert × 4 marginals)
plabel(axMix, 'E')                                       # sec-2 full pairwise axis mixing
for ax, Lc in zip([axLf, axMf, axN1], ['F', 'G', 'H']): plabel(ax, Lc)
# shared flow-grid axis labels (one x, one y — not one per flow)
fb = np.array([[a.get_position().x0, a.get_position().y0, a.get_position().x1, a.get_position().y1] for a in axF])
fx0, fy0, fx1, fy1 = fb[:, 0].min(), fb[:, 1].min(), fb[:, 2].max(), fb[:, 3].max()
fig.text((fx0 + fx1) / 2, fy0 - 0.010, 'sample axis', ha='center', va='top', fontsize=9)
fig.text(fx0 - 0.028, (fy0 + fy1) / 2, 'choice axis  (+ = lick)', ha='center', va='center', rotation=90, fontsize=9)
# section headers (left-aligned, positioned from each section's top edge) + flow-fit equation
def _top(axs):
    return max(a.get_position().y1 for a in (axs if isinstance(axs, list) else [axs]))
def sechead(y, tx):
    fig.text(0.075, y, tx, ha='left', va='bottom', fontsize=10.5, fontweight='bold')
sechead(_top(axSch) + 0.024, '1.  Low-dimensional dPCA geometry & per-task variance')
sechead(_top(axTrN) + 0.012, '2.  dPCA-axis trajectories (Naive vs Expert) & axis mixing')
sechead(_top(axF) + 0.046, '3.  The computation: partial-pooled gain-modulated flows (sample × choice, pooled)')
fig.text(0.075, _top(axF) + 0.027, r'$\dot z = -z + S(z)\,(A_{\mathrm{sh}}\!+\!\Delta A_r)\,z + c_r,\ \ '
         r"S(z)=\langle\varphi'(\sqrt{a^2\|z\|^2+\delta}\,\xi)\rangle$"
         '   —  shared $A_{\\mathrm{sh}}$ per epoch (delay: sample wells / cue-test: choice wells) + ridge '
         '$\\Delta A_r$ + input $c_r$; $a,\\delta,\\lambda$ CV-tuned',
         ha='left', va='bottom', fontsize=8, color='0.25')
sechead(_top([axLf, axMf]) + 0.024, '4.  Learning pushes the sample memory down into no-lick')
leg = [Line2D([0], [0], ls='', marker='*', mfc='yellow', mec='k', ms=12, label='attractor'),
       Line2D([0], [0], ls='', marker='s', mfc='w', mec='k', ms=8, label='saddle'),
       Line2D([0], [0], color='c', ls='--', label='naive memory level'),
       Line2D([0], [0], color=SAMPLE_COL[0], lw=2, label='sample A'),
       Line2D([0], [0], color=SAMPLE_COL[1], lw=2, label='sample B')]
fig.legend(handles=leg, loc='lower center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.002))
fig.suptitle('The dPCA computation of the dual working-memory task', y=0.984, fontsize=13)

# ── match the section-4 flows to the section-3 panel BOX + add flow-speed colorbars (sec 3 & sec 4) ──
# The figure is portrait, so a figure-fraction square is NOT a pixel square. A section-3 panel spans
# (_w3 × _h3) in figure fraction, which is pixel-square; reuse those exact spans for the section-4 flows.
fig.canvas.draw()
_s3 = axF[0].get_position(); _w3, _h3 = _s3.width, _s3.height
_x0 = axLf.get_position().x0; _fy1 = axN1.get_position().y1; _g = 0.014; _y4 = _fy1 - _h3
axLf.set_position([_x0, _y4, _w3, _h3])
axMf.set_position([_x0 + _w3 + _g, _y4, _w3, _h3])

def _flow_cbar(rect, label='flow speed |ż|  (per panel)'):
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='magma'); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes(rect))
    if label:
        cb.set_label(label, fontsize=7.5)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['slow', 'fast']); cb.ax.tick_params(labelsize=7)
_flow_cbar([_x0 + 2 * (_w3 + _g), _y4, 0.011, _h3], label='')                 # section-4 (label-free; sec-3 carries it)
_r = max(a.get_position().x1 for a in axF)                                    # section-3 grid extent
_t = max(a.get_position().y1 for a in axF); _b = min(a.get_position().y0 for a in axF)
_flow_cbar([_r + 0.010, _b, 0.011, _t - _b])                                 # section-3 (right of the grid)
# stat panels J/K (no-lick push, sample separation): evenly spaced in the right region
_jy = axN1.get_position().y0; _jh = axN1.get_position().height
_jxs = _x0 + 2 * (_w3 + _g) + 0.105; _jgap = 0.10; _stats = [axN1, axN2]
_jw = (0.978 - _jxs - _jgap * (len(_stats) - 1)) / len(_stats)
for _k, _axj in enumerate(_stats):
    _axj.set_position([_jxs + _k * (_jw + _jgap), _jy, _jw, _jh])

OUT = 'figures/pseudo/story'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
out = f'{OUT}/png/fig_dpca_story_main{TAG}.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
plt.close(fig)
print('saved', os.path.abspath(out))
