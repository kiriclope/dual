"""OVERLAPS story composite — the CCGD analog of the dPCA story figure's §3 (flows) and §4 (push).

Two bands, both in the sample × choice CCGD-code plane (unlike dPCA, whose §4 is sample×tasks — in
overlaps the no-lick push lives on the *choice* code, = decoded lick action Hit+FA vs CR+Miss, the analog
of the dPCA `tasks` axis):

  §3 (row 1) — rank-2 gain-modulated flow-field grid, Expert: autonomous + input-driven regimes
               (A/B/Go/NoGo/Cue/C/D). Model ż_d = -z_d + S(z)·(A z)_d + c_r,  S(z)=<φ'(√Δ ξ)>,
               Δ = a²‖z‖²+δ. Fit with the dPCA §3 construction (see HONEST SCOPE).
  §4 (row 2) — the SAME plane's DPA well deepening with learning: Naive autonomous flow (well ≈ choice 0)
               | Expert autonomous flow (well pushed into the negative/no-lick half) | per-mouse
               choice-code depth slopegraph Naive→Expert (+ a sample-memory specificity control).
               Push glue ported from exp_nolick_push_stats.py.

HONEST SCOPE (settled in docs/overlaps/overview.md): fitted flow on the overlaps CCGD data is at the
velocity noise floor (held-out vel-R²≈0, ≈ linear). The rank-2 fit is DESCRIPTIVE, not predictive. To get
coherent, dPCA-like input fields we use the dPCA §3 construction — TWO-GROUP PARTIAL POOLING: one shared
bistable landscape A per epoch group (delay {autonomous,A,B} / choice {Go,NoGo,Cue,C,D}), + a ridge-shrunk
per-regime deviation ΔA_r, + a per-regime input current c_r, so each input rides the group's common
landscape (a single global A can't hold both bistabilities). Model selection is restricted to gains whose
shared autonomous flow stays bistable. Fixed points are root-found on the fitted field (★ attractor /
□ saddle / ✕ repeller), same as the dPCA figure. The figure says so.

Usage:
  cd /home/leon/dual/overlaps
  python fig_overlaps_story_main.py [--panels 8|4] [--all-trials] [--train delay|test|ld|wide]
Saves figures/overlaps/story/{png,svg}/fig_overlaps_story_main[_all].{png,svg}.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, argparse, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.common.options import set_options
from src.pca.io import pkl_load
from src.pca.dynamics import flow_fixed_points

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'

ap = argparse.ArgumentParser()
ap.add_argument('--panels', type=int, default=8, choices=[8, 4])
ap.add_argument('--all-trials', action='store_true', help='correct-only (default) -> all laser-off')
args = ap.parse_args()
CORRECT = not args.all_trials
TAG = '' if CORRECT else '_all'
MARGIN = 3                # calcium tail after event offset: 0.5 s @ 6 Hz — kept < MD (1 s = 6 bins) so a
#                           regime's window/decoder never bleeds into the next epoch (distractor↛cue etc.)
VSTEP = 1                 # adjacent-bin velocity (dPCA exact port)

# ── data / constants ──────────────────────────────────────────────────────────
DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA = '../data/overlaps'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
o = set_options(mice=MICE, tasks=['Dual'], mouse=MICE[0], laser=0, trials='', data_type='dF',
                prescreen=None, pval=0.05, preprocess=None, scaler_BL='standard_BL', avg_noise=False,
                unit_var_BL=False, random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
                scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10, class_weight=0,
                multilabel=0, mne_estimator='generalizing', n_jobs=64, days=['first', 'last'])
# T_WINDOW=0.0 = project-canonical epoch convention (shared_data.md): bins_DELAY=18:54, STIM=12:18,
# DIST=27:33, TEST=54:60 — same as exp_nolick_push_stats, so the flow plane and the push are ONE plane.
BL = slice(0, 12)
BINS_BL = o['bins_BL']
BINS_LATE = np.arange(27, 54)                         # canonical push window (exp_nolick_push_stats)
COL = {0: '#332288', 1: '#44AA99', 'DualGo': '#117733', 'DualNoGo': '#CC6677'}
KEY = ['mouse', 'day', 'tasks', 'sample_odor', 'test_odor', 'odor_pair', 'response', 'odr_perf']
TRAIN_PUSH = o['bins_DELAY']          # §4 depth axis (delay) — matches exp_nolick_push_stats

X = pkl_load(f'X_{DUM}', path=DATA)
y = pkl_load(f'labels_{DUM}', path=DATA)

NODES, WK = np.polynomial.hermite_e.hermegauss(20); WK = WK / np.sqrt(2 * np.pi)
def gd(D, h):
    t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * NODES[None, :] + h[:, None])
    return (WK * (1 - t ** 2)).sum(1)


def _win(bins):
    b = np.asarray(bins); return np.arange(b[0], min(b[-1] + 1 + MARGIN, 84))
# read window per input = its EVENT epoch + calcium MARGIN (GCaMP Ca2+ tail, 0.5 s @ 6 Hz).
WIN_STIM, WIN_DIST = _win(o['bins_STIM']), _win(o['bins_DIST'])
WIN_CUE, WIN_TEST = _win(o['bins_CUE']), _win(o['bins_TEST'])
# DECODER (train epoch) per regime — independent of the READ window (REG below). A/B decoded
# contemporaneously at stim, C/D at test; Go/NoGo/Cue decoded on the RESPONSE (lick) axis — the lick code
# is barely driven at the distractor/cue epochs, so Go↑/NoGo↓/Cue-split only read on the response decoder —
# but READ over their own distractor/cue window. autonomous keeps the delay decoder. Regimes sharing a
# decoder pool ONE landscape and each such group CV-tunes its OWN ridge λ (so C/D keep a low ridge → the
# bistable diagonal, independent of the other groups' optimum).
EPOCHS = {'DELAY': o['bins_DELAY'], 'STIM': WIN_STIM, 'TEST': WIN_TEST, 'RESP': o['bins_CHOICE']}
REG_EPOCH = {'autonomous': 'DELAY', 'A input': 'STIM', 'B input': 'STIM', 'Go input': 'RESP',
             'NoGo input': 'RESP', 'Cue input': 'RESP', 'C input': 'TEST', 'D input': 'TEST'}


def codes(stage, correct, train_bins):
    """Matched [sample, choice] plane at a train epoch for one stage, per-mouse BL-std normed, ×2.8."""
    base = (y.laser == 0) & (y.learning == stage)
    if correct:                                          # canonical overlaps correct: DPA-correct AND
        base = base & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1))  # GNG-correct on Dual

    df = X[..., train_bins, :].mean(-2)[:, 1].astype(float)
    for mo in MICE:
        mm = (y.mouse == mo).to_numpy(); sd = df[mm][:, BL].std()
        if sd > 0:
            df[mm] /= sd

    def block(tgt):
        mb = (base & (y.target == tgt)).to_numpy(); yb = y[mb].reset_index(drop=True)
        order = yb.sort_values(KEY, kind='stable').index.to_numpy()
        return df[mb][order], yb.iloc[order].reset_index(drop=True)
    ds, _ = block('sample'); dc, yc = block('choice')
    Z2 = np.stack([ds, dc], axis=1).astype(float); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
    return Z2, yc


def build_reg(yc):
    samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy(); task = yc['tasks'].to_numpy()
    go, nogo, dpa = task == 'DualGo', task == 'DualNoGo', task == 'DPA'
    return [('autonomous', dpa, np.arange(21, 54), ('sample', (0, 1))),
            ('A input', samp == 0, WIN_STIM, ('sample', (0,))),
            ('B input', samp == 1, WIN_STIM, ('sample', (1,))),
            ('Go input', go, WIN_DIST, ('sample', (0, 1))),
            ('NoGo input', nogo, WIN_DIST, ('sample', (0, 1))),
            ('Cue input', go | nogo, WIN_CUE, ('tasks', ('DualGo', 'DualNoGo'))),
            ('C input', test == 0, WIN_TEST, ('sample', (0, 1))),
            ('D input', test == 1, WIN_TEST, ('sample', (0, 1)))]
PANELS4 = [0, 1, 5, 6]                                # autonomous, A, Cue, C


def regime_means(REG, planes, yc, trm):
    out = {}
    for r, (nm, rm, w, (fac, levs)) in enumerate(REG):
        Z2 = planes[REG_EPOCH[nm]]                       # each regime read on its own decoder plane
        fv = yc[fac].to_numpy()
        out[r] = [(lv, Z2[rm & trm & (fv == lv)].mean(0)) for lv in levs
                  if (rm & trm & (fv == lv)).sum() >= 3]
    return out


def zv_one(REG, means, r):
    w = REG[r][2]; zs, vs = [], []; h = VSTEP // 2
    for _, mu in means[r]:
        M = mu[:, w]
        if VSTEP <= 1:
            zs.append(M[:, :-1].T); vs.append(np.diff(M, axis=1).T)
        elif M.shape[1] > 2 * h:
            zs.append(M[:, h:-h].T); vs.append(((M[:, 2 * h:] - M[:, :-2 * h]) / (2 * h)).T)
    return (np.concatenate(zs), np.concatenate(vs)) if zs else (np.empty((0, 2)), np.empty((0, 2)))


def flow_indep(A, c, a, dd):
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd; S = gd(D, np.zeros(P.shape[1])); AP = A @ P
        return np.vstack([-P[0] + S * AP[0] + c[0], -P[1] + S * AP[1] + c[1]])
    return fl


# ── PARTIAL POOLING by decoder epoch (dPCA §3 method) ──────────────────────────
# Regimes sharing a decoder (same plane) pool ONE landscape: shared A + ridge-penalised per-regime ΔA_r +
# per-regime input current c_r. Groups: {autonomous}@DELAY, {A,B}@STIM, {Go,NoGo,Cue}@RESP, {C,D}@TEST.
def groups_of(REG):
    g = {}
    for r in range(len(REG)):
        g.setdefault(REG_EPOCH[REG[r][0]], []).append(r)
    return list(g.values())


def fit_group(REG, means, a, dd, lam, subset):
    Z_, V_, R_ = [], [], []
    for loc, r in enumerate(subset):
        z, v = zv_one(REG, means, r)
        if len(z):
            Z_.append(z); V_.append(v); R_.append(np.full(len(z), loc))
    if not Z_:
        return {}
    z = np.concatenate(Z_); v = np.concatenate(V_); rid = np.concatenate(R_).astype(int); ng = len(subset)
    D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
    OH = np.eye(ng)[rid]; shF = np.column_stack([S * z[:, 0], S * z[:, 1]])
    devF = (OH[:, :, None] * shF[:, None, :]).reshape(len(z), ng * 2)                  # per-regime S·z
    F = np.column_stack([shF, devF, OH]); Pn = F.shape[1]
    reg = np.zeros((Pn, Pn)); reg[2:2 + 2 * ng, 2:2 + 2 * ng] = lam * np.eye(2 * ng)   # ridge ΔA only
    FtF = F.T @ F + reg; A_sh = np.zeros((2, 2)); dA = np.zeros((ng, 2, 2)); C = np.zeros((ng, 2))
    for d in (0, 1):
        cd = np.linalg.solve(FtF, F.T @ (v[:, d] + z[:, d])); A_sh[d] = cd[0:2]
        for loc in range(ng):
            dA[loc, d] = cd[2 + 2 * loc:2 + 2 * loc + 2]
        C[:, d] = cd[2 + 2 * ng:]
    return {r: flow_indep(A_sh + dA[loc], C[loc], a, dd) for loc, r in enumerate(subset)}


def fit_all(REG, means, a, dd, lam):
    flows = {}
    for subset in groups_of(REG):
        flows.update(fit_group(REG, means, a, dd, lam, subset))
    return flows


GRID = [(a, dd, lam) for a in (0.2, 0.4, 0.7, 1.0) for dd in (0.3, 0.8, 2.0)
        for lam in (0.2, 1.0, 5.0, 20.0, 100.0)]


def fit_stage(stage, correct):
    planes = {}; yc = None
    for en in EPOCHS:
        Z2, yc = codes(stage, correct, EPOCHS[en]); planes[en] = Z2   # one plane per decoder epoch
    REG = build_reg(yc); NREG = len(REG)
    allm = regime_means(REG, planes, yc, np.ones(len(yc), bool))
    span = np.concatenate([mu[:, REG[r][2]] for r in range(NREG) for _, mu in allm[r]], axis=1)
    L = 1.3 * max(float(np.abs(span).max()), 1.0)
    folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(yc))))

    # PER-GROUP ridge: each decoder group CV-tunes its OWN (a,δ,λ) so C/D keep a low ridge (bistable
    # diagonal) while the response group keeps Go↑/NoGo↓. Score = held-out velocity R² on that group.
    def cv_group(subset, a, dd, lam):
        rr = []
        for tr, te in folds:
            trm = np.zeros(len(yc), bool); trm[tr] = True; tem = np.zeros(len(yc), bool); tem[te] = True
            fl = fit_group(REG, regime_means(REG, planes, yc, trm), a, dd, lam, subset)
            me = regime_means(REG, planes, yc, tem); num = den = 0.0
            for r in subset:
                z, v = zv_one(REG, me, r)
                if len(z) >= 3:
                    vp = fl[r](z.T).T; num += ((v - vp) ** 2).sum(); den += ((v - v.mean(0)) ** 2).sum()
            rr.append(1 - num / (den + 1e-12))
        return float(np.mean(rr))

    def auto_bistable(subset, p):                     # keep the WM autonomous landscape bistable
        fl = fit_group(REG, allm, *p, subset)
        return sum(k == 'attractor' for _, k, _ in flow_fixed_points(fl[0], [(-L, L), (-L, L)], n_seed=18)) >= 2

    flows = {}; cvbest = []
    for g in groups_of(REG):
        cvs = {p: cv_group(g, *p) for p in GRID}
        if 0 in g:                                    # autonomous group → restrict to bistable configs
            bist = [p for p in GRID if auto_bistable(g, p)]
            best = max(bist or GRID, key=lambda p: cvs[p]); tag = f'  ({len(bist)}/{len(GRID)} bist)'
        else:
            best = max(GRID, key=lambda p: cvs[p]); tag = ''
        flows.update(fit_group(REG, allm, *best, g)); cvbest.append(cvs[best])
        print(f'  {stage:6s} [{"/".join(REG[r][0].split()[0] for r in g):11s}] '
              f'(a,δ,λ)={best}  vel-R²={cvs[best]:+.3f}{tag}')
    return dict(REG=REG, allm=allm, flows=flows, cv=float(np.mean(cvbest)), L=L)


def panel_lim(allm, idxs):
    arr = np.concatenate([mu[:, REG[r][2]] for r in idxs for _, mu in allm[r]], axis=1)
    return 1.3 * max(np.abs(arr).max(), 1.0)


def draw_panel(ax, st, r, LIM, baseline=False):
    REG, allm, flows = st['REG'], st['allm'], st['flows']
    w = REG[r][2]
    gl = np.linspace(-LIM, LIM, 60); Xg, Yg = np.meshgrid(gl, gl); P = np.vstack([Xg.ravel(), Yg.ravel()])
    F = flows[r](P); U, V = F[0].reshape(Xg.shape), F[1].reshape(Xg.shape)
    ax.pcolormesh(Xg, Yg, np.hypot(U, V), cmap='magma', shading='auto')
    ax.streamplot(Xg, Yg, U, V, color='w', density=0.9, linewidth=0.5, arrowsize=0.7)
    if baseline:
        ax.axhline(0, color='c', lw=1.1, ls='--', zorder=4)     # lick baseline (choice=0)
    for lv, mu in allm[r]:
        col = COL.get(lv, 'c')
        ax.plot(mu[0, w], mu[1, w], '-', color=col, lw=2.0, zorder=5)
        ax.plot(mu[0, w][0], mu[1, w][0], 'o', color=col, ms=5, mfc='w', zorder=6)       # start
        ax.plot(mu[0, w][-1], mu[1, w][-1], '*', color=col, ms=12, zorder=6)             # end
    for pt, kind, _ in flow_fixed_points(flows[r], [(-LIM, LIM), (-LIM, LIM)], n_seed=18):
        mk = {'attractor': ('*', 'yellow', 13), 'saddle': ('s', 'w', 8),
              'repeller': ('X', 'r', 9)}.get(kind, ('*', 'y', 10))
        ax.plot(pt[0], pt[1], mk[0], mfc=mk[1], mec='k', ms=mk[2], zorder=7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


# ── push quantification (ported from exp_nolick_push_stats.py) ─────────────────
STAGES = ['Naive', 'Expert']
SAMPLE_CLASSES = [('A', [0, 1]), ('B', [2, 3])]
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def choice_axis_ep(bins_train):
    X_ep = X[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mo in MICE:
        m = (y.mouse == mo).to_numpy(); sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd
    return X_ep


def depths(bins_train, trial_mask, target='choice', cond='DPA'):
    X_ep = choice_axis_ep(bins_train); out = {}
    for mo in MICE:
        for stage in STAGES:
            for cls, pairs in SAMPLE_CLASSES:
                m = ((y.mouse == mo) & (y.tasks == cond) & (y.stage == stage) &
                     (y.target == target) & trial_mask & y.odor_pair.isin(pairs)).to_numpy()
                out[(mo, stage, cls)] = X_ep[m][:, BINS_LATE].mean() if m.sum() else np.nan
    return out


def per_mouse(dep, stage, cls):
    return np.array([dep[(m, stage, cls)] for m in MICE], float)


def boot_ci(vals, func=np.mean, alpha=0.05):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v))); stat = func(v[idx], axis=1)
    return np.percentile(stat, 100 * alpha / 2), np.percentile(stat, 100 * (1 - alpha / 2))


def cohen_dz(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan


def slopegraph(ax, nai, exp, color, ylabel, title, side='less'):
    keep = ~(np.isnan(nai) | np.isnan(exp)); a, b = nai[keep], exp[keep]; d = b - a
    for n, e in zip(a, b):
        ax.plot([0, 1], [n, e], '-', color=color, lw=1, alpha=0.4)
        ax.scatter([0, 1], [n, e], color=color, s=26, zorder=5, edgecolors='w', linewidths=0.5)
    for x, v in [(0, a), (1, b)]:
        lo, hi = boot_ci(v)
        ax.errorbar(x + 0.07, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                    fmt='o', color='k', ms=7, capsize=4, lw=1.6, zorder=6)
    try:
        p2 = wilcoxon(d).pvalue; p1 = wilcoxon(d, alternative=side).pvalue
    except ValueError:
        p2 = p1 = np.nan
    n_dir = int((d < 0).sum()) if side == 'less' else int((d > 0).sum())
    lo, hi = boot_ci(d)
    ax.axhline(0, ls='--', color='k', lw=0.8)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=8)
    ax.set_title(f'{title}\nΔ={d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]\n'
                 f'p={p2:.3f} (1s {p1:.3f})  dz={cohen_dz(d):+.2f}  {n_dir}/{len(d)}', fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(labelsize=7)
    return d


# ── build ──────────────────────────────────────────────────────────────────────
print(f'=== OVERLAPS story figure  (per-regime decoders, per-group ridge; Go/NoGo/Cue on response; '
      f'{"correct" if CORRECT else "all-laser-off"}, panels={args.panels}) ===')
ST = {s: fit_stage(s, CORRECT) for s in STAGES}
REG = ST['Expert']['REG']
idx3 = list(range(8)) if args.panels == 8 else PANELS4

# push arrays (delay axis, correct/all, DPA)
mask = (y.laser == 0) if not CORRECT else (
    (y.laser == 0) & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1)))
dep_ch = depths(TRAIN_PUSH, mask, target='choice')
pool_nai = np.nanmean([per_mouse(dep_ch, 'Naive', 'A'), per_mouse(dep_ch, 'Naive', 'B')], axis=0)
pool_exp = np.nanmean([per_mouse(dep_ch, 'Expert', 'A'), per_mouse(dep_ch, 'Expert', 'B')], axis=0)
dep_sa = depths(TRAIN_PUSH, mask, target='sample')     # specificity control: sample-memory separation
sep_nai = per_mouse(dep_sa, 'Naive', 'B') - per_mouse(dep_sa, 'Naive', 'A')
sep_exp = per_mouse(dep_sa, 'Expert', 'B') - per_mouse(dep_sa, 'Expert', 'A')

# ── figure ──────────────────────────────────────────────────────────────────────
FH = 15.0 if args.panels == 8 else 9.0
fig = plt.figure(figsize=(19, FH))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0 if args.panels == 8 else 1.0, 1.0], hspace=0.30)

# Row 1 — §3 Expert flow grid
nc3 = 4; nr3 = (len(idx3) + nc3 - 1) // nc3
gsF = gs[0].subgridspec(nr3, nc3, hspace=0.32, wspace=0.12)
LIM3 = ST['Expert']['L']
axF = []
for k, r in enumerate(idx3):
    ax = fig.add_subplot(gsF[k // nc3, k % nc3]); axF.append(ax)
    draw_panel(ax, ST['Expert'], r, LIM3, baseline=(r == 0))
    ax.set_title(f'{REG[r][0]}   [{REG_EPOCH[REG[r][0]]} dec.]', fontsize=9.5)
    if k % nc3 == 0:
        ax.set_ylabel('choice code', fontsize=9)
    if k // nc3 == nr3 - 1:
        ax.set_xlabel('sample code', fontsize=9)

# Row 2 — §4 push
gsL = gs[1].subgridspec(1, 4, width_ratios=[1.1, 1.1, 0.62, 0.62], wspace=0.32)
axNai = fig.add_subplot(gsL[0]); axExp = fig.add_subplot(gsL[1])
axN1 = fig.add_subplot(gsL[2]); axN2 = fig.add_subplot(gsL[3])
LIM4 = max(panel_lim(ST['Naive']['allm'], [0]), panel_lim(ST['Expert']['allm'], [0]))
for ax, s in [(axNai, 'Naive'), (axExp, 'Expert')]:
    draw_panel(ax, ST[s], 0, LIM4, baseline=True)
    ax.set_title(f'{s}: DPA autonomous flow', fontsize=10)
    ax.set_xlabel('sample code', fontsize=9)
axNai.set_ylabel('choice code', fontsize=9)
slopegraph(axN1, pool_nai, pool_exp, '#444444',
           f'choice-code depth\n(late delay, {"correct" if CORRECT else "all"})',
           'push: choice code', side='less')
slopegraph(axN2, sep_nai, sep_exp, '#888844', 'sample memory |B−A|',
           'control: sample sep.', side='greater')

# panel letters + legend + caption
for ax, L in [(axF[0], 'A'), (axNai, 'B'), (axExp, 'C'), (axN1, 'D'), (axN2, 'E')]:
    ax.text(-0.02, 1.06, L, transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom', ha='right')
fp_leg = [Line2D([0], [0], ls='', marker='*', mfc='yellow', mec='k', ms=12, label='attractor'),
          Line2D([0], [0], ls='', marker='s', mfc='w', mec='k', ms=8, label='saddle'),
          Line2D([0], [0], ls='', marker='X', mfc='r', mec='k', ms=9, label='repeller'),
          Line2D([0], [0], color='c', ls='--', lw=1.1, label='lick baseline (choice=0)')]
tr_leg = [Line2D([0], [0], color=COL[0], lw=2.3, label='sample A'),
          Line2D([0], [0], color=COL[1], lw=2.3, label='sample B'),
          Line2D([0], [0], color=COL['DualGo'], lw=2.3, label='Go'),
          Line2D([0], [0], color=COL['DualNoGo'], lw=2.3, label='NoGo'),
          Line2D([0], [0], ls='', marker='o', mfc='w', mec='k', ms=6, label='traj. start'),
          Line2D([0], [0], ls='', marker='*', mfc='0.4', mec='k', ms=11, label='traj. end')]
fig.legend(handles=fp_leg + tr_leg, loc='lower center', ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.015))
cvE, cvN = ST['Expert']['cv'], ST['Naive']['cv']
fig.suptitle('OVERLAPS story — rank-2 gain-modulated flows on the sample×choice CCGD plane   '
             f'[per-regime decoders, per-group ridge; Go/NoGo/Cue on response; {"correct" if CORRECT else "all-laser-off"}]\n'
             '§3 (row 1) computation, Expert: each input decoded on the axis that carries it (A/B stim, C/D '
             'test, Go/NoGo/Cue on the RESPONSE/lick axis), read over its event+GCaMP-margin window; per-decoder '
             'shared landscape + ridge ΔA_r + input c_r → autonomous bistable, Go↑, NoGo↓, C/D split the '
             'diagonals.   §4 (row 2) learning: DPA delay well deepens Naive→Expert into the no-lick half.\n'
             f'DESCRIPTIVE fit — per-group held-out CV vel-R² (mean) = {cvE:+.2f} (Expert) / {cvN:+.2f} (Naive); '
             'fixed points root-found on the fitted field (★ attractor, □ saddle, ✕ repeller).',
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))

# ── flow-speed colorbars (qualitative slow→fast; each panel is per-panel normalised, like the dPCA fig) ──
fig.canvas.draw()
def _flow_cbar(rect, label='flow speed |ż|'):
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='magma'); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes(rect))
    cb.set_ticks([0, 1]); cb.set_ticklabels(['slow', 'fast']); cb.ax.tick_params(labelsize=7)
    if label:
        cb.set_label(label, fontsize=8)
_r = max(a.get_position().x1 for a in axF)                                    # §3 grid extent
_t = max(a.get_position().y1 for a in axF); _b = min(a.get_position().y0 for a in axF)
_flow_cbar([_r + 0.007, _b, 0.007, _t - _b])                                 # §3 (right of the grid)
_pe = axExp.get_position()
_flow_cbar([_pe.x1 + 0.008, _pe.y0, 0.007, _pe.height], label='')            # §4 (right of Expert flow)

OUT = 'figures/overlaps/story'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
stem = f'fig_overlaps_story_main{TAG}' + ('' if args.panels == 8 else '_p4')
fig.savefig(f'{OUT}/png/{stem}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/svg/{stem}.svg', bbox_inches='tight')
print(f'\n§4 push (choice, delay, {"correct" if CORRECT else "all"}):  '
      f'Naive={np.nanmean(pool_nai):+.3f}  Expert={np.nanmean(pool_exp):+.3f}  '
      f'Δ={np.nanmean(pool_exp - pool_nai):+.3f}')
print(f'saved {OUT}/png/{stem}.png')
