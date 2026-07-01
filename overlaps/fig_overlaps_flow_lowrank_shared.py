"""Rank-2 reduced flows on the OVERLAPS (CCGD) sample×choice plane — PORTED VERBATIM from
pca/fig_dpca_flow_lowrank_shared.py (same gain-modulated linear model, regimes, CV, plotting),
only the data section is swapped: instead of dPCA latents, the plane is built from the CCGD
decision-function codes (x = sample code, y = choice code, trainTEST-averaged, per-mouse BL-std
normalised). The two codes live in different y['target'] blocks (same physical trials, reordered);
they are matched by a stable label-key sort to reconstruct a per-trial [sample, choice] vector
(exact for all condition means — verified row-identical on sample/test/tasks/choice).

Model: ż_d = -z_d + S(z)·(A z)_d + c_r,  S(z)=<φ'(√Δ ξ)>,  Δ = a²‖z‖²+δ  (gain-modulated LINEAR).
Modes: partial (shared A + ridge ΔA_r) / independent (per-regime A_r) / shared (one A + currents).
(a,δ[,λ]) CV-tuned on pooled held-out condition-mean velocity R². 8-panel grid (autonomous + inputs).
SINGLE train axis for the whole figure (`--train`, default delay). Only the per-regime READ WINDOW
varies = [odor onset, odor offset + calcium MARGIN] (autonomous = delay maintenance, no odor).
Usage: fig_overlaps_flow_lowrank_shared.py [--shared|--independent|--partial] [--stage Expert|Naive]
       [--train delay|test|ld|wide|diag] [--anchor W] [--margin M] [--smooth σ] [--vstep K]
`diag` = the generalization-matrix diagonal (train-time==test-time per bin) → each regime read on its
contemporaneous decoder; best fits (pooled CV +0.17) but the axis rotates per bin (looser flow)."""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.common.options import set_options
from src.pca.io import pkl_load
from src.pca.dynamics import flow_fixed_points

MODE = ('shared' if '--shared' in sys.argv else 'partial' if '--partial' in sys.argv else 'independent')
STAGE = sys.argv[sys.argv.index('--stage') + 1] if '--stage' in sys.argv else 'Expert'
SMOOTH = float(sys.argv[sys.argv.index('--smooth') + 1]) if '--smooth' in sys.argv else 0.0
# Velocity estimate: K-bin CENTRED difference (denoises the slow CCGD condition-mean velocity, like
# the empirical-flow VEL_STEP=5). K=1 = the dPCA adjacent-bin np.diff (default, exact port).
VSTEP = int(sys.argv[sys.argv.index('--vstep') + 1]) if '--vstep' in sys.argv else 1
# Train epoch for the code plane. DEFAULT 'delay': we draw the DELAY flow, so read the codes on the
# DELAY-trained axis — CV-honest, smoother (jag 0.045 vs 0.064), better A/B separation, and it moves B
# from +0.46 (buried near the saddle on the TEST axis) to +1.25 (a real well) → faithful bistability.
TRAINSEL = sys.argv[sys.argv.index('--train') + 1] if '--train' in sys.argv else 'delay'
# Endpoint anchoring weight: pin the fitted field's fixed points onto the condition-mean trajectory
# endpoints (v=0 anchors in the LS) so streamlines converge to where the data settles (INDEPENDENT mode).
ANCHOR = float(sys.argv[sys.argv.index('--anchor') + 1]) if '--anchor' in sys.argv else 8.0
# Input-driven regimes span [odor ONSET, odor OFFSET + MARGIN]; the margin captures the slow calcium
# tail after the odor turns off (GCaMP decay ~1.5–2 s). Default 12 bins ≈ 2 s at 6 Hz.
MARGIN = int(sys.argv[sys.argv.index('--margin') + 1]) if '--margin' in sys.argv else 12
# Fit objective: 'vel' (default) = closed-form LS on the condition-mean velocity; 'traj' = integrate the
# flow from the trajectory start and minimise POSITION error (least_squares). Velocity R² scores the
# noise-amplified derivative; the trajectory objective/metric scores the observable (position) and is
# validated by held-out trajectory R² vs a LINEAR baseline (a=0). Independent mode only.
FIT = sys.argv[sys.argv.index('--fit') + 1] if '--fit' in sys.argv else 'vel'

# ── data: matched [sample, choice] plane from the CCGD codes (replaces dPCA latents) ──
DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
o = set_options(mice=MICE, tasks=['Dual'], mouse=MICE[0], laser=0, days=['first', 'last'], mne_estimator='generalizing')
BL = slice(0, 12)
X = pkl_load(f'X_{DUM}', path='../data/overlaps'); y = pkl_load(f'labels_{DUM}', path='../data/overlaps')
base = (y.laser == 0) & (y.learning == STAGE) & (y.performance == 1)
KEY = ['mouse', 'day', 'tasks', 'sample_odor', 'test_odor', 'odor_pair', 'response', 'odr_perf']
def codes(train_bins):                                          # matched [sample,choice] plane at a train epoch
    if isinstance(train_bins, str) and train_bins == 'diag':    # diagonal: train-time == test-time per bin
        df = np.diagonal(X[:, 1].astype(float), axis1=1, axis2=2).copy()   # (n,84); axis rotates per bin
    else:
        df = X[..., train_bins, :].mean(-2)[:, 1].astype(float)
    for mo in MICE:
        mm = (y.mouse == mo).to_numpy(); sd = df[mm][:, BL].std()
        if sd > 0:
            df[mm] /= sd
    def block(tgt):
        mb = (base & (y.target == tgt)).to_numpy(); yb = y[mb].reset_index(drop=True)
        order = yb.sort_values(KEY, kind='stable').index.to_numpy()
        return df[mb][order], yb.iloc[order].reset_index(drop=True)
    ds, ys = block('sample'); dc, yc = block('choice')
    Z2 = np.stack([ds, dc], axis=1).astype(float); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
    if SMOOTH > 0:
        from scipy.ndimage import gaussian_filter1d
        Z2 = gaussian_filter1d(Z2, SMOOTH, axis=2)
    return Z2, yc
def _win(bins):                                                  # [odor onset, odor offset + MARGIN]
    b = np.asarray(bins); return np.arange(b[0], min(b[-1] + 1 + MARGIN, 84))
WIN_STIM, WIN_DIST, WIN_TEST = _win(o['bins_STIM']), _win(o['bins_DIST']), _win(o['bins_TEST'])
# ONE train axis for the whole figure (default DELAY — draws the delay flow, faithful autonomous).
# Only the per-regime READ WINDOW varies = the input presentation + calcium MARGIN (autonomous = delay).
TRAIN = 'diag' if TRAINSEL == 'diag' else {'delay': o['bins_DELAY'], 'test': o['bins_TEST'],
        'ld': o['bins_LD'], 'wide': np.arange(18, 63)}[TRAINSEL]
Z2, yc = codes(TRAIN)
samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy(); task = yc['tasks'].to_numpy(); ch = yc['choice'].to_numpy()
go, nogo, dpa = task == 'DualGo', task == 'DualNoGo', task == 'DPA'
OUTDIR = 'figures/overlaps/flow_lowrank_traj' if FIT == 'traj' else 'figures/overlaps/flow_lowrank_shared'
os.makedirs(f'{OUTDIR}/png', exist_ok=True)
os.makedirs(f'{OUTDIR}/svg', exist_ok=True)

# ── model + CV + plotting (dPCA script verbatim) — single train axis, per-regime input windows ──
# autonomous is NOT input-driven → delay maintenance window; inputs = [odor onset, offset + margin].
REG = [('autonomous', dpa, np.arange(21, 54), ('sample', (0, 1))),
       ('A input', samp == 0, WIN_STIM, ('sample', (0,))),
       ('B input', samp == 1, WIN_STIM, ('sample', (1,))),
       ('Go input', go, WIN_DIST, ('sample', (0, 1))),
       ('NoGo input', nogo, WIN_DIST, ('sample', (0, 1))),
       ('Cue input', go | nogo, WIN_DIST, ('tasks', ('DualGo', 'DualNoGo'))),
       ('C input', test == 0, WIN_TEST, ('sample', (0, 1))),
       ('D input', test == 1, WIN_TEST, ('sample', (0, 1)))]
NREG = len(REG); COL = {0: '#332288', 1: '#44AA99', 'DualGo': '#117733', 'DualNoGo': '#CC6677'}
NODES, WK = np.polynomial.hermite_e.hermegauss(20); WK = WK / np.sqrt(2 * np.pi)
def gd(D, h): t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * NODES[None, :] + h[:, None]); return (WK * (1 - t ** 2)).sum(1)


def regime_means(trm):
    out = {}
    for r, (nm, rm, w, (fac, levs)) in enumerate(REG):
        fv = yc[fac].to_numpy()
        out[r] = [(lv, Z2[rm & trm & (fv == lv)].mean(0)) for lv in levs if (rm & trm & (fv == lv)).sum() >= 3]
    return out
def zv_one(means, r):
    w = REG[r][2]; zs, vs = [], []; h = VSTEP // 2
    for _, mu in means[r]:
        M = mu[:, w]                                     # (2, |w|)
        if VSTEP <= 1:
            zs.append(M[:, :-1].T); vs.append(np.diff(M, axis=1).T)
        elif M.shape[1] > 2 * h:                         # centred K-bin velocity at position M[:, c]
            zs.append(M[:, h:-h].T); vs.append(((M[:, 2 * h:] - M[:, :-2 * h]) / (2 * h)).T)
    return (np.concatenate(zs), np.concatenate(vs)) if zs else (np.empty((0, 2)), np.empty((0, 2)))
def stack(means):
    Z_, V_, R_ = [], [], []
    for r in range(NREG):
        z, v = zv_one(means, r)
        if len(z): Z_.append(z); V_.append(v); R_.append(np.full(len(z), r))
    return np.concatenate(Z_), np.concatenate(V_), np.concatenate(R_).astype(int)


# ---- shared model: one A + per-regime current (additive c_r, in-gain h_r) ----
def fit_shared(z, v, rid, a, dd, H):
    D = a ** 2 * (z ** 2).sum(1) + dd; OH = np.eye(NREG)[rid]; g0 = gd(D, H[rid, 0]); g1 = gd(D, H[rid, 1])
    F0 = np.column_stack([g0 * z[:, 0], g0 * z[:, 1], OH]); F1 = np.column_stack([g1 * z[:, 0], g1 * z[:, 1], OH])
    c0 = np.linalg.lstsq(F0, v[:, 0] + z[:, 0], rcond=None)[0]; c1 = np.linalg.lstsq(F1, v[:, 1] + z[:, 1], rcond=None)[0]
    return np.array([[c0[0], c0[1]], [c1[0], c1[1]]]), np.column_stack([c0[2:], c1[2:]])
def flow_shared(A, c, a, dd, h):
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd; AP = A @ P
        return np.vstack([-P[0] + gd(D, np.full(P.shape[1], h[0])) * AP[0] + c[0],
                          -P[1] + gd(D, np.full(P.shape[1], h[1])) * AP[1] + c[1]])
    return fl
def fit_all_shared(z, v, rid, a, dd, n_iter=4):
    H = np.zeros((NREG, 2))
    for _ in range(n_iter):
        A, C = fit_shared(z, v, rid, a, dd, H)
        for r in range(1, NREG):
            mk = rid == r
            if mk.sum():
                H[r] = least_squares(lambda h, A=A, C=C: (v[mk] - flow_shared(A, C[r], a, dd, h)(z[mk].T).T).ravel(), H[r], max_nfev=150).x
    A, C = fit_shared(z, v, rid, a, dd, H)
    return {r: flow_shared(A, C[r], a, dd, H[r]) for r in range(NREG)}, (A, C, H)


# ---- independent model: per-regime A_r + c_r (isotropic gain), optional endpoint anchoring ----
def fit_indep_one(z, v, a, dd, anch=None, wa=0.0):
    if len(z) < 3:
        return np.zeros((2, 2)), np.zeros(2)
    if anch is not None and wa > 0 and len(anch):
        z = np.vstack([z] + [e[None] for e in anch]); v = np.vstack([v] + [np.zeros(2)] * len(anch))
        ww = np.concatenate([np.ones(len(z) - len(anch)), wa * np.ones(len(anch))])   # anchors: v=0 at endpoints
    else:
        ww = np.ones(len(z))
    D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z))); Wt = np.sqrt(ww)[:, None]
    F = np.column_stack([S * z[:, 0], S * z[:, 1], np.ones(len(z))]) * Wt
    c0 = np.linalg.lstsq(F, (v[:, 0] + z[:, 0]) * Wt[:, 0], rcond=None)[0]
    c1 = np.linalg.lstsq(F, (v[:, 1] + z[:, 1]) * Wt[:, 0], rcond=None)[0]
    return np.array([[c0[0], c0[1]], [c1[0], c1[1]]]), np.array([c0[2], c1[2]])
def flow_indep(A, c, a, dd):
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd; S = gd(D, np.zeros(P.shape[1])); AP = A @ P
        return np.vstack([-P[0] + S * AP[0] + c[0], -P[1] + S * AP[1] + c[1]])
    return fl
def sim(flow_fn, z0, n):                                             # Euler-integrate (dt=1 bin) with a guard
    z = np.asarray(z0, float).copy(); out = [z.copy()]
    for _ in range(n - 1):
        z = np.clip(z + flow_fn(z[:, None]).ravel(), -30, 30); out.append(z.copy())
    return np.array(out).T                                          # (2, n)
def fit_indep_traj_one(means_r, w, a, dd, anch, wa):
    # fit A,c so the flow INTEGRATED from each condition-mean start reproduces its trajectory (position).
    tgt = [mu[:, w] for _, mu in means_r if mu[:, w].shape[1] >= 3]
    if not tgt:
        return np.zeros((2, 2)), np.zeros(2)
    zs = np.concatenate([t[:, :-1].T for t in tgt]); vs = np.concatenate([np.diff(t, 1).T for t in tgt])
    A0, c0 = fit_indep_one(zs, vs, a, dd, anch, wa)                 # velocity-fit init
    def resid(p):
        A = p[:4].reshape(2, 2); c = p[4:]; fl = flow_indep(A, c, a, dd); r = []
        for t in tgt:
            r.append((sim(fl, t[:, 0], t.shape[1]) - t).ravel())
        for e in (anch if wa > 0 else []):
            r.append(np.sqrt(wa) * fl(np.asarray(e, float)[:, None]).ravel())   # v=0 at endpoint
        return np.concatenate(r)
    sol = least_squares(resid, np.concatenate([A0.ravel(), c0]), max_nfev=150)
    return sol.x[:4].reshape(2, 2), sol.x[4:]
def fit_all_indep(means, a, dd, wa=0.0):
    flows = {}
    for r in range(NREG):
        anch = [mu[:, REG[r][2]][:, -5:].mean(1) for _, mu in means[r]]   # settled endpoints
        wr = wa if r == 0 else 0.0                                        # anchor the autonomous only
        if FIT == 'traj':
            A, c = fit_indep_traj_one(means[r], REG[r][2], a, dd, anch, wr)
        else:
            z, v = zv_one(means, r); A, c = fit_indep_one(z, v, a, dd, anch, wr)
        flows[r] = flow_indep(A, c, a, dd)
    return flows, None
def traj_r2_one(flow_fn, mu, w):                                    # in-sample trajectory R² for one cond
    t = mu[:, w]
    if t.shape[1] < 3: return np.nan
    zsim = sim(flow_fn, t[:, 0], t.shape[1])
    return 1 - ((zsim - t) ** 2).sum() / (((t - t.mean(1, keepdims=True)) ** 2).sum() + 1e-12)
def traj_r2(flows, means, r):
    vals = [traj_r2_one(flows[r], mu, REG[r][2]) for _, mu in means[r]]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


# ---- partial pooling: shared A + ridge-penalized per-regime deviation ΔA_r (+ per-regime c) ----
def fit_all_partial(means, a, dd, lam):
    z, v, rid = stack(means); D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
    OH = np.eye(NREG)[rid]; shF = np.column_stack([S * z[:, 0], S * z[:, 1]])
    devF = (OH[:, :, None] * shF[:, None, :]).reshape(len(z), NREG * 2)
    F = np.column_stack([shF, devF, OH]); Pn = F.shape[1]
    reg = np.zeros((Pn, Pn)); reg[2:2 + 2 * NREG, 2:2 + 2 * NREG] = lam * np.eye(2 * NREG)
    FtF = F.T @ F + reg; A_sh = np.zeros((2, 2)); dA = np.zeros((NREG, 2, 2)); C = np.zeros((NREG, 2))
    for d in (0, 1):
        cd = np.linalg.solve(FtF, F.T @ (v[:, d] + z[:, d])); A_sh[d] = cd[0:2]
        for r in range(NREG):
            dA[r, d] = cd[2 + 2 * r:2 + 2 * r + 2]
        C[:, d] = cd[2 + 2 * NREG:]
    return {r: flow_indep(A_sh + dA[r], C[r], a, dd) for r in range(NREG)}, (A_sh, dA, C)


def fit_mode(means, params, wa=0.0):
    # wa = endpoint-anchoring weight (independent mode only); CV passes wa=0, the display uses ANCHOR.
    if MODE == 'shared':
        z, v, rid = stack(means); return fit_all_shared(z, v, rid, *params)
    if MODE == 'partial':
        return fit_all_partial(means, *params)
    return fit_all_indep(means, *params, wa=wa)
def vr2(flows, means, r):
    z, v = zv_one(means, r)
    if len(z) < 3: return np.nan
    vp = flows[r](z.T).T; return 1 - ((v - vp) ** 2).sum() / (((v - v.mean(0)) ** 2).sum() + 1e-12)


folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(yc))))
def _cv_perreg(params):                                            # held-out {vel|traj} SSE/var per regime
    num = np.zeros(NREG); den = np.zeros(NREG)
    for tr, te in folds:
        trm = np.zeros(len(yc), bool); trm[tr] = True; tem = np.zeros(len(yc), bool); tem[te] = True
        fl, _ = fit_mode(regime_means(trm), params); me = regime_means(tem)
        for r in range(NREG):
            if FIT == 'traj':
                for _, mu in me[r]:
                    t = mu[:, REG[r][2]]
                    if t.shape[1] >= 3:
                        zsim = sim(fl[r], t[:, 0], t.shape[1])
                        num[r] += ((zsim - t) ** 2).sum(); den[r] += ((t - t.mean(1, keepdims=True)) ** 2).sum()
            else:
                z, v = zv_one(me, r)
                if len(z) >= 3:
                    vp = fl[r](z.T).T; num[r] += ((v - vp) ** 2).sum(); den[r] += ((v - v.mean(0)) ** 2).sum()
    return 1 - num / (den + 1e-12)                                 # per-regime held-out R²
def cv(params):
    r2 = _cv_perreg(params); return float(np.nanmean(r2))          # pooled held-out R²
if FIT == 'traj':                                                  # traj fit is expensive → small grid
    ADS = [(0.2, 2.0), (0.5, 2.0), (1.0, 0.8)]
    GRID = ([(a, dd, 1.0) for (a, dd) in ADS] if MODE == 'partial' else ADS)
else:
    ADS = [(a, dd) for a in (0.2, 0.4, 0.7, 1.0) for dd in (0.3, 0.8, 2.0)]
    GRID = ([(a, dd, lam) for (a, dd) in ADS for lam in (0.2, 1.0, 5.0, 20.0, 100.0)] if MODE == 'partial' else ADS)
cvs = {p: cv(p) for p in GRID}; best = max(cvs, key=cvs.get)
_metric = 'traj' if FIT == 'traj' else 'vel'
print(f'[{MODE}] {STAGE}  best params={best}  pooled CV {_metric}-R²={cvs[best]:+.3f}')
allm = regime_means(np.ones(len(yc), bool)); flows, _ = fit_mode(allm, best, wa=ANCHOR)
if FIT == 'traj':
    # GOAL = descriptive fit: does a rank-2 dynamics REPRODUCE the observed trajectories? -> in-sample
    # trajectory R² per regime (the headline). Held-out rank-2-vs-linear is a secondary generalization note.
    print('  in-sample trajectory R² (descriptive fit — does rank-2 reproduce the trajectories?):')
    isr2 = [traj_r2(flows, allm, r) for r in range(NREG)]
    for r, reg in enumerate(REG):
        print(f'    {reg[0]:11s} {isr2[r]:+.2f}')
    print(f'    {"POOLED":11s} {np.nanmean(isr2):+.2f}')
    r2_rank2 = _cv_perreg(best); r2_lin = _cv_perreg((0.0,) + tuple(best[1:]))
    print(f'  (secondary — held-out generalization: rank-2 pooled {np.nanmean(r2_rank2):+.2f} vs '
          f'linear {np.nanmean(r2_lin):+.2f}; poor & ≈linear = descriptive, not predictive)')
else:
    for r, (nm, rm, w, _) in enumerate(REG):
        print(f'  {nm:11s} vel-R²={vr2(flows, allm, r):+.2f}')

LIM = 1.3 * max(np.abs(np.concatenate([mu[:, w] for r, (nm, rm, w, _) in enumerate(REG) for _, mu in allm[r]], axis=1)).max(), 1.0)


# Fixed points = the condition-mean TRAJECTORY ENDPOINTS (settled = last 5 bins), NOT root-finding on
# the fitted field. Overlaps subproject convention (plot_flow2d.py): "trajectory endpoint is correct by
# construction; the root-found / speed-minimum fp is displaced in a noisy field." Verified here: at the
# CV gain the field's own attractor already sits on the A endpoint (speed ≈0.02), but B's shallow well
# is not classified by root-finding though the data settles there (speed ≈0.03) — so mark endpoints,
# which makes the trajectories terminate at the marked fixed points by construction. (Earlier autonomous
# gain-raising to force 2 wells was WRONG — it manufactured a 2nd attractor off the B endpoint; removed.)
def endpoints(r):
    return [mu[:, REG[r][2]][:, -5:].mean(1) for _, mu in allm[r]]   # settled (x,y) per condition
gl = np.linspace(-LIM, LIM, 60); Xg, Yg = np.meshgrid(gl, gl); P = np.vstack([Xg.ravel(), Yg.ravel()])
fig, axes = plt.subplots(2, 4, figsize=(20, 10.2))
for ax, (r, (nm, rm, w, _)) in zip(axes.ravel(), enumerate(REG)):
    F = flows[r](P); U, V = F[0].reshape(Xg.shape), F[1].reshape(Xg.shape)
    ax.pcolormesh(Xg, Yg, np.hypot(U, V), cmap='magma', shading='auto')
    ax.streamplot(Xg, Yg, U, V, color='w', density=1.0, linewidth=0.5, arrowsize=0.6)
    for lv, mu in allm[r]:
        col = COL.get(lv, 'c'); ax.plot(mu[0, w], mu[1, w], '-', color=col, lw=2.3, zorder=5); ax.plot(mu[0, w][-1], mu[1, w][-1], 'o', color=col, ms=6, zorder=6)
    for ep in endpoints(r):                                          # fixed points = trajectory endpoints
        ax.plot(ep[0], ep[1], '*', mfc='yellow', mec='k', ms=15, zorder=7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    _r2 = traj_r2(flows, allm, r) if FIT == 'traj' else vr2(flows, allm, r)
    ax.set_title(f'{nm}   ({_metric}-R²={_r2:+.2f})', fontsize=11)
    ax.set_xlabel('sample code'); ax.set_ylabel('choice code')
fp_leg = [Line2D([0], [0], ls='', marker='*', mfc='yellow', mec='k', ms=13, label='fixed point = trajectory endpoint')]
tr_leg = [Line2D([0], [0], color='#332288', lw=2.3, label='sample A'), Line2D([0], [0], color='#44AA99', lw=2.3, label='sample B'),
          Line2D([0], [0], color='#117733', lw=2.3, label='Go'), Line2D([0], [0], color='#CC6677', lw=2.3, label='NoGo'),
          Line2D([0], [0], ls='', marker='o', color='k', ms=6, label='trajectory end')]
l1 = fig.legend(handles=fp_leg, loc='lower center', ncol=3, bbox_to_anchor=(0.3, -0.02), frameon=False, fontsize=10, title='fixed points')
fig.legend(handles=tr_leg, loc='lower center', ncol=5, bbox_to_anchor=(0.72, -0.02), frameon=False, fontsize=10, title='trajectories'); fig.add_artist(l1)
_hdr = (f'descriptive in-sample traj-R²={np.nanmean([traj_r2(flows, allm, r) for r in range(NREG)]):+.2f}'
        if FIT == 'traj' else f'pooled CV vel-R²={cvs[best]:+.2f}')
fig.suptitle(f'OVERLAPS rank-2 gain-modulated flows [{MODE}, train axis = {TRAINSEL}, fit={FIT}, anchor={ANCHOR:g}] — '
             f'{STAGE}: autonomous + inputs  (single {TRAINSEL} axis; per-regime windows = odor±{MARGIN}bin margin; '
             f'params={best}; {_hdr})', y=1.0)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = f'{OUTDIR}/png/overlaps_lowrank_{MODE}_{STAGE}_train{TRAINSEL}.png'
fig.savefig(out, dpi=300, bbox_inches='tight'); fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
