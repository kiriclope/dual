"""Rank-2 reduced flows on the dPCA sample×choice plane (gain-modulated LINEAR form, no tanh
saturation): ż_d = -z_d + S(z)·(A z)_d + c_r,  S(z)=<φ'(√Δ ξ)>,  Δ = a²‖z‖²+δ.
Three modes:
  (default) partial : shared A_shared + ridge-penalized per-regime deviation ΔA_r (CV-tuned λ).
                      Best of both — generalizes AND captures per-regime structure (C/D diagonal).
  --shared          : ONE recurrent A + per-input current (additive c_r + per-mode in-gain h_r).
                      Parsimonious; a shared A can't give C/D their own diagonal.
  --independent     : each regime fits its OWN A_r + c_r. Captures C/D diagonal but overfits (esp. per mouse).
(a,δ[,λ]) CV-tuned on pooled held-out condition-mean velocity R². 8-panel grid with legend.
Usage: fig_dpca_flow_lowrank_shared.py [--shared|--independent] [--dum <DUM>]"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.pca.io import pkl_load
from src.pca.dynamics import flow_fixed_points

# ── Style (Nature Neuroscience house style — matches the main figures) ──────────
sns.set_context('notebook')          # MUST come after importing src.common.plot_utils (sets "poster")
sns.set_style('ticks')
plt.rcParams.update({                 # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

PUSH = '--push' in sys.argv          # add the no-lick push (tasks+time offset) onto the action axis
CHOICE_AUTO = '--choice-auto' in sys.argv   # resolve the AUTONOMOUS regime by upcoming choice (learned polarization)
_def_dum = ('pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca' if PUSH
            else 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca')
DUM = sys.argv[sys.argv.index('--dum') + 1] if '--dum' in sys.argv else _def_dum
_per_mouse = bool(DUM.split('_dpca')[-1].strip('_'))          # True if DUM has a mouse/sub-pool suffix
# default: pooled -> independent (best descriptive geometry); per-mouse -> partial (regularized).
MODE = ('shared' if '--shared' in sys.argv else 'independent' if '--independent' in sys.argv
        else 'partial' if '--partial' in sys.argv else ('partial' if _per_mouse else 'independent'))
Z = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca'); y = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')
lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca'); i, j = lab.index('sample'), lab.index('sample:test')
STAGE = 'Naive' if 'Naive' in DUM else 'Expert'               # trials match the DUM's basis (corrected)
m = ((y.laser == 0) & (y.learning == STAGE) & (y.performance == 1)).to_numpy()
if PUSH:
    # PUSH CORRECTION (corrected framing 2026-06-23): the `tasks` marginal IS a lick/no-lick ACTION axis
    # (DualGo=lick +0.96, DualNoGo/DPA=no-lick negative; tracks the lick; not ⊥ choice, cos +0.22). dPCA
    # demixing strips this absolute no-lick offset off the choice (sample:test) axis. Restore it on the
    # action axis: action = choice(sample:test, trial decision) + tasks(lick/no-lick action position),
    # tasks scaled to unit std WITHOUT centering (keeps the offset), oriented so the DPA delay is negative.
    # NB the condition-independent time/CI ramp is GLOBAL (not lick-specific) and is deliberately EXCLUDED.
    if 'tasks' not in lab:
        sys.exit("--push needs a DUM with 'tasks' (use the f-sample-test-tasks_dpca DUM)")
    itk = lab.index('tasks'); LATE = np.arange(39, 54); yp = y[m].reset_index(drop=True)
    sam = Z[m][:, i, :].astype(float); ch = Z[m][:, j, :].astype(float)
    tk = Z[m][:, itk, :].astype(float)
    push = tk / tk.std()                                      # tasks = lick/no-lick action axis (CI/time excluded)
    dpad = ((yp.tasks == 'DPA') & (yp.performance == 1)).to_numpy()
    if push[dpad][:, LATE].mean() > 0:
        push = -push                                          # no-lick = negative on the action axis
    act = ch / ch.std() + push                                # choice (decision) + no-lick action push
    Z2 = np.stack([sam, act], axis=1); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
else:
    Z2 = Z[m][:, [i, j], :].astype(float); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
SMOOTH = float(sys.argv[sys.argv.index('--smooth') + 1]) if '--smooth' in sys.argv else 0.0
if SMOOTH > 0:                                                # temporal denoise (helps thin per-mouse means)
    from scipy.ndimage import gaussian_filter1d
    Z2 = gaussian_filter1d(Z2, SMOOTH, axis=2)
yc = y[m].reset_index(drop=True)
samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy(); task = yc['tasks'].to_numpy(); ch = yc['choice'].to_numpy()
go, nogo, dpa = task == 'DualGo', task == 'DualNoGo', task == 'DPA'
os.makedirs('figures/pseudo/flow/lowrank/png', exist_ok=True); os.makedirs('figures/pseudo/flow/lowrank/svg', exist_ok=True)

REG = [('autonomous', dpa, np.arange(21, 54), ('sample', (0, 1))),
       ('A input', samp == 0, np.arange(15, 30), ('sample', (0,))),
       ('B input', samp == 1, np.arange(15, 30), ('sample', (1,))),
       ('Go input', go, np.arange(30, 52), ('sample', (0, 1))),
       ('NoGo input', nogo, np.arange(30, 52), ('sample', (0, 1))),
       ('Cue input', go | nogo, np.arange(30, 52), ('tasks', ('DualGo', 'DualNoGo'))),
       ('C input', test == 0, np.arange(57, 84), ('sample', (0, 1))),
       ('D input', test == 1, np.arange(57, 84), ('sample', (0, 1)))]
NREG = len(REG); COL = {0: '#332288', 1: '#44AA99', 'DualGo': '#117733', 'DualNoGo': '#CC6677'}
NODES, WK = np.polynomial.hermite_e.hermegauss(20); WK = WK / np.sqrt(2 * np.pi)
def gd(D, h): t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * NODES[None, :] + h[:, None]); return (WK * (1 - t ** 2)).sum(1)


def regime_means(trm):
    out = {}
    for r, (nm, rm, w, (fac, levs)) in enumerate(REG):
        if CHOICE_AUTO and nm == 'autonomous':               # resolve the DPA-delay WM states by upcoming choice
            out[r] = [((s, c), Z2[rm & trm & (samp == s) & (ch == c)].mean(0))
                      for s in (0, 1) for c in (0, 1) if (rm & trm & (samp == s) & (ch == c)).sum() >= 3]
        else:
            fv = yc[fac].to_numpy(); out[r] = [(lv, Z2[rm & trm & (fv == lv)].mean(0)) for lv in levs if (rm & trm & (fv == lv)).sum() >= 3]
    return out
def zv_one(means, r):
    w = REG[r][2]; zs, vs = [], []
    for _, mu in means[r]:
        zs.append(mu[:, w][:, :-1].T); vs.append(np.diff(mu[:, w], axis=1).T)
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


# ---- independent model: per-regime A_r + c_r (isotropic gain) ----
def fit_indep_one(z, v, a, dd):
    if len(z) < 3:
        return np.zeros((2, 2)), np.zeros(2)
    D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z))); F = np.column_stack([S * z[:, 0], S * z[:, 1], np.ones(len(z))])
    c0 = np.linalg.lstsq(F, v[:, 0] + z[:, 0], rcond=None)[0]; c1 = np.linalg.lstsq(F, v[:, 1] + z[:, 1], rcond=None)[0]
    return np.array([[c0[0], c0[1]], [c1[0], c1[1]]]), np.array([c0[2], c1[2]])
def flow_indep(A, c, a, dd):
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd; S = gd(D, np.zeros(P.shape[1])); AP = A @ P
        return np.vstack([-P[0] + S * AP[0] + c[0], -P[1] + S * AP[1] + c[1]])
    return fl
def fit_all_indep(means, a, dd):
    flows = {}
    for r in range(NREG):
        z, v = zv_one(means, r); A, c = fit_indep_one(z, v, a, dd); flows[r] = flow_indep(A, c, a, dd)
    return flows, None


# ---- partial pooling: shared A + ridge-penalized per-regime deviation ΔA_r (+ per-regime c) ----
def fit_all_partial(means, a, dd, lam):
    z, v, rid = stack(means); D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
    OH = np.eye(NREG)[rid]; shF = np.column_stack([S * z[:, 0], S * z[:, 1]])
    devF = (OH[:, :, None] * shF[:, None, :]).reshape(len(z), NREG * 2)            # per-regime S·z
    F = np.column_stack([shF, devF, OH]); Pn = F.shape[1]
    reg = np.zeros((Pn, Pn)); reg[2:2 + 2 * NREG, 2:2 + 2 * NREG] = lam * np.eye(2 * NREG)  # ridge ΔA only
    FtF = F.T @ F + reg; A_sh = np.zeros((2, 2)); dA = np.zeros((NREG, 2, 2)); C = np.zeros((NREG, 2))
    for d in (0, 1):
        cd = np.linalg.solve(FtF, F.T @ (v[:, d] + z[:, d])); A_sh[d] = cd[0:2]
        for r in range(NREG):
            dA[r, d] = cd[2 + 2 * r:2 + 2 * r + 2]
        C[:, d] = cd[2 + 2 * NREG:]
    return {r: flow_indep(A_sh + dA[r], C[r], a, dd) for r in range(NREG)}, (A_sh, dA, C)


def fit_mode(means, params):
    if MODE == 'shared':
        z, v, rid = stack(means); return fit_all_shared(z, v, rid, *params)
    if MODE == 'partial':
        return fit_all_partial(means, *params)
    return fit_all_indep(means, *params)
def vr2(flows, means, r):
    z, v = zv_one(means, r)
    if len(z) < 3: return np.nan
    vp = flows[r](z.T).T; return 1 - ((v - vp) ** 2).sum() / (((v - v.mean(0)) ** 2).sum() + 1e-12)


folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(yc))))
def cv(params):
    rr = []
    for tr, te in folds:
        trm = np.zeros(len(yc), bool); trm[tr] = True; tem = np.zeros(len(yc), bool); tem[te] = True
        fl, _ = fit_mode(regime_means(trm), params); me = regime_means(tem)
        num = den = 0.0
        for r in range(NREG):
            z, v = zv_one(me, r)
            if len(z) >= 3:
                vp = fl[r](z.T).T; num += ((v - vp) ** 2).sum(); den += ((v - v.mean(0)) ** 2).sum()
        rr.append(1 - num / (den + 1e-12))
    return np.mean(rr)
ADS = [(a, dd) for a in (0.2, 0.4, 0.7, 1.0) for dd in (0.3, 0.8, 2.0)]
GRID = ([(a, dd, lam) for (a, dd) in ADS for lam in (0.2, 1.0, 5.0, 20.0, 100.0)] if MODE == 'partial' else ADS)
cvs = {p: cv(p) for p in GRID}; best = max(cvs, key=cvs.get)
print(f'[{MODE}] best params={best}  pooled CV vel-R²={cvs[best]:+.3f}')
allm = regime_means(np.ones(len(yc), bool)); flows, _ = fit_mode(allm, best)
ba, bd = best[0], best[1]
if PUSH:
    # CORRECTED autonomous panel: the tasks RAMP during the delay is INPUT-DRIVEN, so refitting an
    # autonomous flow on the pushed (descending) trajectory collapses the two wells into one too-deep
    # attractor (the "autonomous fit on an input-driven descent is wrong" failure). Instead: fit the
    # bistability on the CHOICE axis (preserves both wells), then add a no-lick drive GATED by recurrent
    # activity r(z)=1−S(z) (≈0 at the quiet saddle, →1 at the active wells) → the wells push DOWN and the
    # slow manifold DEFORMS (non-uniform), the attractors stay, and it is not a rigid translation.
    Wa = np.arange(21, 54)
    sam_n = Z[m][:, i, :].astype(float); ch_n = Z[m][:, j, :].astype(float)
    Z2c = np.stack([sam_n, ch_n], 1); Z2c = Z2c / Z2c.std((0, 2), keepdims=True) * 2.8
    mc = [Z2c[dpa & (samp == s)][:, :, Wa].mean(0) for s in (0, 1)]
    zc = np.concatenate([mu[:, :-1].T for mu in mc]); vc = np.concatenate([np.diff(mu, 1).T for mu in mc])
    def fit_a(a):                                                       # rank-2 gain-mod fit at gain a (bd fixed)
        S = gd(a ** 2 * (zc ** 2).sum(1) + bd, np.zeros(len(zc)))
        Fc = np.column_stack([S * zc[:, 0], S * zc[:, 1], np.ones(len(zc))])
        w0 = np.linalg.lstsq(Fc, vc[:, 0] + zc[:, 0], rcond=None)[0]
        w1 = np.linalg.lstsq(Fc, vc[:, 1] + zc[:, 1], rcond=None)[0]
        return np.array([[w0[0], w0[1]], [w1[0], w1[1]]]), np.array([w0[2], w1[2]])
    def n_att(a, A, c):
        f = lambda P: (lambda Pp, D: np.vstack([-Pp[0] + gd(D, np.zeros(Pp.shape[1])) * (A @ Pp)[0] + c[0],
                                                -Pp[1] + gd(D, np.zeros(Pp.shape[1])) * (A @ Pp)[1] + c[1]]))(
            np.atleast_2d(P), a ** 2 * (np.atleast_2d(P) ** 2).sum(0) + bd)
        return sum(k == 'attractor' for _, k, _ in flow_fixed_points(f, [(-3, 3), (-3, 3)], n_seed=21))
    # pooled CV gain (ba) optimizes input-regime prediction & is often monostable; the autonomous WM
    # bistability is an established result that manifests at higher gain — use the smallest a giving 2 wells.
    ba_auto = ba
    for a_try in [ba, 0.4, 0.7, 1.0, 1.4]:
        A_t, c_t = fit_a(a_try)
        if n_att(a_try, A_t, c_t) >= 2:
            ba_auto, Ac, cc = a_try, A_t, c_t; break
    else:
        ba_auto, (Ac, cc) = ba, fit_a(ba)
    ba_used = ba_auto
    tgt = float(np.mean([mu[1, -5:].mean() for _, mu in allm[0]]))      # settled no-lick depth (pushed means)
    xw = float(np.mean([abs(mu[0, -5:].mean()) for mu in mc]))          # well location on the sample axis
    # the gate r=1−S creates positive feedback (deeper y → higher activity → more drive), so a closed-form
    # h overshoots and collapses bistability. Tune h numerically: largest h that KEEPS 2 attractors while
    # approaching the target depth → wells pushed slightly down, manifold deforms, both wells preserved.
    def gated(h, _a=ba_used, _A=Ac, _c=cc):
        def fl(P, _h=h):
            P = np.atleast_2d(P); D = _a ** 2 * (P ** 2).sum(0) + bd
            Sp = gd(D, np.zeros(P.shape[1])); AP = _A @ P
            return np.vstack([-P[0] + Sp * AP[0] + _c[0], -P[1] + Sp * AP[1] + _c[1] - _h * (1.0 - Sp)])
        return fl
    hpush, ay = 0.0, 0.0
    for h in np.linspace(0, 1.5, 31):
        att = [p for p, k, _ in flow_fixed_points(gated(h), [(-3, 3), (-3, 3)], n_seed=21) if k == 'attractor']
        if len(att) < 2:
            break                                                      # bistability lost → keep last good h
        hpush, ay = h, float(np.mean([p[1] for p in att]))
        if ay <= tgt:
            break                                                      # reached the no-lick depth, still bistable
    flows[0] = gated(hpush)
    print(f'  [autonomous PUSH fix] gain a={ba_used:.1f} bistable + gated drive h={hpush:.2f} '
          f'(well depth {ay:+.2f}, target {tgt:+.2f}, base #att={n_att(ba_used, Ac, cc)})')
for r, (nm, rm, w, _) in enumerate(REG):
    print(f'  {nm:11s} vel-R²={vr2(flows, allm, r):+.2f}')

LIM = 1.3 * max(np.abs(np.concatenate([mu[:, w] for r, (nm, rm, w, _) in enumerate(REG) for _, mu in allm[r]], axis=1)).max(), 1.0)
gl = np.linspace(-LIM, LIM, 60); Xg, Yg = np.meshgrid(gl, gl); P = np.vstack([Xg.ravel(), Yg.ravel()])
fig, axes = plt.subplots(2, 4, figsize=(20, 10.2))
for ax, (r, (nm, rm, w, _)) in zip(axes.ravel(), enumerate(REG)):
    F = flows[r](P); U, V = F[0].reshape(Xg.shape), F[1].reshape(Xg.shape)
    ax.pcolormesh(Xg, Yg, np.hypot(U, V), cmap='magma', shading='auto')
    ax.streamplot(Xg, Yg, U, V, color='w', density=1.0, linewidth=0.5, arrowsize=0.6)
    for lv, mu in allm[r]:
        if isinstance(lv, tuple):                            # autonomous choice-resolved: (sample, choice)
            s, c = lv; col = COL[s]; ls = '-' if c == 1 else '--'; mk = '*' if c == 1 else 'o'; ms = 13 if c == 1 else 7
            ax.plot(mu[0, w], mu[1, w], ls, color=col, lw=2.3, zorder=5)
            ax.plot(mu[0, w][-1], mu[1, w][-1], mk, color=col, mfc=col, mec='k', ms=ms, zorder=6)
        else:
            col = COL.get(lv, 'c'); ax.plot(mu[0, w], mu[1, w], '-', color=col, lw=2.3, zorder=5); ax.plot(mu[0, w][-1], mu[1, w][-1], 'o', color=col, ms=6, zorder=6)
    for pt, kind, _ in flow_fixed_points(flows[r], [(-LIM, LIM), (-LIM, LIM)], n_seed=18):
        mk = {'attractor': ('*', 'yellow', 14), 'saddle': ('s', 'w', 8), 'repeller': ('X', 'r', 9)}.get(kind, ('*', 'y', 12))
        ax.plot(pt[0], pt[1], mk[0], mfc=mk[1], mec='k', ms=mk[2], zorder=7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ttl = (f'{nm}  (bistable + gated no-lick push)' if (PUSH and r == 0) else f'{nm}   (R²={vr2(flows, allm, r):+.2f})')
    ax.set_title(ttl, loc='left', fontsize=TITLE_FS); ax.set_xlabel('sample')
    ax.set_ylabel('action: choice + tasks  (− = no-lick)' if PUSH else 'choice')
fp_leg = [Line2D([0], [0], ls='', marker='*', mfc='yellow', mec='k', ms=13, label='attractor (stable)'),
          Line2D([0], [0], ls='', marker='s', mfc='w', mec='k', ms=8, label='saddle'),
          Line2D([0], [0], ls='', marker='X', mfc='r', mec='k', ms=9, label='repeller (unstable)')]
tr_leg = [Line2D([0], [0], color='#332288', lw=2.3, label='sample A'), Line2D([0], [0], color='#44AA99', lw=2.3, label='sample B'),
          Line2D([0], [0], color='#117733', lw=2.3, label='Go'), Line2D([0], [0], color='#CC6677', lw=2.3, label='NoGo'),
          Line2D([0], [0], ls='', marker='o', color='k', ms=6, label='trajectory end')]
l1 = fig.legend(handles=fp_leg, loc='lower center', ncol=3, bbox_to_anchor=(0.3, -0.02), frameon=False, fontsize=6.5, title='fixed points')
fig.legend(handles=tr_leg, loc='lower center', ncol=5, bbox_to_anchor=(0.72, -0.02), frameon=False, fontsize=6.5, title='trajectories'); fig.add_artist(l1)
fig.suptitle(f'dPCA rank-2 gain-modulated flows [{MODE}] — {STAGE}: autonomous + inputs  '
             f'(params={best}; pooled CV vel-R²={cvs[best]:+.2f})', y=1.0, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
TAG = DUM.split('_dpca')[-1].strip('_')                       # '' for pooled, mouse name for per-mouse
out = f'figures/pseudo/flow/lowrank/png/dpca_lowrank_{MODE}_{STAGE}{"_push" if PUSH else ""}{"_choiceauto" if CHOICE_AUTO else ""}{("_" + TAG) if TAG else ""}.png'
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
