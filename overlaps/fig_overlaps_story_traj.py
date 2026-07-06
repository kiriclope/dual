"""PROTOTYPE — trajectory-fit variant of fig_overlaps_story_main.py.

Identical to the story figure in EVERY respect except the flow-fit objective, so the two can be compared
side by side:
  - SAME data plane: per-group decoders (autonomous+A/B @DELAY, Go/NoGo @RESP, Cue @CUE, C/D on the mixed
    sample@TEST × choice@RESP plane), canonical correct mask, per-mouse BL-std norm.
  - SAME layout: §3 flow grid (Expert) + §4 no-lick push (unchanged — the push stats don't use the fit).
  - SAME fixed-point display: root-found on the fitted field (white ○, edge = attractor/saddle/repeller).

WHAT CHANGES — the fit objective:
  story_main: closed-form LS on the one-bin finite-difference VELOCITY of the condition means, scored by
              held-out velocity R² (a noise-amplified derivative → sits at the ≈0 noise floor).
  this:       fit the flow so the INTEGRATED trajectory reproduces the observed condition-mean path
              (POSITION), via least_squares; scored by TRAJECTORY R² (the observable). PARTIAL POOLING is
              kept, now under the trajectory objective: each decoder GROUP shares one recurrent landscape
              A_sh + a ridge-λ-shrunk per-regime deviation ΔA_r, fit JOINTLY (nonlinear LS, warm-started
              from the closed-form velocity partial-pool). Every regime is endpoint-anchored (v=0 at the
              drawn trajectory end → the fitted field has a fixed point there). CV-tunes (a,δ,λ).
              NO bistable-gain restriction — the autonomous landscape is fit honestly, not forced.

THE INPUT ENTERS THE LOW-RANK FORM (--input, default inside):
  inside  : rank-2 low-rank RNN (Mastrogiuseppe & Ostojic 2018; Dubreuil et al. 2022) —
            ż = -z + S(z)·(A z + b_r),  b_r the input drive INSIDE the gain, its variance β_r added to
            Δ = a²‖z‖²+δ+β_r.  (Fit b_r and β_r per regime; velocity warm-start uses the S·b_r column.)
  outside : the previous form — ż = -z + S(z)·(A z) + c_r, external additive current (β ignored).

Usage:
  cd /home/leon/dual/overlaps
  python fig_overlaps_story_traj.py [--panels 8|4] [--all-trials]
                                    [--mode partial|shared|independent] [--input inside|outside]
Saves figures/overlaps/story_traj/{png,svg}/fig_overlaps_story_traj_<mode>_<input>[_all].{png,svg}.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, argparse, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
from scipy.optimize import least_squares
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
ap.add_argument('--mode', choices=['partial', 'shared', 'independent'], default='partial',
                help='per-group landscape pooling: partial = shared A_sh + ridge ΔA_r (default); '
                     'shared = ONE A_sh per group (ΔA_r≡0); independent = per-regime A_r (no shared, no ridge)')
ap.add_argument('--input', choices=['inside', 'outside'], default='inside',
                help='where the per-regime input drive enters: inside (default, low-rank RNN form) = '
                     'ż=-z+S(z)(A z + b_r), b_r inside the gain, its variance β_r added to Δ; '
                     'outside (previous form) = ż=-z+S(z)(A z)+c_r, external additive current')
args = ap.parse_args()
CORRECT = not args.all_trials
MODE = args.mode
INSIDE = args.input == 'inside'          # low-rank RNN: input drive inside the nonlinearity (gain-modulated)
TAG = '' if CORRECT else '_all'
MARGIN = 3
VSTEP = 1
ANCHOR = 8.0                       # endpoint-anchoring weight (autonomous only): pin v=0 at settled endpoints
ADS_TRAJ = [(0.2, 2.0), (0.5, 2.0), (1.0, 0.8)]    # (a, δ) gain grid
LAMS = [1.0, 20.0]                                 # ridge on ΔA_r (partial mode only; low vs high shrink)
GRID = [(a, dd, lam) for (a, dd) in ADS_TRAJ for lam in (LAMS if MODE == 'partial' else [0.0])]

# ── data / constants (verbatim from fig_overlaps_story_main.py) ─────────────────
DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA = '../data/overlaps'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
o = set_options(mice=MICE, tasks=['Dual'], mouse=MICE[0], laser=0, trials='', data_type='dF',
                prescreen=None, pval=0.05, preprocess=None, scaler_BL='standard_BL', avg_noise=False,
                unit_var_BL=False, random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
                scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10, class_weight=0,
                multilabel=0, mne_estimator='generalizing', n_jobs=64, days=['first', 'last'])
BL = slice(0, 12)
BINS_BL = o['bins_BL']
BINS_LATE = np.arange(27, 54)
COL = {0: '#332288', 1: '#44AA99', 'DualGo': '#117733', 'DualNoGo': '#CC6677'}
KEY = ['mouse', 'day', 'tasks', 'sample_odor', 'test_odor', 'odor_pair', 'response', 'odr_perf']
TRAIN_PUSH = o['bins_DELAY']

X = pkl_load(f'X_{DUM}', path=DATA)
y = pkl_load(f'labels_{DUM}', path=DATA)

NODES, WK = np.polynomial.hermite_e.hermegauss(20); WK = WK / np.sqrt(2 * np.pi)
def gd(D, h):
    t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * NODES[None, :] + h[:, None])
    return (WK * (1 - t ** 2)).sum(1)


def _win(bins):
    b = np.asarray(bins); return np.arange(b[0], min(b[-1] + 1 + MARGIN, 84))
WIN_STIM, WIN_DIST = _win(o['bins_STIM']), _win(o['bins_DIST'])
WIN_CUE, WIN_TEST = _win(o['bins_CUE']), _win(o['bins_TEST'])
WIN_MD, WIN_RWD = _win(o['bins_MD']), _win(o['bins_RWD'])   # Go/NoGo read over MD; Cue read over GNG reward
WIN_CD = np.arange(57, 63)          # C/D read: ±0.5 s straddling test→choice (t 9.5–10.5), where the diagonal forms
EPOCHS = {'DELAY': (o['bins_DELAY'],), 'CUE': (o['bins_CUE'],),
          'TEST': (o['bins_TEST'], o['bins_CHOICE']), 'RESP': (o['bins_CHOICE'],)}
REG_EPOCH = {'autonomous': 'DELAY', 'A input': 'DELAY', 'B input': 'DELAY', 'Go input': 'RESP',
             'NoGo input': 'RESP', 'Cue input': 'CUE', 'C input': 'TEST', 'D input': 'TEST'}


def codes(stage, correct, sample_bins, choice_bins=None, mice=None):
    choice_bins = sample_bins if choice_bins is None else choice_bins
    base = (y.laser == 0) & (y.learning == stage)
    if correct:
        base = base & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1))
    if mice is not None:
        base = base & y.mouse.isin(mice)

    def block(tgt, tb):
        df = X[..., tb, :].mean(-2)[:, 1].astype(float)
        for mo in MICE:
            mm = (y.mouse == mo).to_numpy(); sd = df[mm][:, BL].std()
            if sd > 0:
                df[mm] /= sd
        mb = (base & (y.target == tgt)).to_numpy(); yb = y[mb].reset_index(drop=True)
        order = yb.sort_values(KEY, kind='stable').index.to_numpy()
        return df[mb][order], yb.iloc[order].reset_index(drop=True)
    ds, _ = block('sample', sample_bins); dc, yc = block('choice', choice_bins)
    Z2 = np.stack([ds, dc], axis=1).astype(float); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
    return Z2, yc


def build_reg(yc):
    samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy(); task = yc['tasks'].to_numpy()
    go, nogo, dpa = task == 'DualGo', task == 'DualNoGo', task == 'DPA'
    return [('autonomous', dpa, np.arange(21, 54), ('sample', (0, 1))),
            ('A input', samp == 0, np.arange(12, 30), ('sample', (0,))),
            ('B input', samp == 1, np.arange(12, 30), ('sample', (1,))),
            ('Go input', go, WIN_MD, ('sample', (0, 1))),        # read Go over the memory delay (MD)
            ('NoGo input', nogo, WIN_MD, ('sample', (0, 1))),    # read NoGo over MD
            ('Cue input', go | nogo, WIN_RWD, ('tasks', ('DualGo', 'DualNoGo'))),   # read Cue over GNG reward
            ('C input', test == 0, WIN_CD, ('sample', (0, 1))),  # ±0.5 s around test→choice
            ('D input', test == 1, WIN_CD, ('sample', (0, 1)))]
PANELS4 = [0, 1, 5, 6]


def regime_means(REG, planes, yc, trm):
    out = {}
    for r, (nm, rm, w, (fac, levs)) in enumerate(REG):
        Z2 = planes[REG_EPOCH[nm]]
        fv = yc[fac].to_numpy()
        out[r] = [(lv, Z2[rm & trm & (fv == lv)].mean(0)) for lv in levs
                  if (rm & trm & (fv == lv)).sum() >= 3]
    return out


def groups_of(REG):
    g = {}
    for r in range(len(REG)):
        g.setdefault(REG_EPOCH[REG[r][0]], []).append(r)
    return list(g.values())


def build_flow(A, inp, a, dd, beta=0.0):
    """Rank-2 reduced flow for one regime.  --input inside (low-rank RNN):  ż = -z + S(z)·(A z + b),  the
    input drive b = inp is INSIDE the gain and its variance β adds to Δ = a²‖z‖²+δ+β.  --input outside
    (previous form):  ż = -z + S(z)·(A z) + c,  external additive current c = inp (β ignored)."""
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd + (beta if INSIDE else 0.0)
        S = gd(D, np.zeros(P.shape[1])); AP = A @ P
        if INSIDE:
            return np.vstack([-P[0] + S * (AP[0] + inp[0]), -P[1] + S * (AP[1] + inp[1])])
        return np.vstack([-P[0] + S * AP[0] + inp[0], -P[1] + S * AP[1] + inp[1]])
    return fl


# ── TRAJECTORY FIT (ported from fig_overlaps_flow_lowrank_shared.py --fit traj) ──
def sim(flow_fn, z0, n):                                         # Euler-integrate the flow (dt = 1 bin)
    z = np.asarray(z0, float).copy(); out = [z.copy()]
    for _ in range(n - 1):
        z = np.clip(z + flow_fn(z[:, None]).ravel(), -30, 30); out.append(z.copy())
    return np.array(out).T


def zv_one(REG, means, r):                                     # (position, one-bin velocity) pairs of regime r
    w = REG[r][2]; zs, vs = [], []
    for _, mu in means[r]:
        M = mu[:, w]
        zs.append(M[:, :-1].T); vs.append(np.diff(M, axis=1).T)
    return (np.concatenate(zs), np.concatenate(vs)) if zs else (np.empty((0, 2)), np.empty((0, 2)))


def traj_r2_one(flow_fn, mu, w):                               # in-sample trajectory R² for one condition
    t = mu[:, w]
    if t.shape[1] < 3:
        return np.nan
    zsim = sim(flow_fn, t[:, 0], t.shape[1])
    return 1 - ((zsim - t) ** 2).sum() / (((t - t.mean(1, keepdims=True)) ** 2).sum() + 1e-12)


def traj_r2(REG, flows, means, r):
    vals = [traj_r2_one(flows[r], mu, REG[r][2]) for _, mu in means[r]]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def fit_stage(stage, correct):
    planes = {}; yc = None
    for en in EPOCHS:
        Z2, yc = codes(stage, correct, *EPOCHS[en]); planes[en] = Z2
    REG = build_reg(yc); NREG = len(REG)
    allm = regime_means(REG, planes, yc, np.ones(len(yc), bool))
    span = np.concatenate([mu[:, REG[r][2]] for r in range(NREG) for _, mu in allm[r]], axis=1)
    L = 1.3 * max(float(np.abs(span).max()), 1.0)
    folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(yc))))
    fold_data = []
    for tr, te in folds:
        trm = np.zeros(len(yc), bool); trm[tr] = True; tem = np.zeros(len(yc), bool); tem[te] = True
        fold_data.append((regime_means(REG, planes, yc, trm), regime_means(REG, planes, yc, tem)))

    def endpoints(means, r):
        return [mu[:, REG[r][2]][:, -1] for _, mu in means[r]]     # DRAWN trajectory end → pin the fixed point there

    def vel_pool_params(means, a, dd, lam, subset):
        # closed-form VELOCITY partial-pool over the group (story_main fit_group) → warm-start for the
        # nonlinear trajectory fit: shared A_sh + ridge-λ per-regime ΔA_r + current c_r.
        Z_, V_, R_ = [], [], []
        for loc, r in enumerate(subset):
            z, v = zv_one(REG, means, r)
            if len(z):
                Z_.append(z); V_.append(v); R_.append(np.full(len(z), loc))
        ng = len(subset)
        if not Z_:
            return np.zeros((2, 2)), np.zeros((ng, 2, 2)), np.zeros((ng, 2))
        z = np.concatenate(Z_); v = np.concatenate(V_); rid = np.concatenate(R_).astype(int)
        D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
        OH = np.eye(ng)[rid]; shF = np.column_stack([S * z[:, 0], S * z[:, 1]])
        devF = (OH[:, :, None] * shF[:, None, :]).reshape(len(z), ng * 2)
        inpF = (S[:, None] * OH) if INSIDE else OH          # input cols: S·b_r (inside) vs c_r (outside)
        F = np.column_stack([shF, devF, inpF]); Pn = F.shape[1]
        reg = np.zeros((Pn, Pn)); reg[2:2 + 2 * ng, 2:2 + 2 * ng] = lam * np.eye(2 * ng)
        reg += 1e-6 * np.eye(Pn)                       # tiny floor: F is rank-deficient (shF = Σ devF) at λ→0
        FtF = F.T @ F + reg; A_sh = np.zeros((2, 2)); dA = np.zeros((ng, 2, 2)); C = np.zeros((ng, 2))
        for d in (0, 1):
            cd = np.linalg.solve(FtF, F.T @ (v[:, d] + z[:, d])); A_sh[d] = cd[0:2]
            for loc in range(ng):
                dA[loc, d] = cd[2 + 2 * loc:2 + 2 * loc + 2]
            C[:, d] = cd[2 + 2 * ng:]
        return A_sh, dA, C

    def _traj_anchor_resid(fl, tgts, anch, regs):                # trajectory-integration + endpoint anchors
        out = []
        for r in regs:
            for t in tgts[r]:
                out.append((sim(fl[r], t[:, 0], t.shape[1]) - t).ravel())
            for e in anch[r]:                                    # pin v=0 at EVERY regime's settled endpoint
                out.append(np.sqrt(ANCHOR) * fl[r](np.asarray(e, float)[:, None]).ravel())  # → trajectory ends at a fixed point
        return out

    # per-regime input block = drive (2) [+ β-raw (1) when the input is inside the gain, β=raw²≥0]
    NIN = 3 if INSIDE else 2
    def _inp(blk):                                           # (drive, β) from a per-regime input block
        return blk[:2], (blk[2] ** 2 if INSIDE else 0.0)
    def _inp0(bi):                                           # warm-start block: velocity-fit drive, β raw = 0
        return np.concatenate([bi, [0.0]]) if INSIDE else bi

    def fit_group(means, a, dd, lam, subset):
        # Fit the group's flows under the trajectory objective, warm-started from the closed-form velocity
        # pool. --mode sets the recurrent-landscape pooling (partial / shared / independent); --input sets
        # whether the per-regime drive is INSIDE the gain (b_r, +β_r on Δ) or an external current (c_r).
        regs = list(subset); ng = len(regs)
        tgts = {r: [mu[:, REG[r][2]] for _, mu in means[r] if mu[:, REG[r][2]].shape[1] >= 3] for r in regs}
        if all(len(tgts[r]) == 0 for r in regs):
            return {r: build_flow(np.zeros((2, 2)), np.zeros(2), a, dd) for r in regs}
        lam_init = lam if MODE == 'partial' else (1e6 if MODE == 'shared' else 0.0)
        A0, dA0, C0 = vel_pool_params(means, a, dd, lam_init, regs)
        anch = {r: endpoints(means, r) for r in regs}

        if MODE == 'independent':                                # per regime: A_r (4) + input (NIN)
            BLK = 4 + NIN
            p0 = np.concatenate([np.concatenate([(A0 + dA0[i]).ravel(), _inp0(C0[i])]) for i in range(ng)])
            def unpack(p):
                fl = {}
                for i, r in enumerate(regs):
                    blk = p[BLK * i:BLK * i + BLK]; inp, beta = _inp(blk[4:])
                    fl[r] = build_flow(blk[0:4].reshape(2, 2), inp, a, dd, beta)
                return fl
            def resid(p):
                return np.concatenate(_traj_anchor_resid(unpack(p), tgts, anch, regs))
        elif MODE == 'shared':                                   # A_sh (4) + per regime input (NIN)
            p0 = np.concatenate([A0.ravel()] + [_inp0(C0[i]) for i in range(ng)])
            def unpack(p):
                A_sh = p[:4].reshape(2, 2); rest = p[4:]; fl = {}
                for i, r in enumerate(regs):
                    inp, beta = _inp(rest[NIN * i:NIN * i + NIN])
                    fl[r] = build_flow(A_sh, inp, a, dd, beta)
                return fl
            def resid(p):
                return np.concatenate(_traj_anchor_resid(unpack(p), tgts, anch, regs))
        else:                                                    # partial: A_sh (4) + per regime [ΔA_r (4) + input (NIN)]
            BLK = 4 + NIN
            p0 = np.concatenate([A0.ravel()] +
                                [np.concatenate([dA0[i].ravel(), _inp0(C0[i])]) for i in range(ng)])
            def unpack(p):
                A_sh = p[:4].reshape(2, 2); rest = p[4:]; fl = {}
                for i, r in enumerate(regs):
                    blk = rest[BLK * i:BLK * i + BLK]; inp, beta = _inp(blk[4:])
                    fl[r] = build_flow(A_sh + blk[0:4].reshape(2, 2), inp, a, dd, beta)
                return fl
            def resid(p):
                fl = unpack(p); rest = p[4:]
                out = _traj_anchor_resid(fl, tgts, anch, regs)
                for i in range(ng):
                    out.append(np.sqrt(lam) * rest[BLK * i:BLK * i + 4])   # ridge ΔA_r → shrink toward A_sh
                return np.concatenate(out)
        sol = least_squares(resid, p0, max_nfev=200)
        return unpack(sol.x)

    def cv_traj(subset, a, dd, lam):                            # held-out TRAJECTORY R² (5-fold)
        num = den = 0.0
        for trm_m, tem_m in fold_data:
            fl = fit_group(trm_m, a, dd, lam, subset)
            for r in subset:
                for _, mu in tem_m[r]:
                    t = mu[:, REG[r][2]]
                    if t.shape[1] >= 3:
                        zsim = sim(fl[r], t[:, 0], t.shape[1])
                        num += ((zsim - t) ** 2).sum(); den += ((t - t.mean(1, keepdims=True)) ** 2).sum()
        return 1 - num / (den + 1e-12)

    flows = {}; best_by = {}; cv_groups = []
    for g in groups_of(REG):
        cvs = {p: cv_traj(g, *p) for p in GRID}
        best = max(GRID, key=lambda p: cvs[p])
        flows.update(fit_group(allm, *best, g))
        glabel = "/".join(REG[r][0].split()[0] for r in g)
        cv_groups.append((glabel, cvs[best]))
        best_by[REG_EPOCH[REG[g[0]][0]]] = best
        isr = np.nanmean([traj_r2_one(flows[r], mu, REG[r][2]) for r in g for _, mu in allm[r]])
        print(f'  {stage:6s} [{glabel:11s}] (a,δ,λ)={best}  held-out τR²={cvs[best]:+.3f}  in-sample τR²={isr:+.3f}')
    isr2 = {r: traj_r2(REG, flows, allm, r) for r in range(NREG)}
    return dict(REG=REG, allm=allm, flows=flows, cv_groups=cv_groups, isr2=isr2, L=L, best=best_by)


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
        ax.axhline(0, color='c', lw=1.1, ls='--', zorder=4)
    for lv, mu in allm[r]:
        col = COL.get(lv, 'c')
        ax.plot(mu[0, w], mu[1, w], '-', color=col, lw=2.0, zorder=5)
        ax.plot(mu[0, w][0], mu[1, w][0], 'o', color=col, ms=5, mfc='w', zorder=6)
        ax.plot(mu[0, w][-1], mu[1, w][-1], '*', color=col, ms=12, zorder=6)
    for pt, kind, _ in flow_fixed_points(flows[r], [(-LIM, LIM), (-LIM, LIM)], n_seed=18):
        mec = {'attractor': 'k', 'saddle': '0.55', 'repeller': 'r'}.get(kind, 'k')     # edge = FP type
        ms = 10 if kind == 'attractor' else 8
        ax.plot(pt[0], pt[1], 'o', mfc='white', mec=mec, ms=ms, mew=1.3, zorder=8)     # white-filled circle
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


# ── push quantification (verbatim from exp_nolick_push_stats.py) ────────────────
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


def lmm_paired(nai_A, exp_A, nai_B, exp_B, side):
    """Push test with sample A & B kept SEPARATE (2 obs/mouse) — LMM on the Expert−Naive differences with a
    per-mouse random intercept (honest about the A/B clustering; n=18 rather than the pooled n=9)."""
    dA, dB = exp_A - nai_A, exp_B - nai_B
    df = pd.DataFrame({'d': np.concatenate([dA, dB]), 'mouse': list(MICE) * 2}).dropna()
    try:
        r = smf.mixedlm('d ~ 1', df, groups=df['mouse']).fit(reml=False, method='lbfgs')
        beta, p2 = float(r.params['Intercept']), float(r.pvalues['Intercept']); how = 'LMM'
    except Exception:                                    # RE variance → 0 etc.: fall back to n=18 Wilcoxon
        p2 = wilcoxon(df['d']).pvalue; beta = float(df['d'].mean()); how = 'Wilc18'
    p1 = p2 / 2 if ((side == 'less' and beta < 0) or (side == 'greater' and beta > 0)) else 1 - p2 / 2
    return beta, p2, p1, len(df), how


def slopegraph_ab(ax, nai_A, exp_A, nai_B, exp_B, ylabel, title, side='less'):
    for nai, exp, c in [(nai_A, exp_A, COL[0]), (nai_B, exp_B, COL[1])]:      # sample A indigo, B teal
        keep = ~(np.isnan(nai) | np.isnan(exp))
        for n, e in zip(nai[keep], exp[keep]):
            ax.plot([0, 1], [n, e], '-', color=c, lw=1, alpha=0.45, zorder=4)
            ax.scatter([0, 1], [n, e], color=c, s=22, zorder=5, edgecolors='w', linewidths=0.5)
    allN = np.concatenate([nai_A, nai_B]); allE = np.concatenate([exp_A, exp_B])
    for x, v in [(0, allN), (1, allE)]:
        lo, hi = boot_ci(v); mu = np.nanmean(v)
        ax.errorbar(x + 0.10, mu, yerr=[[mu - lo], [hi - mu]], fmt='o', color='k', ms=7, capsize=4,
                    lw=1.6, zorder=6)
    beta, p2, p1, n, how = lmm_paired(nai_A, exp_A, nai_B, exp_B, side)
    ax.axhline(0, ls='--', color='k', lw=0.8)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=8)
    ax.set_title(f'{title}\nΔ={beta:+.2f}  {how} p={p2:.3f} (1s {p1:.3f})\n'
                 f'A,B separate  n={n} (9×2, 1|mouse)', fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(labelsize=7)
    mark = '***' if p1 < 1e-3 else '**' if p1 < 1e-2 else '*' if p1 < 5e-2 else 'ns'
    yt = np.concatenate([allN, allE]); yt = yt[~np.isnan(yt)]; ytop, ybot = float(yt.max()), float(yt.min())
    pad = 0.07 * (ytop - ybot + 1e-9); ybar = ytop + 1.4 * pad
    ax.plot([0, 0, 1, 1], [ybar - 0.5 * pad, ybar, ybar, ybar - 0.5 * pad], color='k', lw=1.0, clip_on=False)
    ax.text(0.5, ybar, mark, ha='center', va='bottom', fontweight='bold', fontsize=13 if mark != 'ns' else 9.5)
    ax.set_ylim(ybot - pad, ybar + 3.4 * pad)
    return beta


# ── build ──────────────────────────────────────────────────────────────────────
print(f'=== OVERLAPS story figure — TRAJECTORY FIT prototype  [mode={MODE}, input={args.input} '
      f'({"low-rank: b_r inside gain" if INSIDE else "external current c_r"})]  '
      f'({"correct" if CORRECT else "all-laser-off"}, panels={args.panels}) ===')
ST = {s: fit_stage(s, CORRECT) for s in STAGES}
REG = ST['Expert']['REG']
idx3 = list(range(8)) if args.panels == 8 else PANELS4

# push arrays (delay axis, correct/all, DPA)
mask = (y.laser == 0) if not CORRECT else (
    (y.laser == 0) & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1)))
dep_ch = depths(TRAIN_PUSH, mask, target='choice')     # choice-code push, A & B kept SEPARATE
chA_nai, chA_exp = per_mouse(dep_ch, 'Naive', 'A'), per_mouse(dep_ch, 'Expert', 'A')
chB_nai, chB_exp = per_mouse(dep_ch, 'Naive', 'B'), per_mouse(dep_ch, 'Expert', 'B')
pool_nai = np.nanmean([chA_nai, chB_nai], axis=0); pool_exp = np.nanmean([chA_exp, chB_exp], axis=0)  # for print
dep_sa = depths(TRAIN_PUSH, mask, target='sample')     # specificity control: sample-memory strength
saA_nai, saA_exp = -per_mouse(dep_sa, 'Naive', 'A'), -per_mouse(dep_sa, 'Expert', 'A')   # A codes − → flip +
saB_nai, saB_exp = per_mouse(dep_sa, 'Naive', 'B'), per_mouse(dep_sa, 'Expert', 'B')     # B codes +

# ── figure ──────────────────────────────────────────────────────────────────────
FH = 15.0 if args.panels == 8 else 9.0
fig = plt.figure(figsize=(19, FH))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0 if args.panels == 8 else 1.0, 1.0], hspace=0.30)

nc3 = 4; nr3 = (len(idx3) + nc3 - 1) // nc3
gsF = gs[0].subgridspec(nr3, nc3, hspace=0.32, wspace=0.12)
LIM3 = ST['Expert']['L']
axF = []
for k, r in enumerate(idx3):
    ax = fig.add_subplot(gsF[k // nc3, k % nc3]); axF.append(ax)
    draw_panel(ax, ST['Expert'], r, LIM3, baseline=(r == 0))
    ax.set_title(f'{REG[r][0]}   τR²={ST["Expert"]["isr2"][r]:+.2f}', fontsize=11, fontweight='bold')
    if k % nc3 == 0:
        ax.set_ylabel('choice code', fontsize=9)
    if k // nc3 == nr3 - 1:
        ax.set_xlabel('sample code', fontsize=9)

gsL = gs[1].subgridspec(1, 4, width_ratios=[1.1, 1.1, 0.62, 0.62], wspace=0.32)
axNai = fig.add_subplot(gsL[0]); axExp = fig.add_subplot(gsL[1])
axN1 = fig.add_subplot(gsL[2]); axN2 = fig.add_subplot(gsL[3])
LIM4 = max(panel_lim(ST['Naive']['allm'], [0]), panel_lim(ST['Expert']['allm'], [0]))
for ax, s in [(axNai, 'Naive'), (axExp, 'Expert')]:
    draw_panel(ax, ST[s], 0, LIM4, baseline=True)
    ax.set_title(f'{s}: DPA autonomous flow  (τR²={ST[s]["isr2"][0]:+.2f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('sample code', fontsize=9)
axNai.set_ylabel('choice code', fontsize=9)
slopegraph_ab(axN1, chA_nai, chA_exp, chB_nai, chB_exp,
              f'choice-code depth\n(late delay, {"correct" if CORRECT else "all"})',
              'push: choice code', side='less')
slopegraph_ab(axN2, saA_nai, saA_exp, saB_nai, saB_exp, 'sample-memory strength\n(A,B oriented +)',
              'control: sample sep.', side='greater')

for ax, L in [(axF[0], 'A'), (axNai, 'B'), (axExp, 'C'), (axN1, 'D'), (axN2, 'E')]:
    ax.text(-0.02, 1.06, L, transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom', ha='right')
fp_leg = [Line2D([0], [0], ls='', marker='o', mfc='white', mec='k', ms=8, mew=1.3, label='attractor'),
          Line2D([0], [0], ls='', marker='o', mfc='white', mec='0.55', ms=7, mew=1.3, label='saddle'),
          Line2D([0], [0], ls='', marker='o', mfc='white', mec='r', ms=7, mew=1.3, label='repeller'),
          Line2D([0], [0], color='c', ls='--', lw=1.1, label='lick baseline (choice=0)')]
tr_leg = [Line2D([0], [0], color=COL[0], lw=2.3, label='sample A'),
          Line2D([0], [0], color=COL[1], lw=2.3, label='sample B'),
          Line2D([0], [0], color=COL['DualGo'], lw=2.3, label='Go'),
          Line2D([0], [0], color=COL['DualNoGo'], lw=2.3, label='NoGo'),
          Line2D([0], [0], ls='', marker='o', mfc='w', mec='k', ms=6, label='traj. start'),
          Line2D([0], [0], ls='', marker='*', mfc='0.4', mec='k', ms=11, label='traj. end')]
fig.tight_layout(rect=(0, 0.05, 1, 0.99))

fig.canvas.draw()
def _flow_cbar(rect, label='flow speed |ż|'):
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='magma'); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes(rect))
    cb.set_ticks([0, 1]); cb.set_ticklabels(['slow', 'fast']); cb.ax.tick_params(labelsize=7)
    if label:
        cb.set_label(label, fontsize=8)
_r = max(a.get_position().x1 for a in axF)
_t = max(a.get_position().y1 for a in axF); _b = min(a.get_position().y0 for a in axF)
_flow_cbar([_r + 0.007, _b, 0.007, _t - _b])
_pe = axExp.get_position()
_flow_cbar([_pe.x1 + 0.008, _pe.y0, 0.007, _pe.height], label='')

_r2b = min(a.get_position().y0 for a in [axNai, axExp, axN1, axN2])          # bottom of the push row
fig.legend(handles=fp_leg + tr_leg, loc='upper center', ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, _r2b - 0.045))                              # legend hugs the last row

OUT = 'figures/overlaps/story_traj'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
stem = f'fig_overlaps_story_traj_{MODE}_{args.input}{TAG}' + ('' if args.panels == 8 else '_p4')
fig.savefig(f'{OUT}/png/{stem}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/svg/{stem}.svg', bbox_inches='tight')
print(f'\n§4 push (choice, delay, {"correct" if CORRECT else "all"}):  '
      f'Naive={np.nanmean(pool_nai):+.3f}  Expert={np.nanmean(pool_exp):+.3f}  '
      f'Δ={np.nanmean(pool_exp - pool_nai):+.3f}')
print(f'saved {OUT}/png/{stem}.png')
