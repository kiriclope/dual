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
coherent, dPCA-like input fields we use the dPCA §3 construction — PARTIAL POOLING BY DECODER GROUP: each
group of regimes sharing a decoder epoch pools one shared landscape A. FOUR groups: {autonomous,A,B}@DELAY,
{Go,NoGo}@RESP, {Cue}@CUE, {C,D}@TEST (the last on a mixed sample@TEST × choice@RESP plane). Each group adds
a ridge-shrunk per-regime deviation ΔA_r + a per-regime input current c_r, so each input rides its group's
common landscape (a single global A can't hold every bistability). Model selection is restricted to gains
whose shared autonomous flow stays bistable. Fixed points are root-found on the fitted field (★ attractor /
□ saddle / ✕ repeller), same as the dPCA figure. The figure says so.

Usage:
  cd /home/leon/dual/overlaps
  python fig_overlaps_story_main.py [--panels 8|4] [--all-trials] [--input inside|outside]
                                   [--mode partial|shared|independent] [--stability B]
    --input inside (default): low-rank RNN, drive b_r INSIDE the gain S(z)(A z + b_r) (closed-form: the
                              input column becomes S·b_r; the β_r variance term is omitted here).
    --input outside         : previous form, external additive current S(z)(A z) + c_r.
    --mode partial (default): per group, shared A_sh + ridge ΔA_r.  shared: one A_sh.  independent: per-regime A_r.
Saves figures/overlaps/story/{png,svg}/fig_overlaps_story_main_<mode>_<input>[_all].{png,svg}.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, argparse, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
from sklearn.model_selection import RepeatedKFold
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
ap.add_argument('--stability', type=int, default=0, metavar='B',
                help='mouse-subsample stability of the §3 flows (B resamples of 7/9 mice); prints, no figure')
ap.add_argument('--input', choices=['inside', 'outside'], default='inside',
                help='per-regime drive: inside (default, low-rank RNN) = S(z)(A z + b_r), b_r inside the '
                     'gain (closed-form: input column S·b_r; the β_r variance term is omitted here — the '
                     'trajectory figure fits it); outside (previous) = S(z)(A z) + c_r, external current')
ap.add_argument('--mode', choices=['partial', 'shared', 'independent'], default='partial',
                help='per-group recurrent-landscape pooling: partial = shared A_sh + ridge ΔA_r (default); '
                     'shared = ONE A_sh per group (ΔA_r≡0); independent = per-regime A_r (no shared, no ridge)')
args = ap.parse_args()
CORRECT = not args.all_trials
INSIDE = args.input == 'inside'
MODE = args.mode
TAG = '' if CORRECT else '_all'
MARGIN = 3                # calcium tail after event offset: 0.5 s @ 6 Hz (< MD, so no epoch bleed)
VSTEP = 1                 # adjacent-bin velocity (dPCA exact port)
W_ANCHOR = 20.0           # weight of the v=0 endpoint anchors (pin a fixed point where each trajectory settles)

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
WIN_MD, WIN_RWD = _win(o['bins_MD']), _win(o['bins_RWD'])   # Go/NoGo read over MD; Cue read over GNG reward
WIN_CD = np.arange(57, 63)          # C/D read: ±0.5 s straddling test→choice (t 9.5–10.5), where the diagonal forms
# DECODER (train epoch) per regime — independent of the READ window (REG below). A/B ride the DELAY
# decoder WITH autonomous (their sample wells sit on the WM bistable landscape; the choice axis is
# meaningless at stim, so a stim decoder gave A/B a garbage vertical position). C/D decoded at test;
# Go/NoGo on the RESPONSE (lick) axis, Cue on the CUE decoder — the lick code is barely driven early, so
# Go↑/NoGo↓ only read on the response decoder; READ windows: Go/NoGo over MD, Cue over the GNG reward (RWD).
# Regimes sharing a decoder pool ONE landscape and each such group CV-tunes its OWN ridge λ.
# each value = (sample_bins, choice_bins) for that regime's plane. C/D use a MIXED plane: sample @TEST
# (A neg / B pos — symmetric x separation; the delay decoder lets B fade to ~0) × choice/lick @RESPONSE
# (match AC/BD → +y, mismatch AD/BC → −y). Result: C = AC(top-left)/BC(bottom-right), D = AD(bottom-left)/
# BD(top-right) — the two opposite DIAGONALS. A single test decoder for both axes gives weak y (mismatch≈0).
EPOCHS = {'DELAY': (o['bins_DELAY'],), 'CUE': (o['bins_CUE'],),
          'TEST': (o['bins_TEST'], o['bins_CHOICE']), 'RESP': (o['bins_CHOICE'],)}
REG_EPOCH = {'autonomous': 'DELAY', 'A input': 'DELAY', 'B input': 'DELAY', 'Go input': 'RESP',
             'NoGo input': 'RESP', 'Cue input': 'CUE', 'C input': 'TEST', 'D input': 'TEST'}


def codes(stage, correct, sample_bins, choice_bins=None, mice=None):
    """Matched [sample, choice] plane; sample decoded at sample_bins, choice at choice_bins (default = same
    epoch). Per-mouse BL-std normed, ×2.8. Separate epochs let C/D read sample memory @delay × match @test.
    `mice` restricts to a subset (stability subsampling)."""
    choice_bins = sample_bins if choice_bins is None else choice_bins
    base = (y.laser == 0) & (y.learning == stage)
    if correct:                                          # canonical overlaps correct: DPA-correct AND
        base = base & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1))  # GNG-correct on Dual
    if mice is not None:                                 # mouse-subsample for the stability check
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
            ('A input', samp == 0, np.arange(12, 30), ('sample', (0,))),   # stim→early delay: well forms
            ('B input', samp == 1, np.arange(12, 30), ('sample', (1,))),
            ('Go input', go, WIN_MD, ('sample', (0, 1))),        # read Go over the memory delay (MD)
            ('NoGo input', nogo, WIN_MD, ('sample', (0, 1))),    # read NoGo over MD
            ('Cue input', go | nogo, WIN_RWD, ('tasks', ('DualGo', 'DualNoGo'))),   # read Cue over GNG reward
            ('C input', test == 0, WIN_CD, ('sample', (0, 1))),
            ('D input', test == 1, WIN_CD, ('sample', (0, 1)))]
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


def build_flow(A, inp, a, dd):
    """--input inside (low-rank RNN): ż = -z + S(z)·(A z + b), drive b INSIDE the gain (β variance term
    omitted in this closed-form fit).  --input outside: ż = -z + S(z)·(A z) + c, external current."""
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + dd; S = gd(D, np.zeros(P.shape[1])); AP = A @ P
        if INSIDE:
            return np.vstack([-P[0] + S * (AP[0] + inp[0]), -P[1] + S * (AP[1] + inp[1])])
        return np.vstack([-P[0] + S * AP[0] + inp[0], -P[1] + S * AP[1] + inp[1]])
    return fl


# ── POOLING by decoder epoch (dPCA §3 method) ──────────────────────────────────
# Regimes sharing a decoder (same plane) form a group: {autonomous,A,B}@DELAY, {Go,NoGo}@RESP, {Cue}@CUE,
# {C,D}@TEST.  --mode sets how the recurrent landscape is pooled within a group (fit_group):
#   partial (default): shared A_sh + ridge-λ per-regime ΔA_r  (A_r = A_sh + ΔA_r).
#   shared           : ONE A_sh per group (ΔA_r ≡ 0) — inputs differ only by their drive.
#   independent      : per-regime A_r, no shared term, no ridge.
# NOTE the CUE group is a SINGLE regime, so partial/shared/independent coincide there.
def groups_of(REG):
    g = {}
    for r in range(len(REG)):
        g.setdefault(REG_EPOCH[REG[r][0]], []).append(r)
    return list(g.values())


def fit_group(REG, means, a, dd, lam, subset):
    Z_, V_, R_, W_ = [], [], [], []
    for loc, r in enumerate(subset):
        z, v = zv_one(REG, means, r)
        if len(z):
            Z_.append(z); V_.append(v); R_.append(np.full(len(z), loc)); W_.append(np.ones(len(z)))
        for _, mu in means[r]:                        # anchor v=0 at the DRAWN trajectory end (last bin) so
            e = mu[:, REG[r][2]][:, -1]               #   the root-found fixed point sits exactly there
            Z_.append(e[None]); V_.append(np.zeros((1, 2))); R_.append(np.full(1, loc)); W_.append(np.full(1, W_ANCHOR))
    if not Z_:
        return {}
    z = np.concatenate(Z_); v = np.concatenate(V_); rid = np.concatenate(R_).astype(int)
    w = np.concatenate(W_); ng = len(subset)
    D = a ** 2 * (z ** 2).sum(1) + dd; S = gd(D, np.zeros(len(z)))
    OH = np.eye(ng)[rid]; shF = np.column_stack([S * z[:, 0], S * z[:, 1]])
    devF = (OH[:, :, None] * shF[:, None, :]).reshape(len(z), ng * 2)                  # per-regime S·z
    inpF = (S[:, None] * OH) if INSIDE else OH        # input cols: S·b_r (inside gain) vs c_r (external)
    Wt = np.sqrt(w)[:, None]                           # weighted LS — anchor rows carry weight W_ANCHOR
    A = np.zeros((ng, 2, 2)); C = np.zeros((ng, 2))
    if MODE == 'shared':                              # one A_sh per group + per-regime drive (no ΔA_r)
        F = np.column_stack([shF, inpF]) * Wt
        for d in (0, 1):
            cd = np.linalg.lstsq(F, (v[:, d] + z[:, d]) * Wt[:, 0], rcond=None)[0]
            A[:, d] = cd[0:2]; C[:, d] = cd[2:]       # same A_sh broadcast to every regime
    elif MODE == 'independent':                       # per-regime A_r + drive, no shared term, no ridge
        F = np.column_stack([devF, inpF]) * Wt
        for d in (0, 1):
            cd = np.linalg.lstsq(F, (v[:, d] + z[:, d]) * Wt[:, 0], rcond=None)[0]
            for loc in range(ng):
                A[loc, d] = cd[2 * loc:2 * loc + 2]
            C[:, d] = cd[2 * ng:]
    else:                                             # partial: shared A_sh + ridge-λ per-regime ΔA_r
        Fu = np.column_stack([shF, devF, inpF]); F = Fu * Wt; Pn = Fu.shape[1]
        reg = np.zeros((Pn, Pn)); reg[2:2 + 2 * ng, 2:2 + 2 * ng] = lam * np.eye(2 * ng)   # ridge ΔA only
        FtF = F.T @ F + reg
        for d in (0, 1):
            cd = np.linalg.solve(FtF, F.T @ ((v[:, d] + z[:, d]) * Wt[:, 0]))
            for loc in range(ng):
                A[loc, d] = cd[0:2] + cd[2 + 2 * loc:2 + 2 * loc + 2]                        # A_sh + ΔA_r
            C[:, d] = cd[2 + 2 * ng:]
    return {r: build_flow(A[loc], C[loc], a, dd) for loc, r in enumerate(subset)}


def fit_all(REG, means, a, dd, lam):
    flows = {}
    for subset in groups_of(REG):
        flows.update(fit_group(REG, means, a, dd, lam, subset))
    return flows


GRID = [(a, dd, lam) for a in (0.2, 0.4, 0.7, 1.0) for dd in (0.3, 0.8, 2.0)
        for lam in ((0.2, 1.0, 5.0, 20.0, 100.0) if MODE == 'partial' else (0.0,))]   # ridge only in partial


def fit_stage(stage, correct):
    planes = {}; yc = None
    for en in EPOCHS:
        Z2, yc = codes(stage, correct, *EPOCHS[en]); planes[en] = Z2   # one plane per decoder epoch
    REG = build_reg(yc); NREG = len(REG)
    allm = regime_means(REG, planes, yc, np.ones(len(yc), bool))
    span = np.concatenate([mu[:, REG[r][2]] for r in range(NREG) for _, mu in allm[r]], axis=1)
    L = 1.3 * max(float(np.abs(span).max()), 1.0)
    folds = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=0).split(np.arange(len(yc))))
    fold_data = []                                        # precompute per-fold train/test regime means ONCE
    for tr, te in folds:                                  #   (they don't depend on (a,δ,λ) — reuse across grid)
        trm = np.zeros(len(yc), bool); trm[tr] = True; tem = np.zeros(len(yc), bool); tem[te] = True
        fold_data.append((regime_means(REG, planes, yc, trm), regime_means(REG, planes, yc, tem)))

    # PER-GROUP ridge: each decoder group CV-tunes its OWN (a,δ,λ) so C/D keep a low ridge (bistable
    # diagonal) while the response group keeps Go↑/NoGo↓. Score = held-out velocity R², averaged over the
    # 50 splits of 10×5-fold repeated CV (more stable hyperparameter selection than a single 5-fold split).
    def cv_group(subset, a, dd, lam):
        rr = []
        for trm_m, tem_m in fold_data:
            fl = fit_group(REG, trm_m, a, dd, lam, subset); num = den = 0.0
            for r in subset:
                z, v = zv_one(REG, tem_m, r)
                if len(z) >= 3:
                    vp = fl[r](z.T).T; num += ((v - vp) ** 2).sum(); den += ((v - v.mean(0)) ** 2).sum()
            rr.append(1 - num / (den + 1e-12))
        return float(np.mean(rr))

    def auto_bistable(subset, p):                     # keep the WM autonomous landscape bistable
        fl = fit_group(REG, allm, *p, subset)
        return sum(k == 'attractor' for _, k, _ in flow_fixed_points(fl[0], [(-L, L), (-L, L)], n_seed=18)) >= 2

    flows = {}; cvbest = []; best_by = {}; cv_groups = []
    for g in groups_of(REG):
        cvs = {p: cv_group(g, *p) for p in GRID}
        if 0 in g:                                    # autonomous group → restrict to bistable configs
            bist = [p for p in GRID if auto_bistable(g, p)]
            best = max(bist or GRID, key=lambda p: cvs[p]); tag = f'  ({len(bist)}/{len(GRID)} bist)'
        else:
            best = max(GRID, key=lambda p: cvs[p]); tag = ''
        glabel = "/".join(REG[r][0].split()[0] for r in g)
        flows.update(fit_group(REG, allm, *best, g)); cvbest.append(cvs[best])
        cv_groups.append((glabel, cvs[best]))         # per-group held-out CV (kept unaveraged for the caption)
        best_by[REG_EPOCH[REG[g[0]][0]]] = best       # per decoder-group hyperparams (for stability refits)
        print(f'  {stage:6s} [{glabel:11s}] (a,δ,λ)={best}  vel-R²={cvs[best]:+.3f}{tag}')
    return dict(REG=REG, allm=allm, flows=flows, cv=float(np.mean(cvbest)), cv_groups=cv_groups, L=L, best=best_by)


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
        mec = {'attractor': 'k', 'saddle': '0.55', 'repeller': 'r'}.get(kind, 'k')     # edge = FP type
        ms = 10 if kind == 'attractor' else 8
        ax.plot(pt[0], pt[1], 'o', mfc='white', mec=mec, ms=ms, mew=1.3, zorder=8)     # white-filled circle
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
print(f'=== OVERLAPS story figure  (per-regime decoders; mode={MODE}; Go/NoGo on response, Cue on cue; '
      f'input={args.input} [{"low-rank: b_r inside gain" if INSIDE else "external current c_r"}]; '
      f'{"correct" if CORRECT else "all-laser-off"}, panels={args.panels}) ===')
ST = {s: fit_stage(s, CORRECT) for s in STAGES}
REG = ST['Expert']['REG']
idx3 = list(range(8)) if args.panels == 8 else PANELS4


def run_stability(B, K=7):
    """Subsample K of the 9 mice B times; refit the §3 flows at the full-data per-group hyperparams and
    record whether each qualitative feature survives. (§4 push already has a per-mouse bootstrap CI in D.)"""
    rng = np.random.default_rng(0); best = ST['Expert']['best']
    keys = ['auto bistable', 'A/B split (Ax<Bx)', 'Go↑', 'NoGo↓', 'C diag (AC>BC)', 'D diag (BD>AD)']
    hits = {k: [] for k in keys}
    for _ in range(B):
        sub = list(rng.choice(MICE, size=K, replace=False))
        planes = {en: codes('Expert', CORRECT, *EPOCHS[en], mice=sub)[0] for en in EPOCHS}
        yc = codes('Expert', CORRECT, *EPOCHS['DELAY'], mice=sub)[1]
        rg = build_reg(yc); allm = regime_means(rg, planes, yc, np.ones(len(yc), bool))
        span = np.concatenate([mu[:, rg[r][2]] for r in range(len(rg)) for _, mu in allm[r]], axis=1)
        L = 1.3 * max(float(np.abs(span).max()), 1.0)
        fl = {}
        for g in groups_of(rg):
            fl.update(fit_group(rg, allm, *best[REG_EPOCH[rg[g[0]][0]]], g))

        def end(r, lv, ax):                              # last-3-bin mean of regime r, level lv, on axis ax
            for l, mu in allm[r]:
                if l == lv:
                    return mu[ax, rg[r][2]][-3:].mean()
            return np.nan
        na = sum(k == 'attractor' for _, k, _ in flow_fixed_points(fl[0], [(-L, L), (-L, L)], n_seed=18))
        hits['auto bistable'].append(na >= 2)
        hits['A/B split (Ax<Bx)'].append(end(1, 0, 0) < end(2, 1, 0))
        hits['Go↑'].append(np.nanmean([end(3, 0, 1), end(3, 1, 1)]) > 0)
        hits['NoGo↓'].append(np.nanmean([end(4, 0, 1), end(4, 1, 1)]) < 0)
        hits['C diag (AC>BC)'].append(end(6, 0, 1) > end(6, 1, 1))
        hits['D diag (BD>AD)'].append(end(7, 1, 1) > end(7, 0, 1))
    print(f'\n=== §3 flow stability — {B}× subsample {K}/{len(MICE)} mice (Expert, {"correct" if CORRECT else "all"}, '
          f'fixed full-data hyperparams) ===')
    for k in keys:
        print(f'  {k:20s}: {100 * np.nanmean(hits[k]):5.1f}%  of subsamples')


if args.stability:
    run_stability(args.stability)
    sys.exit(0)

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

# Row 1 — §3 Expert flow grid
nc3 = 4; nr3 = (len(idx3) + nc3 - 1) // nc3
gsF = gs[0].subgridspec(nr3, nc3, hspace=0.32, wspace=0.12)
LIM3 = ST['Expert']['L']
axF = []
for k, r in enumerate(idx3):
    ax = fig.add_subplot(gsF[k // nc3, k % nc3]); axF.append(ax)
    draw_panel(ax, ST['Expert'], r, LIM3, baseline=(r == 0))
    ax.set_title(REG[r][0], fontsize=11, fontweight='bold')
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
    ax.set_title(f'{s}: DPA autonomous flow', fontsize=10, fontweight='bold')
    ax.set_xlabel('sample code', fontsize=9)
axNai.set_ylabel('choice code', fontsize=9)
slopegraph_ab(axN1, chA_nai, chA_exp, chB_nai, chB_exp,
              f'choice-code depth\n(late delay, {"correct" if CORRECT else "all"})',
              'push: choice code', side='less')
slopegraph_ab(axN2, saA_nai, saA_exp, saB_nai, saB_exp, 'sample-memory strength\n(A,B oriented +)',
              'control: sample sep.', side='greater')

# panel letters + legend + caption
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

_r2b = min(a.get_position().y0 for a in [axNai, axExp, axN1, axN2])          # bottom of the push row
fig.legend(handles=fp_leg + tr_leg, loc='upper center', ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, _r2b - 0.045))                              # legend hugs the last row

OUT = 'figures/overlaps/story'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
stem = f'fig_overlaps_story_main_{MODE}_{args.input}{TAG}' + ('' if args.panels == 8 else '_p4')
fig.savefig(f'{OUT}/png/{stem}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/svg/{stem}.svg', bbox_inches='tight')
print(f'\n§4 push (choice, delay, {"correct" if CORRECT else "all"}):  '
      f'Naive={np.nanmean(pool_nai):+.3f}  Expert={np.nanmean(pool_exp):+.3f}  '
      f'Δ={np.nanmean(pool_exp - pool_nai):+.3f}')
print(f'saved {OUT}/png/{stem}.png')
