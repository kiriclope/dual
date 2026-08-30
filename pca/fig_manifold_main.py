"""fig_manifold_main.py — Fig 3 CANDIDATE (2026-08-12): ONE manifold, seen in a fixed frame.

Message: the states of both tasks live in a single frame built from a few near-orthogonal, abstract
axes — and the ONLY pair that shares geometry is distractor x action, whose overlap GROWS with
learning. That is the bridge into Fig 4: learning does not add a dimension, it moves the state along
the action axis (and drags the distractor code toward it).

  a  the frame itself — held-out pseudo-trials plotted IN the fixed axes (x = sample axis @ mid-delay,
     y = behavioural lick axis @ decision, both trained on an independent trial half). DPA | dual x
     mid-delay | decision. Metric, unlike a t-SNE map: every offset is in z units.
  b  axis geometry — attenuation-corrected |cos| between sample / action / distractor (AXIS_FRAME),
     Naive vs Expert. sample is orthogonal to both; action x distractor is 0.39 -> 0.53.
  c  the shared ACTION code — Go/NoGo <-> DPA-lick cross-decoding (ACT_Mms) with the off/within
     ratio + CI. High transfer is what a d'~6 code with |cos|~0.4 predicts; the ratio's CI excludes 1,
     so the two codes are NOT interchangeable.
  d  abstraction is not built by learning — per-mouse CCGP, Naive vs Expert (permouse_ccgp_cache).

Reads caches only (no 20 GB X, no overlaps tensor): pca results.pkl + fits_inputs.pkl, and the
overlaps ccgp caches by absolute path.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_manifold_main.py
Output: figures/pseudo/dimensionality/{png,svg}/fig_manifold_main.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import wilcoxon
from decoders import fit_axis, SUF, NOPCA          # THE shared decoder (see decoders.py)
FIGSUF = '' if NOPCA else '_pca20'                 # filename: plain = no denoising

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
SAMPC = {0: '#332288', 1: '#44AA99'}
SC = {'Naive': '0.55', 'Expert': '#332288'}
VAR_COL = {'sample': '#332288', 'test': '#377eb8', 'choice': '#4daf4a',
           'tasks': '#cc3311', 'gng': '#ee7733'}          # house palette, as in Fig 2
# ONE name per code across every panel. The four task variables are sample / dist / test / choice;
# "lick", "action" and "GNG" are the legacy aliases that used to appear panel-to-panel.
CODE_ORDER = ['sample', 'dist', 'test', 'choice']
CODE_NAME = {'sample': 'sample', 'GNG': 'dist', 'gng': 'dist', 'distractor': 'dist',
             'test': 'test', 'lick': 'choice', 'action': 'choice', 'choice': 'choice'}
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
STAGE = 'Expert'
KP = 30
OVL = '/home/leon/dual/overlaps/figures/overlaps/ccgp'
# the pooled panels (c, d) and panel e come from the overlaps caches; pick the variant that MATCHES
# this figure's decoder, so a --nopca figure is not silently half PCA-fitted.
MAT_CACHE = f'{OVL}/matrices_cache_acc{SUF}.pkl'          # SUF = '' | '_nopca'
CCGP_CACHE = f'{OVL}/permouse_ccgp_cache{"" if NOPCA else "_pca"}.pkl'
for _f in (MAT_CACHE, CCGP_CACHE):
    assert os.path.exists(_f), (f'missing {_f} — run the matching overlaps analysis:\n'
                                '  cd ../overlaps && python fig_ccgp_matrices_pseudo.py --acc [--nopca]\n'
                                '  cd ../overlaps && python fig_ccgp.py [--pca]')

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
AXF = RES['AXIS_FRAME' + SUF]
_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)          # BEHAVIOURAL lick (error trials included)


def plabel(ax, s, dx=-0.06):
    ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')


# ══ a — the frame: held-out trials plotted IN the fixed axes ══════════════════
def neuron_scale(M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, STAGE)]
        tr = np.where((MOUSE == m) & (LEARN == STAGE) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def make_halves(rng):
    """half 0 trains the axes, half 1 is plotted — no self-inclusion leakage."""
    H = {}
    for m in MICE:
        for ci, (t, s, te) in enumerate(ALL12):
            for pf in (0, 1):
                idx = np.where((MOUSE == m) & (LEARN == STAGE) & (LAS == 0) & (PERF == pf)
                               & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
                p = rng.permutation(idx); h = len(p) // 2
                H[(m, ci, pf)] = (p[:h], p[h:])
    return H


def _pseudo(M, sd, pools, K, rng):
    """Composite pseudo-trials: one random trial per mouse per pseudo-trial, each mouse filling its
    own neurons. `pools` maps mouse -> trial indices. Returns (K, N) in sd-scaled units."""
    X = np.zeros((K, N))
    for m, idx in pools.items():
        if not len(idx):
            continue
        val = VALIDIX[(m, STAGE)]
        cm = np.nanmean(M[np.ix_(idx, val)], 0)
        blk = M[np.ix_(rng.choice(idx, K, replace=True), val)]
        bad = ~np.isfinite(blk)
        if bad.any():
            blk[bad] = np.broadcast_to(cm, blk.shape)[bad]
        X[:, val] = blk
    return X / sd[None, :]


def _axis_and_origin(Xpos, Xneg):
    """Fit the shared decoder on two class blocks and return (unit axis, ORIGIN).

    The origin is the training-set midpoint between the two class means along the axis — i.e. the
    decision boundary. Subtracting it (instead of each panel's own mean) is what lets the panels be
    compared: a state's SIGN then says which side of the lick/no-lick boundary it sits on, which is
    the quantity Fig 4 measures. Per-panel centring destroys exactly that (the push is a grand-mean
    translation), which is why an earlier version of this panel erased it."""
    w, _ = fit_axis(np.vstack([Xpos, Xneg]), np.r_[np.ones(len(Xpos), int), np.zeros(len(Xneg), int)])
    return w, 0.5 * ((Xpos @ w).mean() + (Xneg @ w).mean())


def sample_axis(sd, H, K=60):
    """ONE sample axis for the whole figure (all 12 conditions), fit on half-0 pseudo-trials @ md."""
    M = AW['md']; rng = np.random.RandomState(3)
    blocks = {0: [], 1: []}
    for ci, cd in enumerate(ALL12):
        pools = {m: H[(m, ci, 1)][0] for m in MICE}
        blocks[cd[1]].append(_pseudo(M, sd, pools, K, rng))
    return _axis_and_origin(np.vstack(blocks[1]), np.vstack(blocks[0]))


def test_axis(sd, H, K=60):
    """ONE test axis (Odor C vs D) for the trajectory row, fit on half-0 pseudo-trials @ the TEST
    window — the 4th code of the original Fig-3 panel A."""
    M = AW['test']; rng = np.random.RandomState(5)
    blocks = {0: [], 1: []}
    for ci, cd in enumerate(ALL12):
        pools = {m: H[(m, ci, 1)][0] for m in MICE}
        blocks[cd[2]].append(_pseudo(M, sd, pools, K, rng))
    return _axis_and_origin(np.vstack(blocks[1]), np.vstack(blocks[0]))


def lick_axis(sd, H, K=60):
    """ONE action axis for the whole figure: BEHAVIOURAL lick split (error trials included),
    all tasks, fit on half-0 trials @ decision."""
    M = AW['decision']; rng = np.random.RandomState(4)
    out = []
    for lickval in (True, False):
        pools = {}
        for m in MICE:
            pool = [H[(m, ci, pf)][0] for ci in range(len(ALL12)) for pf in (0, 1)]
            idx = np.concatenate(pool) if pool else np.array([], int)
            pools[m] = idx[LICK[idx] == lickval]
        out.append(_pseudo(M, sd, pools, K, rng))
    return _axis_and_origin(out[0], out[1])


def cloud(sname, wn, sd, H, rng):
    conds = SETS[sname]; M = AW[wn]
    X = np.zeros((len(conds) * KP, N)); crow = np.repeat(np.arange(len(conds)), KP)
    for ci_l, cd in enumerate(conds):
        ci = ALL12.index(cd)
        for m in MICE:
            val = VALIDIX[(m, STAGE)]; idx = H[(m, ci, 1)][1]
            if not len(idx):
                continue
            cm = np.nanmean(M[np.ix_(idx, val)], 0)
            block = M[np.ix_(rng.choice(idx, KP, replace=True), val)]
            bad = ~np.isfinite(block)
            if bad.any():
                block[bad] = np.broadcast_to(cm, block.shape)[bad]
            X[ci_l * KP:(ci_l + 1) * KP, val] = block
    return X / sd[None, :], crow, conds


TASKM = {'DPA': 'o', 'DualGo': '^', 'DualNoGo': 's'}


def build_frame():
    """The ONE frame used by both the trajectory row and the state scatters: neuron scale, the
    trial-half split, the two axes with their decision boundaries, and the BASELINE origin.

    Panels a and b must share BOTH the units and the zero, or the same axis name means two different
    coordinates. Both now use the pooled whole-population projection with the pre-trial baseline as
    zero; the decision boundary is DRAWN (a line) in each rather than being the origin of one of
    them."""
    sd = neuron_scale(AW['delay+dec'])
    H = make_halves(np.random.RandomState(7))
    (w_s, b_s), (w_l, b_l) = sample_axis(sd, H), lick_axis(sd, H)
    (w_t, b_t) = test_axis(sd, H)
    CM = np.asarray(RES['CMBIN'][STAGE], dtype=float)
    BL = {}
    for key, w in (('s', w_s), ('l', w_l), ('t', w_t)):
        v = np.zeros(CM.shape[2])
        for m in MICE:
            val = VALIDIX[(m, STAGE)]
            if len(val):
                v = v + w[val] @ (CM[:, val, :] / sd[val][None, :, None]).mean(0)
        BL[key] = float(v[BINS_BL].mean())
    return sd, H, (w_s, b_s), (w_l, b_l), (w_t, b_t), BL


def panel_a(fig, gsA, sd, H, w_s, b_s, w_l, b_l, BL):
    """COLUMN 1: the frame states stacked (DPA|dual x window)."""
    axes = []
    # LATE DELAY is essential for the dual set: it is the only plotted window AFTER the Go/NoGo cue
    # (6.5-7 s) and its lick, and it is where the Go state actually crosses into the action region
    # (Go +2.8 vs NoGo -5.0). Mid-delay alone is PRE-cue, so Go is still short of the boundary there.
    SPECS = [('DPA', 'md'), ('DPA', 'decision'),
             ('dual', 'md'), ('dual', 'delay'), ('dual', 'decision')]
    # PRECOMPUTE all four so they can share one x/y range. Autoscaling each panel would push the
    # y=0 boundary off-screen at mid-delay (the states sit ~5 z below it) and hide the very thing
    # the fixed origin exists to show.
    DATA = []
    for sname, wn in SPECS:
        X, crow, conds = cloud(sname, wn, sd, H, np.random.RandomState(2))
        DATA.append((X @ w_s - BL['s'], X @ w_l - BL['l'], crow, conds))
    ALLX = np.concatenate([d[0] for d in DATA]); ALLY = np.concatenate([d[1] for d in DATA])
    pad = 0.06 * (ALLX.max() - ALLX.min())
    XL = (min(ALLX.min(), 0) - pad, max(ALLX.max(), 0) + pad)
    padY = 0.06 * (ALLY.max() - ALLY.min())
    YL = (min(ALLY.min(), 0) - padY, max(ALLY.max(), 0) + padY)
    for j, (sname, wn) in enumerate(SPECS):
        ax = fig.add_subplot(gsA[j, 0]); axes.append(ax)     # stacked, not side-by-side
        xs, ys, crow, conds = DATA[j]
        # baseline only — the decision-boundary lines were removed (they invited being read as zero)
        ax.axhline(0, ls='--', color='k', lw=0.5, zorder=0)
        ax.axvline(0, ls='--', color='k', lw=0.4, zorder=0)
        for ci, cd in enumerate(conds):
            sel = crow == ci; col = SAMPC[cd[1]]; licks = cd[1] == cd[2]
            ax.scatter(xs[sel], ys[sel], s=3, marker='.', color=col, lw=0, alpha=0.18, zorder=1)
            C = np.cov(xs[sel], ys[sel]); ev, evec = np.linalg.eigh(C)
            ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
            ax.add_patch(Ellipse((xs[sel].mean(), ys[sel].mean()), 2 * np.sqrt(ev[-1]),
                                 2 * np.sqrt(ev[0]), angle=ang, fc=col, alpha=0.10, ec=col,
                                 lw=0.9, ls='-' if licks else '--', zorder=2))
            ax.scatter(xs[sel].mean(), ys[sel].mean(), marker=TASKM[cd[0]], s=30,
                       facecolor=col if licks else 'w', edgecolor=col, linewidths=1.0, zorder=5)
        ax.set_xlim(*XL); ax.set_ylim(*YL)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        sx = x0 + 0.05 * (x1 - x0); sy = y0 + 0.06 * (y1 - y0)
        ax.plot([sx, sx + 5], [sy, sy], '-', color='0.3', lw=1.1)
        ax.text(sx + 2.5, sy + 0.015 * (y1 - y0), '5 z', ha='center', va='bottom',
                fontsize=5.4, color='0.3')
        WLAB = {'md': 'mid-delay (pre-cue)', 'delay': 'late delay (post-lick)', 'decision': 'decision'}
        ax.set_title(f'{sname} · {WLAB[wn]}', loc='left', fontsize=TITLE_FS)
        if j == len(SPECS) - 1:
            ax.set_xlabel('sample axis   A ← · → B', fontsize=7)
        ax.set_ylabel('choice axis\n← no-lick · lick →', fontsize=7)
        if j == 0:                                   # colour/fill key once, on the top panel
            hs = [mlines.Line2D([], [], marker='o', ls='', ms=4, color=SAMPC[0], label='sample A'),
                  mlines.Line2D([], [], marker='o', ls='', ms=4, color=SAMPC[1], label='sample B'),
                  mlines.Line2D([], [], marker='o', ls='', ms=4, mfc='0.4', mec='0.4', label='lick'),
                  mlines.Line2D([], [], marker='o', ls='', ms=4, mfc='w', mec='0.4', label='no-lick')]
            ax.legend(handles=hs, frameon=False, fontsize=5.4, loc='upper left', ncols=2,
                      handletextpad=0.15, columnspacing=0.5, labelspacing=0.25, borderaxespad=0.0)
        if j == 2:                                    # marker key on the first dual panel
            hs = [mlines.Line2D([], [], marker='^', ls='', ms=4, mfc='none', mec='0.3', label='Go'),
                  mlines.Line2D([], [], marker='s', ls='', ms=4, mfc='none', mec='0.3', label='NoGo')]
            ax.legend(handles=hs, frameon=False, fontsize=5.4, loc='upper left', ncols=2,
                      handletextpad=0.15, columnspacing=0.5, borderaxespad=0.0)
        lk = np.array([conds[c][1] == conds[c][2] for c in crow])
        print(f'a: {sname:4s} {wn:9s} action-axis mean {ys.mean():+6.2f} '
              f'(lick {ys[lk].mean():+6.2f} / no-lick {ys[~lk].mean():+6.2f})  '
              f'sample sep {xs[[conds[c][1] == 1 for c in crow]].mean() - xs[[conds[c][1] == 0 for c in crow]].mean():+.2f}')
    return axes[0]


# ══ a — TRAJECTORIES on the two axes of the frame (top row) ═══════════════════
#   Uses CMBIN (per-bin condition means, 12 x 3319 x 84 per stage, cached by exp_antact_traj.py),
#   projected on the SAME two axes and the SAME origins as the scatters below, so the trajectory
#   row and the state row are literally the same coordinates. t = bin/6 - 0.5 s (exact).
TBIN = lambda b: np.asarray(b) / 6.0 - 0.5
EVENTS = [('sample', 2.0, 3.0, SAMPC[0]), ('distractor', 4.5, 5.5, '#cc3311'),
          ('GNG cue', 6.5, 7.5, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]


BINS_BL = np.arange(0, 12)          # baseline bins (0-11), shared_data.md convention
# ONE PANEL PER CODE with its own two-class colour pair — the original Fig-3 panel-A convention
# (overlaps/main_panels.py:486-491): sample A/B indigo/teal · no-lick/lick blue/green ·
# NoGo/Go green/blue. Each trace is BASELINE-CENTRED (its own 0-11 bin mean subtracted) so 0 = the
# pre-trial state and the traces are readable; the faint grey line marks where the decision
# boundary sits in those units, so "crossed into the action region" is still legible.
TRAJ_SPECS = [
    ('sample code', 'mem', 'DPA', lambda c: c[1], ['Odor A', 'Odor B'], ['#332288', '#44AA99']),
    ('lick code (DPA)', 'act', 'DPA', lambda c: int(c[1] == c[2]), ['No lick', 'Lick'],
     ['#377eb8', '#4daf4a']),
    ('test code', 'tst', 'DPA', lambda c: c[2], ['Odor C', 'Odor D'], ['#CC6677', '#999933']),
    ('GNG code (dual)', 'act', 'dual', lambda c: int(c[0] == 'DualGo'), ['NoGo', 'Go'],
     ['#2ca02c', '#1f77b4']),
]


def panel_traj(fig, gsT):
    """2 rows (Naive | Expert) x 4 codes — THE ORIGINAL Fig-3 panel A, replayed from ORIG_TRACES
    (`exp_traj_orig.py`, which extracts overlaps/main_panels' validated per-mouse CCGD projections).

    These are NOT re-derived here. Fitting fresh axes from the cached window matrices gave axes
    contaminated by the trial's global ramp (29-45% of the code size), which dragged both classes
    off zero and made the traces asymmetric; the overlaps decoders (per mouse, per stage, ridge,
    bootstrapped, lick axis orthogonalised to sample) do not have that problem. Units are the
    overlaps per-mouse shared unit, so amplitudes are comparable ACROSS CODES within this panel —
    but they are NOT the pseudo-population z of panel b, which answers a different question
    (position in the frame, not code depth).
    """
    # panel a must come from the SAME decoder variant as the rest of the figure: the default
    # figure uses PCA(20) everywhere, so it reads the traces from the _pca20 overlaps tensor
    # (run_overlaps.py --pca 20); --nopca reads the un-denoised one.
    TKEY = 'ORIG_TRACES' if NOPCA else 'ORIG_TRACES_pca20'
    assert TKEY in RES, (f'missing {TKEY} — build it with:\n'
                         '  cd ../overlaps && python run_overlaps.py --scaler none --save-weights '
                         '--targets sample choice gng test --pca 20\n'
                         '  cd ../pca && python exp_traj_orig.py --pca')
    TR = RES[TKEY]; xt = np.asarray(RES['ORIG_XTIME'])
    SP = sorted(RES['ORIG_SPECS'], key=lambda sp: CODE_ORDER.index(CODE_NAME[sp['code']]))
    axes = []
    for r, stage in enumerate(['Naive', 'Expert']):
        for k, spec in enumerate(SP):
            ax = fig.add_subplot(gsT[r, k]); axes.append(ax)
            for nm, lo, hi, col in EVENTS:
                ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0)
                if r == 0 and k == 0:
                    ax.text((lo + hi) / 2, 0.98, nm, transform=ax.get_xaxis_transform(),
                            ha='center', va='top', fontsize=5.2, color=col)
            for lv, lab, col in zip(spec['levels'], spec['labels'], spec['colors']):
                M = np.asarray(TR[(stage, spec['code'], int(lv))], dtype=float)
                if not len(M):
                    continue
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                ax.plot(xt, mu, color=col, lw=1.5, label=f'{lab} (n={len(M)})', zorder=3)
                ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.20, lw=0, zorder=2)
                print(f"traj {stage:6s} {spec['code']:7s} {lab:8s} peak {mu.max():+5.2f} "
                      f"trough {mu.min():+5.2f}")
            ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
            ax.set_xlim(0, 14); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
            if r == 1:
                ax.set_xlabel('time (s)', fontsize=7)
            else:
                ax.tick_params(labelbottom=False)
                ax.set_title(f"{CODE_NAME[spec['code']]} code", loc='left', fontsize=TITLE_FS)
                ax.legend(frameon=False, fontsize=5.4, handlelength=1.2,
                          loc='lower right' if k == 0 else 'upper left')
            ax.set_ylabel(f'{stage}\ncode depth' if k == 0 else 'code depth', fontsize=7)
    return axes[0]


# ══ b — the frame's geometry: corrected |cos| between the three axes ══════════
def panel_b(fig, gsB):
    labs = ['sample', 'choice', 'dist']
    axes = []
    for j, stage in enumerate(['Naive', 'Expert']):
        ax = fig.add_subplot(gsB[0, j]); axes.append(ax)
        C = np.asarray(AXF[stage]['cos'])
        Cm = C.copy(); np.fill_diagonal(Cm, np.nan)
        ax.imshow(np.ma.masked_invalid(Cm), cmap='Oranges', vmin=0, vmax=1, aspect='equal')
        for i in range(3):
            for k in range(3):
                if i == k:                              # 1 by construction — shown, greyed
                    ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, fc='0.93', ec='none'))
                    ax.text(k, i, '1', ha='center', va='center', fontsize=6.4, color='0.45')
                    continue
                ax.text(k, i, f'{C[i, k]:.2f}', ha='center', va='center', fontsize=6.4,
                        color='w' if C[i, k] > 0.55 else 'k')
        ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.2, rotation=35, ha='right')
        ax.set_yticks(range(3))
        ax.set_yticklabels(labs if j == 0 else [], fontsize=6.2)
        ax.set_title(stage, loc='left', fontsize=TITLE_FS)
        if j == 0:
            ax.set_ylabel('axis geometry\n|cos|', fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'b: {stage} sample-action {C[0,1]:.2f}  sample-distr {C[0,2]:.2f}  '
              f'action-distr {C[1,2]:.2f}  (reliab {np.round(AXF[stage]["rel"],2)})')
    return axes[0]


# ══ c — the shared action code: Go/NoGo <-> DPA-lick cross-decoding ═══════════
def panel_c(fig, gsC):
    CC = pickle.load(open(MAT_CACHE, 'rb'))
    axes = []
    for j, stage in enumerate(['Naive', 'Expert']):
        ax = fig.add_subplot(gsC[0, j]); axes.append(ax)
        M = np.asarray(CC['ACT_Mms'][stage])
        ax.imshow(M, cmap='Reds', vmin=0.5, vmax=1.0, aspect='equal')
        for i in range(2):
            for k in range(2):
                ax.text(k, i, f'{M[i, k]:.2f}', ha='center', va='center', fontsize=6.6,
                        color='w' if M[i, k] > 0.82 else 'k')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['dist', 'choice'], fontsize=5.8)  # rot=0: the ratio
        ax.set_yticks([0, 1])                                                     # text sits below
        ax.set_yticklabels(['dist', 'choice'] if j == 0 else [], fontsize=5.8)
        ax.set_title(stage, loc='left', fontsize=TITLE_FS)
        if j == 0:
            ax.set_ylabel('dist ↔ choice\ncross-dec. (bal. acc.)', fontsize=7)
        S = CC['ACT_SUMM'][stage]
        ax.text(0.5, -0.26, f"off/within {S['offdiag']:.2f}\n[{S['offdiag_lo']:.2f}, {S['offdiag_hi']:.2f}]",
                transform=ax.transAxes, ha='center', va='top', fontsize=5.6, color='0.3')
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'c: {stage} off/within {S["offdiag"]:.2f} [{S["offdiag_lo"]:.2f},{S["offdiag_hi"]:.2f}]')
    return axes[0]


# ══ d — the codes are REUSED: decoders trained in one task work in the others ══
#   OFF-DIAGONAL = Nms = (acc - 0.5) / (within-task acc of the TEST task - 0.5): the fraction of the
#   test task's OWN decodable signal that transfers. Normalising by COLUMN is essential — the within-
#   task sample code is 0.88 in DPA but 0.58/0.65 in Go/NoGo, so a raw cross value of 0.53 is 90% of
#   what is achievable there, NOT a failure. Reading the raw heatmap against a flat 0.5 chance line
#   produced a spurious "the memory code does not transfer out of DPA" (retracted 2026-08-12).
#   DIAGONAL = the raw within-task accuracy (the ceiling itself), greyed to mark the different unit.
#   Expert; Naive -> ED. The GNG block is OFF: it saturates at 0.99-1.00 in BOTH stages.
def panel_gen(fig, gsG):
    CC = pickle.load(open(MAT_CACHE, 'rb'))
    TL = list(CC['TLAB'])
    axes = []
    for j, var in enumerate(['sample', 'choice', 'test']):
        ax = fig.add_subplot(gsG[0, j]); axes.append(ax)
        M = np.asarray(CC['Mms'][('Expert', var)])
        Nn = np.asarray(CC['Nms'][('Expert', var)])
        disp = Nn.copy(); np.fill_diagonal(disp, np.nan)
        ax.imshow(np.ma.masked_invalid(disp), cmap='Reds', vmin=0, vmax=1, aspect='equal')
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if i == k:                                  # fully normalised: diagonal is 1 by
                    ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, fc='0.93', ec='none'))
                    ax.text(k, i, '1', ha='center', va='center', fontsize=5.8, color='0.45')
                else:
                    ax.text(k, i, f'{Nn[i, k]:.2f}', ha='center', va='center', fontsize=5.8,
                            color='w' if Nn[i, k] > 0.6 else 'k')
        # A ratio is only meaningful if its column has a ceiling to divide by: when the TEST task's
        # own within-task accuracy is barely above chance the denominator is ~0 and the ratio
        # explodes (test: within 0.57-0.59 -> Nms 1.5-1.9, which is a denominator artefact, NOT
        # transfer better than within-task). Hatch those columns instead of letting them read as
        # the strongest cells on the panel.
        weak = (np.diag(M) - 0.5) < 0.10
        for k in np.where(weak)[0]:
            for i in range(M.shape[0]):
                if i != k:
                    ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, fill=False, hatch='////',
                                           edgecolor='0.45', lw=0.0, zorder=3))
        # the ceiling now rides in the COLUMN label, so the matrix itself is purely normalised
        ax.set_xticks(range(len(TL)))
        ax.set_xticklabels([f'{t}\n{d:.2f}' for t, d in zip(TL, np.diag(M))],
                           fontsize=5.6, rotation=35, ha='right')
        ax.set_yticks(range(len(TL)))
        ax.set_yticklabels(TL if j == 0 else [], fontsize=6.0)
        ax.set_title(var, loc='left', fontsize=7)
        if j == 0:
            ax.set_ylabel('train', fontsize=7)      # column labels carry each test task's ceiling
        for sp in ax.spines.values():
            sp.set_visible(True)
        EYE = np.eye(len(M), dtype=bool)
        print(f'gen: {var:7s} Expert within {np.round(np.diag(M),2)}  transferred frac '
              f'{np.round(Nn[~EYE], 2)}  mean {Nn[~EYE].mean():.2f}')
    return axes[0]


# ══ e — abstraction present from the start: per-mouse CCGP Naive vs Expert ════
#   Conventions COPIED from the original Fig 3e (overlaps/fig_overlaps_manifold.py draw_scatters):
#   colour = mouse, marker = opsin group, white edge, black diamond = mean, dashed unity, Δ and p.
#   The three constants are replicated here rather than imported: `import main_panels` would load the
#   ~1.9 GB overlaps tensor at import time, and this figure is otherwise cache-only.
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
GROUP = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
         **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
_pal = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal[i] for i, m in enumerate(ALL_MICE)}


def _mouse_scatter(ax, nv, ev, lo, hi, xlab, ylab, title):
    """The house per-mouse idiom (same as panel e): colour = mouse, marker = opsin group,
    black diamond = mean, dashed unity, Wilcoxon across the 9 animals."""
    ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
    for m in ALL_MICE:
        if m in nv and m in ev:
            ax.scatter(nv[m], ev[m], s=34, color=MOUSE_COLOR[m], marker=GMARKER[GROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
    n = np.array([nv[m] for m in ALL_MICE if m in nv and m in ev])
    e = np.array([ev[m] for m in ALL_MICE if m in nv and m in ev])
    ax.scatter(n.mean(), e.mean(), s=80, color='k', marker='D', edgecolors='w',
               linewidths=0.6, zorder=5)
    p = float(wilcoxon(e, n).pvalue)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlab, fontsize=7); ax.set_ylabel(ylab, fontsize=7)
    ax.set_title(f'{title}  ({"∗" if p < .05 else "n.s."})', loc='left', fontsize=TITLE_FS)
    ax.text(0.05, 0.96, f'Δ={e.mean() - n.mean():+.2f}\np={p:.3f}', transform=ax.transAxes,
            va='top', ha='left', fontsize=6, color='0.3')
    return p


def panel_b_mouse(fig, gsX):
    """b companion: per-mouse |cos|(action, distractor), Naive vs Expert. The pooled panel says this
    overlap grows; this asks whether it grows WITHIN animals (mice = the exchangeable unit)."""
    PC = RES['PM_COS' + SUF]
    ax = fig.add_subplot(gsX[0, 0])
    nv = {m: PC[(m, 'Naive')]['ad'] for m in ALL_MICE if (m, 'Naive') in PC}
    ev = {m: PC[(m, 'Expert')]['ad'] for m in ALL_MICE if (m, 'Expert') in PC}
    p = _mouse_scatter(ax, nv, ev, 0.0, 1.0, '|cos| Naive', '|cos| Expert', 'choice × dist')
    print(f"b': per-mouse |cos|(action,distr) {np.mean(list(nv.values())):.2f} -> "
          f"{np.mean(list(ev.values())):.2f}  p={p:.3f}")
    return ax


def panel_c_mouse(fig, gsX):
    """c companion: per-mouse gng<->lick cross-decoding (mean of both directions)."""
    PA = RES['PM_ACT' + SUF]
    ax = fig.add_subplot(gsX[0, 0])
    nv = {m: np.nanmean([PA[(m, 'Naive')]['g2l'], PA[(m, 'Naive')]['l2g']])
          for m in ALL_MICE if (m, 'Naive') in PA}
    ev = {m: np.nanmean([PA[(m, 'Expert')]['g2l'], PA[(m, 'Expert')]['l2g']])
          for m in ALL_MICE if (m, 'Expert') in PA}
    p = _mouse_scatter(ax, nv, ev, 0.40, 0.85, 'cross-dec. Naive', 'cross-dec. Expert', 'dist ↔ choice')
    ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
    ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
    print(f"c': per-mouse cross-decode {np.mean(list(nv.values())):.3f} -> "
          f"{np.mean(list(ev.values())):.3f}  p={p:.3f}")
    return ax


def panel_gen_mouse(fig, gsX, off=0):
    """d companion: per-mouse WITHIN-task (x) vs CROSS-task (y) accuracy, Expert. Plotted as raw
    accuracies on purpose — the chance-corrected RATIO is unusable per animal (within-task sits near
    chance for sample/test, so the denominator ~0 and the ratio reaches 1e6). Distance below the
    unity line IS the generalisation loss, and the x position shows the ceiling that sets it."""
    PG = RES['PM_GEN' + SUF]
    axes = []
    for j, v in enumerate(['sample', 'choice', 'test']):
        ax = fig.add_subplot(gsX[0, j + off]); axes.append(ax)
        E3 = np.eye(3, dtype=bool)
        ax.plot([0.45, 0.95], [0.45, 0.95], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6); ax.axvline(0.5, ls=':', color='0.85', lw=0.6)
        wi, cr = [], []
        for m in ALL_MICE:
            if (m, 'Expert', v) not in PG:
                continue
            M = PG[(m, 'Expert', v)]
            x = float(np.diag(M).mean()); y = float(M[~E3].mean())
            wi.append(x); cr.append(y)
            ax.scatter(x, y, s=34, color=MOUSE_COLOR[m], marker=GMARKER[GROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        ax.scatter(np.mean(wi), np.mean(cr), s=80, color='k', marker='D', edgecolors='w',
                   linewidths=0.6, zorder=5)
        ax.set_xlim(0.45, 0.95); ax.set_ylim(0.45, 0.95); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
        ax.set_title(v, loc='left', fontsize=7)
        ax.set_xlabel('within-task', fontsize=7)
        if j == 0:
            ax.set_ylabel('cross-task', fontsize=7)
        print(f"d': {v:7s} per-mouse within {np.mean(wi):.3f}  cross {np.mean(cr):.3f}")
    return axes[0]


def panel_d(fig, gsD):
    R = pd.read_pickle(CCGP_CACHE)
    axes = []
    for j, v in enumerate(['sample', 'GNG', 'test', 'choice']):   # cache keys
        ax = fig.add_subplot(gsD[0, j]); axes.append(ax)
        piv = (R[R['variable'] == v].pivot_table(index='mouse', columns='stage', values='ccgp')
               .dropna(subset=['Naive', 'Expert']))
        ax.plot([0.42, 1.0], [0.42, 1.0], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        for m, rr in piv.iterrows():
            ax.scatter(rr['Naive'], rr['Expert'], s=42, color=MOUSE_COLOR.get(m, '0.5'),
                       marker=GMARKER[GROUP.get(m, 'Jaws')], edgecolors='w', linewidths=0.5, zorder=3)
        ax.scatter(piv['Naive'].mean(), piv['Expert'].mean(), s=95, color='k', marker='D',
                   edgecolors='w', linewidths=0.6, zorder=5)
        p = float(wilcoxon(piv['Expert'], piv['Naive']).pvalue)
        ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{CODE_NAME[v]}  ({"∗" if p < .05 else "n.s."})', loc='left', fontsize=TITLE_FS)
        ax.set_xlabel('CCGP — Naive', fontsize=7.5)
        ax.text(0.05, 0.96, f'Δ={piv.Expert.mean() - piv.Naive.mean():+.2f}\np={p:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=6, color='0.3')
        print(f'e: {v:7s} N {piv["Naive"].mean():.3f} -> E {piv["Expert"].mean():.3f}  p={p:.3f}')
    axes[0].set_ylabel('CCGP — Expert', fontsize=7.5)
    return axes[0]


# ══ ASSEMBLE (paper figure: no suptitle, no footnotes — prose is caption+Methods) ══
#  row 0: A (one row of 4) | row 1: B + C + D (all the matrix panels) | row 2: E
#  Each of b/c/d keeps its pooled matrices and its per-mouse scatter TOGETHER, at the same cell size:
#  row 0  a   the frame (4 scatters)
#  row 1  b = 2 matrices + 1 scatter  |  c = 2 matrices + 1 scatter
#  row 2  d = 3 matrices + 3 scatters
#  row 3  e   per-mouse CCGP
#  TWO COLUMNS: a stacked down column 1; b, c, d(matrices), d(per-mouse), e down column 2.
fig = plt.figure(figsize=(13.2, 16.0))
outer = fig.add_gridspec(2, 12, height_ratios=[1.15, 3.0], hspace=0.11,
                         left=0.080, right=0.982, top=0.972, bottom=0.040, wspace=0.9)
SD_F, H_F, (WS_F, BS_F), (WL_F, BL_F), (WT_F, BT_F), BLINE = build_frame()
gsT = outer[0, 0:12].subgridspec(2, 4, wspace=0.34, hspace=0.18)
axT = panel_traj(fig, gsT)
gsA = outer[1, 0:4].subgridspec(5, 1, hspace=0.34)
axA = panel_a(fig, gsA, SD_F, H_F, WS_F, BS_F, WL_F, BL_F, BLINE)

gsR = outer[1, 5:12].subgridspec(5, 1, hspace=0.62)
gsB = gsR[0, 0].subgridspec(1, 3, wspace=0.60)      # [Naive | Expert | per-mouse], equal cells
axB = panel_b(fig, gsB)
panel_b_mouse(fig, gsB[0, 2].subgridspec(1, 1))
gsC = gsR[1, 0].subgridspec(1, 3, wspace=0.60)
axC = panel_c(fig, gsC)
panel_c_mouse(fig, gsC[0, 2].subgridspec(1, 1))
gsDa = gsR[2, 0].subgridspec(1, 3, wspace=0.60)     # d(a) pooled matrices
axD = panel_gen(fig, gsDa)
gsDb = gsR[3, 0].subgridspec(1, 3, wspace=0.60)     # d(b) the same three, per mouse
axDb = panel_gen_mouse(fig, gsDb)
gsE = gsR[4, 0].subgridspec(1, 4, wspace=0.55)
axE = panel_d(fig, gsE)

plabel(axT, 'A', dx=-0.05); plabel(axA, 'B'); plabel(axB, 'C', dx=-0.42)
plabel(axC, 'D', dx=-0.42); plabel(axD, 'E', dx=-0.42); plabel(axE, 'F', dx=-0.34)

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_manifold_main{FIGSUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_main{FIGSUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_main{FIGSUF}.png'),
      '(NO PCA in the decoder)' if NOPCA else '')
