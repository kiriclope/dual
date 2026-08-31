"""fig_manifold_main.py — Fig 3: ONE manifold, seen in a fixed frame.

RESTRUCTURED 2026-08-30 (user decision, "redistribute"): this figure now carries ONLY the frame
itself — the four codes in time, the task states positioned in the sample x choice plane, and the
axis geometry. The LEARNING panels moved to Fig 4 (dist<->choice cross-decode matrices + the two
per-mouse learning scatters — see overlaps/fig_overlaps_main_native.py panel A); the cross-task
GENERALISATION matrices moved to Fig 2 (fig_dimensionality_main.py panel E); the per-mouse
generalisation row and the CCGP panel moved to ED (fig_manifold_supp.py). Canonical pipeline =
no-PCA (--nopca, unsuffixed filenames); the PCA-20 build is the ED robustness variant.

  A  the four codes over time (Naive | Expert rows x sample/dist/test/choice), replayed from the
     overlaps CCGD projections (ORIG_TRACES).
  B  the frame itself — held-out pseudo-trials plotted IN the fixed axes (x = sample axis @ mid-delay,
     y = behavioural lick axis @ decision, both trained on an independent trial half), a 2 (Naive |
     Expert) x 5 storyboard (DPA·md -> DPA·decision -> dual·md -> dual·late -> dual·decision), each
     row in that stage's OWN frame (axes re-fit per stage; added 2026-08-31 — the slimmed figure
     read empty next to Figs 2/4). Metric, unlike a t-SNE map: every offset is in z units.
  C  axis geometry — attenuation-corrected |cos| between sample / choice / dist (AXIS_FRAME),
     Naive vs Expert; sample orthogonal to both, choice x dist modest and growing (the growth is
     QUANTIFIED per-mouse in Fig 4's panel A, not here; values live in the cache — don't hardcode).
  D  per-mouse raw |cos| Naive-vs-Expert scatters, one per axis pair (PM_COS raw values):
     orthogonality holds in EVERY animal at both stages; choice x dist higher, most mice above
     unity (tested/starred in Fig 4A, deliberately no stats here).

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
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse, Rectangle
from decoders import fit_axis, SUF, NOPCA, NPC     # THE shared decoder (see decoders.py)
# the overlaps caches (matrices, ccgp, ORIG_TRACES) exist only for PCA-20 and no-PCA; an --npc N!=20
# run would silently mix npc-N pca-side caches with PCA-20 overlaps caches AND overwrite the _pca20
# files — refuse loudly instead.
assert NOPCA or NPC == 20, (f'--npc {NPC}: no matching overlaps caches (they are PCA-20 only); '
                            'build matrices/ccgp/ORIG_TRACES at that NPC first')
FIGSUF = '' if NOPCA else f'_pca{NPC}'             # filename: plain = no denoising

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
# (the overlaps matrices/ccgp caches are no longer read here — the cross-decode panel lives in
#  Fig 4, the generalisation matrices in Fig 2, and CCGP in fig_manifold_supp.py)
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
    """Fit the shared decoder on two class blocks and return (unit axis, boundary midpoint).

    The midpoint (training-set decision boundary along the axis) is kept for reference/printing;
    the PLOTTED zero is the pre-trial BASELINE (build_frame's BL), which both the trajectory row
    and the state scatters share. Per-panel centring is still avoided — the push is a grand-mean
    translation and per-panel means would erase it."""
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


def build_frame(stage='Expert'):
    """The frame FOR ONE STAGE: neuron scale, the trial-half split, the two axes, the BASELINE
    origin. Axes are re-fit per stage on that stage's own independent trial half — the Naive row
    shows Naive's OWN frame (per-stage units; the quantitative Naive→Expert push, on ONE fixed
    axis, is Fig 4's job, not this panel's).

    Panels a and b must share BOTH the units and the zero, or the same axis name means two different
    coordinates. Both use the pooled whole-population projection with the pre-trial baseline as
    zero (the dashed lines in the state scatters mark that baseline, not the decision boundary —
    boundary lines were removed: they invited being read as zero).

    NB sets the module-global STAGE (every helper — neuron_scale/make_halves/_pseudo/cloud — reads
    it); call build_frame + frame_states for one stage before moving to the next."""
    global STAGE
    STAGE = stage
    sd = neuron_scale(AW['delay+dec'])
    H = make_halves(np.random.RandomState(7))
    (w_s, b_s), (w_l, b_l) = sample_axis(sd, H), lick_axis(sd, H)
    CM = np.asarray(RES['CMBIN'][STAGE], dtype=float)
    BL = {}
    for key, w in (('s', w_s), ('l', w_l)):
        v = np.zeros(CM.shape[2])
        for m in MICE:
            val = VALIDIX[(m, STAGE)]
            if len(val):
                v = v + w[val] @ (CM[:, val, :] / sd[val][None, :, None]).mean(0)
        BL[key] = float(v[BINS_BL].mean())
    return sd, H, (w_s, b_s), (w_l, b_l), BL


# LATE DELAY is essential for the dual set: it is the only plotted window AFTER the Go/NoGo cue
# (6.5-7 s) and its lick, and it is where the Go state actually crosses into the action region.
# Mid-delay alone is PRE-cue, so Go is still short of the boundary there.
B_SPECS = [('DPA', 'md'), ('DPA', 'decision'),
           ('dual', 'md'), ('dual', 'delay'), ('dual', 'decision')]


def frame_states(sd, H, w_s, w_l, BL):
    """Held-out state clouds for the five storyboard windows of the CURRENT stage (call while the
    module-global STAGE is set by build_frame — cloud() reads it)."""
    out = []
    for sname, wn in B_SPECS:
        X, crow, conds = cloud(sname, wn, sd, H, np.random.RandomState(2))
        out.append((X @ w_s - BL['s'], X @ w_l - BL['l'], crow, conds))
    return out


def panel_a(fig, gsA, STAGEDATA):
    """The storyboard, 2 rows (Naive | Expert) x 5 windows — each row in that stage's OWN frame
    (axes re-fit per stage; per-stage units — the fixed-axis quantitative push is Fig 4's).
    All ten frames share one x/y range so positions are comparable at a glance; autoscaling
    would push the baseline off-screen where states sit far from it."""
    ALLX = np.concatenate([d[0] for _, DATA in STAGEDATA for d in DATA])
    ALLY = np.concatenate([d[1] for _, DATA in STAGEDATA for d in DATA])
    pad = 0.06 * (ALLX.max() - ALLX.min())
    XL = (min(ALLX.min(), 0) - pad, max(ALLX.max(), 0) + pad)
    padY = 0.06 * (ALLY.max() - ALLY.min())
    YL = (min(ALLY.min(), 0) - padY, max(ALLY.max(), 0) + padY)
    WLAB = {'md': 'mid-delay (pre-cue)', 'delay': 'late delay (post-lick)', 'decision': 'decision'}
    axes = []
    for r, (stage, DATA) in enumerate(STAGEDATA):
        for j, (sname, wn) in enumerate(B_SPECS):
            ax = fig.add_subplot(gsA[r, j]); axes.append(ax)
            xs, ys, crow, conds = DATA[j]
            # baseline only — boundary lines were removed (they invited being read as zero)
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
            if r == 0:
                ax.set_title(f'{sname} · {WLAB[wn]}', loc='left', fontsize=TITLE_FS)
            if r == 1:
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                sx = x0 + 0.05 * (x1 - x0); sy = y0 + 0.06 * (y1 - y0)
                ax.plot([sx, sx + 5], [sy, sy], '-', color='0.3', lw=1.1)
                ax.text(sx + 2.5, sy + 0.015 * (y1 - y0), '5 z', ha='center', va='bottom',
                        fontsize=5.4, color='0.3')
                if j == 2:                           # one shared x-label, centred under the grid
                    ax.set_xlabel('sample axis   A ← · → B', fontsize=7)
            if j == 0:
                ax.set_ylabel(f'{stage}\nchoice axis\n← no-lick · lick →', fontsize=7)
            if r == 0 and j == 0:                    # colour/fill key once
                hs = [mlines.Line2D([], [], marker='o', ls='', ms=4, color=SAMPC[0], label='sample A'),
                      mlines.Line2D([], [], marker='o', ls='', ms=4, color=SAMPC[1], label='sample B'),
                      mlines.Line2D([], [], marker='o', ls='', ms=4, mfc='0.4', mec='0.4', label='lick'),
                      mlines.Line2D([], [], marker='o', ls='', ms=4, mfc='w', mec='0.4', label='no-lick')]
                ax.legend(handles=hs, frameon=False, fontsize=5.4, loc='upper left', ncols=2,
                          handletextpad=0.15, columnspacing=0.5, labelspacing=0.25, borderaxespad=0.0)
            if r == 0 and j == 2:                    # marker key on the first dual panel
                hs = [mlines.Line2D([], [], marker='^', ls='', ms=4, mfc='none', mec='0.3', label='Go'),
                      mlines.Line2D([], [], marker='s', ls='', ms=4, mfc='none', mec='0.3', label='NoGo')]
                ax.legend(handles=hs, frameon=False, fontsize=5.4, loc='upper left', ncols=2,
                          handletextpad=0.15, columnspacing=0.5, borderaxespad=0.0)
            lk = np.array([conds[c][1] == conds[c][2] for c in crow])
            print(f'a: {stage:6s} {sname:4s} {wn:9s} action-axis mean {ys.mean():+6.2f} '
                  f'(lick {ys[lk].mean():+6.2f} / no-lick {ys[~lk].mean():+6.2f})  '
                  f'sample sep {xs[[conds[c][1] == 1 for c in crow]].mean() - xs[[conds[c][1] == 0 for c in crow]].mean():+.2f}')
    return axes[0]


# ══ a — TRAJECTORIES on the two axes of the frame (top row) ═══════════════════
#   Uses CMBIN (per-bin condition means, 12 x 3319 x 84 per stage, cached by exp_antact_traj.py),
#   projected on the SAME two axes and the SAME origins as the scatters below, so the trajectory
#   row and the state row are literally the same coordinates. t = bin/6 - 0.5 s (exact).
TBIN = lambda b: np.asarray(b) / 6.0 - 0.5
EVENTS = [('sample', 2.0, 3.0, SAMPC[0]), ('distractor', 4.5, 5.5, '#cc3311'),
          ('GNG cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]
# cue is 6.5-7.0 s; the reward window 7.0-7.5 s is deliberately unshaded (as everywhere else)


BINS_BL = np.arange(0, 12)          # baseline bins (0-11), shared_data.md convention
# One panel per code with its own two-class colour pair — the original Fig-3 panel-A convention
# (colours/levels/labels come from ORIG_SPECS in the cache, written by exp_traj_orig.py).


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
    # SHARED y-limits per code COLUMN (across Naive and Expert): the panel's point is the
    # Naive -> Expert depth comparison, which independent autoscaling silently defeated.
    YLK = {}
    for k, spec in enumerate(SP):
        lo, hi = 0.0, 0.0
        for stage in ['Naive', 'Expert']:
            for lv in spec['levels']:
                M = np.asarray(TR[(stage, spec['code'], int(lv))], dtype=float)
                if not len(M):
                    continue
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                lo = min(lo, (mu - se).min()); hi = max(hi, (mu + se).max())
        pad = 0.05 * (hi - lo)
        YLK[k] = (lo - pad, hi + pad)
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
            ax.set_ylim(*YLK[k])                    # shared per code column, Naive vs Expert
            ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12])
            if r == 1:
                ax.set_xlabel('time (s)', fontsize=7)
                if k == 0:                              # sample legend lives in the EXPERT panel:
                    ax.legend(frameon=False, fontsize=5.4, handlelength=1.2, loc='lower right')
                    # (the Naive panel's top band carries the epoch names and its lower-right is
                    #  crossed by the Odor-A tail — both collide with a legend there)
            else:
                ax.tick_params(labelbottom=False)
                ax.set_title(f"{CODE_NAME[spec['code']]} code", loc='left', fontsize=TITLE_FS)
                if k > 0:
                    ax.legend(frameon=False, fontsize=5.4, handlelength=1.2, loc='upper left')
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
        ax.set_anchor('NW')                             # top line shared with panel D
        ax.set_title(stage, loc='left', fontsize=TITLE_FS)
        if j == 0:
            ax.set_ylabel('axis geometry\n|cos|', fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'b: {stage} sample-action {C[0,1]:.2f}  sample-distr {C[0,2]:.2f}  '
              f'action-distr {C[1,2]:.2f}  (reliab {np.round(AXF[stage]["rel"],2)})')
    return axes[0]


# ══ d — the same geometry in EVERY animal: per-mouse raw |cos|, Naive vs Expert scatters ══
_pmpal = sns.color_palette('tab10', n_colors=len(MICE))
PMCOL = {m: _pmpal[i] for i, m in enumerate(MICE)}                 # same mouse = same colour (Figs 2-4)
PMGROUP = {**{m: 'Jaws' for m in MICE[:5]}, **{m: 'ChR' for m in MICE[5:7]},
           **{m: 'ACC' for m in MICE[7:]}}
PMMARK = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}


def panel_pm_cos(fig, gsD):
    """Per-mouse RAW split-half |cos| for the three axis pairs, Naive (x) vs Expert (y) — the house
    per-mouse scatter idiom (PC cache raw values; the attenuation correction is unusable at
    per-animal reliabilities, see exp_permouse_frame.py). sample×choice and sample×dist hug the
    floor in every animal at both stages; choice×dist sits higher with most mice above unity.
    NO stats drawn — the choice×dist increase is the whitelisted, starred test in Fig 4A;
    duplicating it here would double-report."""
    PC = RES['PM_COS' + SUF]
    PAIRS = [('sa', 'sample × choice'), ('sd', 'sample × dist'), ('ad', 'choice × dist')]
    lo, hi = 0.0, 0.25
    axes = []
    for j, (key, lab) in enumerate(PAIRS):
        ax = fig.add_subplot(gsD[0, j]); axes.append(ax)
        ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
        nv, ev = [], []
        for m in MICE:
            if (m, 'Naive') not in PC or (m, 'Expert') not in PC:
                continue
            a = PC[(m, 'Naive')][key + '_raw']; b = PC[(m, 'Expert')][key + '_raw']
            nv.append(a); ev.append(b)
            ax.scatter(a, b, s=26, color=PMCOL[m], marker=PMMARK[PMGROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        ax.scatter(np.mean(nv), np.mean(ev), s=60, color='k', marker='D', edgecolors='w',
                   linewidths=0.6, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('NW')                             # top line shared with panel C
        ax.set_xticks([0, 0.1, 0.2]); ax.set_yticks([0, 0.1, 0.2])
        if j:
            ax.tick_params(labelleft=False)
        ax.set_title(lab, loc='left', fontsize=6.5)
        if j == 0:
            ax.set_ylabel('raw |cos|\nExpert', fontsize=7)
        if j == 1:
            ax.set_xlabel('raw |cos| — Naive', fontsize=7)
        print(f"d: {key} raw |cos| {np.mean(nv):.3f} -> {np.mean(ev):.3f}")
    return axes[0]



# ══ ASSEMBLE (paper figure: no suptitle, no footnotes — prose is caption+Methods) ══
#  Three full-width rows (filled back out 2026-08-31 — the slimmed figure read empty next to 2/4):
#  row 0  A  the four codes over time (2 x 4 traces)
#  row 1  B  the frame — 2 (Naive | Expert) x 5 storyboard, each row in its stage's OWN frame
#  row 2  C  axis-geometry |cos| matrices  +  D  per-mouse raw-|cos| strip
# row 1 deliberately SHORTER than its content suggests: the shared y-range is set by the decision
# licks (+9 z), so tall frames leave the mid-delay panels mostly empty — compressing the row fills
# the data band. Row 2 taller so the aspect-locked matrices/scatters grow.
fig = plt.figure(figsize=(12.4, 10.8))
outer = fig.add_gridspec(3, 12, height_ratios=[1.45, 1.55, 1.0], hspace=0.28,
                         left=0.062, right=0.982, top=0.972, bottom=0.035, wspace=0.9)
gsT = outer[0, 0:12].subgridspec(2, 4, wspace=0.34, hspace=0.18)
axT = panel_traj(fig, gsT)
STAGEDATA = []
for _st in ['Naive', 'Expert']:
    _sd, _H, (_ws, _bs), (_wl, _bl), _BL = build_frame(_st)        # sets the module STAGE global
    STAGEDATA.append((_st, frame_states(_sd, _H, _ws, _wl, _BL)))
gsA = outer[1, 0:12].subgridspec(2, 5, wspace=0.16, hspace=0.10)
axA = panel_a(fig, gsA, STAGEDATA)
gsB = outer[2, 0:6].subgridspec(1, 2, wspace=0.28)   # left-aligned: C's letter lines up with A/B
axB = panel_b(fig, gsB)
gsD = outer[2, 7:12].subgridspec(1, 3, wspace=0.28)
axD = panel_pm_cos(fig, gsD)

plabel(axT, 'A', dx=-0.05); plabel(axA, 'B', dx=-0.10)
plabel(axB, 'C', dx=-0.30); plabel(axD, 'D', dx=-0.16)

# ── CAPTION (justified, drawn below — same mechanism as Fig 2; edit CAP_PARAS + re-render) ──
CAP_PARAS = [
    'Figure 3 | One manifold: the states of both tasks live in a single fixed sample × choice frame.',
    'A. The four codes over time (sample / dist / test / choice; Naive top, Expert bottom): per-mouse '
    'cross-validated decoder projections (CCGD), mean ± SEM across mice (n = 9), correct laser-off '
    'trials, one shared per-mouse unit so amplitudes are comparable across codes; y-axes shared '
    'within each code column. Shaded bands: sample, distractor, GNG cue and test epochs. The sample '
    'code is maintained throughout the delay in both stages; the dist and choice codes rise at their '
    'own epochs.',
    'B. The frame itself, Naive (top) and Expert (bottom): held-out pseudo-trials projected onto '
    'ONE sample axis (trained at mid-delay) and ONE behavioural choice axis (trained at the '
    'decision epoch), both fit on an independent trial half — no self-inclusion; axes are re-fit '
    'per stage (per-stage units — the quantitative fixed-axis Naive→Expert push is Fig. 4B). '
    'Dashed lines = the pre-trial baseline (zero); ellipses = 1 SD of the pseudo-trial cloud; '
    'fill = lick, open = no-lick; circle / triangle / square = DPA / Go / NoGo; colour = sample '
    'A / B; scale bar 5 z. Read left to right along the trial: DPA states separate along the '
    'sample axis only (mid-delay) and split along the choice axis at decision; the dual Go and '
    'NoGo states sit apart along the choice axis already at mid-delay (weakly in Naive, strongly '
    'in Expert — the distractor precedes this window), and after the cue the Go state crosses '
    'toward lick (late delay) — in both stages, every separation in every task is carried by the '
    'same two axes. Note the Expert DPA delay states sit below the choice-axis baseline where the '
    'Naive ones do not — the repositioning quantified on a fixed axis in Fig. 4B.',
    'C. Axis geometry: attenuation-corrected split-half |cos| between the three axes '
    '(pseudo-population). The sample axis is orthogonal to both action codes (|cos| ≈ 0.07–0.09, '
    'both stages); the choice × dist overlap is partial and grows with learning '
    f'({np.asarray(AXF["Naive"]["cos"])[1, 2]:.2f} → {np.asarray(AXF["Expert"]["cos"])[1, 2]:.2f}) — '
    'quantified per animal in Fig. 4A.',
    'D. The same geometry in every animal: per-mouse raw split-half |cos|, Naive (x) vs Expert '
    '(y), one panel per axis pair (colour = mouse, marker = opsin line, diamond = mean; raw '
    'values — the attenuation correction is unusable at per-animal reliabilities). sample × '
    'choice and sample × dist hug the floor in all 9 mice at both stages; choice × dist sits '
    'higher with most mice above the unity line — that increase is tested (and starred) in '
    'Fig. 4A, not here.',
]
from figcaption import draw_justified                  # shared with fig_dimensionality_main.py
draw_justified(fig, CAP_PARAS)

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_manifold_main{FIGSUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_main{FIGSUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_main{FIGSUF}.png'),
      '(NO PCA in the decoder)' if NOPCA else '')
