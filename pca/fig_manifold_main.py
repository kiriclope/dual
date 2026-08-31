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
  C  (RESTRUCTURED 2026-08-31: the PROOF row moved up, directly answering B's plotted-in-a-plane
     objection) SUFFICIENCY summary bars: mean ± SEM of the per-mouse plane / out-of-plane / full
     accuracies (PM_PLANE / exp_permouse_plane.py), canonical timeline sample/dist/test/choice,
     ALL pairwise Wilcoxons drawn with knob-robust verdicts; choice plane-vs-full flips (.94/.012)
     -> drawn as † only. Adaptive bracket heights.
  D  the per-animal breakdown: 3x4 Naive-vs-Expert scatters (rows = plane/out/full). Double
     dissociation per mouse; learning deltas n.s. everywhere EXCEPT the whitelisted dist-plane
     increase (p=.020/.027, 8/9 & 7/9 up) — starred.
  E  axis geometry — attenuation-corrected |cos| between sample / choice / dist (AXIS_FRAME),
     Naive vs Expert; sample orthogonal to both, choice x dist modest and growing (quantified
     per-mouse in Fig 4A; values from the cache — don't hardcode). The per-mouse cosine scatters
     MOVED TO ED (fig_manifold_supp.py) — their choice x dist subpanel duplicated Fig 4A's data
     point-for-point.
  F  ONE frame across LEARNING (new): cross-stage decoding 2x2s (XSTAGE_DEC / exp_plane_frame.py)
     — train the sample/choice decoders in one stage, test held-out in the other (registered
     neurons); transfer/within 0.90 / 0.87, knob-robust. Learning moves the state, not the frame.

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
from matplotlib.patches import Ellipse, Rectangle, Patch
from scipy.stats import wilcoxon
from decoders import SUF, NOPCA, NPC               # the shared decoder config (see decoders.py)
# the overlaps caches (matrices, ccgp, ORIG_TRACES) exist only for PCA-20 and no-PCA; an --npc N!=20
# run would silently mix npc-N pca-side caches with PCA-20 overlaps caches AND overwrite the _pca20
# files — refuse loudly instead.
assert NOPCA or NPC == 20, (f'--npc {NPC}: no matching overlaps caches (they are PCA-20 only); '
                            'build matrices/ccgp/ORIG_TRACES at that NPC first')
ANTACT = '--antact' in sys.argv[1:]                # variant: choice axis = ANTICIPATORY action
ASUF = '_antact' if ANTACT else ''                 #   axis (overlaps train bins 48-62, vs 57-62)
FIGSUF = ('' if NOPCA else f'_pca{NPC}') + ASUF    # filename: plain = no denoising

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
           'tasks': '#cc3311', 'gng': '#ee7733', 'dist': '#ee7733'}   # house palette, as in Fig 2
# ONE name per code across every panel. The four task variables are sample / dist / test / choice;
# "lick", "action" and "GNG" are the legacy aliases that used to appear panel-to-panel.
CODE_ORDER = ['sample', 'dist', 'test', 'choice']
CODE_NAME = {'sample': 'sample', 'GNG': 'dist', 'gng': 'dist', 'distractor': 'dist',
             'test': 'test', 'lick': 'choice', 'action': 'choice', 'choice': 'choice'}
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
# (the overlaps matrices/ccgp caches are no longer read here — the cross-decode panel lives in
#  Fig 4, the generalisation matrices in Fig 2, and CCGP in fig_manifold_supp.py)
RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
AXF = RES['AXIS_FRAME' + SUF]
assert 'PM_PLANE' + SUF in RES, ('missing PM_PLANE' + SUF +
                                 ' — run: python exp_permouse_plane.py' +
                                 (' --nopca' if NOPCA else ''))
PMPL = RES['PM_PLANE' + SUF]                       # (plane, full, out-of-plane) per mouse/stage/var
assert 'XSTAGE_DEC' + SUF in RES, ('missing XSTAGE_DEC' + SUF +
                                   ' — run: python exp_plane_frame.py' +
                                   (' --nopca' if NOPCA else ''))
XSD = RES['XSTAGE_DEC' + SUF]                      # cross-stage decoding (train x test stage)
# (fits_inputs.pkl is no longer read here — the storyboard replays FRAME_STATES, and every other
#  panel reads a results.pkl cache; the raw-tensor machinery lives in the exp_* scripts)


def plabel(ax, s, dx=-0.06):
    ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')


# ══ a — the frame: REPLAYED per-mouse CCGD states (exp_frame_states.py) ═══════
# The storyboard uses the SAME projections as the panel-A traces — SAMPLE_D (x) and LICK_D (y)
# from overlaps/main_panels: cross-validated CCGD decision functions, per-mouse baseline zero,
# one shared per-mouse unit. Two earlier fresh-axis builds FAILED the A↔B consistency check
# (user 2026-08-31): a freshly-fit single-window axis carries the trial's condition-independent
# ramp (29-45% of the code), so NO single origin worked — baseline-zero dragged every window to
# one side, boundary-zero put mid-delay 3-5 z below the lick line while A's traces sat AT
# baseline. Replaying the CCGD projections makes A and B literally the same coordinates.
TASKM = {'DPA': 'o', 'DualGo': '^', 'DualNoGo': 's'}

# LATE DELAY is essential for the dual set: it is the only plotted window AFTER the Go/NoGo cue
# (6.5-7 s) and its lick, and it is where the Go state actually crosses into the action region.
# Mid-delay alone is PRE-cue, so Go is still short of the boundary there.
B_SPECS = [('DPA', 'md'), ('DPA', 'decision'),
           ('dual', 'md'), ('dual', 'delay'), ('dual', 'decision')]

FS_KEY = ('FRAME_STATES' if NOPCA else 'FRAME_STATES_pca20') + ASUF
assert FS_KEY in RES, (f'missing {FS_KEY} — run: cd /home/leon/dual/pca && python '
                       f'exp_frame_states.py' + ('' if NOPCA else ' --pca')
                       + (' --antact' if ANTACT else ''))
# FRAME_STATES[(stage, set, window)] = {(task, samp, lick): (n_mice, 2) per-mouse [x, y] means}


def _centered(ENT):
    """Per-mouse re-centring of one storyboard window: subtract each mouse's cross-condition
    mean state, so the panel shows the CONDITION GEOMETRY — which codes separate, along which
    axis — free of the common ramp and of between-mouse offsets (both of which swamped the
    clouds in the raw replay). The removed absolute displacement lives in panel A's traces and
    is quantified on a fixed axis in Fig 4B."""
    per = {}
    for cd, (mice, P) in ENT.items():
        for mo, p in zip(mice, P):
            per.setdefault(mo, []).append(p)
    off = {mo: np.mean(v, 0) for mo, v in per.items() if len(v) >= 2}
    out = {}
    for cd, (mice, P) in ENT.items():
        pts = np.array([p - off[mo] for mo, p in zip(mice, P) if mo in off])
        if len(pts) >= 3:
            out[cd] = pts
    # the grand-mean removed offset (where this window's mean state sits relative to baseline)
    # is returned for the console print only — baseline lines were tried and removed (user)
    return out, np.mean(list(off.values()), 0)


def panel_a(fig, gsA):
    """The storyboard, 2 rows (Naive | Expert) x 5 windows, replayed from FRAME_STATES (the
    panel-A CCGD projections) and re-centred per mouse per window (_centered): faint dots =
    per-mouse condition means (>=3 trials), ellipse = 1 SD across mice, marker = the grand mean,
    crosshair = the window's mean state. All ten frames share one x/y range."""
    FS = RES[FS_KEY]
    BOTH = {(st,) + sp: _centered(FS[(st,) + sp]) for st in ['Naive', 'Expert'] for sp in B_SPECS}
    CEN = {k: v[0] for k, v in BOTH.items()}
    OFF = {k: v[1] for k, v in BOTH.items()}
    ALL = np.vstack([pts for ent in CEN.values() for pts in ent.values()])
    # shared range from the 2-98 percentiles of the per-mouse dots (a few outlier dots clip;
    # the raw min/max left the row mostly empty around a thin data band)
    x0, x1 = np.percentile(ALL[:, 0], [2, 98]); y0, y1 = np.percentile(ALL[:, 1], [2, 98])
    XL = (x0 - 0.08 * (x1 - x0), x1 + 0.08 * (x1 - x0))
    YL = (y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    WLAB = {'md': 'mid-delay (pre-cue)', 'delay': 'late delay (post-lick)', 'decision': 'decision'}
    axes = []
    for r, stage in enumerate(['Naive', 'Expert']):
        for j, (sname, wn) in enumerate(B_SPECS):
            ax = fig.add_subplot(gsA[r, j]); axes.append(ax)
            ENT = CEN[(stage, sname, wn)]
            # crosshair = this window's cross-condition mean state (per mouse)
            ax.axhline(0, ls='--', color='k', lw=0.5, zorder=0)
            ax.axvline(0, ls='--', color='k', lw=0.4, zorder=0)
            for cd, P in ENT.items():
                col = SAMPC[int(cd[1])]; licks = cd[2]
                ax.scatter(P[:, 0], P[:, 1], s=6, marker='.', color=col, lw=0, alpha=0.5, zorder=1)
                if len(P) >= 3:
                    C = np.cov(P.T) / len(P)             # SEM ellipse (mean uncertainty; animal
                    ev, evec = np.linalg.eigh(C)         #   spread is panel D's job)
                    ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
                    ax.add_patch(Ellipse(P.mean(0), 2 * np.sqrt(ev[-1]), 2 * np.sqrt(ev[0]),
                                         angle=ang, fc=col, alpha=0.10, ec=col,
                                         lw=0.9, ls='-' if licks else '--', zorder=2))
                ax.scatter(*P.mean(0), marker=TASKM[cd[0]], s=30,
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
                ax.plot([sx, sx + 2], [sy, sy], '-', color='0.3', lw=1.1)
                ax.text(sx + 1.0, sy + 0.015 * (y1 - y0), '2 z', ha='center', va='bottom',
                        fontsize=5.4, color='0.3')
                if j == 2:                           # one shared x-label, centred under the grid
                    ax.set_xlabel('sample axis   A ← · → B', fontsize=7)
            if j == 0:
                yax = 'antic. action axis' if ANTACT else 'choice axis'
                ax.set_ylabel(f'{stage}\n{yax}\n← no-lick · lick →', fontsize=7)
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
            gm = {cd: P.mean(0) for cd, P in ENT.items()}
            ylk = np.mean([g[1] for cd, g in gm.items() if cd[2]])
            ynl = np.mean([g[1] for cd, g in gm.items() if not cd[2]])
            ssep = (np.mean([g[0] for cd, g in gm.items() if cd[1] == 1])
                    - np.mean([g[0] for cd, g in gm.items() if cd[1] == 0]))
            gng = ''
            if sname == 'dual':
                gsplit = (np.mean([g[1] for cd, g in gm.items() if cd[0] == 'DualGo'])
                          - np.mean([g[1] for cd, g in gm.items() if cd[0] == 'DualNoGo']))
                gng = f'  Go-NoGo y {gsplit:+.2f}'
            print(f'a: {stage:6s} {sname:4s} {wn:9s} action-axis '
                  f'lick {ylk:+6.2f} / no-lick {ynl:+6.2f}  sample sep {ssep:+.2f}{gng}  '
                  f'mean-vs-BL y {OFF[(stage, sname, wn)][1]:+.2f}')
    return axes[0]


# ══ a — TRAJECTORIES (top row): replayed CCGD projections (ORIG_TRACES) ═══════
#   Same projections and units as the storyboard below; the traces carry the ABSOLUTE positions
#   (baseline zero, ramp included) while each storyboard window is re-centred on its own mean
#   state — so trace SEPARATIONS match cloud separations window by window, but the trace value
#   is not the cloud's crosshair offset. t = bin/6 - 0.5 s (exact).
TBIN = lambda b: np.asarray(b) / 6.0 - 0.5
EVENTS = [('sample', 2.0, 3.0, SAMPC[0]), ('distractor', 4.5, 5.5, '#cc3311'),
          ('GNG cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]
# cue is 6.5-7.0 s; the reward window 7.0-7.5 s is deliberately unshaded (as everywhere else)


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
    TKEY = ('ORIG_TRACES' if NOPCA else 'ORIG_TRACES_pca20') + ASUF
    assert TKEY in RES, (f'missing {TKEY} — run: cd /home/leon/dual/pca && python '
                         f'exp_traj_orig.py' + ('' if NOPCA else ' --pca')
                         + (' --antact' if ANTACT else ''))
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



# ══ NEW f — ONE frame across LEARNING: cross-stage decoding (XSTAGE_DEC, exp_plane_frame.py).
#   Train the sample (@md) / choice (@decision) decoder in one stage, test held-out in the other
#   (neurons are registered across stages — identical valid masks). transfer/within ~0.9 => the
#   frame is functionally the SAME before and after learning; learning moves the STATE (Fig 4),
#   not the frame. Decoding, not cosines: the corrected cross-stage cosine explodes at the raw
#   axes' split-half reliabilities (~0.15) — logged dead-end. ══
def panel_xstage(fig, gsX):
    axes = []
    for k, vn in enumerate(['sample', 'choice']):
        ax = fig.add_subplot(gsX[0, k]); axes.append(ax)
        M = np.array([[np.mean(XSD[(vn, a, b)]) for b in ['Naive', 'Expert']]
                      for a in ['Naive', 'Expert']])
        ax.imshow(M, cmap='Reds', vmin=0.5, vmax=1.0, aspect='equal')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                        color='w' if M[i, j] > 0.82 else 'k')
        off = np.mean([M[0, 1], M[1, 0]]); dia = np.mean([M[0, 0], M[1, 1]])
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=5.8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Naive', 'Expert'] if k == 0 else [], fontsize=5.8)
        ax.set_title(f'{vn} axis', loc='left', fontsize=TITLE_FS)
        ax.set_anchor('NW')
        if k == 0:
            ax.set_ylabel('train stage', fontsize=7)
        ax.set_xlabel('test stage', fontsize=7)
        ax.text(0.5, -0.44, f'transfer/within {(off - .5) / (dia - .5):.2f}',
                transform=ax.transAxes, ha='center', va='top', fontsize=5.8, color='0.3')
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'f-xstage: {vn} within {dia:.2f} cross {off:.2f} ratio {(off - .5) / (dia - .5):.2f}')
    return axes[0]


# ══ d — SUFFICIENCY per animal: decode from the plane / its complement / the full space ══
#   PM_PLANE (exp_permouse_plane.py): per mouse & stage, each variable decoded from (i) only the
#   2 coordinates of that mouse's OWN sample x choice plane, (ii) the out-of-plane residual
#   (plane component removed), (iii) the full population. Held-out halves, canonical windows.
E_VARS = ['sample', 'dist', 'test', 'choice']      # canonical timeline order (dist added 2026-08-31)
E_SPACES = [('plane only (2-D)', 0), ('out-of-plane', 2), ('full space', 1)]


def panel_e_plane(fig, gsE):
    """3x4 per-mouse Naive-vs-Expert scatters (rows = space, cols = variable). Learning deltas are
    n.s. for every cell IN BOTH PIPELINES (p>=.13) with ONE whitelisted exception: the dist
    plane-only accuracy GROWS with learning (p=.020 nopca / .027 pca20, 8/9 & 7/9 mice up) —
    the per-animal 'dist joins the plane' result, starred."""
    lo, hi = 0.42, 1.01
    axes = []
    for r, (rowlab, key) in enumerate(E_SPACES):
        for c, vn in enumerate(E_VARS):
            ax = fig.add_subplot(gsE[r, c]); axes.append(ax)
            ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
            ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
            ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
            nv, ev = [], []
            for m in MICE:
                if (m, 'Naive') not in PMPL or (m, 'Expert') not in PMPL:
                    continue
                if vn not in PMPL[(m, 'Naive')] or vn not in PMPL[(m, 'Expert')]:
                    continue
                a = PMPL[(m, 'Naive')][vn][key]; b = PMPL[(m, 'Expert')][vn][key]
                nv.append(a); ev.append(b)
                ax.scatter(a, b, s=26, color=PMCOL[m], marker=PMMARK[PMGROUP[m]],
                           edgecolors='w', linewidths=0.5, zorder=3)
            nv, ev = np.array(nv), np.array(ev)
            ax.scatter(nv.mean(), ev.mean(), s=56, color='k', marker='D', edgecolors='w',
                       linewidths=0.6, zorder=5)
            p = float(wilcoxon(ev, nv).pvalue)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
            if c:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(vn, loc='left', fontsize=7)
            if r < 2:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(f'{rowlab}\nExpert', fontsize=6.8)
            if r == 2 and c == 1:
                ax.set_xlabel('accuracy — Naive', fontsize=7, loc='right')   # ~grid centre (4 cols)
            # whitelisted verdict (knob-robust .020/.027): dist gains in-plane decodability
            star = '  ∗' if (vn == 'dist' and key == 0 and p < 0.05) else ''
            ax.text(0.05, 0.96, f'Δ={ev.mean() - nv.mean():+.02f}\np={p:.2f}{star}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=5.4 if not star else 6.2, color='0.3' if not star else 'k',
                    fontweight='normal' if not star else 'bold')
            print(f'f: {rowlab:16s} {vn:7s} {nv.mean():.2f} -> {ev.mean():.2f}  p={p:.2f}{star}')
    return axes[0]


# ══ f — the summary: the three spaces compared, mean ± SEM, WITH the paired tests ══
#   STAR POLICY (verified across BOTH pipelines, 2026-08-31): sample out-vs-full p=.0039/.0039 ∗,
#   test plane-vs-full p=.0039/.0039 ∗, choice out-vs-full p=.0195/.0078 ∗; sample plane-vs-full
#   and test out-vs-full n.s. in both (drawn n.s.). choice plane-vs-full FLIPS with the knob
#   (p=.94 nopca / .012 pca20) — NOT drawn; the caption states the knob-dependence.
def panel_f_spaces(fig, gsF):
    ax = fig.add_subplot(gsF[0, 0])
    trips = {}
    gtop = {}                                       # per-group tallest bar+SEM (bracket anchoring)
    for gi, vn in enumerate(E_VARS):
        trips[vn] = {}
        for si, (slab, key) in enumerate([('plane', 0), ('out', 2), ('full', 1)]):
            vals = np.array([np.mean([PMPL[(m, st)][vn][key] for st in ['Naive', 'Expert']])
                             for m in MICE
                             if all((m, st) in PMPL and vn in PMPL[(m, st)]
                                    for st in ['Naive', 'Expert'])])
            trips[vn][slab] = vals
            gtop[gi] = max(gtop.get(gi, 0),
                           vals.mean() + vals.std(ddof=1) / np.sqrt(len(vals)))
            x = gi + (si - 1) * 0.27
            col = VAR_COL[vn]
            if slab == 'plane':
                sty = dict(facecolor=col, edgecolor=col)
            elif slab == 'out':
                sty = dict(facecolor='w', edgecolor=col, hatch='///')
            else:
                sty = dict(facecolor=col, alpha=0.35, edgecolor=col)
            ax.bar(x, vals.mean(), 0.25, lw=0.9, zorder=2, **sty)
            ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1) / np.sqrt(len(vals)),
                        color='k', capsize=2, lw=0.9, zorder=3)

    def bracket(gi, s1, s2, y, mode='verdict'):
        """mode='verdict': knob-robust pair, draw ∗/n.s. from p. mode='dagger': the comparison
        flips with the decoder knob — draw † (defined in the caption), never a verdict."""
        POS = {'plane': -0.27, 'out': 0.0, 'full': 0.27}
        x1, x2 = gi + POS[s1], gi + POS[s2]
        p = float(wilcoxon(trips[E_VARS[gi]][s1], trips[E_VARS[gi]][s2]).pvalue)
        ax.plot([x1, x1, x2, x2], [y - 0.008, y, y, y - 0.008], color='0.25', lw=0.8,
                clip_on=False)
        if mode == 'dagger':
            ax.text((x1 + x2) / 2, y + 0.001, '†', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='0.4')
            print(f'e-bars: {E_VARS[gi]:7s} {s1}-vs-{s2} p={p:.4f} † (knob-dependent)')
            return
        sig = p < 0.05
        ax.text((x1 + x2) / 2, y + 0.001, '∗' if sig else 'n.s.', ha='center', va='bottom',
                fontsize=11 if sig else 6.5, fontweight='bold',
                color='k' if sig else '0.55')
        print(f'e-bars: {E_VARS[gi]:7s} {s1}-vs-{s2} p={p:.4f} {"*" if sig else "n.s."}')

    # ALL pairwise comparisons drawn (2026-08-31): knob-robust verdicts + 1 † (choice
    # plane-vs-full flips .94 nopca / .012 pca20 — no verdict, caption defines †). dist added:
    # its triplet is fully robust (plane-vs-out .0039/.0039, out-vs-full n.s. 1.0/.65,
    # plane-vs-full .0039/.0039) — the test-like pattern but with an above-chance, learning-
    # growing plane share (starred in F's dist plane cell).
    ytop = 0.45
    for gi in range(len(E_VARS)):
        b0 = gtop[gi] + 0.025                       # brackets stacked ABOVE the group's bars
        bracket(gi, 'plane', 'out', b0)
        bracket(gi, 'out', 'full', b0 + 0.045)
        bracket(gi, 'plane', 'full', b0 + 0.095,
                mode='dagger' if E_VARS[gi] == 'choice' else 'verdict')
        ytop = max(ytop, b0 + 0.095)
    ax.axhline(0.5, ls='--', color='0.6', lw=0.8, zorder=1)
    ax.set_xticks(range(len(E_VARS))); ax.set_xticklabels(E_VARS, fontsize=7)
    ax.set_ylim(0.45, ytop + 0.045); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel('accuracy (mean ± SEM, n = 9)', fontsize=7)
    ax.legend(handles=[Patch(fc='0.35', ec='0.35', label='plane (2-D)'),
                       Patch(fc='w', ec='0.35', hatch='///', label='out-of-plane'),
                       Patch(fc='0.35', alpha=0.35, ec='0.35', label='full space')],
              frameon=False, fontsize=5.6, loc='lower right', bbox_to_anchor=(1.0, 1.01),
              ncols=3, handlelength=1.1, handletextpad=0.4, labelspacing=0.3,
              columnspacing=0.8, borderaxespad=0.0)
    return ax



# ══ ASSEMBLE (paper figure: no suptitle, no footnotes — prose is caption+Methods) ══
#  Three full-width rows (filled back out 2026-08-31 — the slimmed figure read empty next to 2/4):
#  row 0  A  the four codes over time (2 x 4 traces)
#  row 1  B  the frame — 2 (Naive | Expert) x 5 storyboard, each row in its stage's OWN frame
#  row 2  C  axis-geometry |cos| matrices  +  D  per-mouse raw-|cos| strip
# RESTRUCTURED 2026-08-31 (review, user-approved): the PROOF row (bars + per-mouse block) moves UP
# to row 2, directly answering B's "plotted-in-a-plane is by construction" objection; the geometry
# row (cosine matrices + NEW cross-stage 2x2s) closes the figure. The per-mouse cosine scatters
# moved to ED (fig_manifold_supp.py) — their choice x dist subpanel duplicated Fig 4A's starred
# scatter point-for-point.
# Row 1 deliberately SHORTER than its content suggests: the shared y-range is set by the decision
# licks (+9 z), so tall frames leave the mid-delay panels mostly empty — compressing fills the band.
fig = plt.figure(figsize=(12.4, 14.6))
outer = fig.add_gridspec(4, 12, height_ratios=[1.45, 1.55, 1.65, 0.95], hspace=0.28,
                         left=0.062, right=0.982, top=0.978, bottom=0.028, wspace=0.9)
gsT = outer[0, 0:12].subgridspec(2, 4, wspace=0.34, hspace=0.18)
axT = panel_traj(fig, gsT)
gsA = outer[1, 0:12].subgridspec(2, 5, wspace=0.16, hspace=0.10)
axA = panel_a(fig, gsA)
gsC = outer[2, 0:4].subgridspec(1, 1)                # C = the proof: summary bars
axC = panel_f_spaces(fig, gsC)
gsD = outer[2, 4:12].subgridspec(3, 4, wspace=0.24, hspace=0.20)   # D = per-mouse 3x4
axD = panel_e_plane(fig, gsD)
gsE = outer[3, 0:5].subgridspec(1, 2, wspace=0.28)   # E = cosine matrices
axE = panel_b(fig, gsE)
gsF = outer[3, 6:10].subgridspec(1, 2, wspace=0.30)  # F = cross-stage identity 2x2s
axF = panel_xstage(fig, gsF)

plabel(axT, 'A', dx=-0.05); plabel(axA, 'B', dx=-0.10)
plabel(axC, 'C', dx=-0.14); plabel(axD, 'D', dx=-0.34)
plabel(axE, 'E', dx=-0.30); plabel(axF, 'F', dx=-0.24)

# ── CAPTION (justified, drawn below — same mechanism as Fig 2; edit CAP_PARAS + re-render) ──
CAP_PARAS = [
    'Figure 3 | One manifold: a single sample × choice plane is necessary and sufficient for the '
    'memory and choice codes of both tasks — and learning pulls the distractor code into it.',
    'A. The four codes over time (sample / dist / test / choice; Naive top, Expert bottom): per-mouse '
    'cross-validated decoder projections (CCGD), mean ± SEM across mice (n = 9), correct laser-off '
    'trials, one shared per-mouse unit so amplitudes are comparable across codes; y-axes shared '
    'within each code column. Shaded bands: sample, distractor, GNG cue and test epochs. The sample '
    'code is maintained throughout the delay in both stages; the dist and choice codes rise at their '
    'own epochs.',
    'B. The frame itself, Naive (top) and Expert (bottom): the same cross-validated per-mouse '
    'CCGD projections as A — sample axis (x), choice axis (y) — read at one window per panel '
    'and re-centred, per mouse, on that window’s mean state, so each panel shows the condition '
    'GEOMETRY at that moment: which codes are separated, and along which axis. (The common ramp '
    'and the absolute displacement along the trial are carried by the traces in A and quantified '
    'on a fixed axis in Fig. 4B.) Windows: mid-delay 5.5–6.3 s (pre-cue), late delay 7.5–8.8 s '
    '(post-cue), decision 10–11 s — the first second of the response window (the choice axis '
    'itself is trained at the lick moment, 9.5–10.5 s, and read here). Dots = per-mouse '
    'condition means (correct trials, ≥3 per '
    'mouse), ellipse = SEM '
    'across mice, marker = grand mean; fill = lick, open = no-lick; '
    'circle / triangle / square = DPA / Go / NoGo; colour = sample A / B; scale bar 2 z. Read '
    'left to right along the trial: at mid-delay the DPA states separate along the sample axis '
    'only — the choice axis carries nothing; at decision they split along the choice axis; the '
    'dual Go and NoGo states already differ along the choice axis at mid-delay (weakly in '
    'Naive, strongly in Expert — the distractor precedes this window), and after the cue the Go '
    'state sits toward lick (late delay) — in both stages, every separation in every task is '
    'carried by the same two axes.',
    'C. The plane is necessary and sufficient for the memory and choice codes: each variable '
    'decoded from only the 2 coordinates of each mouse’s own sample × choice plane, from the '
    'out-of-plane residual (plane component removed) and from the full population (mean ± SEM, '
    'n = 9 mice, stages averaged; held-out trial halves, each variable at its canonical window). '
    'Paired Wilcoxons, all comparisons drawn: sample — plane = full (n.s.), out-of-plane '
    'collapses (both p = .004); dist — the mirror pattern at partial strength: out-of-plane = '
    'full (n.s.) and the plane carries a real but smaller share (plane-vs-out and plane-vs-full '
    'p = .004); test — plane at chance (both p = .004), out-of-plane = full (n.s.); choice — '
    'plane-vs-out and out-vs-full p = .020. Every ∗ / n.s. is robust across both decoder '
    'pipelines; † marks the one pipeline-dependent comparison (choice plane-vs-full: '
    'p = .94 / .012 — no verdict drawn).',
    'D. The same, per animal: Naive (x) vs Expert (y), rows = plane / out-of-plane / full space, '
    'columns = variables. sample and choice decode as well from the plane as from the full space '
    'and collapse when the plane is removed; test is at chance from the plane yet keeps its full '
    'accuracy without it. Learning changes are n.s. in both decoder pipelines (annotations) with '
    'one robust exception, starred: the dist code’s PLANE-only accuracy grows with learning '
    '(0.57 → 0.65, p = .020 / .027 across pipelines, 8/9 and 7/9 mice up) — per animal, learning '
    'pulls the distractor code into the manifold (cf. Fig. 4A). Residual above-chance decoding '
    'out-of-plane is expected (only the estimated plane is removed; population codes are '
    'redundant).',
    'E. Axis geometry: attenuation-corrected split-half |cos| between the three axes '
    '(pseudo-population). The sample axis is orthogonal to both action codes (|cos| ≈ 0.07–0.09, '
    'both stages); the choice × dist overlap is partial and grows with learning '
    f'({np.asarray(AXF["Naive"]["cos"])[1, 2]:.2f} → {np.asarray(AXF["Expert"]["cos"])[1, 2]:.2f}) — '
    'quantified per animal in Fig. 4A (per-mouse cosine companions in the ED supplement).',
    'F. The frame is the SAME frame across learning: decoders trained in one stage read the other '
    'stage’s activity (registered neurons; held-out trials) at ~90% of the within-stage ceiling — '
    'transfer/within 0.90 (sample) and 0.87 (choice), robust across both decoder pipelines. '
    'Learning moves the state within the frame (Fig. 4B); it does not rotate the frame.',
]
if ANTACT:
    CAP_PARAS = [p + (' [AXIS VARIANT: the choice axis in A (choice trace) and B (y-axis) is '
                      'the ANTICIPATORY action axis — decoders trained over overlaps bins 48–62 '
                      '(anticipatory + action) instead of the action window 57–62. Panels C–F '
                      'are unchanged (pca-side decision-window axis).]'
                      if p.startswith('B. ') else '') for p in CAP_PARAS]
from figcaption import draw_justified                  # shared with fig_dimensionality_main.py
draw_justified(fig, CAP_PARAS)

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_manifold_main{FIGSUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_main{FIGSUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_main{FIGSUF}.png'),
      '(NO PCA in the decoder)' if NOPCA else '')
