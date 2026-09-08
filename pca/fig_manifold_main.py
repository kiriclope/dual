"""fig_manifold_main.py — Fig 3: ONE manifold, seen in a fixed frame.

RESTRUCTURED 2026-08-30 (user decision, "redistribute"): this figure now carries ONLY the frame
itself — the four codes in time, the task states positioned in the sample x choice plane, and the
axis geometry. The LEARNING panels moved to Fig 4 (dist<->choice cross-decode matrices + the two
per-mouse learning scatters — see overlaps/fig_overlaps_main_native.py panel A); the cross-task
GENERALISATION matrices moved to Fig 2 (fig_dimensionality_main.py panel E); the per-mouse
generalisation row and the CCGP panel moved to ED (fig_manifold_supp.py). Canonical pipeline =
no-PCA (--nopca, unsuffixed filenames); the PCA-20 build is the ED robustness variant.

  A  the TWO FRAME AXES read per task (Naive | Expert rows x DPA|Go|NoGo x sample/choice),
     replayed from the overlaps CCGD projections (ORIG_TRACES @go/@nogo keys). ADOPTED
     2026-08-31 (user): A and B are then the SAME data — time courses vs window snapshots.
     The four-code 2x4 row (dist + test axes) moved to fig_manifold_supp.py panel A.
  B  the frame itself — the SAME per-mouse CCGD projections as A (FRAME_STATES /
     exp_frame_states.py), read at one window per panel and re-centred per mouse on that window's
     cross-condition mean: a 2 (Naive | Expert) x 5 storyboard (DPA·md -> DPA·decision -> dual·md
     -> dual·late -> dual·decision) of condition GEOMETRY only — dots = per-mouse condition means,
     SEM ellipses; the absolute ramp/push lives in A and Fig 4B. Metric: offsets are in z units.
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

Reads caches only (no 20 GB X, no overlaps tensor): everything comes from pca results.pkl
(FRAME_STATES / ORIG_TRACES / PM_PLANE / AXIS_FRAME / XSTAGE_DEC).
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
FIG2AX = '--fig2axes' in sys.argv[1:]              # Fig-2 windows for the CCGD axes (sample 36-38, choice 57-65)
EVWIN = '--evwin' in sys.argv[1:]                  # event windows (sample 33-38, choice/test 54-59) — pca caches from DUAL_RES
PCABINS = '--pcabins' in sys.argv[1:]              # pca bins in both pipelines (sample 36-38, choice/test 57-59) — DUAL_RES
AXENV = __import__('os').environ.get('DUAL_AXSUF', '')   # env-driven variant (see main_panels)
ASUF = ('_antact' if ANTACT else ('_f2' if FIG2AX else ('_ev' if EVWIN else ('_pb' if PCABINS else '')))) + AXENV
FIGSUF = ('' if NOPCA else f'_pca{NPC}') + ASUF    # filename: plain = no denoising

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.3      # print-scale: typography sized for 183 mm reproduction (2026-09-02)
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
RES = pickle.load(open(__import__('os').environ.get('DUAL_RES', 'figures/pseudo/dimensionality/results.pkl'), 'rb'))   # DUAL_RES = variant cache
AXF = RES['AXIS_FRAME' + SUF]
assert 'PM_PLANE' + SUF in RES, ('missing PM_PLANE' + SUF +
                                 ' — run: python exp_permouse_plane.py' +
                                 (' --nopca' if NOPCA else ''))
PMPL = RES['PM_PLANE' + SUF]                       # (plane, full, out-of-plane) per mouse/stage/var
assert 'XSTAGE_DEC' + SUF in RES, ('missing XSTAGE_DEC' + SUF +
                                   ' — run: python exp_plane_frame.py' +
                                   (' --nopca' if NOPCA else ''))
XSD = RES['XSTAGE_DEC' + SUF]                      # cross-stage decoding (train x test stage)
PMC = RES['PM_COS' + SUF]                          # per-mouse raw |cos| (E scatters; exp_permouse_frame.py)
assert 'PM_XSTAGE' + SUF in RES, ('missing PM_XSTAGE' + SUF +
                                  ' — run: python exp_permouse_xstage.py' +
                                  (' --nopca' if NOPCA else ''))
PMX = RES['PM_XSTAGE' + SUF]                       # per-mouse cross-stage decoding (F scatters)
# (fits_inputs.pkl is no longer read here — the storyboard replays FRAME_STATES, and every other
#  panel reads a results.pkl cache; the raw-tensor machinery lives in the exp_* scripts)


def plabel(ax, s, dx=-0.06):
    ax.text(dx, 1.05, s.lower(), transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


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
B_SPECS = [('DPA', 'md'), ('DPA', 'decision'), ('dual', 'md'), ('dual', 'decision')]
#   (the dual late-delay column was removed 2026-09-08, Leon)

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
    per-mouse condition means (>=3 trials), SEM ellipse across mice, marker = the grand mean,
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
                ax.scatter(*P.mean(0), marker=TASKM[cd[0]], s=34,
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
                        fontsize=PS*6.0, color='0.3')
                if j == 2:                           # one shared x-label, centred under the grid
                    ax.set_xlabel('sample axis   A ← · → B', fontsize=PS*7)
            if j == 0:
                yax = 'antic. action axis' if ANTACT else 'choice axis'
                ax.set_ylabel(f'{stage}\n{yax}\n← no-lick · lick →', fontsize=PS*7)
            if r == 0 and j == 0:                    # colour/fill key once
                hs = [mlines.Line2D([], [], marker='o', ls='', ms=4, color=SAMPC[0], label='sample A'),
                      mlines.Line2D([], [], marker='o', ls='', ms=4, color=SAMPC[1], label='sample B'),
                      mlines.Line2D([], [], marker='o', ls='', ms=4, mfc='0.4', mec='0.4', label='lick'),
                      mlines.Line2D([], [], marker='o', ls='', ms=4, mfc='w', mec='0.4', label='no-lick')]
                ax.legend(handles=hs, frameon=False, fontsize=PS*6.0, loc='upper left', ncols=2,
                          handletextpad=0.15, columnspacing=0.5, labelspacing=0.25, borderaxespad=0.0)
            if r == 0 and j == 2:                    # marker key on the first dual panel
                hs = [mlines.Line2D([], [], marker='^', ls='', ms=4, mfc='none', mec='0.3', label='Go'),
                      mlines.Line2D([], [], marker='s', ls='', ms=4, mfc='none', mec='0.3', label='NoGo')]
                ax.legend(handles=hs, frameon=False, fontsize=PS*6.0, loc='upper left', ncols=2,
                          handletextpad=0.15, columnspacing=0.5, borderaxespad=0.0)
            if r == 0 and j == 0:                    # crosshair semantics (Codex review 2026-09-01:
                ax.text(0.98, 0.03, 'window-centred', transform=ax.transAxes, ha='right',
                        va='bottom', fontsize=PS*6.0, color='0.45', style='italic')
                #  the zero lines invite a decision-boundary reading; they are the window mean)
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
TBIN = lambda b: np.asarray(b) / 6.0          # bin b = [b/6, (b+1)/6) s from trial start — same clock as Figs 1/2/6 (the old −0.5 s shift was wrong)
EVENTS = [('sample', 2.0, 3.0, SAMPC[0]), ('distractor', 4.5, 5.5, '#cc3311'),
          ('cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]
# cue is 6.5-7.0 s; the reward window 7.0-7.5 s is deliberately unshaded (as everywhere else)


# ══ a — sample & choice codes read PER TASK: DPA | Go | NoGo (ADOPTED 2026-08-31) ══
#   User rationale: rows A and B are then the SAME data — the same per-mouse CCGD sample/choice
#   projections with the same task split; A = time courses, B = window snapshots. The four-code
#   2x4 trace row (with the dist and test axes) moved to fig_manifold_supp.py panel A.
def panel_traj(fig, gsT):
    """2 rows (Naive | Expert) x 6 cols: sample and choice codes per task set (@go/@nogo
    ORIG_TRACES keys, exp_traj_orig.py), y shared per CODE across all task columns so the
    DPA-vs-dual amplitude comparison is direct."""
    TKEY = ('ORIG_TRACES' if NOPCA else 'ORIG_TRACES_pca20') + ASUF
    assert TKEY in RES and ('Naive', 'sample@go', 0) in RES[TKEY], \
        f'missing {TKEY} @go/@nogo keys — run exp_traj_orig.py' + ('' if NOPCA else ' --pca')
    TR = RES[TKEY]; xt = np.asarray(RES['ORIG_XTIME'])
    SCL = (['Odor A', 'Odor B'], [SAMPC[0], SAMPC[1]])
    LCL = (['No lick', 'Lick'], ['#377eb8', '#4daf4a'])
    COLS = [('DPA · sample code', 'sample', *SCL), ('DPA · choice code', 'lick', *LCL),
            ('Go · sample code', 'sample@go', *SCL), ('Go · choice code', 'lick@go', *LCL),
            ('NoGo · sample code', 'sample@nogo', *SCL), ('NoGo · choice code', 'lick@nogo', *LCL)]
    YLg = {}
    for grp, keys in [('sample', ['sample', 'sample@go', 'sample@nogo']),
                      ('choice', ['lick', 'lick@go', 'lick@nogo'])]:
        lo, hi = 0.0, 0.0
        for key in keys:
            for stage in ['Naive', 'Expert']:
                for lv in (0, 1):
                    M = np.asarray(TR[(stage, key, lv)], dtype=float)
                    mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                    lo = min(lo, (mu - se).min()); hi = max(hi, (mu + se).max())
        pad = 0.05 * (hi - lo)
        YLg[grp] = (lo - pad, hi + pad)
    axes = []
    for r, stage in enumerate(['Naive', 'Expert']):
        for k, (ttl, key, labs, cols) in enumerate(COLS):
            ax = fig.add_subplot(gsT[r, k]); axes.append(ax)
            for nm, lo, hi, col in EVENTS:
                ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0)
                if r == 0 and k == 0:
                    yl = 0.905 if nm == 'distractor' else 0.98
                    ax.text((lo + hi) / 2, yl, nm, transform=ax.get_xaxis_transform(),
                            ha='center', va='top', fontsize=PS*6.0, color=col)
            for lv, lab, col in zip((0, 1), labs, cols):
                M = np.asarray(TR[(stage, key, lv)], dtype=float)
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                ax.plot(xt, mu, color=col, lw=1.3, label=f'{lab} (n={len(M)})', zorder=3)
                ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.20, lw=0, zorder=2)
            ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
            ax.set_ylim(*YLg['sample' if 'sample' in key else 'choice'])
            ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12])
            if r == 0:
                ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
                ax.tick_params(labelbottom=False)
                if k == 1:                          # one legend per code type
                    ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.2, loc='upper left')
            else:
                ax.set_xlabel('time (s)', fontsize=PS*7)
                if k == 0:
                    ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.2, loc='lower right')
            ax.set_ylabel(f'{stage}\ncode depth' if k == 0 else 'code depth', fontsize=PS*7)
    return axes[0]


# ══ b — the frame's geometry: corrected |cos| between the three axes ══════════
def panel_b(fig, gsB):
    labs = ['sample', 'choice']                     # dist dropped from the matrices (Leon 2026-09-08)
    axes = []
    for j, stage in enumerate(['Naive', 'Expert']):
        ax = fig.add_subplot(gsB[0, j]); axes.append(ax)
        C3 = np.asarray(AXF[stage]['cos']); C = C3[:2, :2]
        Cm = C.copy(); np.fill_diagonal(Cm, np.nan)
        ax.imshow(np.ma.masked_invalid(Cm), cmap='Oranges', vmin=0, vmax=1, aspect='equal')
        for i in range(2):
            for k in range(2):
                if i == k:                              # 1 by construction — shown, greyed
                    ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, fc='0.93', ec='none'))
                    ax.text(k, i, '1', ha='center', va='center', fontsize=PS*6.4, color='0.45')
                    continue
                ax.text(k, i, f'{C[i, k]:.2f}', ha='center', va='center', fontsize=PS*6.4,
                        color='w' if C[i, k] > 0.55 else 'k')
        ax.set_xticks(range(2)); ax.set_xticklabels(labs, fontsize=PS*6.2, rotation=35, ha='right')
        ax.set_yticks(range(2))
        ax.set_yticklabels(labs if j == 0 else [], fontsize=PS*6.2)
        ax.set_anchor('C')                              # vertically centred with the scatters
        ax.set_title(stage, loc='left', fontsize=TITLE_FS)
        if j == 0:
            ax.set_ylabel('axis geometry\n|cos|', fontsize=PS*7)
        # the attenuation correction divides by sqrt(rel_i*rel_j) — disclose the reliabilities
        # (sample/choice sit at 0.23-0.39; only dist is comfortably high). Review 2026-08-31.
        ax.text(1.0, 1.10, 'rel ' +                      # just above the stage title line
                '/'.join(f'{r:.2f}'.lstrip('0') for r in np.asarray(AXF[stage]['rel'])[:2]),
                transform=ax.transAxes, fontsize=PS*6.0, color='0.3', ha='right', va='bottom')
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'b: {stage} sample-action {C3[0,1]:.2f}  sample-distr {C3[0,2]:.2f}  '
              f'action-distr {C3[1,2]:.2f}  (reliab {np.round(AXF[stage]["rel"],2)})')
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
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=PS*6.6,
                        color='w' if M[i, j] > 0.82 else 'k')
        off = np.mean([M[0, 1], M[1, 0]]); dia = np.mean([M[0, 0], M[1, 1]])
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'], fontsize=PS*6.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Naive', 'Expert'] if k == 0 else [], fontsize=PS*6.2)
        ax.set_title(f'{vn} axis   T/W {(off - .5) / (dia - .5):.2f}', loc='left', fontsize=TITLE_FS)   # ratio in the title (2026-09-08)
        ax.set_anchor('C')                              # vertically centred with the scatters
        if k == 0:
            ax.set_ylabel('train stage', fontsize=PS*7)
        ax.set_xlabel('test stage', fontsize=PS*7)
        if False: ax.text(0.5, -0.44, f'transfer/within {(off - .5) / (dia - .5):.2f}',   # moved into the title
                transform=ax.transAxes, ha='center', va='top', fontsize=PS*6.2, color='0.3')
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'f-xstage: {vn} within {dia:.2f} cross {off:.2f} ratio {(off - .5) / (dia - .5):.2f}')
    return axes[0]


# ══ E right — the same geometry PER ANIMAL: raw |cos| Naive-vs-Expert scatters (PM_COS).
#   RAW (uncorrected) within-mouse cosines — magnitudes attenuated, ordering honest. NO stats
#   drawn: the choice x dist increase is starred in Fig 4A (drawing it here would double-report).
#   (Returned to the main 2026-08-31 at user request; the ED copies were removed.) ══
def panel_e_pm(fig, gs):
    PAIRS = [('sa', 'sample × choice')]   # sample × dist and choice × dist dropped 2026-09-08 (Leon); choice × dist = Fig 4a
    lo, hi = 0.0, 0.25
    axes = []
    for j, (key, lab) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[0, j]); axes.append(ax)
        ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
        nv, ev = [], []
        for m in MICE:
            if (m, 'Naive') not in PMC or (m, 'Expert') not in PMC:
                continue
            a = PMC[(m, 'Naive')][key + '_raw']; b = PMC[(m, 'Expert')][key + '_raw']
            nv.append(a); ev.append(b)
            ax.scatter(a, b, s=34, color=PMCOL[m], marker=PMMARK[PMGROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0, 0.1, 0.2]); ax.set_yticks([0, 0.1, 0.2])
        if j:
            ax.tick_params(labelleft=False)
        ax.set_title(lab, loc='left', fontsize=PS*6.5)
        if j == 0:
            ax.set_ylabel('raw |cos|\nExpert', fontsize=PS*6.8)
        if j == len(PAIRS) // 2:
            ax.set_xlabel('raw |cos| — Naive', fontsize=PS*7)
        print(f'e-pm: {key} raw |cos| {np.mean(nv):.3f} -> {np.mean(ev):.3f}')
    return axes[0]


# ══ F right — cross-stage transfer PER ANIMAL (PM_XSTAGE, exp_permouse_xstage.py): each mouse's
#   own decoder trained in one stage, tested held-out in the other (registered neurons); x =
#   within-stage, y = cross-stage accuracy. Points on the unity line = perfect transfer. NO
#   verdicts — the annotation is the mean chance-referenced transfer/within ratio. ══
def panel_f_pm(fig, gs):
    lo, hi = 0.45, 0.95
    axes = []
    for k, vn in enumerate(['sample', 'choice']):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        wi, cr, rat = [], [], []
        for m in MICE:
            if vn not in PMX.get(m, {}):
                continue
            w, c = PMX[m][vn]['within'], PMX[m][vn]['cross']
            wi.append(w); cr.append(c)
            if w > 0.52:                            # ratio undefined at the chance floor
                rat.append((c - 0.5) / (w - 0.5))
            ax.scatter(w, c, s=34, color=PMCOL[m], marker=PMMARK[PMGROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
        if k:
            ax.tick_params(labelleft=False)
        ax.set_title(vn, loc='left', fontsize=PS*6.5)
        ax.text(0.05, 0.96, f'T/W {np.mean(rat):.2f}', transform=ax.transAxes, va='top',
                ha='left', fontsize=PS*6.0, color='0.3')
        if k == 0:
            ax.set_ylabel('cross-stage\naccuracy', fontsize=PS*6.8)
            ax.set_xlabel('within-stage accuracy', fontsize=PS*7, loc='left')
        print(f'f-pm: {vn} within {np.mean(wi):.2f} cross {np.mean(cr):.2f} '
              f'T/W {np.mean(rat):.2f} (n={len(wi)})')
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
                ax.scatter(a, b, s=34, color=PMCOL[m], marker=PMMARK[PMGROUP[m]],
                           edgecolors='w', linewidths=0.5, zorder=3)
            nv, ev = np.array(nv), np.array(ev)
            p = float(wilcoxon(ev, nv).pvalue)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
            if c:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(vn, loc='left', fontsize=PS*7)
            if r < 2:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(f'{rowlab}\nExpert', fontsize=PS*6.8)
            if r == 2 and c == 1:
                ax.set_xlabel('accuracy — Naive', fontsize=PS*7, loc='right')   # ~grid centre (4 cols)
            # whitelisted verdict (knob-robust .020/.027): dist gains in-plane decodability
            star = '  ∗' if (vn == 'dist' and key == 0 and p < 0.05) else ''
            ax.text(0.05, 0.96, f'Δ={ev.mean() - nv.mean():+.02f}\np={p:.2f}{star}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=PS*6.0 if not star else 6.5, color='0.3' if not star else 'k',
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
                    fontsize=PS*8, fontweight='bold', color='0.4')
            print(f'e-bars: {E_VARS[gi]:7s} {s1}-vs-{s2} p={p:.4f} † (knob-dependent)')
            return
        sig = p < 0.05
        ax.text((x1 + x2) / 2, y + 0.001, '∗' if sig else 'n.s.', ha='center', va='bottom',
                fontsize=PS*12 if sig else 8, fontweight='bold',
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
        bracket(gi, 'plane', 'full', b0 + 0.095)     # 2026-09-08: verdicts only — the pca20 decoder variant is
                                                     # no longer consulted (Leon), so no † anywhere
        ytop = max(ytop, b0 + 0.095)
    ax.axhline(0.5, ls='--', color='0.6', lw=0.8, zorder=1)
    ax.set_xticks(range(len(E_VARS))); ax.set_xticklabels(E_VARS, fontsize=PS*7)
    ax.set_ylim(0.45, ytop + 0.045); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel('accuracy (mean ± SEM, n = 9)', fontsize=PS*7)
    ax.legend(handles=[Patch(fc='0.35', ec='0.35', label='plane (2-D)'),
                       Patch(fc='w', ec='0.35', hatch='///', label='out-of-plane'),
                       Patch(fc='0.35', alpha=0.35, ec='0.35', label='full space')],
              frameon=False, fontsize=PS*6.0, loc='lower right', bbox_to_anchor=(1.0, 1.01),
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
fig = plt.figure(figsize=(12.4, 12.4))
outer = fig.add_gridspec(3, 12, height_ratios=[1.45, 1.55, 1.55], hspace=0.28,
                         left=0.062, right=0.982, top=0.978, bottom=0.028, wspace=0.9)
gsT = outer[0, 0:12].subgridspec(2, 6, wspace=0.42, hspace=0.18)
axT = panel_traj(fig, gsT)
gsA = outer[1, 0:12].subgridspec(2, 4, wspace=0.16, hspace=0.10)
axA = panel_a(fig, gsA)
gsC = outer[2, 0:3].subgridspec(1, 1)                # C = the proof: summary bars
axC = panel_f_spaces(fig, gsC)
# D (cosine matrices + 2 per-mouse scatters) and E (cross-stage 2x2s + 2 per-mouse scatters) share
# the rest of row 2; the per-mouse plane grid (old d) is rendered on its own for ED 6g (2026-09-08).
# All eight axes are aspect-locked squares of the same size on one centre line.
gsR = outer[2, 4:12].subgridspec(2, 1, hspace=0.45)      # D above E, where the old d grid was (col 3 = air for the y-labels)
gsDrow = gsR[0, 0].subgridspec(1, 5, wspace=0.42, width_ratios=[1, 1, 0.15, 1, 1])
gsE = gsDrow[0, 0:2].subgridspec(1, 2, wspace=0.28)      # D = cosine matrices
axE = panel_b(fig, gsE)
gsE2 = gsDrow[0, 3:4].subgridspec(1, 1)                  # D right = per-mouse raw sample × choice |cos| (slot 5 empty, aligned with E)
panel_e_pm(fig, gsE2)
gsErow = gsR[1, 0].subgridspec(1, 5, wspace=0.42, width_ratios=[1, 1, 0.15, 1, 1])
gsF = gsErow[0, 0:2].subgridspec(1, 2, wspace=0.30)      # E = cross-stage identity 2x2s
axF = panel_xstage(fig, gsF)
gsF2 = gsErow[0, 3:5].subgridspec(1, 2, wspace=0.45)     # E right = per-mouse transfer scatters
panel_f_pm(fig, gsF2)

plabel(axT, 'A', dx=-0.05); plabel(axA, 'B', dx=-0.10)
plabel(axC, 'C', dx=-0.14); plabel(axE, 'D', dx=-0.30); plabel(axF, 'E', dx=-0.24)

# ── CAPTION (justified, drawn below — same mechanism as Fig 2; edit CAP_PARAS + re-render) ──
CAP_PARAS = [
    'Figure 3 | One manifold. The two-dimensional sample × choice plane carries the memory and '
    'choice codes on every trial type, excludes the test code, and is the same plane before and after dual task learning. A '
    'code is the projection of population activity onto a per-mouse cross-validated decoder axis '
    '(the decoder never sees the trials it projects), baseline-zeroed, in units of that mouse’s '
    'evoked s.d.; correct laser-off trials; mean ± SEM across nine mice; p values uncorrected.',
    'a, The two axes of the frame, read in each task (columns, DPA | Go | NoGo × sample / choice; '
    'rows, naïve | expert). The DPA sample code is maintained across the delay. On dual trials '
    'the same readout decays after the distractor, the code-morphing signature that follows an '
    'interfering stimulus (lower in 9/9 naïve and 8/9 expert mice); whether the memory itself '
    'survives, or only this readout, is answered by the transfer in Fig. 2e. On the choice axis '
    'the Go trace rises at the cue in both trial classes, a motor and reward transient (every '
    'correct Go trial licks at the cue), and the lick/no-lick split opens only at the test; the '
    'expert NoGo trace runs below baseline through the late delay (7/9 mice), consistent with '
    'active withholding. Distractor and test codes are shown in Extended Data.',
    'b, The same data as geometry. Snapshots of the sample × choice plane at mid-delay (5.5–6.5 '
    's) and decision (10.0–11.2 s, the response window); the choice axis is trained '
    'during the test (9.0–10.5 s). Each panel is re-centered per mouse on the mean state of that window, so '
    'it shows the geometry of the conditions rather than their absolute position (the shared ramp '
    'and the push are carried by a and by Fig. 4b). Dots, per-mouse condition means (at least '
    'three correct trials); ellipses, SEM across mice; large marker, grand mean; filled = lick, '
    'open = no-lick; circle, triangle and square = DPA, Go and NoGo; color = sample; scale bar, '
    '2 z. Whatever the task and the moment, the conditions separate along the same two axes.',
    'c, What lives in the plane. Each variable is decoded from the two '
    'coordinates of the plane, from the residual population after the plane is removed, and from '
    'the full population (mean ± SEM, n = 9, stages averaged; withheld trial halves; paired '
    'Wilcoxon tests, all comparisons drawn). Sample and choice decode as well from the plane as from '
    'the full population, as they must, since the plane is built from their own decoder axes, and '
    'removing the plane reduces but does not abolish their decoding (sample 0.70 → 0.56, p = .004; '
    'choice 0.63 → 0.56, p = .012); the test code is at chance from the plane (0.52 against 0.57 from the full population, p = .055) and untouched without it (p = .82), so '
    'it lives outside the manifold; the distractor’s share is real but partial (p = .004).',
    'd, The memory and choice axes are orthogonal. |cos| between the sample and choice decoder axes, corrected for attenuation using the split-half reliabilities printed above each matrix (0 = orthogonal): 0.06 in naïve and 0.08 in expert mice, the static layer of protection. Right, the raw within-mouse sample × choice |cos|, naïve against expert (below 0.10 in every mouse at both stages). The choice × distractor overlap, which grows with learning, is quantified in Fig. 4a. No tests are drawn here.',
    'e, The frame is fixed across dual task learning. Axes trained in one stage read the withheld activity '
    'of the other stage (registered neurons) at 90% of the within-stage ceiling for the sample and 72% for the choice '
    '(transfer/within 0.90 and 0.72; cross-stage accuracy 0.88 and 0.74 against within-stage 0.92 and 0.83; '
    'robust to resampling and to a common-scaling check, with ratios shifting '
    'by at most 0.02). Right, the same test within each animal (transfer/within 0.86 for sample, 0.59 for choice). This is within-manifold learning: '
    'the state moves inside the frame (Fig. 4b), and the frame does not rotate.',
]
if AXENV:
    CAP_PARAS[0] += (f' [BUILD VARIANT {AXENV}: sample/distractor axes on bins '
                     f'{__import__("os").environ["DUAL_SAMPLE_BINS"]}, choice/test axes on bins '
                     f'{__import__("os").environ["DUAL_CHOICE_BINS"]} in every panel; panel annotations carry this '
                     'build’s statistics, the entries quote the canonical build.]')
if PCABINS:
    CAP_PARAS[0] += (' [BUILD VARIANT _pb: every axis on the pca bins — sample 6.0–6.5 s (post-distractor, pre-cue, '
                     'after the GCaMP rise), choice and test 9.5–10.0 s (second half of the test odor); panel '
                     'annotations carry this build’s statistics, the entries quote the canonical build.]')
if EVWIN:
    CAP_PARAS[0] += (' [BUILD VARIANT _ev: every axis on event windows — sample 5.5–6.5 s (post-distractor, '
                     'pre-cue), choice and test 9.0–10.0 s (test odor); panel annotations carry this build’s '
                     'statistics, the numbers quoted in the entries are the canonical build’s.]')
if ANTACT:
    CAP_PARAS = [p + (' [AXIS VARIANT: the choice axis in a (choice trace) and b (y-axis) is '
                      'the ANTICIPATORY action axis — decoders trained over overlaps bins 48–62 '
                      '(anticipatory + action) instead of the action window 57–62. Panels c–f '
                      'are unchanged (pca-side decision-window axis).]'
                      if p.startswith('b, ') else '') for p in CAP_PARAS]
from figcaption import draw_justified                  # shared with fig_dimensionality_main.py
if '--nocap' not in sys.argv[1:]:   # submission build: legend goes below the figure
    draw_justified(fig, CAP_PARAS, fontsize=PS*7.2)

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_manifold_main{FIGSUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_manifold_main{FIGSUF}.svg', bbox_inches='tight')

# ── ED 6g: the per-mouse plane grid (Fig 3's former panel d), rendered on its own (2026-09-08) ──
figD = plt.figure(figsize=(7.6, 5.6))
gsD2 = figD.add_gridspec(3, 4, wspace=0.10, hspace=0.22, left=0.10, right=0.98, top=0.93, bottom=0.10)
panel_e_plane(figD, gsD2)
figD.savefig(f'{OUT}/png/fig_manifold_permouse_plane{FIGSUF}.png', bbox_inches='tight')
figD.savefig(f'{OUT}/svg/fig_manifold_permouse_plane{FIGSUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_permouse_plane{FIGSUF}.png'))
print('saved', os.path.abspath(f'{OUT}/png/fig_manifold_main{FIGSUF}.png'),
      '(NO PCA in the decoder)' if NOPCA else '')
