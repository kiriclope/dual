"""fig_dimensionality_main.py — Fig 2: one dedicated axis per task variable (minimal, factorised geometry).

ADOPTED 2026-08-10 (user decision): the DECODE build IS main Fig 2 — B/C/D share one grid,
DPA vs dual (x) mid-delay vs decision. Three claims:
  1. B cvPCA reliable spectra (+ leave-one-mouse-out jackknife 95% CIs, SPEC_JK): the memory state is
     a single reliable component; dual adds exactly the GNG axis (~7%); decision ~3 components.
  2. C per-variable DECODING POWER (held-out pseudo-trials along each variable's demixed axis, vs
     shuffle nulls; DPCA_COUNT / DPA_GNG_C): a variable decodes only when in play — the amplitude-free
     existence metric (Kobak et al. 2016), replacing the variance-weighted PR bars. Each stage is
     drawn against ITS OWN null (Expert solid, Naive dashed); the dist-cross verdict uses the
     1000-draw permutation null (2026-08-30; at 100 draws the margin was seed-flippable).
  3. D CROSS-VALIDATED eta^2 PC-coding matrices (Expert; DPA then dual) + the boxed 'dist x' cross-decode
     column on DPA: the axes ARE the variables; the DPA geometry carries the GNG code only
     weakly. Since 2026-09-09 the basis is fit on one trial half and the eta^2 AND the row-label
     percentages are read on the other (exp_pceta_cv.py -> pceta_cv / cm_var_cv), so the labels ARE
     panel b's reliable fractions and non-replicating rows go flat (~1/nfactor) instead of looking
     coded. Naive overlaid in B/C; Naive matrices identical (Extended Data). Row fade rank =
     cumulative-95%-of-reliable-variance rule (see _rank_b). NB the dual eta^2 rows do NOT sum to 1
     exactly (4 of 7 centred contrasts shown; dual-md PC2 leaks ~6% to unshown interactions) —
     caption must not claim they do. Display names are canonical sample/dist/test/choice ('gng'
     is the cache key for dist).
  4. E cross-task generalisation matrices (MOVED FROM Fig 3, 2026-08-30 "redistribute"), timeline
     order sample → test → choice: decoders trained in one task decode the same variable in the
     others — the axes are not just one-per-variable, they are the SAME axes in every task. Cells =
     transferred fraction (cross-0.5)/(within-0.5), column labels carry each test task's within-task
     ceiling; hatched = weak ceiling or ratio>1. Canonical NO-PCA overlaps cache
     (matrices_cache_acc_nopca.pkl). The ratio/hatch key lives in the CAPTION (removed in-figure
     2026-08-31). A dist matrix was built and removed same day (see panelE_gen comment;
     exp_dist_task.py / DIST_TASK keeps the analysis).
  5. F the shared frame is STABLE across learning (added 2026-08-31): per-mouse mean cross-task
     accuracy Naive vs Expert (PM_GEN_nopca), sample/test/choice — points hug unity, all p>=.30 in
     both pipeline variants, pooled bootstrap Δ n.s. Generalisation is in place from the start;
     learning changes the state's position (Fig 4), not the shared geometry. Per-mouse full
     companions in ED (fig_manifold_supp.py).
Windows: mid-delay = bins_MD 36-38 (post-GNG, PRE-cue/PRE-lick), decision = 57-65; B/C/D all
share these two windows. The 'all tasks' set and its context contrasts are OFF this figure.

--pr: the PREVIOUS build (all-tasks spectra + PR bars + jackknife CIs, dual-first D, no gng column)
-> fig_dimensionality_main_pr.{png,svg} — kept as the ED/caption source for the PR numbers.
(A dot-strip-over-PR-bars variant was built and REJECTED 2026-08-10 — don't rebuild it.)

Data: figures/pseudo/dimensionality/results.pkl (CV / FITDATA / PR_JK / SPEC_JK / DPCA_COUNT /
DPA_GNG / DPA_GNG_C — merged caches; no recompute here).

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_dimensionality_main.py
Output: figures/pseudo/dimensionality/{png,svg}/fig_dimensionality_main.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Patch
import matplotlib.lines as mlines

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.15      # print-scale: typography sized for 183 mm reproduction (2026-09-02)
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

LEGACY = '--pr' in sys.argv          # previous PR/all-tasks build (ED source)
CDEC = not LEGACY                    # the adopted main Fig 2

EVWIN = '--evwin' in sys.argv[1:]                 # event-window variant: caches from DUAL_RES, matrices _mdte, stem _ev
PCABINS = '--pcabins' in sys.argv[1:]             # pca-bins variant: caches from DUAL_RES, matrices _pb, stem _pb
AXENV = __import__('os').environ.get('DUAL_AXSUF', '')   # env-driven variant: caches from DUAL_RES, matrices/stem AXENV
RES = pickle.load(open(__import__('os').environ.get('DUAL_RES', 'figures/pseudo/dimensionality/results.pkl'), 'rb'))
CV, FITDATA = RES['CV'], RES['FITDATA']
# panel E (cross-task generalisation) reads the CANONICAL no-PCA overlaps cache — deliberately
# hardcoded (Figs 3-5 are no-PCA canonical; the PCA-20 build is an ED robustness variant)
MAT_CACHE = (f'/home/leon/dual/overlaps/figures/overlaps/ccgp/matrices_cache{AXENV}_acc_nopca.pkl' if AXENV else
             '/home/leon/dual/overlaps/figures/overlaps/ccgp/matrices_cache_pb_acc_nopca.pkl' if PCABINS else
             '/home/leon/dual/overlaps/figures/overlaps/ccgp/matrices_cache_mdte_acc_nopca.pkl' if EVWIN else
             '/home/leon/dual/overlaps/figures/overlaps/ccgp/matrices_cache_acc_nopca.pkl')   # canonical axes (sample 36-38, choice/test 54-62)   # Fig 2 windows: sample @ md, test/choice @ decision (Leon 2026-09-08; was LD/TEST)
assert os.path.exists(MAT_CACHE), (f'missing {MAT_CACHE} — '
                                   'run: cd ../overlaps && python fig_ccgp_matrices_pseudo.py --acc --nopca')
STAGES = ['Naive', 'Expert']
SC = {'Naive': '0.55', 'Expert': '#332288'}
VAR_COL = {'sample': '#332288', 'test': '#377eb8', 'choice': '#4daf4a', 'tasks': '#cc3311', 'gng': '#ee7733'}


def plabel(ax, s):
    ax.text(-0.06, 1.04, s.lower(), transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


# ══ A — schematic: trial timeline, the two read-out states, and the cvPCA (repeated 2-fold CV) logic ══
def schematic(ax):
    ax.set_xlim(0, 14); ax.set_ylim(0, 1); ax.axis('off')
    y0, h = 0.82, 0.10                                                     # timeline bar
    ax.add_patch(Rectangle((0, y0), 14, h, fc='#f4f4f4', ec='0.5', lw=0.7))
    # data epochs (s): sample 2-3 | GNG odor 4.5-5.5 | MD 5.5-6.5 | GNG cue 6.5-7, reward 7-7.5 |
    # LD 7.5-9 | test 9-10.  The GNG cue/lick is AFTER the mid-delay window — show it, or the
    # "pre-cue, pre-lick" justification for MD is invisible to the reader.
    for nm, lo, hi, col in [('sample', 2.0, 3.0, VAR_COL['sample']), ('GNG', 4.5, 5.5, VAR_COL['tasks']),
                            ('cue', 6.5, 7.0, VAR_COL['gng']),   # honest length (cue = 6.5-7.0 s;
                            ('test', 9.0, 10.0, VAR_COL['test']),    #  7.0-7.5 is the reward window)
                            ('lick', 10.0, 11.5, VAR_COL['choice'])]:
        ax.add_patch(Rectangle((lo, y0), hi - lo, h, fc=col, alpha=0.75, lw=0))
        ax.text((lo + hi) / 2, y0 + h + 0.025, nm, ha='center', va='bottom', fontsize=PS*6.0, color=col)
    ax.text(0.1, y0 + h + 0.025, 'trial', ha='left', va='bottom', fontsize=PS*6.0, color='0.4')
    brackets = ([(5.6, 6.4, 'memory / delay state (6.0–6.5 s)', 'right', 5.5),   # bins_MD 36–38
                 (9.0, 10.5, 'decision state\n(9.0–10.5 s)', 'left', 9.1)] if CDEC else   # bins 54–62 (canonical 2026-09-08)
                [(8.0, 8.9, 'memory / delay state', 'right', 7.9),      # legacy: late delay
                 (9.5, 11.0, 'decision state', 'left', 10.0)])
    for lo, hi, lab, hal, xt in brackets:
        ax.plot([lo, lo, hi, hi], [y0 - 0.015, y0 - 0.045, y0 - 0.045, y0 - 0.015], color='0.25', lw=0.9)
        ax.text(xt, y0 - 0.065, lab, ha=hal, va='top', fontsize=PS*6.0, color='0.25')
    ax.text(0.1, 0.56, 'pseudo-population: 3,319 neurons\n× 12 conditions', ha='left', va='center', fontsize=PS*6.0)
    for yb, lab, res in [(0.36, 'trial half 1', 'PCA basis'), (0.20, 'trial half 2', 'cross-projected variance')]:
        ax.add_patch(Rectangle((0.6, yb - 0.06), 3.1, 0.12, fc='#e8e6f0', ec='0.5', lw=0.7))
        ax.text(2.15, yb, lab, ha='center', va='center', fontsize=PS*6.0)
        ax.annotate('', xy=(6.0, yb), xytext=(3.9, yb), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='k'))
        ax.text(6.3, yb, res, ha='left', va='center', fontsize=PS*6.0)
    ax.text(0.6, 0.05, 'repeated 2-fold CV (30 random half-splits,\nboth directions averaged): only variance\n'
            'that REPLICATES across halves counts (cvPCA)',
            ha='left', va='center', fontsize=PS*6.0, style='italic', color='0.35')


# ══ B — cvPCA reliable spectra, one mini-panel per state (1:1 with panel C's bars) ══
def panelB(fig, gsB):
    specs = [('memory\n(DPA delay)', lambda st: FITDATA[('DPA', 'delay', st)]['cv'], None, [1, 2, 3, 4]),
             ('delay\n(all tasks)', lambda st: CV[(st, 'delay')]['cv'], CV[('Expert', 'delay')]['cvn'], [1, 6, 12]),
             ('decision\n(all tasks)', lambda st: CV[(st, 'decision')]['cv'], CV[('Expert', 'decision')]['cvn'], [1, 6, 12])]
    axes = []
    for c, (ttl, get, cvn, xt) in enumerate(specs):
        ax = fig.add_subplot(gsB[0, c]); axes.append(ax)
        for stage in STAGES:
            pos = np.clip(get(stage), 0, None); frac = pos / pos.sum()
            ax.plot(np.arange(1, len(frac) + 1), frac, '-o', ms=2.6, color=SC[stage], label=stage)
            print(f'B: {ttl.splitlines()[0]:9s} {stage:6s} fractions {np.round(frac[:4], 3)}')
        if cvn is not None:
            real_tot = np.clip(get('Expert'), 0, None).sum()
            ax.plot(np.arange(1, len(cvn) + 1), np.clip(cvn, 0, None) / real_tot, '--', color='0.7',
                    lw=1.0, label='null (shuffled)')
        ax.axhline(0, color='0.85', lw=0.6)
        ax.set_xticks(xt); ax.set_ylim(-0.04, 1.06)
        ax.set_title(ttl, loc='left', fontsize=PS*7)
        if c == 0:
            ax.set_ylabel('reliable variance\n(fraction)')
        else:
            ax.tick_params(labelleft=False)
        if c == 1:
            ax.set_xlabel('cvPCA component')
        if c == 2:
            ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.2, loc='upper right')
    return axes[0]


# ══ C — dimensionality = # variables in play: 1 (memory) → 2 (delay) → 3 (decision); Naive ≈ Expert.
#     Error bars = 95% CI from a LEAVE-ONE-MOUSE-OUT JACKKNIFE (mice are the exchangeable unit; the
#     split-half percentiles only measure trial-split stability and are anti-conservative). ══
def panelC(ax, show_title=True):
    PJ = RES['PR_JK']
    groups = [('DPA', 'delay', 'memory\n(DPA delay)'), ('all', 'delay', 'delay\n(all tasks)'),
              ('all', 'decision', 'decision')]
    xp = np.arange(len(groups))
    for j, stage in enumerate(STAGES):
        prs = [PJ[(ts, wn, stage)]['pr'] for ts, wn, _ in groups]
        # 95% CI with the t(8) multiplier for n=9 mice (the cached 'ci' was built with z=1.96,
        # ~15% too narrow) — recomputed here from the stored jackknife SE, no producer rerun
        cis = np.array([[PJ[(ts, wn, stage)]['pr'] - 2.306 * PJ[(ts, wn, stage)]['se'],
                         PJ[(ts, wn, stage)]['pr'] + 2.306 * PJ[(ts, wn, stage)]['se']]
                        for ts, wn, _ in groups])
        xj = xp + (j - 0.5) * 0.32
        ax.bar(xj, prs, 0.30, color=SC[stage], label=stage)
        ax.vlines(xj, cis[:, 0], cis[:, 1], color='0.25', lw=0.9)
        for x, (lo, hi) in zip(xj, cis):
            ax.hlines([lo, hi], x - 0.05, x + 0.05, color='0.25', lw=0.9)
        for x, v, chi in zip(xj, prs, cis[:, 1]):
            ax.text(x, chi + 0.09, f'{v:.1f}', ha='center', va='bottom', fontsize=PS*6.5)
    ax.set_xticks(xp); ax.set_xticklabels([g[2] for g in groups], fontsize=PS*7)
    ax.set_ylim(0, 4.6); ax.set_ylabel('participation ratio')
    if show_title:
            ax.legend(frameon=False, fontsize=PS*6.5, loc='upper left')
    ax.text(0.03, 0.76, 'error bars: 95% CI,\njackknife across mice (n=9)', transform=ax.transAxes,
            ha='left', va='top', fontsize=PS*6.0, color='0.35')
    for ts, wn, _ in groups:
        for st in STAGES:
            P = PJ[(ts, wn, st)]
            print(f'C: {ts:4s} {wn:9s} {st:6s} PR={P["pr"]:.2f} jkSE={P["se"]:.3f} '
                  f'CI [{P["ci"][0]:.2f}, {P["ci"][1]:.2f}]')


# ══ B (--cdecode) — cvPCA reliable spectra in the SAME grid as C and D: DPA vs dual (cols) ×
#     mid-delay vs decision (rows). Mid-delay (bins_MD 36-38, post-GNG PRE-cue/PRE-lick — the
#     clean maintenance window, no consummatory residue) from MD_CHECK; decision from FITDATA. ══
def panelB_sets(fig, gsB2):
    SJ = RES['SPEC_JK']; SN = RES.get('SPEC_NULL', {})
    axes = []
    for r, (wn, wlab) in enumerate([('md', 'mid-delay'), ('decision', 'decision')]):
        for c, ts in enumerate(['DPA', 'dual']):
            ax = fig.add_subplot(gsB2[r, c]); axes.append(ax)
            for j, stage in enumerate(STAGES):
                S = SJ[(ts, wn, stage)]
                frac = np.asarray(S['frac'])
                ks = np.arange(1, len(frac) + 1) + (j - 0.5) * 0.16
                ax.plot(ks, frac, '-o', ms=2.4, color=SC[stage], label=stage)
                ax.vlines(ks, S['lo'], S['hi'], color=SC[stage], lw=0.8, alpha=0.9)
                print(f'B-sets: {ts:4s} {wn:9s} {stage:6s} fractions {np.round(frac[:4], 3)} '
                      f'CI1 [{S["lo"][0]:.2f},{S["hi"][0]:.2f}]')
            if (ts, wn) in SN:                       # label-shuffle null, ÷ the real positive total
                nf = np.clip(np.asarray(SN[(ts, wn)]), 0, None)
                ax.plot(np.arange(1, len(nf) + 1), nf, '--', color='0.7', lw=0.9,
                        label='null (shuffled)', zorder=1)
            ax.axhline(0, color='0.85', lw=0.6)
            ax.set_ylim(-0.05, 1.06); ax.set_yticks([0, 0.5, 1.0])   # short labels: taller axes would
            ax.set_xlim(0.4, 6.6); ax.set_xticks([1, 2, 3, 4, 5, 6])  # auto-add 0.25 steps and collide
            if r == 0:
                ax.set_title(ts, loc='left', fontsize=PS*7)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('component', fontsize=PS*7)
            if c == 1:
                ax.tick_params(labelleft=False)
            ax.text(0.96, 0.94, wlab, transform=ax.transAxes, ha='right', va='top',   # window tag on BOTH
                    fontsize=PS*6.5, color='0.35', style='italic')                     # columns (2026-09-08)
            # geometry callouts + cartoons (the point of evidence, not the caption)
            if r == 0 and c == 0:                    # DPA mid-delay: ONE axis — a sample line
                ax.text(0.96, 0.84, '1 reliable axis —\nthe sample line', transform=ax.transAxes,
                        ha='right', va='top', fontsize=PS*6.0, color='0.25')
                # (the line/plane cartoon glyphs were removed 2026-09-01 — they overlapped the
                #  spectra and were unreadable at panel scale; the text callouts carry the message)
            if r == 0 and c == 1:                    # dual mid-delay: 2 axes = GNG × sample.
                # VERIFIED by projecting held-out cond means on the cvPCA basis (2026-08-12): comp1
                # (0.93) carries gng η²=0.99, comp2 (0.07) carries sample η²=0.78 — the GNG
                # dominates and the memory line survives as the small axis. Do NOT swap these.
                _fd = np.asarray(SJ[('dual', 'md', 'Expert')]['frac'])   # drawn values, not hardcoded
                ax.text(0.96, 0.84, f'2 axes: GNG ({_fd[0]:.2f})\n× sample ({_fd[1]:.2f})',
                        transform=ax.transAxes, ha='right', va='top', fontsize=PS*6.0, color='0.25')
            if r == 1:                               # decision: COUNT the reliable axes (Leon 2026-09-09)
                # was a hardcoded "≈3 reliable axes" on both columns; under the canonical axes only the
                # DPA column has three. An axis counts as reliable when its jackknife lower bound clears
                # the shuffle null, the same comparison the error bars and the dashed null line draw.
                _e = SJ[(ts, 'decision', 'Expert')]; _nl = np.clip(np.asarray(SN[(ts, 'decision')], float), 0, None)
                _lo = np.asarray(_e['lo'], float); _m = min(len(_lo), len(_nl))
                _nax = int(sum(_lo[i] > _nl[i] for i in range(_m)))
                ax.text(0.96, 0.84, f'{_nax} reliable axes', transform=ax.transAxes,
                        ha='right', va='top', fontsize=PS*6.0, color='0.25')
            if r == 1 and c == 0:
                ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.3, loc='center right')
    p0, p3 = axes[0].get_position(), axes[2].get_position()
    fig.text(p0.x0 - 0.028, (p3.y0 + p0.y1) / 2, 'reliable variance (fraction)',
             rotation=90, va='center', ha='center', fontsize=PS*8)
    return axes[0]


# ══ C (--cdecode) — DECODING POWER per variable, DPA vs dual × delay vs decision: held-out
#     pseudo-trial accuracy along each variable's demixed axis (exp_dpca_count.py, Kobak-style).
#     Amplitude-free existence metric: replaces the variance-weighted PR bars. ══
def panelC_decode(fig, gsC):
    """2×2 grid in panel-b format (Leon 2026-09-08): DPA | dual columns × mid-delay | decision rows, each
    its own axes. Expert bars, naive open circles, one null mark per bar (expert 95th pct of the MATCHED
    label-shuffle null), † = naive above its own null (the anticipatory choice; explained in the caption).
    The DPA-subspace GNG cross-decode is printed (→ panel d's orange column), not drawn."""
    DC = RES['DPCA_COUNT']; GC = RES['DPA_GNG_C']
    setsvars = [('DPA', ['sample', 'test', 'choice']), ('dual', ['sample', 'gng', 'test', 'choice'])]
    for wn in ('md', 'decision'):
        for st in STAGES:
            _g = GC[(wn, st)]
            print(f'C-dec: DPA-subspace GNG cross-decode {wn:9s} {st:6s} acc={_g["acc"]:.2f} '
                  f'null95={_g["null95"]:.2f} sig={_g["sig"]} p={_g.get("p", float("nan")):.3f}')
    axes = []
    for r, (wn, wlab) in enumerate([('md', 'mid-delay'), ('decision', 'decision')]):
        for c, (sname, vs) in enumerate(setsvars):
            ax = fig.add_subplot(gsC[r, c]); axes.append(ax)
            for xb, v in enumerate(vs):
                d = DC[(sname, wn, 'Expert')][v]; dn = DC[(sname, wn, 'Naive')][v]
                ax.bar(xb, d['acc'], 0.72, color=VAR_COL[v], zorder=2)
                ax.hlines(d['null95'], xb - 0.36, xb + 0.36, color='0.2', lw=0.8, zorder=3)
                ax.plot(xb, dn['acc'], 'o', ms=2.8, mfc='w', mec='0.3', mew=0.7, zorder=4)
                if dn['sig'] and not d['sig']:                 # naive-only signal (the bias state)
                    ax.text(xb + 0.15, dn['acc'] + 0.01, '†', fontsize=PS*7, color='0.25',
                            ha='left', va='bottom', zorder=5)
                print(f'C-dec: {wn:9s} {sname:4s} {v:6s} E {d["acc"]:.2f} (n95 {d["null95"]:.2f})'
                      f'{" *" if d["sig"] else "  "} N {dn["acc"]:.2f} (n95 {dn["null95"]:.2f})'
                      f'{" *" if dn["sig"] else ""}')
            ax.axhline(0.5, color='0.6', lw=0.7, ls='--', zorder=1)
            ax.set_ylim(0.35, 1.04); ax.set_yticks([0.5, 0.75, 1.0]); ax.set_yticklabels(['0.5', '', '1.0'])
            ax.set_xlim(-0.6, len(vs) - 0.4); ax.set_xticks(range(len(vs)))
            if r == 0:
                ax.set_title(sname, loc='left', fontsize=PS*7)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xticklabels([('GNG' if v == 'gng' else v) for v in vs],
                                   fontsize=PS*6.0, rotation=35, ha='right')
            if c == 1:
                ax.tick_params(labelleft=False)
                ax.text(0.96, 0.94, wlab, transform=ax.transAxes, ha='right', va='top',
                        fontsize=PS*6.5, color='0.35', style='italic')
    hs = [Patch(fc='0.45', label='Expert'),
          mlines.Line2D([], [], marker='o', ls='', ms=2.8, mfc='w', mec='0.3', mew=0.7, label='Naive'),
          mlines.Line2D([], [], color='0.2', lw=0.8, label='null 95%')]
    axes[0].legend(handles=hs, frameon=False, fontsize=PS*5.5, loc='upper right', handlelength=1.1,
                   handletextpad=0.4, labelspacing=0.25, borderaxespad=0.15)   # inside DPA mid-delay: its
    p0, p3 = axes[0].get_position(), axes[2].get_position()                    # test/choice bars sit at chance
    fig.text(p0.x0 - 0.032, (p3.y0 + p0.y1) / 2, 'held-out decoding accuracy',
             rotation=90, va='center', ha='center', fontsize=PS*8)
    return axes[0]


# ══ D — the axes ARE the variables: η² of each condition-mean PC on the factor contrasts (Expert).
#     Adopted build: DPA first (matches B/C), PC1–4 in both sets (DPA PC4 = the degenerate null
#     direction of the 4-condition set, ~0%). Legacy (--pr): dual first, DPA PC1–3. ══
if CDEC:
    D_SPECS = [('DPA', 'md', 'DPA — mid-delay'), ('DPA', 'decision', 'DPA — decision'),
               ('dual', 'md', 'dual — mid-delay'), ('dual', 'decision', 'dual — decision')]
else:
    D_SPECS = [('dual', 'delay', 'dual — delay'), ('dual', 'decision', 'dual — decision'),
               ('DPA', 'delay', 'DPA — delay'), ('DPA', 'decision', 'DPA — decision')]


def _rank_b(ts, wn):
    """Reliable rank for the panel-D fade: the number of leading cvPCA components needed to reach
    95% of the reliable variance, each also exceeding 2x its own label-shuffle level.

    Replaces the old rule (jackknife lo > 1%), which was knife-edge on BOTH knobs: dual-md comp2
    passed by lo=0.012 vs the 1% constant, and switching the CI multiplier from z=1.96 to the
    t8=2.306 appropriate for n=9 flipped dual-md to rank 1 and collapsed DPA-decision 3->1 (its
    comp2 CI spans 0 while comp3 is solidly reliable — stop-at-first-failure). The cumulative rule
    keeps the CI multiplier out of the decision entirely.

    RANKS UNDER THE CANONICAL AXES (re-measured 2026-09-09, post-flip; the pre-flip numbers this
    docstring used to quote were stale): DPA-md 1, DPA-dec 3, dual-md 2, dual-dec 2. Margins:
    dual-md cum1=0.924 < 0.95 < cum2=0.995, dual-dec cum1=0.837 < 0.95 < cum2=0.954 (the tightest,
    0.004). The panel-b annotation uses the STRICTER CI rule instead (count of components whose
    jackknife lower bound clears the shuffle null: DPA-dec 3, dual-dec 2 — same on the decision
    rows it annotates; the two rules differ only on dual-md, 2 here vs 1 there)."""
    S = RES['SPEC_JK'][(ts, wn, 'Expert')]
    frac = np.clip(np.asarray(S['frac']), 0, None)
    null = np.clip(np.asarray(RES['SPEC_NULL'][(ts, wn)]), 0, None)
    r, cum = 0, 0.0
    for i in range(len(frac)):
        if frac[i] <= 2 * null[i]:                  # indistinguishable from the shuffle level
            break
        r += 1; cum += frac[i]
        if cum >= 0.95 * frac.sum():
            break
    return max(r, 1)


DCROSS = False      # 2026-09-08 (Leon): the DPA 'dist × (cross-dec)' column is OUT of panel d — the
                    # GNG-in-the-memory-subspace result lives in Fig 3c (per mouse) and the pooled
                    # number (0.61 @ md, p=.031) is quoted in §3; DPA_GNG stays cached and printed.


def panelD_mats(fig, gsD):
    axes = []
    for c, (ts, wn, ttl) in enumerate(D_SPECS):
        ax = fig.add_subplot(gsD[0, c]); axes.append(ax)
        F = FITDATA[(ts, wn, 'Expert')]
        nk = 3                                       # PC1-3 only (Leon 2026-09-08: PC4 removed — DPA PC4 is
                                                     # the degenerate 0% direction, dual PC4 sits below the rank)
        # CROSS-VALIDATED cells + row labels (Leon 2026-09-09: "we should cross validate panel d").
        # exp_pceta_cv.py fits the PC basis on one trial half and measures BOTH the variance and the
        # eta^2 on the held-out half (30 splits x 2 directions), so the percentages are panel b's
        # reliable fractions and a row that does not replicate goes flat instead of looking coded.
        # The raw keys stay in the pickle as the fallback.
        _cv = 'pceta_cv' in F and 'cm_var_cv' in F
        M = np.asarray(F['pceta_cv' if _cv else 'pceta'])[:nk]
        FO = list(F['pceta_cv_factors'] if _cv else F['factors'])
        cmv = np.asarray(F['cm_var_cv' if _cv else 'cm_var'])[:nk]
        rk = _rank_b(ts, wn) if CDEC else nk
        if DCROSS and ts == 'DPA':                  # dist CROSS-decode column (DPA_GNG, above-chance frac)
            g = np.asarray(RES['DPA_GNG'][(wn, 'Expert')])[:nk]
            M = np.insert(M, 1, g, axis=1); FO = FO[:1] + ['GNG ×\n(cross-dec)'] + FO[1:]
        FO = ['GNG' if f == 'gng' else f for f in FO]   # canonical code names (as in Figs 3-4)
        ax.imshow(M, cmap='Purples', vmin=0, vmax=1, aspect='auto')   # square BOX (Leon 2026-09-08)
        ax.set_box_aspect(1)
        if DCROSS and ts == 'DPA':
            ax.add_patch(Rectangle((0.5, -0.5), 1.0, nk, fill=False,
                                   edgecolor=VAR_COL['gng'], lw=1.0, zorder=4, clip_on=False))
        ax.set_anchor('C')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                veiled = i >= rk and not (DCROSS and ts == 'DPA' and j == 1)
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=PS*6.2,
                        color='0.62' if veiled else ('w' if M[i, j] > 0.55 else 'k'))
        if CDEC and rk < M.shape[0]:                # fade rows beyond B's reliable rank (future = noise);
            spans = ([(-0.5, 1.0), (1.5, M.shape[1] - 2.0)] if (DCROSS and ts == 'DPA')   # keep the boxed gng×
                     else [(-0.5, float(M.shape[1]))])                          # column readable
            for x0, wdt in spans:
                ax.add_patch(Rectangle((x0, rk - 0.5), wdt, M.shape[0] - rk,
                                       fc='white', alpha=0.55, ec='none', zorder=2.5))
            ax.plot([-0.5, M.shape[1] - 0.5], [rk - 0.5] * 2, color='0.3', lw=0.8,
                    ls=(0, (3, 2)), zorder=4, clip_on=False)
        ax.set_xticks(range(len(FO))); ax.set_xticklabels(FO, fontsize=PS*6.6)
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([f'PC{k+1} ({cmv[k]:.0%})' for k in range(M.shape[0])], fontsize=PS*6.0)
        if CDEC:
            for i, tl in enumerate(ax.get_yticklabels()):
                if i >= rk:
                    tl.set_color('0.55')
        ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
        for sp in ax.spines.values():
            sp.set_visible(True)
    p = axes[0].get_position()
    fig.text(0.014, (p.y0 + p.y1) / 2, 'Expert\n(PC coding, η²)', rotation=90,
             va='center', ha='center', fontsize=PS*7.5, fontweight='bold')
    return axes[0]


# ══ E — the axes are SHARED across tasks: cross-task generalisation (from Fig 3, 2026-08-30) ══
#   OFF-DIAGONAL = Nms = (acc - 0.5) / (within-task acc of the TEST task - 0.5): the fraction of the
#   test task's OWN decodable signal that transfers. Normalising by COLUMN is essential — the within-
#   task sample code is ~0.9 in DPA but ~0.6 in Go/NoGo, so a raw cross value of 0.53 is ~90% of what
#   is achievable there, NOT a failure (a flat 0.5-chance reading produced a retracted claim).
#   DIAGONAL = raw within-task accuracy (the ceiling itself), greyed; the column label carries it.
#   Hatch = weak ceiling (<0.10 above chance) OR ratio>1 (both denominator artefacts). Expert only.
def panelE_gen(fig, gsE):
    CC = pickle.load(open(MAT_CACHE, 'rb'))
    TL = list(CC['TLAB'])
    axes = []
    # TASK-TIMELINE order (user, 2026-08-31): sample → test → choice. (A dist matrix was built and
    # REMOVED same day — with no within-task training possible for Go-vs-NoGo, every cell is a
    # transfer through a geometry built without the contrast and none can reach 1, which read as
    # broken next to the ratio matrices. The analysis survives in exp_dist_task.py / DIST_TASK.)
    for j, var in enumerate(['sample', 'test', 'choice']):
        ax = fig.add_subplot(gsE[0, j]); axes.append(ax)
        M = np.asarray(CC['Mms'][('Expert', var)])
        Nn = np.asarray(CC['Nms'][('Expert', var)])
        disp = Nn.copy(); np.fill_diagonal(disp, np.nan)
        ax.imshow(np.ma.masked_invalid(disp), cmap='Reds', vmin=0, vmax=1, aspect='equal')
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if i == k:
                    ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, fc='0.93', ec='none'))
                    ax.text(k, i, '1', ha='center', va='center', fontsize=PS*6.0, color='0.45')
                else:
                    ax.text(k, i, f'{Nn[i, k]:.2f}', ha='center', va='center', fontsize=PS*6.0,
                            color='w' if Nn[i, k] > 0.6 else 'k')
        weak = (np.diag(M) - 0.5) < 0.10
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if i != k and (weak[k] or Nn[i, k] > 1.0):
                    ax.add_patch(Rectangle((k - .5, i - .5), 1, 1, fill=False, hatch='////',
                                           edgecolor='0.45', lw=0.0, zorder=3))
        ax.set_xticks(range(len(TL)))
        ax.set_xticklabels(TL,                        # plain task labels (ceilings → legend; Leon 2026-09-08)
                           fontsize=PS*6.0, rotation=35, ha='right')
        ax.set_yticks(range(len(TL)))
        ax.set_yticklabels(TL if j == 0 else [], fontsize=PS*6.0)
        ax.set_title(var, loc='left', fontsize=PS*7)
        ax.set_anchor('C')
        if j == 0:
            ax.set_ylabel('train', fontsize=PS*7)
        for sp in ax.spines.values():
            sp.set_visible(True)
        # PARALLELISM SCORE (Bernardi's geometric twin of the transfer test; exp_parallelism.py,
        # pipeline-invariant — condition-mean vectors, no decoder). Added 2026-09-01 (craft review).
        _ps = RES['PS_nopca'][('Expert', var)]
        ax.text(0.5, -0.42, f"PS {_ps['raw']:.2f}",             # null → legend (Leon 2026-09-08)
                transform=ax.transAxes, ha='center', va='top', fontsize=PS*6.0, color='0.3')
        EYE = np.eye(len(M), dtype=bool)
        print(f'E-gen: {var:7s} Expert within {np.round(np.diag(M),2)}  transferred frac '
              f'{np.round(Nn[~EYE], 2)}  mean {Nn[~EYE].mean():.2f}  PS {_ps["raw"]:.2f} '
              f'corr {_ps["corrected"]:.2f} null95 {_ps["null95"]:.2f}')
    # (the in-figure key was removed 2026-08-31 — the ratio/hatch explanation is caption/Methods
    #  material; its cell now hosts panel F, the learning-stability scatters)
    return axes[0]


# ══ F — the shared frame is STABLE across learning: per-mouse cross-task accuracy, Naive vs
#     Expert (PM_GEN, canonical no-PCA). Each point = one mouse's mean OFF-DIAGONAL accuracy of
#     its own 3×3 generalisation matrix (raw accuracy on purpose — the per-animal chance-corrected
#     ratio explodes when within-task sits near chance). Points hug unity: generalisation is in
#     place in Naive and learning does not change it (all p≥.30 in BOTH pipeline variants; pooled
#     bootstrap Δ likewise n.s.) — the foil for Fig 4, where learning DOES change the state's
#     position and the dist↔choice coupling. No title verdicts (star policy: whitelist only). ══
F_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
F_GROUP = {**{m: 'Jaws' for m in F_MICE[:5]}, **{m: 'ChR' for m in F_MICE[5:7]},
           **{m: 'ACC' for m in F_MICE[7:]}}
F_GMARK = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
_fpal = sns.color_palette('tab10', n_colors=len(F_MICE))
F_MCOL = {m: _fpal[i] for i, m in enumerate(F_MICE)}


def panelG_biplot(fig, gsG):
    """Per-neuron selectivity biplot (NEURON_SEL, exp_neuron_sel.py; Expert): d' for sample at
    mid-delay (x) vs d' for choice at decision (y), one dot per neuron (n = 3,319). A factorised
    code is a CROSS: neurons selective for one variable or neither, few for both (co-selectivity
    at the independence product). Model-free (condition means / pooled SD) — no decoder shapes
    the cloud. Added 2026-09-01 (craft review: the paper's first neuron-level display)."""
    NS = RES['NEURON_SEL_nopca']['Expert']         # canonical no-PCA, like MAT_CACHE
    ds, dc = NS['ds'], NS['dc']
    ok = np.isfinite(ds) & np.isfinite(dc)
    ax = fig.add_subplot(gsG[0, 0])
    lim = 2.0
    ax.axhline(0, color='0.8', lw=0.6, zorder=0); ax.axvline(0, color='0.8', lw=0.6, zorder=0)
    n95 = NS['null95_abs']
    ax.add_patch(Rectangle((-n95, -n95), 2 * n95, 2 * n95, fc='0.92', ec='none', zorder=0))
    # colour by selectivity class (Leon 2026-09-08: "make the two clouds explicit"): |d'| above the
    # label-shuffle floor for the sample only (indigo), the choice only (green), both (orange) or neither (grey)
    x, yv = np.clip(ds[ok], -lim, lim), np.clip(dc[ok], -lim, lim)
    ssel, csel = np.abs(ds[ok]) > n95, np.abs(dc[ok]) > n95
    both = ssel & csel
    dom_s = np.abs(ds[ok]) >= np.abs(dc[ok])          # two clouds: the axis with the larger |d'| (Leon
    classes = [('sample cloud', dom_s & ~both, VAR_COL['sample'], 0.45, 3.0, 2),   # 2026-09-08: no separate
               ('choice cloud', ~dom_s & ~both, VAR_COL['choice'], 0.45, 3.0, 2),  # 'neither' colour)
               ('both selective', both, '#E69F00', 0.85, 5.0, 3)]
    hs = []
    for lab, msk, col, al, sz, z in classes:
        ax.scatter(x[msk], yv[msk], s=sz, marker='.', color=col, alpha=al, lw=0, zorder=z, rasterized=True)
        hs.append(mlines.Line2D([], [], marker='o', ls='', ms=3.2, color=col,
                                label=(f'{lab} {100 * msk.mean():.0f}%' if lab == 'both selective' else lab)))
        print(f'G: class {lab:15s} n={int(msk.sum()):4d} ({100 * msk.mean():.1f}%)')
    print(f'G: sample-only {int((ssel & ~csel).sum())} choice-only {int((~ssel & csel).sum())} '
          f'neither {int((~ssel & ~csel).sum())} both {int(both.sum())}')
    ax.legend(handles=hs, frameon=False, fontsize=PS*5.5, loc='lower left', bbox_to_anchor=(0.0, 1.02),
              ncols=1, handletextpad=0.2, labelspacing=0.15, borderaxespad=0.0, markerscale=1.0)
    #   one column ABOVE the axes, within the panel width (inside, any key crosses the vertical arm)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.set_xlabel("sample d′ (mid-delay)", fontsize=PS*7)
    ax.set_ylabel("choice d′ (decision)", fontsize=PS*7)
    ax.text(0.03, 0.03, f"r = {NS['r_abs']:+.02f}\nboth {100 * both.mean():.1f}%\nindep. {100 * ssel.mean() * csel.mean():.1f}%", transform=ax.transAxes,
            va='bottom', ha='left', fontsize=PS*6.0, color='0.3')   # lower-left quadrant (few points)
    print(f"G: biplot n={NS['n_ok']} r_abs={NS['r_abs']:+.3f} null95={n95:.2f}")
    return ax


def panelF_gen_learning(fig, gsF):
    from scipy.stats import wilcoxon
    PG = RES['PM_GEN_mddec_nopca']                     # canonical no-PCA per-mouse matrices, Fig 2 windows (md / decision)
    E3 = np.eye(3, dtype=bool)
    lo, hi = 0.45, 0.73
    axes = []
    for j, var in enumerate(['sample', 'test', 'choice']):
        ax = fig.add_subplot(gsF[0, j]); axes.append(ax)
        ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
        ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
        nv, ev = [], []
        for m in F_MICE:
            if (m, 'Naive', var) not in PG or (m, 'Expert', var) not in PG:
                continue
            xn = float(PG[(m, 'Naive', var)][~E3].mean())
            ye = float(PG[(m, 'Expert', var)][~E3].mean())
            nv.append(xn); ev.append(ye)
            ax.scatter(xn, ye, s=34, color=F_MCOL[m], marker=F_GMARK[F_GROUP[m]],
                       edgecolors='w', linewidths=0.5, zorder=3)
        nv, ev = np.array(nv), np.array(ev)
        p = float(wilcoxon(ev, nv).pvalue)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('C')                             # one centre line with E and G
        ax.set_xticks([0.5, 0.6, 0.7]); ax.set_yticks([0.5, 0.6, 0.7])
        if j:
            ax.tick_params(labelleft=False)
        ax.set_title(var, loc='left', fontsize=PS*7)
        if j == 0:
            ax.set_ylabel('cross-task acc.\nExpert', fontsize=PS*7)
        if j == 1:
            ax.set_xlabel('cross-task acc. — Naive', fontsize=PS*7)
        ax.text(0.05, 0.96, f'Δ={ev.mean() - nv.mean():+.2f}\np={p:.2f}', transform=ax.transAxes,
                va='top', ha='left', fontsize=PS*6.0, color='0.3')
        print(f'F: {var:7s} per-mouse cross {nv.mean():.3f} -> {ev.mean():.3f}  p={p:.3f} '
              f'({int((ev > nv).sum())}/{len(nv)} up)')
    return axes[0]


# ══ ASSEMBLE ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.6, 7.9))
# NO suptitle / NO footnotes: this is a paper figure — panel prose lives in the CAPTION drawn below
# (added 2026-08-31, user request) + Methods.
# wspace 1.5 (was 1.0): explicit air between A | B | C (subgridspecs keep their own internal wspace).
# Row 1 is a thin SPACER: it moves D away from the first row without widening the D→E/F gap
# (uniform hspace can't do one-sided spacing).
gs = fig.add_gridspec(4, 12, height_ratios=[1.0, 0.02, 0.70, 0.60], hspace=0.26, wspace=1.5,
                      left=0.076, right=0.978, top=0.955, bottom=0.045)

axSch = fig.add_subplot(gs[0, 0:4])

schematic(axSch)
if CDEC:
    gsB2 = gs[0, 4:9].subgridspec(2, 2, wspace=0.20, hspace=0.25)
    axB0 = panelB_sets(fig, gsB2)
else:
    gsB = gs[0, 4:9].subgridspec(1, 3, wspace=0.22)
    axB0 = panelB(fig, gsB)
if CDEC:
    gsC = gs[0, 9:12].subgridspec(2, 2, wspace=0.20, hspace=0.25, width_ratios=[3, 4])   # panel-b format
    axC = panelC_decode(fig, gsC)
else:
    axC = fig.add_subplot(gs[0, 9:12])
    panelC(axC)
gsD = gs[2, 0:12].subgridspec(1, 4, wspace=0.50,
                              width_ratios=[4, 4, 4, 4] if CDEC else [4, 4, 3.2, 3.2])
axD0 = panelD_mats(fig, gsD)
if CDEC:
    # equal-width slots + centred anchors: E matrices, F scatters and G biplot are all
    # aspect-locked squares of the SAME size on one centre line (user 2026-09-01)
    gsBot = gs[3, 0:12].subgridspec(1, 9, wspace=0.45,
                                    width_ratios=[1, 1, 1, 0.22, 1, 1, 1, 0.22, 1])
    gsE = gsBot[0, 0:3].subgridspec(1, 3, wspace=0.28)   # same internal gap as F ->
    axE0 = panelE_gen(fig, gsE)
    gsF = gsBot[0, 4:7].subgridspec(1, 3, wspace=0.28)   #   same slot width -> same-size
    axF0 = panelF_gen_learning(fig, gsF)
    gsG = gsBot[0, 8:9].subgridspec(1, 1)            #   squares, one shared centre line
    axG = panelG_biplot(fig, gsG)
    plabel(axE0, 'E'); plabel(axF0, 'F'); plabel(axG, 'G')

plabel(axSch, 'A'); plabel(axB0, 'B'); plabel(axC, 'C'); plabel(axD0, 'D')

# ── CAPTION (drawn below the panels; user request 2026-08-31). JUSTIFIED: matplotlib has no
#    native justification, so words are measured with the Agg renderer and the slack is spread
#    across the gaps; the last line of each paragraph stays flush-left (print convention). ──
if CDEC:
    CAP_PARAS = [
        'Figure 2 | The population geometry is minimal and factorized. The working memory occupies a '
        'single dimension, each task variable has its own nearly orthogonal coding axis, and the memory and choice axes are shared across trial types. All panels use the pseudo-population (3,319 '
        'neurons, nine mice, 12 conditions). The memory state is the mid-delay window (6.0–6.5 s, '
        'after the Go/NoGo odor and before any cue or lick); the decision state runs from test onset to 0.5 s after test offset (9.0–10.5 s).',
        'a, Trial timeline, the two analyzed states, and the logic of cross-validated PCA (cvPCA). '
        'Condition means are estimated on one half of the trials and evaluated on the other half (30 '
        'random half-splits, both directions averaged), so only structure that replicates across '
        'independent trial halves counts toward the geometry.',
        'b, The memory manifold is a line. Fraction of reliable condition-mean variance per cvPCA '
        'component (error bars, leave-one-mouse-out jackknife 95% CI, t(8); dashed gray, within-mouse '
        'label-shuffle null). The DPA mid-delay state occupies a single reliable dimension. The dual '
        'tasks add exactly one, the GNG axis (0.92 against sample 0.07), and the decision '
        'state spreads to three reliable axes on DPA trials and two on Go and NoGo trials (an axis counts as '
        'reliable when its jackknife interval clears the shuffle null). Naïve and expert spectra are near-identical; learning does not '
        'change the dimensionality.',
        'c, Each axis carries its variable when, and only when, the task engages it. Decoding accuracy along each demixed coding axis on withheld pseudo-trials (expert, bars; naïve, open circles), against the expert label-shuffle null (95th percentile of a null matched to the plotted statistic, short line). The dagger marks the single exception, an anticipatory choice signal in the naïve mid-delay state (0.66 against its own null) that disappears with learning.',
        'd, The principal components are the task variables. η² of each condition-mean PC against the '
        'design contrasts, cross-validated exactly as in b: the components are fitted on one half of '
        'the trials and both the η² and the row percentages are measured on the other (30 random '
        'half-splits, both directions averaged; components are matched to the full-data axes before '
        'averaging, because two components of nearly equal size otherwise change places from split to '
        'split and their rows blend). Row labels therefore give each matched component’s share of the '
        'reliable variance of b, and a cell near 1 means that the PC codes that variable alone. The '
        'geometry is factorized rather than mixed. Rows beyond the reliable rank of b are faded, and '
        'they are also flat by construction, since a component that does not replicate carries no '
        'coding on held-out trials; dual rows show 4 of the 7 centered contrasts.',
        'e, Cross-task transfer of the decoders (expert; sample at mid-delay, test and choice at decision, the states of b–d; each decoder is trained and tested in the same window). Cells give the transferred fraction of decodable signal, (cross − 0.5)/(within − 0.5); the within-task accuracies are 0.94/0.78/0.80 for the sample, 0.68/0.58/0.56 for the test and 0.86/0.75/0.81 for the choice (DPA/Go/NoGo); hatched cells have a ratio above 1 (cross above within) and are not read as fractions. The choice transfers largely (0.41–0.97), and the test completely (0.53 and above; four of six cells exceed the within-task level, whose accuracies are low). The sample transfer is partial and asymmetric (0.27–0.90): decoders trained on Go or NoGo trials read the DPA trials well (0.76–0.80), whereas the DPA-trained decoder reads the dual trials less well (0.27–0.44), consistent with the shift of the sample readout within the plane after the Go/NoGo odor (Fig. 3a; Extended Data Fig. 6e). Below each matrix is the parallelism score (PS), the geometric twin of the transfer test (sample 0.28, test 0.06, choice 0.16; label-shuffle 95th percentiles 0.04–0.05).',
        'f, The shared frame precedes dual task learning. Per-mouse mean cross-task accuracy (same windows as e), naïve against expert; points on the unity line indicate no change. The sample is unchanged (Δ = 0.00, 95% CI [−0.05, +0.05], Wilcoxon p = 1.00, n = 9), and so are the test (+0.01, [−0.01, +0.03], p = .43) and the choice (+0.01, [−0.03, +0.06], p = .82); the fraction transferred is unchanged (per-mouse medians 0.41–0.88, all p ≥ .65).',
        'g, The factorization is visible neuron by neuron. Per-neuron discriminability (d′, within '
        'mouse) for sample at mid-delay against choice at decision (n = 3,319; gray square, label-'
        'shuffle floor; color, the axis with the larger |d′|, sample in indigo or choice in green; orange, neurons above the floor on both axes). |d′| across the two variables is uncorrelated (r = −0.02), and the fraction '
        'of both-selective neurons (5.2%) equals the independence prediction (5.2%). Largely separate '
        'populations carry the two axes, which is the single-neuron basis of the factorized geometry.',
    ]
    from figcaption import draw_justified              # shared with fig_manifold_main.py
    if AXENV:
        CAP_PARAS[0] += (f' [BUILD VARIANT {AXENV}: sample/GNG axes on bins '
                         f'{__import__("os").environ["DUAL_SAMPLE_BINS"]}, choice/test axes on bins '
                         f'{__import__("os").environ["DUAL_CHOICE_BINS"]} in every panel; panel annotations carry '
                         'this build’s statistics, the entries quote the canonical build.]')
    if PCABINS:
        CAP_PARAS[0] += (' [BUILD VARIANT _pb: sample axis 6.0–6.5 s (post-GNG, pre-cue, after the GCaMP rise), '
                         'choice and test axes 9.5–10.0 s (second half of the test odor) in every panel; panel annotations '
                         'carry this build’s statistics, the entries quote the canonical build.]')
    if EVWIN:
        CAP_PARAS[0] += (' [BUILD VARIANT _ev: sample axis 5.5–6.5 s (post-GNG, pre-cue), choice and test axes '
                         '9.0–10.0 s (test odor) in every panel; panel annotations carry this build’s statistics, the '
                         'numbers quoted in the entries are the canonical build’s.]')
    if '--nocap' not in sys.argv[1:]:   # submission build: legend goes below the figure
        draw_justified(fig, CAP_PARAS, fontsize=PS*7.2)

OUT = 'figures/pseudo/dimensionality'
STEM = ('fig_dimensionality_main' if CDEC else 'fig_dimensionality_main_pr') + ('_ev' if EVWIN else ('_pb' if PCABINS else '')) + AXENV
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/{STEM}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/{STEM}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/{STEM}.png'))
