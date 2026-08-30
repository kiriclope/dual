"""fig_dimensionality_main.py — Fig 2: one dedicated axis per task variable (minimal, factorised geometry).

ADOPTED 2026-08-10 (user decision): the DECODE build IS main Fig 2 — B/C/D share one grid,
DPA vs dual (x) mid-delay vs decision. Three claims:
  1. B cvPCA reliable spectra (+ leave-one-mouse-out jackknife 95% CIs, SPEC_JK): the memory state is
     a single reliable component; dual adds exactly the distractor axis (~7%); decision ~3 components.
  2. C per-variable DECODING POWER (held-out pseudo-trials along each variable's demixed axis, vs
     shuffle nulls; DPCA_COUNT / DPA_GNG_C): a variable decodes only when in play — the amplitude-free
     existence metric (Kobak et al. 2016), replacing the variance-weighted PR bars.
  3. D eta^2 PC-coding matrices (Expert; DPA then dual, PC1-4 both) + the boxed 'gng x' cross-decode
     column on DPA: the axes ARE the variables; the DPA geometry carries the distractor code only
     weakly. Naive overlaid in B/C; Naive matrices identical (Extended Data).
Windows: mid-delay = bins_MD 36-38 (post-distractor, PRE-cue/PRE-lick), decision = 57-65; B/C/D all
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

LEGACY = '--pr' in sys.argv          # previous PR/all-tasks build (ED source)
CDEC = not LEGACY                    # the adopted main Fig 2

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
CV, FITDATA = RES['CV'], RES['FITDATA']
STAGES = ['Naive', 'Expert']
SC = {'Naive': '0.55', 'Expert': '#332288'}
VAR_COL = {'sample': '#332288', 'test': '#377eb8', 'choice': '#4daf4a', 'tasks': '#cc3311', 'gng': '#ee7733'}


def plabel(ax, s):
    ax.text(-0.06, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')


# ══ A — schematic: trial timeline, the two read-out states, and the cvPCA (repeated 2-fold CV) logic ══
def schematic(ax):
    ax.set_xlim(0, 14); ax.set_ylim(0, 1); ax.axis('off')
    y0, h = 0.82, 0.10                                                     # timeline bar
    ax.add_patch(Rectangle((0, y0), 14, h, fc='#f4f4f4', ec='0.5', lw=0.7))
    # data epochs (s): sample 2-3 | distractor 4.5-5.5 | MD 5.5-6.5 | GNG cue 6.5-7, reward 7-7.5 |
    # LD 7.5-9 | test 9-10.  The GNG cue/lick is AFTER the mid-delay window — show it, or the
    # "pre-cue, pre-lick" justification for MD is invisible to the reader.
    for nm, lo, hi, col in [('sample', 2.0, 3.0, VAR_COL['sample']), ('distractor', 4.5, 5.5, VAR_COL['tasks']),
                            ('GNG cue', 6.5, 7.5, VAR_COL['gng']),   # longer label hits 'distractor'
                            ('test', 9.0, 10.0, VAR_COL['test']), ('lick', 10.0, 11.5, VAR_COL['choice'])]:
        ax.add_patch(Rectangle((lo, y0), hi - lo, h, fc=col, alpha=0.75, lw=0))
        ax.text((lo + hi) / 2, y0 + h + 0.025, nm, ha='center', va='bottom', fontsize=5.8, color=col)
    ax.text(0.1, y0 + h + 0.025, 'trial', ha='left', va='bottom', fontsize=5.8, color='0.4')
    brackets = ([(5.6, 6.4, 'memory / delay state (5.5–6.3 s)', 'right', 5.5),   # bins_MD 36–38
                 (9.5, 10.8, 'decision state', 'left', 9.6)] if CDEC else       # keep SHORT: a longer
                [(8.0, 8.9, 'memory / delay state', 'right', 7.9),      # legacy: late delay
                 (9.5, 11.0, 'decision state', 'left', 10.0)])
    for lo, hi, lab, hal, xt in brackets:
        ax.plot([lo, lo, hi, hi], [y0 - 0.015, y0 - 0.045, y0 - 0.045, y0 - 0.015], color='0.25', lw=0.9)
        ax.text(xt, y0 - 0.065, lab, ha=hal, va='top', fontsize=5.8, color='0.25')
    ax.text(0.1, 0.56, 'pseudo-population: 3,319 neurons\n× 12 conditions', ha='left', va='center', fontsize=6.0)
    for yb, lab, res in [(0.36, 'trial half 1', 'PCA basis'), (0.20, 'trial half 2', 'cross-projected variance')]:
        ax.add_patch(Rectangle((0.6, yb - 0.06), 3.1, 0.12, fc='#e8e6f0', ec='0.5', lw=0.7))
        ax.text(2.15, yb, lab, ha='center', va='center', fontsize=5.8)
        ax.annotate('', xy=(6.0, yb), xytext=(3.9, yb), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='k'))
        ax.text(6.3, yb, res, ha='left', va='center', fontsize=5.8)
    ax.text(0.6, 0.05, 'repeated 2-fold CV (30 random half-splits,\nboth directions averaged): only variance\n'
            'that REPLICATES across halves counts (cvPCA)',
            ha='left', va='center', fontsize=6.0, style='italic', color='0.35')


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
        ax.set_title(ttl, loc='left', fontsize=7)
        if c == 0:
            ax.set_ylabel('reliable variance\n(fraction)')
        else:
            ax.tick_params(labelleft=False)
        if c == 1:
            ax.set_xlabel('cvPCA component')
        if c == 2:
            ax.legend(frameon=False, fontsize=5.5, handlelength=1.2, loc='upper right')
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
        cis = np.array([PJ[(ts, wn, stage)]['ci'] for ts, wn, _ in groups])
        xj = xp + (j - 0.5) * 0.32
        ax.bar(xj, prs, 0.30, color=SC[stage], label=stage)
        ax.vlines(xj, cis[:, 0], cis[:, 1], color='0.25', lw=0.9)
        for x, (lo, hi) in zip(xj, cis):
            ax.hlines([lo, hi], x - 0.05, x + 0.05, color='0.25', lw=0.9)
        for x, v, chi in zip(xj, prs, cis[:, 1]):
            ax.text(x, chi + 0.09, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(xp); ax.set_xticklabels([g[2] for g in groups], fontsize=7)
    ax.set_ylim(0, 4.6); ax.set_ylabel('participation ratio')
    if show_title:
        ax.set_title('one dimension per variable in play', loc='left', fontsize=TITLE_FS)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    ax.text(0.03, 0.76, 'error bars: 95% CI,\njackknife across mice (n=9)', transform=ax.transAxes,
            ha='left', va='top', fontsize=5.8, color='0.35')
    for ts, wn, _ in groups:
        for st in STAGES:
            P = PJ[(ts, wn, st)]
            print(f'C: {ts:4s} {wn:9s} {st:6s} PR={P["pr"]:.2f} jkSE={P["se"]:.3f} '
                  f'CI [{P["ci"][0]:.2f}, {P["ci"][1]:.2f}]')


# ══ B (--cdecode) — cvPCA reliable spectra in the SAME grid as C and D: DPA vs dual (cols) ×
#     mid-delay vs decision (rows). Mid-delay (bins_MD 36-38, post-distractor PRE-cue/PRE-lick — the
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
                ax.set_title(ts, loc='left', fontsize=7)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('component', fontsize=7)
            if c == 1:
                ax.tick_params(labelleft=False)
                ax.text(0.90, 0.94, wlab, transform=ax.transAxes, ha='right', va='top',
                        fontsize=6.5, color='0.35', style='italic')
            # geometry callouts + cartoons (the point of evidence, not the caption)
            if r == 0 and c == 0:                    # DPA mid-delay: ONE axis — a sample line
                ax.text(0.97, 0.95, '1 reliable axis —\nthe sample line', transform=ax.transAxes,
                        ha='right', va='top', fontsize=5.8, color='0.25')
                ax.plot([0.62, 0.93], [0.60, 0.70], '-', color='0.45', lw=1.0, transform=ax.transAxes)
                ax.plot([0.62], [0.60], 'o', ms=3.2, color='#332288', transform=ax.transAxes, clip_on=False)
                ax.plot([0.93], [0.70], 'o', ms=3.2, color='#44AA99', transform=ax.transAxes, clip_on=False)
                ax.text(0.62, 0.53, 'A', transform=ax.transAxes, fontsize=5, color='#332288', ha='center', va='top')
                ax.text(0.93, 0.63, 'B', transform=ax.transAxes, fontsize=5, color='#44AA99', ha='center', va='top')
            if r == 0 and c == 1:                    # dual mid-delay: 2 axes = distractor × sample.
                # VERIFIED by projecting held-out cond means on the cvPCA basis (2026-08-12): comp1
                # (0.93) carries gng η²=0.99, comp2 (0.07) carries sample η²=0.78 — the DISTRACTOR
                # dominates and the memory line survives as the small axis. Do NOT swap these.
                ax.text(0.96, 0.84, '2 axes: distractor (0.93)\n× sample (0.07)', transform=ax.transAxes,
                        ha='right', va='top', fontsize=5.8, color='0.25')
                ax.add_patch(Polygon([(0.54, 0.50), (0.86, 0.57), (0.96, 0.72), (0.64, 0.65)],
                                     closed=True, fc='0.93', ec='0.6', lw=0.6,
                                     transform=ax.transAxes, zorder=1))
                ax.plot([0.59, 0.89], [0.535, 0.62], '-', color='0.45', lw=1.0, transform=ax.transAxes)
                ax.plot([0.59], [0.535], 'o', ms=2.8, color='#332288', transform=ax.transAxes)
                ax.plot([0.89], [0.62], 'o', ms=2.8, color='#44AA99', transform=ax.transAxes)
                ax.annotate('', xy=(0.71, 0.705), xytext=(0.65, 0.555),
                            arrowprops=dict(arrowstyle='-|>', lw=0.9, color=VAR_COL['gng']),
                            xycoords=ax.transAxes, textcoords=ax.transAxes)
                ax.text(0.725, 0.70, 'gng', transform=ax.transAxes, fontsize=5,
                        color=VAR_COL['gng'], ha='left', va='center')
            if r == 1:                               # decision: ~3 reliable axes
                ax.text(0.96, 0.94 if c == 0 else 0.84, '≈3 reliable axes', transform=ax.transAxes,
                        ha='right', va='top', fontsize=5.8, color='0.25')
            if r == 1 and c == 0:
                ax.legend(frameon=False, fontsize=5.2, handlelength=1.3, loc='center right')
    p0, p3 = axes[0].get_position(), axes[2].get_position()
    fig.text(p0.x0 - 0.028, (p3.y0 + p0.y1) / 2, 'reliable variance (fraction)',
             rotation=90, va='center', ha='center', fontsize=8)
    return axes[0]


# ══ C (--cdecode) — DECODING POWER per variable, DPA vs dual × delay vs decision: held-out
#     pseudo-trial accuracy along each variable's demixed axis (exp_dpca_count.py, Kobak-style).
#     Amplitude-free existence metric: replaces the variance-weighted PR bars. ══
def panelC_decode(fig, gsC):
    DC = RES['DPCA_COUNT']; GC = RES['DPA_GNG_C']
    setsvars = [('DPA', ['sample', 'gng', 'test', 'choice']), ('dual', ['sample', 'gng', 'test', 'choice'])]
    gap = 0.9
    xpos, x = {}, 0.0
    for sname, vs in setsvars:
        for v in vs:
            xpos[(sname, v)] = x; x += 1.0
        x += gap
    xdiv = xpos[('dual', 'sample')] - (gap + 1.0) / 2.0 + 0.5
    axs = []
    for r, (wn, wlab) in enumerate([('md', 'mid-delay'), ('decision', 'decision')]):
        ax = fig.add_subplot(gsC[r, 0]); axs.append(ax)
        for sname, vs in setsvars:
            for v in vs:
                xb = xpos[(sname, v)]
                if sname == 'DPA' and v == 'gng':      # CROSS-decode: gng from the DPA-state subspace
                    d, dn = GC[(wn, 'Expert')], GC[(wn, 'Naive')]
                    ax.bar(xb, d['acc'], 0.72, facecolor='none', edgecolor=VAR_COL['gng'],
                           hatch='/////', lw=0.9, zorder=2)
                else:
                    d = DC[(sname, wn, 'Expert')][v]; dn = DC[(sname, wn, 'Naive')][v]
                    ax.bar(xb, d['acc'], 0.72, color=VAR_COL[v], zorder=2)
                ax.hlines(d['null95'], xb - 0.36, xb + 0.36, color='0.15', lw=0.8, zorder=3)
                ax.plot(xb, dn['acc'], 'o', ms=2.8, mfc='w', mec='0.3', mew=0.7, zorder=4)
                if dn['sig'] and not d['sig']:                 # naive-only signal (the bias state)
                    ax.text(xb + 0.15, dn['acc'] + 0.01, '†', fontsize=7, color='0.25',
                            ha='left', va='bottom', zorder=5)
                print(f'C-dec: {wn:9s} {sname:4s} {v:6s} E {d["acc"]:.2f} (n95 {d["null95"]:.2f})'
                      f'{" *" if d["sig"] else "  "} N {dn["acc"]:.2f}{" *" if dn["sig"] else ""}')
        if r == 0:                                   # the salience fix: 0.61 is WEAK next to dual's 1.0
            ax.annotate('weak transfer\n(dual gng = 1.0)', xy=(xpos[('DPA', 'gng')] + 0.30,
                        GC[('md', 'Expert')]['acc']), xytext=(xpos[('DPA', 'gng')] + 1.05, 0.80),
                        fontsize=5.2, color=VAR_COL['gng'], ha='left', va='center',
                        arrowprops=dict(arrowstyle='-', lw=0.6, color=VAR_COL['gng'],
                                        shrinkA=0, shrinkB=1))
        ax.axhline(0.5, color='0.6', lw=0.7, ls='--', zorder=1)
        ax.axvline(xdiv, color='0.85', lw=0.7)
        ax.set_ylim(0.35, 1.04); ax.set_yticks([0.5, 0.75, 1.0]); ax.set_yticklabels(['0.5', '', '1.0'])
        ax.set_xlim(-0.7, x - gap - 0.3)
        ax.text(0.985, 0.965, wlab, transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='0.35', style='italic')
        if r == 0:
            ax.tick_params(labelbottom=False); ax.set_xticks([])
        else:
            ax.set_xticks([xpos[k] for k in xpos])
            ax.set_xticklabels(['gng cross' if k == ('DPA', 'gng') else k[1] for k in xpos],
                               fontsize=6.0, rotation=42, ha='right')   # NOT '×': rotated it reads '+'
            for sname, vs in setsvars:
                xc = np.mean([xpos[(sname, v)] for v in vs])
                ax.text(xc, -0.44, sname, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=6.5, color='0.2')
    hs = [Patch(fc='0.45', label='Expert'),
          mlines.Line2D([], [], marker='o', ls='', ms=2.8, mfc='w', mec='0.3', mew=0.7, label='Naive'),
          mlines.Line2D([], [], color='0.15', lw=0.8, label='null 95%'),
          Patch(fc='none', ec=VAR_COL['gng'], hatch='/////', label='gng cross-dec ← DPA PCs')]
    axs[0].legend(handles=hs, frameon=False, fontsize=5.2, loc='upper left', ncols=2,
                  handlelength=1.1, handletextpad=0.4, labelspacing=0.3, columnspacing=0.7,
                  borderaxespad=0.15)
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    fig.text(p0.x0 - 0.032, (p1.y0 + p0.y1) / 2, 'held-out decoding accuracy',
             rotation=90, va='center', ha='center', fontsize=8)
    return axs[0]


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
    """# leading components whose LOMO-jackknife CI stays above 1% reliable variance (= panel B's
    reliable rank: DPA md 1, dual md 2, DPA/dual decision 3). Drives the panel-D fade."""
    lo = np.asarray(RES['SPEC_JK'][(ts, wn, 'Expert')]['lo'])
    r = 0
    for v in lo:
        if v > 0.01:
            r += 1
        else:
            break
    return max(r, 1)


def panelD_mats(fig, gsD):
    axes = []
    for c, (ts, wn, ttl) in enumerate(D_SPECS):
        ax = fig.add_subplot(gsD[0, c]); axes.append(ax)
        F = FITDATA[(ts, wn, 'Expert')]
        nk = 4 if (ts == 'dual' or CDEC) else 3
        M = np.asarray(F['pceta'])[:nk]; FO = list(F['factors']); cmv = np.asarray(F['cm_var'])[:nk]
        rk = _rank_b(ts, wn) if CDEC else nk
        if CDEC and ts == 'DPA':                    # gng CROSS-decode column (DPA_GNG, above-chance frac)
            g = np.asarray(RES['DPA_GNG'][(wn, 'Expert')])[:nk]
            M = np.insert(M, 1, g, axis=1); FO = FO[:1] + ['gng ×\n(cross-dec)'] + FO[1:]
        ax.imshow(M, cmap='Purples', vmin=0, vmax=1, aspect='equal')
        if CDEC and ts == 'DPA':
            ax.add_patch(Rectangle((0.5, -0.5), 1.0, nk, fill=False,
                                   edgecolor=VAR_COL['gng'], lw=1.0, zorder=4, clip_on=False))
        ax.set_anchor('NW')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                veiled = i >= rk and not (CDEC and ts == 'DPA' and j == 1)
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.2,
                        color='0.62' if veiled else ('w' if M[i, j] > 0.55 else 'k'))
        if CDEC and rk < M.shape[0]:                # fade rows beyond B's reliable rank (future = noise);
            spans = ([(-0.5, 1.0), (1.5, M.shape[1] - 2.0)] if ts == 'DPA'     # keep the boxed gng×
                     else [(-0.5, float(M.shape[1]))])                          # column readable
            for x0, wdt in spans:
                ax.add_patch(Rectangle((x0, rk - 0.5), wdt, M.shape[0] - rk,
                                       fc='white', alpha=0.55, ec='none', zorder=2.5))
            ax.plot([-0.5, M.shape[1] - 0.5], [rk - 0.5] * 2, color='0.3', lw=0.8,
                    ls=(0, (3, 2)), zorder=4, clip_on=False)
        ax.set_xticks(range(len(FO))); ax.set_xticklabels(FO, fontsize=6.6)
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([f'PC{k+1} ({cmv[k]:.0%})' for k in range(M.shape[0])], fontsize=6.0)
        if CDEC:
            for i, tl in enumerate(ax.get_yticklabels()):
                if i >= rk:
                    tl.set_color('0.55')
        ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
        for sp in ax.spines.values():
            sp.set_visible(True)
    p = axes[0].get_position()
    fig.text(0.014, (p.y0 + p.y1) / 2, 'Expert\n(PC coding, η²)', rotation=90,
             va='center', ha='center', fontsize=7.5, fontweight='bold')
    return axes[0]


# ══ ASSEMBLE ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.6, 6.2))
# NO suptitle / NO footnotes: this is a paper figure — all prose lives in the caption + Methods.
# wspace must leave each block's gap >= shared y-label + tick labels (~0.035 fig width) or the
# first-row panels collide; 1.0 at 12 columns gives ~0.039.
gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 0.88], hspace=0.42, wspace=1.0,
                      left=0.076, right=0.978, top=0.945, bottom=0.055)

axSch = fig.add_subplot(gs[0, 0:4])

schematic(axSch)
if CDEC:
    gsB2 = gs[0, 4:9].subgridspec(2, 2, wspace=0.20, hspace=0.25)
    axB0 = panelB_sets(fig, gsB2)
else:
    gsB = gs[0, 4:9].subgridspec(1, 3, wspace=0.22)
    axB0 = panelB(fig, gsB)
if CDEC:
    gsC = gs[0, 9:12].subgridspec(2, 1, hspace=0.18)
    axC = panelC_decode(fig, gsC)
else:
    axC = fig.add_subplot(gs[0, 9:12])
    panelC(axC)
gsD = gs[1, 0:12].subgridspec(1, 4, wspace=0.70,
                              width_ratios=[4, 4, 4, 4] if CDEC else [4, 4, 3.2, 3.2])
axD0 = panelD_mats(fig, gsD)

plabel(axSch, 'A'); plabel(axB0, 'B'); plabel(axC, 'C'); plabel(axD0, 'D')

OUT = 'figures/pseudo/dimensionality'
STEM = 'fig_dimensionality_main' if CDEC else 'fig_dimensionality_main_pr'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/{STEM}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/{STEM}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/{STEM}.png'))
