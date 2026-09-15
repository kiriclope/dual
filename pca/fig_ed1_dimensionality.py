"""fig_ed1_dimensionality.py — Extended Data Fig. 1: dimensionality, provenance and robustness
(companion to Fig. 2b–d). Built 2026-09-15 (Leon: "keep only what is essential for the paper's
argumentation") — every panel here is one the Results, Methods or Discussion cite; nothing else.

  a  the full twelve-condition spectra (Results §2: "the full twelve-condition spectra are given in ED")
  b  the participation-ratio ladder with leave-one-mouse-out CIs (Methods: "on the PR (ED)")
  c  the shattering dimension of the twelve conditions (Results §3: "0.67 at both stages")
  d  the per-mouse cvPCA companion (Discussion: "including the dimensionality spectra themselves, n = 9")
  e  the UNCROSS-VALIDATED eta^2 matrices, DPA delay (Methods: "kept ... as the demonstration")
  f  learning removes the premature choice signal from the dual delay (Results §2, twice)

Reads caches only: results.pkl (CV / FITDATA / PR_JK / PM_CVPCA / ANTACT_TRAJ / DPCA_COUNT).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ed1_dimensionality.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig1.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import wilcoxon
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.0      # 10-in canvas -> 183 mm: an 8 pt label prints at 5.8 pt (CLAUDE.md print-scale rule)
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
STAGES = ['Naive', 'Expert']
SC = {'Naive': '0.55', 'Expert': '#332288'}
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MC = dict(zip(MICE, sns.color_palette('tab10', n_colors=len(MICE))))
NOCAP = '--nocap' in sys.argv[1:]

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
CV, FITDATA, PJ, PM = RES['CV'], RES['FITDATA'], RES['PR_JK'], RES['PM_CVPCA']
AT, DC = RES['ANTACT_TRAJ'], RES['DPCA_COUNT']


def plabel(ax, s, dx=-0.10):
    ax.text(dx, 1.06, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


# ══ a — the twelve-condition spectra (plus the DPA memory spectrum the ladder starts from) ══════
def panel_a(fig, gs):
    specs = [('memory (DPA delay)', lambda st: FITDATA[('DPA', 'delay', st)]['cv'], None, [1, 2, 3, 4]),
             ('delay (all 12 conditions)', lambda st: CV[(st, 'delay')]['cv'], CV[('Expert', 'delay')]['cvn'], [1, 6, 12]),
             ('decision (all 12 conditions)', lambda st: CV[(st, 'decision')]['cv'], CV[('Expert', 'decision')]['cvn'], [1, 6, 12])]
    axes = []
    for c, (ttl, get, cvn, xt) in enumerate(specs):
        ax = fig.add_subplot(gs[0, c]); axes.append(ax)
        for stage in STAGES:
            pos = np.clip(get(stage), 0, None); frac = pos / pos.sum()
            ax.plot(np.arange(1, len(frac) + 1), frac, '-o', ms=2.6, color=SC[stage], label=stage)
            print(f'a: {ttl:28s} {stage:6s} fractions {np.round(frac[:4], 3)}')
        if cvn is not None:
            real_tot = np.clip(get('Expert'), 0, None).sum()
            ax.plot(np.arange(1, len(cvn) + 1), np.clip(cvn, 0, None) / real_tot, '--', color='0.7', lw=1.0,
                    label='shuffle null')
        ax.axhline(0, color='0.85', lw=0.6)
        ax.set_xticks(xt); ax.set_ylim(-0.04, 1.06)
        ax.set_title(ttl, loc='left', fontsize=PS*7)
        ax.set_xlabel('cvPCA component')
        if c == 0:
            ax.set_ylabel('reliable variance (fraction)')
        else:
            ax.tick_params(labelleft=False)
        if c == 2:
            ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.2, loc='upper right')
    return axes[0]


# ══ b — the participation-ratio ladder, leave-one-mouse-out 95% CI (t(8)) ═════════════════════
def panel_b(ax):
    groups = [('DPA', 'delay', 'memory\n(DPA delay)'), ('all', 'delay', 'delay\n(all tasks)'),
              ('all', 'decision', 'decision\n(all tasks)')]
    xp = np.arange(len(groups))
    for j, stage in enumerate(STAGES):
        prs = [PJ[(ts, wn, stage)]['pr'] for ts, wn, _ in groups]
        cis = np.array([[PJ[(ts, wn, stage)]['pr'] - 2.306 * PJ[(ts, wn, stage)]['se'],
                         PJ[(ts, wn, stage)]['pr'] + 2.306 * PJ[(ts, wn, stage)]['se']] for ts, wn, _ in groups])
        xj = xp + (j - 0.5) * 0.32
        ax.bar(xj, prs, 0.30, color=SC[stage], label=stage)
        ax.vlines(xj, cis[:, 0], cis[:, 1], color='0.25', lw=0.9)
        for x, (lo, hi) in zip(xj, cis):
            ax.hlines([lo, hi], x - 0.05, x + 0.05, color='0.25', lw=0.9)
        for x, v, chi in zip(xj, prs, cis[:, 1]):
            ax.text(x, chi + 0.08, f'{v:.1f}', ha='center', va='bottom', fontsize=PS*6.5)
        for (ts, wn, _), pr, ci in zip(groups, prs, cis):
            print(f'b: {ts:4s} {wn:9s} {stage:6s} PR={pr:.2f} CI [{ci[0]:.2f}, {ci[1]:.2f}]')
    ax.set_xticks(xp); ax.set_xticklabels([g[2] for g in groups], fontsize=PS*7)
    ax.set_ylim(0, 4.2); ax.set_ylabel('participation ratio')
    ax.legend(frameon=False, fontsize=PS*6.5, loc='upper left')


# ══ c — the shattering dimension of the twelve conditions ════════════════════════════════════
def panel_c(ax):
    wins = [('delay', 'delay'), ('decision', 'decision')]
    xp = np.arange(len(wins))
    for j, stage in enumerate(STAGES):
        sds = [FITDATA[('all', wn, stage)]['sd'] for wn, _ in wins]
        xj = xp + (j - 0.5) * 0.32
        ax.bar(xj, sds, 0.30, color=SC[stage], label=stage)
        for x, v in zip(xj, sds):
            ax.text(x, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=PS*6.5)
            print(f'c: all {wins[list(xj).index(x)][0]:9s} {stage:6s} shattering {v:.3f}')
    ax.axhline(0.5, ls='--', color='0.6', lw=0.8)
    ax.text(1.42, 0.505, 'shuffle', fontsize=PS*6, color='0.5', ha='right', va='bottom')
    ax.axhline(1.0, ls=':', color='0.6', lw=0.8)
    ax.text(1.42, 0.995, 'unstructured', fontsize=PS*6, color='0.5', ha='right', va='top')
    ax.set_xticks(xp); ax.set_xticklabels([w[1] for w in wins], fontsize=PS*7)
    ax.set_xlim(-0.6, 1.45); ax.set_ylim(0.4, 1.04); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_ylabel('shattering dimension\n(balanced accuracy)')


# ══ d — per-mouse cvPCA: the memory spectrum is one-dimensional animal by animal ═══════════════
def panel_d(fig, gs):
    axes = []
    for k, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        for m in MICE:
            pts = []
            for x, win in [(0, 'md'), (1, 'decision')]:
                r = PM.get((m, stage, win))
                pts.append(None if r is None else (x, r['top1'], r['ok']))
            if pts[0] and pts[1] and pts[0][2] and pts[1][2]:
                ax.plot([0, 1], [pts[0][1], pts[1][1]], '-', color=MC[m], lw=0.8, alpha=0.5, zorder=2)
            for pt in pts:
                if pt is None:
                    continue
                x, yv, ok = pt
                ax.scatter(x, yv, s=30, color=MC[m] if ok else 'none', edgecolors=MC[m] if ok else '0.6',
                           linewidths=0.8, zorder=3)
        both = [m for m in MICE if all((m, stage, w) in PM and PM[(m, stage, w)]['ok'] for w in ('md', 'decision'))]
        a = np.array([PM[(m, stage, 'md')]['top1'] for m in both])
        b = np.array([PM[(m, stage, 'decision')]['top1'] for m in both])
        okmd = [PM[(m, stage, 'md')]['top1'] for m in MICE if (m, stage, 'md') in PM and PM[(m, stage, 'md')]['ok']]
        if len(both) >= 5:
            p = wilcoxon(a, b).pvalue; sig = p < .05
            ax.text(0.5, 1.02, '∗' if sig else 'n.s.', ha='center', va='bottom', fontsize=PS*(12 if sig else 8),
                    fontweight='bold', color='k' if sig else '0.55')
            ax.text(0.5, 0.955, f'p = {p:.3f}, n = {len(both)}', ha='center', va='bottom', fontsize=PS*6.5, color='0.3')
            print(f'd: {stage}: md top-1 median {np.median(okmd):.2f} (n={len(okmd)}); md vs decision '
                  f'{np.median(a):.2f} vs {np.median(b):.2f}, Wilcoxon p={p:.4f}, {int((a > b).sum())}/{len(both)}')
        ax.axhline(1 / 3, ls=':', color='0.6', lw=0.8)
        ax.text(1.38, 1 / 3 + 0.01, 'uniform', fontsize=PS*6.0, color='0.5', va='bottom', ha='right')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['memory\n(mid-delay)', 'decision'])
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1.16); ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(stage, loc='left', fontsize=TITLE_FS)
        if k == 0:
            ax.set_ylabel('top-1 reliable-variance\nfraction (per mouse)')
        else:
            ax.tick_params(labelleft=False)
            ax.legend(handles=[mlines.Line2D([0], [0], marker='o', color='0.4', ls='none', ms=4.5, label='resolvable'),
                               mlines.Line2D([0], [0], marker='o', mfc='none', color='0.6', ls='none', ms=4.5,
                                             label='noise-limited')],
                      frameon=False, fontsize=PS*6.0, loc='lower left', handletextpad=0.3)
    return axes[0]


# ══ e — the uncross-validated eta^2 matrices (DPA delay): the artifact Fig. 2d removes ════════
def panel_e(fig, gs):
    axes = []
    im = None
    for k, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        F = FITDATA[('DPA', 'delay', stage)]; FO = list(F['factors']); cmv = F['cm_var']
        M = F['pceta'][:3]
        im = ax.imshow(M, cmap='Purples', vmin=0, vmax=1, aspect='equal')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=PS*6.2,
                        color='w' if M[i, j] > 0.55 else 'k')
        ax.set_xticks(range(3)); ax.set_xticklabels(FO, fontsize=PS*6.5)
        ax.set_yticks(range(3)); ax.set_yticklabels([f'PC{i+1} ({cmv[i]:.0%})' for i in range(3)], fontsize=PS*6.2)
        ax.tick_params(length=0)
        ax.set_title(f'{stage}, DPA delay', loc='left', fontsize=TITLE_FS)
        for sp in ax.spines.values():
            sp.set_visible(True)
        print(f'e: {stage} DPA delay uncross-validated eta2 rows', np.round(M, 2).tolist())
    cb = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.04, shrink=0.8)
    cb.set_label('η² (not cross-validated)', fontsize=PS*6.5); cb.ax.tick_params(labelsize=PS*6)
    return axes[0]


# ══ f — learning removes the premature choice signal from the dual delay ══════════════════════
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}


def panel_f(fig, gs):
    tt = np.arange(84) / 6.0
    axes = []
    for p, sname in enumerate(['dual', 'DPA']):
        ax = fig.add_subplot(gs[0, p]); axes.append(ax)
        conds = SETS[sname]
        for stage in STAGES:
            m = [AT[(stage, sname, 'delay', c)] for c in conds if c[1] == c[2]]
            n = [AT[(stage, sname, 'delay', c)] for c in conds if c[1] != c[2]]
            sep = np.mean([d['mean'] for d in m], 0) - np.mean([d['mean'] for d in n], 0)
            var = (np.mean([d['sd'] ** 2 for d in m], 0) / len(m) + np.mean([d['sd'] ** 2 for d in n], 0) / len(n))
            ax.plot(tt, sep, color=SC[stage], lw=1.2, label=stage)
            ax.fill_between(tt, sep - np.sqrt(var), sep + np.sqrt(var), color=SC[stage], alpha=0.15, lw=0)
        ax.axhline(0, color='0.8', lw=0.6)
        for lo, hi, col in [(2.0, 3.0, '#332288'), (4.5, 5.5, '#cc3311'), (9.0, 10.0, '#377eb8')]:
            ax.axvspan(lo, hi, color=col, alpha=0.06, lw=0)
        ax.axvline(9.0, color='0.6', lw=0.6, ls=':')
        ax.set_title('dual trials' if sname == 'dual' else 'DPA trials', loc='left', fontsize=TITLE_FS)
        ax.set_xlabel('time (s)'); ax.set_xlim(0, 14); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12, 14])
        ax.set_ylim(-1.6, 4.2)
        if p == 0:
            ax.set_ylabel('future-choice separation\non the delay-defined axis (z)')
            ax.legend(frameon=False, fontsize=PS*6.5, loc='upper left')
        else:
            ax.tick_params(labelleft=False)
    WINS = [('ed', 'early\ndelay'), ('md', 'mid-\ndelay'), ('delay', 'late\ndelay'), ('decision', 'decision')]
    for p, sname in enumerate(['dual', 'DPA']):
        ax = fig.add_subplot(gs[0, 2 + p]); axes.append(ax)
        xp = np.arange(len(WINS))
        for j, stage in enumerate(STAGES):
            accs = [DC[(sname, wn, stage)]['choice']['acc'] for wn, _ in WINS]
            n95s = [DC[(sname, wn, stage)]['choice']['null95'] for wn, _ in WINS]
            xj = xp + (j - 0.5) * 0.34
            ax.bar(xj, accs, 0.30, color=SC[stage], zorder=2)
            ax.hlines(n95s, xj - 0.16, xj + 0.16, color='0.15', lw=0.8, zorder=3)
            for x, a, n9 in zip(xj, accs, n95s):
                if a > n9:
                    ax.text(x, a + 0.012, '∗', ha='center', va='bottom', fontsize=PS*8, fontweight='bold')
                print(f'f: {sname:4s} {stage:6s} choice {WINS[list(xp).index(round(x - (j - 0.5) * 0.34))][0]:8s} '
                      f'{a:.2f} (null95 {n9:.2f}){" *" if a > n9 else ""}')
        ax.axhline(0.5, color='0.6', lw=0.7, ls='--', zorder=1)
        ax.axvline(2.5, color='0.85', lw=0.7)
        ax.set_xticks(xp); ax.set_xticklabels([lb for _, lb in WINS], fontsize=PS*6.5)
        ax.set_ylim(0.38, 1.04); ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_title('dual trials' if sname == 'dual' else 'DPA trials', loc='left', fontsize=TITLE_FS)
        if p == 0:
            ax.set_ylabel('held-out choice decoding')
        else:
            ax.tick_params(labelleft=False)
    return axes[0], axes[2]


# ══ ASSEMBLE ═══════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 8.2))
outer = fig.add_gridspec(3, 12, height_ratios=[1.0, 1.0, 1.0], hspace=0.55, wspace=1.0,
                         left=0.065, right=0.985, top=0.965, bottom=0.07)
gsA = outer[0, 0:6].subgridspec(1, 3, wspace=0.22)
axA = panel_a(fig, gsA)
axB = fig.add_subplot(outer[0, 6:9]); panel_b(axB)
axC = fig.add_subplot(outer[0, 10:12]); panel_c(axC)
gsD = outer[1, 0:5].subgridspec(1, 2, wspace=0.12)
axD = panel_d(fig, gsD)
gsE = outer[1, 6:12].subgridspec(1, 2, wspace=0.55)
axE = panel_e(fig, gsE)
gsF = outer[2, 0:12].subgridspec(1, 4, wspace=0.28, width_ratios=[1.25, 1.25, 1, 1])
axF, axF2 = panel_f(fig, gsF)
plabel(axA, 'a', dx=-0.28); plabel(axB, 'b', dx=-0.30); plabel(axC, 'c', dx=-0.34)
plabel(axD, 'd', dx=-0.30); plabel(axE, 'e', dx=-0.42); plabel(axF, 'f', dx=-0.22)

CAP = [
    'Extended Data Fig. 1 | Dimensionality: provenance and robustness (companion to Fig. 2b–d). '
    'a, Cross-validated spectra of the DPA delay state (four conditions) and of the full twelve-condition '
    'state at the delay and at the decision (naïve and expert; dashed, the shuffle null of the expert fit). '
    'b, The participation ratio of the same three spectra, 95% CI from a leave-one-mouse-out jackknife '
    '(t(8)): memory 1.0 → delay 2.0 → decision 2.5, unchanged by learning. c, The shattering dimension: '
    'mean withheld-trial accuracy over all 462 balanced dichotomies of the twelve conditions, against the '
    'shuffle floor (0.50) and the unstructured ceiling (1); 0.64–0.68 at both stages.',
    'd, The memory spectrum is one-dimensional animal by animal: top-1 reliable-variance fraction of the '
    'DPA state from each mouse’s own simultaneously recorded population (same estimator as Fig. 2b), at '
    'mid-delay and at the decision; open symbols, noise-limited cells (reliable total < 5), excluded from '
    'the paired Wilcoxon test. e, The uncross-validated η² matrices of the DPA delay state, kept as the '
    'demonstration of the artifact Fig. 2d removes: on condition means estimated from all trials, the '
    'second and third components carry apparent test and choice coding that cannot be anticipatory, '
    'because the test odor is drawn independently of the sample.',
    'f, Learning removes the premature choice signal from the dual delay. Left, the withheld match/nonmatch '
    'separation projected on a choice axis defined at late delay (pre-test, hence reward-free), naïve '
    'against expert, on dual and on DPA trials (band, split SEM). Right, withheld decoding of the upcoming '
    'choice along the demixed choice axis per window (ticks, shuffle-null 95th percentile; ∗, above null): '
    'in naïve mice the dual delay carries the choice from early through late delay, in expert mice it '
    'sits at chance until the test; on DPA trials the pre-test signal is at most marginal (≤ 0.59) and absent at late delay.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig1.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/ed_fig1.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig1.png')
