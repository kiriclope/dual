"""(RENUMBERED 2026-09-15 to citation order: this script draws Extended Data Fig. 2.)
fig_ed1_dimensionality.py — Extended Data Fig. 2: dimensionality, provenance and robustness
(companion to Fig. 2b–d). Built 2026-09-15 (Leon: "keep only what is essential for the paper's
argumentation") — every panel here is one the Results, Methods or Discussion cite; nothing else.

  a  the full twelve-condition spectra (Results §2: "the full twelve-condition spectra are given in ED")
  b  the participation-ratio ladder with leave-one-mouse-out CIs (Methods: "on the PR (ED)")
  c  the shattering dimension of the twelve conditions (Results §3: "0.67 at both stages")
  d  the per-mouse cvPCA companion (Discussion: "including the dimensionality spectra themselves, n = 9")
  e  the naive 'premature choice' signal is a correct-trial selection effect (why Fig. 2c reads all trials)
     (2026-09-15: the eta^2 demonstration and the bias-cleanup traces were cut — Leon: "cut e"; the naive signal
     did not survive all trials, exp_dpca_count.py --alltrials)

Reads caches only: results.pkl (CV / FITDATA / PR_JK / PM_CVPCA / ANTACT_TRAJ / DPCA_COUNT).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ed1_dimensionality.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig2.{png,svg}
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
PS = 1.2      # 10-in canvas -> 183 mm is x0.72: 1.2 keeps every literal (5.5-8 pt) at >= 5 pt in print (review 2026-09-15)
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
FITDATA, PM = RES['FITDATA'], RES['PM_CVPCA']


def plabel(ax, s, dx=-0.10):
    ax.text(dx, 1.06, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


# ══ a — the DPA and twelve-condition spectra at Fig. 2b's windows, Fig. 2b's estimator (ED1_SPEC, exp_ed1_spectra.py) ══
ES = RES['ED1_SPEC']
A_SPECS = [('DPA', 'md', 'memory (DPA, mid-delay)', [1, 2, 3, 4]), ('all', 'md', 'mid-delay, 12 conditions', [1, 6, 12]),
           ('all', 'decision', 'decision, 12 conditions', [1, 6, 12])]


def panel_a(fig, gs):
    axes = []
    for c, (ts, wn, ttl, xt) in enumerate(A_SPECS):
        ax = fig.add_subplot(gs[0, c]); axes.append(ax)
        for stage in STAGES:
            sp = np.asarray(ES[(ts, wn, stage)]['spec']); pos = np.clip(sp, 0, None); frac = pos / pos.sum()
            ax.plot(np.arange(1, len(frac) + 1), frac, '-o', ms=2.6, color=SC[stage], label=stage.lower().replace('naive', 'naïve'))
            print(f'a: {ttl:30s} {stage:6s} fractions {np.round(frac[:4], 3)}')
        nul = ES[(ts, wn, 'Expert')]['null']
        if nul is not None:
            real_tot = np.clip(np.asarray(ES[(ts, wn, 'Expert')]['spec']), 0, None).sum()
            ax.plot(np.arange(1, len(nul) + 1), np.clip(np.asarray(nul), 0, None) / real_tot, '--', color='0.7', lw=1.0, label='shuffle null')
        ax.axhline(0, color='0.85', lw=0.6)
        ax.set_xticks(xt); ax.set_ylim(-0.04, 1.06)
        ax.set_title(ttl, loc='left', fontsize=PS*7)
        ax.set_xlabel('cvPCA component')
        if c == 0:
            ax.set_ylabel('reliable variance (fraction)')
        else:
            ax.tick_params(labelleft=False)
        if c == 2:
            ax.legend(frameon=False, fontsize=PS*6.0, handlelength=1.2, loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0)
    return axes[0]


# ══ b — the participation ratio of the same three spectra, leave-one-mouse-out 95% CI (t(8)) ══════
def panel_b(ax):
    groups = [('DPA', 'md', 'memory\n(DPA)'), ('all', 'md', 'mid-delay\n(12 cond.)'), ('all', 'decision', 'decision\n(12 cond.)')]
    xp = np.arange(len(groups))
    for j, stage in enumerate(STAGES):
        prs = [ES[(ts, wn, stage)]['pr'] for ts, wn, _ in groups]
        cis = np.array([[max(1.0, ES[(ts, wn, stage)]['pr'] - 2.306 * ES[(ts, wn, stage)]['se']),     # PR >= 1 by definition
                         ES[(ts, wn, stage)]['pr'] + 2.306 * ES[(ts, wn, stage)]['se']] for ts, wn, _ in groups])
        xj = xp + (j - 0.5) * 0.32
        ax.bar(xj, prs, 0.30, color=SC[stage], label=stage.lower().replace('naive', 'naïve'))
        ax.vlines(xj, cis[:, 0], cis[:, 1], color='0.25', lw=0.9)
        for x, (lo, hi) in zip(xj, cis):
            ax.hlines([lo, hi], x - 0.05, x + 0.05, color='0.25', lw=0.9)
        for x, v, chi in zip(xj, prs, cis[:, 1]):
            ax.text(x, chi + 0.08, f'{v:.1f}', ha='center', va='bottom', fontsize=PS*6.5)
        for (ts, wn, _), pr, ci in zip(groups, prs, cis):
            print(f'b: {ts:4s} {wn:9s} {stage:6s} PR={pr:.2f} CI [{ci[0]:.2f}, {ci[1]:.2f}]')
    ax.set_xticks(xp); ax.set_xticklabels([g[2] for g in groups], fontsize=PS*7)
    ax.set_ylim(0, 5.3); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_ylabel('participation ratio')
    ax.legend(frameon=False, fontsize=PS*6.5, loc='upper left')


# ══ c — the shattering dimension of the twelve conditions (SD_FULL: all 462 balanced dichotomies at the
#     decision window 54–62, exp_dimensionality_ci.py — the numbers Results §3 quotes) ═════════════════
def panel_c(fig, gs):
    """Left: pseudo-population SD (SD_FULL dots/mean; bar = across-animal leave-one-mouse-out jackknife 95% CI,
    SD_LOO). Right: each mouse's own population (SD_MOUSE), paired naive -> expert, Wilcoxon n = 9."""
    SDF, LOO, SDM = RES['SD_FULL'], RES['SD_LOO'], RES['SD_MOUSE']
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    rng = np.random.RandomState(3)
    for j, stage in enumerate(STAGES):
        acc = np.asarray(SDF[stage]['acc']); nul = np.asarray(SDF[stage]['null']); ci = LOO[stage]['ci']
        x = j + rng.uniform(-0.22, 0.22, len(acc))
        ax.scatter(x, acc, s=5, color=SC[stage], alpha=0.35, lw=0, zorder=2)
        ax.plot([j - 0.3, j + 0.3], [acc.mean()] * 2, color='k', lw=1.3, zorder=4)
        ax.vlines(j, ci[0], ci[1], color='k', lw=0.9, zorder=4)
        ax.plot([j - 0.3, j + 0.3], [nul.mean()] * 2, color='0.45', lw=0.9, ls='--', zorder=3)
        ax.text(j, 1.005, f'{acc.mean():.3f}', ha='center', va='bottom', fontsize=PS*6.5)
        print(f'c: {stage:6s} shattering {acc.mean():.3f} [LOO jackknife CI {ci[0]:.3f}, {ci[1]:.3f}; resample CI '
              f'{SDF[stage]["ci"][0]:.3f}, {SDF[stage]["ci"][1]:.3f}]  null {nul.mean():.3f}  '
              f'({len(acc)} dichotomies, range {acc.min():.2f}–{acc.max():.2f})')
    D = LOO['delta']
    print(f'c: Expert−Naive Δ {D["mean"]:+.3f} [{D["ci"][0]:+.3f}, {D["ci"][1]:+.3f}] t(8) {D["t"]:+.2f} p {D["p"]:.3f}')
    ax.axhline(1.0, ls=':', color='0.6', lw=0.8)
    ax.text(1.45, 0.985, 'unstructured', fontsize=PS*6, color='0.5', ha='right', va='top')
    ax.text(1.45, 0.505, 'shuffle', fontsize=PS*6, color='0.5', ha='right', va='bottom')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['naïve', 'expert'], fontsize=PS*7)
    ax.set_xlim(-0.6, 1.5); ax.set_ylim(0.44, 1.06); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_ylabel('balanced accuracy\n(462 dichotomies, decision)')
    ax.set_title('pooled', loc='left', fontsize=TITLE_FS)
    # right: own-population shattering, one line per mouse
    a, b = SDM['Naive'], SDM['Expert']
    for k, m in enumerate(MICE):
        ax2.plot([0, 1], [a[k], b[k]], '-', color=MC[m], lw=0.8, alpha=0.5, zorder=2)
        ax2.scatter([0, 1], [a[k], b[k]], s=22, color=MC[m], linewidths=0.6, zorder=3)
    pw = wilcoxon(b, a).pvalue; sig = pw < .05
    ax2.text(0.5, 1.02, '∗' if sig else 'n.s.', ha='center', va='bottom', fontsize=PS*(12 if sig else 8),
             fontweight='bold', color='k' if sig else '0.55')
    ax2.text(0.5, 0.93, f'p = {pw:.3f}\n9 mice', ha='center', va='bottom', fontsize=PS*6.5, color='0.3')
    ax2.axhline(0.5, ls='--', color='0.45', lw=0.9, zorder=1)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['naïve', 'expert'], fontsize=PS*7); ax2.set_xlim(-0.5, 1.5)
    plt.setp(ax2.get_yticklabels(), visible=False); ax2.tick_params(axis='y', length=0)
    ax2.set_title('per mouse', loc='left', fontsize=TITLE_FS)
    print(f'c: per-mouse own-population SD medians {np.median(a):.3f} / {np.median(b):.3f}, Δ median '
          f'{np.median(b - a):+.3f}, Wilcoxon p {pw:.3f}, {int((b > a).sum())}/9 up')
    return ax


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
            ax.text(0.5, 1.12, '∗' if sig else 'n.s.', ha='center', va='bottom', fontsize=PS*(12 if sig else 8),
                    fontweight='bold', color='k' if sig else '0.55')
            ax.text(0.5, 1.05, f'p = {p:.3f}, n = {len(both)}', ha='center', va='bottom', fontsize=PS*6.5, color='0.3')
            print(f'd: {stage}: md top-1 median {np.median(okmd):.2f} (n={len(okmd)}); md vs decision '
                  f'{np.median(a):.2f} vs {np.median(b):.2f}, Wilcoxon p={p:.4f}, {int((a > b).sum())}/{len(both)}')
        ax.axhline(1 / 3, ls=':', color='0.6', lw=0.8)
        ax.text(1.38, 1 / 3 + 0.01, 'uniform', fontsize=PS*6.0, color='0.5', va='bottom', ha='right')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['memory\n(mid-delay)', 'decision'])
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1.28); ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(stage.lower().replace('naive', 'naïve'), loc='left', fontsize=TITLE_FS)
        if k == 0:
            ax.set_ylabel('top-1 reliable-variance\nfraction (per mouse)')
        else:
            ax.tick_params(labelleft=False)
            ax.legend(handles=[mlines.Line2D([0], [0], marker='o', color='0.4', ls='none', ms=4.5, label='resolvable'),
                               mlines.Line2D([0], [0], marker='o', mfc='none', color='0.6', ls='none', ms=4.5,
                                             label='noise-limited')],
                      frameon=False, fontsize=PS*6.0, loc='lower left', handletextpad=0.3)
    return axes[0]


# ══ e — the naive "premature choice" signal is a correct-trial selection effect (DPCA_COUNT vs DPCA_COUNT_all) ══
WINS_E = [('ed', 'early'), ('md', 'mid'), ('delay', 'late'), ('decision', 'test')]


def panel_e(fig, gs):
    axes = []
    for p, sname in enumerate(['dual', 'DPA']):
        for k, (key, lab) in enumerate([('DPCA_COUNT', 'correct trials'), ('DPCA_COUNT_all', 'all trials')]):
            ax = fig.add_subplot(gs[0, 2 * p + k]); axes.append(ax)
            DC = RES[key]; xp = np.arange(len(WINS_E))
            for j, stage in enumerate(STAGES):
                accs = [DC[(sname, wn, stage)]['choice']['acc'] for wn, _ in WINS_E]
                n95s = [DC[(sname, wn, stage)]['choice']['null95'] for wn, _ in WINS_E]
                xj = xp + (j - 0.5) * 0.34
                ax.bar(xj, accs, 0.30, color=SC[stage], zorder=2, label=stage.lower().replace('naive', 'naïve'))
                ax.hlines(n95s, xj - 0.16, xj + 0.16, color='0.15', lw=0.8, zorder=3)
                for x, a, n9 in zip(xj, accs, n95s):
                    if a > n9:
                        ax.text(x, a + 0.012, '∗', ha='center', va='bottom', fontsize=PS*8, fontweight='bold')
                print(f'e: {sname:4s} {lab:14s} {stage:6s} choice ' + ' '.join(f'{wn}={a:.2f}{"*" if a > n9 else ""}' for (wn, _), a, n9 in zip(WINS_E, accs, n95s)))
            ax.axhline(0.5, color='0.6', lw=0.7, ls='--', zorder=1); ax.axvline(2.5, color='0.85', lw=0.7)
            ax.set_xticks(xp); ax.set_xticklabels([lb for _, lb in WINS_E], fontsize=PS*6.2)
            if 2 * p + k == 0:
                ax.set_xlabel('delay window · decision', fontsize=PS*6.5, loc='left')
            ax.set_ylim(0.38, 1.04); ax.set_yticks([0.5, 0.75, 1.0])
            ax.set_title(f'{sname}, {lab.split()[0]}', loc='left', fontsize=TITLE_FS)
            if 2 * p + k == 0:
                ax.set_ylabel('held-out choice decoding'); ax.legend(frameon=False, fontsize=PS*6.5, loc='upper left')
            else:
                ax.tick_params(labelleft=False)
    return axes[0]


# ══ ASSEMBLE ═══════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 6.2))
outer = fig.add_gridspec(2, 24, height_ratios=[1.0, 1.0], hspace=0.55, wspace=1.0,
                         left=0.065, right=0.985, top=0.955, bottom=0.09)
gsA = outer[0, 0:12].subgridspec(1, 3, wspace=0.30)
axA = panel_a(fig, gsA)
axB = fig.add_subplot(outer[0, 12:18]); panel_b(axB)
gsC = outer[0, 19:24].subgridspec(1, 2, wspace=0.15, width_ratios=[1.15, 1.0])
axC = panel_c(fig, gsC)
gsD = outer[1, 0:8].subgridspec(1, 2, wspace=0.12)
axD = panel_d(fig, gsD)
gsE = outer[1, 10:24].subgridspec(1, 4, wspace=0.22)
axE = panel_e(fig, gsE)
plabel(axA, 'a', dx=-0.28); plabel(axB, 'b', dx=-0.30); plabel(axC, 'c', dx=-0.42)
plabel(axD, 'd', dx=-0.30); plabel(axE, 'e', dx=-0.34)

CAP = [
    'Extended Data Fig. 2 | Dimensionality: provenance and robustness (companion to Fig. 2b–d). '
    'a, Cross-validated spectra of the DPA state (four conditions) and of the full twelve-condition state, at '
    'mid-delay (5.5–6.5 s) and at the decision (9.0–10.5 s), the windows and estimator of Fig. 2b (30 half-splits; '
    'naïve and expert; dashed, the shuffle null of the expert fit). b, The participation ratio of the same three '
    'spectra, 95% CI from a leave-one-mouse-out jackknife (t(8); floored at 1, the minimum of a participation ratio): '
    'the memory state is one-dimensional, the twelve-condition state two- to three-dimensional at both windows, with '
    'overlapping intervals across stages. c, The shattering dimension: withheld-trial balanced accuracy of every one '
    'of the 462 balanced dichotomies of the twelve conditions at the decision window (dots), its mean (line; bar, 95% '
    'across-animal interval from a leave-one-mouse-out jackknife, t(8): 0.660 [0.615, 0.705] naïve, 0.672 [0.621, 0.723] '
    'expert; difference +0.012 [−0.022, +0.046], p = .43) against the shuffle mean (dashed) and the unstructured ceiling '
    '(1); right, the same estimator on each mouse’s own simultaneously recorded population (real trials, 20 half-splits; '
    'lower because populations and trial counts are smaller): medians 0.585 naïve, 0.602 expert, Wilcoxon p = .055, 7/9 '
    'mice up.',
    'd, The memory spectrum is one-dimensional animal by animal: top-1 reliable-variance fraction of the DPA state '
    'from each mouse’s own simultaneously recorded population (same estimator and windows as Fig. 2b), at mid-delay '
    'and at the decision; open symbols, noise-limited cells (reliable total < 5), excluded from the paired Wilcoxon '
    'test. e, Why Fig. 2c decodes all trials. Withheld decoding of the match/nonmatch (choice) contrast along the '
    'demixed choice axis per window (ticks, shuffle-null 95th percentile; ∗, above null), on correct trials only '
    '(left of each pair) and on all laser-off trials (right). On correct trials the future choice coincides with the '
    'lick, and the naïve dual-trial delay appears to carry it (0.64–0.66 from early through late delay); on all '
    'trials, where the contrast is fixed by the odors alone, one of the twelve pre-test cells remains above its null '
    '(naïve, Go and NoGo trials, mid-delay: 0.56 against 0.50 ± 0.02), the nominal false-positive rate for a contrast '
    'that cannot be known before the test — a selection effect of a lick-prone naïve state, not premature '
    'deliberation. Cells below chance on all trials (down to 0.41) arise because the two halves of a small pool of '
    'error trials have complementary lick composition, which anti-correlates the halves’ choice contrasts. The delay '
    'carries no trial-by-trial choice information at either stage; the decision does in expert mice at both trial '
    'types and in naïve mice on Go and NoGo trials (DPA 0.54, marginal).',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig2.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/ed_fig2.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig2.png')
