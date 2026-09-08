"""ED panel — the dual-Naive premature-choice (bias) signal and its removal by learning.

CLAIM: in naive mice the dual-task delay state carries a held-out-decodable premature choice signal
(the upcoming match/nonmatch lick decision is readable from the delay state, long before the test);
with learning this signal disappears — the delay is held clean of the decision. DPA = control (no
such signal at either stage).

Panels (all from caches, no recompute):
  a  dual: held-out match-nonmatch separation on the LD-defined choice axis across the trial,
     Naive vs Expert (ANTACT_TRAJ; the axis is defined PRE-test, bins 48-53, so it is reward-free)
  b  DPA: same — flat at both stages (no anticipatory direction exists; its axis fails validity)
  c  dual: held-out choice decodability per window (demixed axis vs shuffle-null 95th pct;
     DPCA_COUNT): Naive 0.64-0.66* at ED/MD/LD vs Expert at chance; both decode at the decision
  d  DPA: same — chance everywhere pre-test, decodable at the decision only

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_bias_cleanup_ed.py
Output: figures/pseudo/dimensionality/{png,svg}/fig_bias_cleanup_ed.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt

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
SC = {'Naive': '0.55', 'Expert': '#332288'}
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}
NBINS = 84

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
AT, DC = RES['ANTACT_TRAJ'], RES['DPCA_COUNT']

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4))
tt = np.arange(NBINS) / 6.0


def plabel(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')


# ── a, b: held-out m−nm separation on the LD-defined (pre-test, reward-free) choice axis ──
for p, sname in enumerate(['dual', 'DPA']):
    ax = axs[0, p]
    conds = SETS[sname]
    for stage in STAGES:
        m = [AT[(stage, sname, 'delay', c)] for c in conds if c[1] == c[2]]
        n = [AT[(stage, sname, 'delay', c)] for c in conds if c[1] != c[2]]
        sep = np.mean([d['mean'] for d in m], 0) - np.mean([d['mean'] for d in n], 0)
        var = (np.mean([d['sd'] ** 2 for d in m], 0) / len(m)
               + np.mean([d['sd'] ** 2 for d in n], 0) / len(n))
        ax.plot(tt, sep, color=SC[stage], lw=1.2, label=stage)
        ax.fill_between(tt, sep - np.sqrt(var), sep + np.sqrt(var), color=SC[stage], alpha=0.15, lw=0)
    ax.axhline(0, color='0.8', lw=0.6)
    for lo, hi, col in [(2.0, 3.0, '#332288'), (4.5, 5.5, '#cc3311'), (9.0, 10.0, '#377eb8')]:
        ax.axvspan(lo, hi, color=col, alpha=0.06, lw=0)
    ax.axvspan(8.0, 9.0, color='0.5', alpha=0.10, lw=0)
    ax.axvline(9.0, color='0.6', lw=0.6, ls=':')
    ax.text(8.9, 0.96, 'test onset', transform=ax.get_xaxis_transform(), ha='right', va='top',
            fontsize=5.8, color='0.45', rotation=90)
    ax.set_title('dual — premature choice signal' if sname == 'dual' else 'DPA — control (none)',
                 loc='left', fontsize=TITLE_FS)
    ax.set_xlabel('time (s)')
    if p == 0:
        ax.set_ylabel('future-choice separation\non the delay-defined axis (z)')
        ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    ax.set_ylim(-1.6, 4.2)

# ── c, d: held-out choice decodability per window (demixed axis vs shuffle null) ──
WINS = [('ed', 'ED'), ('md', 'MD'), ('delay', 'LD'), ('decision', 'decision')]
for p, sname in enumerate(['dual', 'DPA']):
    ax = axs[1, p]
    xp = np.arange(len(WINS))
    for j, stage in enumerate(STAGES):
        accs = [DC[(sname, wn, stage)]['choice']['acc'] for wn, _ in WINS]
        n95s = [DC[(sname, wn, stage)]['choice']['null95'] for wn, _ in WINS]
        xj = xp + (j - 0.5) * 0.34
        ax.bar(xj, accs, 0.30, color=SC[stage], label=stage, zorder=2)
        ax.hlines(n95s, xj - 0.16, xj + 0.16, color='0.15', lw=0.8, zorder=3)
        for x, a, n9 in zip(xj, accs, n95s):
            if a > n9:
                ax.text(x, a + 0.015, '*', ha='center', va='bottom', fontsize=9, fontweight='bold')
            print(f'{sname:4s} {stage:6s} choice: {a:.2f} (n95 {n9:.2f}){" *" if a > n9 else ""}')
    ax.axhline(0.5, color='0.6', lw=0.7, ls='--', zorder=1)
    ax.set_xticks(xp); ax.set_xticklabels([lb for _, lb in WINS], fontsize=7)
    ax.set_ylim(0.38, 1.02); ax.set_yticks([0.5, 0.75, 1.0])
    ax.axvline(2.5, color='0.85', lw=0.7)
    ax.set_title(f'{sname} — choice decodability by window', loc='left', fontsize=TITLE_FS)
    if p == 0:
        ax.set_ylabel('held-out decoding accuracy')

plabel(axs[0, 0], 'a'); plabel(axs[0, 1], 'b'); plabel(axs[1, 0], 'c'); plabel(axs[1, 1], 'd')
fig.suptitle('Learning removes the premature choice signal from the dual-task delay',
             x=0.085, ha='left', y=0.99, fontsize=11)
fig.text(0.085, 0.005,
         '(a,b) held-out match−nonmatch separation projected on the choice axis DEFINED AT LATE DELAY '
         '(bins 48–53, pre-test → reward-free); axis from one trial-half, traces from the other '
         '(both directions, 12 splits; band = split SEM).\n(c,d) balanced accuracy decoding held-out '
         'pseudo-trials along the choice demixed axis per window; black tick = shuffle-null 95th pct, '
         '* = above null; windows pre-test (ED/MD/LD) vs post-test (decision).',
         fontsize=5.8, color='0.35', va='bottom', ha='left')
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/fig_bias_cleanup_ed.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_bias_cleanup_ed.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_bias_cleanup_ed.png'))
