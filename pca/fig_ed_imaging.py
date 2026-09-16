"""fig_ed_imaging.py — Extended Data Fig. 2: the imaged population and the session-by-session stability of the
held-out codes (companion to Fig. 2a and Fig. 3e). Built 2026-09-16 after the supplementary review ("how do you
know the axis change isn't day-to-day drift or a registration failure?").

  a  neurons per mouse entering the pseudo-population (the same registered neurons index both stages)
  b  held-out decodability of the sample (mid-delay, decoders trained at 33-38) and of the choice (decision,
     54-62) per mouse and session, from the canonical CCGD tensor: every session's trials are read from the
     fold that held them out, so a code that were an artefact of one session or of drifting registration would
     not hold across the six sessions. Naïve sessions grey, expert indigo; thin lines per mouse, bold mean.
Data: data/pca/mouse_slices.pkl; data/overlaps/{X,labels}_<canonical BDUM>.pkl (targets sample + choice).
Run:  cd /home/leon/dual/pca && python fig_ed_imaging.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig2.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns, matplotlib.pyplot as plt
from figcaption import draw_justified
from src.pca.io import pkl_load

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.2
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
NOCAP = '--nocap' in sys.argv[1:]
ED_N = 2
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MC = dict(zip(MICE, sns.color_palette('tab10', n_colors=len(MICE))))
GROUP = {**{m: 'Jaws' for m in MICE[:5]}, **{m: 'ChR2' for m in MICE[5:7]}, **{m: 'ACC' for m in MICE[7:]}}
SC = {'Naive': '0.55', 'Expert': '#332288'}
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
CODES = [('sample', np.arange(33, 39), 'sample code, mid-delay'), ('choice', np.arange(54, 63), 'choice code, decision')]

SL = pickle.load(open('../data/pca/mouse_slices.pkl', 'rb'))
NN = {m: SL[m].stop - SL[m].start for m in MICE}
print('a: neurons per mouse', NN, 'total', sum(NN.values()))

# ── held-out accuracy per mouse x session from the canonical tensor ──
y = pkl_load(f'labels_{BDUM}', path='../data/overlaps'); X = pkl_load(f'X_{BDUM}', path='../data/overlaps')
ACC = {}
for code, W, _ in CODES:
    sel = ((y.target == code) & (y.laser == 0)).to_numpy()
    dfs = np.asarray(X[sel][:, 1][:, W][:, :, W]).mean((1, 2)); lab = y[sel].reset_index(drop=True)
    lab['pred'] = (dfs > 0).astype(int); lab['lab'] = lab['labels'].astype(int)
    for (m, d), g in lab.groupby(['mouse', 'day']):
        if g['lab'].nunique() == 2:
            ACC[(code, m, int(d))] = (balanced_accuracy_score(g['lab'], g['pred']), g.stage.iloc[0], len(g))
del X


def plabel(ax, s, dx=-0.10, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


fig = plt.figure(figsize=(10.0, 3.6))
outer = fig.add_gridspec(1, 24, wspace=1.0, left=0.07, right=0.985, top=0.90, bottom=0.16)
# a
ax = fig.add_subplot(outer[0, 0:7]); axA = ax
for i, m in enumerate(MICE):
    ax.bar(i, NN[m], color=MC[m], width=0.7, edgecolor='k', linewidth=0.5)
    ax.text(i, NN[m] + 12, str(NN[m]), ha='center', va='bottom', fontsize=PS*6.0, color='0.3')
ax.set_xticks(range(len(MICE))); ax.set_xticklabels([f'{m}\n({GROUP[m]})' for m in MICE], rotation=60, ha='right', fontsize=PS*6.0)
ax.set_ylabel('neurons'); ax.set_ylim(0, max(NN.values()) * 1.15)
ax.set_title(f'imaged neurons per mouse ({sum(NN.values()):,} in all)', loc='left', fontsize=TITLE_FS)
# b
axes = []
for k, (code, W, ttl) in enumerate(CODES):
    ax = fig.add_subplot(outer[0, 9 + 8 * k:16 + 8 * k]); axes.append(ax)
    tab = pd.DataFrame([(m, d, a, st) for (c, m, d), (a, st, n) in ACC.items() if c == code], columns=['mouse', 'day', 'acc', 'stage'])
    pm = tab.pivot(index='mouse', columns='day', values='acc')
    for m in MICE:
        ax.plot(pm.columns, pm.loc[m], '-', color=MC[m], lw=0.8, alpha=0.7, zorder=2)
    for st in ['Naive', 'Expert']:
        days = sorted(tab[tab.stage == st].day.unique())
        mu = [pm[d].mean() for d in days]; se = [pm[d].std(ddof=1) / np.sqrt(pm[d].notna().sum()) for d in days]
        ax.errorbar(days, mu, se, fmt='o-', color=SC[st], ms=4, lw=1.8, capsize=2, zorder=4, label=st.lower())
        print(f'b: {code:6s} {st:6s} per-session held-out accuracy ' + ' '.join(f'd{d}:{v:.2f}' for d, v in zip(days, mu)) + f'   per-mouse s.d. across sessions {pm[days].std(1).mean():.3f}')
    ax.axhline(0.5, ls=':', color='0.6', lw=0.8); ax.set_ylim(0.4, 1.02); ax.set_xticks(sorted(tab.day.unique()))
    ax.set_xlabel('dual-task session'); ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
    if k == 0:
        ax.set_ylabel('held-out balanced accuracy'); ax.legend(frameon=False, loc='lower right')
    else:
        ax.tick_params(labelleft=False)
plabel(axA, 'a', dx=-0.20); plabel(axes[0], 'b', dx=-0.22)

CAP = [
    f'Extended Data Fig. {ED_N} | The imaged population and the session-by-session stability of its codes (companion '
    'to Fig. 2a and Fig. 3e). a, Neurons per mouse entering the pseudo-population (3,319 in all; the same registered '
    'neurons index the naïve and expert stages; opsin group in parentheses). b, Held-out decodability of the sample '
    '(mid-delay, on the mid-delay decoder) and of the choice (decision, on the decision decoder) per mouse and '
    'session, from the cross-validated decoders of Figs 3 and 4 (every trial read from the fold that held it out; '
    'thin lines, mice; bold, mean ± SEM; grey, naïve sessions; indigo, expert). Both codes are read at a similar '
    'level in every session, so the cross-stage transfer of Fig. 3e is not limited by session-to-session drift of '
    'the recorded population. Registration quality, fields of view and cell-count criteria: Methods.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
fig.savefig(f'{OUT}/png/ed_fig{ED_N}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig{ED_N}.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig{ED_N}.png')
