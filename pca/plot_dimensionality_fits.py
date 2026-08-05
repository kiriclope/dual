"""Three figures (DPA, dual, all=dpa+dual), each same layout:
 row 1: scree (cvPCA, 3 windows) · PR bars · shattering-SD bars   (Naive vs Expert)
 rows 2-3: PC-coding η² matrices (Naive, Expert) × (delay, decision, delay+dec).

Flag --gng : add a 'gng' column to the DPA matrices = Go-vs-NoGo cross-DECODING above-chance per DPA PC
(2·(bal-acc−0.5)) — gng doesn't exist within DPA, so it's the dual data projected onto the DPA axes.
Outputs get a _gng suffix, so the default (no gng) figures are kept."""
import os, sys, pickle
os.chdir('/home/leon/dual/pca')
import numpy as np, seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 400, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'none', 'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7})
TITLE_FS = 8
d = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
FITDATA = d['FITDATA']
WITH_GNG = '--gng' in sys.argv                                            # add gng cross-decode col to DPA
SUF = '_gng' if WITH_GNG else ''
DPA_GNG = d.get('DPA_GNG', {})
SC = {'Naive': '0.55', 'Expert': '#332288'}
WINS = ['delay', 'decision', 'delay+dec']
WCOL = {'delay': '#4477AA', 'decision': '#EE6677', 'delay+dec': '#228833'}
TITLE = {'DPA': 'DPA (4 conditions)', 'dual': 'dual = DualGo + DualNoGo (8 conditions)',
         'all': 'dpa + dual (all 12 conditions)'}


NPC_SHOW = {'DPA': 3, 'dual': 4, 'all': 6}                                 # #PCs per fit (all: 6 so test shows)


def pc_heatmap(ax, ts, wn, stage, title):
    F = FITDATA[(ts, wn, stage)]; FO = list(F['factors']); cmv = F['cm_var']
    cap = 4 if (WITH_GNG and ts == 'DPA') else NPC_SHOW.get(ts, 4)         # DPA+gng: 4 PCs → 4×4 square
    npc = min(cap, F['pceta'].shape[0]); M = F['pceta'][:npc]
    if WITH_GNG and ts == 'DPA':                                          # insert gng cross-decode col after sample
        g = np.asarray(DPA_GNG[(wn, stage)])[:npc][:, None]
        M = np.column_stack([M[:, [0]], g, M[:, 1:]]); FO = ['sample', 'gng', 'test', 'choice']
    im = ax.imshow(M, cmap='Purples', vmin=0, vmax=1, aspect='equal')
    for i in range(npc):
        for j in range(len(FO)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.2,
                    color='w' if M[i, j] > 0.55 else 'k')
    ax.set_xticks(range(len(FO))); ax.set_xticklabels(FO, fontsize=6.4)
    ax.set_yticks(range(npc)); ax.set_yticklabels([f'PC{k+1} ({cmv[k]:.0%})' for k in range(npc)], fontsize=6.0)
    if title:
        ax.set_title(title, loc='center', fontsize=TITLE_FS, fontweight='bold')
    for sp in ax.spines.values():
        sp.set_visible(True)
    return im


for ts in ['DPA', 'dual', 'all']:
    fig = plt.figure(figsize=(10.5, 10.0 + 0.75 * (NPC_SHOW.get(ts, 4) - 4)))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.55, wspace=0.5,
                          left=0.09, right=0.94, top=0.9, bottom=0.06)

    ax = fig.add_subplot(gs[0, 0])                                          # scree (3 windows, Expert solid / Naive dashed)
    for wn in WINS:
        for stage, ls, a in [('Expert', '-', 1.0), ('Naive', '--', 0.7)]:
            var = FITDATA[(ts, wn, stage)]['var']
            ax.plot(np.arange(1, len(var) + 1), var, ls, color=WCOL[wn], lw=1.3, alpha=a, ms=3,
                    marker='o' if stage == 'Expert' else None,
                    label=wn if stage == 'Expert' else None)
    ax.set_xlabel('cvPCA component'); ax.set_ylabel('reliable variance (frac)')
    ax.set_title('cvPCA scree', loc='left', fontsize=TITLE_FS)
    ax.legend(frameon=False, fontsize=6, loc='upper right', title='(solid E · dashed N)', title_fontsize=5.5)

    axp = fig.add_subplot(gs[0, 1])                                         # PR bars
    axs = fig.add_subplot(gs[0, 2])                                         # SD bars
    xw = np.arange(len(WINS))
    for j, stage in enumerate(['Naive', 'Expert']):
        prs = [FITDATA[(ts, wn, stage)]['pr'] for wn in WINS]
        sds = [FITDATA[(ts, wn, stage)]['sd'] for wn in WINS]
        axp.bar(xw + (j - 0.5) * 0.34, prs, 0.32, color=SC[stage], label=stage)
        axs.bar(xw + (j - 0.5) * 0.34, sds, 0.32, color=SC[stage], label=stage)
        for x, v in zip(xw + (j - 0.5) * 0.34, prs):
            axp.text(x, v + 0.03, f'{v:.1f}', ha='center', va='bottom', fontsize=6)
        for x, v in zip(xw + (j - 0.5) * 0.34, sds):
            axs.text(x, v + 0.005, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    for a, ttl, yl in [(axp, 'participation ratio (cross-val.)', 'PR'), (axs, 'shattering SD (bal. acc.)', 'SD')]:
        a.set_xticks(xw); a.set_xticklabels(WINS, fontsize=6.6); a.set_title(ttl, loc='left', fontsize=TITLE_FS)
        a.set_ylabel(yl); a.legend(frameon=False, fontsize=6.5, loc='upper left')
    axs.axhline(0.5, ls=':', color='0.6', lw=0.8); axs.set_ylim(0.45, 1.0)
    axp.set_ylim(0, max(3.2, axp.get_ylim()[1]))

    im = None
    for r, stage in ((1, 'Naive'), (2, 'Expert')):
        for c, wn in enumerate(WINS):
            ax = fig.add_subplot(gs[r, c])
            im = pc_heatmap(ax, ts, wn, stage, wn if r == 1 else '')
        p = fig.axes[-3].get_position()                                     # leftmost matrix of this row
        fig.text(0.02, (p.y0 + p.y1) / 2, stage, rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
    cb = fig.colorbar(im, ax=[fig.axes[-1]], fraction=0.05, pad=0.04); cb.set_label('η²', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    ttl = TITLE[ts] + ('   (+ gng cross-decode column)' if (WITH_GNG and ts == 'DPA') else '')
    fig.suptitle(f'Dimensionality & PC coding — {ttl}', x=0.01, ha='left', y=0.955, fontsize=11)
    OUT = 'figures/pseudo/dimensionality'
    fig.savefig(f'{OUT}/png/dim_{ts}{SUF}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/dim_{ts}{SUF}.svg', bbox_inches='tight')
    plt.close(fig); print('saved', f'{OUT}/png/dim_{ts}{SUF}.png')
