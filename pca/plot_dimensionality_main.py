"""Re-render the dimensionality figure from cached results (no recompute).
Row 1: cvPCA spectrum · PR · shattering.  Rows 2-3: PC-coding η² matrices (Naive, Expert) — dual (4 codes) & DPA (3 codes)."""
import os, pickle
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
CV, SD, DICH, STAGES = d['CV'], d['SD'], d['DICH'], d['STAGES']
PCETA8, PCETA_DPA = d['PCETA8'], d['PCETA_DPA']
SC = {'Naive': '0.55', 'Expert': '#332288'}

fig = plt.figure(figsize=(15.0, 8.6))
gs = fig.add_gridspec(3, 12, height_ratios=[1.15, 1.0, 1.0], hspace=0.5, wspace=1.15,
                      left=0.07, right=0.965, top=0.93, bottom=0.05)

ax = fig.add_subplot(gs[0, 0:4])                                           # cvPCA spectrum
allcv = np.concatenate([CV[(s, 'delay')]['cv'] for s in STAGES]); floor = max(allcv.max() * 1e-4, 1e-9)
for stage in STAGES:
    cv = CV[(stage, 'delay')]['cv']
    ax.plot(np.arange(1, len(cv) + 1), np.clip(cv, floor, None), '-o', ms=3, color=SC[stage], label=stage)
cvn = CV[('Expert', 'delay')]['cvn']
ax.plot(np.arange(1, len(cvn) + 1), np.clip(cvn, floor, None), '--', color='0.7', lw=1.0, label='null (shuffled)')
ax.set_yscale('log'); ax.set_xlabel('cvPCA component'); ax.set_ylabel('reliable variance (cross-validated)')
ax.set_title('cvPCA — delay-state geometry (12 conds)', loc='left', fontsize=TITLE_FS)
ax.legend(frameon=False, fontsize=6.5)

ax = fig.add_subplot(gs[0, 4:8])                                          # PR bars
labs = ['delay', 'decision']; xp = np.arange(len(labs))
for j, stage in enumerate(STAGES):
    prs = [CV[(stage, k)]['pr'] for k in labs]
    ax.bar(xp + (j - 0.5) * 0.32, prs, 0.30, color=SC[stage], label=stage)
    for x, v in zip(xp + (j - 0.5) * 0.32, prs):
        ax.text(x, v + 0.05, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(xp); ax.set_xticklabels(['delay\n(WM state)', 'decision\n(all 12 conds)']); ax.set_ylim(0, 4)
ax.set_ylabel('participation ratio'); ax.set_title('reliable dimensionality (PR)', loc='left', fontsize=TITLE_FS)
ax.legend(frameon=False, fontsize=6.5, loc='upper left')

ax = fig.add_subplot(gs[0, 8:12])                                         # shattering
for j, stage in enumerate(STAGES):
    a = SD[stage]['acc']
    parts = ax.violinplot(a, positions=[j], widths=0.7, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(SC[stage]); pc.set_alpha(0.5)
    parts['cmeans'].set_color('k')
    ax.text(j, a.mean() + 0.01, f'{a.mean():.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.axhline(0.5, ls=':', color='0.6', lw=0.8); ax.text(1.55, 0.505, 'chance', fontsize=5.5, color='0.6', ha='right')
ax.axhline(SD['Expert']['null'].mean(), ls='--', color='0.7', lw=0.9)
ax.set_xticks([0, 1]); ax.set_xticklabels(STAGES); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.45, 1.02)
ax.set_ylabel('decoding bal. acc. / dichotomy')
ax.set_title(f'shattering dimension ({len(DICH)} dichotomies)', loc='left', fontsize=TITLE_FS)


def pc_heatmap(ax, PC, stage, wn, title, cbar=False):                     # one η²-by-factor matrix (square cells)
    P = PC[(stage, wn)]; FO = P['factors']; M = P['eta']; nk = len(P['var'])
    im = ax.imshow(M, cmap='Purples', vmin=0, vmax=1, aspect='equal')
    for i in range(nk):
        for j in range(len(FO)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.4,
                    color='w' if M[i, j] > 0.55 else 'k')
    ax.set_xticks(range(len(FO))); ax.set_xticklabels(FO, fontsize=6.8)
    ax.set_yticks(range(nk)); ax.set_yticklabels([f'PC{k+1} ({P["var"][k]:.0%})' for k in range(nk)], fontsize=6.2)
    if title:
        ax.set_title(title, loc='left', fontsize=TITLE_FS)
    for sp in ax.spines.values():
        sp.set_visible(True)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.06); cb.set_label('η²', fontsize=6.5); cb.ax.tick_params(labelsize=6)


SPECS = [(PCETA8, 'delay', 'dual tasks — delay'), (PCETA8, 'decision', 'dual tasks — decision'),
         (PCETA_DPA, 'delay', 'DPA only — delay'), (PCETA_DPA, 'decision', 'DPA only — decision')]
COLS = [(0, 3), (3, 6), (6, 9), (9, 12)]
for r, stage in ((1, 'Naive'), (2, 'Expert')):
    first = None
    for (PC, wn, ttl), (c0, c1) in zip(SPECS, COLS):
        ax = fig.add_subplot(gs[r, c0:c1])
        first = first or ax
        pc_heatmap(ax, PC, stage, wn, ttl if r == 1 else '', cbar=(c1 == 12))
    p = first.get_position()
    fig.text(0.018, (p.y0 + p.y1) / 2, f'{stage}\n(PC coding, η²)', rotation=90, va='center', ha='center',
             fontsize=9, fontweight='bold')

fig.suptitle('Honest dimensionality of the dual-task pseudo-population — cvPCA + shattering + PC coding (dual vs DPA)',
             x=0.008, ha='left', y=0.965, fontsize=10)
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/dimensionality.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/dimensionality.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/dimensionality.png'))
