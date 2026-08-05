"""Per-fit scree grid: rows = task-set (dual, DPA), cols = window (delay, decision, delay+dec). Each panel =
condition-mean scree (Naive vs Expert), with participation ratio (PR) and shattering SD annotated."""
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
FITDATA = d['FITDATA']
SC = {'Naive': '0.55', 'Expert': '#332288'}
WINS = ['delay', 'decision', 'delay+dec']; TS = ['dual', 'DPA']

fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), gridspec_kw=dict(wspace=0.28, hspace=0.42))
for r, ts in enumerate(TS):
    for c, wn in enumerate(WINS):
        ax = axes[r, c]; nc = FITDATA[(ts, wn, 'Expert')]['nconds']
        for stage in ['Naive', 'Expert']:
            F = FITDATA[(ts, wn, stage)]; var = F['var']
            ax.plot(np.arange(1, len(var) + 1), var, '-o', ms=3.5, color=SC[stage],
                    label=f'{stage}:  PR {F["pr"]:.1f} · SD {F["sd"]:.2f}')
        ax.set_title(f'{ts} — {wn}   ({nc} conds)', loc='left', fontsize=TITLE_FS)
        ax.set_xticks(range(1, nc))
        ax.set_xlabel('cvPCA component' if r == 1 else ''); ax.set_ylabel('reliable variance (frac)' if c == 0 else '')
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1]))
        ax.legend(frameon=False, fontsize=6.2, loc='upper right', handlelength=1.3, title='PR = cross-validated',
                  title_fontsize=5.5)
fig.suptitle('Per-fit dimensionality — cvPCA reliable-variance scree, participation ratio (PR, cross-validated) '
             '& shattering SD (bal.acc.),  Naive vs Expert', x=0.008, ha='left', y=0.98, fontsize=9.5)
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/dimensionality_fits.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/dimensionality_fits.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/dimensionality_fits.png'))
