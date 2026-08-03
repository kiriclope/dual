"""
Mixing between task components for a pseudo-population PCA run.

Two panels:
  left  — coding strength |contrast score| per variable (Sample/Choice/Test)
          per PC; shows which PC carries each variable and the leakage onto others
  right — variable x variable mixing = |cos| between coding vectors over PCs
          (1 = identical direction, 0 = orthogonal/demixed)

The scalar mixing index (mean off-diagonal |cos|) and per-variable participation
ratio are printed and shown in the title.

Loads pseudo_{traj,labels}_<dum>.pkl from ../data/pca/.
Figures -> figures/pseudo/mixing/<scale>/<factor>/{png,svg}/<dum>_mixing.{png,svg}
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys, os
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from src.common.plot_utils import add_vlines
from src.pca.io import pkl_load
from src.pca.identify import (variable_mixing, participation_ratio, mixing_index,
                              variable_mixing_time, identify_pcs, pc_label)

# ── Style (Nature Neuroscience house style — matches the main figures) ──────────
sns.set_context('notebook')          # MUST come after importing src.common.plot_utils (sets "poster")
sns.set_style('ticks')
plt.rcParams.update({                 # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

# ── config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Plot task-component mixing for a pseudo-population PCA run.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dum', default='pseudo_DELAY_Expert_zscore_5x1')
parser.add_argument('--data-dir', default='../data/pca', dest='data_dir')
parser.add_argument('--n-show', type=int, default=6, dest='n_show',
                    help='Number of PCs shown in the coding heatmap')
args = parser.parse_args()

DATA, DUM = args.data_dir, args.dum
SCALE = DUM.split('_scale_')[1].split('_')[0] if '_scale_' in DUM else 'center'
FACTOR = DUM.split('_f-')[1] if '_f-' in DUM else 'odor_pair-tasks'
EPOCH = DUM.split('_')[1].lower()
CI = 'ci' + DUM.split('_ci')[1].split('_')[0] if '_ci' in DUM else 'ci0'
OUT = os.path.join('figures/pseudo/mixing', EPOCH, SCALE, CI, FACTOR)
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# ── load + compute ─────────────────────────────────────────────────────────────

X = pkl_load(f'pseudo_traj_{DUM}',   path=DATA)
y = pkl_load(f'pseudo_labels_{DUM}', path=DATA)

M, C, names = variable_mixing(X, y, stage='Expert')   # M (3,3), C (3,n_comp)
PR = participation_ratio(C)
mix = mixing_index(M)
PC_ID = identify_pcs(X, y, stage='Expert')
ns = min(args.n_show, C.shape[1])

print('traj', X.shape)
print('mixing index (mean off-diag |cos|):', round(mix, 3))
for v, pr in zip(names, PR):
    print(f'  {v:7s} participation ratio: {pr:.2f}')

# ── plot ───────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 + 0.5 * ns, 3.2),
                               gridspec_kw={'width_ratios': [ns, 3.2]})

# coding strength |contrast score| (variables × PCs)
sns.heatmap(np.abs(C[:, :ns]), ax=ax1, cmap='magma', annot=True, fmt='.2f',
            cbar_kws={'label': '|contrast score|'},
            xticklabels=[pc_label(k, PC_ID).replace('PC ', 'PC') for k in range(ns)],
            yticklabels=names, annot_kws={'fontsize': 6.5})
ax1.set_title('Coding strength per PC', loc='left', fontsize=TITLE_FS)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=35, ha='right', fontsize=7)

# variable × variable mixing |cos|
sns.heatmap(M, ax=ax2, cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
            cbar_kws={'label': '|cos| (mixing)'},
            xticklabels=names, yticklabels=names, square=True,
            annot_kws={'fontsize': 7})
ax2.set_title(f'Component mixing\n(mean off-diag = {mix:.3f})', loc='left', fontsize=TITLE_FS)

fig.suptitle(f'{SCALE} / {FACTOR}', fontsize=TITLE_FS, y=1.02)
fig.tight_layout()

png = os.path.join(OUT, 'png', f'{DUM}_mixing.png')
fig.savefig(png, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'svg', f'{DUM}_mixing.svg'), bbox_inches='tight')
plt.close(fig)
print('saved', png)

# ── time-resolved mixing ───────────────────────────────────────────────────────

Mt, Ct, energy, _ = variable_mixing_time(X, y, stage='Expert')   # (V,V,T),(V,T)
xtime = np.linspace(0, 14, Mt.shape[-1])
pairs = [(0, 1), (0, 2), (1, 2)]
pair_col = {'Sample-Choice': '#332288', 'Sample-Test': '#CC6677', 'Choice-Test': '#999933'}

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.5, 5), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
for i, j in pairs:
    name = f'{names[i]}-{names[j]}'
    axa.plot(xtime, gaussian_filter1d(Mt[i, j], 1.5), color=pair_col[name], lw=2, label=name)
add_vlines(axa, if_dpa=0)
axa.set_ylabel('|cos|  (mixing)')
axa.set_ylim(0, 1)
axa.legend(fontsize=6.5, frameon=False, loc='upper left')
axa.set_title('Time-resolved component mixing', loc='left', fontsize=TITLE_FS)

# coding-vector energy per variable (where |cos| is meaningful)
var_col = {'Sample': '#117733', 'Choice': '#332288', 'Test': '#CC6677'}
for i, v in enumerate(names):
    axb.plot(xtime, gaussian_filter1d(energy[i], 1.5), color=var_col.get(v, f'C{i}'),
             lw=1.6, label=v)
add_vlines(axb, if_dpa=0)
axb.set_ylabel('coding\nenergy')
axb.set_xlabel('Time (s)')
axb.set_xlim(0, 14)
axb.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=3)

fig.suptitle(f'{SCALE} / {FACTOR}', fontsize=TITLE_FS, y=1.0)
fig.tight_layout()
pngt = os.path.join(OUT, 'png', f'{DUM}_mixing_time.png')
fig.savefig(pngt, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'svg', f'{DUM}_mixing_time.svg'), bbox_inches='tight')
plt.close(fig)
print('saved', pngt)
