"""BRIDGE Fig 2 → Fig 4 (ALTERNATIVE, 2026-08-11): the no-lick push drawn INSIDE the memory × action
plane that Fig 2 established — x = per-mouse sample-code depth, y = per-mouse CALIBRATED lick-axis
depth (main_panels.lick_depth = Fig 4's quantity: DPA action decoder @57–63, LD-window depth,
class-signed pooled-evoked norm, ONE unit per mouse shared across stages → Naive→Expert
displacements are literal). Arrows = each mouse's Naive→Expert displacement of its sample-A and
sample-B DPA delay states; bold arrows = group means. The push = the A arrows sliding toward the
no-lick side while B stays — "geometric editing" in Fig 2's coordinate frame.

Why this pipeline and not the pseudo-population (settled): condition-centred analyses subtract the
grand mean (the push IS a grand-mean translation), per-stage z framing destroys cross-stage
position, and the pseudo-pop lick axis carries a sample-uniform Expert outcome offset that buries
the push. Statistics are Fig 4's (per-mouse Wilcoxon n.s.; C(sample) LMM p≈.03–.05) — this panel
adds no new test, it renders Fig 4's quantity in Fig 2's frame.

Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python fig_push_bridge.py
Output: figures/overlaps/main/{png,svg}/fig_push_bridge.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import main_panels as MP
import seaborn as sns, matplotlib.pyplot as plt
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
SAMPC = {0: '#332288', 1: '#44AA99'}
PAIRS = {0: [0, 1], 1: [2, 3]}                                        # A, B odor pairs

sample_depth = MP.SAMPLE_D[:, MP.BINS_LATE].mean(1)
Ys = MP.Y_SAM
pos = {}                                                              # (mouse, cls, stage) -> (x, y)
for mo in MP.ALL_MICE:
    for cls, pairs in PAIRS.items():
        for stage in MP.STAGES:
            my = ((MP.Lm.mouse == mo) & (MP.Lm.stage == stage) & MP.L_dpa
                  & MP.Lm.odor_pair.isin(pairs)).values
            mx = ((Ys.mouse == mo) & (Ys.stage == stage) & (Ys.laser == 0).values
                  & (Ys.tasks == 'DPA').values & Ys.odor_pair.isin(pairs).values)
            pos[(mo, cls, stage)] = (float(sample_depth[mx].mean()) if mx.sum() else np.nan,
                                     float(MP.lick_depth[my].mean()) if my.sum() else np.nan)

fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.axhspan(-4.5, 0, color='#cc3311', alpha=0.045, lw=0)
ax.text(0.985, 0.03, 'no-lick side', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=6.5, color='#a03020', alpha=0.8)
for cls in (0, 1):
    col = SAMPC[cls]
    for mo in MP.ALL_MICE:
        xn, yn = pos[(mo, cls, 'Naive')]; xe, ye = pos[(mo, cls, 'Expert')]
        if np.isnan(xn) or np.isnan(xe):
            continue
        ax.annotate('', xy=(xe, ye), xytext=(xn, yn),
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color=col, alpha=0.35))
        ax.plot(xn, yn, 'o', ms=3.6, mfc='w', mec=col, mew=0.8, alpha=0.8)
        ax.plot(xe, ye, 'o', ms=3.6, mfc=col, mec=col, alpha=0.8)
    XN = np.nanmean([pos[(mo, cls, 'Naive')][0] for mo in MP.ALL_MICE])
    YN = np.nanmean([pos[(mo, cls, 'Naive')][1] for mo in MP.ALL_MICE])
    XE = np.nanmean([pos[(mo, cls, 'Expert')][0] for mo in MP.ALL_MICE])
    YE = np.nanmean([pos[(mo, cls, 'Expert')][1] for mo in MP.ALL_MICE])
    ax.annotate('', xy=(XE, YE), xytext=(XN, YN),
                arrowprops=dict(arrowstyle='-|>', lw=2.6, color=col))
    lab = 'A' if cls == 0 else 'B'
    ax.text(XE, YE - 0.28, f'sample {lab}', ha='center', va='top', fontsize=7.5,
            color=col, fontweight='bold')
    print(f'sample {lab}: Naive ({XN:+.2f}, {YN:+.2f}) -> Expert ({XE:+.2f}, {YE:+.2f})  '
          f'Δdepth {YE - YN:+.2f}')
ax.axhline(0, color='0.8', lw=0.7)
ax.set_xlabel('sample-code depth (per-mouse units)   A ← · → B')
ax.set_ylabel('action-axis depth (per-mouse units)\n← no-lick · lick →')
ax.set_title('learning repositions the sample-A memory state toward no-lick',
             loc='left', fontsize=TITLE_FS)
hs = [mlines.Line2D([], [], marker='o', ls='', ms=4.5, mfc='w', mec='0.35', label='Naive'),
      mlines.Line2D([], [], marker='o', ls='', ms=4.5, mfc='0.35', mec='0.35', label='Expert'),
      mlines.Line2D([], [], color=SAMPC[0], lw=2, label='sample A'),
      mlines.Line2D([], [], color=SAMPC[1], lw=2, label='sample B')]
ax.legend(handles=hs, frameon=False, fontsize=6.5, loc='upper left')
fig.suptitle('The push in the memory × action plane (bridge Fig 2 → Fig 4)',
             x=0.09, ha='left', y=0.985, fontsize=11)
fig.text(0.09, 0.005,
         'DPA delay states, per mouse (thin arrows = Naive→Expert; bold = group mean): x = sample-'
         'code depth, y = the CALIBRATED action-axis depth (Fig 4\'s quantity: per-mouse DPA lick '
         'decoder @57–63,\nlate-delay depth, class-signed pooled-evoked norm — one unit per mouse '
         'shared across stages, so displacements are literal). Statistics are Fig 4\'s (per-mouse '
         'Wilcoxon n.s.; C(sample) LMM p≈.03–.05);\nthis panel renders that quantity in Fig 2\'s '
         'coordinate frame — "pushed into the no-lick region, where the NoGo states sit" '
         '(no dynamical/attractor claim).',
         fontsize=5.8, color='0.35', va='bottom', ha='left')
fig.tight_layout(rect=(0.02, 0.06, 1, 0.93))
PSUF = '_pca20' if getattr(MP, 'PCA20', False) else ''   # --pca -> separate file
OUT = 'figures/overlaps/main'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_push_bridge{PSUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_push_bridge{PSUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_push_bridge{PSUF}.png'))
