"""fig_traj_tasksplit.py — panel-A VARIANT (user request 2026-08-31): sample and choice codes
per TASK SET — columns DPA·sample | DPA·choice | dual·sample | dual·choice, rows Naive | Expert.

Replays ORIG_TRACES (exp_traj_orig.py; the '@dual' keys added 2026-08-31 carry the dual-trial
reads of the DPA-filtered canonical traces). Y-limits are shared per CODE across the two task
columns AND both stages, so DPA-vs-dual amplitude differences are directly readable.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_traj_tasksplit.py [--pca]
Output: figures/pseudo/dimensionality/png/fig_traj_tasksplit[_pca20].png (+svg)
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt

PCA = '--pca' in sys.argv[1:]
TKEY = 'ORIG_TRACES_pca20' if PCA else 'ORIG_TRACES'
FIGSUF = '_pca20' if PCA else ''

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
EVENTS = [('sample', 2.0, 3.0, SAMPC[0]), ('distractor', 4.5, 5.5, '#cc3311'),
          ('GNG cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]

RES = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))
assert TKEY in RES and ('Naive', 'sample@dual', 0) in RES[TKEY], \
    f'missing {TKEY} @dual keys — run exp_traj_orig.py' + (' --pca' if PCA else '')
TR = RES[TKEY]; xt = np.asarray(RES['ORIG_XTIME'])

# columns: (title, trace key, class labels, class colours) — Go and NoGo SEPARATED (2026-08-31)
SCL = (['Odor A', 'Odor B'], [SAMPC[0], SAMPC[1]])
LCL = (['No lick', 'Lick'], ['#377eb8', '#4daf4a'])
COLS = [('DPA · sample code',  'sample',       *SCL),
        ('DPA · choice code',  'lick',         *LCL),
        ('Go · sample code',   'sample@go',    *SCL),
        ('Go · choice code',   'lick@go',      *LCL),
        ('NoGo · sample code', 'sample@nogo',  *SCL),
        ('NoGo · choice code', 'lick@nogo',    *LCL)]

# y-limits shared per CODE (all task columns, both stages) — the cross-task comparison IS the point
YL = {}
for code_group, keys in [('sample', ['sample', 'sample@go', 'sample@nogo']),
                         ('choice', ['lick', 'lick@go', 'lick@nogo'])]:
    lo, hi = 0.0, 0.0
    for key in keys:
        for stage in ['Naive', 'Expert']:
            for lv in (0, 1):
                M = np.asarray(TR[(stage, key, lv)], dtype=float)
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                lo = min(lo, (mu - se).min()); hi = max(hi, (mu + se).max())
    pad = 0.05 * (hi - lo)
    YL[code_group] = (lo - pad, hi + pad)

fig, axs = plt.subplots(2, 6, figsize=(12.4, 4.2), sharex=True)
fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.12, wspace=0.42, hspace=0.16)
for r, stage in enumerate(['Naive', 'Expert']):
    for k, (ttl, key, labs, cols) in enumerate(COLS):
        ax = axs[r, k]
        for ei, (nm, lo, hi, col) in enumerate(EVENTS):
            ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0)
            if r == 0 and k == 0:
                yl = 0.905 if nm == 'distractor' else 0.98   # stagger the wide middle label
                ax.text((lo + hi) / 2, yl, nm, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=5.8, color=col)
        for lv, lab, col in zip((0, 1), labs, cols):
            M = np.asarray(TR[(stage, key, lv)], dtype=float)
            mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
            ax.plot(xt, mu, color=col, lw=1.5, label=f'{lab} (n={len(M)})', zorder=3)
            ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.20, lw=0, zorder=2)
        ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
        ax.set_ylim(*YL['sample' if 'sample' in key else 'choice'])
        ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4.5, 6.5, 9, 12])
        if r == 0:
            ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
            if k == 1:                              # ONE legend per code type (cols repeat them)
                ax.legend(frameon=False, fontsize=6.0, handlelength=1.2, loc='upper left')
        else:
            ax.set_xlabel('time (s)', fontsize=7)
            if k == 0:
                ax.legend(frameon=False, fontsize=6.0, handlelength=1.2, loc='lower right')
        ax.set_ylabel(f'{stage}\ncode depth' if k == 0 else 'code depth', fontsize=7)
        gap = np.abs(np.asarray(TR[(stage, key, 1)], float).mean(0)
                     - np.asarray(TR[(stage, key, 0)], float).mean(0))
        print(f'{stage:6s} {key:12s} max class gap {gap.max():5.2f} @ t={xt[gap.argmax()]:.1f}s')

OUT = 'figures/pseudo/dimensionality'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/fig_traj_tasksplit{FIGSUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_traj_tasksplit{FIGSUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_traj_tasksplit{FIGSUF}.png'))
