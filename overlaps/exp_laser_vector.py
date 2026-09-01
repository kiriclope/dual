"""exp_laser_vector.py — the ACC input's displacement VECTOR, decomposed by axis (idea #1,
2026-09-01 analysis menu). Question: when ACC->mPFC is silenced, does the delay state move
ALONG THE ACTION AXIS specifically, leaving the memory axis untouched? If yes, Fig 6's
"moves position, spares content" becomes a directional claim: the input is AIMED along the
shared action axis.

Per mouse x stage (DPA trials, late-delay read bins_LD 45-53): displacement = mean(laser ON)
- mean(laser OFF), computed WITHIN each sample class then averaged (removes class-composition
confounds), separately on
  - the CHOICE axis  (choice laser tensor, canonical trainLD_TEST axis 45-59;  = panel-I depth)
  - the SAMPLE axis  (sample laser tensor, generalisation train window 16-47)
each in that mouse's own baseline-SD unit (the opto figure's convention) so the two axes share
a per-mouse scale. Stats: Jaws (inhibition, n=5) |d_choice| vs |d_sample| paired Wilcoxon over
the 10 mouse x stage cells + per stage; ChR (n=2) printed, not tested.

Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_laser_vector.py
Output: figures/overlaps/laservec/png/exp_laser_vector.png (+svg) + printed stats + cache pkl.
"""
import matplotlib; matplotlib.use('Agg')
import os, sys, warnings, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from src.pca.io import pkl_load
from src.common.options import set_options

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

JAWS = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18']
CHR = ['ChRM04', 'ChRM23']
LASER_MICE = JAWS + CHR
_pal = sns.color_palette('tab10', n_colors=9)
ALL_MICE = JAWS + CHR + ['ACCM03', 'ACCM04']
MOUSE_COLOR = {m: _pal[i] for i, m in enumerate(ALL_MICE)}
DATA_IN = '../data/overlaps'
STAGES = ['Naive', 'Expert']

options = set_options(mice=LASER_MICE, tasks=['Dual'], mouse=LASER_MICE[0], laser=0,
                      trials='', data_type='dF', T_WINDOW=0.0, days=['first', 'last'],
                      n_comp=3, pca='pca', scaler=None, class_weight=0, multilabel=0,
                      mne_estimator='generalizing', n_jobs=4, prescreen=None, pval=0.05,
                      preprocess=None, scaler_BL='standard_BL', avg_noise=False,
                      unit_var_BL=False, random_state=None, l1_ratio=0.95,
                      bootstrap=1, n_boots=128, n_splits=5, n_repeats=10)
BINS_BL = np.asarray(options['bins_BL'])
BINS_LD = np.asarray(options['bins_LD'])                       # 45-53
TRAIN_LDTEST = np.concatenate([options['bins_LD'], options['bins_TEST']])   # 45-59

def _proj(tag, train_bins, cls_fn):
    """Load a laser tensor, average the train window; return LD reads in TWO per-mouse units:
    'bl' = baseline-SD (opto-fig convention) and 'ev' = class-signed pooled-evoked SD (the
    Figs 3/4 canonical unit; laser-OFF DPA trials, both stages)."""
    X = pkl_load(f'X_{tag}', path=DATA_IN)
    yy = pkl_load(f'labels_{tag}', path=DATA_IN)
    Xe = X[..., train_bins, :].mean(-2)[:, 1].astype(float); del X
    out = {}
    for unit in ('bl', 'ev'):
        Z = Xe.copy()
        for m in LASER_MICE:
            mm = (yy.mouse == m).values
            if unit == 'bl':
                sd = Z[mm][:, BINS_BL].std()
            else:
                pool = mm & (yy.laser == 0).values & (yy.tasks == 'DPA').values
                s = np.where(cls_fn(yy), 1.0, -1.0)[pool]
                vbar = (s[:, None] * Xe[pool]).mean(0)
                sd = (vbar - vbar[BINS_BL].mean()).std()
            if sd > 0:
                Z[mm] /= sd
        out[unit] = Z[:, BINS_LD].mean(1)
    return out, yy

print('loading choice laser tensor ...')
dep_c, y_c = _proj('log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice',
                   TRAIN_LDTEST, lambda yy: (yy.choice == 1).values)
print('loading sample laser tensor ...')
dep_s, y_s = _proj('log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_sample',
                   np.arange(16, 48), lambda yy: (yy.odor_pair < 2).values)

def _disp(dep, yy, mouse, stage):
    """ON-OFF displacement on DPA trials, within-class then class-averaged."""
    base = (yy.mouse == mouse).values & (yy.stage == stage).values & (yy.tasks == 'DPA').values
    ds = []
    for cls in (yy.odor_pair.values < 2, yy.odor_pair.values >= 2):   # A, B
        on = dep[base & cls & (yy.laser == 1).values]
        off = dep[base & cls & (yy.laser == 0).values]
        on, off = on[np.isfinite(on)], off[np.isfinite(off)]
        if len(on) >= 3 and len(off) >= 3:
            ds.append(on.mean() - off.mean())
    return np.mean(ds) if ds else np.nan

DFU = {}
for unit in ('bl', 'ev'):
    rows = []
    for m in LASER_MICE:
        for st in STAGES:
            rows.append(dict(mouse=m, stage=st, grp='Jaws' if m in JAWS else 'ChR',
                             d_choice=_disp(dep_c[unit], y_c, m, st),
                             d_sample=_disp(dep_s[unit], y_s, m, st)))
    DFU[unit] = pd.DataFrame(rows)
    print(f'\n══ unit = {unit} ══')
    print(DFU[unit].round(3).to_string(index=False))
    J = DFU[unit][DFU[unit].grp == 'Jaws'].dropna()
    ac, as_ = np.abs(J.d_choice.values), np.abs(J.d_sample.values)
    w_all = wilcoxon(ac, as_)
    print(f'Jaws |d_choice| vs |d_sample| (10 cells): medians {np.median(ac):.2f} vs '
          f'{np.median(as_):.2f}, Wilcoxon p={w_all.pvalue:.4f}')
    for st in STAGES:
        js = J[J.stage == st]
        if len(js) >= 4:
            w = wilcoxon(np.abs(js.d_choice), np.abs(js.d_sample))
            print(f'  {st}: |d_c| md {np.abs(js.d_choice).median():.2f} vs |d_s| md '
                  f'{np.abs(js.d_sample).median():.2f}, p={w.pvalue:.3f} (n={len(js)})')
DF = DFU['ev']

OUT = 'figures/overlaps/laservec'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
pickle.dump(DF, open(f'{OUT}/laser_vector.pkl', 'wb'))

fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
for ax, st in zip(axs, STAGES):
    sub = DF[DF.stage == st]
    for _, r in sub.iterrows():
        if not (np.isfinite(r.d_choice) and np.isfinite(r.d_sample)):
            continue
        mk = 'o' if r.grp == 'Jaws' else '^'
        ax.plot([0, 1], [abs(r.d_sample), abs(r.d_choice)], '-', color=MOUSE_COLOR[r.mouse],
                lw=0.8, alpha=0.5, zorder=2)
        ax.scatter([0, 1], [abs(r.d_sample), abs(r.d_choice)], s=34, marker=mk,
                   color=MOUSE_COLOR[r.mouse], edgecolors='w', linewidths=0.6, zorder=3)
    js = sub[sub.grp == 'Jaws'].dropna()
    if len(js) >= 4:
        p = wilcoxon(np.abs(js.d_choice), np.abs(js.d_sample)).pvalue
        sig = p < .05
        ax.text(0.5, 0.95, '*' if sig else 'n.s.', transform=ax.transAxes, ha='center',
                fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
        ax.text(0.5, 0.85, f'p={p:.3f}', transform=ax.transAxes, ha='center',
                fontsize=6.5, color='0.3')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['sample\naxis', 'choice\naxis'])
    ax.set_xlim(-0.4, 1.4); ax.set_ylabel('|laser ON−OFF displacement| (BL-SD)' if st == 'Naive' else '')
    ax.set_title(st, loc='left', fontsize=TITLE_FS)
fig.suptitle('The ACC input displaces the delay state along the choice axis, not the memory axis',
             x=0.02, ha='left', fontsize=TITLE_FS)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{OUT}/png/exp_laser_vector.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/exp_laser_vector.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/exp_laser_vector.png'))
