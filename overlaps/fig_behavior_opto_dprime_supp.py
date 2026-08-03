"""
fig_behavior_opto_dprime_supp.py — supplementary figure S17.

Standalone reproduction of ONLY panels K and L of the behavioural OPTO main figure
(fig_behavior_opto_main.py): the neural DISCRIMINABILITY (d′) laser ON-vs-OFF scatters
showing the codes are SPARED under ACC→Prl(mPFC) silencing.

  A  DPA memory code d′(odor A vs B) on the SAMPLE axis at LATE delay (bins_LD 45-53).
  B  GNG code d′(Go vs NoGo) on the CHOICE/GNG axis at MID delay (bins_MD, the Go/NoGo cue).

Each panel: 5 Jaws mice × {Naive ○, Expert ●} = 10 points, laser ON (y) vs laser OFF (x).
Points on the dashed unity line = discriminability unchanged by the laser = code spared.
Stat = LMM d′ ~ C(laser) + C(stage) + (1|mouse): sample laser p≈0.34, GNG laser p≈0.74 (ns).

Data dependency: needs BOTH laser tensors in ../data/overlaps (gitignored — regenerate):
  run_overlaps.py --scaler none --no-raw --with-laser --targets choice   (GNG panel B)
  run_overlaps.py --scaler none --no-raw --with-laser --targets sample   (sample panel A)

Helpers (_dprime, _build_dpr, _lmm_laser, _dprime_scatter) copied inline from
fig_behavior_opto_main.py per repo convention, so that figure stays untouched.

Output: figures/overlaps/behavior/{png,svg}/behavior_opto_dprime.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_opto_dprime_supp.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import statsmodels.formula.api as smf

from src.common.options import set_options
from src.pca.io import pkl_load

# ── canonical Nature-Neuroscience house style ─────────────────────────────────
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

JAWS = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18']   # ACC→Prl INHIBITION
CHR  = ['ChRM04', 'ChRM23']                                      # ACC→Prl EXCITATION
LASER_MICE = JAWS + CHR
ALL_MICE = JAWS + CHR + ['ACCM03', 'ACCM04']                     # keep colour keying identical to main fig
_pal = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal[i] for i, m in enumerate(ALL_MICE)}
STAGES = ['Naive', 'Expert']

DATA_IN = '../data/overlaps'
DUM   = 'log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice'   # GNG / choice axis
DUM_S = 'log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_sample'   # sample axis

OUT = 'figures/overlaps/behavior'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)


def _require(name):
    try:
        return pkl_load(name, path=DATA_IN)
    except Exception as e:
        target = 'sample' if 'targets_sample' in name else 'choice'
        sys.exit(
            f"\nMISSING TENSOR: {name}.pkl not loadable from {DATA_IN} ({e})\n"
            f"Regenerate with:\n"
            f"  cd /home/leon/dual/overlaps && "
            f"python run_overlaps.py --scaler none --no-raw --with-laser --targets {target}\n")


# ── read windows (bins_LD for the late-delay sample read, bins_MD for the mid-delay GNG read) ──
options = set_options(
    mice=LASER_MICE, tasks=['Dual'], mouse=LASER_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_LD = options['bins_LD']
BINS_MD = options['bins_MD']

# ── LOAD choice tensor → GNG d′ (panel B) ─────────────────────────────────────
print('loading choice/GNG laser tensor …')
X = _require(f'X_{DUM}')
y = _require(f'labels_{DUM}')
print(f'  X {X.shape}  y {len(y)}')
cdf_diag = np.stack([X[:, 1, t, t] for t in range(X.shape[-1])], axis=1).astype(float)  # choice DV diag(t)
del X                                                          # free ~1 GB

# ── LOAD sample tensor → sample d′ (panel A) ──────────────────────────────────
print('loading sample-axis laser tensor …')
Xs = _require(f'X_{DUM_S}')
ys = _require(f'labels_{DUM_S}')
print(f'  Xs {Xs.shape}  ys {len(ys)}')
sdf_diag = np.stack([Xs[:, 1, t, t] for t in range(Xs.shape[-1])], axis=1).astype(float)  # sample DV diag(t)
del Xs                                                         # free ~1 GB


def _dprime(v, mask, pos, neg):
    """Neural d' = (μ_pos − μ_neg)/σ_pooled of the decision function under `mask`."""
    a = v[mask & pos]; b = v[mask & neg]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


def _build_dpr(v, base, mo_arr, la_arr, st_arr, pos, neg):
    """One d′ per (Jaws mouse × stage) at laser OFF and ON — 10 rows."""
    rows = []
    for st in STAGES:
        for m in JAWS:
            cell = base & (mo_arr == m) & (st_arr == st)
            rows.append(dict(mouse=m, stage=st,
                             off=_dprime(v, cell & (la_arr == 0), pos, neg),
                             on=_dprime(v, cell & (la_arr == 1), pos, neg)))
    return pd.DataFrame(rows)


def _lmm_laser(dfw):
    """LMM d′ ~ C(laser) + C(stage) + (1|mouse); returns (β_laser, p_laser)."""
    long = pd.concat([dfw.assign(d=dfw.off, laser=0), dfw.assign(d=dfw.on, laser=1)]).dropna(subset=['d'])
    r = smf.mixedlm('d ~ C(laser) + C(stage)', long, groups=long['mouse']).fit(reml=False)
    lc = [k for k in r.params.index if 'laser' in k][0]
    return float(r.params[lc]), float(r.pvalues[lc])


# ── decision-function values in the read windows + class masks ────────────────
# A: sample axis, odor A (sample==1) vs B (sample==0), DPA trials, late delay bins_LD.
_sA, _sB = (ys['sample'].values == 1), (ys['sample'].values == 0)
_sdpa = (ys.tasks == 'DPA').values
sLD = sdf_diag[:, BINS_LD].mean(1)
# B: choice axis, Go vs NoGo, dual trials, mid delay bins_MD (the Go/NoGo cue).
_gGo, _gNo = (y.tasks == 'DualGo').values, (y.tasks == 'DualNoGo').values
cMD = cdf_diag[:, BINS_MD].mean(1)

DPR = {'sample': _build_dpr(sLD, _sdpa, ys.mouse.values, ys.laser.values, ys.stage.values, _sA, _sB),
       'gng': _build_dpr(cMD, (_gGo | _gNo), y.mouse.values, y.laser.values, y.stage.values, _gGo, _gNo)}
LMM_DPR = {k: _lmm_laser(v) for k, v in DPR.items()}
for k in ('sample', 'gng'):
    n = int(np.isfinite(DPR[k].off.values).sum() + np.isfinite(DPR[k].on.values).sum())
    print(f'{k} d′: LMM laser β={LMM_DPR[k][0]:+.3f} p={LMM_DPR[k][1]:.3f}  ({n} finite d′ over 10 mice×stage)')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — 2 square d′ scatters
# ══════════════════════════════════════════════════════════════════════════════
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 3.3))
fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.15, wspace=0.42)


def _dprime_scatter(ax, dfw, lmm, title):
    vals = np.concatenate([dfw.off.values, dfw.on.values]); vals = vals[np.isfinite(vals)]
    lo = min(0.0, vals.min()) - 0.1; hi = vals.max() + 0.15
    ax.plot([lo, hi], [lo, hi], '--', color='0.5', lw=1, zorder=1)          # unity = spared
    for _, r in dfw.iterrows():
        fc = MOUSE_COLOR[r.mouse] if r.stage == 'Expert' else 'w'           # Expert filled / Naive open
        ax.scatter(r.off, r.on, facecolors=fc, edgecolors=MOUSE_COLOR[r.mouse], s=55, lw=1.1, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
    ax.set_xlabel("d′  laser OFF"); ax.set_ylabel("d′  laser ON")
    ax.set_title(title, loc='left', fontsize=TITLE_FS)
    ax.text(0.5, 0.02, f'LMM laser p={lmm[1]:.2f}  (n=10, +1|mouse)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=6.5, color='0.3')


_dprime_scatter(axA, DPR['sample'], LMM_DPR['sample'], 'DPA memory code (A vs B, late delay)')
_dprime_scatter(axB, DPR['gng'], LMM_DPR['gng'], 'GNG code (Go vs NoGo, mid-delay)')
axB.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=6, label='Expert'),
                    mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=6, label='Naive')],
           frameon=False, fontsize=6.5, loc='upper left', handletextpad=0.3)

# ── bold panel letters ────────────────────────────────────────────────────────
for ax, L in [(axA, 'A'), (axB, 'B')]:
    p = ax.get_position()
    fig.text(p.x0 - 0.055, p.y1 + 0.045, L, fontsize=11, fontweight='bold', va='top', ha='left')

for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_opto_dprime.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
