"""
plot_scatter_laser.py — CAUSAL analog of the Δdepth ↔ Δperformance scatter.

Instead of Δ(Expert − Naive), plot Δ(laser_ON − laser_OFF): does the optogenetic
manipulation that moves the delay-period choice-code depth also move accuracy?
(cf. Fig 7: delay-period ACC→mPFC inhibition displaces activity along the lick axis
and impairs DPA.)

Depth is read from the `_laser` CCGD tensor (laser-ON trials projected through the
laser-OFF-trained decoders — same axis, held out), on the LOCKED trainLD_TEST read-out
axis. Both laser conditions come from one run, so on−off is a valid within-mouse Δ.

Groups (Expert only; ACCM03/04 have NO laser trials → excluded):
  - Jaws (n=5) = INHIBITION   (JawsM01/06/12/15/18)  — red circles
  - ChR  (n=2) = EXCITATION   (ChRM04, ChRM23)       — blue triangles
Points are colored/marked by group, but the regression + r/ρ are POOLED over all 7
mice (no sign flip) — mixes opposite-signed manipulations, so read as descriptive.

Panels (1×2, shared y): Δdepth(DPA) vs Δ DPA accuracy | vs Δ GNG accuracy.
x is always the DPA choice-code depth; the GNG panel is the specificity control.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python plot_scatter_laser.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, linregress, t as t_dist

from src.common.options import set_options
from src.pca.io import pkl_load

sns.set_style('ticks')
matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none',
})

# ── Config ────────────────────────────────────────────────────────────────────

# Both-stages _laser tensor (Naive + Expert). STAGE mode selects which rows to use.
DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/scatter_laser'

# stage mode: 'pooled' (Naive+Expert) | 'expert' (Expert only). CLI token; default expert.
MODE = next((m for m in ('pooled', 'expert') if m in sys.argv[1:]), 'expert')

JAWS = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18']   # inhibition
CHR  = ['ChRM04', 'ChRM23']                                      # excitation
LASER_MICE = JAWS + CHR
GROUP = {**{m: 'Jaws' for m in JAWS}, **{m: 'ChR' for m in CHR}}
GMARKER = {'Jaws': 'o', 'ChR': '^'}                 # ● inhibition / ▲ excitation

# One color per animal — IDENTICAL to the main-figure scatter (plot_scatter_perf.py):
# tab10 indexed by position in the full 9-mouse list.
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
_pal = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal[i] for i, m in enumerate(ALL_MICE)}

options = set_options(
    mice=LASER_MICE, tasks=['Dual'], mouse=LASER_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL   = options['bins_BL']
BINS_LATE = np.arange(27, 54)                                    # test-time late-delay window

# Read-out (train) axis, selectable on the CLI: ld_test (default) | ld | delay
AXES = {
    'ld_test': np.concatenate([options['bins_LD'], options['bins_TEST']]),  # 45-59
    'ld':      options['bins_LD'],                                           # 45-53
    'delay':   options['bins_DELAY'],                                        # 18-53
}
AXIS = next((a for a in AXES if a in sys.argv[1:]), 'ld_test')
BINS_TRAIN = AXES[AXIS]

# ── Load the _laser tensor ────────────────────────────────────────────────────

X = pkl_load(f'X_{DUM}', path=DATA_IN)
y = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'X {X.shape}  y {len(y)}')

# choice-code decision function on the trainLD_TEST axis, per-mouse BL-std normalised
X_ep = X[..., BINS_TRAIN, :].mean(-2)[:, 1].astype(float)        # (n, 84) over test-time
for m in LASER_MICE:
    mm = (y.mouse == m).values
    sd = X_ep[mm][:, BINS_BL].std()
    if sd > 0:
        X_ep[mm] /= sd
depth_all = X_ep[:, BINS_LATE].mean(1)                          # per-trial late-delay depth

is_choice = (y.target == 'choice').values
is_dpa    = (y.tasks == 'DPA').values
is_gng    = (y.tasks != 'DPA').values
# stage mask: 'expert' keeps Expert rows only; 'pooled' keeps Naive+Expert
STAGE_MASK = (y.stage == 'Expert').values if MODE == 'expert' \
    else np.ones(len(y), dtype=bool)
print(f'stage mode = {MODE}  ({STAGE_MASK.sum()} of {len(y)} rows kept)')

def _pooled_depth(mmask, laser_val):
    """Equal-weight A&B pooled depth: mean of the per-sample-class (A,B) means —
    matches the main-figure deepening panel (exp_nolick_push_stats), not a raw
    trial-weighted average. A = odor_pairs [0,1], B = [2,3]."""
    vals = []
    for pairs in ([0, 1], [2, 3]):
        m = (mmask & is_choice & is_dpa & STAGE_MASK & (y.laser == laser_val).values
             & y.odor_pair.isin(pairs).values)
        if m.sum():
            vals.append(depth_all[m].mean())
    return float(np.mean(vals)) if vals else np.nan

def _perf_mean(col, task_mask, laser_val):
    m = (y.target == 'choice') & task_mask & STAGE_MASK & (y.laser == laser_val)
    v = y.loc[m.values, col].dropna()
    return v.mean() if len(v) else np.nan

# ── Per-mouse Δ(on − off) ─────────────────────────────────────────────────────

rows = []
for mouse in LASER_MICE:
    mmask = (y.mouse == mouse).values
    d_on  = _pooled_depth(mmask, 1)
    d_off = _pooled_depth(mmask, 0)
    dpa_on  = _perf_mean('performance', (y.tasks == 'DPA') & (y.mouse == mouse), 1)
    dpa_off = _perf_mean('performance', (y.tasks == 'DPA') & (y.mouse == mouse), 0)
    gng_on  = _perf_mean('odr_perf',    (y.tasks != 'DPA') & (y.mouse == mouse), 1)
    gng_off = _perf_mean('odr_perf',    (y.tasks != 'DPA') & (y.mouse == mouse), 0)
    rows.append(dict(
        mouse=mouse, group=GROUP[mouse],
        d_depth=d_on - d_off,
        d_dpa=dpa_on - dpa_off,
        d_gng=gng_on - gng_off,
    ))

print(f'\nPer-mouse Δ(on−off)   [depth on train{AXIS.upper()}, late delay bins 27–53]')
print(f'{"mouse":9s} {"grp":5s} {"Δdepth":>8s} {"ΔDPA":>8s} {"ΔGNG":>8s}')
for r in rows:
    print(f'{r["mouse"]:9s} {r["group"]:5s} {r["d_depth"]:+8.3f} '
          f'{r["d_dpa"]:+8.3f} {r["d_gng"]:+8.3f}')

# ── Day-paired Δ(on−off) for the LMM ──────────────────────────────────────────
# Laser is interleaved WITHIN day, so each (mouse, day) yields a paired on−off Δ.
# This is the hierarchical boost of the 7-point scatter: unit = (mouse, day),
# Δperf ~ Δdepth + (1+Δdepth|mouse). Depth is DPA choice-code; ΔDPA from DPA trials,
# ΔGNG from Go/NoGo trials in the same (mouse,day,laser) cell.
_dfl = pd.DataFrame({
    'mouse':  y.mouse.values, 'stage': y.stage.values, 'day': y.day.values,
    'laser':  y.laser.values, 'depth': depth_all,
    'dpa':    y.performance.values, 'gng': y.odr_perf.values,
    'is_dpa': (y.tasks == 'DPA').values, 'is_choice': (y.target == 'choice').values,
})
_dfl = _dfl[_dfl.is_choice & _dfl.mouse.isin(LASER_MICE)]
if MODE == 'expert':
    _dfl = _dfl[_dfl.stage == 'Expert']

def _delta_by_day(sub, col):
    """(mouse,stage,day) on−off difference of the cell-mean of `col`. Grouping by
    stage keeps Naive/Expert day cells separate when pooling."""
    u = sub.groupby(['mouse', 'stage', 'day', 'laser'])[col].mean().unstack('laser')
    if 0.0 not in u or 1.0 not in u:
        return pd.Series(dtype=float)
    return (u[1.0] - u[0.0]).dropna()

delta = pd.concat({
    'ddepth': _delta_by_day(_dfl[_dfl.is_dpa],  'depth'),
    'ddpa':   _delta_by_day(_dfl[_dfl.is_dpa],  'dpa'),
    'dgng':   _delta_by_day(_dfl[~_dfl.is_dpa], 'gng'),
}, axis=1).reset_index()
print(f'\nDay-paired Δ: {len(delta)} (mouse,stage,day) cells across '
      f'{delta.mouse.nunique()} mice [{MODE}]')

def _lmm_one(d, ycol, reform):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = smf.mixedlm(f'{ycol} ~ ddepth', d, groups=d['mouse'],
                              re_formula=reform).fit(reml=True, method='lbfgs')
        return dict(b=float(res.params['ddepth']), se=float(res.bse['ddepth']),
                    p=float(res.pvalues['ddepth']), conv=bool(res.converged))
    except Exception:
        return None

def lmm_report(df, ycol):
    """Δycol ~ Δdepth with mouse REs. Returns dict with 'max' (random slope,
    lab-standard maximal), 'ri' (random intercept), and 'ols' (no RE) fits, plus n.
    Primary = maximal if it converged, else random-intercept, else OLS."""
    d = df.dropna(subset=['ddepth', ycol])
    out = {'n': len(d),
           'max': _lmm_one(d, ycol, '~ddepth'),
           'ri':  _lmm_one(d, ycol, '~1')}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            o = smf.ols(f'{ycol} ~ ddepth', d).fit()
        out['ols'] = dict(b=float(o.params['ddepth']), se=float(o.bse['ddepth']),
                          p=float(o.pvalues['ddepth']), conv=True)
    except Exception:
        out['ols'] = None
    if out['max'] and out['max']['conv']:
        out['primary'], out['ptag'] = out['max'], 'max'
    elif out['ri']:
        out['primary'], out['ptag'] = out['ri'], 'ri'
    else:
        out['primary'], out['ptag'] = out['ols'], 'ols'
    return out

# ── Plot helpers ──────────────────────────────────────────────────────────────

def regression_band(ax, xs, ys, color='k'):
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return
    xv, yv = xs[ok], ys[ok]
    slope, icpt, _, _, se = linregress(xv, yv)
    xl = np.linspace(xv.min(), xv.max(), 100)
    yl = slope * xl + icpt
    ssx = np.sum((xv - xv.mean()) ** 2)
    seb = se * np.sqrt(1/len(xv) + (xl - xv.mean())**2 / ssx)
    tc = t_dist.ppf(0.975, df=len(xv) - 2)
    ax.plot(xl, yl, color=color, lw=1.5, zorder=4)
    ax.fill_between(xl, yl - tc*seb, yl + tc*seb, color=color, alpha=0.15, zorder=2)

def stats_txt(xs, ys):
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return 'n<3', 1.0
    r, pr = pearsonr(xs[ok], ys[ok]); rho, ps = spearmanr(xs[ok], ys[ok])
    return (f'all (n={ok.sum()}): r={r:+.2f} p={pr:.3f}  ρ={rho:+.2f} p={ps:.3f}',
            float(ps))

# ── Figure: 1×2 (ΔDPA | ΔGNG) vs Δdepth ───────────────────────────────────────

os.makedirs(os.path.join(FIG_BASE, 'png'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'svg'), exist_ok=True)

xdepth = np.array([r['d_depth'] for r in rows])
groups = np.array([r['group'] for r in rows])
specs = [('d_dpa', 'Δ DPA accuracy (on−off)'), ('d_gng', 'Δ GNG accuracy (on−off)')]

ally = np.array([r[k] for k, _ in specs for r in rows], float)
ally = ally[~np.isnan(ally)]
pad = (ally.max() - ally.min()) * 0.15 or 0.05
ylim = (ally.min() - pad, ally.max() + pad)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
for ax, (key, ylabel) in zip(axes, specs):
    yv = np.array([r[key] for r in rows])
    # one color per animal (same tab10 colors as the main-figure scatter);
    # marker still encodes inhibition (Jaws ●) vs excitation (ChR ▲)
    for i, mouse in enumerate(LASER_MICE):
        ax.scatter(xdepth[i], yv[i], color=MOUSE_COLOR[mouse],
                   marker=GMARKER[GROUP[mouse]], s=90, edgecolors='w',
                   linewidths=0.6, zorder=5,
                   label=mouse if ax is axes[1] else None)
    # pooled regression over ALL 7 laser mice (Jaws + ChR, NO sign flip)
    regression_band(ax, xdepth, yv, color='0.25')
    ax.axhline(0, ls=':', color='k', lw=0.8); ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_ylim(ylim)
    txt, ps = stats_txt(xdepth, yv)
    # Figure reports the 7-mouse Spearman (stable across Expert/pooled, matches the 7
    # dots, honest about n=animals). The day-paired LMM is unstable here (≤3 days,
    # 7 clusters → maximal NC / random-intercept singular, model-dependent p) so it is
    # kept in STDOUT only, NOT shown on the figure.
    L = lmm_report(delta, key.replace('d_', 'd'))
    pr = L['primary']
    def _f(m): return f"β={m['b']:+.3f} p={m['p']:.4f}{'' if m['conv'] else '*NC'}" if m else 'n/a'
    print(f"  {ylabel:26s} {txt}")
    print(f"      [stdout only] LMM day-paired PRIMARY[{L['ptag']}]:[{_f(pr)}]  "
          f"max:[{_f(L['max'])}]  ri:[{_f(L['ri'])}]  ols:[{_f(L['ols'])}]  (n={L['n']} pairs)")
    ax.text(0.5, 1.02, txt, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8.5, color='0.3')
    star = '*' if ps < 0.05 else 'n.s.'
    ax.text(0.9, 0.93, star, transform=ax.transAxes, ha='center', va='top',
            fontsize=22, fontweight='bold', color='k' if ps < 0.05 else '0.55')
    ax.set_xlabel('Δ DPA choice-code depth (on−off)')
    ax.set_ylabel(ylabel)

axes[1].legend(frameon=False, fontsize=8, loc='upper left',
               bbox_to_anchor=(1.01, 1), title='mouse (● Jaws / ▲ ChR)',
               title_fontsize=8)
_stage_lbl = 'Naive+Expert' if MODE == 'pooled' else 'Expert'
fig.suptitle(f'Laser ON−OFF: Δ depth vs Δ performance  ({_stage_lbl}; '
             f'train{AXIS.upper()}, late delay 27–53)', fontsize=11, y=1.02)
fig.tight_layout()
for ext in ('png', 'svg'):
    out = os.path.join(FIG_BASE, ext, f'{DUM}_onoff_{AXIS}_{MODE}.{ext}')
    fig.savefig(out, bbox_inches='tight')
    print(f'saved {out}')
plt.close(fig)
