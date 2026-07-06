"""
fig_behavior_opto_main.py — behavioural OPTO main figure (companion to
fig_behavior_main.py). One unified story about the ACC→mPFC(Prl) projection:

  A  Scheme (~/dual/opto.png): recorded-cohort design — hSyn-GCaMP6s imaging in mPFC +
     CaMKII-Jaws-tdTomato in ACC, 635 nm laser-on 50% pseudo-random delay trials.
  ── Training batch · chronic every-trial silencing · BETWEEN-group opto vs control ──
  B  DPA performance   control vs opto   (ACC-Prl batch, 9v9)   vs day
  C  GNG performance   control vs opto
  D  DPA unpaired      control vs opto   (the strongest deficit)
  E  LMM group effect (opto−control) per metric: group β (○) + group×day slope (□).
  ── Recorded cohort · transient delay-only laser · WITHIN-mouse ON vs OFF ──
  F  DPA performance   OFF vs ON  (Jaws inhibition, n=5)   vs session
  G  GNG performance   OFF vs ON  (Jaws, n=5)              vs session
  H  Per-mouse choice-code depth, laser OFF vs ON (Jaws, A&B pooled) — the manipulation
     moves each animal's code (its shift is the x-axis of I–K); group mean flat.
  ── Same projection · overlaps causal coupling (laser ON−OFF) ──
  I  TRADE-OFF contrast (headline): Δdepth vs (ΔDPA − ΔGNG) — depth↑ predicts DPA↑ AND GNG↓
     jointly. 20 pts, Pearson r=+0.48 p=0.034 (Expert-10 r=+0.75 p=0.013) — significant on the
     pre-committed trainLD_TEST axis, no window search. J/K are its two arms (K `*`, J n.s. trend).
  J  Δ DPA choice-code depth (on−off)  vs  Δ DPA accuracy   (5 Jaws; Naive▲+Expert● × A&B, 20 pts)
  K  Δ DPA choice-code depth (on−off)  vs  Δ GNG accuracy   (the coupled one; r=−0.61 p=.004,
     ρ=−0.56 p=.011 — a between-animal coupling; per-mouse-mean r=−0.80, robust across slicings)
  Depth read on the trainLD_TEST axis (bins 45-59, main-overlaps-fig convention); readout window
  27-53 (delay, pre-response); I–K square, all trials.
  ── Last row: behavioural balance under silencing + code discriminability (recorded, 5 Jaws) ──
  L  DPA vs GNG performance in laser-ON trials (balance plane of the non-opto main figure),
     5 Jaws × {Naive○, Expert●} = 10 pts; optimal corner starred (r=+0.44 p=0.20, descriptive).
  M  d′ laser ON vs OFF scatter, DPA memory code: sample-axis d′(A vs B) at late delay
     (bins_LD 45-53). 5 Jaws × {Naive○, Expert●} = 10 pts; on unity = spared.
     LMM d′~laser+stage+(1|mouse): laser p=0.34 (ns).
  N  d′ ON vs OFF scatter, GNG code: choice-axis d′(Go vs NoGo) at mid-delay
     (bins_MD 33-38, the Go/NoGo cue). 10 pts; LMM laser p=0.74 (ns) — spared.

Message: chronic every-trial ACC→Prl silencing degrades DPA, carried by unpaired trials
(B–E); transient delay-only ACC→Prl perturbation spares GROSS behaviour (F,G,H) but
demonstrably moves the choice code's POSITION (I), while sparing its READOUT (L) and its
DISCRIMINABILITY (M,N) — and Δdepth predicts Δaccuracy across animals (I–K).

Data dependency: needs BOTH laser tensors in ../data/overlaps (gitignored — regenerate):
  run_overlaps.py --scaler none --no-raw --with-laser --targets choice   (depth, I–L)
  run_overlaps.py --scaler none --no-raw --with-laser --targets sample   (sensitivity, M,N)

Helpers copied inline (per repo convention) from fig_behavior_learning_offon.py,
fig_behavior_learning_batch.py (--ctrlopto) and plot_scatter_laser.py, so those stay
untouched. Two statistical designs: recorded = within-mouse (interleaved laser);
batch = between-group (every-trial silencing, different animals).

Output: figures/overlaps/behavior/{png,svg}/behavior_opto_main.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_opto_main.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, glob, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import seaborn as sns
import scipy.io as sio
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_ind, pearsonr, spearmanr, linregress, t as t_dist, norm

from src.common.options import set_options
from src.pca.io import pkl_load

sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.9, 'lines.linewidth': 1.8,
})

RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'
OFF_C, ON_C = '#888888', '#332288'          # OFF/control grey · ON/opto indigo
N_MIN = 3
TITLE_FS = 10.5

JAWS = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18']   # ACC→Prl INHIBITION
CHR  = ['ChRM04', 'ChRM23']                                      # ACC→Prl EXCITATION
LASER_MICE = JAWS + CHR
GROUP = {**{m: 'Jaws' for m in JAWS}, **{m: 'ChR' for m in CHR}}
GMARKER = {'Jaws': 'o', 'ChR': '^'}
ALL_MICE = JAWS + CHR + ['ACCM03', 'ACCM04']
_pal = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal[i] for i, m in enumerate(ALL_MICE)}

DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice'
DATA_IN = '../data/overlaps'
BATCH = 'DualTask-Silencing-ACC-Prl'
DATA_ROOT = '/storage/leon/dual_task/data/behavior'

OUT = 'figures/overlaps/behavior'
for sub in ('png', 'svg', 'assets'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


# ══════════════════════════════════════════════════════════════════════════════
# LOAD 1 — recorded laser tensor + labels (choice rows) : behaviour + overlaps depth
# ══════════════════════════════════════════════════════════════════════════════
print('loading recorded laser tensor …')
X = pkl_load(f'X_{DUM}', path=DATA_IN)
y = pkl_load(f'labels_{DUM}', path=DATA_IN)
print(f'  X {X.shape}  y {len(y)}')

options = set_options(
    mice=LASER_MICE, tasks=['Dual'], mouse=LASER_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL = options['bins_BL']
BINS_LATE = np.arange(27, 54)                                    # test-time late-delay window


def _depth_on_axis(bins_train):
    """Per-trial late-delay DPA choice-code depth read on `bins_train` (train axis),
    per-mouse BL-std normalised."""
    Xe = X[..., bins_train, :].mean(-2)[:, 1].astype(float)     # (n, 84) over test-time
    for m in LASER_MICE:
        mm = (y.mouse == m).values
        sd = Xe[mm][:, BINS_BL].std()
        if sd > 0:
            Xe[mm] /= sd
    return Xe[:, BINS_LATE].mean(1)


TRAIN_LDTEST = np.concatenate([options['bins_LD'], options['bins_TEST']])   # 45-59 (main-fig axis)
depth_all = _depth_on_axis(TRAIN_LDTEST)                       # trainLD_TEST axis — I, J/K, L
cdf_diag = np.stack([X[:, 1, t, t] for t in range(X.shape[-1])], axis=1).astype(float)  # choice DV diag(t)
del X                                                          # free ~1 GB

# ── neural DISCRIMINABILITY: per-mouse Δd′ (ON−OFF) of the code ────────────────
#   M: sample axis, odor A vs B, at LATE delay (bins_LD) — DPA memoranda sensitivity.
#      (separate tensor, run_overlaps.py --with-laser --targets sample; validated d'≈1.2).
#   N: choice axis, Go vs NoGo, at MID delay (bins_MD, the Go/NoGo cue) — GNG sensitivity.
#      (the choice DV separates Go/NoGo, peaking ≈0.56 at mid-delay).
print('loading sample-axis laser tensor …')
DUM_S = 'log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_sample'
Xs = pkl_load(f'X_{DUM_S}', path=DATA_IN)
ys = pkl_load(f'labels_{DUM_S}', path=DATA_IN)
sdf_diag = np.stack([Xs[:, 1, t, t] for t in range(Xs.shape[-1])], axis=1).astype(float)      # diagonal(t)
del Xs                                                         # free ~1 GB


def _dprime(v, mask, pos, neg):
    """Neural d' = (μ_pos − μ_neg)/σ_pooled of the decision function under `mask`."""
    a = v[mask & pos]; b = v[mask & neg]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


# M — sample A vs B at late-delay (sample tensor `ys`); N — choice-axis Go vs NoGo at mid-delay.
# Points = 5 Jaws × {Naive, Expert} (10 per panel); stat = LMM d′ ~ laser + stage + (1|mouse)
# (mouse random effect handles the repeated measures). NB the trial-level signal×laser interaction
# looks p<.001 but is pseudoreplication — it vanishes under a random slope, so we DON'T use it.
STAGES = ['Naive', 'Expert']
_sA, _sB = (ys['sample'].values == 1), (ys['sample'].values == 0)
_sdpa = (ys.tasks == 'DPA').values
sLD = sdf_diag[:, options['bins_LD']].mean(1)
_gGo, _gNo = (y.tasks == 'DualGo').values, (y.tasks == 'DualNoGo').values
cMD = cdf_diag[:, options['bins_MD']].mean(1)


def _build_dpr(v, base, mo_arr, la_arr, st_arr, pos, neg):
    rows = []
    for st in STAGES:
        for m in JAWS:
            cell = base & (mo_arr == m) & (st_arr == st)
            rows.append(dict(mouse=m, stage=st,
                             off=_dprime(v, cell & (la_arr == 0), pos, neg),
                             on=_dprime(v, cell & (la_arr == 1), pos, neg)))
    return pd.DataFrame(rows)


def _lmm_laser(dfw):
    long = pd.concat([dfw.assign(d=dfw.off, laser=0), dfw.assign(d=dfw.on, laser=1)]).dropna(subset=['d'])
    r = smf.mixedlm('d ~ C(laser) + C(stage)', long, groups=long['mouse']).fit(reml=False)
    lc = [k for k in r.params.index if 'laser' in k][0]
    return float(r.params[lc]), float(r.pvalues[lc])


DPR = {'sample': _build_dpr(sLD, _sdpa, ys.mouse.values, ys.laser.values, ys.stage.values, _sA, _sB),
       'gng': _build_dpr(cMD, (_gGo | _gNo), y.mouse.values, y.laser.values, y.stage.values, _gGo, _gNo)}
LMM_DPR = {k: _lmm_laser(v) for k, v in DPR.items()}
for k in ('sample', 'gng'):
    print(f'{k} d′ (10 pts): LMM laser β={LMM_DPR[k][0]:+.3f} p={LMM_DPR[k][1]:.3f}')

is_choice = (y.target == 'choice').values
is_dpa = (y.tasks == 'DPA').values
EXP = (y.stage == 'Expert').values


# ── behaviour view of the same labels (all days, laser 0/1) ───────────────────
d = y[y.target == 'choice'].copy()
d = d[d.mouse.isin(LASER_MICE)]
DAYS_REC = sorted(int(x) for x in d.day.unique())


def per_mouse_day_laser(col, mask):
    """Per-(mouse, day, laser) accuracy proportion over the `mask` trial subset."""
    m = mask.values
    df = pd.DataFrame({'v': d.loc[m, col].values, 'mouse': d.loc[m, 'mouse'].values,
                       'day': d.loc[m, 'day'].values, 'laser': d.loc[m, 'laser'].values}).dropna()
    return df.groupby(['mouse', 'day', 'laser'], observed=True).v.mean().reset_index(name='perf')


IS_DPA_R = d.tasks == 'DPA'
IS_DUAL_R = d.tasks.isin(['DualGo', 'DualNoGo'])
UNP_R = d.pair == 0
REC_METRICS = [('DPA', 'performance', IS_DPA_R),
               ('GNG', 'odr_perf', IS_DUAL_R),
               ('DPA unp.', 'performance', IS_DPA_R & UNP_R)]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD 2 — training-batch ACC-Prl .mat  (control vs opto, between-group)
# ══════════════════════════════════════════════════════════════════════════════
def load_session(path, mouse, day):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    Tr = np.atleast_2d(m['Trials'])
    out = Tr[:, 2].astype(int)
    n = len(out)
    perf = np.isin(out, [1, 4]).astype(float)
    pair = np.isin(out, [1, 2]).astype(int)
    has_gng = 'Trials1' in m and np.size(m['Trials1']) > 0
    if has_gng and 'SampleP' in m and np.size(m['SampleP']) > 0:
        S = np.atleast_2d(m['Sample'])[:, 0].astype('int64')
        SP = np.atleast_2d(m['SampleP'])[:, 0].astype('int64')
        isP = np.isin(S, SP)
    else:
        isP = np.ones(n, bool) if not has_gng else np.zeros(n, bool)
    tasks = np.where(isP, 'DPA', '').astype(object)
    odr = np.full(n, np.nan)
    if has_gng:
        Tr1 = np.atleast_2d(m['Trials1'])
        gout = Tr1[:, 2].astype(int)
        dual_idx = np.where(~isP)[0]
        k = min(len(dual_idx), len(gout))
        dual_idx, gout = dual_idx[:k], gout[:k]
        tasks[dual_idx] = np.where(np.isin(gout, [1, 2]), 'DualGo', 'DualNoGo')
        odr[dual_idx] = np.isin(gout, [1, 4]).astype(float)
    return pd.DataFrame({'mouse': mouse, 'day': day, 'tasks': tasks,
                         'performance': perf, 'odr_perf': odr, 'pair': pair})


def load_batch(batch, group):
    folders = sorted(glob.glob(f'{DATA_ROOT}/{batch}/{group}_mouse_*'),
                     key=lambda p: int(p.rsplit('_', 1)[1]))
    rows = []
    for fol in folders:
        mouse = os.path.basename(fol)
        for f in glob.glob(f'{fol}/session_*.mat'):
            day = int(os.path.basename(f).split('_')[1].split('.')[0]) + 1
            try:
                rows.append(load_session(f, mouse, day))
            except Exception as e:
                print(f'  !! {mouse} {os.path.basename(f)}: {e}')
    return pd.concat(rows, ignore_index=True)


print('loading ACC-Prl batch …')
b_ctrl, b_opto = load_batch(BATCH, 'control'), load_batch(BATCH, 'opto')
DAYS_B = list(range(1, int(max(b_ctrl.day.max(), b_opto.day.max())) + 1))
XT_B = DAYS_B if len(DAYS_B) <= 10 else list(range(2, len(DAYS_B) + 1, 2))
print(f'  control n={b_ctrl.mouse.nunique()}  opto n={b_opto.mouse.nunique()}  days {DAYS_B[-1]}')

BATCH_METRICS = [('DPA', 'performance', lambda df: df.tasks == 'DPA'),
                 ('GNG', 'odr_perf', lambda df: df.tasks.isin(['DualGo', 'DualNoGo'])),
                 ('DPA unp.', 'performance', lambda df: (df.tasks == 'DPA') & (df.pair == 0))]


def batch_pmd(df, col, mask_fn):
    return (df[mask_fn(df)].groupby(['mouse', 'day'])[col].mean()
            .reset_index().rename(columns={col: 'perf'}))


def batch_lmm(col, mask_fn):
    """Between-group opto−control: perf ~ C(grp)*dayc + (1|mouse). Returns
    (grp β, lo, hi, p, grp×day β, lo, hi, p)."""
    p1 = batch_pmd(b_ctrl, col, mask_fn).assign(grp='control')
    p2 = batch_pmd(b_opto, col, mask_fn).assign(grp='opto')
    g = pd.concat([p1, p2], ignore_index=True)
    g['dayc'] = g['day'] - g['day'].mean()
    try:
        res = smf.mixedlm("perf ~ C(grp, Treatment('control'))*dayc", g, groups=g['mouse']).fit()
        ci = res.conf_int()
        gn = [i for i in res.params.index if i.startswith('C(grp') and ':' not in i][0]
        it = [i for i in res.params.index if i.startswith('C(grp') and ':' in i][0]
        return (float(res.params[gn]), float(ci.loc[gn, 0]), float(ci.loc[gn, 1]), float(res.pvalues[gn]),
                float(res.params[it]), float(ci.loc[it, 0]), float(ci.loc[it, 1]), float(res.pvalues[it]))
    except Exception:
        return (np.nan,) * 8


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAPS — per-mouse Δ(on−off) depth & accuracy  (Expert only)
# ══════════════════════════════════════════════════════════════════════════════
def _pooled_depth(mmask, laser_val):
    """Equal-weight A&B pooled late-delay DPA choice-code depth."""
    vals = []
    for pairs in ([0, 1], [2, 3]):
        m = (mmask & is_choice & is_dpa & EXP & (y.laser == laser_val).values
             & y.odor_pair.isin(pairs).values)
        if m.sum():
            vals.append(depth_all[m].mean())
    return float(np.mean(vals)) if vals else np.nan


def _perf_mean(col, task_mask, laser_val):
    m = (y.target == 'choice') & task_mask & EXP & (y.laser == laser_val)
    v = y.loc[m.values, col].dropna()
    return v.mean() if len(v) else np.nan


rows = []
for mouse in LASER_MICE:
    mmask = (y.mouse == mouse).values
    rows.append(dict(
        mouse=mouse, group=GROUP[mouse],
        d_depth=_pooled_depth(mmask, 1) - _pooled_depth(mmask, 0),
        d_dpa=(_perf_mean('performance', (y.tasks == 'DPA') & (y.mouse == mouse), 1)
               - _perf_mean('performance', (y.tasks == 'DPA') & (y.mouse == mouse), 0)),
        d_gng=(_perf_mean('odr_perf', (y.tasks != 'DPA') & (y.mouse == mouse), 1)
               - _perf_mean('odr_perf', (y.tasks != 'DPA') & (y.mouse == mouse), 0)),
    ))
xdepth = np.array([r['d_depth'] for r in rows])
print('\nOverlaps per-mouse Δ(on−off) [Expert]:')
for r in rows:
    print(f'  {r["mouse"]:9s} {r["group"]:5s} Δdepth={r["d_depth"]:+.3f} '
          f'ΔDPA={r["d_dpa"]:+.3f} ΔGNG={r["d_gng"]:+.3f}')


# ── A&B-independent Δ(on−off) for panels I/J — Jaws only, each mouse → 2 points ──
#   odor A = odor_pairs [0,1], odor B = [2,3]; depth/accuracy read per sample class
#   (no A&B averaging), so the 5 Jaws mice give 10 independent points.
SAMPLE_CLASSES = [(0, [0, 1]), (1, [2, 3])]


def _depth_sample(depth, mmask, laser_val, pairs, st):
    m = (mmask & is_choice & is_dpa & st.values & (y.laser == laser_val).values
         & y.odor_pair.isin(pairs).values)
    return float(depth[m].mean()) if m.sum() else np.nan


def _perf_mean_sample(col, task_mask, laser_val, pairs, st):
    m = ((y.target == 'choice') & task_mask & st & (y.laser == laser_val)
         & y.odor_pair.isin(pairs))
    v = y.loc[m.values, col].dropna()
    return v.mean() if len(v) else np.nan


rows_ab = []
for st_name in ['Naive', 'Expert']:                     # Naive + Expert as independent points
    st = (y.stage == st_name)
    for mouse in JAWS:                                  # Jaws inhibition only (n=5)
        mmask = (y.mouse == mouse).values
        tm = (y.mouse == mouse)
        for cls, pairs in SAMPLE_CLASSES:               # odor A [0,1] / B [2,3]
            rows_ab.append(dict(
                mouse=mouse, cls=cls, stage=st_name,
                d_depth=_depth_sample(depth_all, mmask, 1, pairs, st) - _depth_sample(depth_all, mmask, 0, pairs, st),
                d_dpa=(_perf_mean_sample('performance', tm & (y.tasks == 'DPA'), 1, pairs, st)
                       - _perf_mean_sample('performance', tm & (y.tasks == 'DPA'), 0, pairs, st)),
                d_gng=(_perf_mean_sample('odr_perf', tm & (y.tasks != 'DPA'), 1, pairs, st)
                       - _perf_mean_sample('odr_perf', tm & (y.tasks != 'DPA'), 0, pairs, st)),
            ))
print(f'\nOverlaps A&B-independent Δ(on−off) [Jaws, {len(rows_ab)} pts]:')
for r in rows_ab:
    print(f'  {r["mouse"]:9s} {r["stage"][:3]} {"A" if r["cls"] == 0 else "B"} '
          f'Δdepth={r["d_depth"]:+.3f} ΔDPA={r["d_dpa"]:+.3f} ΔGNG={r["d_gng"]:+.3f}')


def regression_band(ax, xs, ys, color='0.25'):
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return
    xv, yv = xs[ok], ys[ok]
    slope, icpt, _, _, se = linregress(xv, yv)
    xl = np.linspace(xv.min(), xv.max(), 100)
    yl = slope * xl + icpt
    ssx = np.sum((xv - xv.mean()) ** 2)
    seb = se * np.sqrt(1 / len(xv) + (xl - xv.mean()) ** 2 / ssx)
    tc = t_dist.ppf(0.975, df=len(xv) - 2)
    ax.plot(xl, yl, color=color, lw=1.5, zorder=4)
    ax.fill_between(xl, yl - tc * seb, yl + tc * seb, color=color, alpha=0.15, zorder=2)


# Row-4 mechanism data is computed above: panel L uses `rows_ab` (the trade-off contrast),
# panels M/N use `DPR`/`LMM_DPR` (per mouse×stage d′). The former trial-level GEE
# readout-vs-silencing analysis (and an earlier behavioural d′/criterion SDT control) were
# removed when panel L became the trade-off contrast — see docs/behavior.md §6.


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 19.0))
outer = fig.add_gridspec(2, 1, height_ratios=[2.1, 4.6], hspace=0.05,
                         left=0.055, right=0.985, top=0.99, bottom=0.035)
gs_body = outer[1].subgridspec(4, 12, height_ratios=[1.0, 1.0, 1.22, 1.0], hspace=0.42, wspace=0.62)


def panel_letter(ax, L, dx=0.020, dy=0.016):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, L, fontsize=15, fontweight='bold', va='top', ha='left')


def show_scheme(ax, path, aspect='equal'):
    im = mpimg.imread(path)
    g = im[..., :3].mean(-1); mk = g < 0.985
    r = np.where(mk.any(1))[0]; c = np.where(mk.any(0))[0]
    ax.imshow(im[r.min():r.max() + 1, c.min():c.max() + 1], aspect=aspect); ax.axis('off')


# ── A: opto scheme banner ─────────────────────────────────────────────────────
SCHEME = '../opto.png'                        # recorded-cohort design (self-labelled a/b)
axA = fig.add_subplot(outer[0])
show_scheme(axA, SCHEME)                      # aspect='equal' — no distortion

# ── B, C: recorded OFF vs ON learning curves (Jaws inhibition) — ROW 2 ─────────
axB = fig.add_subplot(gs_body[1, 0:4])
axC = fig.add_subplot(gs_body[1, 4:8])
for ax, (short, col, mask), msg in [
        (axB, REC_METRICS[0], 'Delay silencing spares DPA'),
        (axC, REC_METRICS[1], '…and spares GNG')]:
    g = per_mouse_day_laser(col, mask)
    g = g[g.mouse.isin(JAWS)]
    lo, hi = [], []
    for lasval, color, lab in [(0, OFF_C, 'laser OFF'), (1, ON_C, 'laser ON')]:
        sub = g[g.laser == lasval]
        m, s = [], []
        for day in DAYS_REC:
            v = sub.loc[sub.day == day, 'perf'].dropna().values
            m.append(v.mean() if len(v) else np.nan)
            s.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else (0.0 if len(v) else np.nan))
        m, s, x = np.array(m), np.array(s), np.array(DAYS_REC, float)
        ok = ~np.isnan(m)
        ax.plot(x[ok], m[ok], '-o', color=color, lw=2, ms=4, label=lab, zorder=3)
        ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)
        if ok.any():
            lo.append(np.nanmin((m - s)[ok])); hi.append(np.nanmax((m + s)[ok]))
    ylo = max(0.0, min(lo) - 0.05) if lo else 0.3
    yhi = min(1.06, max(hi) + 0.06) if hi else 1.05
    # per-day within-mouse ON−OFF stars
    piv = g.pivot_table(index=['mouse', 'day'], columns='laser', values='perf').dropna()
    if {0, 1}.issubset(piv.columns):
        delta = (piv[1] - piv[0]).reset_index(name='dd')
        for day in DAYS_REC:
            dv = delta.loc[delta.day == day, 'dd'].values
            if len(dv) >= N_MIN and not np.allclose(dv, dv[0]):
                pv = float(ttest_1samp(dv, 0.0).pvalue)
                if star(pv):
                    ax.text(day, yhi - 0.02 * (yhi - ylo), star(pv), ha='center', va='top',
                            fontsize=10, fontweight='bold')
    ax.set_ylim(ylo, yhi)
    if ylo < 0.5 < yhi:
        ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_xticks(DAYS_REC); ax.set_xlabel('session')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title(msg, loc='left', fontweight='bold', fontsize=TITLE_FS)
axB.set_ylabel('performance')

# ── H: per-mouse laser effect on the choice code (OFF vs ON depth, Jaws) ──────
#   Absolute A&B-pooled DPA choice-code depth per mouse under laser OFF vs ON — shows
#   the laser reliably moves each animal's code (the shift that is the scatters' x-axis).
#   (Sits in the recorded within-mouse row; replaced the old LMM laser forest.)
axK = fig.add_subplot(gs_body[1, 8:12])
_offon = {m: (_pooled_depth((y.mouse == m).values, 0),
              _pooled_depth((y.mouse == m).values, 1)) for m in JAWS}
_offs = np.array([_offon[m][0] for m in JAWS]); _ons = np.array([_offon[m][1] for m in JAWS])
for m in JAWS:
    axK.plot([0, 1], _offon[m], '-o', color=MOUSE_COLOR[m], lw=1.3, ms=6,
             mec='w', mew=0.6, label=m, zorder=3)
for xx, vv in [(-0.22, _offs), (1.22, _ons)]:                # group mean ± SEM
    mn = np.nanmean(vv); se = np.nanstd(vv, ddof=1) / np.sqrt(np.isfinite(vv).sum())
    axK.errorbar(xx, mn, yerr=se, fmt='s', color='k', ms=7, capsize=4, lw=1.5, zorder=5)
axK.axhline(0, ls=':', color='0.5', lw=1)
axK.set_xticks([0, 1]); axK.set_xticklabels(['laser\nOFF', 'laser\nON'])
axK.set_xlim(-0.5, 1.5)
axK.set_ylabel('DPA choice-code depth\n(late delay, trainLD_TEST)')
axK.set_title('Laser moves the code per mouse', loc='left', fontweight='bold', fontsize=TITLE_FS)
axK.legend(frameon=True, framealpha=0.85, edgecolor='0.85', fontsize=6.5, loc='center left',
           ncol=1, handletextpad=0.3)

# ── J, K: overlaps causal coupling — Δdepth vs Δaccuracy (square) ─────────────
#   Jaws only, A&B taken as INDEPENDENT points (each mouse → odor-A solid + odor-B open,
#   joined by a thin line); stats over all 10 points.
axE = fig.add_subplot(gs_body[2, 4:8])
axF = fig.add_subplot(gs_body[2, 8:12])
ally = np.array([r[k] for k in ('d_dpa', 'd_gng') for r in rows_ab], float)
ally = ally[~np.isnan(ally)]
pad = (ally.max() - ally.min()) * 0.15 or 0.05
ylim = (ally.min() - pad, ally.max() + pad)
_STMK = {'Expert': 'o', 'Naive': '^'}                       # Expert circle / Naive triangle
for ax, key, ylab, msg in [
        (axE, 'd_dpa', 'Δ DPA accuracy (on−off)', 'Depth vs ΔDPA (positive trend)'),
        (axF, 'd_gng', 'Δ GNG accuracy (on−off)', 'Depth change predicts ΔGNG')]:
    xdep = np.array([r['d_depth'] for r in rows_ab])
    yv = np.array([r[key] for r in rows_ab])
    for mouse in JAWS:                                   # join A&B within each mouse×stage
        for stg in ('Naive', 'Expert'):
            idx = [i for i, r in enumerate(rows_ab) if r['mouse'] == mouse and r['stage'] == stg]
            ax.plot(xdep[idx], yv[idx], '-', color=MOUSE_COLOR[mouse], lw=0.6, alpha=0.35, zorder=3)
    for i, r in enumerate(rows_ab):
        face = MOUSE_COLOR[r['mouse']] if r['cls'] == 0 else 'w'    # A solid / B open
        ax.scatter(xdep[i], yv[i], facecolors=face, edgecolors=MOUSE_COLOR[r['mouse']],
                   marker=_STMK[r['stage']], s=80, linewidths=1.1, zorder=5)
    regression_band(ax, xdep, yv, color='0.25')
    ax.axhline(0, ls=':', color='k', lw=0.8); ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_ylim(ylim)
    ok = ~(np.isnan(xdep) | np.isnan(yv))
    r_p, p_p = pearsonr(xdep[ok], yv[ok]); rho, ps = spearmanr(xdep[ok], yv[ok])
    ax.text(0.5, 0.02, f'n={ok.sum()}: r={r_p:+.2f} p={p_p:.3f}  ρ={rho:+.2f} p={ps:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5, color='0.3')
    ax.text(0.85, 0.93, '*' if p_p < 0.05 else 'n.s.', transform=ax.transAxes, ha='center',
            va='top', fontsize=20, fontweight='bold', color='k' if p_p < 0.05 else '0.55')
    ax.set_xlabel('Δ DPA choice-code depth (on−off, trainLD_TEST)'); ax.set_ylabel(ylab)
    ax.set_title(msg, loc='left', fontweight='bold', fontsize=TITLE_FS)
    ax.set_box_aspect(1)                                  # square panels
_leg_h = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=7, label='odor A'),
          mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=7, label='odor B'),
          mlines.Line2D([0], [0], marker='o', color='k', mfc='0.5', ls='none', ms=7, label='Expert'),
          mlines.Line2D([0], [0], marker='^', color='k', mfc='0.5', ls='none', ms=7, label='Naive')]
axE.legend(handles=_leg_h, frameon=False, fontsize=6.5, loc='upper left', handletextpad=0.3, ncol=2, columnspacing=0.8)

# ── G, H, I: batch ACC-Prl control vs opto learning curves — ROW 1 ────────────
axG = fig.add_subplot(gs_body[0, 0:3])
axH = fig.add_subplot(gs_body[0, 3:6])
axI = fig.add_subplot(gs_body[0, 6:9])
for ax, (short, col, mask_fn), msg in [
        (axG, BATCH_METRICS[0], 'Chronic silencing impairs DPA'),
        (axH, BATCH_METRICS[1], 'GNG is spared'),
        (axI, BATCH_METRICS[2], 'Deficit carried by unpaired')]:
    p1, p2 = batch_pmd(b_ctrl, col, mask_fn), batch_pmd(b_opto, col, mask_fn)
    lo, hi = [], []
    for p, color, lab in [(p1, OFF_C, f'control (n={p1.mouse.nunique()})'),
                          (p2, ON_C, f'opto (n={p2.mouse.nunique()})')]:
        m, s = [], []
        for day in DAYS_B:
            v = p.loc[p.day == day, 'perf'].dropna().values
            m.append(v.mean() if len(v) else np.nan)
            s.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else (0.0 if len(v) else np.nan))
        m, s, x = np.array(m), np.array(s), np.array(DAYS_B, float)
        ok = ~np.isnan(m)
        ax.plot(x[ok], m[ok], '-o', color=color, lw=2, ms=4, label=lab, zorder=3)
        ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)
        if ok.any():
            lo.append(np.nanmin((m - s)[ok])); hi.append(np.nanmax((m + s)[ok]))
    ylo = max(0.0, min(lo) - 0.05) if lo else 0.3
    yhi = min(1.06, max(hi) + 0.06) if hi else 1.05
    for day in DAYS_B:
        a = p1.loc[p1.day == day, 'perf'].dropna().values
        b = p2.loc[p2.day == day, 'perf'].dropna().values
        if len(a) >= 4 and len(b) >= 4 and not (np.all(a == a[0]) and np.all(b == b[0])):
            pv = float(ttest_ind(b, a, equal_var=False).pvalue)
            if star(pv):
                ax.text(day, yhi - 0.02 * (yhi - ylo), star(pv), ha='center', va='top',
                        fontsize=9, fontweight='bold')
    ax.set_ylim(ylo, yhi)
    if ylo < 0.5 < yhi:
        ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_xticks(XT_B); ax.set_xlabel('training day')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title(msg, loc='left', fontweight='bold', fontsize=TITLE_FS)
axG.set_ylabel('performance')

# ── J: batch LMM group-effect forest (opto−control β ○ + group×day slope □) ────
axJ = fig.add_subplot(gs_body[0, 9:12])
for i, (short, col, mask_fn) in enumerate(BATCH_METRICS):
    gb, glo, ghi, gp, ib, ilo, ihi, ip = batch_lmm(col, mask_fn)
    for dx, val, vlo, vhi, pv, mk, fill in [(-0.14, gb, glo, ghi, gp, 'o', True),
                                            (0.14, ib, ilo, ihi, ip, 's', False)]:
        if not np.isfinite(val):
            continue
        cc = 'k' if (np.isfinite(pv) and pv < 0.05) else '0.6'
        axJ.errorbar(i + dx, val, yerr=[[val - vlo], [vhi - val]], fmt=mk,
                     color=cc, mfc=(cc if fill else 'white'), ms=7 if fill else 6,
                     capsize=3, lw=1.5, zorder=3)
        if np.isfinite(pv) and star(pv):
            axJ.text(i + dx, vhi + 0.004, star(pv), ha='center', va='bottom',
                     fontsize=9, fontweight='bold')
axJ.axhline(0, ls='--', color='0.4', lw=1)
axJ.set_xticks(range(len(BATCH_METRICS)))
axJ.set_xticklabels([m[0] for m in BATCH_METRICS], rotation=15, ha='right')
axJ.set_xlim(-0.6, len(BATCH_METRICS) - 0.4)
axJ.set_ylabel('opto−control  (Δ perf.)')
axJ.set_title('Silencing effect (LMM)', loc='left', fontweight='bold', fontsize=TITLE_FS)
axJ.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=7, label='group (at mean day)'),
                    mlines.Line2D([0], [0], marker='s', color='k', mfc='white', ls='none', ms=6, label='group×day (slope)')],
           frameon=False, fontsize=7.5, loc='best')

# ── I: depth → DPA/GNG TRADE-OFF contrast (headline coupling stat) — with J,K ──
#   Trade-off hypothesis (depth↑ → DPA↑ AND GNG↓) makes one joint prediction: depth
#   positively predicts (ΔDPA − ΔGNG). On the pre-committed trainLD_TEST axis this pools
#   both arms (J/K, same row) and is significant with no window search: r=+0.48 p=0.034.
axL = fig.add_subplot(gs_body[2, 0:4])
_xdep = np.array([r['d_depth'] for r in rows_ab])
_ytr = np.array([r['d_dpa'] - r['d_gng'] for r in rows_ab])          # trade-off contrast
for mouse in JAWS:
    for stg in ('Naive', 'Expert'):
        idx = [i for i, r in enumerate(rows_ab) if r['mouse'] == mouse and r['stage'] == stg]
        axL.plot(_xdep[idx], _ytr[idx], '-', color=MOUSE_COLOR[mouse], lw=0.6, alpha=0.35, zorder=3)
for i, r in enumerate(rows_ab):
    face = MOUSE_COLOR[r['mouse']] if r['cls'] == 0 else 'w'         # A solid / B open
    axL.scatter(_xdep[i], _ytr[i], facecolors=face, edgecolors=MOUSE_COLOR[r['mouse']],
                marker=_STMK[r['stage']], s=70, linewidths=1.0, zorder=5)
regression_band(axL, _xdep, _ytr, color='0.25')
axL.axhline(0, ls=':', color='k', lw=0.8); axL.axvline(0, ls=':', color='k', lw=0.8)
_ok = ~(np.isnan(_xdep) | np.isnan(_ytr))
_rp, _pp = pearsonr(_xdep[_ok], _ytr[_ok]); _rs, _ps = spearmanr(_xdep[_ok], _ytr[_ok])
axL.text(0.5, 0.02, f'n={_ok.sum()}: r={_rp:+.2f} p={_pp:.3f}  ρ={_rs:+.2f} p={_ps:.3f}',
         transform=axL.transAxes, ha='center', va='bottom', fontsize=6.2, color='0.3')
axL.text(0.85, 0.93, '*' if _pp < 0.05 else 'n.s.', transform=axL.transAxes, ha='center',
         va='top', fontsize=20, fontweight='bold', color='k' if _pp < 0.05 else '0.55')
axL.set_xlabel('Δ choice-code depth (on−off, trainLD_TEST)')
axL.set_ylabel('Δ DPA − Δ GNG accuracy (on−off)')
axL.set_title('Depth drives a DPA↑/GNG↓ trade-off', loc='left', fontweight='bold', fontsize=TITLE_FS)
axL.set_box_aspect(1)

# ── M, N: neural d′ laser ON vs OFF (per mouse) — points on unity = spared ──────
axM = fig.add_subplot(gs_body[3, 4:8])
axN = fig.add_subplot(gs_body[3, 8:12])


def _dprime_scatter(ax, dfw, lmm, title):
    vals = np.concatenate([dfw.off.values, dfw.on.values]); vals = vals[np.isfinite(vals)]
    lo = min(0.0, vals.min()) - 0.1; hi = vals.max() + 0.15
    ax.plot([lo, hi], [lo, hi], '--', color='0.5', lw=1, zorder=1)          # unity = spared
    for _, r in dfw.iterrows():
        fc = MOUSE_COLOR[r.mouse] if r.stage == 'Expert' else 'w'           # Expert filled / Naive open
        ax.scatter(r.off, r.on, facecolors=fc, edgecolors=MOUSE_COLOR[r.mouse], s=105, lw=1.3, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
    ax.set_xlabel("d′  laser OFF"); ax.set_ylabel("d′  laser ON")
    ax.set_title(title, loc='left', fontweight='bold', fontsize=TITLE_FS)
    ax.text(0.5, 0.02, f'LMM laser p={lmm[1]:.2f}  (n=10, +1|mouse)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=7.5, color='0.3')


_dprime_scatter(axM, DPR['sample'], LMM_DPR['sample'], 'DPA memory code (A vs B, late delay)')
_dprime_scatter(axN, DPR['gng'], LMM_DPR['gng'], 'GNG code (Go vs NoGo, mid-delay)')
axN.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=7, label='Expert'),
                    mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=7, label='Naive')],
           frameon=False, fontsize=7, loc='upper left', handletextpad=0.3)

# ── L: DPA vs GNG performance in laser-ON trials (Naive ○ + Expert ●, 10 pts) ──
#   Balance plane of the non-opto main figure, restricted to laser-ON trials: per mouse ×
#   stage, where does the ON-trial behaviour sit relative to the DPA=GNG diagonal / optimum?
axBal = fig.add_subplot(gs_body[3, 0:4])
_dON = y[(y.target == 'choice') & y.mouse.isin(JAWS) & (y.laser == 1)]


def _pf_on(mo, st, is_dpa):
    s = _dON[(_dON.mouse == mo) & (_dON.stage == st)]
    v = s[s.tasks == 'DPA']['performance'] if is_dpa else s[s.tasks != 'DPA']['odr_perf']
    v = v.dropna()
    return v.mean() if len(v) else np.nan


_bpts = [(m, st, _pf_on(m, st, True), _pf_on(m, st, False)) for st in ('Naive', 'Expert') for m in JAWS]
_bx = np.array([p[2] for p in _bpts]); _by = np.array([p[3] for p in _bpts])
_bok = np.isfinite(_bx) & np.isfinite(_by)
_blim = (max(0.3, np.concatenate([_bx[_bok], _by[_bok]]).min() - 0.05), 1.0)
axBal.plot(_blim, _blim, ls='--', color='0.7', lw=0.9, zorder=1)
axBal.scatter(0.99, 0.99, marker='*', s=200, color='#E8A100', edgecolor='k', linewidths=0.6, zorder=6)
axBal.text(0.985, 0.955, 'optimal', ha='right', va='top', fontsize=8, color='#7a5600', transform=axBal.transAxes)
for m in JAWS:                                            # join each mouse's Naive→Expert
    xs = [p[2] for p in _bpts if p[0] == m]; ys = [p[3] for p in _bpts if p[0] == m]
    axBal.plot(xs, ys, '-', color=MOUSE_COLOR[m], lw=0.6, alpha=0.35, zorder=3)
for m, st, xd, yd in _bpts:
    fc = MOUSE_COLOR[m] if st == 'Expert' else 'w'       # Expert filled / Naive open
    axBal.scatter(xd, yd, marker='o', s=90, facecolors=fc, edgecolors=MOUSE_COLOR[m], lw=1.2, zorder=5)
_brp, _bpp = pearsonr(_bx[_bok], _by[_bok]); _brs, _bps = spearmanr(_bx[_bok], _by[_bok])
axBal.text(0.5, 0.02, f'ON, n={_bok.sum()}: r={_brp:+.2f} p={_bpp:.3f}  ρ={_brs:+.2f} p={_bps:.3f}',
           transform=axBal.transAxes, ha='center', va='bottom', fontsize=6.2, color='0.3')
axBal.set_xlim(_blim); axBal.set_ylim(_blim); axBal.set_box_aspect(1)
axBal.set_xlabel('DPA performance (laser ON)'); axBal.set_ylabel('GNG performance (laser ON)')
axBal.set_title('DPA–GNG balance, laser ON', loc='left', fontweight='bold', fontsize=TITLE_FS)
axBal.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=7, label='Expert'),
                      mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=7, label='Naive')],
             frameon=False, fontsize=7, loc='lower left', handletextpad=0.3)


# ── panel letters + row banners ───────────────────────────────────────────────
# reading order: A scheme · B–E batch · F–H recorded(+depth) · I–K overlaps · L–N last row
for _ax, _L in [(axA, 'A'), (axG, 'B'), (axH, 'C'), (axI, 'D'), (axJ, 'E'),
                (axB, 'F'), (axC, 'G'), (axK, 'H'), (axL, 'I'), (axE, 'J'), (axF, 'K'),
                (axBal, 'L'), (axM, 'M'), (axN, 'N')]:
    panel_letter(_ax, _L)


def row_banner(ax_left, text, dy=0.014):
    p = ax_left.get_position()
    fig.text(0.055, p.y1 + dy, text, fontsize=9.5, fontweight='bold',
             va='bottom', ha='left', color='0.35')


row_banner(axG, 'Training batch · chronic every-trial silencing · BETWEEN-group opto vs control (ACC-Prl, 9 v 9)')
row_banner(axB, 'Recorded cohort · transient delay-only laser · WITHIN-mouse ON vs OFF (n=5 Jaws inhibition)')
row_banner(axL, 'overlaps · laser ON−OFF: depth drives a DPA↑/GNG↓ trade-off (I) with its two arms — ΔDPA (J) & ΔGNG (K); 5 Jaws · Naive▲+Expert● × A&B')
row_banner(axBal, 'Laser-ON DPA–GNG balance (L) · code discriminability d′ ON≈OFF (on unity) — DPA memory (M) & GNG (N)')

fig.text(0.5, 0.004,
         'ACC→Prl(mPFC) projection.  B–E training batch, between-group (every-trial silencing), mean ± SEM; '
         'LMM perf ~ group×day + (1|mouse); per-day stars Welch, uncorrected.  '
         'F–G recorded cohort within-mouse (interleaved laser), Jaws inhibition n=5; LMM perf ~ laser×day + (1|mouse); '
         'per-day stars = one-sample ΔON−OFF.  H per-mouse OFF-vs-ON choice-code depth (Jaws, A&B pooled). '
         'I depth vs the trade-off contrast ΔDPA−ΔGNG (joint test of the DPA↑/GNG↓ trade-off, 20 pts, '
         'Pearson r=+0.48 p=0.034 on the pre-committed trainLD_TEST axis — no window search); J/K are its two arms. '
         'J–K overlaps Δ(on−off), depth = DPA choice-code on trainLD_TEST (45-59), readout 27-53; 5 Jaws × '
         '{Naive ▲, Expert ●} × A&B = 20 pts; star = Pearson (Spearman agrees). Between-animal coupling.  '
         'L DPA vs GNG performance in laser-ON trials (balance plane of the non-opto main figure), 5 Jaws × '
         '{Naive ○, Expert ●} = 10 pts, optimal corner starred (r=+0.44 p=0.20, descriptive).  '
         'M,N code discriminability d′ laser ON vs OFF, 5 Jaws × {Naive ○, Expert ●} = 10 pts (points on unity = spared): '
         'M = sample axis odor A vs B at late delay (bins_LD 45-53, DPA memoranda; --targets sample tensor); '
         'N = choice axis Go vs NoGo at mid-delay (bins_MD 33-38, GNG cue). Dashed = unity; stat = LMM '
         'd′ ~ laser + stage + (1|mouse) [trial-level signal×laser interaction is pseudoreplication, ns under random slope].  '
         '* p<0.05  ** p<0.01  *** p<0.001',
         ha='center', va='bottom', fontsize=7.3, color='0.45')

for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_opto_main.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
