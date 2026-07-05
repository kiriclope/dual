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
  H  Within-mouse LMM laser effect (ON−OFF) per metric: Jaws β±95%CI (○).
     perf ~ laser*day + (1|mouse).  (Jaws inhibition only, n=5.)
  ── Same projection · overlaps causal coupling (laser ON−OFF) ──
  I  Per-mouse choice-code depth, laser OFF vs ON (Jaws, A&B pooled) — the manipulation
     moves each animal's code (its shift is the x-axis of J/K); group mean flat.
  J  Δ DPA choice-code depth (on−off)  vs  Δ DPA accuracy   (5 Jaws, Expert; A&B indep., 10 pts)
  K  Δ DPA choice-code depth (on−off)  vs  Δ GNG accuracy   (specificity / the coupled one)
  Depth read on the trainLD axis (bins 45-53); late-delay window (27-53); J/K square, all trials.

Message: chronic every-trial ACC→Prl silencing degrades DPA, carried by unpaired trials
(B–E); transient delay-only ACC→Prl perturbation spares GROSS behaviour (F,G,H) but
demonstrably moves the choice code, and Δdepth predicts Δaccuracy across animals (I,J).

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
import statsmodels.api as sm
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


depth_all = _depth_on_axis(options['bins_LD'])                 # trainLD axis (45-53) — both I & J
del X                                                          # free ~1 GB

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


def recorded_lmm(col, mask, mice):
    """Within-mouse laser effect: perf ~ laser*dayc + (1|mouse) over `mice`.
    Returns (laser β, lo, hi, p) or (mean Δ, nan,nan,nan) when n<3 (ChR)."""
    g = per_mouse_day_laser(col, mask)
    g = g[g.mouse.isin(mice)].copy()
    g['dayc'] = g['day'] - g['day'].mean()
    if g.mouse.nunique() >= 3:
        try:
            res = smf.mixedlm('perf ~ laser*dayc', g, groups=g['mouse']).fit()
            ci = res.conf_int()
            return (float(res.params['laser']), float(ci.loc['laser', 0]),
                    float(ci.loc['laser', 1]), float(res.pvalues['laser']))
        except Exception:
            pass
    piv = g.pivot_table(index=['mouse', 'day'], columns='laser', values='perf').dropna()
    b = float((piv[1] - piv[0]).mean()) if {0, 1}.issubset(piv.columns) else np.nan
    return (b, np.nan, np.nan, np.nan)


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


def _depth_sample(depth, mmask, laser_val, pairs):
    m = (mmask & is_choice & is_dpa & EXP & (y.laser == laser_val).values
         & y.odor_pair.isin(pairs).values)
    return float(depth[m].mean()) if m.sum() else np.nan


def _perf_mean_sample(col, task_mask, laser_val, pairs):
    m = ((y.target == 'choice') & task_mask & EXP & (y.laser == laser_val)
         & y.odor_pair.isin(pairs))
    v = y.loc[m.values, col].dropna()
    return v.mean() if len(v) else np.nan


rows_ab = []
for mouse in JAWS:                                      # Jaws inhibition only (n=5)
    mmask = (y.mouse == mouse).values
    tm = (y.mouse == mouse)
    for cls, pairs in SAMPLE_CLASSES:
        rows_ab.append(dict(
            mouse=mouse, cls=cls,
            d_depth=_depth_sample(depth_all, mmask, 1, pairs) - _depth_sample(depth_all, mmask, 0, pairs),
            d_dpa=(_perf_mean_sample('performance', tm & (y.tasks == 'DPA'), 1, pairs)
                   - _perf_mean_sample('performance', tm & (y.tasks == 'DPA'), 0, pairs)),
            d_gng=(_perf_mean_sample('odr_perf', tm & (y.tasks != 'DPA'), 1, pairs)
                   - _perf_mean_sample('odr_perf', tm & (y.tasks != 'DPA'), 0, pairs)),
        ))
print(f'\nOverlaps A&B-independent Δ(on−off) [Jaws, {len(rows_ab)} pts]:')
for r in rows_ab:
    print(f'  {r["mouse"]:9s} {"A" if r["cls"] == 0 else "B"} Δdepth={r["d_depth"]:+.3f} '
          f'ΔDPA={r["d_dpa"]:+.3f} ΔGNG={r["d_gng"]:+.3f}')


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


# ══════════════════════════════════════════════════════════════════════════════
# Recorded within-mouse laser MECHANISM (Jaws, Expert) — for row-4 panels L,M,N
# ══════════════════════════════════════════════════════════════════════════════
_dfl = pd.DataFrame({
    'mouse': y.mouse.values, 'laser': y.laser.values, 'tasks': y.tasks.values,
    'pair': y.pair.values, 'perf': y.performance.values, 'odr': y.odr_perf.values,
    'depth': depth_all, 'is_choice': (y.target == 'choice').values, 'stage': y.stage.values,
})
_dfl = _dfl[_dfl.is_choice & _dfl.mouse.isin(JAWS) & (_dfl.stage == 'Expert')].copy()
_dfl['depth_z'] = _dfl.groupby('mouse')['depth'].transform(          # per-mouse z-score
    lambda v: (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 0 else v * 0.0)


def _gee_depth(sub, ycol):
    """Trial-level cluster-robust logistic: ycol ~ depth_z + laser, GEE grouped by mouse.
    Returns depth OR (per SD), 95% CI, p, n-trials. Handles pseudoreplication."""
    d = sub.dropna(subset=[ycol, 'depth_z', 'laser'])
    try:
        g = smf.gee(f'{ycol} ~ depth_z + laser', groups=d['mouse'], data=d,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = g.conf_int()
        return (float(np.exp(g.params['depth_z'])), float(np.exp(ci.loc['depth_z', 0])),
                float(np.exp(ci.loc['depth_z', 1])), float(g.pvalues['depth_z']), len(d))
    except Exception as e:
        print('  GEE failed:', e)
        return (np.nan, np.nan, np.nan, np.nan, 0)


MM = {'DPA': _gee_depth(_dfl[_dfl.tasks == 'DPA'], 'perf'),
      'GNG': _gee_depth(_dfl[_dfl.tasks != 'DPA'], 'odr')}
print('\nTrial-level GEE  depth→accuracy (OR/SD, p, n):')
for k, v in MM.items():
    print(f'  {k}: OR={v[0]:.3f}  p={v[3]:.4f}  (n={v[4]})')


def _dc(hit, ns, fa, nn):
    """d' and criterion with loglinear (Hautus) correction."""
    HR = (hit + 0.5) / (ns + 1); FA = (fa + 0.5) / (nn + 1)
    return norm.ppf(HR) - norm.ppf(FA), -0.5 * (norm.ppf(HR) + norm.ppf(FA))


# per-(mouse, laser) d' & criterion — DPA (signal=paired) and GNG (signal=Go)
SDT = {tk: {'d': {0: [], 1: []}, 'c': {0: [], 1: []}} for tk in ('DPA', 'GNG')}
for _mo in JAWS:
    for _las in (0, 1):
        _sub = _dfl[(_dfl.mouse == _mo) & (_dfl.laser == _las)]
        _sp, _su = _sub[(_sub.tasks == 'DPA') & (_sub.pair == 1)], _sub[(_sub.tasks == 'DPA') & (_sub.pair == 0)]
        _dp, _cr = (_dc(_sp.perf.sum(), len(_sp), len(_su) - _su.perf.sum(), len(_su))
                    if len(_sp) and len(_su) else (np.nan, np.nan))
        SDT['DPA']['d'][_las].append(_dp); SDT['DPA']['c'][_las].append(_cr)
        _sg, _sn = _sub[_sub.tasks == 'DualGo'], _sub[_sub.tasks == 'DualNoGo']
        _dp, _cr = (_dc(_sg.odr.sum(), len(_sg), len(_sn) - _sn.odr.sum(), len(_sn))
                    if len(_sg) and len(_sn) else (np.nan, np.nan))
        SDT['GNG']['d'][_las].append(_dp); SDT['GNG']['c'][_las].append(_cr)
print('SDT (mean OFF→ON, paired-t p):')
for tk in ('DPA', 'GNG'):
    for met in ('d', 'c'):
        _o, _n = np.array(SDT[tk][met][0], float), np.array(SDT[tk][met][1], float)
        _dd = (_n - _o)[np.isfinite(_n - _o)]
        _pp = float(ttest_1samp(_dd, 0).pvalue) if len(_dd) > 1 else np.nan
        print(f"  {tk} {met}': {np.nanmean(_o):+.2f} → {np.nanmean(_n):+.2f}  (p={_pp:.3f})")


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

# ── H: recorded within-mouse LMM laser forest (Jaws β±CI ○) ───────────────────
axD = fig.add_subplot(gs_body[1, 8:12])
for i, (short, col, mask) in enumerate(REC_METRICS):
    jb, jlo, jhi, jp = recorded_lmm(col, mask, JAWS)
    if np.isfinite(jb):
        cc = 'k' if (np.isfinite(jp) and jp < 0.05) else '0.6'
        has_ci = np.isfinite(jlo) and np.isfinite(jhi)
        axD.errorbar(i, jb, yerr=([[jb - jlo], [jhi - jb]] if has_ci else None),
                     fmt='o', color=cc, ms=7, capsize=3, lw=1.6, zorder=3)
        if np.isfinite(jp) and star(jp):
            axD.text(i, (jhi if has_ci else jb) + 0.004, star(jp), ha='center',
                     va='bottom', fontsize=9, fontweight='bold')
axD.axhline(0, ls='--', color='0.4', lw=1)
axD.set_xticks(range(len(REC_METRICS)))
axD.set_xticklabels([m[0] for m in REC_METRICS], rotation=15, ha='right')
axD.set_xlim(-0.6, len(REC_METRICS) - 0.4)
axD.set_ylabel('ON−OFF  (Δ perf.)')
axD.set_title('No gross behavioural effect', loc='left', fontweight='bold', fontsize=TITLE_FS)
axD.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=7,
                                  label='Jaws inhibition (LMM, n=5)')],
           frameon=False, fontsize=7.5, loc='best')

# ── I: per-mouse laser effect on the choice code (OFF vs ON depth, Jaws) ──────
#   Absolute A&B-pooled DPA choice-code depth per mouse under laser OFF vs ON — shows
#   the laser reliably moves each animal's code (the shift that is the scatters' x-axis).
axK = fig.add_subplot(gs_body[2, 0:4])
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
axK.set_ylabel('DPA choice-code depth\n(late delay, trainLD)')
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
for ax, key, ylab, msg in [
        (axE, 'd_dpa', 'Δ DPA accuracy (on−off)', 'Depth change tracks ΔDPA'),
        (axF, 'd_gng', 'Δ GNG accuracy (on−off)', 'Depth change predicts ΔGNG')]:
    xdep = np.array([r['d_depth'] for r in rows_ab])
    yv = np.array([r[key] for r in rows_ab])
    for mouse in JAWS:                                   # join each mouse's A & B dots
        idx = [i for i, r in enumerate(rows_ab) if r['mouse'] == mouse]
        ax.plot(xdep[idx], yv[idx], '-', color=MOUSE_COLOR[mouse], lw=0.8, alpha=0.5, zorder=3)
    for i, r in enumerate(rows_ab):
        face = MOUSE_COLOR[r['mouse']] if r['cls'] == 0 else 'w'    # A solid / B open
        ax.scatter(xdep[i], yv[i], facecolors=face, edgecolors=MOUSE_COLOR[r['mouse']],
                   marker='o', s=95, linewidths=1.2, zorder=5)
    regression_band(ax, xdep, yv, color='0.25')
    ax.axhline(0, ls=':', color='k', lw=0.8); ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_ylim(ylim)
    ok = ~(np.isnan(xdep) | np.isnan(yv))
    r_p, p_p = pearsonr(xdep[ok], yv[ok]); rho, ps = spearmanr(xdep[ok], yv[ok])
    ax.text(0.5, 0.02, f'n={ok.sum()}: r={r_p:+.2f} p={p_p:.3f}  ρ={rho:+.2f} p={ps:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7, color='0.3')
    ax.text(0.85, 0.93, '*' if p_p < 0.05 else 'n.s.', transform=ax.transAxes, ha='center',
            va='top', fontsize=20, fontweight='bold', color='k' if p_p < 0.05 else '0.55')
    ax.set_xlabel('Δ DPA choice-code depth (on−off, trainLD)'); ax.set_ylabel(ylab)
    ax.set_title(msg, loc='left', fontweight='bold', fontsize=TITLE_FS)
    ax.set_box_aspect(1)                                  # square panels
_sample_h = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=7, label='odor A'),
             mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=7, label='odor B')]
axE.legend(handles=_sample_h, frameon=False, fontsize=7, loc='upper left', handletextpad=0.3)

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

# ── L: trial-level GEE — choice-code depth → accuracy (cluster-robust) — ROW 4 ─
axL = fig.add_subplot(gs_body[3, 0:4])
for i, tk in enumerate(('DPA', 'GNG')):
    orr, lo, hi, pv, n = MM[tk]
    if not np.isfinite(orr):
        continue
    cc = RED if tk == 'DPA' else BLUE
    sig = np.isfinite(pv) and pv < 0.05
    axL.errorbar(i, orr, yerr=[[orr - lo], [hi - orr]], fmt='o', color=cc,
                 mfc=cc if sig else 'white', ms=10, capsize=4, lw=1.8, zorder=3)
    lab = star(pv) if star(pv) else 'n.s.'
    axL.text(i, hi + (hi - lo) * 0.06, lab, ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='k' if sig else '0.5')
    axL.text(i, lo - (hi - lo) * 0.10, f'OR={orr:.2f}', ha='center', va='top', fontsize=7, color='0.35')
axL.axhline(1, ls='--', color='0.4', lw=1)
axL.set_xticks([0, 1]); axL.set_xticklabels(['DPA', 'GNG'])
axL.set_xlim(-0.6, 1.6)
_his = [MM[t][2] for t in ('DPA', 'GNG') if np.isfinite(MM[t][2])]
_los = [MM[t][1] for t in ('DPA', 'GNG') if np.isfinite(MM[t][1])]
axL.set_ylim(min(_los + [0.9]) - 0.08, max(_his + [1.1]) * 1.16)   # headroom for star
axL.set_ylabel('depth → accuracy\n(OR per SD, GEE)')
axL.set_title('Code depth predicts accuracy (trial-level)', loc='left', fontweight='bold', fontsize=TITLE_FS)

# ── M, N: signal-detection — d′ (sensitivity) and criterion (bias), OFF vs ON ──
axM = fig.add_subplot(gs_body[3, 4:8])
axN = fig.add_subplot(gs_body[3, 8:12])
_XG = {'DPA': (0.0, 1.0), 'GNG': (2.3, 3.3)}
for ax, met, ylab, ttl in [(axM, 'd', "sensitivity  d′", 'Sensitivity (d′) unchanged'),
                           (axN, 'c', 'criterion  c', 'Bias (criterion) unchanged')]:
    for tk, (x0, x1) in _XG.items():
        off = np.array(SDT[tk][met][0], float); on = np.array(SDT[tk][met][1], float)
        for j, m in enumerate(JAWS):
            if np.isfinite(off[j]) and np.isfinite(on[j]):
                ax.plot([x0, x1], [off[j], on[j]], '-', color=MOUSE_COLOR[m], lw=0.9, alpha=0.5, zorder=2)
        for xx, vv, col in [(x0, off, OFF_C), (x1, on, ON_C)]:
            ok = np.isfinite(vv)
            mn = np.nanmean(vv); se = np.nanstd(vv, ddof=1) / np.sqrt(ok.sum())
            ax.errorbar(xx, mn, yerr=se, fmt='o', color=col, ms=11, capsize=4, lw=2,
                        mec='k', mew=0.6, zorder=5)
        dd = on - off; dd = dd[np.isfinite(dd)]
        pv = float(ttest_1samp(dd, 0).pvalue) if len(dd) > 1 else np.nan
        lab = star(pv) if star(pv) else 'n.s.'
        ytop = np.nanmax(np.concatenate([off, on]))
        ax.text((x0 + x1) / 2, ytop, lab, ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='k' if (np.isfinite(pv) and pv < 0.05) else '0.5')
    ax.axhline(0, ls=':', color='0.6', lw=0.9)
    ax.set_xticks([0, 1, 2.3, 3.3]); ax.set_xticklabels(['OFF', 'ON', 'OFF', 'ON'])
    ax.set_xlim(-0.5, 3.8)
    for xc, tk in [(0.5, 'DPA'), (2.8, 'GNG')]:
        ax.text(xc, -0.15, tk, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=8.5, fontweight='bold', color='0.3')
    ax.set_ylabel(ylab)
    ax.set_title(ttl, loc='left', fontweight='bold', fontsize=TITLE_FS)
axN.legend(handles=[mlines.Line2D([0], [0], marker='o', color=OFF_C, ls='none', ms=8, mec='k', label='laser OFF'),
                    mlines.Line2D([0], [0], marker='o', color=ON_C, ls='none', ms=8, mec='k', label='laser ON')],
           frameon=False, fontsize=7.5, loc='best')

# ── panel letters + row banners ───────────────────────────────────────────────
# reading order: A scheme · B–E batch · F–H recorded · I–K overlaps · L–N mechanism
for _ax, _L in [(axA, 'A'), (axG, 'B'), (axH, 'C'), (axI, 'D'), (axJ, 'E'),
                (axB, 'F'), (axC, 'G'), (axD, 'H'), (axK, 'I'), (axE, 'J'), (axF, 'K'),
                (axL, 'L'), (axM, 'M'), (axN, 'N')]:
    panel_letter(_ax, _L)


def row_banner(ax_left, text, dy=0.014):
    p = ax_left.get_position()
    fig.text(0.055, p.y1 + dy, text, fontsize=9.5, fontweight='bold',
             va='bottom', ha='left', color='0.35')


row_banner(axG, 'Training batch · chronic every-trial silencing · BETWEEN-group opto vs control (ACC-Prl, 9 v 9)')
row_banner(axB, 'Recorded cohort · transient delay-only laser · WITHIN-mouse ON vs OFF (n=5 Jaws inhibition)')
row_banner(axE, 'Same projection · overlaps: laser ON−OFF moves the choice code (Expert, 5 Jaws · A&B independent, 10 pts)')
row_banner(axL, 'Mechanism · trial-level coupling + signal-detection (Expert, 5 Jaws; choice code predicts DPA, transient laser spares d′ & bias)')

fig.text(0.5, 0.004,
         'ACC→Prl(mPFC) projection.  B–E training batch, between-group (every-trial silencing), mean ± SEM; '
         'LMM perf ~ group×day + (1|mouse); per-day stars Welch, uncorrected.  '
         'F–H recorded cohort within-mouse (interleaved laser), Jaws inhibition n=5; LMM perf ~ laser×day + (1|mouse); '
         'F/G per-day stars = one-sample ΔON−OFF.  I per-mouse OFF-vs-ON choice-code depth (Jaws, A&B pooled). '
         'J–K overlaps Δ(on−off), depth = DPA choice-code late-delay (trainLD), odor A&B as independent points '
         '(5 Jaws → 10 pts); star = Pearson.  '
         'L trial-level GEE logistic accuracy ~ depth_z + laser, cluster-robust by mouse (OR per within-mouse SD of depth). '
         'M/N signal-detection d′ & criterion per mouse (loglinear-corrected), OFF vs ON, star = paired t on ΔON−OFF.  '
         '* p<0.05  ** p<0.01  *** p<0.001',
         ha='center', va='bottom', fontsize=7.3, color='0.45')

for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_opto_main.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
