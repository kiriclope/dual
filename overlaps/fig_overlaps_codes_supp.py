"""
fig_overlaps_codes_supp.py — SUPPLEMENT to the overlaps main figure.

Holds the code-characterisation panels pulled out of the main figure (2026-07-15) so the main figure's
panel A can headline the code TRACES + the shared-action TRAJECTORIES:

  Row 1  per-mouse neural d′ (Naive x vs Expert y, unity = unchanged) for sample / test / GNG codes,
         each on its generalisation best axis — the sensory/context/Go-NoGo codes are decodable and
         stable with learning. + CODE ALIGNMENT: pairwise |cos| between the code axes (decoder weights,
         broad late-delay bins 27–53), Naive→Expert — codes near-orthogonal, the CHOICE code demixes
         from both stimulus codes with learning (sample–choice *, choice–test ***; sample–test at chance).
  Row 2  SHARED ACTION/LICK code via cross-cosine: choice axis (DPA lick, rows) × gng axis (Go/NoGo lick,
         cols) at every epoch pair, Naive | Expert — orthogonal during the delay (diagonal ≈ chance) but
         ALIGNED off-diagonally at each task's action/reward moment (DPA resp/rwd 60–84 × GNG rwd 42–60),
         strengthening with learning. The main figure now shows the same result as trajectories on a
         shared axis; this is the weight-space (cosine) corroboration.

Output: figures/overlaps/main/{png,svg}/fig_overlaps_codes_supp.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_codes_supp.py
"""
import matplotlib
matplotlib.use('Agg')
import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import ttest_rel, ttest_1samp
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load

sns.set_context('notebook')
sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
_pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal_mice[i] for i, m in enumerate(ALL_MICE)}

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0, trials='', data_type='dF',
    prescreen=None, pval=0.05, preprocess=None, scaler_BL='standard_BL', avg_noise=False,
    unit_var_BL=False, random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
    scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10, class_weight=0,
    multilabel=0, mne_estimator='generalizing', n_jobs=4, days=['first', 'last'])
_BDUM = f'{DUM}_raw_targets_choice-gng-sample-test'

print('loading main tensor …')
Xb = pkl_load(f'X_{_BDUM}',      path=DATA_IN)
yb = pkl_load(f'labels_{_BDUM}', path=DATA_IN)
_sct = (yb.target != 'gng').to_numpy()
X = Xb[_sct]; y = yb[_sct].reset_index(drop=True)


# ── d′ on generalisation best axes (sample/test), trained AND read on the window ────────────────
def _diag(win):
    return X[:, 1, win, :][:, :, win].mean((1, 2)).astype(float)
_dp_sample = _diag(np.arange(16, 48))
_dp_test   = _diag(np.arange(58, 84))
del X

_gm = (yb.target == 'gng').to_numpy()
Xg = Xb[_gm]; yg = yb[_gm].reset_index(drop=True)
_GNG_WIN = np.arange(34, 59)
_gng_dp = Xg[:, 1, _GNG_WIN, :][:, :, _GNG_WIN].mean((1, 2)).astype(float)
del Xg, Xb

DPRIME_SPECS = [('sample d′', 'sample', 'sample_odor', 1, 0, _dp_sample, True),
                ('test d′',   'test',   'test_odor',   1, 0, _dp_test,   True)]


def _code_dprime(v, target, col, pos, neg, mouse, stage, dpa_only):
    base = ((y.target == target) & (y.mouse == mouse) & (y.stage == stage) &
            (y.laser == 0) & (y.performance == 1)).values
    if dpa_only:
        base = base & (y.tasks == 'DPA').values
    a = v[base & (y[col].values == pos)]; b = v[base & (y[col].values == neg)]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


dpr = {}
for _title, _tgt, _col, _pos, _neg, _v, _dpaonly in DPRIME_SPECS:
    dN, dE, mice = [], [], []
    for mo in ALL_MICE:
        n = _code_dprime(_v, _tgt, _col, _pos, _neg, mo, 'Naive', _dpaonly)
        e = _code_dprime(_v, _tgt, _col, _pos, _neg, mo, 'Expert', _dpaonly)
        if np.isfinite(n) and np.isfinite(e):
            dN.append(n); dE.append(e); mice.append(mo)
    dpr[_title] = dict(naive=np.array(dN), expert=np.array(dE), mice=mice)


def _gng_dprime(mouse, stage):
    base = ((yg.target == 'gng') & (yg.mouse == mouse) & (yg.stage == stage)
            & (yg.laser == 0) & (yg.tasks != 'DPA')).to_numpy()
    a = _gng_dp[base & (yg.gng.to_numpy() == 1)]; b = _gng_dp[base & (yg.gng.to_numpy() == 0)]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


_gN, _gE, _gmice = [], [], []
for mo in ALL_MICE:
    n = _gng_dprime(mo, 'Naive'); e = _gng_dprime(mo, 'Expert')
    if np.isfinite(n) and np.isfinite(e):
        _gN.append(n); _gE.append(e); _gmice.append(mo)
dpr['GNG d′'] = dict(naive=np.array(_gN), expert=np.array(_gE), mice=_gmice)


# ── code-alignment cosine (broad late-delay 27–53) ──────────────────────────────────────────────
BINS_COS = np.arange(27, 54)
_WBLOB = pkl_load(f'weights_{_BDUM}', path=DATA_IN)['weights']
COS_PAIRS = [('sample', 'choice'), ('sample', 'test'), ('choice', 'test')]


def _code_axis(mouse, stage, target, win):
    ws = np.asarray(_WBLOB[(mouse, stage, 'all', target)], float)
    v = ws[win].mean(0); nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


cosmix = {}
for _a, _b in COS_PAIRS:
    cN, cE = [], []
    for mo in ALL_MICE:
        cN.append(abs(_code_axis(mo, 'Naive', _a, BINS_COS) @ _code_axis(mo, 'Naive', _b, BINS_COS)))
        cE.append(abs(_code_axis(mo, 'Expert', _a, BINS_COS) @ _code_axis(mo, 'Expert', _b, BINS_COS)))
    cosmix[(_a, _b)] = dict(naive=np.array(cN), expert=np.array(cE))
COS_CHANCE = 1.0 / np.sqrt(np.mean([np.asarray(_WBLOB[(m, 'Naive', 'all', 'sample')]).shape[1] for m in ALL_MICE]))


# ── shared-action cross-cosine: choice axis × gng axis at every epoch pair ───────────────────────
_EP_ACT = [('stim', 12, 18), ('eDel', 18, 27), ('distr', 27, 33), ('mDel', 33, 39), ('cue', 39, 42),
           ('gngRwd', 42, 45), ('lDel', 45, 54), ('test', 54, 60), ('resp', 60, 72), ('dpaRwd', 72, 84)]
_ACT_CH, _ACT_GN = (60, 84), (42, 60)


def _wax_act(mo, st, tg, a, c):
    k = (mo, st, 'all', tg)
    if k not in _WBLOB:
        return None
    v = np.asarray(_WBLOB[k], float)[a:c].mean(0); n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _act_epmat(st):
    per = []
    for mo in ALL_MICE:
        R = np.full((len(_EP_ACT), len(_EP_ACT)), np.nan)
        for i, (_, a1, c1) in enumerate(_EP_ACT):
            A = _wax_act(mo, st, 'choice', a1, c1)
            if A is None:
                continue
            for j, (_, a2, c2) in enumerate(_EP_ACT):
                B = _wax_act(mo, st, 'gng', a2, c2)
                if B is not None:
                    R[i, j] = A @ B
        per.append(R)
    return np.nanmean(per, 0)


def _act_block(mo, st):
    A = _wax_act(mo, st, 'choice', *_ACT_CH); B = _wax_act(mo, st, 'gng', *_ACT_GN)
    return float(A @ B) if (A is not None and B is not None) else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — 2 rows
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.55,
                      left=0.07, right=0.98, top=0.93, bottom=0.09)

# ── Row 1: d′ scatters + code alignment ─────────────────────────────────────────
gsD = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.55)
DPRIME_TITLES = [s[0] for s in DPRIME_SPECS] + ['GNG d′']
for c, _title in enumerate(DPRIME_TITLES):
    axd = fig.add_subplot(gsD[0, c])
    P = dpr[_title]
    _av = np.concatenate([P['naive'], P['expert']]) if len(P['naive']) else np.array([0.0, 1.0])
    _lo, _hi = float(np.nanmin(_av)), float(np.nanmax(_av)); _pd = (_hi - _lo) * 0.12 or 0.2
    _lim = (min(_lo - _pd, -0.1), _hi + _pd)
    axd.plot(_lim, _lim, ls='--', color='0.6', lw=0.8, zorder=1)
    axd.axhline(0, ls=':', color='0.8', lw=0.6, zorder=0); axd.axvline(0, ls=':', color='0.8', lw=0.6, zorder=0)
    for mo, xn, ye in zip(P['mice'], P['naive'], P['expert']):
        axd.scatter(xn, ye, s=26, facecolors=MOUSE_COLOR[mo], edgecolors=MOUSE_COLOR[mo], linewidths=0.6, zorder=4)
    _n = len(P['naive'])
    _tp = float(ttest_rel(P['expert'], P['naive']).pvalue) if _n >= 3 else np.nan
    _dm = float((P['expert'] - P['naive']).mean()) if _n else np.nan
    _sg = (_tp == _tp and _tp < 0.05)
    axd.set_xlim(_lim); axd.set_ylim(_lim); axd.set_box_aspect(1)
    axd.set_title(_title, fontsize=8); axd.set_xlabel('Naive d′', fontsize=7.5)
    if c == 0:
        axd.set_ylabel('Expert d′', fontsize=7.5)
    axd.text(0.06, 0.94, '*' if _sg else 'n.s.', transform=axd.transAxes, ha='left', va='top',
             fontsize=11 if _sg else 7, fontweight='bold', color='k' if _sg else '0.55')
    axd.text(0.5, 0.02, f'Δ={_dm:+.2f}\np={_tp:.3f}', transform=axd.transAxes, ha='center', va='bottom',
             fontsize=6, color='0.3')
    print(f"d′[{_title.strip()}] Δ={_dm:+.3f} p={_tp:.3f} n={_n}")

axCos = fig.add_subplot(gsD[0, 3])
axCos.axhline(COS_CHANCE, ls=':', color='0.6', lw=0.8, zorder=1)
axCos.text(-0.25, COS_CHANCE, 'chance', ha='left', va='bottom', fontsize=5.5, color='0.6')
_LSH = {'sample': 'S', 'choice': 'C', 'test': 'T'}
_sig_i = 0
for _pr in COS_PAIRS:
    _cN = cosmix[_pr]['naive'].mean(); _cE = cosmix[_pr]['expert'].mean()
    _tp = float(ttest_rel(cosmix[_pr]['expert'], cosmix[_pr]['naive']).pvalue)
    _sig = _tp < 0.05
    _col = '#cc3311' if _sig else '0.75'
    axCos.plot([0, 1], [_cN, _cE], '-o', color=_col, lw=2.4 if _sig else 1.0, ms=6 if _sig else 3, zorder=5 if _sig else 2)
    if _sig:
        _star = '***' if _tp < 0.001 else ('**' if _tp < 0.01 else '*')
        axCos.annotate(f'{_LSH[_pr[0]]}–{_LSH[_pr[1]]} {_star}', (1, _cE), xytext=(5, 6 - 12 * _sig_i),
                       textcoords='offset points', va='center', ha='left', color=_col, fontsize=6.5, fontweight='bold')
        _sig_i += 1
    print(f'cos[{_pr[0]}-{_pr[1]}] N={_cN:.3f}→E={_cE:.3f} p={_tp:.3f}')
axCos.set_xticks([0, 1]); axCos.set_xticklabels(['Naive', 'Expert'], fontsize=7)
axCos.set_xlim(-0.3, 1.75); axCos.set_ylim(bottom=0.0)
axCos.set_title('code alignment', fontsize=8); axCos.set_ylabel('|cos| between codes', fontsize=7.5)
axCos.set_box_aspect(1)

# ── Row 2: shared-action cross-cosine heatmaps + paired growth ──────────────────
gsA = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.5)
_eplab = [n for n, _, _ in _EP_ACT]
_bi = {n: k for k, (n, _, _) in enumerate(_EP_ACT)}
_br = (_bi['resp'], _bi['dpaRwd']); _bc = (_bi['gngRwd'], _bi['test'])
_axAct = []
for _ci, _st in enumerate(('Naive', 'Expert')):
    axm = fig.add_subplot(gsA[0, _ci]); _axAct.append(axm)
    _M = _act_epmat(_st)
    im_act = axm.imshow(_M, vmin=-0.25, vmax=0.25, cmap='RdBu_r', aspect='equal')
    axm.add_patch(Rectangle((_bc[0] - 0.5, _br[0] - 0.5), _bc[1] - _bc[0] + 1, _br[1] - _br[0] + 1,
                            fill=False, ec='k', lw=1.6))
    axm.set_xticks(range(len(_EP_ACT))); axm.set_xticklabels(_eplab, rotation=90, fontsize=5.5)
    axm.set_yticks(range(len(_EP_ACT))); axm.set_yticklabels(_eplab, fontsize=5.5)
    axm.set_title(_st, fontsize=TITLE_FS)
    axm.set_xlabel('GNG code (Go/NoGo lick)', fontsize=7)
    if _ci == 0:
        axm.set_ylabel('choice code (DPA lick)', fontsize=7)
    axm.tick_params(length=2)
_cbA = fig.colorbar(im_act, ax=_axAct, fraction=0.023, pad=0.02)
_cbA.set_label('signed cos', fontsize=6.5); _cbA.ax.tick_params(labelsize=6)

axActp = fig.add_subplot(gsA[0, 2])
_aN = np.array([_act_block(m, 'Naive') for m in ALL_MICE]); _aE = np.array([_act_block(m, 'Expert') for m in ALL_MICE])
for _i, _m in enumerate(ALL_MICE):
    axActp.plot([0, 1], [_aN[_i], _aE[_i]], '-o', color=MOUSE_COLOR[_m], lw=0.9, ms=4.5, mec='w', mew=0.5, zorder=3)
for _x, _v in ((-0.18, _aN), (1.18, _aE)):
    _mu = np.nanmean(_v); _se = np.nanstd(_v, ddof=1) / np.sqrt(np.isfinite(_v).sum())
    axActp.errorbar(_x, _mu, yerr=_se, fmt='s', color='k', ms=6, capsize=3.5, lw=1.3, zorder=5)
axActp.axhline(COS_CHANCE, ls=':', color='0.6', lw=0.8); axActp.axhline(0, color='0.85', lw=0.6)
axActp.text(1.6, COS_CHANCE, 'chance', fontsize=5.5, color='0.6', va='bottom', ha='right')
axActp.set_xticks([0, 1]); axActp.set_xticklabels(['Naive', 'Expert'], fontsize=8); axActp.set_xlim(-0.5, 1.7)
axActp.set_ylabel('action-code alignment\ncos(choice@resp/rwd · gng@rwd)', fontsize=7.5)
_apN = float(ttest_1samp(_aN[np.isfinite(_aN)], 0).pvalue); _apE = float(ttest_1samp(_aE[np.isfinite(_aE)], 0).pvalue)
_apNE = float(ttest_1samp((_aE - _aN)[np.isfinite(_aE - _aN)], 0).pvalue)
_sigAct = _apNE < 0.05
axActp.set_title('shared action code', loc='left', fontsize=TITLE_FS)
axActp.text(0.06, 0.96, '*' if _sigAct else 'n.s.', transform=axActp.transAxes, ha='left', va='top',
            fontsize=12 if _sigAct else 8, fontweight='bold', color='k' if _sigAct else '0.55')
axActp.text(0.5, 0.02, f'Naive {np.nanmean(_aN):+.2f} (p={_apN:.3f})  Expert {np.nanmean(_aE):+.2f} (p={_apE:.3f})\n'
                       f'Δ paired-t p={_apNE:.3f}', transform=axActp.transAxes, ha='center', va='bottom',
            fontsize=6, color='0.3')
axActp.set_box_aspect(1)
print(f'action block cos Naive={np.nanmean(_aN):+.3f} Expert={np.nanmean(_aE):+.3f} Δ p={_apNE:.3f}')

fig.suptitle('Supplement — code characterisation: decodability (d′), axis alignment, and the shared action code (weight-space cosine)',
             fontsize=9.5, y=0.985)

OUT = 'figures/overlaps/main'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_codes_supp.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
