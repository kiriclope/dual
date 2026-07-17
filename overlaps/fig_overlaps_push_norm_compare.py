"""COMPARISON — the DPA no-lick push on the DPA ACTION axis under four per-mouse normalisations.

Push quantity = all-correct-DPA late-delay (LD 45–53) depth on the action axis (choice decoder @57–63),
Naive→Expert (regardless of lick/no-lick trial type). Denominators compared (per mouse):
  per-group (per-stage) = std of THAT stage's mean trajectory (the A 3rd-row recipe; Naive & Expert in
                          DIFFERENT units, so its 'deepening' is not a literal displacement)
  pooled evoked-std     = std of the all-DPA mean trajectory pooled over BOTH stages → ONE unit/mouse,
                          shared by Naive & Expert (literal Δ; equalises mice by evoked amplitude)
  shared baseline-std   = baseline-window std (current push panel; SNR-weights low-noise mice)
  shared signal-std     = all-trials std (eqnorm; includes trial noise → washes the push out)

Row 1: all-DPA mean±SEM trajectory, Naive vs Expert (LD shaded). Row 2: per-mouse Naive→Expert paired
LD depth + mean±SEM + deepening mixed model depth ~ stage + C(sample) + (1|mouse).

Output: figures/overlaps/action/{png,svg}/overlaps_push_norm_compare.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DATA = '../data/overlaps'
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BL = np.arange(0, 12); LD = np.arange(45, 53); xt = np.linspace(0, 14, 84)
_pal = sns.color_palette('tab10', n_colors=len(MICE)); MC = {m: _pal[i] for i, m in enumerate(MICE)}

# --gng = read the DPA state on the GNG lick axis (gng decoder @39–45) instead of the DPA action axis (@57–63)
GNG = '--gng' in sys.argv[1:]
TARGET = 'gng' if GNG else 'choice'
ACT = np.arange(39, 45) if GNG else np.arange(57, 63)
AXLAB = 'GNG action axis (gng decoder @39–45)' if GNG else 'DPA action axis (choice decoder @57–63)'

y = pkl_load(f'labels_{BDUM}', path=DATA); X = np.asarray(pkl_load(f'X_{BDUM}', path=DATA))
ch = (y.target == TARGET).to_numpy(); D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True)

ALLTR = '--all' in sys.argv[1:]                                        # --all = every DPA trial (else correct only)
PERF_S = pd.Series(True, index=yc.index) if ALLTR else (yc.performance == 1)
TRSET = 'ALL DPA trials' if ALLTR else 'correct DPA trials'
SUF = ('_gng' if GNG else '') + ('_all' if ALLTR else '')

NORMS = ['per-group\n(per-stage)', 'pooled evoked-std\n(shared stages)', 'shared\nbaseline-std', 'shared\nsignal-std']


def denom(norm, m, stage):
    sm = (yc.mouse == m).values
    if norm == 0:      # per-group per-stage: std of THIS stage's all-DPA mean traj
        b = ((yc.laser == 0) & (yc.stage == stage) & (yc.tasks == 'DPA') & PERF_S).values & sm
        v = D[b].mean(0); return (v - v[BL].mean()).std()
    if norm == 1:      # pooled evoked: std of all-DPA mean traj pooled over both stages
        b = ((yc.laser == 0) & (yc.tasks == 'DPA') & PERF_S).values & sm
        v = D[b].mean(0); return (v - v[BL].mean()).std()
    if norm == 2:      # shared baseline std
        return D[sm][:, BL].std()
    return D[sm].std()  # shared signal std


def traj(norm, m, stage, pairs=None):
    sm = (yc.mouse == m).values
    b = ((yc.laser == 0) & (yc.stage == stage) & (yc.tasks == 'DPA') & PERF_S).values & sm
    if pairs is not None:
        b = b & yc.odor_pair.isin(pairs).values
    if b.sum() < 3:
        return None
    v = D[b].mean(0); v = v - v[BL].mean()
    return v / (denom(norm, m, stage) + 1e-9)


fig, axs = plt.subplots(2, len(NORMS), figsize=(14, 6.6))
STC = {'Naive': '#4477AA', 'Expert': '#CC3311'}
for ci in range(len(NORMS)):
    # ── row 1: mean±SEM trajectory ──
    ax = axs[0, ci]
    for stage in ('Naive', 'Expert'):
        T = [traj(ci, m, stage) for m in MICE]; T = np.stack([t for t in T if t is not None])
        mu = T.mean(0); se = T.std(0, ddof=1) / np.sqrt(len(T))
        ax.plot(xt, mu, color=STC[stage], lw=2.2, label=stage, zorder=3)
        ax.fill_between(xt, mu - se, mu + se, color=STC[stage], alpha=0.15, lw=0)
    ax.axhline(0, color='0.7', lw=0.7); ax.axvspan(xt[LD[0]], xt[LD[-1]], color='0.85', alpha=0.5, lw=0)
    ax.set_title(NORMS[ci], fontsize=9, fontweight='bold')
    if ci == 0:
        ax.set_ylabel('all-DPA proj.\n(action axis)', fontsize=8); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    ax.set_xlabel('time (s)', fontsize=8)
    # ── row 2: per-mouse Naive→Expert LD depth + stat ──
    ax2 = axs[1, ci]
    N = np.array([np.nanmean([traj(ci, m, 'Naive', p)[LD].mean() if traj(ci, m, 'Naive', p) is not None else np.nan
                              for p in ([0, 1], [2, 3])]) for m in MICE])
    E = np.array([np.nanmean([traj(ci, m, 'Expert', p)[LD].mean() if traj(ci, m, 'Expert', p) is not None else np.nan
                              for p in ([0, 1], [2, 3])]) for m in MICE])
    for i, m in enumerate(MICE):
        ax2.plot([0, 1], [N[i], E[i]], '-o', color=MC[m], lw=0.9, ms=4.5, mec='w', mew=0.4, zorder=3)
    for x, v in ((-0.16, N), (1.16, E)):
        ax2.errorbar(x, np.nanmean(v), yerr=np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()),
                     fmt='s', color='k', ms=7, capsize=4, lw=1.4, zorder=5)
    ax2.axhline(0, ls=':', color='0.6', lw=0.8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['Naive', 'Expert'], fontsize=8); ax2.set_xlim(-0.5, 1.5)
    # deepening mixed model per (mouse, sample)
    rows = []
    for m in MICE:
        for slab, pairs in [('A', [0, 1]), ('B', [2, 3])]:
            for st, k in [('Naive', 0), ('Expert', 1)]:
                t = traj(ci, m, st, pairs)
                if t is not None:
                    rows.append(dict(mouse=m, sample=slab, st=k, depth=t[LD].mean()))
    dfp = pd.DataFrame(rows)
    fit = smf.mixedlm('depth ~ st + C(sample)', dfp, groups=dfp['mouse']).fit()
    b, p = float(fit.params['st']), float(fit.pvalues['st'])
    push = np.nanmean(E - N); cv = np.nanstd(E - N) / (abs(push) + 1e-9)
    if ci == 0:
        ax2.set_ylabel('LD depth\n← no lick    lick →', fontsize=8)
    ax2.set_title(f'Naive={np.nanmean(N):+.2f}  Expert={np.nanmean(E):+.2f}\n'
                  f'push={push:+.2f}  CV={cv:.1f}\nβ={b:+.2f}, p={p:.3f}', fontsize=7.5,
                  color=('k' if p < 0.05 else '0.35'))
    print(f'{NORMS[ci][:12]:14s} Naive={np.nanmean(N):+.3f} Expert={np.nanmean(E):+.3f} push={push:+.3f} p={p:.3f}')
fig.suptitle(f'DPA state on the {AXLAB} ({TRSET}) — four per-mouse normalisations.  '
             'Naive vs Expert LD depth, regardless of trial type.\n'
             'pooled-evoked (col 2) = one shared unit across stages (a literal Δ).',
             fontsize=9.5, y=1.02)
fig.tight_layout(rect=(0, 0, 1, 0.94))
OUT = 'figures/overlaps/action'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
pth = f'{OUT}/png/overlaps_push_norm_compare{SUF}.png'
fig.savefig(pth, dpi=300, bbox_inches='tight'); fig.savefig(pth.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', pth)
