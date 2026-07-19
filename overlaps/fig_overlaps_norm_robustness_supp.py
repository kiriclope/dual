"""SUPPLEMENT — is the figure's result an artifact of the (bespoke) class-signed pooled-evoked
normalisation? Recompute the two headline effects under SIX per-mouse normalisations of the same
DPA action-axis depth (choice decoder @57-63, read @LD 45-53):
  raw (none) | baseline-std | eqnorm (whole-trial std) | pooled-evoked (figure default) |
  d'-action (within-class σ at the action window — a STANDARD signal-detection unit) |
  gap-action (lick−nolick class separation).
Left panel  = PUSH   (within-mouse deepening LMM depth~stage+C(sample)+(1|mouse), 36 obs).
Right panel = COUPLING (between-mouse per-mouse n=9 Spearman Δdepth↔ΔDPA-acc).
Message: the depth↔behaviour COUPLING is significant under EVERY normalisation (incl. raw); the PUSH is
normalisation-sensitive. d'-action is the recommended interpretable primary unit (ties to the opto d').
Output: figures/overlaps/controls/{png,svg}/overlaps_norm_robustness.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr
from src.pca.io import pkl_load
from src.common.options import set_options

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({          # shared house style (matches fig_overlaps_main_native.py; see CLAUDE.md)
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
DATA = '../data/overlaps'
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BL = np.arange(0, 12); LD = np.asarray(set_options()['bins_LD']); ACT = np.arange(57, 63); SAMPLES = [('A', [0, 1]), ('B', [2, 3])]  # LD = figure's bins_LD [48..53]
NORMS = ['raw', 'baseline-std', 'eqnorm', 'pooled-evoked', "d'-action", 'gap-action']

y = pkl_load(f'labels_{BDUM}', path=DATA); X = np.asarray(pkl_load(f'X_{BDUM}', path=DATA))
ch = (y.target == 'choice').to_numpy(); D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True); del X
lo = (yc.laser == 0).to_numpy(); dpa = (yc.tasks == 'DPA').to_numpy(); cho = yc.choice.to_numpy()
mo = yc.mouse.to_numpy(); st = yc.stage.to_numpy(); op = yc.odor_pair.to_numpy(); perf = yc.performance.to_numpy()

MU, DEN = {}, {}
for m in MICE:
    pool = lo & dpa & (mo == m); Dp = D[pool]; cp = cho[pool]
    MU[m] = Dp[:, BL].mean()
    s = np.where(cp == 1, 1.0, -1.0); vbar = (s[:, None] * Dp).mean(0)
    a = Dp[:, ACT].mean(1); al, an = a[cp == 1], a[cp == 0]
    DEN[m] = {'raw': 1.0, 'baseline-std': Dp[:, BL].std(), 'eqnorm': Dp.std(),
              'pooled-evoked': (vbar - vbar[BL].mean()).std(),
              "d'-action": np.sqrt(0.5 * (al.var(ddof=1) + an.var(ddof=1))), 'gap-action': abs(al.mean() - an.mean())}
rdel = D[:, LD].mean(1) - np.array([MU[m] for m in mo])


def dd_ms(m, pr, nm):
    def a(stg):
        b = lo & dpa & (mo == m) & np.isin(op, pr) & (st == stg); return (rdel[b] / DEN[m][nm]).mean() if b.sum() else np.nan
    return a('Expert') - a('Naive')

def dacc(m, pr):
    def a(stg):
        b = lo & dpa & (mo == m) & np.isin(op, pr) & (st == stg); return perf[b].mean() if b.sum() else np.nan
    return a('Expert') - a('Naive')
DACC = {m: np.nanmean([dacc(m, pr) for _, pr in SAMPLES]) for m in MICE}

push, coup = {}, {}
for nm in NORMS:
    rows = []
    for m in MICE:
        for sl, pr in SAMPLES:
            for stg, k in [('Naive', 0), ('Expert', 1)]:
                b = lo & dpa & (mo == m) & np.isin(op, pr) & (st == stg)
                if b.sum():
                    rows.append(dict(mouse=m, sample=sl, stg=k, depth=(rdel[b] / DEN[m][nm]).mean()))
    df = pd.DataFrame(rows)
    f = smf.mixedlm('depth ~ stg + C(sample)', df, groups=df['mouse']).fit()
    push[nm] = (float(f.params['stg']), float(f.pvalues['stg']))
    dd = np.array([np.nanmean([dd_ms(m, pr, nm) for _, pr in SAMPLES]) for m in MICE])
    da = np.array([DACC[m] for m in MICE]); ok = np.isfinite(dd) & np.isfinite(da)
    rho, p = spearmanr(dd[ok], da[ok]); coup[nm] = (float(rho), float(p))

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
yv = np.arange(len(NORMS))[::-1]
for ax, dat, lab, xlab in [(axs[0], push, 'PUSH (within-mouse deepening)', 'LMM β (depth ~ stage)'),
                           (axs[1], coup, 'COUPLING (between-mouse Δdepth↔ΔDPA-acc)', 'Spearman ρ (n=9)')]:
    for i, nm in enumerate(NORMS):
        val, p = dat[nm]; sig = p < 0.05
        c = '#CC3311' if sig else '0.6'
        ax.scatter(val, yv[i], s=45, color=c, zorder=3, edgecolors='k', linewidths=0.5)
        ax.text(val, yv[i] + 0.28, f'p={p:.3f}' + (' *' if sig else ''), ha='center', va='bottom',
                fontsize=6.5, color=('k' if sig else '0.45'))
    ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_yticks(yv); ax.set_yticklabels(NORMS)
    ax.set_xlabel(xlab); ax.set_title(lab, loc='left', fontsize=TITLE_FS)
    ax.margins(x=0.22); ax.set_ylim(-0.6, len(NORMS) - 0.15)   # headroom so the top p-label clears the title
axs[0].text(0.5, -0.22, "d'-action = within-class σ at the action window (a signal-detection unit; ties to the opto figure's behavioural d′)",
            transform=axs[0].transAxes, ha='center', fontsize=6, color='0.3')
fig.suptitle('Normalisation robustness: the depth↔behaviour COUPLING holds under every normalisation '
             '(incl. raw); the PUSH is normalisation-sensitive', fontsize=9, y=1.02)
fig.tight_layout()
OUT = 'figures/overlaps/controls'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
p = f'{OUT}/png/overlaps_norm_robustness.png'
fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
for nm in NORMS:
    print(f'  {nm:14s} push β={push[nm][0]:+.3f} p={push[nm][1]:.3f}   coupling ρ={coup[nm][0]:+.2f} p={coup[nm][1]:.3f}')
