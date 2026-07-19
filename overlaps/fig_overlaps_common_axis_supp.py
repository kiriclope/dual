"""SUPPLEMENT — is the no-lick PUSH a genuine state movement or a per-(mouse,stage) decoder-axis
ROTATION artifact? Project the delay state onto a FIXED axis for BOTH stages and re-test.

Cells are registered across days (Naive∩Expert valid-neuron Jaccard = 1.00 for all 9 mice), so a
single fixed axis is well defined. Depth = (delay activity · choice axis) normalised by the
class-signed pooled evoked-std (the main figure's normalisation); choice axis = mean of the choice
decoder weight over the DPA action window (train bins 57-63).

Three axis definitions × two rows:
  row 1  PUSH   : per-mouse Naive→Expert LD-depth paired plot + deepening LMM depth~stage+C(sample)+(1|mouse)
  row 2  COUPLING: Δdepth vs ΔDPA-accuracy, per-mouse n=9 Spearman
Columns: per-stage (current figure) | COMMON = Expert axis | COMMON = pooled axis.

Message: the push is strong on per-stage axes but attenuates to a trend on a fixed axis (part of it is
axis reorganisation); the behaviour coupling survives on the pooled fixed axis (ρ≈−0.72 p≈.03).
Output: figures/overlaps/controls/{png,svg}/overlaps_common_axis_control.{png,svg}
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
ACT = np.arange(57, 63); LD = np.asarray(set_options()['bins_LD']); BL = np.arange(0, 12); SAMPLES = [('A', [0, 1]), ('B', [2, 3])]  # LD = figure's bins_LD [48..53]
_pal = sns.color_palette('tab10', n_colors=len(MICE)); MC = {m: _pal[i] for i, m in enumerate(MICE)}

Wb = pkl_load(f'weights_{BDUM}', path=DATA); W, VALID = Wb['weights'], Wb['valid']
print('loading X_all …', flush=True)
Xall = np.asarray(pkl_load('X_all_nan_', path='../data/pca')); yall = pkl_load('y_all_nan_', path='../data/pca')
print('  X_all', Xall.shape, flush=True)


def axis_act(m, stage):
    return np.asarray(W[(m, stage, 'all', 'choice')])[ACT].mean(0)


def project(m, stage, axis, both):
    idx = ((yall.mouse == m) & (yall.learning == stage) & (yall.tasks == 'DPA') & (yall.laser == 0)).to_numpy()
    Xs = Xall[idx][:, both, :]
    P = np.nansum(Xs * axis[None, :, None], axis=1)
    return P, yall.loc[idx].reset_index(drop=True)


def build(mode):
    """returns depth DataFrame (mouse,sample,st,depth) + per-(mouse,sample) Δdepth dict."""
    rows, dd = [], {}
    for m in MICE:
        vN, vE = VALID[(m, 'Naive')], VALID[(m, 'Expert')]; both = vN & vE
        wE = axis_act(m, 'Expert')[vN[vE]]; wN = axis_act(m, 'Naive')[vE[vN]]; wP = 0.5 * (wE + wN)
        P, ys = {}, {}
        for stg in ('Naive', 'Expert'):
            ax = {'perstage': (wE if stg == 'Expert' else wN), 'commonE': wE, 'commonPool': wP}[mode]
            P[stg], ys[stg] = project(m, stg, ax, both)
        Pall = np.vstack([P['Naive'], P['Expert']])
        chall = np.concatenate([ys['Naive'].choice.to_numpy(), ys['Expert'].choice.to_numpy()])
        s = np.where(chall == 1, 1.0, -1.0); vbar = (s[:, None] * Pall).mean(0)
        sd = (vbar - vbar[BL].mean()).std() + 1e-9; mu = Pall[:, BL].mean()
        dep = {}
        for stg, k in [('Naive', 0), ('Expert', 1)]:
            for sl, pr in SAMPLES:
                sel = ys[stg].odor_pair.isin(pr).to_numpy()
                if sel.sum():
                    d = ((P[stg][sel][:, LD].mean(1) - mu) / sd).mean(); rows.append(dict(mouse=m, sample=sl, st=k, depth=d)); dep[(sl, k)] = d
        for sl, _ in SAMPLES:
            if (sl, 0) in dep and (sl, 1) in dep:
                dd[(m, sl)] = dep[(sl, 1)] - dep[(sl, 0)]
    return pd.DataFrame(rows), dd


def dacc(m):
    def one(pr, stg):
        b = ((yall.mouse == m) & (yall.learning == stg) & (yall.tasks == 'DPA') & (yall.laser == 0) & yall.odor_pair.isin(pr)).to_numpy()
        return yall.loc[b, 'performance'].mean() if b.sum() else np.nan
    return np.nanmean([one(pr, 'Expert') - one(pr, 'Naive') for _, pr in SAMPLES])
DACC = {m: dacc(m) for m in MICE}

MODES = [('perstage', 'per-stage axis\n(current figure)'), ('commonE', 'fixed COMMON axis\n(Expert)'),
         ('commonPool', 'fixed COMMON axis\n(pooled Naive+Expert)')]
fig, axs = plt.subplots(2, 3, figsize=(8.5, 5.4))
for ci, (mode, title) in enumerate(MODES):
    df, dd = build(mode)
    # ── row 1: push paired ──
    ax = axs[0, ci]
    piv = df.pivot_table(index=['mouse', 'sample'], columns='st', values='depth')
    for (m, sl), r in piv.iterrows():
        ax.plot([0, 1], [r[0], r[1]], '-o', color=MC[m], lw=0.8, ms=4, mec='w', mew=0.4,
                mfc=(MC[m] if sl == 'A' else 'w'), zorder=3)
    for x, k in ((-0.16, 0), (1.16, 1)):
        v = df[df.st == k].depth.values
        ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='s', color='k', ms=7, capsize=4, lw=1.4, zorder=5)
    fit = smf.mixedlm('depth ~ st + C(sample)', df, groups=df['mouse']).fit()
    b, p = float(fit.params['st']), float(fit.pvalues['st'])
    ax.axhline(0, ls=':', color='0.6', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert']); ax.set_xlim(-0.5, 1.5)
    ax.set_title(title, loc='left', fontsize=TITLE_FS)
    ax.text(0.03, 0.03, f'push β={b:+.2f}\np={p:.3f}', transform=ax.transAxes, va='bottom', fontsize=6.5,
            color='k' if p < .05 else '0.4')
    ax.text(0.95, 0.95, '*' if p < .05 else 'n.s.', transform=ax.transAxes, ha='right', va='top',
            fontsize=12 if p < .05 else 8, fontweight='bold', color='k' if p < .05 else '0.55')
    if ci == 0:
        ax.set_ylabel('LD choice-code depth\n← no lick     lick →')
    # ── row 2: coupling ──
    ax2 = axs[1, ci]
    ddm = np.array([np.nanmean([dd.get((m, sl), np.nan) for sl, _ in SAMPLES]) for m in MICE])
    dam = np.array([DACC[m] for m in MICE]); ok = np.isfinite(ddm) & np.isfinite(dam)
    for i, m in enumerate(MICE):
        ax2.scatter(ddm[i], dam[i], facecolors=MC[m], edgecolors=MC[m], s=44, zorder=4)
    if ok.sum() > 2:
        z = np.polyfit(ddm[ok], dam[ok], 1); xx = np.array([ddm[ok].min(), ddm[ok].max()])
        ax2.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.6, zorder=3)
    rho, pc = spearmanr(ddm[ok], dam[ok])
    ax2.axhline(0, ls=':', color='k', lw=0.7); ax2.axvline(0, ls=':', color='k', lw=0.7)
    ax2.text(0.03, 0.03, f'n=9 Spearman\nρ={rho:+.2f}, p={pc:.3f}', transform=ax2.transAxes, va='bottom', fontsize=6.5,
             color='k' if pc < .05 else '0.4')
    ax2.text(0.95, 0.95, '*' if pc < .05 else 'n.s.', transform=ax2.transAxes, ha='right', va='top',
             fontsize=12 if pc < .05 else 8, fontweight='bold', color='k' if pc < .05 else '0.55')
    ax2.set_xlabel('Δ choice-code depth (Exp−Naive)')
    if ci == 0:
        ax2.set_ylabel('Δ DPA accuracy (Exp−Naive)')
fig.suptitle('Fixed-axis control: the no-lick push is partly decoder-axis reorganisation, but the '
             'depth↔accuracy coupling survives on a fixed pooled axis', fontsize=9, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT = 'figures/overlaps/controls'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
p = f'{OUT}/png/overlaps_common_axis_control.png'
fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
