"""SUPPLEMENT S8b — reviewer-facing robustness of the panel-D coupling
(Δdepth ↔ ΔDPA-accuracy) rendered as a compact FOREST plot.

Same build as the main figure's panel D (pooled-evoked class-signed depth of the DPA
choice/action axis @57-63, read at the late delay LD 45-53). For the between-mouse
Δdepth↔Δaccuracy coupling we run the resampling battery a NatNeuro reviewer expects and
show every estimate as a point ± interval against the null (x=0):
  - per-mouse n=9 Spearman ρ  (headline)          — point ± bootstrap 95% CI (resample mice)
  - jackknife leave-one-mouse-out                  — range of the 9 LOO ρ
  - permutation (shuffle Δacc across mice)         — observed ρ vs the n=9 null band
  - Mundlak within–between LMM between-mouse β     — the principled multilevel estimator (own β scale)
ΔGNG is carried through the whole battery as the null control (should straddle 0).

All numbers are computed live here (identical logic + RNG seed 0 to
/home/leon/.claude/jobs/a9688faa/tmp/coupling_robustness.py, so the plotted values match the
diagnostic print verbatim: ΔDPA ρ=-0.833 p=.005, Mundlak β=-0.041 p=.006, jackknife 9/9,
bootstrap CI [-1.00,-0.26], permutation p=.008; ΔGNG null throughout).

Output: figures/overlaps/controls/{png,svg}/overlaps_coupling_battery.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from src.pca.io import pkl_load

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

# ---------------------------------------------------------------- battery (mirrors coupling_robustness.py)
DATA = '../data/overlaps'
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BL = np.arange(0, 12); LD = np.arange(45, 53); ACT = np.arange(57, 63); SAMPLES = [('A', [0, 1]), ('B', [2, 3])]
RNG = np.random.default_rng(0)

y = pkl_load(f'labels_{BDUM}', path=DATA); X = np.asarray(pkl_load(f'X_{BDUM}', path=DATA))
ch = (y.target == 'choice').to_numpy(); D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True); del X
lo = (yc.laser == 0).to_numpy(); dpa = (yc.tasks == 'DPA').to_numpy(); cho = yc.choice.to_numpy()
mo = yc.mouse.to_numpy(); st = yc.stage.to_numpy(); op = yc.odor_pair.to_numpy(); perf = yc.performance.to_numpy(); odr = yc.odr_perf.to_numpy()

MU, SD = {}, {}
for m in MICE:
    pool = lo & dpa & (mo == m); Dp = D[pool]; cp = cho[pool]
    MU[m] = Dp[:, BL].mean(); s = np.where(cp == 1, 1.0, -1.0); vbar = (s[:, None] * Dp).mean(0)
    SD[m] = (vbar - vbar[BL].mean()).std() + 1e-9
rdel = D[:, LD].mean(1) - np.array([MU[m] for m in mo])


def dd_mouse_sample(m, pr):
    def a(stg):
        b = lo & dpa & (mo == m) & np.isin(op, pr) & (st == stg); return (rdel[b] / SD[m]).mean() if b.sum() else np.nan
    return a('Expert') - a('Naive')


def dperf(m, pr, is_dpa):
    tk = dpa if is_dpa else ~dpa; col = perf if is_dpa else odr
    def a(stg):
        b = lo & tk & (mo == m) & np.isin(op, pr) & (st == stg); v = col[b]; v = v[np.isfinite(v)]; return v.mean() if len(v) else np.nan
    return a('Expert') - a('Naive')


dd = np.array([np.nanmean([dd_mouse_sample(m, pr) for _, pr in SAMPLES]) for m in MICE])
RES = {}
for lab, is_dpa in [('DPA', True), ('GNG', False)]:      # order matters: keeps RNG stream == diagnostic
    da = np.array([np.nanmean([dperf(m, pr, is_dpa) for _, pr in SAMPLES]) for m in MICE])
    rho, p = spearmanr(dd, da)
    # (a) Mundlak within-between LMM on 18 obs -> between-mouse slope β (+ Wald 95% CI)
    rows = []
    for m in MICE:
        for sl, pr in SAMPLES:
            rows.append(dict(mouse=m, ddi=dd_mouse_sample(m, pr), dpi=dperf(m, pr, is_dpa)))
    d18 = pd.DataFrame(rows).dropna()
    d18['dd_bw'] = d18.groupby('mouse')['ddi'].transform('mean')
    d18['dd_wi'] = d18['ddi'] - d18['dd_bw']
    f = smf.mixedlm('dpi ~ dd_bw + dd_wi', d18, groups=d18['mouse']).fit()
    mb, mp, mse = float(f.params['dd_bw']), float(f.pvalues['dd_bw']), float(f.bse['dd_bw'])
    # (b) jackknife
    jk = np.array([spearmanr(np.delete(dd, i), np.delete(da, i))[0] for i in range(len(MICE))])
    jkp = np.array([spearmanr(np.delete(dd, i), np.delete(da, i))[1] for i in range(len(MICE))])
    # (c) bootstrap CI (resample mice)
    boot = []
    for _ in range(5000):
        idx = RNG.integers(0, len(MICE), len(MICE))
        if len(np.unique(dd[idx])) > 2:
            boot.append(spearmanr(dd[idx], da[idx])[0])
    blo, bhi = np.nanpercentile(boot, [2.5, 97.5])
    # (d) permutation (shuffle Δacc across mice) -> p + n=9 null 95% band
    perm = np.array([spearmanr(dd, RNG.permutation(da))[0] for _ in range(10000)])
    pperm = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)
    nlo, nhi = np.percentile(perm, [2.5, 97.5])
    RES[lab] = dict(rho=rho, p=p, mb=mb, mp=mp, mse=mse,
                    jkmin=jk.min(), jkmax=jk.max(), jkp_ok=int((jkp < .05).sum()),
                    blo=blo, bhi=bhi, pperm=pperm, nlo=nlo, nhi=nhi)
    print(f'{lab}: rho={rho:+.3f} p={p:.3f} | Mundlak b={mb:+.4f} p={mp:.3f} | jk[{jk.min():+.2f},{jk.max():+.2f}] '
          f'{int((jkp<.05).sum())}/9 | boot[{blo:+.2f},{bhi:+.2f}] | perm p={pperm:.4f}')

# ----------------------------------------------------------------------------------- forest render
CDPA, CGNG = '#222222', '#9a9a9a'                        # ΔDPA dark, ΔGNG grey
OFF = 0.18                                               # vertical split of the two groups within a row
GROUPS = [('DPA', CDPA, +OFF), ('GNG', CGNG, -OFF)]

fig = plt.figure(figsize=(7.5, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.05], wspace=0.30)
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])

# --- LEFT: three correlation-scale checks on the Spearman ρ axis ---
ROWS = ['Per-mouse Spearman (n=9)\n± bootstrap 95% CI', 'Jackknife\n(leave-one-out, 9)',
        'Permutation (10 000)\nvs n=9 null band']
YR = [2, 1, 0]                                           # row 0 (Spearman) on top
TXTX = 1.18                                              # left edge of the right-hand numeric column


def whisk(ax, x, y, lo_, hi_, c):
    ax.plot([lo_, hi_], [y, y], color=c, lw=1.3, solid_capstyle='round', zorder=3)
    ax.plot([lo_, lo_], [y - 0.06, y + 0.06], color=c, lw=1.0, zorder=3)
    ax.plot([hi_, hi_], [y - 0.06, y + 0.06], color=c, lw=1.0, zorder=3)


for lab, c, dy in GROUPS:
    r = RES[lab]
    # row 0: Spearman point ± bootstrap CI
    whisk(axL, r['rho'], YR[0] + dy, r['blo'], r['bhi'], c)
    axL.scatter(r['rho'], YR[0] + dy, s=34, color=c, ec='k', lw=0.5, zorder=4)
    # row 1: jackknife range (point = full-sample ρ)
    whisk(axL, r['rho'], YR[1] + dy, r['jkmin'], r['jkmax'], c)
    axL.scatter(r['rho'], YR[1] + dy, s=34, color=c, ec='k', lw=0.5, zorder=4)
    # row 2: permutation observed ρ (null band drawn once below)
    axL.scatter(r['rho'], YR[2] + dy, s=34, color=c, ec='k', lw=0.5, zorder=4)

# permutation n=9 null band (n=9 Spearman null is set by rank count -> shared by both labels)
nb_lo, nb_hi = RES['DPA']['nlo'], RES['DPA']['nhi']
axL.add_patch(Rectangle((nb_lo, YR[2] - 0.34), nb_hi - nb_lo, 0.68, facecolor='0.88', edgecolor='none', zorder=0))
axL.text(0.0, YR[2] - 0.46, 'n=9 permutation null (95%)', ha='center', va='top', fontsize=5.6, color='0.45')

# numeric column (whiskers already draw the CI / jackknife range, so print only ρ/p/verdict) + sig markers
ANN = {
    'DPA': [f"ρ={RES['DPA']['rho']:+.2f}  p={RES['DPA']['p']:.3f}",
            f"{RES['DPA']['jkp_ok']}/9 LOO  p<.05",
            f"p={RES['DPA']['pperm']:.3f}"],
    'GNG': [f"ρ={RES['GNG']['rho']:+.2f}  p={RES['GNG']['p']:.2f}",
            f"{RES['GNG']['jkp_ok']}/9 LOO",
            f"p={RES['GNG']['pperm']:.2f}"],
}
SIG = {'DPA': [RES['DPA']['p'] < .05, RES['DPA']['jkp_ok'] == 9, RES['DPA']['pperm'] < .05],
       'GNG': [RES['GNG']['p'] < .05, RES['GNG']['jkp_ok'] > 0, RES['GNG']['pperm'] < .05]}
for i in range(3):
    for lab, c, dy in GROUPS:
        tc = '0.15' if lab == 'DPA' else '0.5'
        axL.text(TXTX, YR[i] + dy, ANN[lab][i], ha='left', va='center', fontsize=6.5, color=tc)
        sig = SIG[lab][i]
        axL.text(2.28, YR[i] + dy, '*' if sig else 'n.s.', ha='center', va='center',
                 fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')

axL.axvline(0, ls=':', color='k', lw=0.8, zorder=1)
axL.plot([1.12, 1.12], [-0.62, 2.55], color='0.8', lw=0.7, zorder=1)          # column divider
axL.set_xlim(-1.18, 2.55); axL.set_ylim(-0.75, 2.6)
axL.set_xticks([-1, -0.5, 0, 0.5, 1]); axL.spines['bottom'].set_bounds(-1.05, 1.05)
axL.set_yticks(YR); axL.set_yticklabels(ROWS)
axL.set_xlabel('Spearman ρ  (Δdepth ↔ Δaccuracy, between mice)')
axL.set_title('Coupling resampling battery — correlation scale', loc='left', fontsize=TITLE_FS)

# --- RIGHT: Mundlak within–between LMM on its own β scale (β ± Wald 95% CI) ---
for lab, c, dy in GROUPS:
    r = RES[lab]; ci = 1.96 * r['mse']; tc = '0.15' if lab == 'DPA' else '0.5'; sig = r['mp'] < .05
    whisk(axR, r['mb'], 1 + dy, r['mb'] - ci, r['mb'] + ci, c)
    axR.scatter(r['mb'], 1 + dy, s=34, color=c, ec='k', lw=0.5, zorder=4)
    if lab == 'DPA':                                   # DPA text above its marker, GNG below -> no overlap
        axR.text(r['mb'], 1 + dy + 0.20, f"β={r['mb']:+.3f}  p={r['mp']:.3f}", ha='center', va='bottom', fontsize=6.5, color=tc)
    else:
        axR.text(r['mb'], 1 + dy - 0.20, f"β={r['mb']:+.3f}  p={r['mp']:.3f}", ha='center', va='top', fontsize=6.5, color=tc)
    axR.text(r['mb'] - ci - 0.008 if r['mb'] < 0 else r['mb'] + ci + 0.008, 1 + dy,
             '*' if sig else 'n.s.', ha='right' if r['mb'] < 0 else 'left', va='center',
             fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
axR.axvline(0, ls=':', color='k', lw=0.8, zorder=1)
axR.set_xlim(-0.105, 0.165); axR.set_ylim(-0.75, 2.6)
axR.set_yticks([]); axR.spines['left'].set_visible(False)
axR.set_xlabel('Between-mouse slope β')
axR.set_title('Multilevel estimator\n(Mundlak within–between LMM)', loc='left', fontsize=TITLE_FS)

# shared legend + suptitle
handles = [Line2D([0], [0], marker='o', color=CDPA, mfc=CDPA, mec='k', mew=0.5, lw=1.3, label='ΔDPA (memory accuracy)'),
           Line2D([0], [0], marker='o', color=CGNG, mfc=CGNG, mec='k', mew=0.5, lw=1.3, label='ΔGNG (go/no-go accuracy — null control)')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False, fontsize=6.5)
fig.tight_layout(rect=[0, 0, 1, 0.965])
# (no centered banner — the two left-aligned panel titles + the ED-figure legend carry the message)

OUT = 'figures/overlaps/controls'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
pth = f'{OUT}/png/overlaps_coupling_battery.png'
fig.savefig(pth, bbox_inches='tight'); fig.savefig(pth.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', pth)
