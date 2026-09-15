"""(RENUMBERED 2026-09-15 to citation order: this script draws Extended Data Fig. 6.)
fig_ed5_laser7.py — Extended Data Fig. 6: the acute laser ON−OFF coupling over all seven laser mice
(companion to Fig. 6g–i). Built 2026-09-15 (Leon: "keep only what is essential") — Results §6 cites it
once: "Computing the same coupling over all seven mice that received laser gives the same answer".

Same estimator as Fig. 6 (fig_behavior_opto_main.py, canonical build): laser-ON trials projected through
the laser-OFF-trained choice decoder (axis bins 54–62, the decision window), depth read at late delay
(bins 45–53), RAW units, DPA trials, expert stage; Δ = ON − OFF within mouse, equal-weight A/B pooling.
Five Jaws (inhibition) + two ChR2 (excitation) mice; ACC-implant mice have no laser trials. The pooled
rank test is sign-agnostic: it asks only whether a displacement of either sign tracks accuracy.

The seven per-mouse rows are cached (figures/overlaps/behavior/ed5_laser7_cache.pkl); --recompute reloads
the ~1 GB laser tensor.
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python fig_ed5_laser7.py [--nocap] [--recompute]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig6.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); sys.path.insert(0, '/home/leon/dual/pca')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import spearmanr, linregress
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 0.75     # 7.2-in canvas prints 1:1 at 183 mm, so 8 pt x 0.75 = 6 pt
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = PS*8
NOCAP = '--nocap' in sys.argv[1:]
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
JAWS = ALL_MICE[:5]; CHR = ALL_MICE[5:7]; LASER_MICE = JAWS + CHR
GROUP = {**{m: 'Jaws' for m in JAWS}, **{m: 'ChR' for m in CHR}}
GMARK = {'Jaws': 'o', 'ChR': '^'}
MC = dict(zip(ALL_MICE, sns.color_palette('tab10', n_colors=len(ALL_MICE))))
CACHE = 'figures/overlaps/behavior/ed5_laser7_cache.pkl'

if os.path.exists(CACHE) and '--recompute' not in sys.argv[1:]:
    rows = pickle.load(open(CACHE, 'rb'))
else:
    from src.common.options import set_options
    from src.pca.io import pkl_load
    DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice'
    print('loading recorded laser tensor …', flush=True)
    X = pkl_load(f'X_{DUM}', path='../data/overlaps'); y = pkl_load(f'labels_{DUM}', path='../data/overlaps')
    options = set_options(
        mice=LASER_MICE, tasks=['Dual'], mouse=LASER_MICE[0], laser=0,
        trials='', data_type='dF', prescreen=None, pval=0.05,
        preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
        random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
        bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
        class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
        days=['first', 'last'])
    BINS_LATE = np.asarray(options['bins_LD'])                    # 45–53
    DEPTH_AXIS = np.arange(54, 63)                                # canonical decision-window axis (2026-09-08)
    Xe = X[..., DEPTH_AXIS, :].mean(-2)[:, 1].astype(float)       # (n, 84) over test time, RAW units
    depth_all = Xe[:, BINS_LATE].mean(1); del X
    is_choice = (y.target == 'choice').values; is_dpa = (y.tasks == 'DPA').values; EXP = (y.stage == 'Expert').values

    def _pooled_depth(mmask, laser_val):
        vals = []
        for pairs in ([0, 1], [2, 3]):
            m = mmask & is_choice & is_dpa & EXP & (y.laser == laser_val).values & y.odor_pair.isin(pairs).values
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
        rows.append(dict(mouse=mouse, group=GROUP[mouse],
                         d_depth=_pooled_depth(mmask, 1) - _pooled_depth(mmask, 0),
                         d_dpa=(_perf_mean('performance', (y.tasks == 'DPA') & (y.mouse == mouse), 1)
                                - _perf_mean('performance', (y.tasks == 'DPA') & (y.mouse == mouse), 0)),
                         d_gng=(_perf_mean('odr_perf', (y.tasks != 'DPA') & (y.mouse == mouse), 1)
                                - _perf_mean('odr_perf', (y.tasks != 'DPA') & (y.mouse == mouse), 0))))
    pickle.dump(rows, open(CACHE, 'wb')); print('cached', CACHE)

R = pd.DataFrame(rows)
for _, r in R.iterrows():
    print(f'  {r.mouse:9s} {r.group:5s} Δdepth={r.d_depth:+.3f} ΔDPA={r.d_dpa:+.3f} ΔGNG={r.d_gng:+.3f}')


def regression_band(ax, xs, ys, color='0.25'):
    slope, icpt, _, _, se = linregress(xs, ys)
    xl = np.linspace(xs.min(), xs.max(), 100); yl = slope * xl + icpt
    seb = se * np.sqrt(1 / len(xs) + (xl - xs.mean()) ** 2 / np.sum((xs - xs.mean()) ** 2))
    ax.plot(xl, yl, '-', color=color, lw=1.2, zorder=2)
    ax.fill_between(xl, yl - 1.96 * seb, yl + 1.96 * seb, color=color, alpha=0.10, lw=0, zorder=1)


fig = plt.figure(figsize=(7.2, 3.1))
gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.11, right=0.985, top=0.86, bottom=0.19)
axes = []
for k, (col, ylab, ttl) in enumerate([('d_dpa', 'Δ DPA accuracy (ON − OFF), DPA trials', 'DPA arm'),
                                       ('d_gng', 'Δ GNG accuracy (ON − OFF), dual trials', 'GNG arm')]):
    ax = fig.add_subplot(gs[0, k]); axes.append(ax)
    xs, ys = R.d_depth.to_numpy(), R[col].to_numpy()
    regression_band(ax, xs, ys)
    for _, r in R.iterrows():
        ax.scatter(r.d_depth, r[col], s=36, color=MC[r.mouse], marker=GMARK[r.group], edgecolors='w', linewidths=0.6, zorder=4)
    rho, p = spearmanr(xs, ys); sig = p < 0.05
    J = R[R.group == 'Jaws']; rj, pj = spearmanr(J.d_depth, J[col])
    ax.text(0.97, 0.96, '∗' if sig else 'n.s.', transform=ax.transAxes, ha='right', va='top',
            fontsize=PS*(12 if sig else 8), fontweight='bold', color='k' if sig else '0.55')
    ax.text(0.03, 0.04, f'all 7: ρ = {rho:+.2f}, p = {p:.3f}\nJaws only: ρ = {rj:+.2f}, p = {pj:.2f}', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=PS*6.5, color='0.3')
    ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.axvline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('Δ choice-code depth (ON − OFF), DPA trials'); ax.set_ylabel(ylab)
    ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
    print(f'{ttl}: rho={rho:+.3f} p={p:.4f} n={len(xs)} | Jaws only rho={rj:+.3f} p={pj:.3f} n={len(J)}')
axes[1].legend(handles=[mlines.Line2D([0], [0], marker='o', color='0.4', ls='none', ms=5, label='Jaws, inhibition (n = 5)'),
                        mlines.Line2D([0], [0], marker='^', color='0.4', ls='none', ms=5, label='ChR2, excitation (n = 2)')],
               frameon=False, fontsize=PS*6.0, loc='lower left', bbox_to_anchor=(0.0, 0.16))
for ax, s in zip(axes, 'ab'):
    ax.text(-0.22, 1.05, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')

CAP = [
    'Extended Data Fig. 6 | The acute laser ON−OFF coupling over all seven laser mice (companion to Fig. 6g–i). '
    'The within-mouse change in choice-code depth under laser against the change in accuracy, one point per mouse, '
    'for every mouse carrying interleaved laser trials (five Jaws inhibition, circles; two ChR2 excitation, triangles; '
    'the ACC-implant mice received no laser); same axis, window and units as Fig. 6 (expert stage, DPA trials, depth '
    'read at late delay on the laser-OFF-trained choice axis). Because the two opsins move the state in opposite '
    'directions, the pooled rank test asks only whether a displacement of either sign tracks the change in accuracy; '
    'each panel also gives the five Jaws mice alone. a, DPA arm: the trend over seven mice (ρ = +0.71, p = .074) is '
    'carried by the two ChR2 mice and is absent in the Jaws mice (ρ = +0.21, p = .74), so it is not read as a coupling. '
    'b, GNG arm: robust over seven (ρ = −0.94, p = .002) and of the same sign and size in the Jaws mice alone '
    '(ρ = −0.82, p = .089, n = 5). Shaded, 95% band of a linear fit drawn for orientation; the statistic is the rank '
    'correlation. Two Jaws mice at ceiling in both tasks show no change in either accuracy (the two points on the '
    'zero line).',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig6.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig6.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig6.png')
