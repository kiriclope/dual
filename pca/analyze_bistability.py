"""Per-mouse bistability of the autonomous WM flow: dynamical depth (barrier = saddle
unstable |eig(I+J)|, well τ) + reproducibility (bootstrap P(2 att)) + sample-memory d'.
Nonlinear rate-network flow on the DPA-delay sample-grouped condition means. Summary figure.
"""
import matplotlib; matplotlib.use('Agg')
import sys, warnings, os
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.random.seed(0)                                  # reproducible bootstrap
import seaborn as sns
import matplotlib.pyplot as plt
from src.pca.io import pkl_load
from src.pca.dynamics import fit_rnn_flow, flow_fixed_points, bootstrap_fixed_points

# ── Style (Nature Neuroscience house style — matches the main figures) ──────────
sns.set_context('notebook')          # MUST come after importing src.common.plot_utils (sets "poster")
sns.set_style('ticks')
plt.rcParams.update({                 # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BASE = 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca'
W, WL, DT, BOX = np.arange(21, 54), np.arange(42, 54), 1 / 6, [(-2.5, 2.5), (-3.2, 3.2)]
RUNS = [('pooled', BASE, None)] + [(mo, f'{BASE}_{mo}', 2.8) for mo in MICE]


def analyze(dum, rescale):
    Z = pkl_load(f'pseudo_traj_{dum}', path='../data/pca')
    y = pkl_load(f'pseudo_labels_{dum}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{dum}', path='../data/pca')
    i, j = lab.index('sample'), lab.index('sample:test')
    m = ((y.laser == 0) & (y.learning == 'Expert') & (y.performance == 1)).to_numpy()
    Z2 = Z[m][:, [i, j], :].astype(float)
    if rescale:
        Z2 = Z2 / Z2.std((0, 2), keepdims=True) * rescale
    yc = y[m].reset_index(drop=True); dpa = (yc['tasks'] == 'DPA').to_numpy()
    gA = dpa & (yc['sample'] == 0).to_numpy(); gB = dpa & (yc['sample'] == 1).to_numpy()
    # sample-memory d' (late delay, z-scored)
    sa = (Z2[:, 0, :] - Z2[:, 0, :].mean()) / Z2[:, 0, :].std()
    a, b = sa[gA][:, WL].mean(1), sa[gB][:, WL].mean(1)
    dprime = abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))
    # nonlinear flow + fixed points
    means = np.stack([Z2[g][:, :, W].mean(0) for g in (gA, gB)])
    flow, _ = fit_rnn_flow(means, gain=1.0, ridge=0.2)
    fps = flow_fixed_points(flow, BOX, n_seed=21)
    natt = sum(1 for _, k, _ in fps if k == 'attractor')
    barrier = max([np.abs(ev).max() for _, k, ev in fps if k == 'saddle'], default=np.nan)
    well_tau = max([(-1 / (np.log(np.abs(ev).min()) * DT)) for _, k, ev in fps if k == 'attractor'], default=np.nan)
    idx = [np.where(g)[0] for g in (gA, gB)]
    resamp = lambda: np.stack([Z2[np.random.choice(g, len(g), True)][:, :, W].mean(0) for g in idx])
    p2 = bootstrap_fixed_points(resamp, BOX, gain=1.0, ridge=0.2, n_boot=120)['p2']
    return dict(dprime=dprime, natt=natt, barrier=barrier, well_tau=well_tau, p2=p2)

R = {}
print(f"{'run':9} d'    natt  barrier|λ|  wellτ(s)  P2")
for name, dum, rs in RUNS:
    R[name] = analyze(dum, rs)
    r = R[name]
    print(f"{name:9} {r['dprime']:.2f}   {r['natt']}    {r['barrier']:.3f}      {r['well_tau']:.1f}    {r['p2']:.2f}")

# ── summary figure ───────────────────────────────────────────────────────────────
names = MICE
dp = np.array([R[n]['dprime'] for n in names]); p2 = np.array([R[n]['p2'] for n in names])
bar = np.array([R[n]['barrier'] for n in names])
col = np.where(p2 >= 0.5, '#44AA99', '#bbbbbb')
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

o = np.argsort(-p2)
ax[0].bar(range(9), p2[o], color=col[o]); ax[0].axhline(0.5, color='k', ls=':', lw=0.8)
ax[0].set_xticks(range(9)); ax[0].set_xticklabels(np.array(names)[o], rotation=60, fontsize=7)
ax[0].set_ylabel('P(2 attractors)  [bootstrap]'); ax[0].set_title('A  bistability survey (9 mice)', loc='left', fontsize=TITLE_FS)

ax[1].scatter(dp, p2, c=col, s=90, edgecolor='k', zorder=3)
ax[1].scatter(R['pooled']['dprime'], R['pooled']['p2'], marker='*', s=320, c='#332288', edgecolor='k', zorder=4, label='pooled')
for n, x, yv in zip(names, dp, p2):
    ax[1].annotate(n, (x, yv), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax[1].set_xlabel("sample-memory d'  (late delay)"); ax[1].set_ylabel('P(2 attractors)')
ax[1].set_title('B  bistability tracks sample memory', loc='left', fontsize=TITLE_FS); ax[1].legend(fontsize=6.5)

bi = p2 >= 0.5
ax[2].scatter(dp[bi], bar[bi], c='#44AA99', s=90, edgecolor='k', zorder=3)
for n, x, yv in zip(np.array(names)[bi], dp[bi], bar[bi]):
    ax[2].annotate(n, (x, yv), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax[2].axhline(1.0, color='k', ls=':', lw=0.8)
ax[2].set_xlabel("sample-memory d'"); ax[2].set_ylabel('barrier  |eig(I+J)| at saddle  (>1)')
ax[2].set_title('C  barrier depth (bistable mice)', loc='left', fontsize=TITLE_FS)
fig.suptitle('Autonomous WM bistability — nonlinear rate-flow fixed-point analysis', y=1.02, fontsize=9)
fig.tight_layout()
out = 'figures/pseudo/flow/bistability_summary.png'
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
