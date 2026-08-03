"""Tell apart (A) a genuine 2D double-well vs (B) a slow 1D manifold carrying 2 wells.
Discriminator = behaviour BETWEEN the wells (not just at the fixed points):
  - SLOW MANIFOLD: persistent fast-transverse / slow-longitudinal Jacobian gap ALL ALONG the
    inter-well axis + low speed everywhere along it + near-flat saddle.
  - 2D DOUBLE WELL: gap only near the fixed points; speed rises between; no persistent marginal dir.
Method: fit the pooled rate-net flow, walk the inter-well axis, at each point take the local
Jacobian, split eigen-rates into longitudinal (along axis) vs transverse, and read the speed.
Rate = -ln|eig(I+J)|/dt  (1/s; >0 contracting, <0 expanding); dt = 1/6 s."""
import matplotlib; matplotlib.use('Agg')
import sys, warnings, os
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.pca.io import pkl_load
from src.pca.dynamics import fit_rnn_flow, flow_fixed_points

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

WHO = sys.argv[1] if len(sys.argv) > 1 else 'pooled'
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 0           # CI-removal order (0/1/2)
_root = 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test'
_ci = '' if Q == 0 else f'_ci{Q}'
DUM = f'{_root}{_ci}_dpca' if WHO == 'pooled' else f'{_root}_dpca_{WHO}{_ci}'
W = np.arange(18, 54); GAIN, RIDGE, DT, LIM = 1.0, 0.2, 1.0 / 6, 2.6

Z = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca')
y = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')
lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca')
i, j = lab.index('sample'), lab.index('sample:test')
m = ((y.laser == 0) & (y.learning == 'Expert') & (y.performance == 1)).to_numpy()
Z2 = Z[m][:, [i, j], :].astype(float); Z2 = Z2 / Z2.std((0, 2), keepdims=True) * 2.8
yc = y[m].reset_index(drop=True); dpa = (yc['tasks'] == 'DPA').to_numpy()
idx = {s: np.where(dpa & (yc['sample'] == s).to_numpy())[0] for s in (0, 1)}
means = [Z2[idx[s]][:, :, W].mean(0) for s in (0, 1)]
flow, _ = fit_rnn_flow(np.stack(means), gain=GAIN, ridge=RIDGE)
fps = flow_fixed_points(flow, bounds=[(-LIM, LIM), (-LIM, LIM)])
att = [p for p, k, _ in fps if k == 'attractor']; sad = [p for p, k, _ in fps if k == 'saddle']
y0 = np.mean([p[1] for p, _, _ in fps]) if fps else 0.0


def jac(p):
    e = 1e-4; J = np.zeros((2, 2))
    for k in range(2):
        d = np.zeros(2); d[k] = e
        J[:, k] = (flow((p + d)[:, None])[:, 0] - flow((p - d)[:, None])[:, 0]) / (2 * e)
    return J


xs = np.linspace(-LIM, LIM, 80)
speed, rate_long, rate_trans = [], [], []
for x in xs:
    p = np.array([x, y0])
    speed.append(np.linalg.norm(flow(p[:, None])[:, 0]))
    ev, V = np.linalg.eig(np.eye(2) + jac(p))
    rate = -np.log(np.abs(ev)) / DT                          # 1/s per eigenmode
    long_i = int(np.argmax(np.abs(V[0, :])))                 # eigvec most aligned with x = longitudinal
    rate_long.append(rate[long_i]); rate_trans.append(rate[1 - long_i])
speed = np.array(speed); rate_long = np.array(rate_long); rate_trans = np.array(rate_trans)

# --- summary discriminators over the inter-well segment (between the two outer attractors) ---
if len(att) >= 2:
    xa = sorted(p[0] for p in att); seg = (xs >= xa[0]) & (xs <= xa[-1])
else:
    seg = np.ones_like(xs, bool)
gap = rate_trans[seg] / np.maximum(np.abs(rate_long[seg]), 1e-3)     # transverse vs longitudinal
spd_ratio = speed[seg].mean() / speed.max()                          # low along axis?
print(f'[{WHO}]  attractors x = {sorted(round(p[0],2) for p in att)}  saddle x = {[round(p[0],2) for p in sad]}')
print(f'  transverse contraction rate (1/s): median {np.median(rate_trans[seg]):+.2f}  -> tau {1/abs(np.median(rate_trans[seg])):.2f}s')
print(f'  longitudinal rate (1/s):           median {np.median(rate_long[seg]):+.2f}  -> tau {1/max(abs(np.median(rate_long[seg])),1e-3):.2f}s')
print(f'  PERSISTENT GAP transverse/longitudinal: median {np.median(gap):.1f}x, min {np.min(gap):.1f}x')
print(f'  mean speed along inter-well axis / max speed = {spd_ratio:.2f}  (low => slow manifold)')
verdict = ('SLOW MANIFOLD + wells' if (np.median(gap) > 3 and spd_ratio < 0.5)
           else '2D double well' if np.median(gap) < 1.5 else 'intermediate')
print(f'  => {verdict}')

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].plot(xs, speed, 'k-', lw=2)
for p in att: ax[0].axvline(p[0], color='#2ca25f', lw=1.4, ls='--')
for p in sad: ax[0].axvline(p[0], color='#d1495b', lw=1.4, ls=':')
ax[0].set_xlabel('position along inter-well (sample) axis'); ax[0].set_ylabel('flow speed |dz|')
ax[0].set_title('Speed along the axis\n(low between wells => slow manifold; high => separate wells)', loc='left', fontsize=TITLE_FS)
ax[1].axhline(0, color='k', lw=.6)
ax[1].plot(xs, rate_trans, '-', color='#1f77b4', lw=2, label='transverse rate (onto axis)')
ax[1].plot(xs, rate_long, '-', color='#d62728', lw=2, label='longitudinal rate (along axis)')
for p in att: ax[1].axvline(p[0], color='#2ca25f', lw=1.2, ls='--')
for p in sad: ax[1].axvline(p[0], color='#d1495b', lw=1.2, ls=':')
ax[1].set_xlabel('position along inter-well (sample) axis'); ax[1].set_ylabel('contraction rate (1/s)')
ax[1].set_title('Jacobian rates along the axis\n(persistent transverse>>longitudinal gap => slow manifold)', loc='left', fontsize=TITLE_FS)
ax[1].legend(fontsize=6.5)
fig.suptitle(f'Slow-manifold vs 2D-double-well test [{WHO}]  —  verdict: {verdict}', y=1.02, fontsize=9)
fig.tight_layout()
out = f'figures/pseudo/flow/slowmanifold_test_{WHO}{_ci}.png'
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
print('saved', out)
