"""Task-computation rank test — do the WM/choice STATES live on a 2-D manifold? (dPCA latents)

The predictive-sufficiency test (exp_rank_dpca.py) says the full latent DYNAMICS is higher-rank
(rank-2 = 62-67% of full-rank multi-step R²). But that counts test coding, the CI time-ramp and fast
transients against rank-2. The task-computation question is narrower: does the WM/choice COMPUTATION —
the geometry the flow portrays — fit in 2-D? A velocity/dynamics-fit criterion FAILS here (the delay is
near-stationary → velocity R² is at the noise floor; over the trial it is input-driven), so we ask the
well-posed GEOMETRIC version: the dimensionality of the condition-mean (sample×test) trajectory manifold
— variance by PC, top-2 fraction, and participation ratio PR=(Σλ)²/Σλ². PR≈2 ⇒ the task states are 2-D
⇒ rank-2 is good enough for the STATE GEOMETRY (not for predicting the full dynamics).
Usage: exp_rank_task.py
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.pca.io import pkl_load

WIN = np.arange(12, 72)                                            # stimulus → response (task-active window)
os.makedirs('figures/pseudo/flow/png', exist_ok=True); os.makedirs('figures/pseudo/flow/svg', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, STAGE in zip(axes, ('Expert', 'Naive')):
    DUM = f'pseudo_ALL_{STAGE}_zscore_5x1_scale_blcenter_f-sample-test_dpca'
    Z = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca'); y = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')
    lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca')
    m = ((y.laser == 0) & (y.learning == STAGE) & (y.performance == 1)).to_numpy()
    Zc = Z[m].astype(float); yc = y[m].reset_index(drop=True)
    samp = yc['sample'].to_numpy(); test = yc['test'].to_numpy()
    subs = {'WM sample+choice': [i for i, L in enumerate(lab) if L in ('sample', 'sample:test')],
            'task (no time)': [i for i, L in enumerate(lab) if L != 'time'],
            'all 8-D': list(range(8))}
    print(f'\n=== {STAGE}: dimensionality of the condition-mean (sample×test) trajectory manifold ===')
    for nm, dd in subs.items():
        M = [Zc[np.ix_(np.where((samp == s) & (test == t))[0], dd, WIN)].mean(0).T
             for s in (0, 1) for t in (0, 1)]                     # 4 cond means, each (T, |dd|)
        X = np.vstack(M); X = X - X.mean(0)
        ev = np.linalg.svd(X, full_matrices=False)[1] ** 2; ev /= ev.sum()
        pr = float(ev.sum() ** 2 / (ev ** 2).sum()); top2 = float(ev[:2].sum())
        print(f'  {nm:18s} (D={len(dd)}): var/PC={np.round(ev[:4], 2)}  top2={top2:.0%}  PR={pr:.2f}')
        ax.plot(range(1, len(ev) + 1), np.cumsum(ev), '-o', label=f'{nm} (PR {pr:.1f})')
    ax.axhline(0.9, ls=':', color='0.6', lw=0.8); ax.axvline(2, ls='--', color='k', lw=0.8)
    ax.set_xlabel('# PCs of the condition-mean manifold'); ax.set_title(STAGE)
    ax.legend(frameon=False, fontsize=8); ax.set_xlim(1, 8)
axes[0].set_ylabel('cumulative variance explained')
fig.suptitle('Task states live on a ~2-D manifold (top-2 PCs ≈ 82–94%) → rank-2 OK for the state geometry', y=1.02)
fig.tight_layout()
p = 'figures/pseudo/flow/png/rank_task_manifold.png'
fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('\nsaved', p)
