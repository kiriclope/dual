"""Is a RANK-2 dynamics good enough? — dPCA latent space (the right substrate, D=8).

With dt=1 bin the discrete transition z_{t+1}=A z_t + b has A ≈ the recurrent connectivity, so
constraining rank(A)≤R (reduced-rank regression) tests the CONNECTIVITY RANK directly. We sweep R
and score HELD-OUT multi-step predictive R² (CV over trials); if it plateaus at R=2, a rank-2
dynamics captures the latent flow as well as full rank ⇒ rank-2 is good enough.

Reported for the full 8-D latent and for the task subspace (sample + sample:test = the WM/choice
plane). Prints R²(R) and the rank at which it reaches 95% / 99% of the full-rank value.
Usage: exp_rank_dpca.py [Naive|Expert] [horizon]
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from src.pca.io import pkl_load

STAGE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('Naive', 'Expert') else 'Expert'
HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DUM = f'pseudo_ALL_{STAGE}_zscore_5x1_scale_blcenter_f-sample-test_dpca'
Z = pkl_load(f'pseudo_traj_{DUM}', path='../data/pca')
y = pkl_load(f'pseudo_labels_{DUM}', path='../data/pca')
lab = pkl_load(f'pseudo_marglabels_{DUM}', path='../data/pca')
m = ((y.laser == 0) & (y.learning == STAGE) & (y.performance == 1)).to_numpy()
Zc = Z[m].astype(float)                                            # (n, 8, T)
TASK = [i for i, L in enumerate(lab) if L in ('sample', 'sample:test')]   # WM/choice subspace
print(f'{STAGE}: Z {Zc.shape}  marglabels {lab}  task-dims {TASK}')


def fit_lds_rank(Ztr, R, ridge=1e-2):
    """z_{t+1} = A z_t + b with rank(A) ≤ R (reduced-rank regression)."""
    d = Ztr.shape[1]
    X = Ztr[:, :, :-1].transpose(0, 2, 1).reshape(-1, d).T         # (d, M) state
    Y = Ztr[:, :, 1:].transpose(0, 2, 1).reshape(-1, d).T          # (d, M) next
    xm = X.mean(1, keepdims=True); ym = Y.mean(1, keepdims=True)
    Xc = X - xm; Yc = Y - ym
    Afull = (Yc @ Xc.T) @ np.linalg.inv(Xc @ Xc.T + ridge * np.eye(d))
    U, S, Vt = np.linalg.svd(Afull @ Xc, full_matrices=False)      # rank-R proj of the fitted output
    A = (U[:, :R] @ U[:, :R].T) @ Afull
    return A, (ym - A @ xm).ravel()


def cv_r2_rank(Zd, R, horizon=HORIZON, n_splits=5, seed=0):
    T = Zd.shape[2]; kf = KFold(n_splits, shuffle=True, random_state=seed); num = den = 0.0
    for tr, te in kf.split(np.arange(Zd.shape[0])):
        A, b = fit_lds_rank(Zd[tr], R)
        zt = Zd[te][:, :, :T - horizon]
        for _ in range(horizon):
            zt = np.einsum('ij,njt->nit', A, zt) + b[None, :, None]
        tgt = Zd[te][:, :, horizon:]
        num += float(((tgt - zt) ** 2).sum()); den += float(((tgt - tgt.mean()) ** 2).sum())
    return 1 - num / den


def sweep(Zd, name):
    D = Zd.shape[1]; ranks = list(range(1, D + 1))
    r2 = np.array([cv_r2_rank(Zd, R) for R in ranks])
    full = r2[-1]
    def rank_at(frac):
        hit = np.where(r2 >= frac * full)[0]
        return ranks[hit[0]] if len(hit) else D
    print(f'\n[{name}] held-out {HORIZON}-step R² vs rank(A):')
    for R, v in zip(ranks, r2):
        print(f'   rank {R}: R²={v:+.3f}' + ('   <- rank-2' if R == 2 else '') + ('   (full)' if R == D else ''))
    print(f'   full-rank R²={full:+.3f};  rank-2 reaches {100*r2[1]/full:.0f}% of full;  '
          f'≥95% at rank {rank_at(0.95)}, ≥99% at rank {rank_at(0.99)}')
    return ranks, r2, full


os.makedirs('figures/pseudo/flow/png', exist_ok=True); os.makedirs('figures/pseudo/flow/svg', exist_ok=True)
fig, ax = plt.subplots(figsize=(5.5, 4))
for Zd, name, col in [(Zc, 'full 8-D', '#2171b5'), (Zc[:, TASK, :], f'task {len(TASK)}-D (sample+choice)', '#cc6677')]:
    ranks, r2, full = sweep(Zd, name)
    ax.plot(ranks, r2 / full, '-o', color=col, label=name)
ax.axvline(2, ls='--', color='k', lw=0.8); ax.axhline(0.95, ls=':', color='0.6', lw=0.8)
ax.set_xlabel('rank of A (connectivity)'); ax.set_ylabel(f'held-out {HORIZON}-step R²  (/ full-rank)')
ax.set_title(f'Is rank-2 good enough? — dPCA latents, {STAGE}'); ax.legend(frameon=False, fontsize=9)
ax.set_ylim(top=1.02); fig.tight_layout()
p = f'figures/pseudo/flow/png/rank_sufficiency_{STAGE}.png'
fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('\nsaved', p)
