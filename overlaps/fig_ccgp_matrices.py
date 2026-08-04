"""Explicit cross-task generalization matrices (the per-condition-pair version of CCGP).

For each variable, a 3x3 matrix: train a linear decoder on trials of one TASK (DPA / DualGo / DualNoGo),
test on trials of another. Diagonal = within-task (cross-validated); off-diagonal = cross-task
generalization. Off-diag ≈ diag ⇒ the code is coded on the same axis in all three tasks = abstract /
reusable (the same low-dimensional geometry is re-used across the composed tasks).

sample @ late delay · choice @ test · test @ test.  Per mouse, averaged over the 9 mice (Expert).
Output: figures/overlaps/ccgp/{png,svg}/overlaps_ccgp_matrices.{png,svg}
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from src.pca.io import pkl_load
from src.common.options import set_options
from src.common import plot_utils  # noqa: sets poster context at import

# ── Style (Nature Neuroscience house style — matches the main figures) ──────────
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
TASKS, TLAB = ['DPA', 'DualGo', 'DualNoGo'], ['DPA', 'Go', 'NoGo']
o = set_options()
W_LD, W_TE = np.asarray(o['bins_LD']), np.asarray(o['bins_TEST'])
STAGE = 'Expert'
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))

print('loading pseudo-population …')
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
y = pkl_load('y_all_nan_', path='../data/pca')
VALID = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
                 path='../data/overlaps')['valid']

VARS = [('sample', 'sample_odor', W_LD), ('choice', 'choice', W_TE), ('test', 'test_odor', W_TE)]
if '--test' in sys.argv[1:]:                       # decode every code during the TEST epoch
    VARS = [(lab, col, W_TE) for lab, col, _ in VARS]; SUF, WINLAB = '_test', 'all codes @ TEST epoch'
else:
    SUF, WINLAB = '', 'sample @ late delay · choice/test @ TEST'


def population(mouse, win):
    val = VALID[(mouse, STAGE)]
    idx = ((y.mouse == mouse) & (y.learning == STAGE) & (y.laser == 0)).to_numpy()
    A = np.nanmean(X[idx][:, val, :][:, :, win], axis=2)
    A = np.where(np.isnan(A), np.nanmean(A, axis=0, keepdims=True), A)
    return A, y[idx].reset_index(drop=True)


def dprime(s, V):
    """|d'| of the (cross-)projected decision function — Fig-3's metric, per-mouse-scaled by construction."""
    a, b = s[V == 1], s[V == 0]
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-9))


def within_cv(A, V):
    n = np.bincount(V.astype(int)).min()
    if n < 3:
        return np.nan
    skf = StratifiedKFold(min(5, n), shuffle=True, random_state=0)
    s = np.full(len(V), np.nan)
    for tr, te in skf.split(A, V):                                   # held-out decision-function scores → d'
        s[te] = CLF().fit(A[tr], V[tr]).decision_function(A[te])
    return dprime(s, V)


def xtask_matrix(A, V, task):
    M = np.full((3, 3), np.nan)
    for i, ti in enumerate(TASKS):
        for j, tj in enumerate(TASKS):
            tr, te = (task == ti), (task == tj)
            if np.unique(V[tr]).size < 2 or np.unique(V[te]).size < 2:
                continue
            if min(np.bincount(V[tr].astype(int)).min(), np.bincount(V[te].astype(int)).min()) < 3:
                continue
            M[i, j] = within_cv(A[tr], V[tr]) if i == j else \
                dprime(CLF().fit(A[tr], V[tr]).decision_function(A[te]), V[te])   # train i → d' on test j
    return M


# per-mouse matrices → average
MATS = {lab: [] for lab, _, _ in VARS}
for lab, col, win in VARS:
    for m in ALL_MICE:
        A, d = population(m, win)
        V = d[col].to_numpy(float); task = d['tasks'].to_numpy()
        ok = np.isfinite(V) & pd.notna(task)
        M = xtask_matrix(A[ok], V[ok], task[ok])
        MATS[lab].append(M)
    Mm = np.nanmean(np.stack(MATS[lab]), axis=0)
    diag, off = np.nanmean(np.diag(Mm)), np.nanmean(Mm[~np.eye(3, dtype=bool)])
    print(f'{lab:7s} diag={diag:.3f} off-diag={off:.3f} gen-index={off/diag:.2f}\n{np.round(Mm,3)}')

# ── figure: one 3×3 matrix per variable ──────────────────────────────────────
Mms = {lab: np.nanmean(np.stack(MATS[lab]), axis=0) for lab, _, _ in VARS}
VMAX = max(np.nanmax(np.abs(m)) for m in Mms.values())              # shared d' scale across variables
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw=dict(wspace=0.55))
for ax, (lab, _, _) in zip(axes, VARS):
    Mm = Mms[lab]
    im = ax.imshow(Mm, cmap='RdBu_r', vmin=-VMAX, vmax=VMAX, aspect='equal')
    for i in range(3):
        for j in range(3):
            v = Mm[i, j]
            ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else '–', ha='center', va='center',
                    fontsize=7.5, color='k', fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(TLAB); ax.set_yticklabels(TLAB)
    ax.set_xlabel('test task'); ax.set_ylabel('train task')
    diag, off = np.nanmean(np.diag(Mm)), np.nanmean(Mm[~np.eye(3, dtype=bool)])
    ax.set_title(f'{lab} code   (off/diag = {off/diag:.2f})', loc='left', fontsize=TITLE_FS)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5); cb.set_label("d′", fontsize=7)
fig.suptitle(f'Cross-task generalization (decoder d′): a code trained on one task reads out in the others '
             f'— {WINLAB}', x=0.01, ha='left', y=1.04, fontsize=9)
OUT = 'figures/overlaps/ccgp'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
fig.savefig(f'{OUT}/png/overlaps_ccgp_matrices{SUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/overlaps_ccgp_matrices{SUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/overlaps_ccgp_matrices{SUF}.png'))
