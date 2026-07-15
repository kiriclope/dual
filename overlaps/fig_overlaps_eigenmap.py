"""SUPPLEMENT — per-code spectral segmentation on the RAW cosine maps ("eigen map").

For each code we eigendecompose the raw 84x84 discriminant-axis cosine matrix M directly (M is a Gram
matrix of unit axes -> already PSD; NO Laplacian normalization, which would manufacture sinusoidal
eigenvectors on a smooth matrix). The eigenvalue spectrum + eigengap give the natural number of
temporal blocks K; the best contiguous K-partition is read off the spectral embedding (DP on the top-K
eigenvectors, minimizing within-segment L2 variance).

Result: gng is the only code with genuine multi-block structure (eigengap K=3; a visible two-block map)
and its boundary sits at the DISTRACTOR; sample/choice/test are smooth drift (K=2) with their single
robust boundary at their driving event (test onset / response). The union of the per-code boundaries is
{distractor, test} — the anchors of the composite figure's 3 read windows. Pooled Naive+Expert.

Decoder variant mirrors the main figure: default ridge; --l1 / --lda load the matching bundled weights.
Output: figures/overlaps/cosine/{png,svg}/overlaps_eigenmap[_l1|_lda].{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

L1  = '--l1'  in sys.argv[1:]
LDA = '--lda' in sys.argv[1:]
CORRECT = '--all' not in sys.argv[1:]     # default correct-only; --all uses all-trials decoders
CTAG = '_correct' if CORRECT else ''
TSET = 'correct' if CORRECT else 'all'
if L1:
    DUM, PFX, SUF, DLAB = f'log_generalizing_overlaps_none{CTAG}_l1_ratio_1.0', 'l1_', '_l1', 'L1 (lasso)'
elif LDA:
    DUM, PFX, SUF, DLAB = f'log_generalizing_overlaps_none{CTAG}_lda',         'lda_', '_lda', 'shrinkage-LDA'
else:
    DUM, PFX, SUF, DLAB = f'log_generalizing_overlaps_none{CTAG}_l1_ratio_0.0', '',    '',    'ridge (logistic L2)'
DLAB = f'{DLAB} · {TSET} trials'
BDUM = f'{DUM}_raw_targets_choice-gng-sample-test'

DATA_IN = '../data/overlaps'
MICE = ['JawsM01','JawsM06','JawsM12','JawsM15','JawsM18','ChRM04','ChRM23','ACCM03','ACCM04']
EP = [('stim',12,18),('eDelay',18,27),('distr',27,33),('mDelay',33,39),('cue',39,42),
      ('gng rwd',42,45),('lDelay',45,54),('test',54,60),('resp',60,72),('dpa rwd',72,84)]
xtime = np.linspace(0, 14, 84)
CODES = ['sample', 'choice', 'test', 'gng']

VDIR = 'l1' if L1 else 'lda' if LDA else 'l2'   # decoder variant → own subfolder
OUT = f'figures/overlaps/cosine/{TSET}/{VDIR}'  # trial set (correct|all) → variant
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

W = pkl_load(f'{PFX}weights_{BDUM}', path=DATA_IN)['weights']


def unit(ws):
    ws = np.asarray(ws, float); n = np.linalg.norm(ws, axis=1, keepdims=True); n[n == 0] = 1; return ws/n
def raw_cos(st, tg):
    S = [unit(W[(m, st, 'all', tg)]) for m in MICE if (m, st, 'all', tg) in W]
    return np.nanmean([U @ U.T for U in S], 0)
def pooled(tg):
    return np.nanmean([raw_cos('Naive', tg), raw_cos('Expert', tg)], 0)
def label_bin(b):
    for nm, a, c in EP:
        if a <= b < c: return nm
    return 'base' if b < 12 else EP[-1][0]


def dp_partition(U, K):
    """Best contiguous K-partition of rows of U minimizing within-segment SSE (L2)."""
    n = len(U); pre = np.vstack([np.zeros(U.shape[1]), np.cumsum(U, 0)])
    pre2 = np.concatenate([[0], np.cumsum((U ** 2).sum(1))])
    def sse(a, b):
        cnt = b - a + 1; s = pre[b + 1] - pre[a]; return float(pre2[b + 1] - pre2[a] - (s @ s) / cnt)
    INF = 1e18; C = np.full((K + 1, n), INF); bk = np.full((K + 1, n), -1, int)
    for b in range(n): C[1][b] = sse(0, b)
    for k in range(2, K + 1):
        for b in range(k - 1, n):
            for a in range(k - 1, b + 1):
                v = C[k - 1][a - 1] + sse(a, b)
                if v < C[k][b]: C[k][b] = v; bk[k][b] = a
    segs = []; b = n - 1; k = K
    while k > 0:
        a = bk[k][b] if k > 1 else 0; segs.append((a, b)); b = a - 1; k -= 1
    return segs[::-1]


fig, axs = plt.subplots(2, 4, figsize=(19, 8.5), gridspec_kw={'height_ratios': [3, 1.5]})
print(f'Per-code spectral segmentation on RAW cosine maps  [{DLAB}]:')
for j, c in enumerate(CODES):
    M = pooled(c)
    w, V = np.linalg.eigh(M); w = np.clip(w[::-1], 0, None); V = V[:, ::-1]
    pct = 100 * w / w.sum()
    gaps = [w[k - 1] - w[k] for k in range(1, 7)]
    Kstar = int(np.argmax(gaps[1:5]) + 2)
    U = V[:, 1:max(Kstar, 2)] * np.sqrt(w[1:max(Kstar, 2)])[None, :]
    segs = dp_partition(U, Kstar)
    names = [f'{label_bin(s)}-{label_bin(e)} ({xtime[s]:.1f}-{xtime[min(e,83)]:.1f}s)' for s, e in segs]
    print(f'  {c:>7}: eigengap K={Kstar}  |  ' + '  |  '.join(names))
    ax = axs[0][j]
    ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r', origin='upper', extent=[0, 14, 14, 0], aspect='equal')
    for s, e in segs:
        ax.add_patch(Rectangle((xtime[s], xtime[s]), xtime[min(e,83)]-xtime[s], xtime[min(e,83)]-xtime[s],
                               fill=False, ec='k', lw=2.2))
    for _, a, cc in EP:
        ax.axvline(xtime[a], color='0.4', lw=0.4, alpha=0.5); ax.axhline(xtime[a], color='0.4', lw=0.4, alpha=0.5)
    ax.set_title(f'{c}  (raw map, K*={Kstar})', fontsize=11, fontweight='bold')
    ax.set_xlabel('time (s)', fontsize=8); ax.tick_params(labelsize=7)
    axe = axs[1][j]
    for r in range(1, 4):
        axe.plot(xtime, V[:, r], lw=1.6, label=f'eig {r+1} (λ%={pct[r]:.0f})')
    for s, e in segs[:-1]:
        axe.axvline(xtime[e], color='k', ls='--', lw=1.0)
    axe.axhline(0, color='0.6', lw=0.5); axe.set_xlabel('time (s)', fontsize=8); axe.tick_params(labelsize=7)
    axe.legend(fontsize=6.5, frameon=False, loc='upper right')
    if j == 0: axe.set_ylabel('eigenvector', fontsize=8)

fig.suptitle(f'Per-code spectral segmentation on the RAW cosine maps (eigendecompose M; K from eigengap; '
             f'DP partition) — pooled Naive+Expert   ·   {DLAB}', fontsize=11, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
stem = f'overlaps_eigenmap'
p = os.path.join(OUT, 'png', f'{stem}.png')
fig.savefig(p, dpi=300, bbox_inches='tight')
fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
