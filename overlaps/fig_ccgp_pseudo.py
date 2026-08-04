"""Pseudo-population CCGP (geometry-of-abstraction) — the boosted version.

Fixes vs the per-mouse CCGP: (i) pseudo-population pools all 3,319 neurons (removes the ~184-neuron
per-mouse ceiling); (ii) correct trials only; (iii) each code at a strong window; (iv) clean, on-thesis
generalization contexts — sample across TEST, choice/GNG/test across SAMPLE (avoids the distractor-
confounded across-task axis). Leakage-free: within-context "decode" uses disjoint train/test halves; CCGP
trains on one context value, tests on the held-out value. Balanced accuracy (chance 0.5). Null = shuffle.

Output: figures/overlaps/ccgp/{png,svg}/overlaps_ccgp_pseudo.{png,svg}
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from src.pca.io import pkl_load
from src.common.options import set_options
from src.common import plot_utils  # noqa

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGE, K, R, NB = 'Expert', 80, 20, 200
RNG = np.random.RandomState(0)
o = set_options()
W = {'ED': np.arange(18, 30), 'LD': np.asarray(o['bins_LD']),
     'MD': np.asarray(o['bins_MD']), 'TE': np.asarray(o['bins_TEST'])}

print('loading pseudo-population …')
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
y = pkl_load('y_all_nan_', path='../data/pca')
VALID = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
                 path='../data/overlaps')['valid']
CORR = ((y.learning == STAGE) & (y.laser == 0) & (y.performance == 1)).to_numpy()
MOUSE = y.mouse.to_numpy(); DUAL = y.tasks.isin(['DualGo', 'DualNoGo']).to_numpy()
print('pre-averaging activity per window …')
AW = {k: np.nanmean(X[:, :, w], axis=2) for k, w in W.items()}
del X

# (label, V-array, context-array, window, base-mask) — each generalized across a MEANINGFUL context:
#   sample/test across TASK (does the code survive the distractor), choice/GNG across SAMPLE (action &
#   distractor axes abstract w.r.t. memory = the sample⟂action factorisation).
SPECS = [
    ('sample', y.sample_odor.to_numpy(float), y.tasks.to_numpy(),             'LD', CORR),
    ('choice', y.choice.to_numpy(float),      y.sample_odor.to_numpy(float),  'TE', CORR),
    ('GNG',    (y.tasks == 'DualGo').to_numpy(float), y.sample_odor.to_numpy(float), 'MD', CORR & DUAL),
    ('test',   y.test_odor.to_numpy(float),   y.tasks.to_numpy(),             'TE', CORR),
]
PIPE = lambda: make_pipeline(StandardScaler(), PCA(n_components=20, random_state=0),
                             LogisticRegression(C=1.0, max_iter=3000))


def idx_of(base, V, C, v, c):
    keep = base & (V == v) & (C == c)
    return {m: np.where(keep & (MOUSE == m))[0] for m in ALL_MICE}


def pseudo(cidx, wkey):
    A = AW[wkey]; P = np.full((K, A.shape[1]), np.nan)
    for m in ALL_MICE:
        ix = cidx[m]
        if len(ix):
            P[:, VALID[(m, STAGE)]] = A[RNG.choice(ix, K, replace=True)][:, VALID[(m, STAGE)]]
    return np.where(np.isnan(P), 0.0, P)


def halves(cidx):
    a, b = {}, {}
    for m, ix in cidx.items():
        p = RNG.permutation(ix); k = len(p) // 2; a[m], b[m] = p[:k], p[k:]
    return a, b


Y2 = np.r_[np.zeros(K), np.ones(K)]


def score(base, V, C, wkey, shuffle=False):
    """CCGP = mean cross-context generalization over all ordered context pairs (2- or 3-valued context);
    decode = within-context balanced acc on a disjoint half (leakage-free reference)."""
    Vs = RNG.permutation(V) if shuffle else V
    vals = sorted([c for c in set(C.tolist()) if c == c], key=str)          # drop NaN (c==c), stable order
    cc, dec = [], []
    for ct in vals:
        clf = PIPE().fit(np.vstack([pseudo(idx_of(base, Vs, C, 0, ct), wkey),
                                    pseudo(idx_of(base, Vs, C, 1, ct), wkey)]), Y2)
        for ce in vals:
            if ce == ct:
                continue
            te = np.vstack([pseudo(idx_of(base, Vs, C, 0, ce), wkey), pseudo(idx_of(base, Vs, C, 1, ce), wkey)])
            cc.append(balanced_accuracy_score(Y2, clf.predict(te)))                    # CCGP: train ct → test ce
        h0a, h0b = halves(idx_of(base, Vs, C, 0, ct)); h1a, h1b = halves(idx_of(base, Vs, C, 1, ct))
        clf2 = PIPE().fit(np.vstack([pseudo(h0a, wkey), pseudo(h1a, wkey)]), Y2)        # decode: within-context split
        dec.append(balanced_accuracy_score(Y2, clf2.predict(np.vstack([pseudo(h0b, wkey), pseudo(h1b, wkey)]))))
    return float(np.mean(cc)), float(np.mean(dec))


rows = []
for lab, V, C, wk, base in SPECS:
    ccs = [score(base, V, C, wk)[0] for _ in range(R)]
    decs = [score(base, V, C, wk)[1] for _ in range(R)]
    nul = [score(base, V, C, wk, shuffle=True)[0] for _ in range(NB // 10)]
    rows.append(dict(var=lab, ccgp=np.mean(ccs), ccgp_sd=np.std(ccs), decode=np.mean(decs), null=np.mean(nul),
                     null_hi=np.percentile(nul, 95)))
    print(f'{lab:7s} CCGP={np.mean(ccs):.3f}±{np.std(ccs):.3f}  decode={np.mean(decs):.3f}  '
          f'null={np.mean(nul):.3f} (95%={np.percentile(nul,95):.3f})')

# ── figure ────────────────────────────────────────────────────────────────────
ORDER = ['sample', 'choice', 'GNG', 'test']
CTX = {'sample': 'task', 'choice': 'sample', 'GNG': 'sample', 'test': 'task'}
D = {r['var']: r for r in rows}
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.5, 1.0], wspace=0.42))
axA.axhline(0.5, ls=':', color='0.6', lw=0.8)
for i, v in enumerate(ORDER):
    r = D[v]
    axA.plot([i - 0.3, i + 0.3], [r['null'], r['null']], color='0.8', lw=6, alpha=0.6, solid_capstyle='butt',
             label='shuffle null (95%)' if i == 0 else None, zorder=1)
    axA.bar(i, r['ccgp'] - 0.5, bottom=0.5, width=0.5, color='#4477AA', alpha=0.85, zorder=2)
    axA.errorbar(i, r['ccgp'], yerr=r['ccgp_sd'], color='k', lw=1.0, capsize=2, zorder=3)
    sig = r['ccgp'] > r['null_hi']
    axA.text(i, 0.965, '∗' if sig else 'n.s.', ha='center', va='top',
             fontsize=13 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    axA.text(i, 0.452, f'/{CTX[v]}', ha='center', va='bottom', fontsize=6, color='0.4')
axA.set_xticks(range(4)); axA.set_xticklabels(ORDER); axA.set_ylim(0.45, 1.0)
axA.set_ylabel('CCGP  (cross-context generalization,\nbalanced acc., pseudo-population)')
axA.set_title('Codes generalize across context (abstract)', loc='left', fontsize=TITLE_FS)
axA.legend(frameon=False, loc='upper right', fontsize=6.5)

VC = {'sample': '#332288', 'choice': '#117733', 'GNG': '#4477AA', 'test': '#999933'}
axB.plot([0.5, 1.0], [0.5, 1.0], ls='--', color='0.6', lw=0.8)
for v in ORDER:
    r = D[v]; axB.plot(r['decode'], r['ccgp'], 'o', ms=7, color=VC[v], mec='k', mew=0.5, label=v)
axB.set_xlim(0.5, 1.0); axB.set_ylim(0.5, 1.0)
axB.set_xlabel('within-context decoding'); axB.set_ylabel('CCGP')
axB.set_title('CCGP ≈ decoding → abstract', loc='left', fontsize=TITLE_FS)
axB.legend(frameon=False, fontsize=6.5, loc='upper left')

fig.suptitle('Pseudo-population CCGP: sample/choice/distractor codes form an abstract, factorised geometry',
             x=0.01, ha='left', y=1.02, fontsize=9)
OUT = 'figures/overlaps/ccgp'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
fig.savefig(f'{OUT}/png/overlaps_ccgp_pseudo.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/overlaps_ccgp_pseudo.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/overlaps_ccgp_pseudo.png'))
