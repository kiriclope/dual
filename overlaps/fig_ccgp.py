"""CCGP — geometry-of-abstraction test (Bernardi/Fusi 2020) on the SAME pseudo-population as Fig 3.

Cross-condition generalization performance: for each task variable, train a linear decoder on one set
of *context* conditions and test on held-out contexts. High CCGP = the variable is coded on a consistent,
low-dimensional, factorised (abstract, reusable) axis — the positive, generalization-based version of the
low-D claim, immune to the "only 4 conditions" critique of a condition-mean scree.

Per mouse (real within-mouse population, valid-neuron mask), per stage. Generalization context per var:
  sample   (A/B)        across TASK (DPA/Go/NoGo)   @ late delay   — memory abstract w.r.t. the distractor
  choice   (lick/no)    across SAMPLE (A/B)         @ test         — the shared lick/no-lick command
  GNG      (Go/NoGo)    across SAMPLE (A/B)         @ mid-delay    — distractor code abstract w.r.t. memory
  test     (C/D)        across TASK (DPA/Go/NoGo)   @ test         — test identity abstract w.r.t. task
Null = label-shuffle CCGP (per mouse). Also standard within-context CV decoding (abstraction = CCGP≈decode).

Output: figures/overlaps/ccgp/{png,svg}/overlaps_ccgp.{png,svg}
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import wilcoxon
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
MOUSE_COL = {m: c for m, c in zip(ALL_MICE, sns.color_palette('tab10', len(ALL_MICE)))}
o = set_options()
W_LD, W_MD, W_TE = np.asarray(o['bins_LD']), np.asarray(o['bins_MD']), np.asarray(o['bins_TEST'])
NSHUF, RNG = 30, np.random.RandomState(0)

print('loading pseudo-population …')
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
y = pkl_load('y_all_nan_', path='../data/pca')
VALID = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
                 path='../data/overlaps')['valid']

# variable spec: (label, target col / builder, context col, window, extra trial mask builder)
VARS = [
    ('sample', lambda d: d['sample_odor'].to_numpy(float), 'tasks',       W_LD, None),
    ('choice', lambda d: d['choice'].to_numpy(float),      'sample_odor', W_TE, None),
    ('GNG',    lambda d: (d['tasks'] == 'DualGo').to_numpy(float), 'sample_odor', W_MD,
     lambda d: d['tasks'].isin(['DualGo', 'DualNoGo']).to_numpy()),
    ('test',   lambda d: d['test_odor'].to_numpy(float),   'tasks',       W_TE, None),
]
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))


def population(mouse, stage, win):
    val = VALID[(mouse, stage)]
    idx = ((y.mouse == mouse) & (y.learning == stage) & (y.laser == 0)
           & (y.performance == 1)).to_numpy()                          # CORRECT trials only (matches Fig 2/3)
    A = np.nanmean(X[idx][:, val, :][:, :, win], axis=2)                 # trials × neurons (window-avg)
    A = np.where(np.isnan(A), np.nanmean(A, axis=0, keepdims=True), A)   # impute residual NaN with neuron mean
    return A, y[idx].reset_index(drop=True)


def ccgp(A, V, C):
    """mean cross-context generalization balanced-accuracy over all ordered context train→test pairs."""
    accs, cvals = [], [c for c in pd.unique(C) if pd.notna(c)]
    for ct in cvals:
        for ce in cvals:
            if ct == ce:
                continue
            tr, te = (C == ct), (C == ce)
            if len(np.unique(V[tr])) < 2 or len(np.unique(V[te])) < 2:
                continue
            if min(np.bincount(V[tr].astype(int)).min(), np.bincount(V[te].astype(int)).min()) < 3:
                continue
            clf = CLF().fit(A[tr], V[tr])
            accs.append(balanced_accuracy_score(V[te], clf.predict(A[te])))
    return float(np.mean(accs)) if accs else np.nan


def decode_cv(A, V):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); accs = []
    for tr, te in skf.split(A, V):
        clf = CLF().fit(A[tr], V[tr]); accs.append(balanced_accuracy_score(V[te], clf.predict(A[te])))
    return float(np.mean(accs))


rows = []
for stage in ['Naive', 'Expert']:
    for lab, vbuild, ctx, win, extra in VARS:
        for m in ALL_MICE:
            A, d = population(m, stage, win)
            keep = np.ones(len(d), bool)
            if extra is not None:
                keep &= extra(d)
            V = vbuild(d); C = d[ctx].to_numpy()
            ok = keep & np.isfinite(V) & pd.notna(C)
            A2, V2, C2 = A[ok], V[ok].astype(float), C[ok]
            if len(np.unique(V2)) < 2 or len(A2) < 20:
                continue
            g = ccgp(A2, V2, C2)
            dec = decode_cv(A2, V2)
            null = np.nanmean([ccgp(A2, RNG.permutation(V2), C2) for _ in range(NSHUF)])
            rows.append(dict(stage=stage, variable=lab, mouse=m, ccgp=g, decode=dec, null=null))
            print(f'{stage:6s} {lab:6s} {m:8s}  CCGP={g:.3f}  decode={dec:.3f}  null={null:.3f}')
R = pd.DataFrame(rows)
import pickle                                                          # cache for fig_overlaps_manifold.py
os.makedirs('figures/overlaps/ccgp', exist_ok=True)
R.to_pickle('figures/overlaps/ccgp/permouse_ccgp_cache.pkl')
print('cached per-mouse CCGP → figures/overlaps/ccgp/permouse_ccgp_cache.pkl')

# ── figure: A CCGP per variable (Naive vs Expert) vs chance/null; B CCGP-vs-decode abstraction ──
ORDER = ['sample', 'choice', 'GNG', 'test']
fig = plt.figure(figsize=(9.6, 6.2))
gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.9)
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3:6])
axS = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]

# panel A: Expert CCGP per variable, per-mouse points + mean, chance 0.5, null band
xE = np.arange(len(ORDER))
axA.axhline(0.5, ls=':', color='0.6', lw=0.8, zorder=0)
for i, v in enumerate(ORDER):
    sub = R[(R['variable'] == v) & (R['stage'] == 'Expert')]
    nb = sub['null'].mean()
    axA.plot([i - 0.28, i + 0.28], [nb, nb], color='0.75', lw=3, alpha=0.7, zorder=1,
             solid_capstyle='butt', label='shuffle null' if i == 0 else None)
    for _, r in sub.iterrows():
        axA.plot(i + RNG.uniform(-0.11, 0.11), r['ccgp'], 'o', ms=4, mfc=MOUSE_COL[r['mouse']],
                 mec='none', alpha=0.85, zorder=3)
    mu, se = sub['ccgp'].mean(), sub['ccgp'].std() / np.sqrt(len(sub))
    axA.plot([i - 0.28, i + 0.28], [mu, mu], color='k', lw=2, zorder=4)
    axA.plot([i, i], [mu - se, mu + se], color='k', lw=1.1, zorder=4)
    p = float(wilcoxon(sub['ccgp'] - 0.5, alternative='greater').pvalue)
    ns = int(p < .05) + int(p < .01) + int(p < .001)
    star = '*' * ns if ns else 'n.s.'
    axA.text(i, 0.985, star, ha='center', va='top',
             fontsize=12 if p < .05 else 8, fontweight='bold', color='k' if p < .05 else '0.55')
axA.set_xticks(xE); axA.set_xticklabels(ORDER); axA.set_ylim(0.42, 1.0)
axA.set_ylabel('CCGP  (cross-condition\ngeneralization, balanced acc.)')
axA.set_title('Codes generalize across context (abstract)', loc='left', fontsize=TITLE_FS)
axA.legend(frameon=False, loc='lower right', fontsize=6.5)

# panel B: CCGP vs standard within-context decoding (Expert) — near unity = fully abstract
VARC = {'sample': '#332288', 'choice': '#117733', 'GNG': '#1f77b4', 'test': '#999933'}
axB.plot([0.45, 1.0], [0.45, 1.0], ls='--', color='0.6', lw=0.8)
for v in ORDER:
    sub = R[(R['variable'] == v) & (R['stage'] == 'Expert')]
    axB.errorbar(sub['decode'].mean(), sub['ccgp'].mean(),
                 xerr=sub['decode'].std() / np.sqrt(len(sub)), yerr=sub['ccgp'].std() / np.sqrt(len(sub)),
                 fmt='o', ms=6, color=VARC[v], mec='k', mew=0.5, capsize=2, lw=1.0, label=v)
axB.set_xlim(0.45, 1.0); axB.set_ylim(0.45, 1.0)
axB.set_xlabel('within-context decoding'); axB.set_ylabel('CCGP')
axB.set_title('CCGP ≈ decoding → abstract', loc='left', fontsize=TITLE_FS)
axB.legend(frameon=False, fontsize=6.5, loc='upper left')
axB.text(0.98, 0.02, 'on unity = code is\nfully generalizable', transform=axB.transAxes,
         ha='right', va='bottom', fontsize=6.0, color='0.4', style='italic')

# panel C: per-mouse abstraction Naive vs Expert — ONE scatter per code (sample / choice / test).
# On the unity line = abstraction already present in Naive (pre-existing geometry, reused from the component
# tasks the animals already know); above = rose with learning. Dots = mice (colour = mouse), black diamond =
# mean, dotted lines = chance (0.5). ∗ = paired Wilcoxon p<.05.
for ax, v in zip(axS, ['sample', 'choice', 'test']):
    piv = (R[R['variable'] == v].pivot_table(index='mouse', columns='stage', values='ccgp')
           .dropna(subset=['Naive', 'Expert']))
    ax.plot([0.42, 1.0], [0.42, 1.0], ls='--', color='0.6', lw=0.8, zorder=0)
    ax.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0); ax.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
    for m, rr in piv.iterrows():
        ax.scatter(rr['Naive'], rr['Expert'], s=30, color=MOUSE_COL[m], edgecolor='k', linewidths=0.4, zorder=3)
    ax.scatter(piv['Naive'].mean(), piv['Expert'].mean(), s=95, color='k', marker='D', zorder=5)
    p = float(wilcoxon(piv['Expert'], piv['Naive']).pvalue)
    ax.set_xlim(0.42, 1.0); ax.set_ylim(0.42, 1.0); ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{v}   ({"∗" if p < .05 else "n.s."})', loc='left', fontsize=TITLE_FS)
    ax.set_xlabel('CCGP — Naive')
    ax.text(0.05, 0.96, f'Δ={piv.Expert.mean() - piv.Naive.mean():+.2f}\np={p:.2f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=6, color='0.3')
axS[0].set_ylabel('CCGP — Expert')

fig.suptitle('CCGP: sample / choice / distractor codes form an abstract low-dimensional geometry — per mouse '
             '(A, B);  abstraction Naive vs Expert, per code (C)', y=1.0, fontsize=9)
OUT = 'figures/overlaps/ccgp'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
fig.savefig(f'{OUT}/png/overlaps_ccgp.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/overlaps_ccgp.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/overlaps_ccgp.png'))

# summary
print('\n=== Expert CCGP (mean ± sem across 9 mice) vs chance ===')
for v in ORDER:
    sub = R[(R['variable'] == v) & (R['stage'] == 'Expert')]
    p = wilcoxon(sub['ccgp'] - 0.5, alternative='greater').pvalue
    print(f'  {v:6s} CCGP={sub.ccgp.mean():.3f}±{sub.ccgp.std()/np.sqrt(len(sub)):.3f}  '
          f'decode={sub.decode.mean():.3f}  null={sub.null.mean():.3f}  p(vs .5)={p:.4f}')

print('\n=== paired Naive→Expert CCGP (per-mouse Wilcoxon; panel C) ===')
for v in ORDER:
    piv = (R[R['variable'] == v].pivot_table(index='mouse', columns='stage', values='ccgp')
           .dropna(subset=['Naive', 'Expert']))
    if len(piv) >= 3:
        p = float(wilcoxon(piv['Expert'], piv['Naive']).pvalue)
        print(f'  {v:6s} n={len(piv)}  Naive={piv.Naive.mean():.3f}  Expert={piv.Expert.mean():.3f}  '
              f'Δ={piv.Expert.mean()-piv.Naive.mean():+.3f}  p={p:.3f}')
