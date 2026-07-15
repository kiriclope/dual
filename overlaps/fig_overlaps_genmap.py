"""SUPPLEMENT — cross-temporal DECODING generalization vs chance, per code.

For each code, the generalising decoder gives a decision value for every (train-time, test-time) pair
(the bundled tensor X[:,1] = train x test decision values per trial). We measure cross-temporal
decoding as AUC(decision, class-label) at each (t_train, t_test), per mouse, then average over mice.
Chance is 0.5 (a real, NON-uniform floor — unlike the cosine map's ~1/sqrt(n)), so the off-block
regions actually go dark and each code's generalization CLUSTER separates out: a bright block that
turns on at that code's driving event (sample ~encoding, gng ~distractor, test ~test, choice ~response)
and generalises within a limited window.

Significance: two-sided t-test across mice vs 0.5 at each bin (p<0.05), with small connected
components removed so the coherent blocks show rather than per-bin speckle. Red dashed lines mark the
two window anchors (distractor, test). Each mouse's decoder is oriented off its own diagonal so folding
does not bias the chance floor upward.

Decoder variant mirrors the main figure: default ridge; --l1 / --lda load the matching bundled tensor.
Stage: Expert (headline); --naive for the Naive stage.
Output: figures/overlaps/cosine/{png,svg}/overlaps_genmap_<stage>[_l1|_lda].{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats, ndimage
import matplotlib.pyplot as plt
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
STAGE = 'Naive' if '--naive' in sys.argv[1:] else 'Expert'
xtime = np.linspace(0, 14, 84)
ANCH = [4.5, 9.0]                                     # distractor, test onsets (s) — the window anchors
MIN_CLUSTER = 30                                      # drop significant blobs smaller than this (bins)
# code -> (title, target, class-column, context)
CODES = [('sample', 'sample', 'sample_odor', 'dpa'), ('choice', 'choice', 'choice', 'dpa'),
         ('test',   'test',   'test_odor',   'dpa'), ('gng',    'gng',    'gng',    'gng')]

VDIR = 'l1' if L1 else 'lda' if LDA else 'l2'   # decoder variant → own subfolder
OUT = f'figures/overlaps/cosine/{TSET}/{VDIR}'  # trial set (correct|all) → variant
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

Xb = np.asarray(pkl_load(f'{PFX}X_{BDUM}', path=DATA_IN))
yb = pkl_load(f'{PFX}labels_{BDUM}', path=DATA_IN)


def auc_map(dec, y):
    """dec (n,84,84), y (n,) binary -> (84,84) AUC via the rank (Mann-Whitney) formula, vectorised."""
    n = len(y); order = np.argsort(dec, axis=0); ranks = np.empty_like(order, float)
    ar = np.arange(1, n + 1)[:, None, None]
    np.put_along_axis(ranks, order, np.broadcast_to(ar, dec.shape).astype(float), axis=0)
    pos = y == 1; n1 = int(pos.sum()); n0 = n - n1
    if n1 < 3 or n0 < 3:
        return np.full((84, 84), np.nan)
    return (ranks[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def clean_sig(mask):
    """Keep only connected significant regions >= MIN_CLUSTER bins (removes per-bin speckle)."""
    lab, k = ndimage.label(mask)
    out = np.zeros_like(mask)
    for i in range(1, k + 1):
        if (lab == i).sum() >= MIN_CLUSTER:
            out[lab == i] = True
    return out


fig, axs = plt.subplots(1, 4, figsize=(19, 5.2))
tt = np.linspace(0, 14, 84)
for j, (name, tgt, col, ctx) in enumerate(CODES):
    tm = (yb.target == tgt).to_numpy()
    Xt = Xb[tm][:, 1].astype(float); yt = yb[tm].reset_index(drop=True)
    mouse = yt.mouse.to_numpy(); lz = yt.laser.to_numpy() == 0; sg = yt.learning.to_numpy() == STAGE
    # trials match the decoder set: correct → DPA perf==1, GNG perf==1 & odr_perf==1; all → no perf filter
    perf = (yt.performance.to_numpy() == 1) if CORRECT else np.ones(len(yt), bool)
    odr  = (yt.odr_perf.to_numpy() == 1)    if CORRECT else np.ones(len(yt), bool)
    ok = (lz & sg & perf & (yt.tasks.to_numpy() == 'DPA')) if ctx == 'dpa' \
        else (lz & sg & (yt.tasks.to_numpy() != 'DPA') & perf & odr)
    yv = yt[col].to_numpy()
    per = []
    for m in MICE:
        sel = ok & (mouse == m)
        if sel.sum() < 10:
            continue
        A = auc_map(Xt[sel], (yv[sel] == 1).astype(int))
        if np.nanmean(np.diag(A)) < 0.5:                 # orient this mouse off its own diagonal
            A = 1 - A
        per.append(A)
    per = np.stack(per); mAUC = np.nanmean(per, 0)
    _, p = stats.ttest_1samp(per, 0.5, axis=0, nan_policy='omit')
    sig = clean_sig((p < 0.05) & (mAUC > 0.55))
    ax = axs[j]
    im = ax.imshow(mAUC, vmin=0.5, vmax=1.0, cmap='viridis', origin='upper',
                   extent=[0, 14, 14, 0], aspect='equal')
    ax.contour(tt, tt, sig.astype(float), levels=[0.5], colors='w', linewidths=1.0)
    for a in ANCH:
        ax.axvline(a, color='r', lw=0.8, ls='--', alpha=0.7); ax.axhline(a, color='r', lw=0.8, ls='--', alpha=0.7)
    ax.set_title(f'{name}  (peak AUC {np.nanmax(np.diag(mAUC)):.2f} @ {xtime[np.nanargmax(np.diag(mAUC))]:.1f}s)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('test time (s)', fontsize=8); ax.set_ylabel('train time (s)', fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, label='AUC')
    print(f"{name:>7}: peak diag AUC={np.nanmax(np.diag(mAUC)):.2f} @ "
          f"{xtime[np.nanargmax(np.diag(mAUC))]:.1f}s  sig-block bins={int(sig.sum())}")
fig.suptitle(f'Cross-temporal decoding generalization vs chance (AUC, chance=0.5) — {STAGE}   ·   {DLAB}'
             f'   ·   white = sig>chance (p<.05, blocks); red = distractor & test anchors',
             fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
stem = f'overlaps_genmap_{STAGE.lower()}'
p = os.path.join(OUT, 'png', f'{stem}.png')
fig.savefig(p, dpi=300, bbox_inches='tight')
fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
