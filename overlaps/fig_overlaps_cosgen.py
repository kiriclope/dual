"""SUPPLEMENT — cross-temporal axis COSINE similarity vs chance, per code (axis-geometry companion to
fig_overlaps_genmap.py's decoding view).

For each code the generalising decoder gives one unit weight axis per bin (raw neuron space). Per mouse
we form the cross-temporal cosine map axis_t . axis_t' (84x84).

NULL = the CROSS-CODE cosine (refit-free): axes for DIFFERENT variables (sample/choice/test/gng) live in
the same population but decode unrelated things, so their cosine is the empirical "how aligned are two
decoders by chance" floor (measured: mean 0.006, |cos| 99th pct ~0.24). A (t,t') bin is SIGNIFICANT when
the within-code cosine beats each mouse's OWN cross-code 99th percentile in >= MIN_MICE of 9 mice
(per-mouse false-positive 1% -> binomial p<1e-4). This removes the earlier arbitrary cos>0.3 floor: the
threshold is now the data's own unrelated-decoder alignment. Small connected components are dropped.

Clusters are extracted from the significant band by average-linkage (grow while the next bin's MEAN cosine
to the window stays above the cross-code floor and its alignment is significant). This reproduces the
decoding-AUC / eigenmap result: gng splits at the DISTRACTOR, choice/test at TEST/response, sample stays
one block — union of boundaries = {distractor, test} = the composite figure's 3 read-window anchors.

Decoder variant mirrors the main figure: default ridge; --l1 / --lda load the matching bundled weights.
Stage: Expert (headline); --naive for Naive.
Output: figures/overlaps/cosine/{png,svg}/overlaps_cosgen_<stage>[_l1|_lda].{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
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
STAGE = 'Naive' if '--naive' in sys.argv[1:] else 'Expert'
EP = [('stim',12,18),('eDelay',18,27),('distr',27,33),('mDelay',33,39),('cue',39,42),
      ('gng rwd',42,45),('lDelay',45,54),('test',54,60),('resp',60,72),('dpa rwd',72,84)]
xtime = np.linspace(0, 14, 84)
NULL_PCT = 99                                         # cross-code (unrelated-decoder) percentile = chance floor
MIN_MICE = 3                                          # mice that must beat their own cross-code floor (binomial p<1e-4)
ANCH = [4.5, 9.0]                                     # distractor, test onsets (s)
MINC = 30                                             # min connected significant-block size (bins)
LO = 12                                               # start clustering after the pre-stim baseline
CODES = ['sample', 'choice', 'test', 'gng']

VDIR = 'l1' if L1 else 'lda' if LDA else 'l2'   # decoder variant → own subfolder
OUT = f'figures/overlaps/cosine/{TSET}/{VDIR}'  # trial set (correct|all) → variant
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

W = pkl_load(f'{PFX}weights_{BDUM}', path=DATA_IN)['weights']


def unit(ws):
    ws = np.asarray(ws, float); n = np.linalg.norm(ws, axis=1, keepdims=True); n[n == 0] = 1; return ws/n
def label_bin(b):
    for nm, a, c in EP:
        if a <= b < c: return nm
    return 'base' if b < LO else EP[-1][0]
def clean(mask):
    lab, k = ndimage.label(mask); out = np.zeros_like(mask)
    for i in range(1, k + 1):
        if (lab == i).sum() >= MINC: out[lab == i] = True
    return out
def extract_blocks(sig, anchor=3):
    """Single-linkage to the block START on the SIGNIFICANCE mask: grow a window while the new bin stays
    significantly aligned (beyond the cross-code floor) to the block's ONSET (its first `anchor` bins, to
    resist single-bin flicker). A boundary marks where the axis has rotated away from where the block
    BEGAN — a genuine reorganization — rather than every incremental drift step (which complete-linkage
    would over-segment on a smoothly rotating axis)."""
    n = len(sig); blocks = []; s = LO
    for i in range(LO + 1, n):
        if sig[s:min(s + anchor, i), i].mean() < 0.5:
            blocks.append((s, i - 1)); s = i
    blocks.append((s, n - 1)); return blocks


# ── cross-code (unrelated-decoder) null: the chance-alignment floor, refit-free ──
# Axes for DIFFERENT variables live in the same population but decode unrelated things, so their cosine
# IS the empirical "how aligned are two decoders by chance" floor. Replaces the arbitrary cos threshold.
AX = {m: {c: unit(W[(m, STAGE, 'all', c)]) for c in CODES if (m, STAGE, 'all', c) in W} for m in MICE}
cross_all = [np.abs(AX[m][a] @ AX[m][b].T).ravel()
             for m in MICE for i, a in enumerate(list(AX[m])) for b in list(AX[m])[i + 1:]]
TAU = float(np.percentile(np.concatenate(cross_all), NULL_PCT))     # pooled cross-code |cos| floor
print(f'Cross-temporal cosine clusters — {STAGE}  [{DLAB}]   cross-code |cos| {NULL_PCT}th pct = {TAU:.3f} '
      f'(data-derived chance floor; replaces the pinned cos threshold)')

fig, axs = plt.subplots(1, 4, figsize=(19, 5.4)); tt = np.linspace(0, 14, 84)
for j, c in enumerate(CODES):
    mm = [m for m in MICE if c in AX[m]]
    within = np.stack([AX[m][c] @ AX[m][c].T for m in mm]); mC = within.mean(0)
    # per mouse: does within-code alignment beat THAT mouse's own cross-code floor (per-mouse FP = 1%)?
    exc = []
    for k, m in enumerate(mm):
        others = [AX[m][c2] for c2 in CODES if c2 != c and c2 in AX[m]]
        cc = np.concatenate([np.abs(AX[m][c] @ O.T).ravel() for O in others])
        exc.append(within[k] > np.percentile(cc, NULL_PCT))
    count = np.stack(exc).sum(0)                       # #mice beyond their own cross-code floor
    sig = clean(count >= MIN_MICE)                     # >=3/9 at per-mouse FP=1% -> binomial p<1e-4
    blocks = extract_blocks(sig)
    names = [f'{label_bin(s)}-{label_bin(e)} ({xtime[s]:.1f}-{xtime[min(e,83)]:.1f}s)' for s, e in blocks]
    print(f'  {c:>7}: ' + '  |  '.join(names))
    ax = axs[j]
    im = ax.imshow(mC, vmin=-1, vmax=1, cmap='RdBu_r', origin='upper', extent=[0, 14, 14, 0], aspect='equal')
    ax.contour(tt, tt, sig.astype(float), levels=[0.5], colors='lime', linewidths=1.2)
    for s, e in blocks:
        ax.add_patch(Rectangle((xtime[s], xtime[s]), xtime[min(e,83)]-xtime[s], xtime[min(e,83)]-xtime[s],
                               fill=False, ec='k', lw=1.7))
    for a in ANCH:
        ax.axvline(a, color='k', lw=0.8, ls='--', alpha=0.5); ax.axhline(a, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(f'{c}  ({len(blocks)} block{"s" if len(blocks)>1 else ""})', fontsize=10, fontweight='bold')
    ax.set_xlabel('time (s)', fontsize=8); ax.set_ylabel('time (s)', fontsize=8); ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, label='cosine')
fig.suptitle(f'Cross-temporal axis cosine vs the CROSS-CODE (unrelated-decoder) null — {STAGE}   ·   {DLAB}'
             f'   ·   lime = aligned beyond cross-code floor ({TAU:.2f}) in >={MIN_MICE}/9 mice; '
             f'black box = clusters; black dashed = distractor & test anchors', fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
stem = f'overlaps_cosgen_{STAGE.lower()}'
p = os.path.join(OUT, 'png', f'{stem}.png')
fig.savefig(p, dpi=300, bbox_inches='tight')
fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
