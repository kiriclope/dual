"""SUPPLEMENT (combined) — per code: clustered within-code cosine matrix + each cluster's
generalization across test-time, in one figure.

One horizontal band per decoder code (sample / choice / test / GNG):
  - LEFT  : within-code epoch×epoch axis cosine with the read windows drawn — DATA-DRIVEN per code:
            maximal diagonal squares whose every within-pair cosine is >= THR (complete-linkage),
            computed on the Naive+Expert-pooled matrix so the same squares apply to both stages.
  - RIGHT : for each block, the decoder TRAINED in that block read at every test-time bin
            (class decision function, Expert, per-mouse BL-z, mean ± SEM over mice); the
            block's own train window is shaded.

Expert stage (headline); both-stage versions are the standalone fig_overlaps_test_cluster_traj.py.
Decoder variant: default ridge; --l1 / --lda load the matching bundled tensor.
Output: figures/overlaps/cosine/{png,svg}/overlaps_code_clusters_combined[_l1|_lda].{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

L1  = '--l1'  in sys.argv[1:]
LDA = '--lda' in sys.argv[1:]
# Correct-only decoders (fit on performance==1 & (DPA | odr_perf==1) trials; run_overlaps --correct).
if L1:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_correct_l1_ratio_1.0', 'l1_', '_l1', 'L1 (lasso)'
elif LDA:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_correct_lda',         'lda_', '_lda', 'shrinkage-LDA'
else:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_correct_l1_ratio_0.0', '',    '',    'ridge (logistic L2)'
BDUM = f'{DUM}_raw_targets_choice-gng-sample-test'

DATA_IN = '../data/overlaps'
MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGE  = 'Naive' if '--naive' in sys.argv[1:] else 'Expert'   # blocks + traces both from this stage
BL     = slice(0, 12)
xtime  = np.linspace(0, 14, 84)
EVENTS = [('sample', 2.0, 3.0), ('distractor', 4.5, 5.5), ('test', 9.0, 10.0)]   # stimulus ON windows (s)


def draw_events(ax, label=False):
    """Task-event stimulus windows (sample / distractor / test) as light-blue bands."""
    for name, t0, t1 in EVENTS:
        ax.axvspan(t0, t1, color='#4c72b0', alpha=0.13, lw=0, zorder=1)
        if label:
            ax.text((t0 + t1) / 2, 0.98, name, transform=ax.get_xaxis_transform(), rotation=90,
                    ha='center', va='top', fontsize=6, color='#33488e', zorder=5)


EPOCHS = [('stim', 12, 18), ('eDelay', 18, 27), ('distr', 27, 33), ('mDelay', 33, 39),
          ('cue', 39, 42), ('gng rwd', 42, 45), ('lDelay', 45, 54), ('test', 54, 60),
          ('resp', 60, 72), ('dpa rwd', 72, 84)]
ENAMES = [e[0] for e in EPOCHS]
# Per-user read windows: 4 windows, same for every code & stage. The delay is split after the CUE,
# at the go/nogo reward (epoch 5 ~7s); the last two share `test`:
# stim-eDelay | distr-cue | gng rwd-test | test-dpaRwd.
FIXED_EPOCH_BLOCKS = [(0, 1), (2, 4), (5, 7), (7, 9)]      # epoch-index windows (inclusive)
FIXED_BIN_BLOCKS   = [(12, 26), (27, 41), (42, 59), (54, 83)]  # same windows, inclusive bin indices (raw path)
CODES = [
    ('sample', 'sample', 'sample_odor', [(0, 'Odor A', '#332288'), (1, 'Odor B', '#44AA99')], 'dpa'),
    ('choice', 'choice', 'choice',      [(0, 'No lick', '#377eb8'), (1, 'Lick', '#4daf4a')],  'dpa'),
    ('test',   'test',   'test_odor',   [(0, 'Odor C', '#CC6677'), (1, 'Odor D', '#999933')], 'dpa'),
    ('GNG',    'gng',    'gng',         [(0, 'NoGo', '#2ca02c'), (1, 'Go', '#1f77b4')],       'gng'),
]

VDIR = 'l1' if L1 else 'lda' if LDA else 'l2'   # decoder variant → own subfolder
OUT = f'figures/overlaps/cosine/{VDIR}'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def unit_rows(ws):
    ws = np.asarray(ws, float)
    nrm = np.linalg.norm(ws, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return ws / nrm


def epoch_cos(target, stage=STAGE):
    stack = []
    for m in MICE:
        k = (m, stage, 'all', target)
        if k in W:
            E = unit_rows(np.stack([np.asarray(W[k], float)[a:b].mean(0) for _, a, b in EPOCHS]))
            stack.append(E @ E.T)
    return np.nanmean(np.stack(stack, 0), 0)


def cluster_blocks(M, thr_cos=0.5):
    """Average-linkage contiguous segmentation (robust to smooth drift; see fig_overlaps_cosine_matrices)."""
    ne = M.shape[0]; blocks, s = [], 0
    for i in range(1, ne):
        if np.mean([M[k, i] for k in range(s, i)]) < thr_cos:
            blocks.append((s, i - 1)); s = i
    blocks.append((s, ne - 1))
    return blocks


CLUSTER_RAW = '--rawclust' in sys.argv[1:]                  # segment the RAW 84×84 matrix, not the 8-epoch one
edges = np.linspace(0, 14, 85)                              # bin edges (s) for raw block extents


def raw_cos(target, stage=STAGE):                           # mean-over-mice 84×84 within-code axis cosine
    S = [unit_rows(W[(m, stage, 'all', target)]) for m in MICE if (m, stage, 'all', target) in W]
    return np.nanmean([U @ U.T for U in S], 0)


# blocks are computed on the Naive+Expert-averaged matrix so they are SHARED across stages
def pooled_epoch_cos(target):
    return np.nanmean([epoch_cos(target, st) for st in ('Naive', 'Expert')], 0)


def pooled_raw_cos(target):
    return np.nanmean([raw_cos(target, st) for st in ('Naive', 'Expert')], 0)


THR = 0.5   # cosine threshold: keep diagonal squares whose every within-pair cosine is >= THR


def squares_complete(M, thr=THR, start=0, min_len=2):
    """Data-driven read windows PER CODE = diagonal squares whose every within-pair cosine is >= thr
    (complete-linkage, cos 0.5 = 60°). OVERLAP ALLOWED: (1) greedy disjoint anchors, then (2) extend
    each anchor L & R while [s..e] stays a complete >= thr square, so a boundary epoch aligned with both
    neighbours is SHARED (e.g. gng distr-cue & cue-test overlap at the cue). Contained duplicates
    dropped; blocks < min_len dropped. Run on the STAGE-POOLED matrix (same squares Naive & Expert)."""
    n = len(M); anchors, s = [], start
    for i in range(start + 1, n):
        if min(M[k, i] for k in range(s, i)) < thr:
            if i - s >= min_len:
                anchors.append((s, i - 1))
            s = i
    if n - s >= min_len:
        anchors.append((s, n - 1))

    def complete(a, b):
        return all(M[p, q] >= thr for p in range(a, b + 1) for q in range(p + 1, b + 1))
    out = []
    for s, e in anchors:
        while s - 1 >= start and complete(s - 1, e):
            s -= 1
        while e + 1 < n and complete(s, e + 1):
            e += 1
        out.append((s, e))
    out = sorted(set(out))
    return [b for b in out if not any(b != c and c[0] <= b[0] and b[1] <= c[1] for c in out)]


def blocks_raw(M, thr=0.5, start=12, min_len=8):
    """Contiguous segmentation of the RAW time×time matrix: pre-stimulus baseline (bins 0..start-1)
    is one block; then split whenever a bin's mean cosine to the growing block falls below thr
    (average-linkage, robust to single-bin noise); finally fold blocks shorter than min_len bins
    into their more-similar neighbour so fast onset-transition bins don't become singletons."""
    n = len(M); bl = [(0, start - 1)]; s = start
    for i in range(start + 1, n):
        if np.mean([M[k, i] for k in range(s, i)]) < thr:
            bl.append((s, i - 1)); s = i
    bl.append((s, n - 1))
    changed = True
    while changed and len(bl) > 2:
        changed = False
        for idx, (s, e) in enumerate(bl):
            if e - s + 1 >= min_len or idx == 0:
                continue
            simp = np.mean(M[bl[idx - 1][0]:bl[idx - 1][1] + 1, s:e + 1])
            simn = np.mean(M[bl[idx + 1][0]:bl[idx + 1][1] + 1, s:e + 1]) if idx < len(bl) - 1 else -9
            if idx < len(bl) - 1 and simn >= simp:
                bl[idx + 1] = (s, bl[idx + 1][1])
            else:
                bl[idx - 1] = (bl[idx - 1][0], e)
            bl.pop(idx); changed = True; break
    return bl


W = pkl_load(f'{PFX}weights_{BDUM}', path=DATA_IN)['weights']
Xb = pkl_load(f'{PFX}X_{BDUM}', path=DATA_IN)
yb = pkl_load(f'{PFX}labels_{BDUM}', path=DATA_IN)

# ── precompute per code: display matrix, matrix blocks, trajectory windows + names ──
CD = []
for spec in CODES:
    title, target, col, levels, context = spec
    if CLUSTER_RAW:
        Mdisp = raw_cos(target)                              # 84×84 raw cosine (this stage, display)
        mblocks = squares_complete(pooled_raw_cos(target), THR, start=12, min_len=6)   # pooled squares
        traj = mblocks
        windows = [(s, e + 1) for s, e in traj]
        tnames = [f'{edges[s]:.1f}–{edges[e + 1]:.1f}s' for s, e in traj]
    else:
        Mdisp = epoch_cos(target)                            # 8×8 epoch cosine (this stage, display)
        mblocks = squares_complete(pooled_epoch_cos(target), THR, start=0, min_len=2)   # pooled squares
        traj = mblocks
        windows = [(EPOCHS[i][1], EPOCHS[j][2]) for i, j in traj]
        tnames = [ENAMES[i] if i == j else f'{ENAMES[i]}–{ENAMES[j]}' for i, j in traj]
    CD.append((spec, Mdisp, mblocks, windows, tnames))
NCOLS = max(len(w) for *_, w, _ in CD)

fig = plt.figure(figsize=(3.2 + 2.7 * NCOLS, 13.5))
outer = GridSpec(len(CODES), 2, figure=fig, width_ratios=[1.0, 0.95 * NCOLS],
                 hspace=0.62, wspace=0.10, left=0.05, right=0.99, top=0.94, bottom=0.05)
ne = len(EPOCHS)
im_mat = None
for r, ((title, target, col, levels, context), Mdisp, mblocks, windows, tnames) in enumerate(CD):
    # ── LEFT: within-code matrix (raw 84×84 or epoch 8×8) with block boundaries ──
    axm = fig.add_subplot(outer[r, 0])
    if CLUSTER_RAW:
        im_mat = axm.imshow(Mdisp, vmin=0, vmax=1, cmap='RdBu_r', origin='upper',
                            extent=[0, 14, 14, 0], aspect='equal', interpolation='nearest')
        for (s, e) in mblocks:
            axm.add_patch(Rectangle((edges[s], edges[s]), edges[e + 1] - edges[s],
                                    edges[e + 1] - edges[s], fill=False, ec='k', lw=1.6))
        axm.set_xticks([0, 7, 14]); axm.set_yticks([0, 7, 14]); axm.tick_params(labelsize=7)
        axm.set_xlabel('time (s)', fontsize=7.5)
        subtitle = f'{len(mblocks)} blocks'
    else:
        im_mat = axm.imshow(Mdisp, vmin=0, vmax=1, cmap='RdBu_r', aspect='equal')
        for (s, e) in mblocks:
            axm.add_patch(Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1, fill=False, ec='k', lw=2.2))
        axm.set_xticks(range(ne)); axm.set_xticklabels(ENAMES, rotation=60, ha='right', fontsize=6)
        axm.set_yticks(range(ne)); axm.set_yticklabels(ENAMES, fontsize=6)
        subtitle = '  |  '.join(tnames)
    axm.set_title(f'{title} code — blocks\n{subtitle}', fontsize=8, fontweight='bold')

    # ── RIGHT: per-block generalization across test-time ──
    tm = (yb.target == target).to_numpy()
    Xt = Xb[tm][:, 1].astype(float)
    yt = yb[tm].reset_index(drop=True)
    mouse = yt.mouse.to_numpy(); sg = yt.learning.to_numpy(); lz = yt.laser.to_numpy() == 0
    # correct-only, matching the correct-fit decoders: DPA needs performance==1; GNG (Dual) needs the
    # go/nogo response correct too (odr_perf==1), i.e. the same rule dataloader(correct=True) applies.
    ok = (lz & (yt.performance.to_numpy() == 1) & (yt.tasks.to_numpy() == 'DPA')) if context == 'dpa' \
        else (lz & (yt.tasks.to_numpy() != 'DPA') & (yt.performance.to_numpy() == 1)
              & (yt.odr_perf.to_numpy() == 1))
    lab_v = yt[col].to_numpy()

    def blz(trace):
        Z = np.full_like(trace, np.nan)
        for mo in MICE:
            mm = mouse == mo
            z = trace[mm] - trace[mm][:, BL].mean()
            Z[mm] = z / (trace[mm][:, BL].std() + 1e-9)
        return Z

    band = outer[r, 1].subgridspec(1, NCOLS, wspace=0.28)
    ax0 = None
    for c in range(NCOLS):
        if c >= len(windows):
            continue
        a, b = windows[c]
        ax = fig.add_subplot(band[0, c], sharey=ax0)
        ax0 = ax0 or ax
        ax.axvspan(edges[a], edges[min(b, 84)], color='0.88', zorder=0)
        draw_events(ax, label=(r == 0 and c == 0))
        ax.axhline(0, color='k', lw=0.5, ls='--')
        Z = blz(Xt[:, a:b, :].mean(1))
        for lv, lab, cclr in levels:
            per = [np.nanmean(Z[ok & (sg == STAGE) & (lab_v == lv) & (mouse == mo)], 0)
                   for mo in MICE if (ok & (sg == STAGE) & (lab_v == lv) & (mouse == mo)).sum() >= 3]
            if len(per) >= 2:
                Mm = np.stack(per); mu = Mm.mean(0); se = Mm.std(0, ddof=1) / np.sqrt(len(per))
                ax.plot(xtime, mu, color=cclr, lw=1.6, label=lab, zorder=2)
                ax.fill_between(xtime, mu - se, mu + se, color=cclr, alpha=0.2, lw=0)
        ax.set_xlim(0, 14); ax.set_xticks([0, 4.5, 9, 14]); ax.tick_params(labelsize=7)
        ax.set_title(tnames[c], fontsize=8)
        if c == 0:
            ax.set_ylabel(f'{title} code (z)', fontsize=8); ax.legend(fontsize=6.5, frameon=False,
                                                                      loc='upper left', handlelength=1.0)
        ax.set_xlabel('test time (s)', fontsize=7.5)
        sns.despine(ax=ax)

cax = fig.add_axes([0.005, 0.35, 0.008, 0.30])
fig.colorbar(im_mat, cax=cax, label='cosine'); cax.yaxis.set_ticks_position('left')
cax.yaxis.set_label_position('left')
fig.suptitle(f'Code read windows (left, boxes on the cosine matrix; data-driven squares, all-pairs cos ≥ '
             f'{THR}, complete-linkage, stage-pooled, per code) and each window read across test-time (right) — '
             f'{STAGE}   ·   {DLAB}', fontsize=13, y=0.975)
NAME = 'raw' if CLUSTER_RAW else 'epoch'      # epoch-averaged vs raw 84-bin left matrix, both windowed
p = os.path.join(OUT, 'png', f'overlaps_code_{NAME}_combined_{STAGE.lower()}.png')
fig.savefig(p, dpi=200, bbox_inches='tight')
fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
plt.close(fig); print('saved', p)
