"""SUPPLEMENT — discriminant-axis cosine similarity matrices (time × time).

For each mouse+stage the generalising decoder gives one weight axis per train-time bin
(raw neuron space, `run_overlaps.py --save-weights`). Unit-normalise each bin's axis;
cosine(axis_A(t), axis_B(t')) is a 84×84 similarity matrix. Cosines are computed PER
MOUSE (targets of a mouse+stage share the neuron basis) then averaged over mice.

Figure = a 4×4 grid over the four decoder codes (sample / choice / test / GNG):
  - DIAGONAL   : within-code temporal stability (is the axis the same over time?)
  - UPPER TRI  : cross-code alignment (≈0 ⇒ orthogonal — the dual-coding claim)
Diagonal is symmetric; cross-code panels are axis_row(t) · axis_col(t').

Decoder variant (mirrors fig_overlaps_main_native): default ridge; --l1 = lasso,
--lda = shrinkage-LDA (loads the matching bundled weights). One figure per stage.
Output: figures/overlaps/cosine/{png,svg}/overlaps_cosine_matrices_<stage>[_l1|_lda].{png,svg}
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

# ── decoder variant (parallel to the main figure) ───────────────────────────────
L1  = '--l1'  in sys.argv[1:]
LDA = '--lda' in sys.argv[1:]
# Trial set: default correct-only (fit on performance==1 & (DPA | odr_perf==1); run_overlaps --correct);
# --all uses the all-trials decoders. Routed to figures/overlaps/cosine/{correct,all}/{l2,l1,lda}/.
CORRECT = '--all' not in sys.argv[1:]
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
MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
CODES  = ['sample', 'choice', 'test', 'gng']
CLABEL = {'sample': 'sample', 'choice': 'choice', 'test': 'test', 'gng': 'GNG'}
CONTEXT = 'all'
xtime  = np.linspace(0, 14, 84)
ONSETS = [(xtime[12], 'sample'), (xtime[27], 'dist'), (xtime[54], 'test')]   # stimulus onsets (s)
# coarse trial epochs for the companion (annotated) panel: (label, bin_start, bin_stop)
EPOCHS = [('stim', 12, 18), ('eDelay', 18, 27), ('distr', 27, 33), ('mDelay', 33, 39),
          ('cue', 39, 42), ('gng rwd', 42, 45), ('lDelay', 45, 54), ('test', 54, 60),
          ('resp', 60, 72), ('dpa rwd', 72, 84)]
ENAMES = [e[0] for e in EPOCHS]
# Per-user read windows: 4 windows, same for every code & stage. The delay is split after the CUE,
# at the go/nogo reward (epoch 5 ~7s); the last two share `test`:
# stim-eDelay | distr-cue | gng rwd-test | test-dpaRwd.
FIXED_EPOCH_BLOCKS = [(0, 1), (2, 4), (5, 7), (7, 9)]      # epoch-index windows (inclusive)
FIXED_BIN_BLOCKS   = [(12, 26), (27, 41), (42, 59), (54, 83)]  # same windows, inclusive bin indices (raw path)
CLUSTER_RAW = '--rawclust' in sys.argv[1:]                  # cluster the raw 84×84 matrix instead of 8-epoch
edges = np.linspace(0, 14, 85)

VDIR = 'l1' if L1 else 'lda' if LDA else 'l2'   # decoder variant → own subfolder
OUT = f'figures/overlaps/cosine/{TSET}/{VDIR}'  # trial set (correct|all) → variant
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def unit_rows(ws):
    """ws (84, n_neurons) -> row-unit-normalised axes (84, n_neurons)."""
    ws = np.asarray(ws, float)
    nrm = np.linalg.norm(ws, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return ws / nrm


# ── load bundled weights ────────────────────────────────────────────────────────
W = pkl_load(f'{PFX}weights_{BDUM}', path=DATA_IN)['weights']    # {(mouse,stage,ctx,target): (84,n)}
print(f'{len(W)} weight axes loaded  [{DLAB}]')
Ns = [np.asarray(W[k]).shape[1] for k in W]
CHANCE = 1.0 / np.sqrt(np.mean(Ns))
print(f'mean n_neurons={np.mean(Ns):.0f}  ->  chance |cos| ~ {CHANCE:.3f}')


def mean_cos_matrix(stage, a, b):
    """Mean over mice of axis_a(t) · axis_b(t')  ->  (84,84)."""
    stack = []
    for m in MICE:
        ka, kb = (m, stage, CONTEXT, a), (m, stage, CONTEXT, b)
        if ka in W and kb in W:
            Ua, Ub = unit_rows(W[ka]), unit_rows(W[kb])           # (84,n) share neuron basis
            stack.append(Ua @ Ub.T)
    return (np.nanmean(np.stack(stack, 0), 0) if stack else np.full((84, 84), np.nan)), len(stack)


def draw_stage(stage):
    nc = len(CODES)
    fig = plt.figure(figsize=(12.5, 11.5))
    gs = GridSpec(nc, nc, figure=fig, wspace=0.18, hspace=0.18,
                  left=0.09, right=0.90, top=0.90, bottom=0.08)
    im = None
    for i, a in enumerate(CODES):            # row = code A (y-axis time)
        for j, b in enumerate(CODES):        # col = code B (x-axis time)
            if j < i:
                continue                     # lower triangle blank (matrices redundant by transpose)
            ax = fig.add_subplot(gs[i, j])
            M, n = mean_cos_matrix(stage, a, b)
            im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r', origin='upper',
                           extent=[0, 14, 14, 0], aspect='equal', interpolation='nearest')
            for onset, _ in ONSETS:
                ax.axvline(onset, color='0.35', lw=0.5, alpha=0.6)
                ax.axhline(onset, color='0.35', lw=0.5, alpha=0.6)
            if i == j:
                ax.set_title(f'{CLABEL[a]}  (within, n={n})', loc='left', fontsize=TITLE_FS)
            else:
                ax.set_title(f'{CLABEL[a]} × {CLABEL[b]}', loc='left', fontsize=TITLE_FS)
            # ticks/labels only on the outer edges
            if i == 0:
                ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
                ax.set_xticks([0, 7, 14]); ax.tick_params(labelsize=7)
            elif i == j:
                ax.set_xticks([0, 7, 14]); ax.set_xlabel(f'{CLABEL[b]} time (s)', fontsize=7.5)
                ax.tick_params(labelsize=7)
            else:
                ax.set_xticks([])
            if j == i:
                ax.set_yticks([0, 7, 14]); ax.set_ylabel(f'{CLABEL[a]} time (s)', fontsize=7.5)
                ax.tick_params(labelsize=7)
            else:
                ax.set_yticks([])
    cax = fig.add_axes([0.925, 0.30, 0.016, 0.40])
    fig.colorbar(im, cax=cax, label='cosine  (signed)')
    fig.suptitle(f'Discriminant-axis cosine similarity (time × time) — {stage}   ·   {DLAB}\n'
                 f'diagonal = within-code temporal stability   ·   upper triangle = cross-code '
                 f'alignment (≈0 ⇒ orthogonal, chance |cos|≈{CHANCE:.2f})',
                 fontsize=TITLE_FS, y=0.965)
    stem = f'overlaps_cosine_matrices_{stage.lower()}'
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('saved', p)


def epoch_axes(m, stage, tgt):
    """(nEpoch, n_neurons) unit axis per coarse epoch, or None if absent."""
    k = (m, stage, CONTEXT, tgt)
    if k not in W:
        return None
    ws = np.asarray(W[k], float)
    return unit_rows(np.stack([ws[a:b].mean(0) for _, a, b in EPOCHS]))


def mean_epoch_cos(stage, a, b):
    stack = []
    for m in MICE:
        Ea, Eb = epoch_axes(m, stage, a), epoch_axes(m, stage, b)
        if Ea is not None and Eb is not None:
            stack.append(Ea @ Eb.T)
    return (np.nanmean(np.stack(stack, 0), 0) if stack else np.full((len(EPOCHS),) * 2, np.nan)), len(stack)


def draw_stage_epochs(stage):
    """Companion: coarse epoch×epoch axis cosine with values printed (drift vs discrete blocks)."""
    nc, ne = len(CODES), len(EPOCHS)
    fig = plt.figure(figsize=(13.5, 12.5))
    gs = GridSpec(nc, nc, figure=fig, wspace=0.12, hspace=0.12,
                  left=0.10, right=0.90, top=0.90, bottom=0.10)
    im = None
    for i, a in enumerate(CODES):
        for j, b in enumerate(CODES):
            if j < i:
                continue
            ax = fig.add_subplot(gs[i, j])
            M, n = mean_epoch_cos(stage, a, b)
            im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
            for r in range(ne):
                for c in range(ne):
                    ax.text(c, r, f'{M[r, c]:+.2f}'.replace('+', ''), ha='center', va='center',
                            fontsize=5.2, color='w' if abs(M[r, c]) > 0.6 else '0.15')
            ax.set_title(f'{CLABEL[a]}  (within, n={n})' if i == j else f'{CLABEL[a]} × {CLABEL[b]}',
                         loc='left', fontsize=TITLE_FS)
            if j == i:
                ax.set_yticks(range(ne)); ax.set_yticklabels(ENAMES, fontsize=6)
                ax.set_ylabel(f'{CLABEL[a]} epoch', fontsize=7.5)
            else:
                ax.set_yticks([])
            if i == j:
                ax.set_xticks(range(ne)); ax.set_xticklabels(ENAMES, rotation=60, ha='right', fontsize=6)
                ax.set_xlabel(f'{CLABEL[b]} epoch', fontsize=7.5)
            else:
                ax.set_xticks([])           # cross panels: epoch order anchored by the diagonal cells
    cax = fig.add_axes([0.925, 0.30, 0.016, 0.40])
    fig.colorbar(im, cax=cax, label='cosine  (signed)')
    fig.suptitle(f'Discriminant-axis cosine by trial epoch — {stage}   ·   {DLAB}\n'
                 f'diagonal = within-code (high adjacent + slow decay ⇒ drifting axis; block ⇒ discrete '
                 f'code)   ·   upper triangle = cross-code (≈0 ⇒ orthogonal, chance |cos|≈{CHANCE:.2f})',
                 fontsize=TITLE_FS, y=0.965)
    stem = f'overlaps_cosine_epochs_{stage.lower()}'
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('saved', p)


def cluster_blocks(M, thr_cos):
    """Average-linkage contiguous segmentation: extend a temporal block while the candidate epoch's
    MEAN cosine to the epochs already in it stays >thr_cos. Robust to smooth drift — a single
    borderline pair no longer shatters the block (the old all-pairs/complete-linkage rule split on
    the 3rd decimal of a cosine sitting right at threshold, which flipped between stages). A code
    whose axis truly rotates still splits; a slowly-ramping/stable one stays one block. Deterministic."""
    ne = M.shape[0]
    blocks, s = [], 0
    for i in range(1, ne):
        if np.mean([M[k, i] for k in range(s, i)]) < thr_cos:   # i cannot join current block [s..i-1]
            blocks.append((s, i - 1)); s = i
    blocks.append((s, ne - 1))
    return blocks


def blocks_raw(M, thr=0.5, start=12, min_len=8):
    """Contiguous segmentation of the RAW 84×84 matrix: pre-stimulus baseline is one block; then
    split when a bin's mean cosine to the growing block drops below thr (average-linkage); finally
    fold blocks shorter than min_len bins into their more-similar neighbour (kills onset singletons)."""
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


def squares_complete(M, thr=0.5, start=0, min_len=2):
    """Data-driven read windows = diagonal SQUARES whose every within-pair cosine is >= thr
    (complete-linkage, cos 0.5 = 60°). Two-step, OVERLAP ALLOWED:
      1. Anchors — greedy disjoint tiling: grow [s..i] while adding i keeps min(M[s:i, i]) >= thr; on
         a drop, close the block and restart at i.
      2. Extend each anchor left and right as far as the whole [s..e] stays a complete >= thr square.
         Because a boundary epoch is often >= thr with BOTH neighbouring blocks, adjacent windows then
         SHARE that epoch (e.g. gng's distr-cue & cue-test overlap at the cue) — the data-driven analogue
         of the old hand-drawn windows that shared `test`. Fully-contained duplicates are dropped.
    A boundary still marks where the axis has rotated past 60° from a block's core — a genuine
    reorganization. Blocks shorter than min_len are dropped. Run on the STAGE-POOLED matrix so the same
    squares draw on Naive and Expert."""
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


def _bname(s, e):
    return ENAMES[s] if s == e else f'{ENAMES[s]}–{ENAMES[e]}'


def pooled_epoch_cos(a):
    """Epoch cosine averaged over BOTH stages → one matrix, so blocks are shared Naive+Expert."""
    return np.nanmean([mean_epoch_cos(st, a, a)[0] for st in STAGES], 0)


def pooled_raw_cos(a):
    return np.nanmean([mean_cos_matrix(st, a, a)[0] for st in STAGES], 0)


def draw_stage_clustered(stage, thr_cos=0.5):
    """Data-driven read windows, PER CODE: maximal diagonal squares whose every within-pair cosine is
    >= thr_cos (complete-linkage, see squares_complete). Blocks are computed on the STAGE-POOLED matrix
    so the same squares are drawn on Naive and Expert. Boxes overlaid on the per-stage cosine matrix."""
    nc, ne = len(CODES), len(EPOCHS)
    fig = plt.figure(figsize=(15.5, 5.0))
    gs = GridSpec(1, nc, figure=fig, wspace=0.42, left=0.05, right=0.93, top=0.72, bottom=0.30)
    im = None
    print(f'  [{stage}] read windows ({"raw 84x84" if CLUSTER_RAW else "epoch"}, squares cos>={thr_cos}):')
    for k, a in enumerate(CODES):
        ax = fig.add_subplot(gs[0, k])
        if CLUSTER_RAW:
            M, n = mean_cos_matrix(stage, a, a)                          # 84×84 raw (this stage, display)
            blocks = squares_complete(pooled_raw_cos(a), thr_cos, start=12, min_len=6)   # pooled squares
            im = ax.imshow(M, vmin=0, vmax=1, cmap='RdBu_r', origin='upper',
                           extent=[0, 14, 14, 0], aspect='equal', interpolation='nearest')
            for (s, e) in blocks:
                ax.add_patch(Rectangle((edges[s], edges[s]), edges[e + 1] - edges[s],
                                       edges[e + 1] - edges[s], fill=False, ec='k', lw=1.6))
            names = [f'{edges[s]:.1f}–{edges[e + 1]:.1f}s' for (s, e) in blocks]
            ax.set_xticks([0, 7, 14]); ax.set_yticks([0, 7, 14]); ax.tick_params(labelsize=7)
            ax.set_xlabel('time (s)', fontsize=7.5)
            subt = f'{len(blocks)} squares'
        else:
            M, n = mean_epoch_cos(stage, a, a)                          # this stage (display)
            blocks = squares_complete(pooled_epoch_cos(a), thr_cos, start=0, min_len=2)   # pooled squares
            im = ax.imshow(M, vmin=0, vmax=1, cmap='RdBu_r', aspect='equal')
            for (s, e) in blocks:
                ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                                       fill=False, ec='k', lw=2.4))
            names = [_bname(s, e) for (s, e) in blocks]
            ax.set_xticks(range(ne)); ax.set_xticklabels(ENAMES, rotation=60, ha='right', fontsize=6)
            ax.set_yticks(range(ne)); ax.set_yticklabels(ENAMES, fontsize=6)
            subt = '  |  '.join(names)
        print(f'      {a:>7}: ' + '  |  '.join(names))
        ax.set_title(f'{CLABEL[a]}  (n={n})\n{subt}', loc='left', fontsize=TITLE_FS)
    cax = fig.add_axes([0.945, 0.32, 0.012, 0.38])
    fig.colorbar(im, cax=cax, label='cosine')
    fig.suptitle(f'Code read windows — {stage}   ·   {DLAB}   ·   data-driven squares (all-pairs cos ≥ '
                 f'{thr_cos}, complete-linkage, stage-pooled), per code', fontsize=TITLE_FS, y=0.95)
    NAME = 'raw' if CLUSTER_RAW else 'epoch'     # epoch-averaged 10×10 vs raw 84-bin, both with read-windows
    stem = f'overlaps_cosine_{NAME}_{stage.lower()}'
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('  saved', p)


for st in STAGES:
    draw_stage(st)
    draw_stage_epochs(st)
    draw_stage_clustered(st)
print(f'\nCosine-matrix supplement → {OUT}/')
