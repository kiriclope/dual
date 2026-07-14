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

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

# ── decoder variant (parallel to the main figure) ───────────────────────────────
L1  = '--l1'  in sys.argv[1:]
LDA = '--lda' in sys.argv[1:]
if L1:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_l1_ratio_1.0', 'l1_', '_l1', 'L1 (lasso)'
elif LDA:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_lda',         'lda_', '_lda', 'shrinkage-LDA'
else:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_l1_ratio_0.0', '',    '',    'ridge (logistic L2)'
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
EPOCHS = [('stim', 12, 18), ('eDelay', 18, 27), ('distr', 27, 33), ('mDelay', 33, 45),
          ('lDelay', 45, 54), ('test', 54, 60), ('resp', 60, 72), ('lResp', 72, 84)]
ENAMES = [e[0] for e in EPOCHS]

OUT = 'figures/overlaps/cosine'
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
                ax.set_title(f'{CLABEL[a]}  (within, n={n})', fontsize=9, fontweight='bold')
            else:
                ax.set_title(f'{CLABEL[a]} × {CLABEL[b]}', fontsize=9)
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
                 fontsize=12, y=0.965)
    stem = f'overlaps_cosine_matrices_{stage.lower()}{SUF}'
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
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
                         fontsize=9, fontweight='bold' if i == j else 'normal')
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
                 fontsize=12, y=0.965)
    stem = f'overlaps_cosine_epochs_{stage.lower()}{SUF}'
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('saved', p)


def cluster_blocks(M, thr_cos):
    """Complete-linkage contiguous segmentation: greedily extend a temporal block as long as EVERY
    pair of epochs already in it stays cos>thr_cos (all-mutually-stable). A drifting axis therefore
    splits into several blocks; a genuinely stable one stays a single block. Deterministic."""
    ne = M.shape[0]
    blocks, s = [], 0
    for i in range(1, ne):
        if min(M[k, i] for k in range(s, i)) < thr_cos:      # i cannot join current block [s..i-1]
            blocks.append((s, i - 1)); s = i
    blocks.append((s, ne - 1))
    return blocks


def _bname(s, e):
    return ENAMES[s] if s == e else f'{ENAMES[s]}–{ENAMES[e]}'


def draw_stage_clustered(stage, thr_cos=0.5):
    """Data-driven blocks (all-pairs cos>thr_cos): within-code matrix with block boundaries drawn."""
    nc, ne = len(CODES), len(EPOCHS)
    fig = plt.figure(figsize=(15.5, 5.0))
    gs = GridSpec(1, nc, figure=fig, wspace=0.42, left=0.05, right=0.93, top=0.72, bottom=0.30)
    im = None
    print(f'  [{stage}] clustered blocks (all-pairs cos>{thr_cos:.1f}):')
    for k, a in enumerate(CODES):
        ax = fig.add_subplot(gs[0, k])
        M, n = mean_epoch_cos(stage, a, a)
        im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
        blocks = cluster_blocks(M, thr_cos)
        for (s, e) in blocks:
            ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                                   fill=False, ec='k', lw=2.4))
        names = [_bname(s, e) for (s, e) in blocks]
        print(f'      {a:>7}: ' + '  |  '.join(names))
        ax.set_xticks(range(ne)); ax.set_xticklabels(ENAMES, rotation=60, ha='right', fontsize=6.5)
        ax.set_yticks(range(ne)); ax.set_yticklabels(ENAMES, fontsize=6.5)
        ax.set_title(f'{CLABEL[a]}  (n={n})\n' + '  |  '.join(names), fontsize=8.5, fontweight='bold')
    cax = fig.add_axes([0.945, 0.32, 0.012, 0.38])
    fig.colorbar(im, cax=cax, label='cosine')
    fig.suptitle(f'Data-driven code blocks — {stage}   ·   {DLAB}   ·   complete-linkage contiguous '
                 f'segmentation (block = all-pairs cos>{thr_cos:.1f})', fontsize=12, y=0.95)
    stem = f'overlaps_cosine_clustered_{stage.lower()}{SUF}'
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('  saved', p)


for st in STAGES:
    draw_stage(st)
    draw_stage_epochs(st)
    draw_stage_clustered(st)
print(f'\nCosine-matrix supplement → {OUT}/')
