"""SUPPLEMENT (combined) — per code: clustered within-code cosine matrix + each cluster's
generalization across test-time, in one figure.

One horizontal band per decoder code (sample / choice / test / GNG):
  - LEFT  : within-code epoch×epoch axis cosine with data-driven block boundaries drawn
            (complete-linkage contiguous, all-pairs cos>0.5) — where the sub-codes are.
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
if L1:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_l1_ratio_1.0', 'l1_', '_l1', 'L1 (lasso)'
elif LDA:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_lda',         'lda_', '_lda', 'shrinkage-LDA'
else:
    DUM, PFX, SUF, DLAB = 'log_generalizing_overlaps_none_l1_ratio_0.0', '',    '',    'ridge (logistic L2)'
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
CODES = [
    ('sample', 'sample', 'sample_odor', [(0, 'Odor A', '#332288'), (1, 'Odor B', '#44AA99')], 'dpa'),
    ('choice', 'choice', 'choice',      [(0, 'No lick', '#377eb8'), (1, 'Lick', '#4daf4a')],  'dpa'),
    ('test',   'test',   'test_odor',   [(0, 'Odor C', '#CC6677'), (1, 'Odor D', '#999933')], 'dpa'),
    ('GNG',    'gng',    'gng',         [(0, 'NoGo', '#2ca02c'), (1, 'Go', '#1f77b4')],       'gng'),
]

OUT = 'figures/overlaps/cosine'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def unit_rows(ws):
    ws = np.asarray(ws, float)
    nrm = np.linalg.norm(ws, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return ws / nrm


def epoch_cos(target):
    stack = []
    for m in MICE:
        k = (m, STAGE, 'all', target)
        if k in W:
            E = unit_rows(np.stack([np.asarray(W[k], float)[a:b].mean(0) for _, a, b in EPOCHS]))
            stack.append(E @ E.T)
    return np.nanmean(np.stack(stack, 0), 0)


def cluster_blocks(M, thr_cos=0.5):
    ne = M.shape[0]; blocks, s = [], 0
    for i in range(1, ne):
        if min(M[k, i] for k in range(s, i)) < thr_cos:
            blocks.append((s, i - 1)); s = i
    blocks.append((s, ne - 1))
    return blocks


CLUSTER_RAW = '--rawclust' in sys.argv[1:]                  # segment the RAW 84×84 matrix, not the 8-epoch one
edges = np.linspace(0, 14, 85)                              # bin edges (s) for raw block extents


def raw_cos(target):                                        # mean-over-mice 84×84 within-code axis cosine
    S = [unit_rows(W[(m, STAGE, 'all', target)]) for m in MICE if (m, STAGE, 'all', target) in W]
    return np.nanmean([U @ U.T for U in S], 0)


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
        Mdisp = raw_cos(target)                              # 84×84 raw cosine
        mblocks = blocks_raw(Mdisp)                          # bin blocks (incl. baseline)
        traj = mblocks[1:]                                   # drop baseline for trajectories
        windows = [(s, e + 1) for s, e in traj]
        tnames = [f'{edges[s]:.1f}–{edges[e + 1]:.1f}s' for s, e in traj]
    else:
        Mdisp = epoch_cos(target)                            # 8×8 epoch cosine
        mblocks = cluster_blocks(Mdisp)
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
        im_mat = axm.imshow(Mdisp, vmin=-1, vmax=1, cmap='RdBu_r', origin='upper',
                            extent=[0, 14, 14, 0], aspect='equal', interpolation='nearest')
        for (s, e) in mblocks:
            axm.add_patch(Rectangle((edges[s], edges[s]), edges[e + 1] - edges[s],
                                    edges[e + 1] - edges[s], fill=False, ec='k', lw=1.6))
        axm.set_xticks([0, 7, 14]); axm.set_yticks([0, 7, 14]); axm.tick_params(labelsize=7)
        axm.set_xlabel('time (s)', fontsize=7.5)
        subtitle = f'{len(mblocks)} blocks'
    else:
        im_mat = axm.imshow(Mdisp, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
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
    ok = (lz & (yt.performance.to_numpy() == 1) & (yt.tasks.to_numpy() == 'DPA')) if context == 'dpa' \
        else (lz & (yt.tasks.to_numpy() != 'DPA'))
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
_mode = 'raw 84×84 matrix' if CLUSTER_RAW else '8-epoch matrix'
fig.suptitle(f'Code blocks (left, clustered on the {_mode}) and each block read across test-time '
             f'(right) — {STAGE}   ·   {DLAB}', fontsize=13, y=0.975)
RAWSUF = '_rawclust' if CLUSTER_RAW else ''
p = os.path.join(OUT, 'png', f'overlaps_code_clusters_combined_{STAGE.lower()}{RAWSUF}{SUF}.png')
fig.savefig(p, dpi=200, bbox_inches='tight')
fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
plt.close(fig); print('saved', p)
