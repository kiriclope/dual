"""SUPPLEMENT — per-cluster generalization of each code across test-time.

Each code's discriminant axis segments into contiguous temporal blocks (clusters) where the axis
is stable (see fig_overlaps_cosine_matrices.py --clustered). For each cluster we take the decoder(s)
TRAINED in that block's bins (average the generalising tensor over those train bins) and read the
decision function at EVERY test-time bin — i.e. how a sub-code trained in one epoch represents its
dimension across the whole trial.

Clusters = that code's Expert blocks (complete-linkage, all-pairs cos>0.5), used as fixed windows
for BOTH stages so Naive vs Expert are comparable. One figure per code (sample/choice/test/GNG):
rows = stage, cols = cluster. Each panel: per-mouse BL-z class decision function (mean ± SEM over
mice), correct DPA trials (Dual trials for GNG); the cluster's own train window is shaded.

Decoder variant mirrors the main figure: default ridge; --l1 / --lda load the matching bundled tensor.
Output: figures/overlaps/cosine/{png,svg}/overlaps_<code>_cluster_traj[_l1|_lda].{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
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
STAGES = ['Naive', 'Expert']
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

# code -> (title, split column, [(level,label,colour)...], context)   context: 'dpa' or 'gng'
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


def epoch_cos(stage, target):
    stack = []
    for m in MICE:
        k = (m, stage, 'all', target)
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


CLUSTER_RAW = '--rawclust' in sys.argv[1:]
edges = np.linspace(0, 14, 85)


def raw_cos(stage, target):                                 # mean-over-mice 84×84 within-code cosine
    S = [unit_rows(W[(m, stage, 'all', target)]) for m in MICE if (m, stage, 'all', target) in W]
    return np.nanmean([U @ U.T for U in S], 0)


def blocks_raw(M, thr=0.5, start=12, min_len=8):
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


def draw_code(title, target, col, levels, context):
    if CLUSTER_RAW:                                          # raw 84×84 blocks (Expert), drop baseline
        traj = blocks_raw(raw_cos('Expert', target))[1:]
        windows = [(s, e + 1) for s, e in traj]
        bnames  = [f'{edges[s]:.1f}–{edges[e + 1]:.1f}s' for s, e in traj]
    else:                                                   # 10-epoch complete-linkage blocks (Expert)
        blocks = cluster_blocks(epoch_cos('Expert', target))
        windows = [(EPOCHS[i][1], EPOCHS[j][2]) for i, j in blocks]
        bnames  = [ENAMES[i] if i == j else f'{ENAMES[i]}–{ENAMES[j]}' for i, j in blocks]
    print(f'{title}: ' + '  |  '.join(f'{n} (bins {a}-{b})' for n, (a, b) in zip(bnames, windows)))

    tm = (yb.target == target).to_numpy()
    Xt = Xb[tm][:, 1].astype(float)                         # (n, 84_train, 84_test)
    yt = yb[tm].reset_index(drop=True)
    mouse = yt.mouse.to_numpy(); stg = yt.learning.to_numpy()
    lz = yt.laser.to_numpy() == 0
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

    nR, nC = len(STAGES), len(windows)
    fig, axs = plt.subplots(nR, nC, figsize=(3.4 * nC, 5.4), sharex=True, sharey='row', squeeze=False)
    for wi, ((a, b), bn) in enumerate(zip(windows, bnames)):
        Z = blz(Xt[:, a:b, :].mean(1))
        for ri, sg in enumerate(STAGES):
            ax = axs[ri][wi]
            ax.axvspan(edges[a], edges[min(b, 84)], color='0.88', zorder=0)
            draw_events(ax, label=(ri == 0 and wi == 0))
            ax.axhline(0, color='k', lw=0.6, ls='--')
            for lv, lab, cclr in levels:
                per = [np.nanmean(Z[ok & (stg == sg) & (lab_v == lv) & (mouse == mo)], 0)
                       for mo in MICE if (ok & (stg == sg) & (lab_v == lv) & (mouse == mo)).sum() >= 3]
                if len(per) >= 2:
                    Mm = np.stack(per); mu = Mm.mean(0); se = Mm.std(0, ddof=1) / np.sqrt(len(per))
                    ax.plot(xtime, mu, color=cclr, lw=1.7, label=f'{lab} (n={len(per)})', zorder=2)
                    ax.fill_between(xtime, mu - se, mu + se, color=cclr, alpha=0.2, lw=0)
            ax.set_xlim(0, 14); ax.set_xticks([0, 2, 4.5, 9, 14])
            if ri == 0:
                ax.set_title(f'cluster: {bn}\n(train bins {a}–{b})', fontsize=9)
                ax.legend(fontsize=6.5, frameon=False, loc='upper left', handlelength=1.1)
            if ri == nR - 1:
                ax.set_xlabel('test time (s)', fontsize=9)
            if wi == 0:
                ax.set_ylabel(f'{sg}\n{title} code (z)', fontsize=9)
            sns.despine(ax=ax)
    fig.suptitle(f'{title}-code clusters read across test-time (train window shaded) — {DLAB}\n'
                 f'each panel: decoder trained in the cluster window, decision function at every test-time bin',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    RAWSUF = '_rawclust' if CLUSTER_RAW else ''
    p = os.path.join(OUT, 'png', f'overlaps_{title.lower()}_cluster_traj{RAWSUF}{SUF}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('saved', p)


for spec in CODES:
    draw_code(*spec)
print(f'\nPer-cluster trajectory supplement → {OUT}/')
