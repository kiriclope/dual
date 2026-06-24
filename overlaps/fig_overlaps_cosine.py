"""Cosine similarity of the overlaps discriminant axes across epochs and codes.

Uses the fold-averaged decoder WEIGHT VECTORS in raw neuron space saved by
`run_overlaps.py --save-weights` (weights_<DUM>.pkl). For each (mouse, stage, target)
the decoder gives one axis per train-time bin; we average those over an epoch window to
get one axis per (target, epoch), then take cosine similarity.

Cosines are computed PER MOUSE (within a mouse+stage the three targets share the same
neuron basis — you cannot take a cosine across mice) and then averaged across mice.

Two figures per stage:
  - stability : per code, epoch x epoch cosine heatmap (is the axis the same over time?)
  - alignment : between-code cosine vs epoch (sample-choice / sample-test / choice-test;
                near 0 = orthogonal — the dual-coding claim)
Output: figures/overlaps/cosine/{png,svg}/
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.options import set_options
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw'
DATA_IN = '../data/overlaps'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
TARGETS = ['sample', 'choice', 'test']
CONTEXT = 'all'

o = set_options(mice=MICE, tasks=['Dual'], mouse=MICE[0], laser=0, days=['first', 'last'],
                mne_estimator='generalizing')
_DELAY = o['bins_DELAY']; _LD = _DELAY[int(0.6 * len(_DELAY)):]
EPOCHS = [('stim', o['bins_STIM']), ('ed', o['bins_ED']), ('md', o['bins_MD']),
          ('gng_rwd', o['bins_RWD']), ('delay', _DELAY), ('ld', _LD),
          ('test', o['bins_TEST']), ('choice', o['bins_CHOICE']), ('dpa_rwd', o['bins_RWD2'])]
ENAMES = [e for e, _ in EPOCHS]
TEST_ONSET = o['bins_TEST'][0]
PAIRS = [('sample', 'choice'), ('sample', 'test'), ('choice', 'test')]

OUT = 'figures/overlaps/cosine'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def axis_per_epoch(ws):
    """ws (n_train, n_neurons) -> dict epoch -> unit axis (n_neurons,)."""
    return {e: unit(ws[bins].mean(0)) for e, bins in EPOCHS}


def save(fig, stem):
    p = os.path.join(OUT, 'png', f'{stem}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig); print('saved', p)


# ── load weights ──────────────────────────────────────────────────────────────
blob = pkl_load(f'weights_{DUM}', path=DATA_IN)
W = blob['weights']                       # {(mouse, stage, context, target): ws}
print(f'{len(W)} weight axes loaded')

# axes[(mouse,stage,target)] = {epoch: unit vector}
axes = {}
for mouse in MICE:
    for stage in STAGES:
        for target in TARGETS:
            key = (mouse, stage, CONTEXT, target)
            if key in W:
                axes[(mouse, stage, target)] = axis_per_epoch(np.asarray(W[key], float))

# ── 1) within-code stability: epoch x epoch cosine, mean over mice, per stage ──
ne = len(EPOCHS)
for stage in STAGES:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    im = None
    for ax, target in zip(axs, TARGETS):
        stack = []
        for mouse in MICE:
            a = axes.get((mouse, stage, target))
            if a is None:
                continue
            M = np.array([[float(np.dot(a[ei], a[ej])) for ej in ENAMES] for ei in ENAMES])
            stack.append(M)
        Mm = np.nanmean(np.stack(stack, 0), 0) if stack else np.full((ne, ne), np.nan)
        im = ax.imshow(Mm, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
        ax.set_xticks(range(ne)); ax.set_xticklabels(ENAMES, rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(ne)); ax.set_yticklabels(ENAMES, fontsize=8)
        ax.set_title(f'{target} axis  (n={len(stack)})', fontsize=11)
        for i in range(ne):
            for j in range(ne):
                ax.text(j, i, f'{Mm[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='k' if abs(Mm[i, j]) < 0.6 else 'w')
    if im is not None:
        fig.colorbar(im, ax=axs, fraction=0.012, pad=0.02, label='cosine')
    fig.suptitle(f'Within-code axis stability across epochs — {stage} '
                 f'(mean cosine over mice)', y=1.02, fontsize=13)
    save(fig, f'cosine_stability_{stage.lower()}')

# chance |cos| floor: two independent unit vectors in R^N have std(cos) ~ 1/sqrt(N)
Ns = [len(next(iter(a.values()))) for a in axes.values()]
chance = 1.0 / np.sqrt(np.mean(Ns))
print(f'mean n_neurons={np.mean(Ns):.0f}  -> chance |cos| ~ {chance:.3f}')

# ── 2) between-code alignment vs epoch, mean ± SEM over mice, per stage ────────
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, stage in zip(axs, STAGES):
    ax.axhspan(-chance, chance, color='0.85', zorder=0,
               label=f'chance ±1/√N ({chance:.2f})')
    for (t1, t2), color in zip(PAIRS, sns.color_palette('Set2', 3)):
        rows = []
        for mouse in MICE:
            a1 = axes.get((mouse, stage, t1)); a2 = axes.get((mouse, stage, t2))
            if a1 is None or a2 is None:
                continue
            rows.append([float(np.dot(a1[e], a2[e])) for e in ENAMES])
        if not rows:
            continue
        R = np.array(rows); mu = R.mean(0); sem = R.std(0, ddof=1) / np.sqrt(len(R))
        ax.plot(range(ne), mu, '-o', color=color, lw=1.8, ms=4, label=f'{t1}–{t2}')
        ax.fill_between(range(ne), mu - sem, mu + sem, color=color, alpha=0.2, lw=0)
    ax.axhline(0, ls='--', color='k', lw=0.7)
    ax.axvline(ENAMES.index('test') - 0.5, ls=':', color='0.5', lw=1)  # S2 onset
    ax.set_xticks(range(ne)); ax.set_xticklabels(ENAMES, rotation=60, ha='right', fontsize=8)
    ax.set_ylim(-1, 1); ax.set_title(stage, fontsize=12)
    ax.set_ylabel('cosine (signed)' if stage == STAGES[0] else '')
    ax.legend(frameon=False, fontsize=9)
fig.suptitle('Between-code axis alignment across epochs (mean ± SEM over mice; '
             '≈0 ⇒ orthogonal)', y=1.02, fontsize=13)
save(fig, 'cosine_alignment_between_codes')

print(f'\nCosine figures → {OUT}/')
