"""Cross-stage Naive→Expert orthogonalization test on the overlaps decoder axes.

Question: do the code axes become more orthogonal with learning? Using the raw-neuron-space
weight vectors (weights_<DUM>.pkl from run_overlaps.py --save-weights), per mouse we take the
delay-window axis of each code (mean over ed/md/delay/ld train bins) and the cosine between
code pairs. Cosines are per-mouse (shared neuron basis), paired Naive vs Expert across 9 mice.

Confound guard: a drop toward the chance floor could be mere axis NOISE in Expert. Control =
within-code self-stability (ed·ld cosine) per stage; if it is unchanged, the Expert axes are
no noisier, so a between-code drop is real orthogonalization.

Result (2026-06-24): only the sample-choice (memory x action) pair orthogonalizes with learning
  - sample-choice: signed delay cos -0.068 -> -0.010 (mildly anti-aligned in Naive -> orthogonal
    in Expert); |cos| 0.083 -> 0.029; paired Wilcoxon p=0.020, 7/9 mice decrease.
  - sample-test: null (p=0.074, wrong direction).  choice-test: trend only (p=0.098).
  - Confound ruled out: within-code ed.ld self-stability is identical across stages
    (sample 0.574/0.582, choice 0.422/0.426, test 0.465/0.468; all p>0.8) -> Expert axes are not
    noisier, so the drop is real orthogonalization, not regression toward the 1/sqrt(N) floor.
  Be precise: "orthogonal" = at the chance floor (statistically independent), not pushed beyond it.

Output: figures/overlaps/cosine/{png,svg}/orthogonalization_naive_expert.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from src.common.options import set_options
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw'
DATA_IN = '../data/overlaps'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
PAIRS = [('sample', 'choice'), ('sample', 'test'), ('choice', 'test')]
CODES = ['sample', 'choice', 'test']

o = set_options(mice=MICE, tasks=['Dual'], mouse=MICE[0], laser=0, days=['first', 'last'],
                mne_estimator='generalizing')
_D = o['bins_DELAY']; _LD = _D[int(0.6 * len(_D)):]
EP = {'ed': o['bins_ED'], 'md': o['bins_MD'], 'delay': _D, 'ld': _LD}
DELAY_EP = ['ed', 'md', 'delay', 'ld']

OUT = 'figures/overlaps/cosine'
for sub in ('png', 'svg'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

W = pkl_load(f'weights_{DUM}', path=DATA_IN)['weights']


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v


def axis(mouse, stage, tgt, ep):
    return unit(np.asarray(W[(mouse, stage, 'all', tgt)], float)[EP[ep]].mean(0))


def delay_cos(mouse, stage, t1, t2):
    return float(np.mean([np.dot(axis(mouse, stage, t1, e), axis(mouse, stage, t2, e))
                          for e in DELAY_EP]))


N = [len(np.asarray(W[(m, 'Naive', 'all', 'sample')])[0]) for m in MICE]
chance = 1.0 / np.sqrt(np.mean(N))

# ── figure: 3 between-code pairs (paired |cos|) + 1 reliability control ────────
C_NAI, C_EXP = '#9ecae1', '#2171b5'
fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))

for ax, (t1, t2) in zip(axs[:3], PAIRS):
    nai = np.array([abs(delay_cos(m, 'Naive', t1, t2)) for m in MICE])
    exp = np.array([abs(delay_cos(m, 'Expert', t1, t2)) for m in MICE])
    sgn_n = np.mean([delay_cos(m, 'Naive', t1, t2) for m in MICE])
    sgn_e = np.mean([delay_cos(m, 'Expert', t1, t2) for m in MICE])
    p = wilcoxon(exp, nai).pvalue
    ax.axhspan(0, chance, color='0.88', zorder=0, label=f'chance (1/√N={chance:.2f})')
    for a, b in zip(nai, exp):
        ax.plot([0, 1], [a, b], '-', color='0.7', lw=0.8, zorder=1)
    ax.plot(np.zeros_like(nai), nai, 'o', color=C_NAI, ms=6, zorder=2)
    ax.plot(np.ones_like(exp), exp, 'o', color=C_EXP, ms=6, zorder=2)
    ax.plot([0, 1], [nai.mean(), exp.mean()], '-k', lw=2.2, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'])
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, max(0.25, nai.max() * 1.15))
    ax.set_title(f'{t1}–{t2}\nsigned {sgn_n:+.3f}→{sgn_e:+.3f}  '
                 f'(Wilcoxon p={p:.3f})', fontsize=10)
    if (t1, t2) == PAIRS[0]:
        ax.set_ylabel('|cosine|  (delay axes)')
    ax.legend(frameon=False, fontsize=7, loc='upper right')

# reliability control: within-code ed·ld self-cosine, Naive vs Expert
axc = axs[3]
for ci, code in enumerate(CODES):
    nai = np.array([np.dot(axis(m, 'Naive', code, 'ed'), axis(m, 'Naive', code, 'ld')) for m in MICE])
    exp = np.array([np.dot(axis(m, 'Expert', code, 'ed'), axis(m, 'Expert', code, 'ld')) for m in MICE])
    p = wilcoxon(exp, nai).pvalue
    x0, x1 = ci - 0.18, ci + 0.18
    for a, b in zip(nai, exp):
        axc.plot([x0, x1], [a, b], '-', color='0.8', lw=0.6, zorder=1)
    axc.plot([x0] * len(nai), nai, 'o', color=C_NAI, ms=4, zorder=2)
    axc.plot([x1] * len(exp), exp, 'o', color=C_EXP, ms=4, zorder=2)
    axc.plot([x0, x1], [nai.mean(), exp.mean()], '-k', lw=2, zorder=3)
    axc.text(ci, 0.97, f'p={p:.2f}', ha='center', fontsize=7, transform=axc.get_xaxis_transform())
axc.set_xticks(range(len(CODES))); axc.set_xticklabels(CODES)
axc.set_ylim(0, 1); axc.set_ylabel('ed·ld self-cosine')
axc.set_title('reliability control\n(within-code stability — unchanged)', fontsize=10)
from matplotlib.lines import Line2D
axc.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NAI, label='Naive'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_EXP, label='Expert')],
           frameon=False, fontsize=8, loc='lower right')

fig.suptitle('Naive→Expert orthogonalization of the code axes (delay window; paired across 9 mice). '
             'Only sample–choice orthogonalizes; axis reliability is unchanged.',
             y=1.03, fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, 'png', 'orthogonalization_naive_expert.png')
fig.savefig(p, dpi=300, bbox_inches='tight')
fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
plt.close(fig); print('saved', p)
