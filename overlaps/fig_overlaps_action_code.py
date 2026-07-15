"""Shared ACTION/LICK code between DPA and GNG (draft panel for the main figure).

The DPA choice decoder is a lick/no-lick decoder (match→lick); the GNG decoder is a Go/NoGo
lick decoder (Go→lick). Both are lick-command readouts for their own task. Are they the SAME
neural direction — a shared action code?

Measure: signed cosine between the choice axis (rows) and the gng axis (cols) at every pair of
epochs, per mouse (shared neuron basis), averaged over 9 mice. Chance |cos| ≈ 1/sqrt(n̄) ≈ 0.05.

Result: orthogonal during the delay (same-time diagonal ≈ chance) but the two axes ALIGN
off-diagonally at each task's ACTION/REWARD moment — DPA response/reward (bins 60–84) × GNG
reward/late-delay (bins 42–60) — and this shared-action alignment STRENGTHENS with learning
(Naive ≈ +0.14 → Expert ≈ +0.25, ~5× chance). The lick command is one shared direction, engaged
at whichever time each task calls for the action.

Output: figures/overlaps/action/{png,svg}/overlaps_action_code.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DATA = '../data/overlaps'
DUM = 'weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
EP = [('stim', 12, 18), ('eDel', 18, 27), ('distr', 27, 33), ('mDel', 33, 39), ('cue', 39, 42),
      ('gngRwd', 42, 45), ('lDel', 45, 54), ('test', 54, 60), ('resp', 60, 72), ('dpaRwd', 72, 84)]
# ACTION/REWARD cross-block: choice at DPA response+reward (60–84) × gng at GNG reward..test (42–60)
CH_BLK = (60, 84)      # DPA lick/reward epochs (rows)
GN_BLK = (42, 60)      # GNG reward/late-delay epochs (cols)

W = pkl_load(DUM, path=DATA)['weights']
Ns = [np.asarray(W[(m, 'Expert', 'all', 'choice')]).shape[1] for m in MICE if (m, 'Expert', 'all', 'choice') in W]
CHANCE = 1.0 / np.sqrt(np.mean(Ns))
_pal = sns.color_palette('tab10', n_colors=len(MICE))
MC = {m: _pal[i] for i, m in enumerate(MICE)}


def _unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v
def _epaxis(m, st, tg, a, c):
    k = (m, st, 'all', tg)
    return _unit(np.asarray(W[k], float)[a:c].mean(0)) if k in W else None
def _epmat(st):                                            # (10 choice-ep, 10 gng-ep) mean signed cos
    per = []
    for m in MICE:
        row = np.full((len(EP), len(EP)), np.nan)
        for i, (_, a1, c1) in enumerate(EP):
            A = _epaxis(m, st, 'choice', a1, c1)
            for j, (_, a2, c2) in enumerate(EP):
                B = _epaxis(m, st, 'gng', a2, c2)
                if A is not None and B is not None:
                    row[i, j] = A @ B
        per.append(row)
    return np.nanmean(per, 0)
def _blockcos(m, st):                                      # per-mouse action/reward block mean cos
    A = _epaxis(m, st, 'choice', *CH_BLK); B = _epaxis(m, st, 'gng', *GN_BLK)
    return float(A @ B) if (A is not None and B is not None) else np.nan


fig = plt.figure(figsize=(9.5, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.45, left=0.07, right=0.98, bottom=0.18, top=0.86)

# ── two heatmaps: Naive | Expert (choice rows × gng cols) ──────────────────────
ep_lab = [n for n, _, _ in EP]
bi = {n: k for k, (n, _, _) in enumerate(EP)}
box_r = (bi['resp'], bi['dpaRwd']); box_c = (bi['gngRwd'], bi['test'])       # boxed action/reward block
for ci, st in enumerate(('Naive', 'Expert')):
    ax = fig.add_subplot(gs[0, ci])
    M = _epmat(st)
    im = ax.imshow(M, vmin=-0.25, vmax=0.25, cmap='RdBu_r', aspect='equal')
    ax.add_patch(Rectangle((box_c[0] - 0.5, box_r[0] - 0.5), box_c[1] - box_c[0] + 1, box_r[1] - box_r[0] + 1,
                           fill=False, ec='k', lw=1.8))
    ax.set_xticks(range(len(EP))); ax.set_xticklabels(ep_lab, rotation=90, fontsize=6)
    ax.set_yticks(range(len(EP))); ax.set_yticklabels(ep_lab, fontsize=6)
    ax.set_title(st, fontsize=10, fontweight='bold')
    if ci == 0:
        ax.set_ylabel('choice code (DPA lick)', fontsize=8)
    ax.set_xlabel('gng code (Go/NoGo lick)', fontsize=8)
    ax.tick_params(length=2)
cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.025, pad=0.02, label='signed cos')
cb.ax.tick_params(labelsize=7)

# ── paired Naive→Expert of the action-block cosine ─────────────────────────────
axp = fig.add_subplot(gs[0, 2])
N = np.array([_blockcos(m, 'Naive') for m in MICE]); E = np.array([_blockcos(m, 'Expert') for m in MICE])
for i, m in enumerate(MICE):
    axp.plot([0, 1], [N[i], E[i]], '-o', color=MC[m], lw=1.0, ms=5, mec='w', mew=0.5, zorder=3)
for x, v in ((-0.18, N), (1.18, E)):
    mu = np.nanmean(v); se = np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())
    axp.errorbar(x, mu, yerr=se, fmt='s', color='k', ms=7, capsize=4, lw=1.5, zorder=5)
axp.axhline(CHANCE, ls=':', color='0.5', lw=1); axp.axhline(-CHANCE, ls=':', color='0.5', lw=1)
axp.axhline(0, color='0.85', lw=0.6)
axp.text(1.5, CHANCE, 'chance', fontsize=6.5, color='0.5', va='bottom', ha='right')
axp.set_xticks([0, 1]); axp.set_xticklabels(['Naive', 'Expert']); axp.set_xlim(-0.5, 1.6)
axp.set_ylabel('action-code alignment\ncos(choice@resp/rwd, gng@rwd)', fontsize=8)
pN = ttest_1samp(N[np.isfinite(N)], 0).pvalue; pE = ttest_1samp(E[np.isfinite(E)], 0).pvalue
pNE = ttest_1samp((E - N)[np.isfinite(E - N)], 0).pvalue
axp.set_title('shared action code', fontsize=10, fontweight='bold')
axp.text(0.5, 0.02, f'N={np.nanmean(N):+.2f} (p={pN:.3f})  E={np.nanmean(E):+.2f} (p={pE:.3f})\n'
                    f'Δ paired-t p={pNE:.3f}', transform=axp.transAxes, ha='center', va='bottom',
         fontsize=6.5, color='0.3')
sns.despine(ax=axp)

fig.suptitle('Shared action/lick code — DPA choice and GNG codes align at each task’s action/reward moment',
             fontsize=10, y=0.99)
OUT = 'figures/overlaps/action'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
p = f'{OUT}/png/overlaps_action_code.png'
fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
print(f'chance|cos|={CHANCE:.3f}  action-block Naive={np.nanmean(N):+.3f} (p={pN:.3f})  '
      f'Expert={np.nanmean(E):+.3f} (p={pE:.3f})  Δ p={pNE:.3f}')
