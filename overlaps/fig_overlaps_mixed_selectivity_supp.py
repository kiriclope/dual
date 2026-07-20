"""SUPPLEMENT — how can the GNG and choice codes be ORTHOGONAL when their panel-A traces look alike?
Because "orthogonal" is a property of the decoder AXES (weight vectors in neuron space), NOT of the
trial-averaged traces, and the two decoders use the SAME neurons weighted differently (mixed selectivity).

A  real data: |w| of the GNG-memory decoder (@bins_MD) vs the choice decoder (@57-63), one point per
   neuron (per-mouse z-scored, pooled over the 9 Expert mice) — the same cells weight into both.
B  real data: cos(GNG-memory axis, choice axis) per mouse, Naive→Expert, vs the 1/sqrt(N) chance floor.
C  toy: two neurons that BOTH fire for Go and for Lick; n1+n2 reads Go/NoGo and n1-n2 reads Lick/No-lick
   — two ORTHOGONAL axes from the SAME neurons.
D  selectivity/conjunction matrix: per-neuron significant tuning (Welch t-test, BH-FDR q<0.05) to each of
   the four task variables (sample / GNG / test / choice) at its window. Diagonal = fraction tuned to
   variable i; off-diagonal = fraction tuned to BOTH i and j. Off-diagonals sit at ~chance (product of the
   diagonals) → single-neuron tuning is sparse and roughly INDEPENDENT across variables (little excess
   conjunctive tuning), which is itself what lays the variables out on near-orthogonal population axes.
   (NB test & choice are correlated on correct DPA trials, so that off-diagonal is if anything inflated.)
Output: figures/overlaps/controls/{png,svg}/overlaps_mixed_selectivity.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from src.pca.io import pkl_load
from src.common.options import set_options

# ── house style (matches fig_overlaps_main_native.py; see CLAUDE.md) ──
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
_pal = sns.color_palette('tab10', n_colors=len(MICE)); MC = {m: _pal[i] for i, m in enumerate(MICE)}
MD = np.asarray(set_options()['bins_MD']); ACT = np.arange(57, 63)
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
Wb = pkl_load(f'weights_{BDUM}', path='../data/overlaps'); W, VALID = Wb['weights'], Wb['valid']


def axes_mouse(m, stage):
    wg = np.asarray(W[(m, stage, 'all', 'gng')])[MD].mean(0)
    wc = np.asarray(W[(m, stage, 'all', 'choice')])[ACT].mean(0)
    return wg, wc


# ── A/B inputs: per-neuron |w| + per-mouse overlap / cos ──
zx, zy, cat, overlaps, cosN, cosE = [], [], [], [], [], []
for m in MICE:
    wg, wc = axes_mouse(m, 'Expert'); ag, ac = np.abs(wg), np.abs(wc)
    tg, tc = ag >= np.percentile(ag, 80), ac >= np.percentile(ac, 80)
    cat.append(np.where(tg & tc, 3, np.where(tg, 1, np.where(tc, 2, 0))))
    zx.append((ac - ac.mean()) / (ac.std() + 1e-9)); zy.append((ag - ag.mean()) / (ag.std() + 1e-9))
    overlaps.append((tg & tc).sum() / max(1, tg.sum()))
    for st, out in [('Naive', cosN), ('Expert', cosE)]:
        g, ch = axes_mouse(m, st); out.append(float(g @ ch / (np.linalg.norm(g) * np.linalg.norm(ch) + 1e-12)))
zx = np.concatenate(zx); zy = np.concatenate(zy); cat = np.concatenate(cat)
rho = np.mean([spearmanr(np.abs(axes_mouse(m, 'Expert')[0]), np.abs(axes_mouse(m, 'Expert')[1]))[0] for m in MICE])
chance = np.mean([1 / np.sqrt(len(axes_mouse(m, 'Naive')[0])) for m in MICE])

# ── D input: per-neuron selectivity to the 4 variables (Welch t + BH-FDR) ──
print('loading X_all …', flush=True)
Xall = np.asarray(pkl_load('X_all_nan_', path='../data/pca')); yall = pkl_load('y_all_nan_', path='../data/pca')
yall['gng'] = np.where(yall.tasks.to_numpy() == 'DualGo', 1.0,           # derive gng (not in raw labels): Go=1 / NoGo=0
                       np.where(yall.tasks.to_numpy() == 'DualNoGo', 0.0, np.nan))
lo = (yall.laser == 0).to_numpy(); dpa = (yall.tasks == 'DPA').to_numpy(); mo = yall.mouse.to_numpy()
# Each variable read at ITS OWN DECODER window (same train-bin averaging as the main-figure codes):
# sample 16-48, GNG bins_MD (36-38), test 58-84, choice 57-63. NB the Go-response lick coincides with
# ~bins_MD (peaks ~5.2-5.8s post-sample; neural↔behaviour alignment uncertain ~1s), so the GNG diagonal
# read here may include the Go-lick (Go=lick/NoGo=withhold) as well as the Go/NoGo odour — see report.
VARS = [('sample', np.arange(16, 48), 'sample_odor', True), ('GNG', MD, 'gng', False),
        ('test', np.arange(58, 84), 'test_odor', True), ('choice', np.arange(57, 63), 'choice', True)]
VLAB = [v[0] for v in VARS]
SEL = []                                                     # per neuron: boolean tuning to each of the 4 variables
for m in MICE:
    val = VALID[(m, 'Expert')]; nsig = np.zeros((int(val.sum()), 4), bool)
    for j, (nm, win, col, is_dpa) in enumerate(VARS):
        idx = lo & (mo == m) & (dpa if is_dpa else ~dpa)
        A = np.nanmean(Xall[idx][:, val, :][:, :, win], axis=2)   # (ntr, nneur)
        lab = yall.loc[idx, col].to_numpy().astype(float); ok = np.isfinite(lab)
        A, lab = A[ok], lab[ok]
        a, b = A[lab == 1], A[lab == 0]
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = ttest_ind(a, b, axis=0, equal_var=False, nan_policy='omit')
        p = np.where(np.isfinite(p), p, 1.0)
        nsig[:, j] = fdrcorrection(p, alpha=0.05)[0]
    SEL.append(nsig)
SEL = np.concatenate(SEL, 0)                                  # (n_neurons, 4)
Mmat = np.zeros((4, 4))
for i in range(4):
    Mmat[i, i] = SEL[:, i].mean()                            # diag: fraction of neurons tuned to variable i
    for j in range(4):
        if i != j:
            Mmat[i, j] = (SEL[:, i] & SEL[:, j]).mean()       # off-diag: fraction tuned to BOTH i and j (conjunctive)
mixed = (SEL.sum(1) >= 2).mean(); anysel = (SEL.sum(1) >= 1).mean()
print('per-variable selectivity:', {VLAB[i]: round(float(Mmat[i, i]), 3) for i in range(4)},
      '| any>=1 %.0f%% mixed>=2 %.0f%%' % (100 * anysel, 100 * mixed))

# ══ FIGURE (2×2) ══
fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.6))
# A: same neurons carry both
a = ax[0, 0]
COL = {0: '0.75', 1: '#1f77b4', 2: '#4daf4a', 3: '#d62728'}; LAB = {1: 'GNG top-20%', 2: 'choice top-20%', 3: 'top-20% in BOTH'}
for cc in (0, 1, 2, 3):
    s = cat == cc; a.scatter(zx[s], zy[s], s=3, color=COL[cc], alpha=0.35 if cc == 0 else 0.7, lw=0, label=LAB.get(cc))
a.set_xlabel('|weight| in choice decoder (z)'); a.set_ylabel('|weight| in GNG decoder (z)')
a.set_title('overlapping but largely distinct neurons', loc='left', fontsize=TITLE_FS)
a.text(0.96, 0.06, f'ρ(|w|)={rho:+.2f}\ntop-20% overlap {np.mean(overlaps):.0%}\n(20% by chance)', transform=a.transAxes, ha='right', va='bottom', fontsize=6)
a.legend(frameon=False, fontsize=5.5, loc='upper left', handletextpad=0.2, markerscale=1.6)
a.text(-0.26, 1.05, 'A', transform=a.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
# B: axes near-orthogonal
b = ax[0, 1]
for i, m in enumerate(MICE):
    b.plot([0, 1], [cosN[i], cosE[i]], '-o', color=MC[m], lw=0.8, ms=4, mec='w', mew=0.4, zorder=3)
for x, v in ((-0.16, cosN), (1.16, cosE)):
    b.errorbar(x, np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)), fmt='s', color='k', ms=6, capsize=3.5, lw=1.3, zorder=5)
b.axhline(chance, ls=':', color='0.6', lw=0.8); b.axhline(0, color='0.85', lw=0.6)
b.text(1.5, chance, 'chance', fontsize=5.5, color='0.6', va='bottom', ha='right')
b.set_xticks([0, 1]); b.set_xticklabels(['Naive', 'Expert']); b.set_xlim(-0.5, 1.6); b.set_ylabel('cos(GNG axis · choice axis)')
b.set_title('axes near-orthogonal', loc='left', fontsize=TITLE_FS)
b.text(0.5, 0.97, f'|cos|≈{np.mean(np.abs(cosE)):.2f} (Expert) → ~{np.degrees(np.arccos(np.mean(np.abs(cosE)))):.0f}° apart',
       transform=b.transAxes, ha='center', va='top', fontsize=6, color='0.3')
b.text(-0.24, 1.05, 'B', transform=b.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
# C: toy
c = ax[1, 0]
rng = np.random.default_rng(0); n = 90; go = rng.choice([-1, 1], n); lk = rng.choice([-1, 1], n)
pts = 0.62 * (go[:, None] * np.array([1, 1]) + lk[:, None] * np.array([1, -1])) + rng.normal(0, 0.42, (n, 2))
for gv, lv, col, mk in [(1, 1, '#1f77b4', 'o'), (1, -1, '#8db6d8', 'o'), (-1, 1, '#4daf4a', '^'), (-1, -1, '#a6d89f', '^')]:
    s = (go == gv) & (lk == lv)
    c.scatter(pts[s, 0], pts[s, 1], s=12, color=col, marker=mk, lw=0, alpha=0.9, label=f'{"Go" if gv > 0 else "NoGo"}·{"Lick" if lv > 0 else "Nolick"}')
for vec, lab, col in [(np.array([1, 1]), 'GNG axis\n(n1+n2)', '#d62728'), (np.array([1, -1]), 'choice axis\n(n1−n2)', '#7030a0')]:
    v = vec / np.linalg.norm(vec) * 2.3
    c.annotate('', xy=v, xytext=-v, arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6))
    c.text(v[0] * 1.02, v[1] * 1.02, lab, color=col, fontsize=6, ha='center', va='center', fontweight='bold')
c.set_xlabel('neuron 1'); c.set_ylabel('neuron 2'); c.set_xlim(-3, 3); c.set_ylim(-3, 3); c.set_aspect('equal')
c.axhline(0, color='0.85', lw=0.6); c.axvline(0, color='0.85', lw=0.6)
c.set_title('toy: orthogonal readouts, same neurons', loc='left', fontsize=TITLE_FS)
c.legend(frameon=False, fontsize=5, loc='lower left', handletextpad=0.1, labelspacing=0.2, borderpad=0.1)
c.text(-0.26, 1.05, 'C', transform=c.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
# D: selectivity / conjunction matrix — diag = tuned to variable i; off-diag = tuned to BOTH i & j
d = ax[1, 1]
vmax = Mmat.max() * 100
im = d.imshow(Mmat * 100, cmap='YlOrRd', vmin=0, vmax=vmax)
for i in range(4):
    for j in range(4):
        v = Mmat[i, j] * 100; tc = 'white' if v > 0.55 * vmax else '0.15'
        if i == j:
            d.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=7.5, fontweight='bold', color=tc)
        else:
            d.text(j, i, f'{v:.1f}%\n({Mmat[i, i]*Mmat[j, j]*100:.1f})', ha='center', va='center', fontsize=5.2, color=tc)
d.set_xticks(range(4)); d.set_xticklabels(VLAB); d.set_yticks(range(4)); d.set_yticklabels(VLAB)
d.tick_params(length=0)
d.set_title('tuned to one (diag) vs both (off-diag)', loc='left', fontsize=TITLE_FS)
d.text(0.5, -0.19, 'each variable read at ITS decoder window (sample 16-48, GNG bins_MD, test 58-84, choice 57-63).\n'
       'off-diag = observed % (chance in parens = product of diagonals); stimulus pairs ≈ chance.\n'
       f'NB the Go-response lick coincides with ~bins_MD, so the GNG diagonal may include it. mixed(≥2)={mixed*100:.0f}%',
       transform=d.transAxes, ha='center', va='top', fontsize=5.1, color='0.3')
cb = fig.colorbar(im, ax=d, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6); cb.set_label('% of neurons', fontsize=6)
d.text(-0.32, 1.05, 'D', transform=d.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')

fig.suptitle('The GNG and choice codes are near-orthogonal — carried by largely separate, '
             'independently-tuned neurons (similar-looking traces ≠ aligned axes)', fontsize=9, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT = 'figures/overlaps/controls'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
p = f'{OUT}/png/overlaps_mixed_selectivity.png'
fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
