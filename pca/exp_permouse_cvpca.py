"""exp_permouse_cvpca.py — the per-mouse companion to Fig 2b's pooled cvPCA ("the memory is a
line"), built 2026-09-02 at user request to blunt the pseudo-population limitation: does the
one-dimensional memory spectrum hold ANIMAL BY ANIMAL, on each mouse's own simultaneously
recorded neurons?

Per mouse x stage x window (DPA set, 4 sample x test conditions -> at most 3 centred dims):
split-half cvPCA exactly as the pooled estimator (30 halvings, both cross directions averaged,
per-mouse condition-agnostic SD scaling on correct laser-off DPA trials), giving the reliable
spectrum, its top-1 fraction, and the reliable-variance TOTAL. Windows: 'md' (maintenance;
prediction top-1 ~ 1) vs 'decision' (prediction: spectrum spreads).

DISCLOSURE RULE (stated wherever plotted): cells with reliable-total < 5 are NOISE-LIMITED —
at their trial counts (min 6/cond required, typical 7-29) the split-half condition means carry
almost no replicating variance, so the top-1 FRACTION has a near-noise denominator; such cells
are drawn open/grey and excluded from the test. Stat: paired Wilcoxon top-1(md) vs
top-1(decision) across mice with BOTH cells valid, per stage.

Merge-dumps {'PM_CVPCA'} into results.pkl (estimator-free: no decoder, no SUF).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_permouse_cvpca.py
Output: figures/pseudo/dimensionality/{png,svg}/fig_permouse_cvpca.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
WINS = ['md', 'decision']
NSPLIT = 30
MIN_TRIALS = 6
TOT_MIN = 5.0                      # reliable-total floor for a trustworthy fraction

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])

def cvpca(mo, stage, win, seed):
    val = VALIDIX[(mo, stage)]
    if not len(val):
        return None
    M = AW[win]
    rng = np.random.RandomState(seed)
    pools = []
    for s in (0, 1):
        for t in (0, 1):
            m = ((MOUSE == mo) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                 & (TSK == 'DPA') & (SAMP == s) & (TESTO == t))
            pools.append(np.where(m)[0])
    if min(len(p) for p in pools) < MIN_TRIALS:
        return None
    X = np.nan_to_num(M[:, val])
    allc = np.concatenate(pools)
    sd = X[allc].std(0); sd = np.where(sd > 1e-6, sd, 1.0)
    specs = []
    for _ in range(NSPLIT):
        S1, S2 = [], []
        for p in pools:
            q = rng.permutation(p); h = len(q) // 2
            S1.append((X[q[:h]] / sd).mean(0)); S2.append((X[q[h:]] / sd).mean(0))
        S1 = np.array(S1) - np.mean(S1, 0); S2 = np.array(S2) - np.mean(S2, 0)
        def one(A, B):
            Vt = np.linalg.svd(A, full_matrices=False)[2]
            return np.einsum('ij,ij->j', A @ Vt.T, B @ Vt.T)
        specs.append(0.5 * (one(S1, S2) + one(S2, S1)))
    sp = np.mean(specs, 0)
    pos = np.clip(sp, 0, None); tot = float(pos.sum())
    frac = pos / tot if tot > 0 else pos
    return dict(frac=[float(f) for f in frac], top1=float(frac[0]) if tot > 0 else np.nan,
                total=tot, min_trials=int(min(len(p) for p in pools)), n_neurons=int(len(val)),
                ok=bool(tot >= TOT_MIN))

PM = {}
print(f'══ PM_CVPCA (DPA set; {NSPLIT} halvings; noise floor total ≥ {TOT_MIN}) ══')
for stage in STAGES:
    for mo in MICE:
        for wi, win in enumerate(WINS):
            r = cvpca(mo, stage, win, seed=100 + wi)
            if r is None:
                print(f'  {stage:6s} {mo:8s} {win:8s} <{MIN_TRIALS} trials/cond — skipped')
                continue
            PM[(mo, stage, win)] = r
            flag = '' if r['ok'] else '  [NOISE-LIMITED]'
            print(f"  {stage:6s} {mo:8s} {win:8s} top1={r['top1']:.2f} "
                  f"spec={np.round(r['frac'][:3], 2)} total={r['total']:5.1f} "
                  f"minTr={r['min_trials']:2d}{flag}")

for stage in STAGES:
    md = {m: PM[(m, stage, 'md')] for m in MICE if (m, stage, 'md') in PM}
    dc = {m: PM[(m, stage, 'decision')] for m in MICE if (m, stage, 'decision') in PM}
    both = [m for m in MICE if m in md and m in dc and md[m]['ok'] and dc[m]['ok']]
    a = np.array([md[m]['top1'] for m in both]); b = np.array([dc[m]['top1'] for m in both])
    okmd = [v['top1'] for v in md.values() if v['ok']]
    if len(both) >= 5:
        w = wilcoxon(a, b)
        print(f'{stage}: valid-md top1 median {np.median(okmd):.2f} (n={len(okmd)}); '
              f'md vs decision (n={len(both)} both-valid): {np.median(a):.2f} vs '
              f'{np.median(b):.2f}, Wilcoxon p={w.pvalue:.4f}, '
              f'{int((a > b).sum())}/{len(both)} mice md>decision')

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PM_CVPCA'] = PM
pickle.dump(d, open(RES, 'wb'))
print('merged PM_CVPCA into', RES)

# ── ED figure ──────────────────────────────────────────────────────────────────
import seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
_pal = sns.color_palette('tab10', n_colors=len(MICE))
MC = {m: _pal[i] for i, m in enumerate(MICE)}
import matplotlib.lines as mlines

fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
for ax, stage in zip(axs, STAGES):
    ns_valid = 0
    for m in MICE:
        pts = []
        for x, win in [(0, 'md'), (1, 'decision')]:
            r = PM.get((m, stage, win))
            if r is None:
                pts.append(None); continue
            pts.append((x, r['top1'], r['ok']))
        if pts[0] and pts[1] and pts[0][2] and pts[1][2]:
            ax.plot([0, 1], [pts[0][1], pts[1][1]], '-', color=MC[m], lw=0.8, alpha=0.5, zorder=2)
        for pt in pts:
            if pt is None:
                continue
            x, y, ok = pt
            ax.scatter(x, y, s=34, color=MC[m] if ok else 'none',
                       edgecolors=MC[m] if ok else '0.6', linewidths=0.8,
                       marker='o', zorder=3, alpha=1.0 if ok else 0.8)
    both = [m for m in MICE if all((m, stage, w) in PM and PM[(m, stage, w)]['ok'] for w in WINS)]
    a = np.array([PM[(m, stage, 'md')]['top1'] for m in both])
    b = np.array([PM[(m, stage, 'decision')]['top1'] for m in both])
    if len(both) >= 5:
        p = wilcoxon(a, b).pvalue
        sig = p < .05
        ax.text(0.5, 1.045, '*' if sig else 'n.s.', ha='center', fontsize=12 if sig else 8,
                fontweight='bold', color='k' if sig else '0.55')
        ax.text(0.5, 0.97, f'p={p:.3f} (n={len(both)})', ha='center', fontsize=6.5, color='0.3')
    ax.axhline(1 / 3, ls=':', color='0.6', lw=0.8)
    ax.text(1.38, 1 / 3, 'uniform\n(3 dims)', fontsize=6.0, color='0.5', va='center', ha='right')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['memory\n(mid-delay)', 'decision'])
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1.1)
    ax.set_title(stage, loc='left', fontsize=TITLE_FS)
axs[0].set_ylabel('top-1 reliable-variance fraction\n(per-mouse cvPCA, DPA set)')
axs[1].legend(handles=[
    mlines.Line2D([0], [0], marker='o', color='0.4', ls='none', ms=5, label='resolvable'),
    mlines.Line2D([0], [0], marker='o', mfc='none', color='0.6', ls='none', ms=5,
                  label='noise-limited (total<5)')],
    frameon=False, fontsize=6.0, loc='lower left')
fig.suptitle('Per mouse, the memory spectrum is one-dimensional and the decision spectrum spreads',
             x=0.02, ha='left', fontsize=TITLE_FS)
fig.tight_layout(rect=(0, 0, 1, 0.92))
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/fig_permouse_cvpca.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/fig_permouse_cvpca.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/fig_permouse_cvpca.png'))
