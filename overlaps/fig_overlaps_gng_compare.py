"""
Dedicated Go-vs-NoGo (GNG) decoder vs the choice axis / current "task code".

The main figure's "task code" is the CHOICE decoder's decision function split by
DualGo/DualNoGo — it is NOT a Go/NoGo decoder. Here we train a real GNG decoder
(target='gng' = DualGo vs DualNoGo, trained on Dual trials, generalizing across all
time; produced by `run_overlaps.py --scaler none --targets gng --save-weights`) and
compare it to:
  (1) DISCRIMINABILITY — dedicated GNG decoder d'(Go,NoGo) vs the choice decoder's
      d'(Go,NoGo) [= the current task readout], per mouse.
  (2) GEOMETRY — |cos| between the GNG axis and the choice/sample/test axes (weights).
  (3) DYNAMICS — GNG decision-function trace (Go vs NoGo) over time + its cross-temporal
      generalization matrix (where it decodes and how it generalizes).

Output: figures/overlaps/gng/{png,svg}/gng_compare.{png,svg}
"""
import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from src.common.options import set_options
from src.pca.io import pkl_load
from src.common.plot_utils import add_vlines
import seaborn as sns; sns.set_context('notebook')   # plot_utils sets poster context at import — undo it

ALL = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
DATA_IN = '../data/overlaps'
CANON = 'log_generalizing_overlaps_none_l1_ratio_0.0'          # sample/choice/test tensor + _raw weights
GNG   = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_gng'
o = set_options(mice=ALL, tasks=['Dual'], mouse=ALL[0], laser=0, trials='', data_type='dF',
                prescreen=None, pval=0.05, preprocess=None, scaler_BL='standard_BL', avg_noise=False,
                unit_var_BL=False, random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca',
                scaler=None, bootstrap=1, n_boots=128, n_splits=5, n_repeats=10, class_weight=0,
                multilabel=0, mne_estimator='generalizing', n_jobs=4, days=['first', 'last'])
xt = np.linspace(0, 14, 84); BL = slice(0, 12); LD = o['bins_LD']; MD = o['bins_MD']; TEST_ONSET = o['bins_TEST'][0]

print('loading tensors …')
Xg = pkl_load(f'X_{GNG}', path=DATA_IN); yg = pkl_load(f'labels_{GNG}', path=DATA_IN)
Xc = pkl_load(f'X_{CANON}', path=DATA_IN); yc = pkl_load(f'labels_{CANON}', path=DATA_IN)
Wg = pkl_load(f'weights_{GNG}', path=DATA_IN)
Wc = pkl_load(f'weights_{CANON}_raw', path=DATA_IN)
Xg1 = Xg[:, 1].astype(np.float32); Xc1 = Xc[:, 1].astype(np.float32)   # toward-"1" pole decision fn
mg, sg, tg = yg.mouse.values, yg.stage.values, yg.tasks.values
mc, sc, tc = yc.mouse.values, yc.stage.values, yc.tasks.values


def _dprime(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5: return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


def tgm_gng():
    """per-mouse d'(Go,NoGo) train×test for the DEDICATED gng decoder, mean over mice."""
    mats = []
    for m in ALL:
        base = (yg.target == 'gng').values & (mg == m) & (yg.laser == 0).values & (tg != 'DPA')
        go = base & (yg.gng.values == 1); ng = base & (yg.gng.values == 0)
        if go.sum() < 5 or ng.sum() < 5: continue
        A = Xg1[go]; B = Xg1[ng]; ps = np.sqrt((A.var(0, ddof=1) + B.var(0, ddof=1)) / 2)
        mats.append(np.where(ps > 0, (A.mean(0) - B.mean(0)) / ps, 0.0))
    return np.nanmean(mats, 0)


def tgm_choice_on_gng():
    """choice decoder's d'(Go,NoGo) train×test = the CURRENT task-code approach."""
    mats = []
    for m in ALL:
        base = (yc.target == 'choice').values & (mc == m) & (yc.laser == 0).values & (tc != 'DPA')
        go = base & (tc == 'DualGo'); ng = base & (tc == 'DualNoGo')
        if go.sum() < 5 or ng.sum() < 5: continue
        A = Xc1[go]; B = Xc1[ng]; ps = np.sqrt((A.var(0, ddof=1) + B.var(0, ddof=1)) / 2)
        mats.append(np.where(ps > 0, (A.mean(0) - B.mean(0)) / ps, 0.0))
    return np.nanmean(mats, 0)


def plateau(G):
    d = np.diag(G).copy(); d[:12] = -1e9
    peak = int(np.argmax(d)); thr = 0.5 * d[peak]; row = G[peak]
    a = peak
    while a - 1 >= 12 and row[a - 1] >= thr: a -= 1
    b = peak
    while b + 1 < 84 and row[b + 1] >= thr: b += 1
    return np.arange(a, b + 1), peak


# ── (1) discriminability: dedicated GNG vs choice-on-GNG, per mouse, at the GNG best window ──
Gg = tgm_gng(); Gc = tgm_choice_on_gng()
win, peak = plateau(Gg)
print(f'GNG best window bins {win[0]}-{win[-1]} ({xt[win[0]]:.1f}-{xt[win[-1]]:.1f}s) peak {xt[peak]:.1f}s')


def dp_percode(stage):
    gd, cd = [], []
    for m in ALL:
        bg = (yg.target == 'gng').values & (mg == m) & (sg == stage) & (yg.laser == 0).values & (tg != 'DPA')
        sc_g = Xg1[:, win, :][:, :, win].mean((1, 2))
        gd.append(_dprime(sc_g[bg & (yg.gng.values == 1)], sc_g[bg & (yg.gng.values == 0)]))
        bc = (yc.target == 'choice').values & (mc == m) & (sc == stage) & (yc.laser == 0).values & (tc != 'DPA')
        sc_c = Xc1[:, win, :][:, :, win].mean((1, 2))
        cd.append(_dprime(sc_c[bc & (tc == 'DualGo')], sc_c[bc & (tc == 'DualNoGo')]))
    return np.array(gd), np.array(cd)


# ── (2) cosine: gng axis vs choice/sample/test axes, per mouse ──
def axis(Wdict, key, w):
    W = Wdict['weights'].get(key)
    if W is None: return None
    v = W[w].mean(0); n = np.linalg.norm(v)
    return v / n if n > 0 else None


def cos_vs(stage, other):
    out = []
    for m in ALL:
        ag = axis(Wg, (m, stage, 'all', 'gng'), MD)         # gng axis at its mid-delay home
        ao = axis(Wc, (m, stage, 'all', other), LD if other != 'test' else o['bins_TEST'])
        if ag is None or ao is None or len(ag) != len(ao): continue
        out.append(abs(float(ag @ ao)))
    return np.array(out)


# ── (3) gng trace: Go vs NoGo over time, on the gng best axis ──
def gng_trace(stage, go):
    sig = Xg1[:, win, :].mean(1)
    per = []
    for m in ALL:
        s = (yg.target == 'gng').values & (mg == m) & (sg == stage) & (yg.laser == 0).values & (tg != 'DPA') & (yg.gng.values == (1 if go else 0))
        if s.sum() >= 3:
            z = sig[s].mean(0); z = z - z[BL].mean(); per.append(z / (sig[s][:, BL].std() + 1e-9))
    return (np.mean(per, 0), np.std(per, 0) / np.sqrt(len(per))) if per else (np.full(84, np.nan),) * 2


# ═══════════════ FIGURE ═══════════════
plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38, left=0.06, right=0.98, top=0.93, bottom=0.09)

# row0col0: GNG decoder TGM
axG = fig.add_subplot(gs[0, 0]); vmx = np.nanmax(np.abs(Gg))
im = axG.imshow(Gg, origin='lower', cmap='RdBu_r', vmin=-vmx, vmax=vmx, extent=[0, 14, 0, 14])
axG.plot([0, 14], [0, 14], 'k--', lw=.5); axG.axhspan(xt[win[0]], xt[win[-1]], color='k', alpha=.08)
axG.set_title(f'GNG decoder d′(Go,NoGo)\nbest {xt[win[0]]:.1f}-{xt[win[-1]]:.1f}s', fontsize=9)
axG.set_xlabel('test (s)'); axG.set_ylabel('train (s)'); plt.colorbar(im, ax=axG, fraction=.046)

# row0col1: choice-on-GNG TGM (current task code)
axC = fig.add_subplot(gs[0, 1]); vmc = np.nanmax(np.abs(Gc))
im2 = axC.imshow(Gc, origin='lower', cmap='RdBu_r', vmin=-vmc, vmax=vmc, extent=[0, 14, 0, 14])
axC.plot([0, 14], [0, 14], 'k--', lw=.5)
axC.set_title(f'choice decoder d′(Go,NoGo)\n= current "task code"', fontsize=9)
axC.set_xlabel('test (s)'); axC.set_ylabel('train (s)'); plt.colorbar(im2, ax=axC, fraction=.046)

# row0col2: d' comparison (dedicated vs choice-on-GNG), Naive+Expert
axD = fig.add_subplot(gs[0, 2])
for i, stage in enumerate(['Naive', 'Expert']):
    gd, cd = dp_percode(stage)
    ok = np.isfinite(gd) & np.isfinite(cd)
    x = np.array([i * 2, i * 2 + 0.8])
    axD.bar(x, [np.nanmean(cd[ok]), np.nanmean(gd[ok])], width=0.75,
            color=['#bbbbbb', '#cc3311'], edgecolor='k', lw=.6)
    for g, c in zip(gd[ok], cd[ok]):
        axD.plot(x, [c, g], '-', color='0.5', lw=.5, alpha=.6)
    p = ttest_rel(gd[ok], cd[ok]).pvalue
    axD.text(i * 2 + 0.4, max(np.nanmean(gd[ok]), np.nanmean(cd[ok])) + .05,
             f'p={p:.3f}', ha='center', fontsize=7)
    print(f'{stage}: dedicated GNG d′={np.nanmean(gd[ok]):.2f}  choice-on-GNG d′={np.nanmean(cd[ok]):.2f}  p={p:.3f}')
axD.set_xticks([0.4, 2.4]); axD.set_xticklabels(['Naive', 'Expert'])
axD.set_ylabel("d′ (Go vs NoGo)"); axD.set_title('discriminability', fontsize=9)
axD.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#bbbbbb'), plt.Rectangle((0, 0), 1, 1, color='#cc3311')],
           labels=['choice decoder', 'GNG decoder'], fontsize=7, frameon=False)

# row1col0: cosine gng-vs-choice/sample/test
axCos = fig.add_subplot(gs[1, 0])
chance = 1 / np.sqrt(184)
axCos.axhline(chance, ls=':', color='0.6', lw=.8); axCos.text(-0.28, chance, 'chance', fontsize=6, color='0.6', va='bottom')
for other, col in [('choice', '#4daf4a'), ('sample', '#332288'), ('test', '#CC6677')]:
    cN = cos_vs('Naive', other); cE = cos_vs('Expert', other)
    p = ttest_rel(cE, cN).pvalue if min(len(cN), len(cE)) > 2 else np.nan
    axCos.plot([0, 1], [cN.mean(), cE.mean()], '-o', color=col, lw=2, ms=5, label=f'GNG–{other} (p={p:.2f})')
    print(f'cos GNG-{other}: N={cN.mean():.3f} E={cE.mean():.3f} p={p:.3f}')
axCos.set_xticks([0, 1]); axCos.set_xticklabels(['Naive', 'Expert']); axCos.set_xlim(-0.3, 1.5)
axCos.set_ylim(0, chance * 1.35)
axCos.set_ylabel('|cos| GNG axis vs code'); axCos.set_title('axis alignment (all ≤ chance → orthogonal)', fontsize=8.5); axCos.legend(fontsize=6, frameon=False)

# row1col1-2: gng trace Naive/Expert
for j, stage in enumerate(['Naive', 'Expert']):
    ax = fig.add_subplot(gs[1, 1 + j]); ax.axhline(0, ls='--', c='k', lw=.5)
    add_vlines(ax, if_dpa=0)
    for go, lab, col in [(1, 'Go', '#1f77b4'), (0, 'NoGo', '#2ca02c')]:
        mu, se = gng_trace(stage, go)
        ax.plot(xt, mu, color=col, lw=1.8, label=lab); ax.fill_between(xt, mu - se, mu + se, color=col, alpha=.2)
    ax.axvspan(xt[win[0]], xt[win[-1]], color='k', alpha=.06)
    ax.set_title(f'GNG code — {stage}', fontsize=9); ax.set_xlabel('time (s)')
    if j == 0: ax.set_ylabel('GNG code (z)')
    ax.legend(fontsize=7, frameon=False)

os.makedirs('figures/overlaps/gng/png', exist_ok=True); os.makedirs('figures/overlaps/gng/svg', exist_ok=True)
fig.savefig('figures/overlaps/gng/png/gng_compare.png', dpi=200, bbox_inches='tight')
fig.savefig('figures/overlaps/gng/svg/gng_compare.svg', bbox_inches='tight')
print('saved figures/overlaps/gng/png/gng_compare.png')
