"""Pseudo-population cross-task generalization matrices (decoder d').

Same idea as fig_ccgp_matrices.py but on a PSEUDO-POPULATION: each pseudo-trial pools ONE random trial of
the matched condition from every mouse (concatenated neurons → all 3,319), which removes the per-mouse
~184-neuron power ceiling that flattened the per-mouse d'. Decoder = StandardScaler → PCA(denoise) →
logistic; averaged over R pseudo-population resamples. Diagonal = within-task (independent resamples),
off-diagonal = train task i → test task j. This is the field-standard way CCGP / cross-gen is presented.

sample @ late delay · choice/test @ TEST  (--test = all @ TEST).  Expert.
Output: figures/overlaps/ccgp/{png,svg}/overlaps_ccgp_matrices_pseudo[_test].{png,svg}
"""
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from src.pca.io import pkl_load
from src.common.options import set_options
from src.common import plot_utils  # noqa: sets poster context at import

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
TASKS, TLAB = ['DPA', 'DualGo', 'DualNoGo'], ['DPA', 'Go', 'NoGo']
K, NPC, B = 60, 20, 100                                 # pseudo-trials/class · PCA dims · bootstrap replicates
EYE = np.eye(3, dtype=bool)
Y2 = np.r_[np.zeros(K), np.ones(K)]
RNG = np.random.RandomState(0)
o = set_options(); W_LD, W_TE, W_MD = np.asarray(o['bins_LD']), np.asarray(o['bins_TEST']), np.asarray(o['bins_MD'])

print('loading pseudo-population …')
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))
y = pkl_load('y_all_nan_', path='../data/pca')
VALID = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
                 path='../data/overlaps')['valid']
MOUSE = y.mouse.to_numpy(); LEARN = y.learning.to_numpy(); LAS = y.laser.to_numpy(); TSK = y.tasks.to_numpy()
print('pre-averaging activity per window (once) …')
AW = {'LD': np.nanmean(X[:, :, W_LD], axis=2), 'TE': np.nanmean(X[:, :, W_TE], axis=2),
      'MD': np.nanmean(X[:, :, W_MD], axis=2)}                                            # 9216×3319 each
del X                                                                                    # free the 20 GB tensor

VARS = [('sample', 'sample_odor', 'LD'), ('choice', 'choice', 'TE'), ('test', 'test_odor', 'TE')]
if '--test' in sys.argv[1:]:
    VARS = [(lab, col, 'TE') for lab, col, _ in VARS]; SUF, WINLAB = '_test', 'all codes @ TEST epoch'
else:
    SUF, WINLAB = '', 'sample @ late delay · choice/test @ TEST'
NOPCA = '--nopca' in sys.argv[1:]           # drop the PCA denoising step (robustness variant)
PSUF = '_nopca' if NOPCA else ''
PIPE = (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))) if NOPCA \
    else (lambda: make_pipeline(StandardScaler(), PCA(n_components=NPC, random_state=0),
                                LogisticRegression(C=1.0, max_iter=3000)))


def dprime(s, V):
    a, b = s[V == 1], s[V == 0]
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-9))


# ── cell metric: d′ (default) or balanced accuracy (--acc). CHANCE is subtracted before forming ratios so the
#    ÷within / off-diag indices are "fraction of above-chance discriminability recovered" for BOTH metrics. ──
ACC = '--acc' in sys.argv[1:]
CHANCE = 0.5 if ACC else 0.0
MLAB = 'bal. acc.' if ACC else "d′"
ASUF = '_acc' if ACC else ''


def cell(clf, Xte):
    return balanced_accuracy_score(Y2, clf.predict(Xte)) if ACC else dprime(clf.decision_function(Xte), Y2)


def cond_indices(col, cls, task, stage):
    base = (LEARN == stage) & (LAS == 0) & (TSK == task) & (y[col].to_numpy() == cls)
    return {m: np.where(base & (MOUSE == m))[0] for m in ALL_MICE}


def pseudo(cidx, wkey, stage):
    A = AW[wkey]; P = np.full((K, A.shape[1]), np.nan)
    for m in ALL_MICE:
        idx = cidx[m]
        if len(idx) == 0:
            continue
        pick = RNG.choice(idx, K, replace=True); val = VALID[(m, stage)]
        P[:, val] = A[pick][:, val]                                  # resample from the precomputed 2-D array
    return np.where(np.isnan(P), 0.0, P)


def split_halves(cidx):                                 # disjoint random halves of each mouse's trials
    h1, h2 = {}, {}
    for m, idx in cidx.items():
        p = RNG.permutation(idx); k = len(p) // 2
        h1[m], h2[m] = p[:k], p[k:]
    return h1, h2


def one_matrix(col, wkey, stage):
    """ONE bootstrap replicate: full 3×3 (train task × test task) from a single pseudo-population resample.
    Each train task → one decoder on a disjoint HALF, tested on its held-out half (diagonal) + each other
    task (off-diagonal). Matched training size + no leakage → fair diag-vs-off comparison."""
    M = np.zeros((3, 3))
    for i, ti in enumerate(TASKS):
        s = {c: split_halves(cond_indices(col, c, ti, stage)) for c in (0, 1)}  # c -> (train_half, test_half)
        clf = PIPE().fit(np.vstack([pseudo(s[0][0], wkey, stage), pseudo(s[1][0], wkey, stage)]), Y2)
        for j, tj in enumerate(TASKS):
            if tj == ti:
                Xte = np.vstack([pseudo(s[0][1], wkey, stage), pseudo(s[1][1], wkey, stage)])  # held-out half
            else:
                cj = {c: cond_indices(col, c, tj, stage) for c in (0, 1)}
                Xte = np.vstack([pseudo(cj[0], wkey, stage), pseudo(cj[1], wkey, stage)])
            M[i, j] = cell(clf, Xte)
    return M


def gng_cond(cls, samp, stage):
    """GNG class (Go=1 / NoGo=0) within a SAMPLE context (odor A=0 / B=1), on Dual trials only."""
    gv = (TSK == 'DualGo')                                                    # Go=True, NoGo=False
    base = ((LEARN == stage) & (LAS == 0) & np.isin(TSK, ['DualGo', 'DualNoGo'])
            & (gv == bool(cls)) & (y['sample_odor'].to_numpy() == samp))
    return {m: np.where(base & (MOUSE == m))[0] for m in ALL_MICE}


def gng_matrix(stage, wkey='MD'):
    """GNG (Go vs NoGo) decoder generalizing across SAMPLE (2×2: train sample-A/B × test sample-A/B).
    GNG has no within-TASK diagonal (it IS the task distinction); its meaningful generalization is across
    memory content — does the distractor code hold regardless of what odor is being remembered."""
    SAMP = [0, 1]
    M = np.zeros((2, 2))
    for i, si in enumerate(SAMP):
        s = {c: split_halves(gng_cond(c, si, stage)) for c in (0, 1)}         # c -> (train_half, test_half)
        clf = PIPE().fit(np.vstack([pseudo(s[0][0], wkey, stage), pseudo(s[1][0], wkey, stage)]), Y2)
        for j, sj in enumerate(SAMP):
            if sj == si:
                Xte = np.vstack([pseudo(s[0][1], wkey, stage), pseudo(s[1][1], wkey, stage)])
            else:
                cj = {c: gng_cond(c, sj, stage) for c in (0, 1)}
                Xte = np.vstack([pseudo(cj[0], wkey, stage), pseudo(cj[1], wkey, stage)])
            M[i, j] = cell(clf, Xte)
    return M


# ── SHARED ACTION AXIS: do the Go/NoGo decision and the DPA lick decision ride ONE action axis? ──
# Cross-decode the two action codes: Go(1)/NoGo(0) @mid-delay (distractor response) vs DPA lick(1)/no-lick(0)
# @test. Off-diagonal = train on one action code, test on the OTHER (different trials, tasks AND epochs → no
# leakage). Above-chance off-diagonal ⇒ a single action/lick axis serves both readouts. This is the robust,
# generalization-based version of the (weak, ~0.18) cos(DPA-lick · GNG-axis) — noise dims dilute cosine but
# not cross-decoding.
ACT_CODES = ['GNG', 'choice']; ACT_WIN = {'GNG': 'MD', 'choice': 'TE'}


def dpa_choice_cond(cls, stage):                          # DPA lick(1)/no-lick(0) at test
    base = (LEARN == stage) & (LAS == 0) & (TSK == 'DPA') & (y['choice'].to_numpy() == cls)
    return {m: np.where(base & (MOUSE == m))[0] for m in ALL_MICE}


def gng_action_cond(cls, stage):                          # Go(1)/NoGo(0) pooled across sample, Dual trials
    gv = (TSK == 'DualGo')
    base = (LEARN == stage) & (LAS == 0) & np.isin(TSK, ['DualGo', 'DualNoGo']) & (gv == bool(cls))
    return {m: np.where(base & (MOUSE == m))[0] for m in ALL_MICE}


def act_cond(which, cls, stage):
    return gng_action_cond(cls, stage) if which == 'GNG' else dpa_choice_cond(cls, stage)


def action_matrix(stage):
    """2×2 (rows/cols = GNG action, DPA choice). diagonal = within-code held-out half; off-diagonal =
    cross-decode between the go/no-go axis and the DPA lick axis. Each code uses its own read window."""
    M = np.zeros((2, 2))
    for i, ci in enumerate(ACT_CODES):
        s = {c: split_halves(act_cond(ci, c, stage)) for c in (0, 1)}
        clf = PIPE().fit(np.vstack([pseudo(s[0][0], ACT_WIN[ci], stage), pseudo(s[1][0], ACT_WIN[ci], stage)]), Y2)
        for j, cj in enumerate(ACT_CODES):
            if cj == ci:
                Xte = np.vstack([pseudo(s[0][1], ACT_WIN[cj], stage), pseudo(s[1][1], ACT_WIN[cj], stage)])
            else:
                d = {c: act_cond(cj, c, stage) for c in (0, 1)}
                Xte = np.vstack([pseudo(d[0], ACT_WIN[cj], stage), pseudo(d[1], ACT_WIN[cj], stage)])
            M[i, j] = cell(clf, Xte)
    return M




# ÷within generalization: divide each cell's ABOVE-CHANCE value by its TEST task's within-task above-chance
# value (the column diagonal) to cancel the test-set SNR term. N[i,j] = (M[i,j]-chance)/(within_j-chance) =
# fraction of task-j's own discriminability the cross decoder recovers; diag ≡ 1. off/diag = mean above-chance
# off ÷ mean above-chance diag. Both indices use the STABLE MEAN diagonal (not the per-replicate one) so their
# bootstrap CIs don't blow up when a weak diagonal lands near chance.
STAGES = ['Naive', 'Expert']
Mms, Nms, SUMM, BOOT = {}, {}, {}, {}                               # keyed by (stage, lab)
for stage in STAGES:
    for lab, col, wkey in VARS:
        Bmat = np.stack([one_matrix(col, wkey, stage) for _ in range(B)])       # (B,3,3) bootstrap replicates
        M = Bmat.mean(0)
        Dmean = np.diag(M)                                                       # STABLE within-task metric/task
        Mms[(stage, lab)] = M
        Nms[(stage, lab)] = (M - CHANCE) / (Dmean - CHANCE)[None, :]
        offd = np.array([(Bmat[b][~EYE] - CHANCE).mean() / (np.diag(Bmat[b]) - CHANCE).mean() for b in range(B)])
        woff = np.array([((Bmat[b] - CHANCE) / (Dmean - CHANCE)[None, :])[~EYE].mean() for b in range(B)])
        olo, ohi = np.percentile(offd, [2.5, 97.5]); wlo, whi = np.percentile(woff, [2.5, 97.5])
        SUMM[(stage, lab)] = dict(offd=offd.mean(), offd_lo=olo, offd_hi=ohi,
                                  woff=woff.mean(), woff_lo=wlo, woff_hi=whi)
        BOOT[(stage, lab)] = dict(offd=offd, woff=woff)
        print(f'{stage:6s} {lab:7s} off/diag={offd.mean():.2f} [{olo:.2f},{ohi:.2f}]  '
              f'÷within off={woff.mean():.2f} [{wlo:.2f},{whi:.2f}]\n{np.round(M,2)}')

# ── Expert − Naive difference (bootstrap 95% CI): does the learning effect exclude 0? ──
LABS = [lab for lab, _, _ in VARS]
DIFF = {}
print(f'\n=== Expert − Naive Δ (metric={MLAB}, B={B}, paired independent bootstrap draws) ===')
for lab in LABS:
    for key in ('offd', 'woff'):
        d = BOOT[('Expert', lab)][key] - BOOT[('Naive', lab)][key]
        lo, hi = np.percentile(d, [2.5, 97.5]); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        DIFF[(lab, key)] = dict(mean=d.mean(), lo=lo, hi=hi, p=p, sig=(lo > 0 or hi < 0))
        print(f'  {lab:7s} Δ{key:4s} = {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  p={p:.3f}  '
              f'{"★ excludes 0" if DIFF[(lab, key)]["sig"] else "n.s."}')

# GNG code: no within-task matrix (GNG IS the task); compute its generalization across SAMPLE (2×2) @ mid-delay
GNG_Mms = {}
for stage in STAGES:
    GNG_Mms[stage] = np.stack([gng_matrix(stage) for _ in range(B)]).mean(0)
    print(f'{stage:6s} GNG    (Go/NoGo × sample A/B) 2×2\n{np.round(GNG_Mms[stage],2)}')

# SHARED ACTION AXIS: Go/NoGo <-> DPA-lick cross-decode (2×2), bootstrap over B draws + Expert−Naive Δ CI
ACT_Mms, ACT_SUMM, ACT_BOOT = {}, {}, {}
E2 = np.eye(2, dtype=bool)
for stage in STAGES:
    Ab = np.stack([action_matrix(stage) for _ in range(B)])                   # (B,2,2)
    ACT_Mms[stage] = Ab.mean(0)
    offr = np.array([(Ab[b][~E2] - CHANCE).mean() / (np.diag(Ab[b]) - CHANCE).mean() for b in range(B)])
    olo, ohi = np.percentile(offr, [2.5, 97.5])
    ACT_SUMM[stage] = dict(off=np.array([Ab[b][~E2].mean() for b in range(B)]).mean(),
                           offdiag=offr.mean(), offdiag_lo=olo, offdiag_hi=ohi)
    ACT_BOOT[stage] = offr
    print(f'{stage:6s} ACTION Go/NoGo<->DPA-lick 2×2\n{np.round(ACT_Mms[stage],3)}  '
          f'off/diag={offr.mean():.2f} [{olo:.2f},{ohi:.2f}]')
dA = ACT_BOOT['Expert'] - ACT_BOOT['Naive']
alo, ahi = np.percentile(dA, [2.5, 97.5]); ap = 2 * min((dA <= 0).mean(), (dA >= 0).mean())
ACT_DIFF = dict(mean=dA.mean(), lo=alo, hi=ahi, p=ap, sig=(alo > 0 or ahi < 0))
print(f'  ACTION Δ off/diag (Expert−Naive) = {dA.mean():+.2f} [{alo:+.2f},{ahi:+.2f}] p={ap:.3f} '
      f'{"★ excludes 0" if ACT_DIFF["sig"] else "n.s."}')

# cache the computed arrays so downstream figures (fig_overlaps_manifold.py) can replot without recomputing
import pickle
os.makedirs('figures/overlaps/ccgp', exist_ok=True)
pickle.dump({'Mms': Mms, 'Nms': Nms, 'SUMM': SUMM, 'DIFF': DIFF, 'CHANCE': CHANCE, 'MLAB': MLAB,
             'STAGES': STAGES, 'LABS': LABS, 'TASKS': TASKS, 'TLAB': TLAB, 'GNG_Mms': GNG_Mms,
             'ACT_Mms': ACT_Mms, 'ACT_SUMM': ACT_SUMM, 'ACT_DIFF': ACT_DIFF, 'ACT_CODES': ACT_CODES},
            open(f'figures/overlaps/ccgp/matrices_cache{SUF}{ASUF}{PSUF}.pkl', 'wb'))
print(f'cached arrays → figures/overlaps/ccgp/matrices_cache{SUF}{ASUF}{PSUF}.pkl')

# ── figure 1: 4 rows (Naive raw · Naive ÷within · Expert raw · Expert ÷within) × 3 variable columns ──
DEV = max(np.nanmax(np.abs(Mms[(s, lab)] - CHANCE)) for s in STAGES for lab, _, _ in VARS)  # shared raw scale
ROWS = [('Naive', 'raw'), ('Naive', 'norm'), ('Expert', 'raw'), ('Expert', 'norm')]
fig, axes = plt.subplots(4, 3, figsize=(8.8, 11.4), gridspec_kw=dict(wspace=0.62, hspace=0.42))
for r, (stage, kind) in enumerate(ROWS):
    for c, (lab, _, _) in enumerate(VARS):
        ax = axes[r, c]
        if kind == 'raw':
            M = Mms[(stage, lab)]
            if ACC:                                    # balanced accuracy: floor = chance 0.5, sequential map
                im = ax.imshow(M, cmap='Reds', vmin=0.5, vmax=0.5 + DEV, aspect='equal')
            else:                                      # d′: diverging around 0
                im = ax.imshow(M, cmap='RdBu_r', vmin=-DEV, vmax=DEV, aspect='equal')
            for i in range(3):
                for j in range(3):
                    hot = (M[i, j] > 0.5 + 0.62 * DEV) if ACC else (abs(M[i, j]) > 0.6 * DEV)
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                            color='w' if hot else 'k', fontweight='bold' if i == j else 'normal')
            offr = (M[~EYE] - CHANCE).mean() / (np.diag(M) - CHANCE).mean()
            ax.set_title(f'{lab}  (off/diag {offr:.2f})', loc='left', fontsize=TITLE_FS)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6); cb.set_label(MLAB, fontsize=6.5)
        else:
            N = Nms[(stage, lab)]
            im = ax.imshow(N, cmap='RdBu_r', vmin=0, vmax=2, aspect='equal')     # 1 = white = within-task ceiling
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{N[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                            color='w' if abs(N[i, j] - 1) > 0.6 else 'k',
                            fontweight='bold' if i == j else 'normal')
            ax.set_title(f'{lab}  (off {N[~EYE].mean():.2f})', loc='left', fontsize=TITLE_FS)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6); cb.set_label('÷ within', fontsize=6)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(TLAB); ax.set_yticklabels(TLAB)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
        if c == 0:
            ax.set_ylabel(f'{stage} · {"raw " + MLAB if kind == "raw" else "÷within"}\ntrain task', fontsize=7.5)
        if r == 3:
            ax.set_xlabel('test task')
fig.suptitle(f'Pseudo-population cross-task generalization — Naive vs Expert (raw {MLAB} + ÷within-test-task, '
             f'SNR-controlled) — {WINLAB}', x=0.01, ha='left', y=0.995, fontsize=9)
OUT = 'figures/overlaps/ccgp'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
fig.savefig(f'{OUT}/png/overlaps_ccgp_matrices_pseudo{SUF}{ASUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/overlaps_ccgp_matrices_pseudo{SUF}{ASUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/overlaps_ccgp_matrices_pseudo{SUF}{ASUF}.png'))

# ── figure 2: SUMMARY — off/diag · ÷within index (Naive vs Expert) · Δ(Expert−Naive) with bootstrap 95% CI ──
xpos = np.arange(len(LABS))
figs, axs = plt.subplots(1, 3, figsize=(10.4, 3.2), gridspec_kw=dict(wspace=0.42))
for ax, (key, ylab, ref) in zip(axs[:2], [('offd', f'off-diag / diag  (raw {MLAB})', None),
                                          ('woff', 'mean off-diag  (÷ within-test-task)', 1.0)]):
    for stage, mk, dx, c in [('Naive', 'o', -0.12, '0.6'), ('Expert', 's', 0.12, '#332288')]:
        ys = np.array([SUMM[(stage, lab)][key] for lab in LABS])
        err = np.array([[ys[k] - SUMM[(stage, lab)][key + '_lo'] for k, lab in enumerate(LABS)],
                        [SUMM[(stage, lab)][key + '_hi'] - ys[k] for k, lab in enumerate(LABS)]])
        ax.errorbar(xpos + dx, ys, yerr=err, fmt=mk, ms=6, color=c, mec='k', mew=0.5, capsize=2.5,
                    lw=1.0, label=stage)
    if ref is not None:
        ax.axhline(ref, ls=':', color='0.6', lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(LABS); ax.set_xlim(-0.5, len(LABS) - 0.5)
    ax.set_ylabel(ylab); ax.set_title(ylab.split('  (')[0], loc='left', fontsize=TITLE_FS)
axs[0].legend(frameon=False, fontsize=6.5, loc='upper left')

axd = axs[2]                                                        # learning effect: Δ ÷within (Expert − Naive)
dm = [DIFF[(lab, 'woff')]['mean'] for lab in LABS]
derr = np.array([[DIFF[(lab, 'woff')]['mean'] - DIFF[(lab, 'woff')]['lo'] for lab in LABS],
                 [DIFF[(lab, 'woff')]['hi'] - DIFF[(lab, 'woff')]['mean'] for lab in LABS]])
axd.axhline(0, ls=':', color='0.6', lw=0.8)
axd.errorbar(xpos, dm, yerr=derr, fmt='D', ms=6, color='#CC6677', mec='k', mew=0.5, capsize=2.5, lw=1.0)
for k, lab in enumerate(LABS):
    st = DIFF[(lab, 'woff')]
    axd.text(k, st['hi'] + 0.02, '★' if st['sig'] else 'n.s.', ha='center', va='bottom',
             fontsize=12 if st['sig'] else 7, fontweight='bold', color='k' if st['sig'] else '0.55')
axd.set_xticks(xpos); axd.set_xticklabels(LABS); axd.set_xlim(-0.5, len(LABS) - 0.5)
axd.set_ylabel('Δ ÷within  (Expert − Naive)'); axd.set_title('learning effect (Δ, 95% CI)', loc='left', fontsize=TITLE_FS)

sig_codes = [lab for lab in LABS if DIFF[(lab, 'woff')]['sig']]
sig_txt = ('learning Δ (÷within) significant for: ' + ', '.join(sig_codes)) if sig_codes \
    else 'no code reaches a significant learning Δ'
figs.suptitle(f'Cross-task generalization index — Naive vs Expert (metric = {MLAB}, bootstrap 95% CI, B={B}).\n'
              f'Right = Expert−Naive Δ (★ = 95% CI excludes 0).  {sig_txt}.', x=0.01, ha='left', y=1.04,
              fontsize=8)
figs.savefig(f'{OUT}/png/overlaps_ccgp_matrices_pseudo{SUF}{ASUF}_summary.png', bbox_inches='tight')
figs.savefig(f'{OUT}/svg/overlaps_ccgp_matrices_pseudo{SUF}{ASUF}_summary.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/overlaps_ccgp_matrices_pseudo{SUF}{ASUF}_summary.png'))

# ── figure 3: FOLD PANEL for Fig 2 — raw metric only, Naive (top) vs Expert (bottom), 2×3 (sample/choice/test) ──
figf, axf = plt.subplots(2, 3, figsize=(7.6, 5.4), gridspec_kw=dict(wspace=0.55, hspace=0.38))
for r, stage in enumerate(STAGES):
    for c, (lab, _, _) in enumerate(VARS):
        ax = axf[r, c]
        M = Mms[(stage, lab)]
        if ACC:                                        # balanced accuracy: chance-floored sequential map
            im = ax.imshow(M, cmap='Reds', vmin=0.5, vmax=0.5 + DEV, aspect='equal')
        else:                                          # d′: diverging around 0
            im = ax.imshow(M, cmap='RdBu_r', vmin=-DEV, vmax=DEV, aspect='equal')
        for i in range(3):
            for j in range(3):
                hot = (M[i, j] > 0.5 + 0.62 * DEV) if ACC else (abs(M[i, j]) > 0.6 * DEV)
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8,
                        color='w' if hot else 'k', fontweight='bold' if i == j else 'normal')
        offr = (M[~EYE] - CHANCE).mean() / (np.diag(M) - CHANCE).mean()
        ax.set_title(f'{lab}  (off/diag {offr:.2f})', loc='left', fontsize=TITLE_FS)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(TLAB); ax.set_yticklabels(TLAB)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
        cb = figf.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6); cb.set_label(MLAB, fontsize=6.5)
        if c == 0:
            ax.set_ylabel(f'{stage}\ntrain task', fontsize=8)
        if r == len(STAGES) - 1:
            ax.set_xlabel('test task')
figf.suptitle(f'Cross-task generalization ({MLAB}): a code trained on one task reads out across DPA / Go / '
              f'NoGo\n(diagonal = within-task; near-uniform rows = shared, abstract geometry) — {WINLAB}',
              x=0.01, ha='left', y=1.02, fontsize=8.5)
figf.savefig(f'{OUT}/png/overlaps_ccgp_foldpanel{SUF}{ASUF}.png', bbox_inches='tight')
figf.savefig(f'{OUT}/svg/overlaps_ccgp_foldpanel{SUF}{ASUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/overlaps_ccgp_foldpanel{SUF}{ASUF}.png'))
