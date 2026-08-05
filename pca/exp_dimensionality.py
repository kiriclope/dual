"""Honest dimensionality of the dual-task pseudo-population — cvPCA + shattering dimension.

WHY: Fig 2b's dPCA scree is near-tautological (restricts to the sample+sample:test demixed axes and measures
only 4 condition-means → ~2-D by construction). This assesses dimensionality WITHOUT that circularity, on the
raw pseudo-population, two complementary ways:

  A. cvPCA  (Stringer et al. 2019) — reliable (cross-validated) variance spectrum + participation ratio.
     Split each (mouse, condition) trial pool into two disjoint halves → two INDEPENDENT condition-mean
     pseudo-populations R1, R2. PCA basis from one half, cross-projected onto the other: signal variance
     replicates (positive), trial-to-trial noise averages to ~0. Non-circular from both sides — single-trial
     splitting removes the noise-inflation ceiling; the FULL neuron space (not demixed axes) and all 12
     conditions (× time) remove the "4 condition-means → 2-D" floor.  PR=(Σλ)²/Σλ² on the reliable (λ>0)
     spectrum = effective linear dimensionality.

  B. Shattering dimension (Bernardi/Fusi 2020) — the FUNCTIONAL dimensionality. Decode every balanced
     dichotomy of the 12 conditions with a leakage-free, cross-validated pseudo-population linear decoder;
     SD = mean balanced accuracy over dichotomies. SD≈0.5 ⇒ low-D/structured; SD→1 ⇒ high-D/general position.
     Complements CCGP (Fig 3): CCGP high + SD moderate = abstract, compressed geometry.

Careful-implementation notes:
  * neurons partition DISJOINTLY across the 9 mice; R1/R2 are filled column-by-mouse from that mouse's own
    trials, so the cross-validation split is independent per mouse (exactly what cvPCA requires).
  * per-neuron z-scoring uses a stage-level, condition-AGNOSTIC scale (std over all of that stage's trials×time)
    → equalises neuron scale without leaking condition/split structure.
  * correct trials only (performance==1), laser off — matches Fig 2/3.
  * memory-safe: precompute the three window-matrices, then free the 20 GB tensor before any resampling.

Run:  cd /home/leon/dual/pca
      /home/leon/mambaforge/envs/dual/bin/python exp_dimensionality.py [--nsplits 30] [--ndich 150] [--quick]
Output: figures/pseudo/dimensionality/{png,svg}/dimensionality.{png,svg}  + printed numbers.
"""
import sys, os, warnings, argparse, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from src.pca.io import pkl_load
from src.common.options import set_options

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

ap = argparse.ArgumentParser()
ap.add_argument('--nsplits', type=int, default=30, help='cvPCA random half-splits to average')
ap.add_argument('--ndich', type=int, default=150, help='balanced dichotomies for shattering (max 462)')
ap.add_argument('--nboot', type=int, default=6, help='pseudo-population resamples per stage (shattering)')
ap.add_argument('--quick', action='store_true', help='fast smoke test')
A = ap.parse_args()
if A.quick:
    A.nsplits, A.ndich, A.nboot = 6, 40, 3

# ── data ──────────────────────────────────────────────────────────────────────
print('loading pseudo-population …')
X = np.asarray(pkl_load('X_all_nan_', path='../data/pca'))                 # (9216, 3319, 84), NaN off-mouse
y = pkl_load('y_all_nan_', path='../data/pca')
VALID = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test',
                 path='../data/overlaps')['valid']
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
TASKS = ['DPA', 'DualGo', 'DualNoGo']
CONDS = [(t, s, te) for t in TASKS for s in (0, 1) for te in (0, 1)]        # 12
NC, N = len(CONDS), X.shape[1]
STAGES = ['Naive', 'Expert']
MOUSE = y.mouse.to_numpy(); LEARN = y.learning.to_numpy(); LAS = y.laser.to_numpy()
TSK = y.tasks.to_numpy(); SAMP = y.sample_odor.to_numpy(); TESTO = y.test_odor.to_numpy()
PERF = y.performance.to_numpy()
o = set_options(); W_LD = np.asarray(o['bins_LD']); WIN_TRAJ = np.arange(12, 72); WIN_TEST = np.arange(57, 66)
NBLK = 12                                                                   # trajectory time-blocks
VALIDIX = {k: np.where(np.asarray(v))[0] for k, v in VALID.items()}
COND_ID = np.full(len(y), -1)                                              # vectorised condition label / trial
for ci, (t, s, te) in enumerate(CONDS):
    COND_ID[(TSK == t) & (SAMP == s) & (TESTO == te)] = ci
BASE_TR = {(m, st): np.where((MOUSE == m) & (LEARN == st) & (LAS == 0) & (PERF == 1))[0]
           for m in MICE for st in STAGES}

# ── precompute window-averaged trial matrices, per-neuron scale, then FREE the 20 GB tensor ──
print('pre-averaging windows + neuron scales …')
SD_SCALE = {}
for st in STAGES:                                                          # per-neuron std (condition-agnostic)
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, st)]; tr = BASE_TR[(m, st)]
        if len(tr):
            s = np.nanstd(X[np.ix_(tr, val)], axis=(0, 2))
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    SD_SCALE[st] = sd
AW_LD = np.nanmean(X[:, :, W_LD], axis=2)                                  # (9216, N)  delay state (WM)
AW_TEST = np.nanmean(X[:, :, WIN_TEST], axis=2)                            # (9216, N)  decision state + shattering
blocks = np.array_split(WIN_TRAJ, NBLK)
AW_TRAJ = np.empty((len(y), N, NBLK))                                      # (9216, N, NBLK) trajectory
for bi, b in enumerate(blocks):                                           # fill in place (no list+stack copy)
    AW_TRAJ[:, :, bi] = np.nanmean(X[:, :, b], axis=2)
del X
print(f'  freed tensor. AW_LD{AW_LD.shape} AW_TEST{AW_TEST.shape} AW_TRAJ{AW_TRAJ.shape}')


# ══════════════════════════ A. cvPCA ══════════════════════════════════════════
def split_means(stage, rng, M, shuffle_cond=False):
    """Two INDEPENDENT condition-mean arrays R1,R2 with leading axis NC, from M (9216,N[,nT]): each
    (mouse,condition) pool split into disjoint halves. shuffle_cond permutes condition labels within each
    mouse (null: destroys condition structure, preserves counts + noise)."""
    extra = M.shape[2:]
    R1 = np.zeros((NC, N) + extra); R2 = np.zeros((NC, N) + extra)
    for m in MICE:
        val = VALIDIX[(m, stage)]; base = BASE_TR[(m, stage)]
        labs = COND_ID[base]
        if shuffle_cond:
            labs = rng.permutation(labs)
        for ci in range(NC):
            idx = base[labs == ci]
            if len(idx) < 2:
                continue
            p = rng.permutation(idx); h = len(p) // 2
            R1[ci][val] = np.nanmean(M[p[:h]][:, val], axis=0)
            R2[ci][val] = np.nanmean(M[p[h:]][:, val], axis=0)
    return R1, R2


def cvpca_spectrum(S1, S2):
    """cvPCA reliable-variance spectrum. S1,S2:(n_states,N). Basis from one half, cross-projected onto the
    other (both directions averaged). cv[k]=<proj1_k,proj2_k> over states — signal replicates(+), noise→0."""
    S1 = S1 - S1.mean(0, keepdims=True); S2 = S2 - S2.mean(0, keepdims=True)

    def one(Atr, Bte):
        Vt = np.linalg.svd(Atr, full_matrices=False)[2]
        return ((Atr @ Vt.T) * (Bte @ Vt.T)).sum(0)
    a, b = one(S1, S2), one(S2, S1)
    k = min(len(a), len(b))
    return 0.5 * (a[:k] + b[:k])


def cvpca(stage, M, shuffle=False):
    rng = np.random.RandomState(0 if not shuffle else 999)
    sd = SD_SCALE[stage]; scale = sd[None, :] if M.ndim == 2 else sd[None, :, None]
    spec = None
    for _ in range(A.nsplits):
        R1, R2 = split_means(stage, rng, M, shuffle_cond=shuffle)
        R1 = R1 / scale; R2 = R2 / scale
        S1 = R1 if M.ndim == 2 else R1.transpose(0, 2, 1).reshape(-1, N)   # states = conds, or conds×time
        S2 = R2 if M.ndim == 2 else R2.transpose(0, 2, 1).reshape(-1, N)
        c = cvpca_spectrum(S1, S2)
        spec = c if spec is None else spec + c
    return spec / A.nsplits


def summarize(cv):
    pos = np.clip(cv, 0, None); tot = pos.sum()
    frac = np.cumsum(pos) / tot
    return dict(pr=float(tot ** 2 / (pos ** 2).sum()), top2=float(pos[:2].sum() / tot),
                d80=int(np.searchsorted(frac, 0.80) + 1), d90=int(np.searchsorted(frac, 0.90) + 1),
                reliable_frac=float(tot / np.abs(cv).sum()), total=float(tot))


print('\n══ A. cvPCA (reliable dimensionality) ══')
CV = {}
for stage in STAGES:
    for kind, M in (('delay', AW_LD), ('decision', AW_TEST), ('traj', AW_TRAJ)):
        cv = cvpca(stage, M); cvn = cvpca(stage, M, shuffle=True)
        s = summarize(cv); sn = summarize(cvn)
        CV[(stage, kind)] = dict(cv=cv, cvn=cvn, pr_null=sn['pr'], **s)
        # null reliable variance as % of real → structure is real iff this is small
        print(f'  {stage:6s} {kind:8s}: PR={s["pr"]:.2f}  top2={s["top2"]:.2f}  d80={s["d80"]}  d90={s["d90"]}'
              f'  reliable/|cv|={s["reliable_frac"]:.2f}   [null reliable var = {100*sn["total"]/s["total"]:.0f}% of real]')


# ══════════════════════ B. Shattering dimension ═══════════════════════════════
def balanced_dichotomies(seed):
    seen, dich = set(), []
    for s in combinations(range(NC), NC // 2):
        s = frozenset(s); key = min(tuple(sorted(s)), tuple(sorted(set(range(NC)) - s)))
        if key not in seen:
            seen.add(key); dich.append(s)
    np.random.RandomState(seed).shuffle(dich)
    return dich[:min(A.ndich, len(dich))]


def cond_pools(stage, rng):
    trp, tep = {}, {}
    for m in MICE:
        base = BASE_TR[(m, stage)]; labs = COND_ID[base]
        for ci in range(NC):
            p = rng.permutation(base[labs == ci]); h = len(p) // 2
            trp[(ci, m)], tep[(ci, m)] = p[:h], p[h:]
    return trp, tep


def make_pseudo(pool, stage, K, rng, M):
    """K pseudo-trials per condition from window-matrix M: each = one random trial per mouse, own neurons filled."""
    Xp = np.zeros((NC * K, N)); crow = np.repeat(np.arange(NC), K)
    for ci in range(NC):
        for m in MICE:
            val = VALIDIX[(m, stage)]; pi = pool[(ci, m)]
            if len(pi):
                Xp[ci * K:(ci + 1) * K, val] = M[np.ix_(pi[rng.randint(0, len(pi), K)], val)]
    return Xp, crow


def shatter(stage, dich, K=24, null=False):
    acc = np.zeros((len(dich), A.nboot))
    for b in range(A.nboot):
        rng = np.random.RandomState(2000 + b)
        trp, tep = cond_pools(stage, rng)
        Xtr, ctr = make_pseudo(trp, stage, K, rng, AW_TEST); Xte, cte = make_pseudo(tep, stage, K, rng, AW_TEST)
        if null:
            ctr = rng.permutation(ctr); cte = rng.permutation(cte)
        pre = make_pipeline(StandardScaler(), PCA(min(30, Xtr.shape[0] - 1), random_state=0)).fit(Xtr)
        Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
        for di, plus in enumerate(dich):
            ytr = np.isin(ctr, list(plus)).astype(int); yte = np.isin(cte, list(plus)).astype(int)
            clf = LinearDiscriminantAnalysis().fit(Ztr, ytr)
            acc[di, b] = balanced_accuracy_score(yte, clf.predict(Zte))
    return acc.mean(1)


# ── what the reliable dimensions CODE: decode each task variable per window (delay/decision) ──
CVARS = {'sample': lambda c: CONDS[c][1], 'test': lambda c: CONDS[c][2],
         'choice': lambda c: int(CONDS[c][1] == CONDS[c][2]), 'task': lambda c: TASKS.index(CONDS[c][0])}
CCHANCE = {'sample': 0.5, 'test': 0.5, 'choice': 0.5, 'task': 1 / 3}       # task is 3-way
CODEWIN = {'delay': AW_LD, 'decision': AW_TEST}


def var_decode(stage, M, vfn, K=24, B=6):
    acc = []
    for b in range(B):
        rng = np.random.RandomState(300 + b)
        trp, tep = cond_pools(stage, rng)
        Xtr, ctr = make_pseudo(trp, stage, K, rng, M); Xte, cte = make_pseudo(tep, stage, K, rng, M)
        ytr = np.array([vfn(c) for c in ctr]); yte = np.array([vfn(c) for c in cte])
        clf = make_pipeline(StandardScaler(), PCA(30, random_state=0), LinearDiscriminantAnalysis()).fit(Xtr, ytr)
        acc.append(balanced_accuracy_score(yte, clf.predict(Xte)))
    return float(np.mean(acc))


print(f'\n══ B. Shattering dimension (decision window bins {WIN_TEST[0]}–{WIN_TEST[-1]}, {A.ndich} dichotomies) ══')
DICH = balanced_dichotomies(seed=0)
SD = {}
for stage in STAGES:
    sd_acc = shatter(stage, DICH); sd_null = shatter(stage, DICH, null=True)
    SD[stage] = dict(acc=sd_acc, null=sd_null)
    print(f'  {stage}: SD = {sd_acc.mean():.3f} ± {sd_acc.std():.3f} (range {sd_acc.min():.2f}-{sd_acc.max():.2f})'
          f'   [shuffle null {sd_null.mean():.3f}]')

print('\n══ C. What the dimensions code (decode each variable per window) ══')
CODING = {}
for stage in STAGES:
    for wn, M in CODEWIN.items():
        for vn, vf in CVARS.items():
            CODING[(stage, wn, vn)] = var_decode(stage, M, vf)
    print(f'  {stage}: ' + '  '.join(f'{wn[:3]}/{vn}={CODING[(stage,wn,vn)]:.2f}' for wn in CODEWIN for vn in CVARS))


# ══ D. What each PC codes — η² of the 12 condition-mean PC scores by task factor (balanced 2×2×3 →
#    orthogonal main effects sample/test/task + the sample:test 'choice' interaction). Answers: does PC1
#    code one variable, PC2 another? ──
# orthogonal ±1 contrasts over the 12 conditions (balanced 2×2×3 → mutually orthogonal); task split into the
# gng distractor code (Go vs NoGo) and the DPA-vs-Dual 'tasks' contrast (distractor present)
_Fs = np.array([c[1] for c in CONDS], float); _Ft = np.array([c[2] for c in CONDS], float)
_Fk = np.array([TASKS.index(c[0]) for c in CONDS])                         # 0=DPA 1=DualGo 2=DualNoGo
CNTR = {'sample': 2 * _Fs - 1, 'test': 2 * _Ft - 1, 'choice': 2 * (_Fs == _Ft) - 1,
        'gng': np.select([_Fk == 1, _Fk == 2], [1.0, -1.0], default=0.0),
        'tasks': np.select([_Fk == 0, _Fk == 1, _Fk == 2], [2.0, -1.0, -1.0])}
FORDER = ['sample', 'gng', 'test', 'choice', 'tasks']


def _eta2(zk):
    zc = zk - zk.mean(); sst = (zc ** 2).sum() + 1e-12
    return {f: (CNTR[f] @ zc) ** 2 / ((CNTR[f] @ CNTR[f]) * sst) for f in FORDER}


def cond_means(stage, M):
    R = np.zeros((NC, N))
    for m in MICE:
        val = VALIDIX[(m, stage)]; base = BASE_TR[(m, stage)]; labs = COND_ID[base]
        for ci in range(NC):
            idx = base[labs == ci]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], axis=0)
    return R


print('\n══ D. What each PC codes (η² of PC scores by factor) ══')
PCETA = {}
for stage in STAGES:
    for wn, M in (('delay', AW_LD), ('decision', AW_TEST)):
        R = cond_means(stage, M); Rc = (R - R.mean(0)) / (R.std(0) + 1e-9); Rc = Rc - Rc.mean(0)
        sv, Vt = np.linalg.svd(Rc, full_matrices=False)[1:]
        var = sv ** 2 / (sv ** 2).sum(); Z = Rc @ Vt.T
        PCETA[(stage, wn)] = dict(eta=np.array([[_eta2(Z[:, k])[f] for f in FORDER] for k in range(4)]),
                                  var=var[:4], factors=FORDER)
    print(f'  {stage}: delay ' + '/'.join(f'PC{k+1}={FORDER[int(np.argmax(PCETA[(stage,"delay")]["eta"][k]))]}'
                                          for k in range(3)))


# cache results so the figure can be re-plotted without the ~12-min recompute
os.makedirs('figures/pseudo/dimensionality', exist_ok=True)
pickle.dump({'CV': CV, 'SD': SD, 'CODING': CODING, 'CCHANCE': CCHANCE, 'PCETA': PCETA, 'DICH': DICH,
             'STAGES': STAGES}, open('figures/pseudo/dimensionality/results.pkl', 'wb'))

# ══════════════════════════ figure ════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(17.0, 3.3), gridspec_kw=dict(wspace=0.55))
SC = {'Naive': '0.55', 'Expert': '#332288'}

ax = axes[0]                                                               # absolute reliable variance (not
allcv = np.concatenate([CV[(s, 'delay')]['cv'] for s in STAGES])          # normalized → null honestly floors)
floor = max(allcv.max() * 1e-4, 1e-9)
for stage in STAGES:
    cv = CV[(stage, 'delay')]['cv']
    ax.plot(np.arange(1, len(cv) + 1), np.clip(cv, floor, None), '-o', ms=3, color=SC[stage], label=stage)
cvn = CV[('Expert', 'delay')]['cvn']
ax.plot(np.arange(1, len(cvn) + 1), np.clip(cvn, floor, None), '--', color='0.7', lw=1.0, label='null (shuffled)')
ax.set_yscale('log'); ax.set_xlabel('cvPCA component'); ax.set_ylabel('reliable variance (cross-validated)')
ax.set_title('cvPCA — delay-state geometry (12 conds)', loc='left', fontsize=TITLE_FS)
ax.legend(frameon=False, fontsize=6.5)

ax = axes[1]                                                               # clean per-epoch PR (null≈0);
labs = ['delay', 'decision']; xp = np.arange(len(labs))                    # traj omitted (time-contaminated null)
for j, stage in enumerate(STAGES):
    prs = [CV[(stage, k)]['pr'] for k in labs]
    ax.bar(xp + (j - 0.5) * 0.32, prs, 0.30, color=SC[stage], label=stage)
    for x, v in zip(xp + (j - 0.5) * 0.32, prs):
        ax.text(x, v + 0.05, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(xp); ax.set_xticklabels(['delay\n(WM state)', 'decision\n(all 12 conds)'])
ax.set_ylim(0, 4)
ax.set_ylabel('participation ratio'); ax.set_title('reliable dimensionality (PR)', loc='left', fontsize=TITLE_FS)
ax.legend(frameon=False, fontsize=6.5, loc='upper left')

ax = axes[2]
for j, stage in enumerate(STAGES):
    a = SD[stage]['acc']
    parts = ax.violinplot(a, positions=[j], widths=0.7, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(SC[stage]); pc.set_alpha(0.5)
    parts['cmeans'].set_color('k')
    ax.text(j, a.mean() + 0.01, f'{a.mean():.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.axhline(0.5, ls=':', color='0.6', lw=0.8); ax.text(1.55, 0.505, 'chance', fontsize=5.5, color='0.6', ha='right')
ax.axhline(SD['Expert']['null'].mean(), ls='--', color='0.7', lw=0.9)
ax.set_xticks([0, 1]); ax.set_xticklabels(STAGES); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.45, 1.02)
ax.set_ylabel('decoding bal. acc. / dichotomy')
ax.set_title(f'shattering dimension ({len(DICH)} dichotomies)', loc='left', fontsize=TITLE_FS)

def pc_heatmap(ax, wn, title, cbar=False):                                 # one η²-by-factor matrix per window
    P = PCETA[('Expert', wn)]; FO = P['factors']; M = np.array([P['eta'][k] for k in range(4)])
    im = ax.imshow(M, cmap='Purples', vmin=0, vmax=1, aspect='auto')
    for i in range(4):
        for j in range(len(FO)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                    color='w' if M[i, j] > 0.55 else 'k')
    ax.set_xticks(range(len(FO))); ax.set_xticklabels(FO, fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels([f'PC{k+1} ({P["var"][k]:.0%})' for k in range(4)], fontsize=6.5)
    ax.set_title(title, loc='left', fontsize=TITLE_FS)
    for sp in ax.spines.values():
        sp.set_visible(True)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04); cb.set_label('η²', fontsize=6.5); cb.ax.tick_params(labelsize=6)


pc_heatmap(axes[3], 'delay', 'what each PC codes — delay (WM state)')
pc_heatmap(axes[4], 'decision', 'what each PC codes — decision', cbar=True)

fig.suptitle('Honest dimensionality of the dual-task pseudo-population — cvPCA + shattering + PC coding',
             x=0.008, ha='left', y=1.02, fontsize=10)
OUT = 'figures/pseudo/dimensionality'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
fig.savefig(f'{OUT}/png/dimensionality.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/dimensionality.svg', bbox_inches='tight')
print('\nsaved', os.path.abspath(f'{OUT}/png/dimensionality.png'))
