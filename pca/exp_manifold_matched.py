"""exp_manifold_matched.py — the per-mouse manifold geometry of exp_manifold_geometry.py, with the two confounds
of that first pass removed (2026-09-18, rigor pass before the geometry panels became Extended Data):

  (1) TRIAL COUNT.  A dual cloud holds ~2x the trials of a DPA cloud in the same mouse (medians 164 vs 88), and the
      participation ratio of a sample covariance rises with the number of samples, so the raw "dual manifold is
      higher-dimensional" contrast is partly a counting artifact. Every set is subsampled to the same n here (the
      smallest set's n in that mouse, stage and window), NDRAW times, and the metrics averaged over draws.
  (2) PER-CLOUD SCALING.  exp_manifold_geometry.py z-scored each cloud by its OWN standard deviations, which forces
      the manifold radius to sqrt(nneu) in every set and stage (hence R = 18.7-18.8 everywhere, a non-result). The
      scale here is condition-agnostic and common to all sets: the s.d. over ALL of that mouse's correct laser-off
      trials at that stage, so DPA and dual clouds are in the same units and R is comparable.

Also adds the CONDITION-COUNT control the DPA-vs-dual contrast needs. The dual set spans eight odor conditions
against the DPA set's four, and two of those are the Go/NoGo odor itself, so part of any extra dimensionality is the
GNG axis rather than a bigger manifold. The Go-only and NoGo-only sets have DPA's design exactly (sample x test, four
conditions, one task context), so DPA vs Go isolates "does distraction enlarge the manifold" from "does adding the
Go/NoGo contrast add a dimension".

  ID      intrinsic dimension of the trial cloud (Two-NN, Facco et al. 2017, 10% trim)
  PR      participation ratio of the trial covariance
  R       mean distance to the cloud centroid, in the common per-neuron units
  SEP(v)  |centroid difference| / mean within-class RMS spread for variable v (decoder-free, unit-free)

Windows: mid-delay (bins 33-38), late delay (45-53), decision (54-62).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_manifold_matched.py [--ndraw 20]
Output: figures/pseudo/dimensionality/manifold_matched.pkl  (R per mouse/set/window/stage, S stage tests, C set tests)
"""
import sys, os, warnings, pickle, time
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'],
        'Go': [c for c in ALL12 if c[0] == 'DualGo'],
        'NoGo': [c for c in ALL12 if c[0] == 'DualNoGo'],
        'dual': [c for c in ALL12 if c[0] != 'DPA']}
WINS = [('md', 'mid-delay'), ('delay', 'late delay'), ('decision', 'decision')]
NDRAW = int(sys.argv[sys.argv.index('--ndraw') + 1]) if '--ndraw' in sys.argv else 20
SEED = 7

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def two_nn_id(X, trim=0.1):
    D = squareform(pdist(X)); np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    ok = r[:, 0] > 1e-12
    mu = np.sort((r[ok, 1] / r[ok, 0])[np.isfinite(r[ok, 1] / r[ok, 0])])
    mu = mu[mu > 1]
    n = len(mu)
    if n < 10:
        return np.nan
    keep = int(n * (1 - trim)); x = np.log(mu[:keep]); F = np.arange(1, n + 1) / n
    return float(np.sum(x * -np.log(1 - F[:keep])) / np.sum(x * x))


def part_ratio(X):
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False)); ev = ev[ev > 1e-12]
    return float(ev.sum() ** 2 / np.sum(ev ** 2)) if len(ev) else np.nan


def separation(X, lab):
    vals = np.unique(lab)
    if len(vals) != 2:
        return np.nan
    A, B = X[lab == vals[0]], X[lab == vals[1]]
    if len(A) < 3 or len(B) < 3:
        return np.nan
    d = np.linalg.norm(A.mean(0) - B.mean(0))
    w = np.mean([np.sqrt(np.mean(np.sum((A - A.mean(0)) ** 2, 1))), np.sqrt(np.mean(np.sum((B - B.mean(0)) ** 2, 1)))])
    return float(d / w) if w > 0 else np.nan


def common_scale(m, stage, wn, val):
    """Condition-agnostic per-neuron centre and scale: ALL of this mouse's correct laser-off trials at this stage."""
    tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
    A = AW[wn][np.ix_(tr, val)]
    return np.nanmean(A, 0), np.nanstd(A, 0)


def trials_of(m, stage, conds):
    keep = np.zeros(len(MOUSE), bool)
    for (t, s, te) in conds:
        keep |= (TSK == t) & (SAMP == s) & (TESTO == te)
    return np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1) & keep)[0]


def strat_subsample(idx, conds, n_take, rng):
    """n_take trials, drawn without replacement, keeping the set's own condition proportions."""
    by = [idx[np.array([(TSK[i], SAMP[i], TESTO[i]) == c for i in idx])] for c in conds]
    frac = np.array([len(b) for b in by], float); frac = frac / frac.sum()
    take = np.floor(frac * n_take).astype(int)
    while take.sum() < n_take:                                   # hand the remainder to the largest pools
        j = int(np.argmax(frac * n_take - take)); take[j] += 1; frac[j] -= 1e-9
    out = []
    for b, k in zip(by, take):
        out.append(rng.permutation(b)[:min(k, len(b))])
    return np.concatenate(out)


t0 = time.time(); rows = []
for wn, wlab in WINS:
    for m in MICE:
        for stage in STAGES:
            val = VALIDIX[(m, stage)]
            mu, sd = common_scale(m, stage, wn, val); sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
            IDX = {k: trials_of(m, stage, cs) for k, cs in SETS.items()}
            good = np.isfinite(AW[wn][np.ix_(np.concatenate(list(IDX.values())), val)]).all(0)
            n_match = min(len(v) for v in IDX.values())
            if n_match < 20 or good.sum() < 10:
                continue
            for k, cs in SETS.items():
                acc = {}
                for d in range(NDRAW):
                    rng = np.random.RandomState(SEED + 1000 * d + hash((m, stage, wn, k)) % 997)
                    sub = strat_subsample(IDX[k], cs, n_match, rng)
                    X = (AW[wn][np.ix_(sub, val)][:, good] - mu[good]) / sd[good]
                    X = np.nan_to_num(X)
                    v = dict(ID=two_nn_id(X), PR=part_ratio(X), R=float(np.mean(np.linalg.norm(X - X.mean(0), axis=1))),
                             SEP_sample=separation(X, SAMP[sub]), SEP_match=separation(X, (SAMP[sub] == TESTO[sub]).astype(int)))
                    if k == 'dual':
                        v['SEP_gng'] = separation(X, (TSK[sub] == 'DualGo').astype(int))
                    for kk, vv in v.items():
                        acc.setdefault(kk, []).append(vv)
                r = dict(set=k, win=wn, mouse=m, stage=stage, n=n_match, nneu=int(good.sum()))
                r.update({kk: float(np.nanmean(vv)) for kk, vv in acc.items()})
                rows.append(r)
    print(f'{wlab}: matched geometry done ({time.time() - t0:.0f} s)', flush=True)
R = pd.DataFrame(rows)

METRICS = ['ID', 'PR', 'R', 'SEP_sample', 'SEP_match', 'SEP_gng']
stats = []
print(f'\n== naive -> expert, n-matched (medians over mice, paired Wilcoxon, two-sided) ==')
for k in SETS:
    for wn, wlab in WINS:
        sub = R[(R.set == k) & (R.win == wn)]
        for mt in METRICS:
            if mt not in sub or sub[mt].isna().all():
                continue
            p_ = sub.pivot_table(index='mouse', columns='stage', values=mt).dropna()
            if len(p_) < 5:
                continue
            pv = float(wilcoxon(p_['Naive'], p_['Expert']).pvalue)
            stats.append(dict(set=k, win=wn, metric=mt, naive=float(p_['Naive'].median()), expert=float(p_['Expert'].median()),
                              p=pv, n=len(p_), nup=int((p_['Expert'] > p_['Naive']).sum())))
            print(f'  {k:4s} {wlab:10s} {mt:11s} {p_["Naive"].median():6.2f} -> {p_["Expert"].median():6.2f}   p = {pv:.3f}  ({int((p_["Expert"] > p_["Naive"]).sum())}/{len(p_)} up)')
S = pd.DataFrame(stats)

cross = []
print(f'\n== set contrasts at matched n (medians over mice, paired Wilcoxon, two-sided) ==')
for a, b in [('DPA', 'dual'), ('DPA', 'Go'), ('DPA', 'NoGo'), ('Go', 'NoGo')]:
    for wn, wlab in WINS:
        for stage in STAGES:
            sub = R[(R.win == wn) & (R.stage == stage) & (R.set.isin([a, b]))]
            for mt in ['ID', 'PR', 'R', 'SEP_sample', 'SEP_match']:
                p_ = sub.pivot_table(index='mouse', columns='set', values=mt).dropna()
                if len(p_) < 5 or a not in p_ or b not in p_:
                    continue
                pv = float(wilcoxon(p_[a], p_[b]).pvalue)
                cross.append(dict(a=a, b=b, win=wn, stage=stage, metric=mt, va=float(p_[a].median()), vb=float(p_[b].median()),
                                  p=pv, n=len(p_), nup=int((p_[b] > p_[a]).sum())))
                print(f'  {a:4s} vs {b:4s} {wlab:10s} {stage:6s} {mt:11s} {p_[a].median():6.2f} vs {p_[b].median():6.2f}   p = {pv:.3f}  ({int((p_[b] > p_[a]).sum())}/{len(p_)} up)')
C = pd.DataFrame(cross)

OUT = 'figures/pseudo/dimensionality/manifold_matched.pkl'
pickle.dump(dict(R=R, S=S, C=C, ndraw=NDRAW, seed=SEED), open(OUT, 'wb'))
print(f'\nwrote {OUT}  ({time.time() - t0:.0f} s, {NDRAW} draws)')
