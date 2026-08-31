"""DIST_TASK cache for Fig 2 panel E (added 2026-08-31, user request): decode dist (Go vs NoGo,
dual trials) THROUGH each task's state geometry — the panel-C 'dist cross <- DPA PCs' analysis
(exp_cdec_support.gng_from_dpa) generalized to all three source tasks.

  rows    = the geometry the readout goes through: top-3 PCs of the source task's condition means
            (DPA / DualGo / DualNoGo x sample x test = 4 conds each) @ md. The Go-vs-NoGo LDA is
            always FIT on held-in dual pseudo-trials projected into that subspace (dist labels only
            exist across the dual pair); what changes per row is WHOSE geometry carries the code.
  columns = test set: DPA trials (FRACTION CLASSIFIED TO THE NoGo SIDE — descriptive, there is no
            Go/NoGo ground truth on DPA trials; the figure marks the column with a dagger and leaves
            the train-DPA x test-DPA cell blank per the user's design), held-out Go trials (Go-class
            accuracy), held-out NoGo trials (NoGo-class accuracy).

Cells are RAW class accuracies / fractions (no within-task ceiling exists for dist, so no ratio
normalisation). Window md = bins 36-38 (T_WINDOW=0.5 pca convention, same as panel C).
Merge-dumps {'DIST_TASK'} into results.pkl:  DIST_TASK[(wn, stage)] = 3x3 (rows/cols DPA,Go,NoGo).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dist_task.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
GO4 = [('DualGo', s, te) for s in (0, 1) for te in (0, 1)]
NOGO4 = [('DualNoGo', s, te) for s in (0, 1) for te in (0, 1)]
DUAL = GO4 + NOGO4
GLAB = np.array([1 if c[0] == 'DualGo' else 0 for c in DUAL])         # 1 = Go
SOURCES = [('DPA', DPA4), ('Go', GO4), ('NoGo', NOGO4)]
WINS = ['md']
NREP = 16                                                             # DPA_GNG_C uses 8; 16 for the 9 cells

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
assert set(WINS) <= set(AW), 'run exp_dimensionality_md.py first'
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def cond_means_all(stage, M, conds):
    R = np.zeros((len(conds), N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci, (t, s, te) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
            if len(idx):
                R[ci][val] = np.nanmean(M[np.ix_(idx, val)], axis=0)
    return R


def pools_split(stage, conds, rng):
    """Per (cond, mouse) trial pools split into disjoint halves (train, test)."""
    tr, te = {}, {}
    for m in MICE:
        for ci, (t, s, te_) in enumerate(conds):
            idx = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                           & (TSK == t) & (SAMP == s) & (TESTO == te_))[0]
            p = rng.permutation(idx); h = len(p) // 2
            tr[(ci, m)], te[(ci, m)] = p[:h], p[h:]
    return tr, te


def make_pseudo(pool, stage, conds, K, rng, M):
    Xp = np.zeros((len(conds) * K, N)); crow = np.repeat(np.arange(len(conds)), K)
    for ci in range(len(conds)):
        for m in MICE:
            val = VALIDIX[(m, stage)]; pi = pool[(ci, m)]
            if len(pi):
                block = M[np.ix_(pi[rng.randint(0, len(pi), K)], val)]
                bad = ~np.isfinite(block)
                if bad.any():
                    block[bad] = 0.0
                Xp[ci * K:(ci + 1) * K, val] = block
    return Xp, crow


def one_rep(stage, M, src_conds, rng):
    """One replicate: subspace from src_conds' condition means (label-blind wrt Go/NoGo), LDA fit on
    held-in dual pseudo-trials, then (frac-NoGo on DPA trials, Go-class acc, NoGo-class acc)."""
    R = cond_means_all(stage, M, src_conds); mu = R.mean(0); sd = R.std(0) + 1e-9
    Rc = (R - mu) / sd; Rc = Rc - Rc.mean(0)
    Vt = np.linalg.svd(Rc, full_matrices=False)[2][:3]                # top-3 = the source geometry
    trp, tep = pools_split(stage, DUAL, rng)
    Xtr, ctr = make_pseudo(trp, stage, DUAL, 24, rng, M)
    Xte, cte = make_pseudo(tep, stage, DUAL, 24, rng, M)
    clf = LinearDiscriminantAnalysis().fit(((Xtr - mu) / sd) @ Vt.T, GLAB[ctr])
    pred = clf.predict(((Xte - mu) / sd) @ Vt.T)
    acc_go = float(np.mean(pred[GLAB[cte] == 1] == 1))                # Go-class accuracy
    acc_ng = float(np.mean(pred[GLAB[cte] == 0] == 0))                # NoGo-class accuracy
    # DPA trials: no Go/NoGo truth — fraction classified to the NoGo side (descriptive). All DPA
    # trials are eligible (they never enter LDA training; the subspace itself is label-blind).
    dpool = {}
    for ci, (t, s, te_) in enumerate(DPA4):
        for m in MICE:
            dpool[(ci, m)] = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                                      & (TSK == t) & (SAMP == s) & (TESTO == te_))[0]
    Xd, _ = make_pseudo(dpool, stage, DPA4, 24, rng, M)
    frac_ng = float(np.mean(clf.predict(((Xd - mu) / sd) @ Vt.T) == 0))
    return frac_ng, acc_go, acc_ng


print('══ DIST_TASK: dist (Go vs NoGo) through each task\'s geometry ══')
DIST_TASK = {}
for wn in WINS:
    for stage in STAGES:
        Mmat = np.zeros((3, 3))
        for i, (sname, src_conds) in enumerate(SOURCES):
            rng = np.random.RandomState(700 + i)
            reps = np.array([one_rep(stage, AW[wn], src_conds, rng) for _ in range(NREP)])
            Mmat[i] = reps.mean(0)                                    # [frac-NoGo(DPA), acc Go, acc NoGo]
            print(f'  {wn} {stage:6s} src={sname:4s}  DPA→NoGo-side {Mmat[i,0]:.2f}  '
                  f'Go acc {Mmat[i,1]:.2f}  NoGo acc {Mmat[i,2]:.2f}', flush=True)
        DIST_TASK[(wn, stage)] = Mmat

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['DIST_TASK'] = DIST_TASK
pickle.dump(d, open(RES, 'wb'))
print('merged DIST_TASK into', RES)
