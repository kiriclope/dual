"""PS cache — cross-task PARALLELISM SCORE per variable (Bernardi et al. 2020's geometric twin
of CCGP), for Fig 2 panel E (user routing 2026-09-01: "PS goes in main").

Per variable (sample @md, test @decision, choice @decision — the three panel-E matrices): the
task-wise CODING VECTOR is the correct-trial class-difference of condition means within each task
(DPA / DualGo / DualNoGo), computed on a trial half. PS(raw) = mean over the 3 task pairs of
cos(v_t1^halfA, v_t2^halfB) (both half pairings averaged — independent halves, so trial noise
cannot fake parallelism). Split-half reliability rel_t = cos(v_t^A, v_t^B) is disclosed and the
corrected PS = raw / mean(rel) is reported alongside (house rule: corrections at low rel are
estimates, not tests). NULL: within-task class-label shuffle (100 draws) -> the 95th percentile
of |PS_raw| under no true coding direction.

Merges {'PS'+SUF}: PS[var] = dict(raw, corrected, rel(mean), null95, per_pair). NREP=10 halves.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_parallelism.py --nopca
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decoders import SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
TASKS = ['DPA', 'DualGo', 'DualNoGo']
NREP, NSHUF = 10, 100

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)

# panel E is Expert; compute Expert (and print Naive for the record)
VAR_DEFS = [('sample', 'md', SAMP), ('test', 'decision', TESTO), ('choice', 'decision', MATCH)]


def coding_vec(stage, tk, wn, labels, idx_pool, rng=None, shuffle=False):
    """Class-difference vector of condition means (correct trials of task tk), on idx_pool."""
    M = AW[wn]
    lab = labels[idx_pool]
    if shuffle:
        lab = rng.permutation(lab)
    v = np.zeros(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        mm = MOUSE[idx_pool] == m
        i1 = idx_pool[mm & (lab == 1)]; i0 = idx_pool[mm & (lab == 0)]
        if len(i1) >= 3 and len(i0) >= 3:
            v[val] = np.nanmean(M[np.ix_(i1, val)], 0) - np.nanmean(M[np.ix_(i0, val)], 0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def halves(rng, idx):
    p = rng.permutation(idx); h = len(p) // 2
    return p[:h], p[h:]


print(f'== PARALLELISM SCORE (SUF="{SUF}") ==', flush=True)
OUT = {}
for stage in ['Naive', 'Expert']:
    for vname, wn, labels in VAR_DEFS:
        raws, rels, pairs = [], [], {}
        null = []
        rng = np.random.RandomState(700)
        for rep in range(NREP):
            VA, VB = {}, {}
            for tk in TASKS:
                pool = np.where((LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == tk))[0]
                a, b = halves(rng, pool)
                VA[tk] = coding_vec(stage, tk, wn, labels, a)
                VB[tk] = coding_vec(stage, tk, wn, labels, b)
            rels.append(np.mean([abs(VA[tk] @ VB[tk]) for tk in TASKS]))
            rep_pairs = {}
            for i in range(3):
                for j in range(i + 1, 3):
                    c = 0.5 * (VA[TASKS[i]] @ VB[TASKS[j]] + VA[TASKS[j]] @ VB[TASKS[i]])
                    rep_pairs[(TASKS[i], TASKS[j])] = c
                    pairs.setdefault((TASKS[i], TASKS[j]), []).append(c)
            raws.append(np.mean(list(rep_pairs.values())))
        # null: label shuffle within task, cross-half cosines (cheaper: 1 half-split per draw)
        rngn = np.random.RandomState(701)
        for _ in range(NSHUF):
            V = {}
            for tk in TASKS:
                pool = np.where((LEARN == stage) & (LAS == 0) & (PERF == 1) & (TSK == tk))[0]
                a, b = halves(rngn, pool)
                V[tk] = (coding_vec(stage, tk, wn, labels, a, rngn, True),
                         coding_vec(stage, tk, wn, labels, b, rngn, True))
            null.append(np.mean([0.5 * abs(V[TASKS[i]][0] @ V[TASKS[j]][1]
                                           + V[TASKS[j]][0] @ V[TASKS[i]][1])
                                 for i in range(3) for j in range(i + 1, 3)]))
        raw = float(np.mean(raws)); rel = float(np.mean(rels))
        ent = dict(raw=raw, rel=rel, corrected=float(raw / max(rel, 1e-9)),
                   null95=float(np.percentile(null, 95)),
                   per_pair={k: float(np.mean(v)) for k, v in pairs.items()})
        OUT[(stage, vname)] = ent
        print(f'  {stage:6s} {vname:7s} PS raw {raw:+.3f}  rel {rel:.3f}  '
              f'corr {ent["corrected"]:+.3f}  null95 {ent["null95"]:.3f}', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['PS' + SUF] = OUT
pickle.dump(d, open(RES, 'wb'))
print('merged PS' + SUF, 'into', RES)
