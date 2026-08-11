"""Kobak-style significant-component count (dPCA-marginalization significance, window states).

Counts, per (condition set x window x stage), how many task variables have a SIGNIFICANT dedicated
(demixed) axis: for each orthogonal design contrast, the axis is the contrast applied to leakage-free
TRAIN condition means; single held-out pseudo-trials are projected on it and classified; the variable
counts if held-out balanced accuracy beats the 95th percentile of a within-mouse label-shuffle null.
This is the decoding-based significance test of Kobak et al. 2016 (dPCA), applied to window-averaged
states — with binary factors each marginalization is rank-1, so "significant components per
marginalization" reduces to one test per contrast. Amplitude-free by construction (a 2% sample axis
and a 90% gng axis face the same criterion): the count companion to the variance-weighted PR.

Inputs: fits_inputs.pkl (AW window matrices, VALIDIX, labels) — no X reload.
Output: merge-dumps {'DPCA_COUNT': {(set, window, stage): {contrast: dict(acc, null95, sig)}}} into
results.pkl and prints the count table.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_dpca_count.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
DUAL = [c for c in ALL12 if c[0] != 'DPA']
DPA4 = [c for c in ALL12 if c[0] == 'DPA']
SETS = {'DPA': DPA4, 'dual': DUAL, 'all': ALL12}
WINS = ['ed', 'md', 'delay', 'test', 'decision']
NSPLIT, NNULL, KPSEUDO = 15, 40, 10

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']; N = _c['N']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])


def contrasts(conds):
    s = np.array([c[1] for c in conds], float) * 2 - 1
    te = np.array([c[2] for c in conds], float) * 2 - 1
    tk = np.array([c[0] for c in conds])
    out = {'sample': s, 'test': te, 'choice': s * te}
    if 'DualGo' in tk and 'DualNoGo' in tk:
        out['gng'] = np.where(tk == 'DualGo', 1.0, np.where(tk == 'DualNoGo', -1.0, 0.0))
    if 'DPA' in tk and 'DualGo' in tk:
        out['tasks'] = np.where(tk == 'DPA', 2.0, -1.0)
    return out


def neuron_scale(stage, M):
    sd = np.ones(N)
    for m in MICE:
        val = VALIDIX[(m, stage)]
        tr = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1))[0]
        if len(tr):
            s = np.nanstd(M[np.ix_(tr, val)], axis=0)
            sd[val] = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return sd


def pools(stage, conds):
    """(mouse, cond_idx) -> trial indices (correct, laser-off)."""
    P = {}
    for m in MICE:
        for ci, (t, s, te) in enumerate(conds):
            P[(m, ci)] = np.where((MOUSE == m) & (LEARN == stage) & (LAS == 0) & (PERF == 1)
                                  & (TSK == t) & (SAMP == s) & (TESTO == te))[0]
    return P


def one_run(M, sd, stage, P, conds, C, rng, shuffle):
    """One train/test split -> {contrast: balanced held-out accuracy}. shuffle=True permutes,
    within each mouse, the trial->condition assignment (the label-shuffle null)."""
    nc = len(conds)
    if shuffle:
        P2 = {}
        for m in MICE:
            allidx = np.concatenate([P[(m, ci)] for ci in range(nc)])
            perm = rng.permutation(allidx); k = 0
            for ci in range(nc):
                n = len(P[(m, ci)]); P2[(m, ci)] = perm[k:k + n]; k += n
        P = P2
    train, test = {}, {}
    for key, idx in P.items():
        p = rng.permutation(idx); h = len(p) // 2
        train[key], test[key] = p[:h], p[h:]
    # train condition means, column-by-mouse
    R = np.zeros((nc, N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci in range(nc):
            tr = train[(m, ci)]
            if len(tr):
                R[ci][val] = np.nanmean(M[np.ix_(tr, val)], 0)
    R = R / sd[None, :]
    mu = R.mean(0, keepdims=True); R = R - mu
    # held-out single pseudo-trials (one random test trial per mouse per pseudo-trial;
    # residual NaNs imputed from the raw-scale train condition mean — train info only, no leakage)
    PT = np.zeros((nc, KPSEUDO, N))
    for m in MICE:
        val = VALIDIX[(m, stage)]
        for ci in range(nc):
            pool = test[(m, ci)]
            if not len(pool):                       # never hit (>=6 trials/cell) — keep at train mean,
                continue                            # which after centering contributes 0 signal
            picks = rng.choice(pool, KPSEUDO, replace=True)
            block = M[np.ix_(picks, val)]
            bad = ~np.isfinite(block)
            if bad.any():
                fill = (R[ci][val] + mu[0][val]) * sd[val]
                block[bad] = np.broadcast_to(fill, block.shape)[bad]
            PT[ci, :, val] = block.T
    PT = PT / sd[None, None, :] - mu[0][None, None, :]
    out = {}
    for f, c in C.items():
        w = (c @ R) / (c @ c)
        pj = PT @ w                                       # (nc, KPSEUDO)
        pr = R @ w                                        # train projections
        pos, neg = c > 0, c < 0
        mp, mn = pr[pos].mean(), pr[neg].mean()
        b = 0.5 * (mp + mn)
        acc_p = np.mean((pj[pos] > b) == (mp > b))
        acc_n = np.mean((pj[neg] > b) == (mn > b))
        out[f] = 0.5 * (acc_p + acc_n)
    return out


OUT = {}
for wn in WINS:
    M = AW[wn]
    for sname, conds in SETS.items():
        C = contrasts(conds)
        for stage in STAGES:
            sd = neuron_scale(stage, M)
            P = pools(stage, conds)
            rng = np.random.RandomState(0)
            real = {f: [] for f in C}
            for _ in range(NSPLIT):
                r = one_run(M, sd, stage, P, conds, C, rng, shuffle=False)
                for f in C:
                    real[f].append(r[f])
            null = {f: [] for f in C}
            for _ in range(NNULL):
                r = one_run(M, sd, stage, P, conds, C, rng, shuffle=True)
                for f in C:
                    null[f].append(r[f])
            res = {}
            for f in C:
                acc = float(np.mean(real[f])); n95 = float(np.percentile(null[f], 95))
                res[f] = dict(acc=acc, null95=n95, sig=bool(acc > n95))
            OUT[(sname, wn, stage)] = res
            sigs = [f for f in C if res[f]['sig']]
            parts = '  '.join(f"{f}={res[f]['acc']:.2f}{'*' if res[f]['sig'] else ' '}(n95 {res[f]['null95']:.2f})"
                              for f in C)
            print(f"{wn:8s} {sname:4s} {stage:6s} n_sig={len(sigs)} {sigs}\n         {parts}", flush=True)

RESPKL = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RESPKL, 'rb'))
d['DPCA_COUNT'] = OUT
pickle.dump(d, open(RESPKL, 'wb'))
print('merged DPCA_COUNT into', RESPKL)
