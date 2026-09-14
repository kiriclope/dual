"""exp_cvpca_bias_check.py — the evidence behind the cvPCA bias disclosure in the Methods.

`docs/paper/methods_notes.md` states that Fig 2b's per-component shares are biased LOW for components
below the half-mean noise floor, and quotes five numbers for it. Those numbers were measured in a
scratch directory that does not survive the session, which left a Methods claim with no script behind
it. This file is that script. Nothing here feeds a figure or a cache; it exists so the disclosure is
reproducible.

WHAT IT REPRODUCES (all five, in the order the Methods paragraph makes them):

  noise      the scale that sets the whole problem: the per-direction noise energy of a half-mean
             against the reliable variance of each component (dual mid-delay: ~400 vs 133 on the
             second component, which is why that direction cannot be located).
  sim        ground truth. Real residuals resampled within mouse, real trial and neuron counts, the
             signal placed on the mice's own measured contrast directions. A component whose TRUE
             share is 10% is reported as ~7%, one at 5% as ~1%, and the bias is gone by 20%. The
             fixed-contrast readout recovers the same truths within about a point.
  folds      a bigger training set helps but does not fix it: a true 10% reads 7.7 at 2-fold, 8.6 at
             5-fold and 8.6 at leave-one-trial-out. 10-fold is impossible, the smallest cell holds 6
             trials.
  trials     the assumption-free check on the REAL data. Subsample the trials and watch the reported
             share climb (Go/NoGo mid-delay component 2: 6.4% at 45% of trials to 10.1% at all of
             them) while the contrast readout stays flat (11.6 to 11.4). Also writes the diagnostic
             figure that is committed beside it.
  scaling    rules out the other suspect: pooled / within-condition-noise / raw per-neuron scaling
             give the Go/NoGo mid-delay sample share as 11.2, 9.8 and 10.7%.
  align      why the contrast readout is a SPECTRUM and not a relabelling, which is what licenses
             quoting it as the unbiased comparison: the design contrasts diagonalise the
             cross-validated signal covariance. Its eigenvalues match the contrast variances to about
             1.5 points in all eight cells (Expert DPA decision 47.6/39.5/12.8 vs 46.4/39.8/13.8).
             NOTE a trap: normalising the off-diagonals by sqrt(diag) explodes for the near-zero
             interaction contrasts, so read the eigenvalue comparison, not those correlations.

Cache-only, no 20 GB reload. The full set runs in about a minute; each check also runs alone.

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_cvpca_bias_check.py [check ...]
      (no arguments runs them all; `--figure` also rewrites the diagnostic PNG)
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pickle
import cvpca

CACHE = 'figures/pseudo/dimensionality/fits_inputs.pkl'
_c = pickle.load(open(CACHE, 'rb'))
AW, VALIDIX = _c['AW'], _c['VALIDIX']
MICE = sorted({m for m, _ in VALIDIX})
cvpca.bind_cache(_c, MICE)
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = {'DPA': [c for c in ALL12 if c[0] == 'DPA'], 'dual': [c for c in ALL12 if c[0] != 'DPA']}


def blocks_for(stage, conds, wn):
    """Per-mouse scaled trial blocks. The full-matrix path in cvpca is ~50x slower for these loops."""
    M = AW[wn]; sd = cvpca.neuron_scale(stage, M)
    blk, idx, off = {}, {}, 0
    for m in MICE:
        val = VALIDIX[(m, stage)]
        blk[m] = [M[np.ix_(cvpca.trials(stage, c, m), val)] / sd[val][None, :] for c in conds]
        idx[m] = (off, off + len(val)); off += len(val)
    return blk, idx, off


def halves(blk, idx, neur, nc, rng):
    A = np.zeros((nc, neur)); B = np.zeros((nc, neur))
    for m in MICE:
        a, b = idx[m]
        for ci, x in enumerate(blk[m]):
            p = rng.permutation(len(x)); h = len(p) // 2
            A[ci, a:b] = x[p[:h]].mean(0); B[ci, a:b] = x[p[h:]].mean(0)
    return A - A.mean(0, keepdims=True), B - B.mean(0, keepdims=True)


def synth_context(stage, conds, wn):
    """Per-mouse residuals and measured contrast directions, for the ground-truth simulations."""
    M = AW[wn]; sd = cvpca.neuron_scale(stage, M); ctx = {}
    for m in MICE:
        val = VALIDIX[(m, stage)]
        X = [M[np.ix_(cvpca.trials(stage, c, m), val)] / sd[val][None, :] for c in conds]
        res = np.vstack([(x - x.mean(0, keepdims=True)) * np.sqrt(len(x) / (len(x) - 1)) for x in X])
        mu = np.array([x.mean(0) for x in X]); mu -= mu.mean(0, keepdims=True)
        ctx[m] = dict(n=[len(x) for x in X], res=res, mu=mu, k=len(val))
    return ctx


def synth(ctx, rng, share2, total, cvecs):
    """Signal of a KNOWN spectrum on the mice's own measured contrast directions, plus real noise."""
    data = {}
    for m in MICE:
        d = ctx[m]
        Q, _ = np.linalg.qr(np.array([d['mu'].T @ v for v in cvecs]).T)
        Q = Q * np.sqrt(1.0 / len(MICE))
        mu = (np.sqrt(total * (1 - share2)) * np.outer(cvecs[0], Q[:, 0])
              + np.sqrt(total * share2) * np.outer(cvecs[1], Q[:, 1]))
        data[m] = [mu[ci][None, :] + d['res'][rng.integers(0, len(d['res']), n)]
                   for ci, n in enumerate(d['n'])]
    return data


def _spec_from(data, rng, mode, npair, neur, nc, cvec2=None):
    """mode: ('half',) repeated 2-fold | ('kfold', K) | ('loo',). Returns (comp-2 share %, contrast %)."""
    sd = {}
    for m in MICE:
        v = np.vstack(data[m]).std(0); sd[m] = np.where(v > 1e-6, v, 1.0)
    off = {}; o = 0
    for m in MICE:
        kk = data[m][0].shape[1]; off[m] = (o, o + kk); o += kk
    pairs = []
    for _ in range(npair):
        if mode[0] == 'kfold':
            K = mode[1]
            TR = [np.zeros((nc, neur)) for _ in range(K)]; TE = [np.zeros((nc, neur)) for _ in range(K)]
            for m in MICE:
                a, b = off[m]
                for ci, x in enumerate(data[m]):
                    parts = np.array_split(rng.permutation(len(x)), K)
                    for f in range(K):
                        rest = np.concatenate([parts[q] for q in range(K) if q != f])
                        TR[f][ci, a:b] = x[rest].mean(0) / sd[m]
                        TE[f][ci, a:b] = x[parts[f]].mean(0) / sd[m]
            pairs += list(zip(TR, TE))
        elif mode[0] == 'loo':
            A = np.zeros((nc, neur)); B = np.zeros((nc, neur))
            for m in MICE:
                a, b = off[m]
                for ci, x in enumerate(data[m]):
                    j = rng.integers(0, len(x)); keep = np.ones(len(x), bool); keep[j] = False
                    A[ci, a:b] = x[keep].mean(0) / sd[m]; B[ci, a:b] = x[j] / sd[m]
            pairs.append((A, B))
        else:
            A = np.zeros((nc, neur)); B = np.zeros((nc, neur))
            for m in MICE:
                a, b = off[m]
                for ci, x in enumerate(data[m]):
                    p = rng.permutation(len(x)); h = len(p) // 2
                    A[ci, a:b] = x[p[:h]].mean(0) / sd[m]; B[ci, a:b] = x[p[h:]].mean(0) / sd[m]
            pairs += [(A, B), (B, A)]
    sp, tt, cv = [], [], []
    for A, B in pairs:
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
        sp.append(cvpca.cvpca_spectrum(A, B))
        tt.append(float(np.sum(A * B)))                     # trace(A^T B) without the N x N product
        if cvec2 is not None:
            cv.append(float(np.dot(A.T @ cvec2, B.T @ cvec2)))
    pos = np.clip(np.mean(sp, 0), 0, None)
    return 100 * pos[1] / pos.sum(), (100 * np.mean(cv) / np.mean(tt) if cvec2 is not None else np.nan)


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def dual_contrasts(conds):
    g = unit([-1.0 if c[0] == 'DualGo' else 1.0 for c in conds])
    s = unit([-1.0 if c[1] == 0 else 1.0 for c in conds])
    t = unit([-1.0 if c[2] == 0 else 1.0 for c in conds])
    return g, s, t


# ── the five checks ───────────────────────────────────────────────────────────────────────────────
def check_noise():
    """The scale that sets the problem: noise energy per direction against each component."""
    print('\n== noise: per-direction noise energy of a half-mean vs the reliable variance ==')
    for sn, wn in [('dual', 'md'), ('dual', 'decision'), ('DPA', 'md'), ('DPA', 'decision')]:
        conds = SETS[sn]; blk, idx, neur = blocks_for('Expert', conds, wn)
        rng = np.random.default_rng(0); sp, nz = [], []
        for _ in range(25):
            A, B = halves(blk, idx, neur, len(conds), rng)
            sp.append(cvpca.cvpca_spectrum(A, B))
            nz.append(np.linalg.svd((A - B) / 2, compute_uv=False) ** 2)
        # (A-B)/2 carries half a half-mean's noise variance, so x2 puts it on the component scale
        print(f'  {sn:4s} {wn:9s} components {np.round(np.mean(sp, 0)[:3], 0)}   '
              f'noise/direction {np.round(2 * np.mean(nz, 0)[:3], 0)}')


def check_sim(reps=8, splits=20, total=1350.0):
    """Ground truth: what the estimator reports when we know the answer."""
    print('\n== sim: TRUE share of component 2 -> what each readout reports (Expert dual md) ==')
    conds = SETS['dual']; ctx = synth_context('Expert', conds, 'md')
    g, s, _ = dual_contrasts(conds); neur = sum(ctx[m]['k'] for m in MICE)
    print(f'{"true %":>8s} | {"fitted basis":>16s} | {"design contrast":>18s}')
    for share in [0.02, 0.05, 0.10, 0.20, 0.30]:
        pc, ct = [], []
        for r in range(reps):
            data = synth(ctx, np.random.default_rng(1000 + r), share, total, [g, s])
            a, b = _spec_from(data, np.random.default_rng(7700 + r), ('half',),
                              splits, neur, len(conds), cvec2=s)
            pc.append(a); ct.append(b)
        print(f'{100*share:8.1f} | {np.mean(pc):9.1f} +-{np.std(pc):4.1f} | '
              f'{np.mean(ct):11.1f} +-{np.std(ct):4.1f}')


def check_folds(reps=8, total=1350.0):
    """A bigger training set shrinks the bias but does not remove it."""
    print('\n== folds: same ground truth, more training data per fit (Expert dual md) ==')
    conds = SETS['dual']; ctx = synth_context('Expert', conds, 'md')
    g, s, _ = dual_contrasts(conds); neur = sum(ctx[m]['k'] for m in MICE)
    modes = [(('half',), '2-fold'), (('kfold', 5), '5-fold'), (('loo',), 'leave-one-trial-out')]
    print(f'{"true %":>8s} |' + ''.join(f'{lab:>22s} |' for _, lab in modes))
    for share in [0.05, 0.10, 0.20]:
        row = f'{100*share:8.1f} |'
        for mode, _ in modes:
            npair = 30 if mode[0] == 'half' else (60 // mode[1] if mode[0] == 'kfold' else 60)
            v = [_spec_from(synth(ctx, np.random.default_rng(500 + r), share, total, [g, s]),
                            np.random.default_rng(7700 + r), mode, npair, neur, len(conds))[0]
                 for r in range(reps)]
            row += f'{np.mean(v):15.1f} +-{np.std(v):4.1f} |'
        print(row)


def check_trials(reps=12, splits=20, figure=False):
    """Assumption-free, on the REAL data: the fitted share is still climbing with trial count."""
    print('\n== trials: reported share vs the fraction of trials used ==')
    fracs = [0.45, 0.60, 0.75, 0.90, 1.00]
    cells = [('Expert', 'dual', 'md', 1, 'sample'), ('Expert', 'dual', 'decision', 1, 'choice'),
             ('Naive', 'dual', 'decision', 1, 'choice'), ('Expert', 'DPA', 'decision', 2, 'test')]
    store = {}
    for stage, sn, wn, pidx, cname in cells:
        conds = SETS[sn]; blk, idx, neur = blocks_for(stage, conds, wn)
        cv = dict(zip(['gng', 'sample', 'test'], dual_contrasts(conds))) if sn == 'dual' else {}
        if sn != 'dual':
            cv['sample'] = unit([-1.0 if c[1] == 0 else 1.0 for c in conds])
            cv['test'] = unit([-1.0 if c[2] == 0 else 1.0 for c in conds])
        cv['choice'] = unit(cv['sample'] * cv['test'] * np.sqrt(len(conds)))
        vec = cv[cname]
        ntr = float(np.mean([len(x) for m in MICE for x in blk[m]]))
        pm, cm = [], []
        for f in fracs:
            pv, tv = [], []
            for r in range(reps):
                rng = np.random.default_rng(9000 + r)
                sub = {m: [x[rng.choice(len(x), max(2, int(round(f * len(x)))), replace=False)]
                           for x in blk[m]] for m in MICE}
                sd = {m: np.where(np.vstack(sub[m]).std(0) > 1e-6,
                                  np.vstack(sub[m]).std(0), 1.0) for m in MICE}
                sp, tt, cc = [], [], []
                for _ in range(splits):
                    A = np.zeros((len(conds), neur)); B = np.zeros((len(conds), neur))
                    for m in MICE:
                        a, b = idx[m]
                        for ci, x in enumerate(sub[m]):
                            p = rng.permutation(len(x)); h = len(p) // 2
                            A[ci, a:b] = x[p[:h]].mean(0) / sd[m]
                            B[ci, a:b] = x[p[h:]].mean(0) / sd[m]
                    A -= A.mean(0, keepdims=True); B -= B.mean(0, keepdims=True)
                    sp.append(cvpca.cvpca_spectrum(A, B)); tt.append(float(np.sum(A * B)))
                    cc.append(float(np.dot(A.T @ vec, B.T @ vec)))
                pos = np.clip(np.mean(sp, 0), 0, None)
                pv.append(100 * pos[pidx] / pos.sum()); tv.append(100 * np.mean(cc) / np.mean(tt))
            pm.append((np.mean(pv), np.std(pv))); cm.append((np.mean(tv), np.std(tv)))
        store[(stage, sn, wn)] = dict(ntr=ntr, pidx=pidx, cname=cname,
                                      pc=np.array(pm), ct=np.array(cm))
        print(f'  {stage:6s} {sn:4s} {wn:9s} component {pidx+1} '
              f'{np.round([p[0] for p in pm], 1)}  |  {cname} contrast '
              f'{np.round([c[0] for c in cm], 1)}')
    if figure:
        _draw_trials(store, fracs)
    return store


def _draw_trials(store, fracs):
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_context('notebook'); sns.set_style('ticks')
    PS = 1.15
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 400, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.labelsize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'axes.linewidth': 0.7, 'lines.linewidth': 1.3})
    fig, axes = plt.subplots(1, len(store), figsize=(11.0, 2.6))
    for ax, ((stage, sn, wn), d) in zip(np.atleast_1d(axes), store.items()):
        x = np.array(fracs) * d['ntr']
        ax.errorbar(x, d['pc'][:, 0], yerr=d['pc'][:, 1], fmt='-o', ms=3.2, lw=1.2, color='#332288')
        ax.errorbar(x, d['ct'][:, 0], yerr=d['ct'][:, 1], fmt='-s', ms=3.2, lw=1.2, color='#44AA99')
        top = max(d['ct'][:, 0].max(), d['pc'][:, 0].max()) * 1.55
        hi = d['pc'][-1, 0] >= d['ct'][-1, 0]; o = 0.11 * top
        ax.text(x[-1], d['pc'][-1, 0] + (o if hi else -o),
                f'  component {d["pidx"]+1}\n  (fitted basis)', color='#332288',
                fontsize=PS*6.0, va='center', ha='left')
        ax.text(x[-1], d['ct'][-1, 0] + (-o if hi else o),
                f'  {d["cname"]} contrast\n  (fixed basis)', color='#44AA99',
                fontsize=PS*6.0, va='center', ha='left')
        ax.set_xlabel('trials per mouse × condition')
        ax.set_title(f'{stage} · {sn} · {"mid-delay" if wn == "md" else "decision"}',
                     loc='left', fontsize=PS*8)
        ax.set_ylim(0, top); ax.set_xlim(x[0] - 1, x[-1] + (x[-1] - x[0]) * 0.95)
        sns.despine(ax=ax)
    np.atleast_1d(axes)[0].set_ylabel('reliable variance (%)')
    fig.tight_layout()
    out = 'figures/pseudo/dimensionality/png/cvpca_small_component_bias.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
    print('  figure ->', os.path.abspath(out))


def check_scaling(splits=25):
    """The other suspect, ruled out: the per-neuron scaling is not what does it."""
    print('\n== scaling: does the per-neuron normalisation explain the small shares? ==')
    for stage, sn, wn in [('Expert', 'dual', 'md'), ('Expert', 'dual', 'decision')]:
        conds = SETS[sn]; M = AW[wn]; g, s, t = dual_contrasts(conds)
        cv = {'gng': g, 'sample': s, 'test': t, 'choice': unit(s * t * np.sqrt(len(conds)))}
        raw, idx, off = {}, {}, 0
        for m in MICE:
            val = VALIDIX[(m, stage)]
            raw[m] = [M[np.ix_(cvpca.trials(stage, c, m), val)] for c in conds]
            idx[m] = (off, off + len(val)); off += len(val)
        for mode in ['pooled', 'noise', 'raw']:
            sd = {}
            for m in MICE:
                allt = np.vstack(raw[m])
                if mode == 'pooled':
                    v = allt.std(0)
                elif mode == 'noise':
                    v = np.vstack([b - b.mean(0, keepdims=True) for b in raw[m]]).std(0) * \
                        np.sqrt(len(allt) / max(len(allt) - len(conds), 1))
                else:
                    v = np.ones(allt.shape[1])
                sd[m] = np.where(np.isfinite(v) & (v > 1e-6), v, 1.0)
            rng = np.random.default_rng(3); acc = {k: 0.0 for k in cv}; tot = 0.0
            for _ in range(splits):
                A = np.zeros((len(conds), off)); B = np.zeros((len(conds), off))
                for m in MICE:
                    a, b = idx[m]
                    for ci, x in enumerate(raw[m]):
                        p = rng.permutation(len(x)); h = len(p) // 2
                        A[ci, a:b] = x[p[:h]].mean(0) / sd[m]
                        B[ci, a:b] = x[p[h:]].mean(0) / sd[m]
                A -= A.mean(0, keepdims=True); B -= B.mean(0, keepdims=True)
                for k, v in cv.items():
                    acc[k] += float(np.dot(A.T @ v, B.T @ v))
                tot += float(np.sum(A * B))
            print(f'  {stage} {sn} {wn:9s} {mode:7s} ' +
                  ' '.join(f'{k} {100*acc[k]/tot:5.1f}%' for k in cv))


def check_align(splits=30):
    """Why the contrast readout is a spectrum: the contrasts diagonalise the signal covariance."""
    print('\n== align: contrast variances (the diagonal) vs the basis-free eigenvalues ==')
    for stage in ['Expert', 'Naive']:
        for sn, wn in [('DPA', 'md'), ('dual', 'md'), ('DPA', 'decision'), ('dual', 'decision')]:
            conds = SETS[sn]; names, V = cvpca.contrast_basis(conds)
            blk, idx, neur = blocks_for(stage, conds, wn)
            rng = np.random.default_rng(7); acc = np.zeros((len(names),) * 2); tot = 0.0
            for _ in range(splits):
                A, B = halves(blk, idx, neur, len(conds), rng)
                SA, SB = V @ A, V @ B
                acc += 0.5 * (SA @ SB.T + SB @ SA.T)
                tot += float(np.sum(A * B))
            C = acc / splits; tot /= splits
            d = np.diag(C); k = min(4, len(names)); o = np.argsort(-d)[:k]
            ev = np.sort(np.linalg.eigvalsh(C))[::-1][:k]
            print(f'  {stage:6s} {sn:4s} {wn:9s} diagonal ' +
                  ' '.join(f'{names[q]} {100*d[q]/tot:5.1f}%' for q in o))
            print(f'  {"":6s} {"":4s} {"":9s} eigenvals ' +
                  ' '.join(f'{100*e/tot:5.1f}%' for e in ev))


CHECKS = {'noise': check_noise, 'sim': check_sim, 'folds': check_folds,
          'trials': check_trials, 'scaling': check_scaling, 'align': check_align}

if __name__ == '__main__':
    want = [a for a in sys.argv[1:] if not a.startswith('--')] or list(CHECKS)
    bad = [w for w in want if w not in CHECKS]
    assert not bad, f'unknown check {bad}; pick from {list(CHECKS)}'
    for w in want:
        if w == 'trials':
            CHECKS[w](figure='--figure' in sys.argv[1:])
        else:
            CHECKS[w]()
    print('\ndone —', ', '.join(want))
