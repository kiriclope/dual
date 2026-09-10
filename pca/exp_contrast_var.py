"""exp_contrast_var.py — panel-b cache: reliable variance per DESIGN CONTRAST (2026-09-10).

Replaces the anonymous cvPCA component spectrum (SPEC_JK) with the same cross-validated reliable
variance read on the FIXED design contrasts. Leon 2026-09-10: "let's switch panel b, c and d to the
contrast decomposition", after the fitted-component shares were shown to be biased low for small
components.

WHY (short version; the long one is in cvpca.contrast_var and docs/pca/dimensionality.md). A fitted
component is measured along a direction estimated from the training half, and a component whose
signal sits below the per-direction noise energy of a half-mean cannot be located — its variance is
under-assigned. A design contrast has no fitted direction, so its cross-validated variance is
unbiased, and the ±1 contrasts of a 2-level factorial form a COMPLETE orthonormal basis of the
centred condition space: this is a change of basis, not a model, and the parts still sum to the same
unbiased total. The interaction contrasts come along for free and are the direct test of
factorisation — main effects carry the variance, interactions carry none.

Conventions are copied from exp_cdec_support.py so panel b keeps its error bars and null unchanged in
kind: 30 half-splits at seed 7, leave-one-mouse-out jackknife with t(8)=2.306, within-mouse
label-shuffle null at seed 11 expressed as a fraction of the REAL total (a null normalised by its own
near-zero total would be meaningless).

Merge-dumps {'CONTRAST_VAR', 'CONTRAST_NULL'} into results.pkl. Cache-only, no X reload (~15 min).

    CONTRAST_VAR[(set, window, stage)] = dict(names, frac, vals, total, se, lo, hi)
    CONTRAST_NULL[(set, window)]       = null fractions, aligned with names (Expert)

Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_contrast_var.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvpca                       # THE cvPCA estimator — one implementation (2026-09-09)

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
DPA4 = [('DPA', s, te) for s in (0, 1) for te in (0, 1)]
DUAL = [(t, s, te) for t in ['DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
SETS = [('DPA', DPA4), ('dual', DUAL)]
WINS = ['md', 'decision']
RES = 'figures/pseudo/dimensionality/results.pkl'

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW, VALIDIX, N = _c['AW'], _c['VALIDIX'], _c['N']
assert set(WINS) <= set(AW), (f'fits_inputs.pkl missing windows {sorted(set(WINS) - set(AW))} — '
                              'run exp_dimensionality_md.py first (merges ed/md/test into the cache)')
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
cvpca.bind(MOUSE=MOUSE, LEARN=LEARN, LAS=LAS, PERF=PERF, TSK=TSK, SAMP=SAMP, TESTO=TESTO,
           VALIDIX=VALIDIX, N=N, MICE=MICE)

NSPL, SEED, SEED_NULL, TCRIT = 30, 7, 11, 2.306    # t(df=8) 97.5% — n=9 mice, not z=1.96

print('══ CONTRAST_VAR: reliable variance per design contrast, + jackknife CIs ══')
CONTRAST_VAR = {}
for ts, conds in SETS:
    for wn in WINS:
        for stage in STAGES:
            names, vals, tot = cvpca.contrast_var(stage, conds, AW[wn], nsplits=NSPL, seed=SEED)
            jkv, jkt = [], []
            for mo in MICE:                        # leave one mouse out: its neurons AND its trials
                _, v, t = cvpca.contrast_var(stage, conds, AW[wn], [m for m in MICE if m != mo],
                                             nsplits=NSPL, seed=SEED)
                jkv.append(v); jkt.append(t)
            jkv = np.array(jkv); jkt = np.array(jkt); n = len(MICE)

            def _jk(point, reps):                  # jackknife SE, exp_dimensionality_jk convention
                return point, np.sqrt((n - 1) / n * ((reps - reps.mean(0)) ** 2).sum(0))
            # TWO normalisations, both from the same replicates so point and CI share one source.
            #  frac  = share of the UNBIASED total tr(A^T B). Exactly unbiased, but a single
            #          contrast can exceed 1 when the others come out negative (DPA mid-delay).
            #  fracp = negatives clipped, renormalised to sum to 1 — the convention avg_frac already
            #          uses for the component spectrum, so the panel keeps a 0-1 axis. This is what
            #          the figure draws; `frac` and `total` are the numbers for Methods.
            frac, se = _jk(vals / tot, jkv / jkt[:, None])
            _p = np.clip(vals, 0, None); fracp = _p / (_p.sum() + 1e-12)
            _pr = np.clip(jkv, 0, None); _pr = _pr / (_pr.sum(1, keepdims=True) + 1e-12)
            fracp, sep = _jk(fracp, _pr)
            CONTRAST_VAR[(ts, wn, stage)] = dict(
                names=names, vals=vals, total=tot,
                frac=frac, se=se, lo=frac - TCRIT * se, hi=frac + TCRIT * se,
                fracp=fracp, sep=sep, lop=np.clip(fracp - TCRIT * sep, 0, 1),
                hip=np.clip(fracp + TCRIT * sep, 0, 1))
            top = ' '.join(f'{nm} {100*f:.1f}%' for nm, f in zip(names, fracp) if f > 0.02)
            print(f'  {ts:4s} {wn:9s} {stage:6s} total {tot:7.0f} | {top}', flush=True)

print('\n══ CONTRAST_NULL: within-mouse label-shuffle null (÷ the real total; Expert) ══')
CONTRAST_NULL = {}
for ts, conds in SETS:
    for wn in WINS:
        _, nul, _ = cvpca.contrast_var('Expert', conds, AW[wn], nsplits=NSPL, seed=SEED_NULL,
                                       shuffle=True)
        # ÷ the real POSITIVE total, matching the fracp normalisation the panel draws
        tot = np.clip(CONTRAST_VAR[(ts, wn, 'Expert')]['vals'], 0, None).sum()
        CONTRAST_NULL[(ts, wn)] = nul / tot
        print(f'  {ts:4s} {wn:9s} null {np.round(100 * CONTRAST_NULL[(ts, wn)], 2)} %', flush=True)

D = pickle.load(open(RES, 'rb'))
D['CONTRAST_VAR'] = CONTRAST_VAR
D['CONTRAST_NULL'] = CONTRAST_NULL
pickle.dump(D, open(RES, 'wb'))
print(f'\nmerged CONTRAST_VAR ({len(CONTRAST_VAR)} keys) + CONTRAST_NULL into {RES}')
