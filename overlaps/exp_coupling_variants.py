"""exp_coupling_variants.py — the Fig. 4c coupling (per-mouse Δ choice-code depth vs Δ DPA accuracy on the
distractor-free DPA trials, n = 9 Spearman) recomputed under the decoder variants, for ED 3c
(2026-09-15). main_panels reads sys.argv at import, so each variant runs in its own subprocess; the nine
points per variant are cached so the ED figure never loads a tensor.

Variants: l2 (canonical ridge logistic, = Fig. 4c) | l1 (lasso) | lda (shrinkage-LDA).
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_coupling_variants.py
Output: figures/overlaps/controls/coupling_variants_cache.pkl  {variant: {mice, dd, dp_dpa, dp_gng, rho, p, ...}}
"""
import sys, os, pickle, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT = 'figures/overlaps/controls/coupling_variants_cache.pkl'
VARIANTS = {'l2': [], 'l1': ['--l1'], 'lda': ['--lda']}

if '--worker' in sys.argv:
    name = sys.argv[sys.argv.index('--worker') + 1]
    sys.argv = [sys.argv[0]] + VARIANTS[name]
    sys.path.insert(0, '/home/leon/dual/')
    import numpy as np
    from scipy.stats import spearmanr
    import main_panels as MP
    # 2026-09-21: Fig. 4c's drawn accuracy arm moved to ALL laser-off trials, but ED 7's other panels (units,
    # fixed axis, licking) hard-code the GNG-free DPA trials for both depth and accuracy, so this battery is
    # PINNED to the DPA arm and the whole ED page stays internally consistent. Do not swap it for
    # MP.delta_dpa_perf_sample, which now follows the main panel. The all-trial value is in the ED 7 legend.
    _dpa_perf = MP._perf_delta_by_sample('performance', MP.y.tasks == 'DPA')
    mice, dd, dpa, gng = [], [], [], []
    for mo in MP.ALL_MICE:
        d = np.nanmean([MP.delta_choice_sample[(mo, cls)] for cls, _ in MP.D_SAMPLE_CLASSES])
        a = np.nanmean([_dpa_perf.get((mo, cls), np.nan) for cls, _ in MP.D_SAMPLE_CLASSES])
        g = np.nanmean([MP.delta_gng_perf_sample.get((mo, cls), np.nan) for cls, _ in MP.D_SAMPLE_CLASSES])
        if np.isfinite(d) and np.isfinite(a):
            mice.append(mo); dd.append(float(d)); dpa.append(float(a)); gng.append(float(g))
    rho, p = spearmanr(dd, dpa); grho, gp = spearmanr(dd, gng)
    res = dict(mice=mice, dd=dd, dp_dpa=dpa, dp_gng=gng, rho=float(rho), p=float(p),
               grho=float(grho), gp=float(gp), axis_label=MP.AXIS_LABEL)
    print(f'{name}: DPA arm rho={rho:+.2f} p={p:.3f}  GNG arm rho={grho:+.2f} p={gp:.3f}  n={len(mice)}')
    pickle.dump(res, open(f'{OUT}.{name}', 'wb'))
    sys.exit(0)

res = {}
for name in VARIANTS:
    subprocess.run([sys.executable, __file__, '--worker', name], check=True)
    res[name] = pickle.load(open(f'{OUT}.{name}', 'rb')); os.remove(f'{OUT}.{name}')
pickle.dump(res, open(OUT, 'wb'))
print('saved', OUT)
