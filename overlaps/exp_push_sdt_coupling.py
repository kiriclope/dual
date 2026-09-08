"""exp_push_sdt_coupling.py — is the no-lick push a response-criterion shift or a sensitivity gain?
(Reviewer objection, 2026-09-07: what the animals learn is "when not to lick", so a shift along a
lick-trained axis could simply track a global criterion shift toward withholding, and would then
correlate with unpaired-trial accuracy for that reason alone.)

Test: signal-detection decomposition of DPA performance per mouse and stage on the same laser-OFF DPA
trials that Fig. 4c uses — hit = lick on a paired trial, false alarm = lick on an unpaired trial,
d' = z(H) − z(FA) (sensitivity), c = −(z(H) + z(FA))/2 (criterion; more positive = more withholding),
log-linear correction. Then the same between-mouse n = 9 Spearman as Fig. 4c (main_panels._panelC_coupling
convention: Δdepth = Expert − Naive, sample classes aggregated within mouse) of Δdepth against Δd', Δc,
ΔH and ΔFA, plus (i) a partial coupling controlling for naive accuracy (regression-to-the-mean check)
and (ii) the level-wise coupling (expert depth vs expert accuracy).
Run from overlaps/:  /home/leon/mambaforge/envs/dual/bin/python exp_push_sdt_coupling.py
"""
import sys, os
sys.path.insert(0, '/home/leon/dual/'); os.chdir('/home/leon/dual/overlaps'); sys.path.insert(0, '/home/leon/dual/overlaps')
import numpy as np
from scipy.stats import spearmanr, pearsonr, norm
import main_panels as MP
y = MP.y
MICE = MP.ALL_MICE
STAGES = ['Naive', 'Expert']
base = MP.idx_laser & MP.idx_choice & (y.tasks == 'DPA')


def sdt(mask):
    d = y.loc[mask]
    paired = d[d.pair == 1]; unp = d[d.pair == 0]
    H = (paired.choice.sum() + 0.5) / (len(paired) + 1)          # log-linear correction
    FA = (unp.choice.sum() + 0.5) / (len(unp) + 1)
    zH, zF = norm.ppf(H), norm.ppf(FA)
    return dict(H=H, FA=FA, dprime=zH - zF, c=-(zH + zF) / 2, acc=d.performance.mean(), n=len(d))


S = {(mo, st): sdt(base & (y.mouse == mo) & (y.stage == st)) for mo in MICE for st in STAGES}
dd = np.array([np.nanmean([MP.delta_choice_sample[(mo, cls)] for cls, _ in MP.D_SAMPLE_CLASSES]) for mo in MICE])
print('per-mouse Δdepth (Expert−Naive):', np.round(dd, 2).tolist())
print('\nmouse      naive: H    FA    d\'    c    acc | expert: H    FA    d\'    c    acc')
for mo in MICE:
    a, b = S[(mo, 'Naive')], S[(mo, 'Expert')]
    print(f"{mo:8s}  {a['H']:.2f} {a['FA']:.2f} {a['dprime']:5.2f} {a['c']:5.2f} {a['acc']:.2f} | "
          f"{b['H']:.2f} {b['FA']:.2f} {b['dprime']:5.2f} {b['c']:5.2f} {b['acc']:.2f}")


def couple(name, vals):
    ok = np.isfinite(vals) & np.isfinite(dd)
    rho, p = spearmanr(dd[ok], vals[ok]); r, pr = pearsonr(dd[ok], vals[ok])
    print(f'  Δdepth vs {name:26s} n={ok.sum()}  Spearman ρ={rho:+.2f} p={p:.3f}   Pearson r={r:+.2f} p={pr:.3f}   mean Δ={np.nanmean(vals):+.2f}')
    return rho, p


print('\n== between-mouse couplings of Δdepth (more negative = deeper into no-lick) ==')
delta = {k: np.array([S[(mo, 'Expert')][k] - S[(mo, 'Naive')][k] for mo in MICE]) for k in ['acc', 'dprime', 'c', 'H', 'FA']}
couple('ΔDPA accuracy (Fig. 4c)', delta['acc'])
couple("Δd' (sensitivity)", delta['dprime'])
couple('Δc (criterion, + = withhold)', delta['c'])
couple('Δhit rate', delta['H'])
couple('Δfalse-alarm rate', delta['FA'])

# regression-to-the-mean check: partial Spearman of Δdepth vs Δacc given naive accuracy
naive_acc = np.array([S[(mo, 'Naive')]['acc'] for mo in MICE])
def partial_spearman(x, yv, z):
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(yv), rankdata(z)
    def resid(a, b):
        A = np.c_[np.ones_like(b), b]; beta = np.linalg.lstsq(A, a, rcond=None)[0]; return a - A @ beta
    return pearsonr(resid(rx, rz), resid(ry, rz))
r_p, p_p = partial_spearman(dd, delta['acc'], naive_acc)
print(f'\npartial (rank) coupling Δdepth vs ΔDPA accuracy | naive accuracy: r={r_p:+.2f} p={p_p:.3f}')
print(f"naive accuracy vs Δaccuracy: Spearman ρ={spearmanr(naive_acc, delta['acc'])[0]:+.2f} p={spearmanr(naive_acc, delta['acc'])[1]:.3f}")
print(f"naive accuracy vs Δdepth:    Spearman ρ={spearmanr(naive_acc, dd)[0]:+.2f} p={spearmanr(naive_acc, dd)[1]:.3f}")

# level-wise coupling: expert depth vs expert accuracy / d' / c (per-mouse means over sample classes,
# the same trials and axis as delta_choice_sample in main_panels)
def depth_level(mo, st):
    v = []
    for cls, pairs in MP.D_SAMPLE_CLASSES:
        m = ((MP.Lm.mouse == mo) & (MP.Lm.stage == st) & MP.L_dpa & MP.Lm.odor_pair.isin(pairs)).values
        v.append(MP.lick_depth[m].mean() if m.sum() else np.nan)
    return np.nanmean(v)
print('\n== level-wise couplings (expert stage) ==')
lev = np.array([depth_level(mo, 'Expert') for mo in MICE])
for k in ['acc', 'dprime', 'c']:
    v = np.array([S[(mo, 'Expert')][k] for mo in MICE]); ok = np.isfinite(lev) & np.isfinite(v)
    rho, p = spearmanr(lev[ok], v[ok])
    print(f"  expert depth vs expert {k:7s}: Spearman ρ={rho:+.2f} p={p:.3f} n={ok.sum()}")
lev_n = np.array([depth_level(mo, 'Naive') for mo in MICE])
for k in ['acc', 'dprime', 'c']:
    v = np.array([S[(mo, 'Naive')][k] for mo in MICE]); ok = np.isfinite(lev_n) & np.isfinite(v)
    rho, p = spearmanr(lev_n[ok], v[ok])
    print(f"  naive  depth vs naive  {k:7s}: Spearman ρ={rho:+.2f} p={p:.3f} n={ok.sum()}")
