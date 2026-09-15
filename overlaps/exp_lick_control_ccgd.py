"""exp_lick_control_ccgd.py — the late-delay lick control on the CCGD depth (ED 3d, reinstated 2026-09-15).

Leon: "make sure all the results are cross validated". The old control (fig_overlaps_lick_control_supp.py)
projected training trials on the decoder weights. Here the depth is Fig. 4's own: the cross-validated CCGD
choice decision function (tensor X_<BDUM>, choice rows, decoders trained at the decision window 54–62), read
at late delay 45–53, RAW log-odds, laser-off DPA trials — identical to main_panels.lick_depth.

Trial alignment is impossible (ccgd_validation returns rows in unseeded fold order), so the control is at the
level the §4 claim needs: per mouse x sample x stage. Late-delay lick rate (7.0–7.5 s after sample onset, the
window of the old control) comes from the behaviour .mat files, per laser-off DPA trial, averaged per
mouse x sample class x stage.
  push     depth ~ stage + sample [+ lick] + (1|mouse), 36 obs
  coupling per-mouse Spearman Δdepth vs ΔDPA accuracy (GNG-free DPA trials), partial for Δlick
  descriptive: per mouse x stage mean depth vs mean lick rate (18 points)
Dumps ed3_cache['lick_ccgd'].
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_lick_control_ccgd.py
"""
import sys, os, glob, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from src.pca.io import pkl_load
from src.common.options import set_options

PATH = '/storage/leon/dual_task/data/2Samples-DualTask-BehavioralData'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
ACT, LD, DELAY = np.arange(54, 63), np.arange(45, 54), (6.0, 7.5)   # 6.0-7.5 s after the .mat 'Sample' stamp = imaging 7.5-9.0 s = bins_LD (the stamp sits 1.5 s before the imaging clock: cue licks peak at 5.0-6.0, test licks at 7.5-9.0)
SAMPLES = [('A', [0, 1]), ('B', [2, 3])]

# ── the CCGD depth (held-out decision functions), exactly as Fig. 4 ──
y = pkl_load(f'labels_{BDUM}', path='../data/overlaps'); X = np.asarray(pkl_load(f'X_{BDUM}', path='../data/overlaps'))
ch = (y.target == 'choice').to_numpy(); D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True); del X
yc['depth'] = D[:, LD].mean(1)
d = yc[(yc.laser == 0) & (yc.tasks == 'DPA')].copy()
d['sample'] = np.where(d.odor_pair.isin([0, 1]), 'A', 'B'); d['st'] = (d.stage == 'Expert').astype(int)

# ── late-delay lick rate per behavioural trial → per mouse x sample x stage ──
def find_mat(folder):
    m = glob.glob(os.path.join(folder, '*.mat')); return m[0] if m else None
beh = []
for mouse in MICE:
    opt = set_options(mouse=mouse)
    for day in range(1, opt['n_days'] + 1):
        f = find_mat(f'{PATH}/{mouse}-DualTask-BehavioralData/day_{day}/')
        if f is None:
            continue
        try:
            m = loadmat(f)
        except Exception as e:                                   # one session file is unreadable (zlib error); the old control skipped it too
            print(f'  skip {mouse} day {day}: {type(e).__name__}'); continue
        onsets = m['Sample'][:, 0] / 1e3; licks = m['lickTime'][:, 0] / 1e3
        tl = [licks[(licks >= onsets[i]) & (licks < onsets[i + 1])] - onsets[i] for i in range(len(onsets) - 1)]
        tl.append(licks[licks >= onsets[-1]] - onsets[-1])
        rate = np.array([np.sum((r >= DELAY[0]) & (r < DELAY[1])) / (DELAY[1] - DELAY[0]) for r in tl])
        tr = m['AllTrials'][0][0][-1]
        df = pd.DataFrame(tr, columns=['sample', 'test', 'outcome', 'pair', 'distractor', 'cue', 'odr_outcome', 'odr_pair', 'laser'])
        df['tasks'] = df['distractor'].map({0: 'DPA', 1: 'DualGo', 2: 'DualNoGo'}); df['lick_delay'] = rate
        df['day'] = day; df['mouse'] = mouse; beh.append(df)
beh = pd.concat(beh, ignore_index=True)
# stage of each behavioural day = the stage the tensor assigns to that (mouse, day)
stage_of = yc.groupby(['mouse', 'day']).stage.first().to_dict()
beh['stage'] = [stage_of.get((m, float(dd))) for m, dd in zip(beh.mouse, beh.day)]
beh = beh[(beh.laser == 0) & (beh.tasks == 'DPA') & beh.stage.notna()].copy()
# sample class from the .mat 'pair' column (1-4 = odor_pair 0-3; pairs 1,2 = sample A, 3,4 = sample B, the
# paper's convention) — verified against the tensor's per-day A/B counts below
beh['sclass'] = np.where(beh['pair'].isin([1, 2]), 'A', 'B')
chk = []
for (m, dd), g in beh.groupby(['mouse', 'day']):
    t = d[(d.mouse == m) & (d.day == dd)]
    chk.append((g.sclass == 'A').sum() == (t['sample'] == 'A').sum())
print(f'sample-class alignment: {np.mean(chk):.0%} of mouse-days agree on the A/B split')
lick = beh.groupby(['mouse', 'sclass', 'stage']).lick_delay.mean().rename('lick').reset_index().rename(columns={'sclass': 'sample'})

# ── per mouse x sample x stage table ──
g = d.groupby(['mouse', 'sample', 'stage']).agg(depth=('depth', 'mean'), acc=('performance', 'mean')).reset_index()
g = g.merge(lick, on=['mouse', 'sample', 'stage'], how='left'); g['st'] = (g.stage == 'Expert').astype(int)
g = g.dropna(subset=['lick']).reset_index(drop=True); print(f'{len(g)} mouse x sample x stage cells with a lick rate')
m0 = smf.mixedlm('depth ~ st + C(sample)', g, groups=g['mouse']).fit()
m1 = smf.mixedlm('depth ~ st + C(sample) + lick', g, groups=g['mouse']).fit()
per = g.pivot_table(index='mouse', columns='st', values=['depth', 'acc', 'lick'], aggfunc='mean')
gd = pd.DataFrame({'dd': per['depth'][1] - per['depth'][0], 'da': per['acc'][1] - per['acc'][0], 'dl': per['lick'][1] - per['lick'][0]}).dropna()
r0, p0 = spearmanr(gd.dd, gd.da)
def rank_resid(u, z):
    ur, zr = pd.Series(u).rank().to_numpy(), pd.Series(z).rank().to_numpy()
    return ur - np.polyval(np.polyfit(zr, ur, 1), zr)
rp, pp = pearsonr(rank_resid(gd.dd, gd.dl), rank_resid(gd.da, gd.dl)); rl, pl = spearmanr(gd.dl, gd.da)
ms = g.groupby(['mouse', 'stage']).agg(depth=('depth', 'mean'), lick=('lick', 'mean')).reset_index()
rm, pm = spearmanr(ms.depth, ms.lick)
frac_lick = (beh.lick_delay > 0).mean()
print(f'late-delay licks on {frac_lick:.1%} of laser-off DPA trials; per-mouse mean rate naive {ms[ms.stage=="Naive"].lick.mean():.2f} Hz, expert {ms[ms.stage=="Expert"].lick.mean():.2f} Hz')
print(f'push  no covariate  β={m0.params["st"]:+.3f} ± {m0.bse["st"]:.3f} p={m0.pvalues["st"]:.3f}')
print(f'push  + lick        β={m1.params["st"]:+.3f} ± {m1.bse["st"]:.3f} p={m1.pvalues["st"]:.3f}   (lick β={m1.params["lick"]:+.3f} p={m1.pvalues["lick"]:.3f})')
print(f'coupling ρ={r0:+.2f} p={p0:.3f}; partial | Δlick r={rp:+.2f} p={pp:.3f}; Δlick vs Δacc ρ={rl:+.2f} p={pl:.2f}')
print(f'mouse x stage depth vs lick rate: ρ={rm:+.2f} p={pm:.2f} (n={len(ms)})')
C = 'figures/overlaps/controls/ed3_cache.pkl'
c = pickle.load(open(C, 'rb')) if os.path.exists(C) else {}
c['lick_ccgd'] = dict(g=g, ms=ms, gd=gd, frac_lick=float(frac_lick),
                      m0=(float(m0.params['st']), float(m0.bse['st']), float(m0.pvalues['st'])),
                      m1=(float(m1.params['st']), float(m1.bse['st']), float(m1.pvalues['st']), float(m1.params['lick']), float(m1.pvalues['lick'])),
                      r0=(float(r0), float(p0)), rp=(float(rp), float(pp)), rl=(float(rl), float(pl)), rm=(float(rm), float(pm)))
pickle.dump(c, open(C, 'wb')); print('cached ed3_cache[lick_ccgd]')
