"""exp_lick_control_trial.py — TRIAL-LEVEL late-delay lick control on the CCGD choice depth (ED 4d, 2026-09-15).

The mouse x sample x stage control (exp_lick_control_ccgd.py) could not align tensor rows to behavioural trials
because ccgd_validation returns rows in fold order. run_overlaps.py now writes a within-session `trial` index into
the labels (and seeds its folds), and the choice decoder was re-run under `--tag trial`:
    log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice_trial
Every held-out decision function therefore maps to one row of the session's behaviour .mat (AllTrials order = imaging
order: tasks and laser agree on 100 % of trials in all 47 readable sessions; .mat `sample` 1/2 = sample_odor 0/1,
.mat `pair` 1-4 = odor_pair 0, 2, 1, 3). Depth as in Fig. 4: decoders trained at the decision window 54-62, read at
late delay 45-53, RAW log-odds; laser-off DPA trials. Late-delay licks = lick times 5.5-7.0 s after the .mat `Sample`
stamp (= imaging 7.5-9.0 s = bins 45-53; the stamp is the sample-odor onset at imaging 2.0 s).

  1. trial-level association   per mouse x stage Spearman(depth, lick rate) -> Wilcoxon over the nine mice (stages
                               averaged); trial-level LMM depth ~ stage + sample + lick rate + (1 | mouse)
  2. push without licks        depth ~ stage + sample + (1 | mouse) on 36 mouse x sample x stage means computed
                               from NO-LICK trials only (vs all trials, vs lick trials)
  3. coupling without licks    per-mouse Spearman Δdepth (no-lick trials) vs ΔDPA accuracy (all DPA trials, as Fig. 4c)
Dumps ed3_cache['lick_trial'].
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_lick_control_trial.py
"""
import sys, os, glob, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.io import loadmat
from scipy.stats import spearmanr, wilcoxon
from src.pca.io import pkl_load
from src.common.options import set_options

PATH = '/storage/leon/dual_task/data/2Samples-DualTask-BehavioralData'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
TDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice_trial'
ACT, LD, DELAY = np.arange(54, 63), np.arange(45, 54), (5.5, 7.0)   # 5.5-7.0 s after the .mat 'Sample' stamp = imaging 7.5-9.0 s = bins_LD 45-53: the stamp IS the sample-odor onset (imaging 2.0 s; Test stamp = +7.0 s = imaging 9.0 s). FIXED 2026-09-16 — the first build used 6.0-7.5 (an assumed 1.5 s offset), which let the first 0.5 s of test-odor licks into the 'late-delay' window

# ── the CCGD depth, one row per held-out trial, with its session trial index ──
y = pkl_load(f'labels_{TDUM}', path='../data/overlaps'); X = np.asarray(pkl_load(f'X_{TDUM}', path='../data/overlaps'))
ch = (y.target == 'choice').to_numpy(); D = X[ch][:, 1, ACT, :].mean(1).astype(float); yc = y[ch].reset_index(drop=True); del X
yc['depth'] = D[:, LD].mean(1)
d = yc[(yc.laser == 0) & (yc.tasks == 'DPA')].copy()
d['sample'] = np.where(d.odor_pair.isin([0, 1]), 'A', 'B'); d['st'] = (d.stage == 'Expert').astype(int)
d['day'] = d.day.astype(int); d['trial'] = d.trial.astype(int)
print(f'{len(d)} laser-off DPA held-out trials, {d.mouse.nunique()} mice')

# ── behaviour: per-trial late-delay lick rate, keyed by (mouse, day, trial) ──
beh = []
for mouse in MICE:
    for day in range(1, set_options(mouse=mouse)['n_days'] + 1):
        f = glob.glob(f'{PATH}/{mouse}-DualTask-BehavioralData/day_{day}/*.mat')
        if not f:
            continue
        try:
            m = loadmat(f[0])
        except Exception as e:
            print(f'  skip {mouse} day {day}: {type(e).__name__}'); continue
        onsets = m['Sample'][:, 0] / 1e3; licks = m['lickTime'][:, 0] / 1e3
        tl = [licks[(licks >= onsets[i]) & (licks < onsets[i + 1])] - onsets[i] for i in range(len(onsets) - 1)]
        tl.append(licks[licks >= onsets[-1]] - onsets[-1])
        n = np.array([np.sum((r >= DELAY[0]) & (r < DELAY[1])) for r in tl])
        tr = m['AllTrials'][0][0][-1]
        beh.append(pd.DataFrame({'mouse': mouse, 'day': day, 'trial': np.arange(len(n)), 'nlick': n,
                                 'lick_rate': n / (DELAY[1] - DELAY[0]), 'mat_sample': tr[:, 0].astype(int) - 1}))
beh = pd.concat(beh, ignore_index=True)
d = d.merge(beh, on=['mouse', 'day', 'trial'], how='inner')
agree = (d.mat_sample == np.where(d['sample'] == 'A', 0, 1)).mean()
print(f'{len(d)} trials matched to behaviour; sample identity agrees on {agree:.1%} (must be 100 %)')
assert agree > 0.999
d['licked'] = (d.nlick > 0).astype(int)
print(f'late-delay licks on {d.licked.mean():.1%} of trials (naive {d[d.st == 0].licked.mean():.1%}, expert {d[d.st == 1].licked.mean():.1%})')

# ── 1. trial-level association ──
rows = []
for (m, st), g in d.groupby(['mouse', 'st']):
    if g.lick_rate.std() > 0:
        r, p = spearmanr(g.depth, g.lick_rate); rows.append(dict(mouse=m, st=st, rho=r, p=p, n=len(g), frac=g.licked.mean()))
RHO = pd.DataFrame(rows)
per = RHO.groupby('mouse').rho.mean()
wt = wilcoxon(per); print(f'trial-level Spearman(depth, lick rate) per mouse x stage: median ρ {RHO.rho.median():+.3f} '
                          f'({len(RHO)} cells); per mouse (stages averaged) median {per.median():+.3f}, Wilcoxon p = {wt.pvalue:.3f}, {(per > 0).sum()}/{len(per)} positive')
mt = smf.mixedlm('depth ~ st + C(sample) + lick_rate', d, groups=d['mouse']).fit()
mt0 = smf.mixedlm('depth ~ st + C(sample)', d, groups=d['mouse']).fit()
print(f'trial LMM  depth ~ st + sample            β_st={mt0.params["st"]:+.3f} ± {mt0.bse["st"]:.3f} p={mt0.pvalues["st"]:.3f}   ({len(d)} trials)')
print(f'trial LMM  depth ~ st + sample + lick     β_st={mt.params["st"]:+.3f} ± {mt.bse["st"]:.3f} p={mt.pvalues["st"]:.3f}   lick β={mt.params["lick_rate"]:+.3f} p={mt.pvalues["lick_rate"]:.3f}')

# ── 2. push on no-lick / lick trials (mouse x sample x stage means, 36 obs) ──
def push(sub, lab):
    g = sub.groupby(['mouse', 'sample', 'st']).depth.mean().reset_index()
    f = smf.mixedlm('depth ~ st + C(sample)', g, groups=g['mouse']).fit()
    print(f'push  {lab:22s} β={f.params["st"]:+.3f} ± {f.bse["st"]:.3f} p={f.pvalues["st"]:.3f}   ({len(g)} obs)')
    return g, (float(f.params['st']), float(f.bse['st']), float(f.pvalues['st']), int(len(g)))
g_all, p_all = push(d, 'all trials')
g_nl, p_nl = push(d[d.licked == 0], 'no-lick trials')
g_lk, p_lk = push(d[d.licked == 1], 'lick trials')

# ── 3. coupling on no-lick trials ──
def dd_of(g):
    pv = g.pivot_table(index='mouse', columns='st', values='depth', aggfunc='mean'); return pv[1] - pv[0]
acc = d.groupby(['mouse', 'st']).performance.mean().unstack(); da = acc[1] - acc[0]
gd = pd.DataFrame({'dd_all': dd_of(g_all), 'dd_nolick': dd_of(g_nl), 'da': da}).dropna()
r_all, p_rall = spearmanr(gd.dd_all, gd.da); r_nl, p_rnl = spearmanr(gd.dd_nolick, gd.da)
print(f'coupling Δdepth vs ΔDPA acc: all trials ρ={r_all:+.2f} p={p_rall:.3f}; no-lick trials ρ={r_nl:+.2f} p={p_rnl:.3f}  ({len(gd)} mice)')

C = 'figures/overlaps/controls/ed3_cache.pkl'
c = pickle.load(open(C, 'rb')) if os.path.exists(C) else {}
c['lick_trial'] = dict(rho=RHO, per=per, wilcoxon_p=float(wt.pvalue), n_trials=int(len(d)), frac_lick=float(d.licked.mean()),
                       frac_lick_stage=(float(d[d.st == 0].licked.mean()), float(d[d.st == 1].licked.mean())),
                       lmm0=(float(mt0.params['st']), float(mt0.bse['st']), float(mt0.pvalues['st'])),
                       lmm_lick=(float(mt.params['st']), float(mt.bse['st']), float(mt.pvalues['st']), float(mt.params['lick_rate']), float(mt.pvalues['lick_rate'])),
                       push_all=p_all, push_nolick=p_nl, push_lick=p_lk, gd=gd,
                       r_all=(float(r_all), float(p_rall)), r_nolick=(float(r_nl), float(p_rnl)))
pickle.dump(c, open(C, 'wb')); print('cached ed3_cache[lick_trial]')
