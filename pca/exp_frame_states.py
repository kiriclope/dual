"""FRAME_STATES cache — the Fig-3 storyboard replayed from the SAME per-mouse CCGD projections
as panel A (user 2026-08-31: "panel B is not consistent with the traces in panel A").

The storyboard's freshly-fit single-window axes were the problem: they carry the trial's
condition-independent ramp (29-45% of the code), so NO single origin makes all five windows read
sensibly — baseline-zero dragged every window off-centre, boundary-zero made the mid-delay states
sit 3-5 z below the lick boundary while panel A's traces showed them AT baseline. The fix is the
same one panel A already uses (exp_traj_orig.py): replay `overlaps/main_panels`' validated
projections — SAMPLE_D (x) and LICK_D (y), cross-validated CCGD decision functions, per-mouse
baseline-zeroed, one shared per-mouse unit — so the storyboard and the traces are literally the
same coordinates and the dashed crosshair is A's baseline zero.

Per (stage, set, window): per-trial window means of SAMPLE_D / LICK_D over the SAME trials
(laser-off, correct, that stage), split by condition (task, sample, lick), reduced to PER-MOUSE
condition means (>=3 trials per mouse per cell; cells kept with >=3 mice).

Windows (overlaps T_WINDOW=0 bins): md = bins_MD 33-38 · delay = BINS_LATE 45-53 (post-cue) ·
decision = 57-62 incl. (the DPA action window).

Output: merges {'FRAME_STATES' (plain) / 'FRAME_STATES_pca20' (--pca)} into results.pkl:
FRAME_STATES[(stage, set, window)] = {(task, samp, lick): (mice, (n_mice, 2) [x, y] means)} —
mouse ids kept so the figure can re-centre each window per mouse (condition geometry only).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_frame_states.py [--pca]
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir('/home/leon/dual/overlaps')                       # main_panels uses relative data paths
sys.path.insert(0, '/home/leon/dual/overlaps')             # ...and is not a package module
import numpy as np
import main_panels as MP                                    # loads the ~1.9 GB overlaps tensor

TSUF = '_pca20' if '--pca' in sys.argv[1:] else ''

# SAMPLE_D rows and LICK_D rows are the SAME trial set but NOT row-aligned (each target's rows
# come out of the CV pipeline in its own order; verified: identical per-(mouse, stage, task)
# group counts, tasks shuffled within mouse blocks). There is no trial id to pair on — so the
# cache stores per-mouse CONDITION MEANS, selected independently in each table: a cell's trial
# set is identical on both sides, so the (mean x, mean y) pair is exact without row pairing.
YS, YL = MP.Y_SAM, MP.Lm
assert len(YS) == len(YL), f'{len(YS)} vs {len(YL)} rows'

WINS = {'md': np.asarray(MP.options['bins_MD']),           # 33-38 mid-delay (pre-cue)
        'delay': np.asarray(MP.BINS_LATE),                 # 45-53 late delay (post-cue/lick)
        'decision': np.arange(57, 63)}                     # 57-62 incl., DPA action window
B_SPECS = [('DPA', 'md'), ('DPA', 'decision'),
           ('dual', 'md'), ('dual', 'delay'), ('dual', 'decision')]

def cols(Y):
    return dict(mouse=Y.mouse.to_numpy(), tasks=Y.tasks.to_numpy(),
                stage=Y.learning.to_numpy(), las=Y.laser.to_numpy(),
                perf=Y.performance.to_numpy(), samp=Y.sample_odor.to_numpy(),
                lick=Y.choice.to_numpy())


CS, CL = cols(YS), cols(YL)


def cell(C, stage, tk, sv, lv, mo):
    return ((C['las'] == 0) & (C['perf'] == 1) & (C['stage'] == stage) & (C['tasks'] == tk)
            & (C['samp'] == sv) & (C['lick'] == lv) & (C['mouse'] == mo))


OUT = {}
for stage in MP.STAGES:
    for sname, wn in B_SPECS:
        bins = WINS[wn]
        x = np.nanmean(MP.SAMPLE_D[:, bins], 1); y = np.nanmean(MP.LICK_D[:, bins], 1)
        tset = ['DPA'] if sname == 'DPA' else ['DualGo', 'DualNoGo']
        ent = {}
        for tk in tset:
            for sv in (0, 1):
                for lv in (0, 1):
                    mice, pm = [], []
                    for mo in MP.ALL_MICE:
                        sx = cell(CS, stage, tk, sv, lv, mo) & np.isfinite(x)
                        sy = cell(CL, stage, tk, sv, lv, mo) & np.isfinite(y)
                        assert sx.sum() == sy.sum(), (stage, tk, sv, lv, mo)
                        if sx.sum() >= 3:
                            mice.append(mo); pm.append([x[sx].mean(), y[sy].mean()])
                    if len(pm) >= 3:
                        ent[(tk, sv, bool(lv))] = (mice, np.asarray(pm))
        OUT[(stage, sname, wn)] = ent
        gm = {cd: v.mean(0) for cd, (_, v) in ent.items()}
        ymn = np.array([g[1] for g in gm.values()])
        ylk = np.array([g[1] for cd, g in gm.items() if cd[2]])
        ynl = np.array([g[1] for cd, g in gm.items() if not cd[2]])
        xa = np.array([g[0] for cd, g in gm.items() if cd[1] == 0])
        xb = np.array([g[0] for cd, g in gm.items() if cd[1] == 1])
        print(f'{stage:6s} {sname:4s} {wn:9s} cells={len(ent)}  '
              f'y mean {ymn.mean():+5.2f} (lick {ylk.mean():+5.2f} / no-lick {ynl.mean():+5.2f})  '
              f'sample sep {xb.mean() - xa.mean():+5.2f}', flush=True)

RES = '/home/leon/dual/pca/figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['FRAME_STATES' + TSUF] = OUT
pickle.dump(d, open(RES, 'wb'))
print('merged FRAME_STATES' + TSUF, 'into', RES)
