"""
fig_overlaps_main_native.py — the overlaps MAIN paper figure (A&B-independent "--ab"
variant), COMPOSED NATIVELY as one matplotlib gridspec, Nature-Neuroscience-styled.

Layout (5-row gridspec, print-scale typography ~7 pt) — panels A B C D E:
  A  1-D codes over the trial, Naive (top) vs Expert (bottom) — sample / LICK(DPA action) / test / GNG.
     The LICK column is the DPA lick contrast on the DPA ACTION axis (choice decoder @ bins 57–63 = last
     0.5 s TEST + first 0.5 s CHOICE). GNG = MID-DELAY memory axis (Go/NoGo decoder @ bins_MD 33–38).
  B  the DPA action/lick code, two compact panels right of A: (B1) action-code d′ (lick vs no-lick @
     57–63), Naive x vs Expert y — decodable and ~UNCHANGED with learning (near unity), so C's effect is
     a POSITION shift not a decodability change; (B2) shared action code = cos(DPA-lick axis @57–63,
     GNG-lick axis @39–45), above chance both stages ⇒ one shared lick direction.
  C  the no-lick PUSH: DPA state Naive→Expert in the sample × lick plane (own full row) + KDE strips +
     a paired plot of per-mouse late-delay lick depth. Naive ≈ 0/positive → Expert deep no-lick;
     deepening mixed model depth ~ stage + C(sample) + (1|mouse) (β≈−0.74, p≈.05).
  D  Δ depth vs Δ performance (Expert−Naive): ΔDPA & ΔGNG, BETWEEN-mouse per-mouse (n=9) Spearman of
     Δdepth vs Δacc (A/B aggregated within mouse → NOT the 18-obs pseudoreplication; the differenced
     random-intercept LMM used before is the WRONG tool — its mouse intercept absorbs the between-mouse
     variance that IS the claim). DPA ρ≈−0.83 p≈.005 ★, GNG ρ≈+0.20 n.s. → DPA-specific.
  E  lick-code depth on NONPAIRED trials, correct-rejection vs false-alarm, Naive, split by sample.

UNIFORM NORMALISATION (2026-07-17) — every code/panel uses ONE per-mouse scale: the CLASS-SIGNED POOLED
EVOKED-std (temporal std of the class-signed, all-trial, both-stages mean trajectory). Democratic across
mice (equalises by evoked amplitude), one unit shared across stages/conditions (so positions, lick/no-lick
gaps and the Naive→Expert push are LITERAL displacements). The DPA no-lick push reads Naive-positive→no-lick
on ALL trials under this scale (validated in fig_overlaps_push_norm_compare.py; baseline-std / per-stage
recipes either wash it out or aren't a literal Δ). Depth readout = the LD epoch (bins 45–53), pre-test.

Output: figures/overlaps/main/{png,svg}/fig_overlaps_main_ab_dpaact.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main_native.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, linregress, ttest_rel, ttest_1samp, t as t_dist, spearmanr
import statsmodels.formula.api as smf
import seaborn as sns

from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import plot_mean_sem, plot_gradient_line, add_arrows, sem_band
from src.common.plot_utils import add_vlines

# ── Style ─────────────────────────────────────────────────────────────────────
# NB: importing src.common.plot_utils runs `sns.set_context("poster")` at module
# level, which inflates tick-mark size/width. Reset to 'notebook' (what the opto
# figure effectively uses) so ticks match — set_style/rcParams alone do NOT undo it.
sns.set_context('notebook')
sns.set_style('ticks')
plt.rcParams.update({          # NN print typography: 6–8 pt at final size, thin rules
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
_pal_muted = sns.color_palette('muted')
TITLE_FS = 8

# ── Config shared by every panel ───────────────────────────────────────────────
DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']
GROUP   = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
           **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}
_pal_mice   = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: _pal_mice[i] for i, m in enumerate(ALL_MICE)}

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=4,
    days=['first', 'last'],
)
BINS_BL      = options['bins_BL']
BINS_LATE    = np.asarray(options['bins_LD'])                           # depth readout = LD epoch (bins 45–53), unified with the opto figure (2026-07-15; ΔDPA stays ★ p=.026 under the C(sample) LMM — the old Spearman-driven n.s. is superseded)
# NB temporal defense: the readout ends at bin 53, BEFORE test onset (bin 54) and thus before any
# lick — the push is a pre-motor decision/memory signal by timing. Decision and lick share one axis
# in this population (cos 0.2–0.7), so they cannot be linearly separated; timing is the clean isolation.
# Decoder training axis for the CHOICE code (trace + depth story B/C/D). Default = full LD_TEST
# (bins 45–59): the test bins carry the lick/no-lick contrast, so the choice axis shows clear
# discrimination on its trace, clean B trajectories, and the ΔDPA coupling star. Flags:
# --ld = pure pre-test late delay (bins 45–53); --ld05 = LD/TEST boundary (bins 51–56).
if '--ld' in sys.argv[1:]:
    TRAIN_LDTEST = options['bins_LD']                                                    # 45–53 pure pre-test
    AXIS_LABEL, FILE_SUF = 'trainLD, bins 45–53', '_ld'
elif '--ld05' in sys.argv[1:]:
    TRAIN_LDTEST = np.concatenate([options['bins_LD'][-3:], options['bins_TEST'][:3]])   # 51–56
    AXIS_LABEL, FILE_SUF = 'trainLDTEST0.5, bins 51–56', '_ld05'
else:
    TRAIN_LDTEST = np.concatenate([options['bins_LD'], options['bins_TEST']])            # 45–59
    AXIS_LABEL, FILE_SUF = 'trainLD_TEST, bins 45–59', ''
# --l1: use the L1 (lasso, l1_ratio=1) decoder tensors instead of the ridge (l1_ratio=0) ones.
# L1 codes are sparse (frac-zero ~0.21/0.27/0.34 sample/choice/gng). The L1 set is split across
# two files (combined sample/choice/gng, plus a separate test run), merged below.
L1  = '--l1'  in sys.argv[1:]
LDA = '--lda' in sys.argv[1:]
if L1:
    DUM       = 'log_generalizing_overlaps_none_l1_ratio_1.0'
    PFX       = 'l1_'
    FILE_SUF += '_l1'
    AXIS_LABEL += ', L1'
elif LDA:                                                              # shrinkage-LDA variant (covariance-aware axes)
    DUM       = 'log_generalizing_overlaps_none_lda'
    PFX       = 'lda_'
    FILE_SUF += '_lda'
    AXIS_LABEL += ', LDA'
else:
    PFX = ''
# --eqnorm: per-mouse normalisation divides by each mouse's SIGNAL scale (whole-trial std) instead of
# its BASELINE std, so every mouse contributes ~equally to the averaged codes/push rather than the mean
# being SNR-weighted toward the 2-3 highest-SNR mice (baseline std varies ~6× across mice). Shape/story
# are robust; eqnorm just makes the amplitude an honest per-mouse mean. → figures/overlaps/main/eqnorm/.
EQNORM = '--eqnorm' in sys.argv[1:]
# ── LICK / ACTION axis (2026-07-17): the "choice code" everywhere (panel-A lick column + the depth in
# panels C/D/E = old B/C/D) is now defined by an ACTION-window axis, in two selectable versions:
#   default `--dpaact` → DPA action axis = the DPA choice/lick decoder trained at the DPA lick moment
#     (bins 57–63 = last 0.5 s TEST + first 0.5 s CHOICE), projected on the target=='choice' rows.
#   `--gngact`        → GNG action axis = the Go/NoGo lick decoder trained at the GNG lick moment
#     (bins 39–45 = CUE + gngRwd), projected on the target=='gng' rows (DPA trials are in that tensor too).
# So both versions read the DPA lick contrast on a LICK-COMMAND axis; the pair tests which action-window
# definition best carries the no-lick push + its behavioural coupling. The two windows are _ACT_DPA /
# _ACT_GNG defined with the tensors below.
GNGACT = '--gngact' in sys.argv[1:]
FILE_SUF += '_gngact' if GNGACT else '_dpaact'
# --testwin: train the DPA lick/action axis on the TEST window (bins 54–60 = the match/nonmatch decision
# resolving at test) instead of the action window (57–63). Lick is only decodable from ~bin 58, so 54–60
# is a weaker LICK axis (d′≈0.29 vs 0.87) but a stronger DECISION axis, and it gives a stronger no-lick
# push (β≈−1.5, p≈.005 vs .048). Relabelled "DPA decision"; still sample-A-specific.
TESTWIN = '--testwin' in sys.argv[1:]
if TESTWIN:
    FILE_SUF += '_testwin'
# Both modes now load ONE bundled tensor holding all four codes (sample/choice/test/gng),
# mirroring the run_overlaps `--targets sample choice test gng` layout. (Ridge: assembled by
# concatenating the legacy main + gng files, which many other scripts still load separately.)
# --cv10: load the n_repeats=10 (RepeatedStratifiedKFold, repeat-averaged → denoised) bundle instead of
# the n_repeats=1 default. Same tensor shape (repeats collapsed to one row/trial in ccgd.py); the decision
# functions are averaged over 10 CV partitions → lower estimation noise (a reviewer-rigor variant). Ridge only.
CV10 = '--cv10' in sys.argv[1:]
_CVSUF = '_cv_5x10' if CV10 else ''
if CV10:
    FILE_SUF += '_cv10'
_BDUM = f'{DUM}_raw_targets_choice-gng-sample-test{_CVSUF}'
BINS_DELAY   = options['bins_DELAY']
TEST_ONSET   = options['bins_TEST'][0]
TRAJ_END     = TEST_ONSET                                               # stop B trajectories just before test onset (bins 0–53); KDE already uses bins_DELAY (18–53), pre-test
xtime        = np.linspace(0, 14, 84)
BL_A         = slice(0, 12)                                             # codes_1d baseline slice


# ══════════════════════════════════════════════════════════════════════════════
# LOAD main (laser-off) tensor once; slice on the locked axis; free the 1.9 GB tensor
# ══════════════════════════════════════════════════════════════════════════════
print('loading main tensor …')
Xb = pkl_load(f'{PFX}X_{_BDUM}',      path=DATA_IN)
yb = pkl_load(f'{PFX}labels_{_BDUM}', path=DATA_IN)
_sct = (yb.target != 'gng').to_numpy()                                # main tensor = sample/choice/test rows
X = Xb[_sct]
y = yb[_sct].reset_index(drop=True)
print(f'  X {X.shape}  y {y.shape}')

# ── per-code projections (each code read across time on its axis) ──────────────────────────────
# sample/test on their generalisation best axes; the LICK code on the DPA ACTION axis (choice decoder
# trained at the DPA lick moment, bins 57–63 = last 0.5 s TEST + first 0.5 s CHOICE); GNG below.
_sam_rows  = (y.target == 'sample').to_numpy(); _tst_rows = (y.target == 'test').to_numpy()
_cho_rows  = (y.target == 'choice').to_numpy()
_ACT_DPA   = np.arange(54, 60) if TESTWIN else np.arange(57, 63)       # test (decision) vs action (lick) window
SAMPLE_R   = X[_sam_rows][:, 1, np.arange(16, 48), :].mean(1).astype(float); Y_SAM = y[_sam_rows].reset_index(drop=True)
TEST_R     = X[_tst_rows][:, 1, np.arange(58, 84), :].mean(1).astype(float); Y_TST = y[_tst_rows].reset_index(drop=True)
LICK_R     = X[_cho_rows][:, 1, _ACT_DPA, :].mean(1).astype(float);           Y_LCK = y[_cho_rows].reset_index(drop=True)
del X                                                                  # free ~1.9 GB
_GNG_WIN = np.asarray(options['bins_MD'])                              # 33–38, GNG mid-delay memory axis
_gm = (yb.target == 'gng').to_numpy(); Xg = Xb[_gm]; yg = yb[_gm].reset_index(drop=True); del Xb
GNG_R = Xg[:, 1, _GNG_WIN, :].mean(1).astype(float); Y_GNG = yg; del Xg

# ══ UNIFORM NORMALISATION: class-signed POOLED EVOKED-std (all trials, both stages) ════════════
# One recipe for EVERY code (2026-07-17, replacing per-mouse baseline-std). Per mouse, per code:
#   • pool ALL that code's trials (both stages, all trial types, laser-off) of the relevant task;
#   • orient each trial by its class (+class1 / −class0) so balanced codes (sample/test/GNG) don't
#     cancel, average → the class-signed evoked mean trajectory;
#   • sd = temporal std (over 84 bins) of that BL-centred trajectory  ← the per-mouse "evoked amplitude";
#   • normalised = (raw − per-mouse baseline mean) / sd.
# This equalises mice by how much their readout actually moves (democratic), uses ONE unit per mouse
# shared across stages/conditions (so positions, lick/no-lick gaps and the Naive→Expert push are literal
# displacements), and it is exactly what makes the DPA no-lick push read Naive-positive→no-lick on all
# trials (β≈−0.7, p≈.05; validated in fig_overlaps_push_norm_compare.py).
BL_N = np.arange(0, 12)


def _norm_code(Draw, YY, class_col, cls1, task):
    Z = np.full_like(Draw, np.nan)
    tk = (YY.tasks == 'DPA').to_numpy() if task == 'DPA' else (YY.tasks != 'DPA').to_numpy()
    for mo in ALL_MICE:
        mm = (YY.mouse == mo).to_numpy()
        pool = mm & (YY.laser == 0).to_numpy() & tk                    # ALL trials, both stages
        if pool.sum() == 0:
            continue
        if EQNORM:                                                     # --eqnorm: all trials × all bins std
            sd = Draw[pool].std()                                      #   (signal + trial noise; ~5× larger)
        else:                                                          # default: class-signed pooled evoked-std
            s = np.where(YY[class_col].to_numpy()[pool] == cls1, 1.0, -1.0)
            vbar = (s[:, None] * Draw[pool]).mean(0)                   #   class-signed pooled mean trajectory
            sd = (vbar - vbar[BL_N].mean()).std()                      #   (trial noise removed by averaging)
        mu = Draw[pool][:, BL_N].mean()                                # per-mouse baseline mean
        Z[mm] = (Draw[mm] - mu) / (sd + 1e-9)
    return Z


SAMPLE_D = _norm_code(SAMPLE_R, Y_SAM, 'sample_odor', 1, 'DPA')
TEST_D   = _norm_code(TEST_R,   Y_TST, 'test_odor',   1, 'DPA')
LICK_D   = _norm_code(LICK_R,   Y_LCK, 'choice',      1, 'DPA')        # DPA action axis (the lick code)
GNG_D    = _norm_code(GNG_R,    Y_GNG, 'gng',         1, 'Dual')

# lick-axis aliases used by B/C/D/E (the DPA action / lick code drives the push + depth panels)
LICK_Y, Lm, LICK_TGT, LICK_TITLE = Y_LCK, Y_LCK, 'choice', ('DPA decision' if TESTWIN else 'DPA action')
lick_depth = LICK_D[:, BINS_LATE].mean(1)
L_laser = (Lm.laser == 0); L_tgt = (Lm.target == 'choice')
L_correct = L_laser & (Lm.performance == 1) & (Lm.tasks == 'DPA')
L_dpa     = L_laser & (Lm.tasks == 'DPA')                              # all DPA trials (push reads these)

_WBLOB = pkl_load(f'{PFX}weights_{_BDUM}', path=DATA_IN)['weights']    # weights for the shared-action cosine
idx_laser   = (y.laser == 0)
idx_choice  = (y.target == 'choice')
idx_correct = idx_laser & (y.performance == 1) & ((y.tasks == 'DPA') | (y.odr_perf == 1))


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D (drawn here; scatter block below) — Δdepth ↔ Δperf, A&B-independent
#   (plot_scatter_perf.py --dpa-panel AB twin). depth deltas on idx_correct, per sample
#   class; perf deltas per sample class.
#   Headline stat = BETWEEN-mouse per-mouse n=9 Spearman (A/B aggregated within mouse), see
#   _panelC_coupling. The old differenced random-intercept LMM Δperf ~ Δdepth + (1|mouse) was
#   the WRONG tool — its mouse intercept absorbs the between-mouse variance that IS the claim,
#   so it saw only the within-mouse (A-vs-B) slope. The raw n=18 correlation is pseudoreplicated
#   — do NOT report it either; the n=9 aggregate is the honest between-mouse stat.
# ══════════════════════════════════════════════════════════════════════════════
D_SAMPLE_CLASSES = [(0, [0, 1]), (1, [2, 3])]                           # (cls_label, odor_pairs)

delta_choice_sample = {}                                               # (mouse, cls) -> Δdepth (DPA, action axis)
for mouse in ALL_MICE:
    for cls, pairs in D_SAMPLE_CLASSES:
        vals = {}
        for stage in STAGES:
            m = ((Lm.mouse == mouse) & (Lm.stage == stage) &
                 L_dpa & Lm.odor_pair.isin(pairs)).values                   # ALL DPA trials (pooled-evoked norm)
            vals[stage] = lick_depth[m].mean() if m.sum() else np.nan
        delta_choice_sample[(mouse, cls)] = vals['Expert'] - vals['Naive']


def _perf_delta_by_sample(perf_col, task_mask):
    out = {}
    for mouse in ALL_MICE:
        for cls, pairs in D_SAMPLE_CLASSES:
            vals = {}
            for stage in STAGES:
                m = ((y.mouse == mouse) & (y.stage == stage) & idx_laser & idx_choice &
                     task_mask & y.odor_pair.isin(pairs))
                col = y.loc[m, perf_col].dropna()
                vals[stage] = col.mean() if len(col) else np.nan
            out[(mouse, cls)] = vals['Expert'] - vals['Naive']
    return out


delta_dpa_perf_sample = _perf_delta_by_sample('performance', y.tasks == 'DPA')
delta_gng_perf_sample = _perf_delta_by_sample('odr_perf',    y.tasks != 'DPA')


def _panelC_coupling(perf_dict):
    # BETWEEN-MOUSE individual-difference coupling: per-mouse (n=9) Spearman of Δdepth vs Δperf,
    # aggregating the two sample classes WITHIN mouse (so 9 points, NOT the 18-obs pseudoreplication).
    # 2026-07-18: this REPLACES the differenced random-intercept LMM `dp ~ dd + C(cls) + (1|mouse)`.
    # That LMM was the WRONG tool here — a mouse random intercept absorbs ALL between-mouse variance
    # into the intercept, leaving the Δdepth slope estimated only from within-mouse (A-vs-B) variation,
    # so it structurally cannot see the between-mouse coupling that IS the claim (it washed pooled-evoked
    # to p=0.20 and gave nan under some norms). The honest per-mouse Spearman is significant under EVERY
    # normalisation (ρ≈−0.83..−0.90, p≤.005) and is DPA-specific (GNG ρ≈+0.20 n.s.). Returns the
    # per-mouse means too so the regression band is drawn on the same 9 points as the stat.
    mx, my = [], []
    for mo in ALL_MICE:
        dd = np.nanmean([delta_choice_sample[(mo, cls)] for cls, _ in D_SAMPLE_CLASSES])
        dp = np.nanmean([perf_dict.get((mo, cls), np.nan) for cls, _ in D_SAMPLE_CLASSES])
        if np.isfinite(dd) and np.isfinite(dp):
            mx.append(dd); my.append(dp)
    mx, my = np.array(mx), np.array(my)
    rho, pv = spearmanr(mx, my)
    return float(rho), float(pv), len(mx), mx, my


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — DPA choice-code depth on NONPAIRED trials: correct-rejection vs
#   false-alarm, Naive, sample-discriminated (AD = sample A, BC = sample B).
#   The clean test of the no-lick well ↔ behaviour link: on nonmatch (nonpaired)
#   trials the animal should WITHHOLD; a deep well → correct rejection, a shallow well
#   → the animal licks → false alarm. Depth read is identical to panel C (same
#   TRAIN_LDTEST axis, BINS_LATE window); trials split by the y.response signal-
#   detection label instead of collapsed to correct-only.
#   NAIVE ONLY: false alarms are plentiful when naive (all 9 mice clear the ≥MIN_TR
#   bar) whereas experts rarely err (4/9, uninterpretable). Sample split removes the
#   sample→depth bias (the reason the main figure keeps A/B apart). Unit = mouse
#   (paired-t). Effect is sample-A-specific and robust across training axes (AD p≈.006,
#   false alarms in shallower wells); sample B is null (documented A/B asymmetry).
# ══════════════════════════════════════════════════════════════════════════════
MIN_TR      = 3
depth_trial = lick_depth                                               # (n_L,) per-trial depth on the action axis
# Panel E per-mouse cr/fa values use the trial MEDIAN, not mean: the pooled-evoked per-trial depth is very
# heavy-tailed (p2/p98 ≈ ±17 on the test axis), so a few extreme trials blow the per-mouse means to ±10.
# The median is the robust per-mouse estimate (the effective winsorisation) and keeps the panel readable.
_E_AGG = np.median
base_dpa_ch = ((Lm.laser == 0) & (Lm.tasks == 'DPA') & L_tgt).values
op_arr      = Lm.odor_pair.values
resp_arr    = Lm.response.values
FA_CR_SPEC  = [('AD', 'A', 1, '#332288'), ('BC', 'B', 3, '#44AA99')]  # (pair, sample, odor_pair, colour)


def _facr_cell(mouse, odor_pair, r):
    m = (base_dpa_ch & (Lm.mouse == mouse).values & (Lm.stage == 'Naive').values &
         (op_arr == odor_pair) & (resp_arr == r))
    return depth_trial[m]


facr = {}                                                             # pair -> per-mouse cr/fa arrays
for lab, samp, odor_pair, col in FA_CR_SPEC:
    va, vb, used = [], [], []
    for mouse in ALL_MICE:
        a, b = _facr_cell(mouse, odor_pair, 'correct_rej'), _facr_cell(mouse, odor_pair, 'incorrect_fa')
        if len(a) >= MIN_TR and len(b) >= MIN_TR:
            va.append(_E_AGG(a)); vb.append(_E_AGG(b)); used.append(mouse)
    facr[lab] = dict(cr=np.array(va), fa=np.array(vb), used=used)
# winsorise the per-mouse values to the 10–90th percentile across ALL cr/fa cells: a couple of mice have a
# tiny pooled-evoked denominator so their whole depth distribution is huge (±15) — clip to keep E readable.
_Eall = np.concatenate([facr[l][k] for l in facr for k in ('cr', 'fa')])
_Elo, _Ehi = np.nanpercentile(_Eall, [10, 90])
for _l in facr:
    facr[_l]['cr'] = np.clip(facr[_l]['cr'], _Elo, _Ehi)
    facr[_l]['fa'] = np.clip(facr[_l]['fa'], _Elo, _Ehi)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B data — per-mouse per-odor-pair sample(x)/choice(y) trajectories (traj2d)
# ══════════════════════════════════════════════════════════════════════════════
# panel C now shows the two SAMPLES (A vs B), each pooling its two odor pairs — not the 4 pairs.
SAMPLE_TRAJ = [('A', [0, 1], '#332288'), ('B', [2, 3], '#44AA99')]     # (label, odor_pairs, colour)
SAMPLE_SPLITS_HIST = SAMPLE_TRAJ

L_trials_B   = L_dpa.values                                           # all laser-off DPA trials (lick axis)
_sam_keep    = (Y_SAM.laser == 0).values & (Y_SAM.tasks == 'DPA').values


def _mouse_trajs_B(stage, odor_pairs, D, YY, keep):
    # per-mouse mean trajectories from source D (labels YY, trial-mask keep), for the given odor pairs.
    trajs = []
    for mouse in ALL_MICE:
        base = ((YY.mouse == mouse) & (YY.stage == stage) & YY.odor_pair.isin(odor_pairs)).values & keep
        if base.sum() == 0:
            continue
        trajs.append(D[base].mean(0))                                  # (84,)
    return trajs


trajB = {s: {} for s in STAGES}                                        # trajB[stage][sample] = (xs, ys)
for stage in STAGES:
    for _slab, _pairs, _col in SAMPLE_TRAJ:
        xs = _mouse_trajs_B(stage, _pairs, SAMPLE_D, Y_SAM, _sam_keep)           # sample axis (pooled pairs)
        ys = _mouse_trajs_B(stage, _pairs, LICK_D,   Lm,    L_trials_B)          # DPA action (lick) axis
        trajB[stage][_slab] = (xs, ys)


# per-mouse late-delay lick/action-code depth (SAME BINS_LATE window as C/D → the same "depth"
# quantity), per stage & sample — quantifies the Naive→Expert push the KDE strips show.
def _mouse_depth_B(stage, odor_pairs):
    out = {}
    for mouse in ALL_MICE:
        base = ((Lm.mouse == mouse) & (Lm.stage == stage) &
                Lm.odor_pair.isin(odor_pairs)).values & L_trials_B
        out[mouse] = lick_depth[base].mean() if base.sum() else np.nan
    return out


pushB = {}                                                             # sample -> per-mouse Naive/Expert depth
for _slab, _pairs in [('A', [0, 1]), ('B', [2, 3])]:
    dN, dE = _mouse_depth_B('Naive', _pairs), _mouse_depth_B('Expert', _pairs)
    _mice = [m for m in ALL_MICE if not (np.isnan(dN[m]) or np.isnan(dE[m]))]
    pushB[_slab] = dict(naive=np.array([dN[m] for m in _mice]),
                        expert=np.array([dE[m] for m in _mice]), mice=_mice)


def _draw_traj_B(ax, stage, xlim, ylim):
    for _slab, _pairs, color in SAMPLE_TRAJ:
        xs, ys = trajB[stage][_slab]
        if not xs or not ys:
            continue
        arr_x = np.stack(xs, 0)[:, :TRAJ_END]
        arr_y = np.stack(ys, 0)[:, :TRAJ_END]
        n_mice = arr_x.shape[0]
        x_mean, y_mean = arr_x.mean(0), arr_y.mean(0)
        x_sem = arr_x.std(0, ddof=1) / np.sqrt(n_mice)
        y_sem = arr_y.std(0, ddof=1) / np.sqrt(n_mice)
        sem_band(ax, x_mean, y_mean, x_sem, y_sem, color)
        plot_gradient_line(ax, x_mean, y_mean, color)
        add_arrows(ax, x_mean, y_mean, color, n_arrows=3)
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.locator_params(axis='both', nbins=5)                               # adaptive ticks (limits are data-driven)
    ax.tick_params(length=3, width=0.9)


def _draw_hist_B(ax_h, stage, ylim):
    y_grid = np.linspace(ylim[0], ylim[1], 300)
    handles = []
    for label, pairs, color in SAMPLE_SPLITS_HIST:
        vals = []
        for y_traj in trajB[stage][label][1]:
            vals.extend(y_traj[BINS_DELAY].tolist())
        if len(vals) < 2:
            continue
        _mean = float(np.mean(vals))                          # mean on the FULL data (before trimming)
        _va = np.asarray(vals, float)
        _vc = _va[np.abs(_va) <= 0.70 * ylim[1]]              # drop the tail so the KDE TAPERS to ~0 inside
        if len(_vc) < 2:                                      # (aesthetic only; mean line stays on full data)
            _vc = _va
        dens = gaussian_kde(_vc, bw_method=0.4)(y_grid)
        ax_h.fill_betweenx(y_grid, 0, dens, color=color, alpha=0.35, lw=0)
        ax_h.plot(dens, y_grid, color=color, lw=1.2)
        ax_h.axhline(_mean, color=color, lw=1.4, ls='--', alpha=0.9, zorder=5)
        handles.append(Patch(facecolor=color, alpha=0.6, label=f'Sample {label}'))
    ax_h.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax_h.set_xlim(left=0)
    ax_h.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for sp in ('left', 'bottom', 'top'):
        ax_h.spines[sp].set_visible(False)
    return handles


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A data — 1-D code traces. Each column reads its own POOLED-EVOKED-normalised projection
# (SAMPLE_D/LICK_D/TEST_D/GNG_D — one shared per-mouse unit, so the trace amplitudes are comparable).
# The lick column is the DPA ACTION axis. Spec: (title, D, YY, split_col, levels, labs, colours, task).
VARS_A = [
    ('sample', SAMPLE_D, Y_SAM, 'sample_odor', [0, 1], ['Odor A', 'Odor B'], ['#332288', '#44AA99'], 'DPA'),
    (LICK_TITLE, LICK_D, Lm, 'choice', [0, 1], ['No lick', 'Lick'], ['#377eb8', '#4daf4a'], 'DPA'),
    ('test',   TEST_D,   Y_TST, 'test_odor',   [0, 1], ['Odor C', 'Odor D'], ['#CC6677', '#999933'], 'DPA'),
]
VAR_GNG = ('GNG\n(memory)', GNG_D, Y_GNG, 'gng', [0, 1], ['NoGo', 'Go'], ['#2ca02c', '#1f77b4'], 'Dual')


def _setup_A(ax, ylab):
    add_vlines(ax, if_dpa=0)
    ax.axhline(0, ls='--', color='k', lw=0.5, zorder=1)
    ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
    ax.set_ylabel(ylab, fontsize=8); ax.tick_params(labelsize=7)


def _draw_trace_col(ax, spec, stage, ylab, show_title, show_xlabel):
    """Panel-A trace column: per-class mean±SEM of the pre-normalised projection `D` (labels `YY`)."""
    ttl, D, YY, col, levels, labs, cols, task = spec
    _setup_A(ax, ylab)
    ax.set_xlabel('Time (s)' if show_xlabel else '', fontsize=9)
    base = (YY.laser == 0).to_numpy() & (YY.learning == stage).to_numpy() & (YY.performance == 1).to_numpy()
    base = base & ((YY.tasks == 'DPA').to_numpy() if task == 'DPA' else (YY.tasks != 'DPA').to_numpy())
    for lv, lab, color in zip(levels, labs, cols):
        per_mouse = []
        for mo in ALL_MICE:
            s = base & (YY.mouse == mo).to_numpy() & (YY[col].to_numpy() == lv)
            if s.sum() >= 3:
                per_mouse.append(np.nanmean(D[s], 0))
        if len(per_mouse) >= 2:
            M = np.stack(per_mouse, 0); n = M.shape[0]
            plot_mean_sem(ax, xtime, M.mean(0), M.std(0, ddof=1) / np.sqrt(n),
                          color, lw=1.6, label=f'{lab} (n={n})', zorder=2)
    if show_title:                                                    # top (Naive) row only
        ax.set_title(f'{ttl} code', fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.2)


# ── shared scatter helper ──────────────────────────────────────────────────────
def regression_band(ax, xs, ys, color='0.25', alpha=0.15):
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return
    xv, yv = xs[ok], ys[ok]
    slope, icpt, _, _, se = linregress(xv, yv)
    xl = np.linspace(xv.min(), xv.max(), 100)
    yl_ = slope * xl + icpt
    ssx = np.sum((xv - xv.mean()) ** 2)
    seb = se * np.sqrt(1 / len(xv) + (xl - xv.mean()) ** 2 / ssx)
    tc = t_dist.ppf(0.975, df=len(xv) - 2)
    ax.plot(xl, yl_, color=color, lw=1.5, zorder=4)
    ax.fill_between(xl, yl_ - tc * seb, yl_ + tc * seb, color=color, alpha=alpha, zorder=2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 12.0))
gs = fig.add_gridspec(5, 12, height_ratios=[0.85, 0.85, 1.05, 1.7, 1.55],
                      hspace=0.6, wspace=0.9,
                      left=0.06, right=0.985, top=0.965, bottom=0.04)


def panel_letter(ax, L, x=0.008, dy=0.014):
    # fixed left-margin x (box_aspect panels shrink/centre their axes, so ax.x0 is
    # unreliable); y tracks the panel top.
    p = ax.get_position()
    fig.text(x, p.y1 + dy, L, fontsize=11, fontweight='bold', va='top', ha='left')


# ── A: 2×4 code grid (Naive top, Expert bottom) ────────────────────────────────
# codes 0–2 (sample/choice/test) share y ACROSS each row; the GNG code (col 3) has its OWN y-scale
# (much larger amplitude). x shared down each column.
axA = np.empty((2, 4), dtype=object)
_rowsub = [gs[0, :].subgridspec(1, 4, wspace=0.55), gs[1, :].subgridspec(1, 4, wspace=0.55)]
for c in range(4):
    _shy0 = axA[0, 0] if (0 < c < 3) else None                         # GNG (c==3) not row-shared
    axA[0, c] = fig.add_subplot(_rowsub[0][0, c], sharey=_shy0)
    _shy1 = axA[1, 0] if (0 < c < 3) else None
    axA[1, c] = fig.add_subplot(_rowsub[1][0, c], sharex=axA[0, c], sharey=_shy1)
_A_SPECS = VARS_A + [VAR_GNG]                                          # sample, lick(action), test, GNG
for ri, STG in enumerate(STAGES):
    for c, spec in enumerate(_A_SPECS):
        ylab = (f'{STG}\ncode (z)' if c == 0 else ('GNG code (z)' if c == 3 else ''))
        _draw_trace_col(axA[ri, c], spec, STG, ylab, show_title=(ri == 0), show_xlabel=(ri == 1))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — the DPA action/lick code, characterised by TWO compact panels on the right of the A block:
#   B1 d′ — lick vs no-lick separation on the action axis (@57–63), Naive x vs Expert y: the code is
#      decodable and roughly UNCHANGED with learning (near unity) — so the learning effect in C is a
#      POSITION shift (the push), not a decodability change.
#   B2 shared action code — signed cosine between the DPA lick axis (@57–63) and the GNG lick axis
#      (@39–45): the two lick-command axes ALIGN above chance ⇒ ONE shared action direction.
# ══════════════════════════════════════════════════════════════════════════════
_ACT_GNG = np.arange(39, 45)
gsBrow = gs[2, 5:12].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.55)   # two squares, right of A


def _lick_dprime(mouse, stage):                                       # d′(lick,no-lick) on the action axis @57–63
    base = ((Lm.mouse == mouse) & (Lm.stage == stage) & (Lm.tasks == 'DPA') & (Lm.laser == 0)).to_numpy()
    a = LICK_D[base & (Lm.choice.to_numpy() == 1)][:, _ACT_DPA].mean(1)
    b = LICK_D[base & (Lm.choice.to_numpy() == 0)][:, _ACT_DPA].mean(1)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    ps = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


_dN = np.array([_lick_dprime(m, 'Naive') for m in ALL_MICE]); _dE = np.array([_lick_dprime(m, 'Expert') for m in ALL_MICE])
axDp = fig.add_subplot(gsBrow[0, 0])
_ok = np.isfinite(_dN) & np.isfinite(_dE)
_av = np.concatenate([_dN[_ok], _dE[_ok]]); _lim = (min(_av.min(), -0.1), _av.max() * 1.12)
axDp.plot(_lim, _lim, ls='--', color='0.6', lw=0.8, zorder=1)
axDp.axhline(0, ls=':', color='0.8', lw=0.6); axDp.axvline(0, ls=':', color='0.8', lw=0.6)
for _m, _xn, _ye in zip(np.array(ALL_MICE)[_ok], _dN[_ok], _dE[_ok]):
    axDp.scatter(_xn, _ye, s=28, facecolors=MOUSE_COLOR[_m], edgecolors=MOUSE_COLOR[_m], linewidths=0.6, zorder=4)
_dp_t = float(ttest_rel(_dE[_ok], _dN[_ok]).pvalue); _dp_d = float((_dE[_ok] - _dN[_ok]).mean())
_dp_sig = _dp_t < 0.05
axDp.set_xlim(_lim); axDp.set_ylim(_lim); axDp.set_box_aspect(1)
axDp.set_title('action-code d′', fontsize=TITLE_FS, loc='left')
axDp.set_xlabel('Naive d′', fontsize=7.5); axDp.set_ylabel('Expert d′', fontsize=7.5)
axDp.text(0.06, 0.95, '*' if _dp_sig else 'n.s.', transform=axDp.transAxes, ha='left', va='top',
          fontsize=11 if _dp_sig else 8, fontweight='bold', color='k' if _dp_sig else '0.55')
axDp.text(0.5, 0.02, f'Δ={_dp_d:+.2f}, p={_dp_t:.3f}', transform=axDp.transAxes, ha='center', va='bottom',
          fontsize=6, color='0.3')
print(f'B1 action-code d′ Naive={np.nanmean(_dN):+.2f} Expert={np.nanmean(_dE):+.2f} Δ={_dp_d:+.2f} p={_dp_t:.3f}')


def _wax(mo, st, tg, win):
    k = (mo, st, 'all', tg)
    if k not in _WBLOB:
        return None
    v = np.asarray(_WBLOB[k], float)[win].mean(0); n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _act_cos(mo, st):                                                  # cos(choice@DPA-action, gng@GNG-action)
    A = _wax(mo, st, 'choice', _ACT_DPA); B = _wax(mo, st, 'gng', _ACT_GNG)
    return float(A @ B) if (A is not None and B is not None) else np.nan


COS_CHANCE = 1.0 / np.sqrt(np.mean([np.asarray(_WBLOB[(m, 'Naive', 'all', 'choice')]).shape[1] for m in ALL_MICE]))
_acN = np.array([_act_cos(m, 'Naive') for m in ALL_MICE]); _acE = np.array([_act_cos(m, 'Expert') for m in ALL_MICE])
axAct = fig.add_subplot(gsBrow[0, 1])
for _i, _m in enumerate(ALL_MICE):
    axAct.plot([0, 1], [_acN[_i], _acE[_i]], '-o', color=MOUSE_COLOR[_m], lw=0.9, ms=4.5, mec='w', mew=0.5, zorder=3)
for _x, _v in ((-0.16, _acN), (1.16, _acE)):
    _mu = np.nanmean(_v); _se = np.nanstd(_v, ddof=1) / np.sqrt(np.isfinite(_v).sum())
    axAct.errorbar(_x, _mu, yerr=_se, fmt='s', color='k', ms=6, capsize=3.5, lw=1.3, zorder=5)
axAct.axhline(COS_CHANCE, ls=':', color='0.6', lw=0.8); axAct.axhline(0, color='0.85', lw=0.6)
axAct.text(1.55, COS_CHANCE, 'chance', fontsize=5.5, color='0.6', va='bottom', ha='right')
axAct.set_xticks([0, 1]); axAct.set_xticklabels(['Naive', 'Expert'], fontsize=8); axAct.set_xlim(-0.45, 1.65)
axAct.set_ylabel('cos(DPA-lick · GNG-lick axis)', fontsize=7.5)
_acp0N = float(ttest_1samp(_acN[np.isfinite(_acN)], 0).pvalue); _acp0E = float(ttest_1samp(_acE[np.isfinite(_acE)], 0).pvalue)
_sigAct = _acp0E < 0.05
axAct.set_title('shared action code', loc='left', fontsize=TITLE_FS)
axAct.text(0.06, 0.96, '*' if _sigAct else 'n.s.', transform=axAct.transAxes, ha='left', va='top',
           fontsize=11 if _sigAct else 8, fontweight='bold', color='k' if _sigAct else '0.55')
axAct.text(0.5, -0.30, f'N={np.nanmean(_acN):+.2f} (p={_acp0N:.3f})  E={np.nanmean(_acE):+.2f} (p={_acp0E:.3f})',
           transform=axAct.transAxes, ha='center', va='top', fontsize=6, color='0.3')
axAct.set_box_aspect(1)
print(f'B2 shared action cos Naive={np.nanmean(_acN):+.3f} (p={_acp0N:.3f}) Expert={np.nanmean(_acE):+.3f} (p={_acp0E:.3f})')

gsB = gs[3, 0:12].subgridspec(1, 5, width_ratios=[5, 1.2, 5, 1.2, 4.4], wspace=0.3)   # C full row: Naive traj|kde, Expert traj|kde, push scatter
# data-driven limits, DECOUPLED x (sample code) and y (choice code) so neither axis wastes space
# (aspect stays equal → the box just goes rectangular). x fits the mean±SEM sample trajectory; y is
# kept TIGHT to the mean±SEM choice trajectory + the KDE bulk (p90) — the long choice tail is not
# allowed to inflate y; instead the hist KDE drops its out-of-range tail (see _draw_hist_B) so the
# distribution TAPERS to ~0 just inside ylim and exactly fits the axis. Means computed first (aesthetic).
_valsx, _valsy_traj, _valsy_kde = [], [], []
for _stg in STAGES:
    for _slab, _p, _c in SAMPLE_TRAJ:
        _xs, _ys = trajB[_stg][_slab]
        if not _xs or not _ys:
            continue
        _ax = np.stack(_xs, 0)[:, :TRAJ_END]; _ay = np.stack(_ys, 0)[:, :TRAJ_END]
        _nm = _ax.shape[0]
        _valsx.append(np.abs(_ax.mean(0)) + _ax.std(0, ddof=1) / np.sqrt(_nm))
        _valsy_traj.append(np.abs(_ay.mean(0)) + _ay.std(0, ddof=1) / np.sqrt(_nm))
        _valsy_kde += [np.abs(np.asarray(_yt)[BINS_DELAY]) for _yt in _ys]   # KDE delay values
_kA = np.concatenate(_valsy_kde) if _valsy_kde else np.array([4.0])
_rBx = float(np.concatenate(_valsx).max()) * 1.12 if _valsx else 4.0
_rBy = float(np.concatenate(_valsy_traj).max()) * 1.28 if _valsy_traj else 4.0   # frame to the TRAJECTORY
xlimB, ylimB = (-_rBx, _rBx), (-_rBy, _rBy)
print(f'B traj limits: x=±{_rBx:.2f}  y=±{_rBy:.2f}  (kde tail dropped to taper inside; p90={np.percentile(_kA,90):.2f})')
axB_traj, axB_hist = [], []
ax0 = None
for ci, stage in enumerate(STAGES):
    at = fig.add_subplot(gsB[0, ci * 2])
    ah = fig.add_subplot(gsB[0, ci * 2 + 1], sharey=at)
    ax0 = at if ax0 is None else ax0
    _draw_traj_B(at, stage, xlimB, ylimB)
    at.set_title(stage, pad=4, fontsize=TITLE_FS)
    at.set_xlabel('Sample code\n← odor A            odor B →')
    if ci == 0:
        at.set_ylabel('Choice code\n← no lick            lick →')
    _draw_hist_B(ah, stage, ylimB)                                     # sample A/B colour = indigo/teal (per pair legend)
    axB_traj.append(at); axB_hist.append(ah)
pair_handles = [Line2D([0], [0], color=_c, lw=2.0, label=f'Sample {_l}') for _l, _p, _c in SAMPLE_TRAJ]
axB_traj[-1].legend(handles=pair_handles, frameon=False, loc='upper right',
                    handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3, fontsize=8)

# ── B depth panel: per-mouse late-delay choice-code depth, Naive → Expert (D-style paired plot) ──
#    Stage on x, per-mouse colour (panel-C palette), sample A filled / B open, each mouse's Naive→
#    Expert pair joined, black group mean±SEM bars. Stat = deepening mixed model
#    depth ~ stage + sample + (1|mouse) (same estimator family as C).
axB_sc = fig.add_subplot(gsB[0, 4])
GX_B = (0.0, 1.0)                                                                  # Naive, Expert x-positions
for _slab, _fill in (('A', True), ('B', False)):
    P = pushB[_slab]
    for mo, xn, ye in zip(P['mice'], P['naive'], P['expert']):
        _mc = MOUSE_COLOR[mo]
        axB_sc.plot(GX_B, [xn, ye], '-', color=_mc, lw=0.7, alpha=0.5, zorder=2)
        axB_sc.scatter(GX_B[0], xn, s=30, zorder=3, linewidths=1.0,
                       facecolors=_mc if _fill else 'w', edgecolors=_mc)
        axB_sc.scatter(GX_B[1], ye, s=30, zorder=3, linewidths=1.0,
                       facecolors=_mc if _fill else 'w', edgecolors=_mc)
_naive_all = np.concatenate([pushB[s]['naive'] for s in ('A', 'B')])              # 18 mouse×sample obs / stage
_expert_all = np.concatenate([pushB[s]['expert'] for s in ('A', 'B')])
for _xx, _vals in ((GX_B[0], _naive_all), (GX_B[1], _expert_all)):
    _mu = _vals.mean(); _se = _vals.std(ddof=1) / np.sqrt(len(_vals))
    axB_sc.plot([_xx - 0.14, _xx + 0.14], [_mu, _mu], color='k', lw=1.8, zorder=4)
    axB_sc.errorbar(_xx, _mu, yerr=_se, color='k', capsize=2.5, lw=1.2, zorder=4)
axB_sc.axhline(0, ls=':', color='0.6', lw=0.7)
# stat: deepening mixed model depth ~ stage + sample + (1|mouse)
_dfp = pd.DataFrame([dict(mouse=mo, sample=_s, st=_st, depth=_v)
                     for _s in ('A', 'B') for _st, _k in ((0, 'naive'), (1, 'expert'))
                     for mo, _v in zip(pushB[_s]['mice'], pushB[_s][_k])])
_pfit = smf.mixedlm('depth ~ st + C(sample)', _dfp, groups=_dfp['mouse']).fit()
_bpush, _ppush = float(_pfit.params['st']), float(_pfit.pvalues['st'])
_nmB, _noB = _dfp['mouse'].nunique(), len(_dfp)
_sigB = _ppush < 0.05
print(f'B depth [mixed model, {_noB} obs] β={_bpush:+.3f} p={_ppush:.3f} ({_nmB} mice)')
axB_sc.set_xlim(-0.5, 1.5); axB_sc.set_xticks(GX_B); axB_sc.set_xticklabels(['Naive', 'Expert'])
axB_sc.set_box_aspect(1)
axB_sc.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=7.5)
axB_sc.set_title('Choice-code depth', loc='left', fontsize=TITLE_FS)
axB_sc.text(0.03, 0.03, f'mixed model ({_nmB} mice, {_noB} obs)\nβ={_bpush:+.3f}, p={_ppush:.3f}',
            transform=axB_sc.transAxes, ha='left', va='bottom', fontsize=6.5, color='0.3')
axB_sc.text(0.06, 0.96, '*' if _sigB else 'n.s.', transform=axB_sc.transAxes, ha='left', va='top',
            fontsize=12 if _sigB else 8, fontweight='bold', color='k' if _sigB else '0.55')   # C-style marker
axB_sc.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
                       mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')],
              frameon=False, loc='upper right', fontsize=6.5, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.3)

# ── C: Δdepth ↔ Δperf (Expert−Naive), A&B independent (ΔDPA | ΔGNG) — right half ─
gsC = gs[4, 0:8].subgridspec(1, 2, wspace=0.55)                        # D + E share the last row
axC = [fig.add_subplot(gsC[0, 0]), fig.add_subplot(gsC[0, 1])]
C_specs = [(delta_dpa_perf_sample, 'Δ DPA accuracy (Exp−Naive)', 'Δ depth vs Δ DPA accuracy',
            _panelC_coupling(delta_dpa_perf_sample)),
           (delta_gng_perf_sample, 'Δ GNG accuracy (Exp−Naive)', 'Δ depth vs Δ GNG accuracy',
            _panelC_coupling(delta_gng_perf_sample))]
_allyC = np.array([d[(m, c)] for d, _, _, _ in C_specs for m in ALL_MICE for c in (0, 1)], float)
_allyC = _allyC[~np.isnan(_allyC)]
_padC = (_allyC.max() - _allyC.min()) * 0.15 or 0.05
ylimC = (_allyC.min() - _padC, _allyC.max() + _padC)
for ax, (yv_dict, ylabel, msg, (rho, pv, n_mice, mx, my)) in zip(axC, C_specs):
    for mouse in ALL_MICE:
        px, py = [], []
        for cls, pairs in D_SAMPLE_CLASSES:
            xx = delta_choice_sample[(mouse, cls)]
            yy = yv_dict.get((mouse, cls), np.nan)
            px.append(xx); py.append(yy)
            if not (np.isnan(xx) or np.isnan(yy)):
                face = MOUSE_COLOR[mouse] if cls == 0 else 'w'         # A solid / B open
                ax.scatter(xx, yy, facecolors=face, edgecolors=MOUSE_COLOR[mouse],
                           marker='o', s=42, linewidths=1.0, zorder=5)
        ax.plot(px, py, '-', color=MOUSE_COLOR[mouse], lw=0.7, alpha=0.5, zorder=3)
    regression_band(ax, mx, my)                                        # band on the per-mouse means (matches the n=9 stat)
    ax.axhline(0, ls=':', color='k', lw=0.7); ax.axvline(0, ls=':', color='k', lw=0.7)
    ax.set_ylim(ylimC)
    sig = pv < 0.05
    ax.text(0.03, 0.03, f'per-mouse (n={n_mice})\nSpearman ρ={rho:+.2f}, p={pv:.3f}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color='0.3')
    ax.text(0.92, 0.94, '*' if sig else 'n.s.', transform=ax.transAxes, ha='center', va='top',
            fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    ax.set_xlabel('Δ DPA choice-code depth'); ax.set_ylabel(ylabel)
    ax.set_title(msg, loc='left', fontsize=TITLE_FS)
    ax.set_box_aspect(1)
    print(f'D[{ylabel[:6]}] per-mouse n={n_mice} Spearman ρ={rho:+.3f} p={pv:.3f}')
_C_leg = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
          mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')]
axC[0].legend(handles=_C_leg, frameon=False, loc='upper center', bbox_to_anchor=(0.42, 1.0),
              ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)

# ── D: Naive nonpaired corr-rej vs false-alarm depth, sample A | sample B ────────
axD = fig.add_subplot(gs[4, 8:12])
GX_FACR = {'AD': (0.0, 0.8), 'BC': (1.9, 2.7)}
for lab, samp, odor_pair, col in FA_CR_SPEC:
    xc, xe = GX_FACR[lab]; r = facr[lab]
    for ya, yb, mouse in zip(r['cr'], r['fa'], r['used']):
        mc = MOUSE_COLOR[mouse]                                                   # per-mouse colour (panel-C palette)
        axD.plot([xc, xe], [ya, yb], '-', color=mc, lw=0.7, alpha=0.5, zorder=2)
        axD.scatter(xc, ya, s=30, facecolors=mc, edgecolors=mc, linewidths=1.0, zorder=3)   # corr-rej = filled
        axD.scatter(xe, yb, s=30, facecolors='w', edgecolors=mc, linewidths=1.1, zorder=3)  # false-alarm = open
    for xx, vals in ((xc, r['cr']), (xe, r['fa'])):
        if len(vals):
            mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            axD.plot([xx - 0.18, xx + 0.18], [mu, mu], color='k', lw=1.8, zorder=4)
            axD.errorbar(xx, mu, yerr=se, color='k', capsize=2.5, lw=1.2, zorder=4)
    n = len(r['cr']); d_mean = float((r['cr'] - r['fa']).mean()) if n else np.nan
    tp = float(ttest_rel(r['cr'], r['fa']).pvalue) if n >= 3 else np.nan
    sig = (tp == tp and tp < 0.05)
    axD.text((xc + xe) / 2, 0.99, f'{lab} (sample {samp})', transform=axD.get_xaxis_transform(),
             ha='center', va='top', fontsize=7, fontweight='bold', color=col)
    axD.text((xc + xe) / 2, 0.88, '*' if sig else 'n.s.', transform=axD.get_xaxis_transform(),   # C-style marker
             ha='center', va='top', fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
    axD.text((xc + xe) / 2, 0.02, f'p={tp:.3f}', transform=axD.get_xaxis_transform(),
             ha='center', va='bottom', fontsize=6.5, color='0.3')
    print(f'D(FA/CR)[Naive {lab} sample {samp}] Δ(cr−fa)={d_mean:+.3f} paired-t p={tp:.3f} n={n}')
axD.axhline(0, ls=':', color='0.6', lw=0.7)
axD.set_xticks([0.0, 0.8, 1.9, 2.7])
axD.set_xticklabels(['corr.\nrej.', 'false\nalarm', 'corr.\nrej.', 'false\nalarm'], fontsize=6.5)
axD.set_xlim(-0.5, 3.2)
axD.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=7.5)
axD.set_title('Naive nonpaired trials', loc='left', fontsize=TITLE_FS)
axD.set_box_aspect(1)

# ── panel letters ───────────────────────────────────────────────────────────────
panel_letter(axA[0, 0], 'A')
panel_letter(axDp, 'B', x=0.42)
panel_letter(axB_traj[0], 'C')
panel_letter(axC[0], 'D')
panel_letter(axD, 'E', x=0.655)

OUT = 'figures/overlaps/main/eqnorm' if EQNORM else 'figures/overlaps/main'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/fig_overlaps_main_ab{FILE_SUF}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
