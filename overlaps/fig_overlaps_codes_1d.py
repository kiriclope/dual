"""1-D overlaps code trajectories over time, dPCA-style (plot_pseudo_traj.py conventions:
mean +/- SEM, project colours, add_vlines, ticks).

Each panel reads a variable on ITS OWN target code (decision function): sample / choice / test
each split by their label; task = per-task average on the choice code. sample/choice/test are
DPA-only (the Dual 2nd-odor distractor ~7-8 s would contaminate the trace); task keeps all tasks.

Sweep: the decoder is averaged over a TRAIN epoch then read across test time. We anchor that
epoch at nine moments spanning the trial (stim / ed / md / gng_rwd / delay / ld / test / choice /
dpa_rwd) so the codes become a movie of WHEN each representation is expressed.

Trial structure (3 odors, 2 outcomes): S1 sample ~2.5 s -> ED ~3.5 s -> Go/NoGo distractor
odor ~5 s -> MD ~6 s -> GNG reward ~7 s -> LD ~8 s -> S2 test ~9.5 s -> DPA choice ~9.5-10 s
-> DPA reward ~11.5 s. CRUCIAL: the Go/NoGo distractor odor AND the GNG reward exist ONLY on
Dual trials (DPA trials have no distractor; odr_choice is NaN). So on the DPA-only panels
(sample/choice/test) ed/md/gng_rwd/ld are just UNINTERRUPTED delay-maintenance timepoints -
nothing happens on those trials. The distractor/GNG-reward events live in the TASK panel
(all tasks), where md/gng_rwd bracket the GNG odor and its outcome (DualGo lick vs DualNoGo).

Interpretation by panel:
  - sample : epoch-invariant (ed~=md~=ld) -> one stable memory held the whole delay. Robustness
    is shown by STABILITY, not by surviving a within-trial distractor (DPA has none).
  - choice (DPA-only) : non-flat already in the delay = a MAINTAINED no-lick action SET held
    alongside the memory (the dual-coding geometry; pushed deeper into no-lick with learning),
    firming into the executed lick at choice/dpa_rwd. NOT foreknowledge of the answer - the
    correct DPA lick is undetermined until S2.
  - test : real only once S2 is on (test/choice/dpa_rwd); any pre-test anchor flags its test
    panel as a pre-test confound (axis trained before the test odor is spurious).

For each (train epoch x stage in Naive/Expert) three figures are written to
figures/overlaps/codes1d/<epoch>/{png,svg}/ (epoch = subdir):
  - grandmean : per-mouse mean trajectory -> mean +/- SEM OVER MICE (the honest error bar)
  - permouse  : 9x4 per-mouse grid (SEM over trials)
  - pooled    : trials pooled over mice, SEM over trials (REFERENCE only, pseudo-replicated)
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.options import set_options
from src.pca.io import pkl_load
from src.plot.traj import plot_mean_sem
from src.common.plot_utils import add_vlines

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)
pal = sns.color_palette('muted')

DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
xtime = np.linspace(0, 14, 84); BL = slice(0, 12); W, H = 3.5, 2.6
o = set_options(mice=MICE, tasks=['Dual'], mouse=MICE[0], laser=0, days=['first', 'last'], mne_estimator='generalizing')
# Train epoch the decoder is averaged over before reading across test time.
# Sweep anchors across the whole trial (stimulus → delay → response → reward) so the codes
# become a movie of WHEN each representation is expressed. _LD = last 40% of delay (BINS_LATE).
_DELAY = o['bins_DELAY']; _LD = _DELAY[int(0.6 * len(_DELAY)):]
TRAIN_EPOCHS = [
    ('trainSTIM',   o['bins_STIM']),   # sample odor on (~2.5 s) — encoding
    ('trainED',     o['bins_ED']),     # early delay (~3.5 s)
    ('trainMD',     o['bins_MD']),     # mid delay, AFTER the Go/NoGo odor (~6 s; Dual-only event)
    ('trainGNGRWD', o['bins_RWD']),    # GNG reward / outcome (~7 s; Dual-only — empty on DPA)
    ('trainDELAY',  _DELAY),           # full delay
    ('trainLD',     _LD),              # late delay (~7-9 s) — pre-test memory
    ('trainTEST',   o['bins_TEST']),   # test odor / readout (~9.5 s)
    ('trainCHOICE', o['bins_CHOICE']), # DPA choice / response (~9.5-10 s)
    ('trainDPARWD', o['bins_RWD2']),   # DPA reward / outcome (~11.5 s)
]
TEST_ONSET = o['bins_TEST'][0]         # test odor absent before this bin
# (title, target code, split column, [levels], [labels], [colours], dpa_only)
# dpa_only=True restricts to DPA trials so the Dual distractor (2nd odor ~7-8 s) doesn't
# contaminate the trace; the task panel keeps all tasks (that's its whole point).
VARS = [
    ('sample', 'sample', 'sample_odor', [0, 1], ['Odor A', 'Odor B'], ['#332288', '#44AA99'], True),
    ('choice', 'choice', 'choice', [0, 1], ['No lick', 'Lick'], ['#377eb8', '#4daf4a'], True),
    ('test',   'test',   'test_odor', [0, 1], ['Odor C', 'Odor D'], ['#377eb8', '#4daf4a'], True),
    ('task',   'choice', 'tasks', ['DPA', 'DualGo', 'DualNoGo'], ['DPA', 'Go', 'NoGo'], [pal[3], pal[0], pal[2]], False),
]
OUT = 'figures/overlaps/codes1d'
# Each train epoch gets its own subdir; png/svg live under it (project convention).
SUBDIR = {'trainSTIM': 'stim', 'trainED': 'ed', 'trainMD': 'md', 'trainGNGRWD': 'gng_rwd',
          'trainDELAY': 'delay', 'trainLD': 'ld', 'trainTEST': 'test',
          'trainCHOICE': 'choice', 'trainDPARWD': 'dpa_rwd'}

X = pkl_load(f'X_{DUM}', path='../data/overlaps')
y = pkl_load(f'labels_{DUM}', path='../data/overlaps')
def setup(ax, ylab):
    add_vlines(ax, if_dpa=0); ax.axhline(0, ls='--', color='k', lw=0.6, zorder=1)
    ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
    ax.set_xlabel('Time (s)', fontsize=10); ax.set_ylabel(ylab, fontsize=10); ax.tick_params(labelsize=8)


def panel_title(ttl, code, dpa_only, test_valid):
    t = f'{ttl} code' + (' (DPA)' if dpa_only else '')
    # A pre-test-trained test axis is spurious (test odor absent → reads a sample/distractor
    # confound, anti-aligned with the true test code). See overview.md.
    if code == 'test' and not test_valid:
        t += '  ⚠ pre-test confound'
    return t


for TRAIN_TAG, TRAIN in TRAIN_EPOCHS:
    df = X[..., TRAIN, :].mean(-2)[:, 1].astype(float)       # (n, 84) decision fn, train-epoch averaged
    for mo in MICE:                                          # per-mouse BL-std norm
        mm = (y.mouse == mo).to_numpy(); sd = df[mm][:, BL].std()
        if sd > 0:
            df[mm] /= sd

    sub = SUBDIR[TRAIN_TAG]                                  # codes1d/<epoch>/{png,svg}/
    test_valid = np.max(TRAIN) >= TEST_ONSET                 # test odor present in this anchor?
    for d in ('png', 'svg'):
        os.makedirs(os.path.join(OUT, sub, d), exist_ok=True)

    for STAGE in ['Naive', 'Expert']:
        tag = STAGE.lower()
        base = ((y.laser == 0) & (y.learning == STAGE) & (y.performance == 1)).to_numpy()

        # ---- per-mouse 9 x 4 grid (per-trial BL subtraction; SEM over trials) ----
        fig, axes = plt.subplots(len(MICE), 4, figsize=(4 * W, len(MICE) * H), sharex=True)
        for r, mo in enumerate(MICE):
            for c, (ttl, code, col, levels, labs, cols, dpa_only) in enumerate(VARS):
                ax = axes[r, c]; setup(ax, mo if c == 0 else '')
                pbase = base & (y.tasks == 'DPA').to_numpy() if dpa_only else base
                sel0 = pbase & (y.target == code).to_numpy() & (y.mouse == mo).to_numpy()
                Z = df[sel0] - df[sel0][:, BL].mean(1, keepdims=True); yc = y[sel0].reset_index(drop=True)
                for lv, lab, color in zip(levels, labs, cols):
                    s = (yc[col].to_numpy() == lv)
                    if s.sum() >= 3:
                        plot_mean_sem(ax, xtime, Z[s].mean(0), Z[s].std(0) / np.sqrt(s.sum()), color, lw=1.4, label=lab, zorder=2)
                if r == 0:
                    ax.set_title(panel_title(ttl, code, dpa_only, test_valid), fontsize=11); ax.legend(fontsize=7, frameon=False, loc='upper left')
        for ax in axes[-1]:
            ax.set_xlabel('Time (s)')
        fig.suptitle(f'Overlaps 1-D codes per mouse — {STAGE} ({TRAIN_TAG}) — mean ± SEM', y=1.004)
        fig.tight_layout()
        p = os.path.join(OUT, sub, 'png', f'overlaps_codes1d_permouse_{tag}.png')
        fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig); print('saved', p)

        # ---- grand mean 1 x 4: per-mouse mean trajectory, then mean ± SEM OVER MICE ----
        # (no trial pooling — each mouse contributes one trace, SEM is across the n<=9 mice)
        fig2, axes2 = plt.subplots(1, 4, figsize=(4 * W, H + 0.4), sharex=True)
        for c, (ttl, code, col, levels, labs, cols, dpa_only) in enumerate(VARS):
            ax = axes2[c]; setup(ax, 'code (BL σ)' if c == 0 else '')
            pbase = base & (y.tasks == 'DPA').to_numpy() if dpa_only else base
            Zc = np.full_like(df, np.nan)
            for mo in MICE:                               # per-mouse BL z-score of the code
                mm = (y.mouse == mo).to_numpy() & (y.target == code).to_numpy()
                z = df[mm]; z = z - z[:, BL].mean(); Zc[mm] = z / (df[mm][:, BL].std() + 1e-9)
            for lv, lab, color in zip(levels, labs, cols):
                per_mouse = []                            # one mean trajectory per mouse
                for mo in MICE:
                    s = (pbase & (y.target == code).to_numpy()
                         & (y.mouse == mo).to_numpy() & (y[col].to_numpy() == lv))
                    if s.sum() >= 3:
                        per_mouse.append(np.nanmean(Zc[s], 0))
                if len(per_mouse) >= 2:
                    M = np.stack(per_mouse, 0); n = M.shape[0]
                    plot_mean_sem(ax, xtime, M.mean(0), M.std(0, ddof=1) / np.sqrt(n),
                                  color, lw=1.8, label=f'{lab} (n={n})', zorder=2)
            ax.set_title(panel_title(ttl, code, dpa_only, test_valid), fontsize=11); ax.legend(fontsize=8, frameon=False, loc='upper left')
        fig2.suptitle(f'Overlaps 1-D codes — {STAGE} (per-mouse mean, then mean ± SEM over mice, {TRAIN_TAG})', y=1.04)
        fig2.tight_layout()
        p2 = os.path.join(OUT, sub, 'png', f'overlaps_codes1d_grandmean_{tag}.png')
        fig2.savefig(p2, dpi=300, bbox_inches='tight'); fig2.savefig(p2.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig2); print('saved', p2)

        # ---- pooled 1 x 4: per-mouse BL z-score, then POOL trials over mice; SEM over trials ----
        # REFERENCE ONLY: tight but pseudo-replicated error bars (SEM over ~thousands of trials,
        # high-trial mice dominate). The honest error bar is the grandmean (over mice) above.
        fig3, axes3 = plt.subplots(1, 4, figsize=(4 * W, H + 0.4), sharex=True)
        for c, (ttl, code, col, levels, labs, cols, dpa_only) in enumerate(VARS):
            ax = axes3[c]; setup(ax, 'code (BL σ)' if c == 0 else '')
            pbase = base & (y.tasks == 'DPA').to_numpy() if dpa_only else base
            Zc = np.full_like(df, np.nan)
            for mo in MICE:                               # per-mouse BL z-score of the code
                mm = (y.mouse == mo).to_numpy() & (y.target == code).to_numpy()
                z = df[mm]; z = z - z[:, BL].mean(); Zc[mm] = z / (df[mm][:, BL].std() + 1e-9)
            selc = pbase & (y.target == code).to_numpy()
            for lv, lab, color in zip(levels, labs, cols):
                s = selc & (y[col].to_numpy() == lv); Zs = Zc[s]
                if Zs.shape[0] >= 3:
                    plot_mean_sem(ax, xtime, np.nanmean(Zs, 0), np.nanstd(Zs, 0) / np.sqrt(Zs.shape[0]),
                                  color, lw=1.8, label=f'{lab} (N={Zs.shape[0]})', zorder=2)
            ax.set_title(panel_title(ttl, code, dpa_only, test_valid), fontsize=11); ax.legend(fontsize=8, frameon=False, loc='upper left')
        fig3.suptitle(f'Overlaps 1-D codes, pooled — {STAGE} (trials pooled over mice; SEM over trials — REFERENCE, {TRAIN_TAG})', y=1.04)
        fig3.tight_layout()
        p3 = os.path.join(OUT, sub, 'png', f'overlaps_codes1d_pooled_{tag}.png')
        fig3.savefig(p3, dpi=300, bbox_inches='tight'); fig3.savefig(p3.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig3); print('saved', p3)
