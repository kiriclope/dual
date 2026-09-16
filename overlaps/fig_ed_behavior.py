"""fig_ed_behavior.py — Extended Data Fig. 1: licking behaviour and per-animal learning in the imaged cohort
(companion to Fig. 1). Built 2026-09-16 after the supplementary review ("we need an ED figure with the actual
licks"; per-animal learning curves; a trial-history control for Fig. 1g).

  a  lick rasters, one example mouse, NoGo laser-off trials of its first and last dual-task session
  b  lick-rate time courses per trial type (DPA / Go / NoGo) and stage, mean ± SEM across the nine mice
     (behaviour-file lick times, drawn on the imaging clock of Figs 3a / ED 4a: the file's sample stamp sits
     1.5 s before it)
  c  delay-lick rate per mouse (the rig's cue-lick flag `odr_choice`, Fig. 1g's predictor), NoGo and Go trials,
     naïve → expert, paired Wilcoxon n = 9
  d  per-animal learning curves: DPA accuracy (DPA trials) and GNG accuracy (Go + NoGo trials) per session,
     thin lines per mouse, mean ± SEM
  e  trial-history control for Fig. 1g: the delay-lick → test-lick propagation (GEE, clustered by mouse) with
     and without the previous trial's outcome, test lick and trial type as covariates
Data: data/pca/y_all_nan_.pkl (session order; laser-off current trials, full sequence for the history) and the
behaviour .mat files (lick times). Run:  cd /home/leon/dual/overlaps && python fig_ed_behavior.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig1.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, glob, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); sys.path.insert(0, '/home/leon/dual/pca')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.io import loadmat
from scipy.stats import wilcoxon
import seaborn as sns, matplotlib.pyplot as plt
from figcaption import draw_justified
from src.pca.io import pkl_load
from src.common.options import set_options

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.2
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = PS*8
NOCAP = '--nocap' in sys.argv[1:]
ED_N = 1
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MC = dict(zip(MICE, sns.color_palette('tab10', n_colors=len(MICE))))
TASK_COL = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
TASK_LAB = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
STAGES = ['Naive', 'Expert']; SC = {'Naive': '0.55', 'Expert': '#332288'}
EVENTS = [('sample', 2.0, 3.0, '#332288'), ('GNG', 4.5, 5.5, '#cc3311'), ('cue', 6.5, 7.0, '#ee7733'), ('test', 9.0, 10.0, '#377eb8')]
OFFSET = 1.5                                   # imaging clock = behaviour-file time + 1.5 s (lick PSTH check, 2026-09-15)
PATH = '/storage/leon/dual_task/data/2Samples-DualTask-BehavioralData'
EXAMPLE = 'ChRM04'

# ── trial table in session order ──
y = pkl_load('y_all_nan_', path='../data/pca')
y['trial'] = y.groupby(['mouse', 'day']).cumcount(); y['day'] = y.day.astype(int)
stage_of = y.groupby(['mouse', 'day']).learning.first().to_dict()

# ── lick times per trial from the behaviour files ──
EDGES = np.arange(-0.5, 12.51, 0.25); TC = (EDGES[:-1] + EDGES[1:]) / 2 + OFFSET
psth = {}; rast = {}
for mouse in MICE:
    for day in range(1, set_options(mouse=mouse)['n_days'] + 1):
        f = glob.glob(f'{PATH}/{mouse}-DualTask-BehavioralData/day_{day}/*.mat')
        if not f:
            continue
        try:
            m = loadmat(f[0])
        except Exception as e:
            print(f'  skip {mouse} day {day}: {type(e).__name__}'); continue
        on = m['Sample'][:, 0] / 1e3; lk = m['lickTime'][:, 0] / 1e3
        tr = m['AllTrials'][0][0][-1]; task = pd.Series(tr[:, 4]).map({0: 'DPA', 1: 'DualGo', 2: 'DualNoGo'}).to_numpy(); las = tr[:, 8]
        tl = [lk[(lk >= on[i]) & (lk < (on[i + 1] if i + 1 < len(on) else on[i] + 20))] - on[i] for i in range(len(on))]
        st = stage_of.get((mouse, day))
        for i in range(len(on)):
            if las[i] != 0:
                continue
            psth.setdefault((mouse, st, task[i]), []).append(np.histogram(tl[i], EDGES)[0] / 0.25)
        if mouse == EXAMPLE and task is not None:
            rast[(day, st)] = [tl[i] + OFFSET for i in range(len(on)) if las[i] == 0 and task[i] == 'DualNoGo']
PS_MEAN = {k: np.mean(v, 0) for k, v in psth.items()}                         # per mouse x stage x task mean rate


def plabel(ax, s, dx=-0.10, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


def shade(ax, labels=False):
    for nm, a, b, col in EVENTS:
        ax.axvspan(a, b, color=col, alpha=0.10, lw=0)
        if labels:
            ax.text((a + b) / 2, 0.98, nm, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=PS*6.0, color=col)


# ── a: rasters ──
def panel_a(fig, gs):
    days = sorted(rast); first = [d for d in days if d[1] == 'Naive'][0]; last = [d for d in days if d[1] == 'Expert'][-1]
    axes = []
    for k, key in enumerate([first, last]):
        ax = fig.add_subplot(gs[k, 0]); axes.append(ax); shade(ax, labels=(k == 0))
        for j, t in enumerate(rast[key]):
            ax.scatter(t, np.full(len(t), j), s=2.5, color='k', lw=0, zorder=3)
        ax.set_xlim(0, 12.5); ax.set_ylim(-1, len(rast[key])); ax.invert_yaxis()
        ax.set_ylabel('NoGo trial'); ax.set_title(f'{EXAMPLE}, {"naïve" if key[1] == "Naive" else "expert"} (session {key[0]})', loc='left', fontsize=TITLE_FS)
        if k == 1:
            ax.set_xlabel('time (s)')
        else:
            ax.tick_params(labelbottom=False)
    return axes[0]


# ── b: lick-rate time courses ──
def panel_b(fig, gs):
    axes = []; hi = 0
    for r, st in enumerate(STAGES):
        ax = fig.add_subplot(gs[r, 0]); axes.append(ax); shade(ax)
        for task in ['DPA', 'DualGo', 'DualNoGo']:
            M = np.array([PS_MEAN[(m, st, task)] for m in MICE if (m, st, task) in PS_MEAN])
            mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M)); hi = max(hi, (mu + se).max())
            ax.plot(TC, mu, color=TASK_COL[task], lw=1.2, label=f'{TASK_LAB[task]} (n = {len(M)})'); ax.fill_between(TC, mu - se, mu + se, color=TASK_COL[task], alpha=0.18, lw=0)
        ax.set_xlim(0, 12.5); ax.set_ylabel(f'{"naïve" if st == "Naive" else "expert"}\nlick rate (Hz)')
        if r == 0:
            ax.legend(frameon=False, loc='upper left', handlelength=1.2); ax.tick_params(labelbottom=False)
            ax.set_title('lick rate by trial type, 9 mice', loc='left', fontsize=TITLE_FS)
        else:
            ax.set_xlabel('time (s)')
    for ax in axes:
        ax.set_ylim(0, hi * 1.08)
    return axes[0]


# ── c: delay-lick rate per mouse (rig cue-lick flag, Fig. 1g's predictor) ──
def panel_c(fig, gs):
    d = y[(y.laser == 0) & y.tasks.isin(['DualGo', 'DualNoGo'])].copy(); d['licked'] = (d.odr_choice > 0).astype(float)
    axes = []
    for k, task in enumerate(['DualNoGo', 'DualGo']):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        r = d[d.tasks == task].groupby(['mouse', 'learning']).licked.mean().unstack()
        for m in MICE:
            ax.plot([0, 1], [r.loc[m, 'Naive'], r.loc[m, 'Expert']], '-o', color=MC[m], lw=0.8, ms=4, alpha=0.8, zorder=3)
        ax.plot([0, 1], [r['Naive'].mean(), r['Expert'].mean()], '-', color='k', lw=1.8, zorder=4)
        p = wilcoxon(r['Naive'], r['Expert']).pvalue; sig = p < .05
        ax.text(0.5, 0.97, '∗' if sig else 'n.s.', transform=ax.transAxes, ha='center', va='top', fontsize=PS*(12 if sig else 8), fontweight='bold', color='k' if sig else '0.55')
        ax.text(0.5, 0.86, f'p = {p:.3f}\n9 mice', transform=ax.transAxes, ha='center', va='top', fontsize=PS*6.5, color='0.3')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['naïve', 'expert']); ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f'{TASK_LAB[task]} trials', loc='left', fontsize=TITLE_FS)
        if k == 0:
            ax.set_ylabel('fraction of trials\nwith a delay lick')
        else:
            ax.tick_params(labelleft=False)
        print(f'c: {TASK_LAB[task]:4s} delay-lick rate {r["Naive"].mean():.2f} -> {r["Expert"].mean():.2f}  Wilcoxon p={p:.3f}  {(r["Expert"] < r["Naive"]).sum()}/9 down')
    return axes[0]


# ── d: per-animal learning curves ──
def panel_d(fig, gs):
    d = y[y.laser == 0]
    axes = []
    for k, (lab, sub, col) in enumerate([('DPA accuracy, DPA trials', d[d.tasks == 'DPA'].assign(acc=lambda t: t.performance), '#e8000b'),
                                         ('GNG accuracy, Go + NoGo trials', d[d.tasks.isin(['DualGo', 'DualNoGo'])].assign(acc=lambda t: t.odr_perf), '#023eff')]):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        pm = sub.groupby(['mouse', 'day']).acc.mean().unstack()
        for m in MICE:
            ax.plot(pm.columns, pm.loc[m], '-', color=MC[m], lw=0.8, alpha=0.7, zorder=2)
        mu, se = pm.mean(0), pm.std(0, ddof=1) / np.sqrt(pm.notna().sum(0))
        ax.errorbar(pm.columns, mu, se, fmt='o-', color=col, ms=4, lw=1.8, capsize=2, zorder=4)
        ax.axhline(0.5, ls=':', color='0.6', lw=0.8); ax.set_ylim(0.3, 1.02); ax.set_xticks(pm.columns)
        ax.set_xlabel('dual-task session'); ax.set_title(lab, loc='left', fontsize=TITLE_FS)
        if k == 0:
            ax.set_ylabel('accuracy')
        else:
            ax.tick_params(labelleft=False)
        print(f'd: {lab}: day means ' + ' '.join(f'{v:.2f}' for v in mu.values))
    return axes[0]


# ── e: trial-history control for Fig. 1g ──
def panel_e(ax):
    s = y.sort_values(['mouse', 'day', 'trial']).copy()
    g = s.groupby(['mouse', 'day'])
    s['prev_perf'] = g.performance.shift(1); s['prev_testlick'] = g.choice.shift(1); s['prev_task'] = g.tasks.shift(1)
    s['licked'] = (s.odr_choice > 0).astype(float); s['testlick'] = (s.choice > 0).astype(float)
    s = s[(s.laser == 0) & s.tasks.isin(['DualGo', 'DualNoGo'])].dropna(subset=['prev_perf', 'prev_testlick', 'prev_task'])
    MODELS = [('Fig. 1g', 'testlick ~ licked'), ('+ trial type', 'testlick ~ licked + C(tasks)'),
              ('+ previous\noutcome', 'testlick ~ licked + C(tasks) + prev_perf'), ('+ previous\ntest lick', 'testlick ~ licked + C(tasks) + prev_perf + prev_testlick'),
              ('+ previous\ntrial type', 'testlick ~ licked + C(tasks) + prev_perf + prev_testlick + C(prev_task)')]
    out = {}
    for j, st in enumerate(STAGES):
        d = s[s.learning == st]
        for i, (lab, f) in enumerate(MODELS):
            r = smf.gee(f, groups=d['mouse'], data=d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            orr = np.exp(r.params['licked']); ci = np.exp(r.conf_int().loc['licked']); p = r.pvalues['licked']
            x = i + (-0.17 if st == 'Naive' else 0.17)
            ax.errorbar(x, orr, [[orr - ci[0]], [ci[1] - orr]], fmt='o', color=SC[st], ms=4.5, capsize=2.5, lw=1.1, label=('naïve' if st == 'Naive' else 'expert') if i == 0 else None)
            out[(st, lab)] = (orr, ci[0], ci[1], p)
            extra = ''
            if 'prev_testlick' in r.params:
                extra += f'  prev test lick OR={np.exp(r.params["prev_testlick"]):.2f} p={r.pvalues["prev_testlick"]:.3f}'
            if 'prev_perf' in r.params:
                extra += f'  prev outcome OR={np.exp(r.params["prev_perf"]):.2f} p={r.pvalues["prev_perf"]:.3f}'
            print(f'e: {st:6s} {lab.replace(chr(10), " "):22s} delay-lick OR={orr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] p={p:.4f}  n={len(d)}{extra}')
    ax.axhline(1, ls=':', color='0.6', lw=0.8); ax.set_yscale('log'); ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(['1', '2', '3', '4']); ax.set_yticks([], minor=True)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([m[0] for m in MODELS], fontsize=PS*6.2); ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylabel('odds ratio, test lick | delay lick'); ax.legend(frameon=False, loc='upper right')
    ax.set_title('Fig. 1g propagation with trial-history covariates', loc='left', fontsize=TITLE_FS)
    return out


# ══ ASSEMBLE ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 7.6))
outer = fig.add_gridspec(2, 24, height_ratios=[1.0, 0.85], hspace=0.55, wspace=1.0, left=0.07, right=0.985, top=0.95, bottom=0.09)
gsA = outer[0, 0:7].subgridspec(2, 1, hspace=0.25); axA = panel_a(fig, gsA)
gsB = outer[0, 9:16].subgridspec(2, 1, hspace=0.25); axB = panel_b(fig, gsB)
gsC = outer[0, 18:24].subgridspec(1, 2, wspace=0.12); axC = panel_c(fig, gsC)
gsD = outer[1, 0:11].subgridspec(1, 2, wspace=0.12); axD = panel_d(fig, gsD)
axE = fig.add_subplot(outer[1, 14:24]); E = panel_e(axE)
plabel(axA, 'a', dx=-0.30); plabel(axB, 'b', dx=-0.24); plabel(axC, 'c', dx=-0.55); plabel(axD, 'd', dx=-0.22); plabel(axE, 'e', dx=-0.14)

CAP = [
    f'Extended Data Fig. {ED_N} | Licking and per-animal learning in the imaged cohort (companion to Fig. 1). '
    'a, Lick rasters of one mouse on the NoGo laser-off trials of its first and last dual-task session (one row per '
    'trial; shading, sample odor, Go/NoGo odor, response cue and test odor; time on the imaging clock of Fig. 3a). '
    'b, Lick rate against time on DPA, Go and NoGo trials (mean ± SEM across nine mice), naïve (top) and expert '
    '(bottom): Go trials carry the required lick at the cue, NoGo trials the delay licks that Fig. 1g follows to the '
    'test, DPA trials no cue and no delay lick. c, The fraction of trials with a delay lick at the cue (the rig\'s '
    'cue-lick flag, the predictor of Fig. 1g) per mouse, naïve against expert, on NoGo (unwarranted) and Go (required) '
    'trials; paired Wilcoxon, n = 9. d, Per-animal learning curves over the six imaged dual-task sessions (thin lines, '
    'mice; bold, mean ± SEM): DPA accuracy on DPA trials and GNG accuracy on Go and NoGo trials (the per-session '
    'means of Fig. 1b). e, The propagation of Fig. 1g with trial-history covariates: odds ratio of a test lick given '
    'a delay lick (GEE, clustered by mouse; 95% CI) in the model of Fig. 1g and after adding, cumulatively, the trial '
    'type, the previous trial\'s outcome, its test lick and its trial type. The propagation does not depend on what '
    'happened on the previous trial.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
fig.savefig(f'{OUT}/png/ed_fig{ED_N}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig{ED_N}.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig{ED_N}.png')
