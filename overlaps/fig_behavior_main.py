"""
fig_behavior_main.py — behavioural MAIN figure (recorded cohort, 9 mice, laser OFF).

Combines the full behaviour_learning panel set with the new mechanistic panels.

  A  Task scheme (DPA + GNG trial structure; dual_task_scheme.svg).
  ── learning curves (per-mouse/day accuracy, mean ± SEM; LMM stats) ──
  B  DPA vs GNG performance vs session.
  C  GNG: Go vs NoGo distractor.
  D  DPA: paired vs unpaired.
  E  DPA unpaired, by task context (DPA / Go / NoGo).
  F  LMM fixed-effect coefficients (condition + condition×day, 95% CI).
  ── new panels ──
  G  Interference: DPA accuracy by condition (pure → NoGo → Go, ordered by lick
     rate); intrusive NoGo CUE lick propagates to the test lick (GEE, Naive ∗; FA route; rebuilt 2026-09-01).
  H  Suboptimal balance: per-animal DPA vs GNG (Expert); none reaches both-optimal.

Panels B–F reproduce fig_behavior_learning.py (helpers copied per repo convention,
so that script stays untouched). Definitions: DPA perf = `performance` on tasks==DPA;
GNG perf = `odr_perf` on Dual; pair==1 paired / 0 unpaired; one row/trial (target==choice).
LMM: perf ~ C(cond)*day_centred + (1|mouse) on per-mouse/day proportions (mildly
anti-conservative — RE variance near boundary). Per-day stars = per-day LMM, uncorrected.

Output: figures/overlaps/behavior/{png,svg}/behavior_main.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_main.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle, subprocess, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

sns.set_style("ticks")
PS = 1.45      # print-scale: typography sized for 183 mm reproduction (2026-09-02)
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})

RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'
GREY = '#555555'
STAGE_SHADE = '#332288'
N_MIN = 4

ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MICE = ALL_MICE
pal_mice = sns.color_palette('tab10', n_colors=len(ALL_MICE))
MOUSE_COLOR = {m: pal_mice[i] for i, m in enumerate(ALL_MICE)}
GROUP = {**{m: 'Jaws' for m in ALL_MICE[:5]}, **{m: 'ChR' for m in ALL_MICE[5:7]},
         **{m: 'ACC' for m in ALL_MICE[7:]}}
GMARKER = {'Jaws': 'o', 'ChR': '^', 'ACC': 's'}

LAB = '../data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl'
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'choice'].copy() if 'target' in y.columns else y.copy()
d = d[(d.laser == 0) & (d.mouse.isin(MICE))]
DAYS = list(range(1, int(d.day.max()) + 1))

OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)
os.makedirs(f'{OUT}/assets', exist_ok=True)
SCHEME = f'{OUT}/assets/dual_task_scheme.png'
if not os.path.exists(SCHEME):
    subprocess.run(['rsvg-convert', '-w', '2600', '/home/leon/dual/dual_task_scheme.svg',
                    '-o', SCHEME], check=True)
TRAIN = f'{OUT}/assets/dual_training_scheme.png'
subprocess.run(['rsvg-convert', '-w', '1600', '/home/leon/dual/dual_training_scheme_vector.svg',
                '-o', TRAIN], check=True)
# Panel-A cartoon: continuous-line B&W vector traced from the original ~/dual/mouse.svg,
# flipped to face the task. Regenerate the SVG with `make_mouse_lineart.py`.
MOUSE = f'{OUT}/assets/mouse_lineart.png'
subprocess.run(['rsvg-convert', '-w', '1800', '/home/leon/dual/mouse_lineart.svg', '-o', MOUSE], check=True)


def show_scheme(ax, path, blank_tl=None):
    """imshow a scheme PNG with surrounding whitespace trimmed. blank_tl=(fh,fw)
    whites out a top-left corner (used to drop the task SVG's built-in 'A')."""
    im = mpimg.imread(path)
    g = im[..., :3].mean(-1); mk = g < 0.985
    r = np.where(mk.any(1))[0]; c = np.where(mk.any(0))[0]
    im = im[r.min():r.max() + 1, c.min():c.max() + 1]
    if blank_tl is not None:
        im = im.copy(); fh, fw = blank_tl
        im[:int(im.shape[0] * fh), :int(im.shape[1] * fw)] = 1.0
    ax.imshow(im); ax.axis('off')

# ── learning-figure helpers (copied from fig_behavior_learning.py) ────────────
IS_DPA = d.tasks == 'DPA'
IS_GO = d.tasks == 'DualGo'
IS_NOGO = d.tasks == 'DualNoGo'
IS_DUAL = IS_GO | IS_NOGO
UNP = d.pair == 0
CORR_DPA = d.performance
CORR_GNG = d.odr_perf
CORR_A = np.where(IS_DPA, d.performance, d.odr_perf)
COND_A = np.where(IS_DPA, 'DPA', 'GNG')
COND_PAIR = np.where(d.pair == 1, 'paired', 'unpaired')
COND_TASK = d.tasks.map({'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'})


def per_mouse_day(col, mask):
    sel = d[mask]
    out = {day: {} for day in DAYS}
    for day in DAYS:
        for m in MICE:
            v = sel[(sel.mouse == m) & (sel.day == day)][col].dropna()
            if len(v):
                out[day][m] = float(v.mean())
    return out


def agg(pmd):
    mean, sem = [], []
    for day in DAYS:
        pm = np.array(list(pmd[day].values()))
        if len(pm):
            mean.append(pm.mean())
            sem.append(pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else 0.0)
        else:
            mean.append(np.nan); sem.append(np.nan)
    return np.array(DAYS, float), np.array(mean), np.array(sem)


def plot_line(ax, pmd, color, label, ls='-'):
    x, m, s = agg(pmd)
    ok = ~np.isnan(m)
    ax.plot(x[ok], m[ok], ls=ls, color=color, lw=1.3, marker='o', ms=3.5,
            mfc=color, mec=color, label=label, zorder=3)
    ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def _prop(mask, cond, correct, by_day=True):
    df = pd.DataFrame({
        'correct': np.asarray(correct)[mask.values],
        'cond': np.asarray(cond)[mask.values],
        'mouse': d.loc[mask, 'mouse'].values,
        'day': d.loc[mask, 'day'].values,
    }).dropna()
    keys = ['mouse', 'day', 'cond'] if by_day else ['mouse', 'cond']
    return df.groupby(keys, observed=True).correct.mean().reset_index(name='perf')


def lmm(mask, cond, correct, ref, tag):
    g = _prop(mask, cond, correct, by_day=True)
    g['dayc'] = g['day'] - g['day'].mean()
    res = smf.mixedlm(f"perf ~ C(cond, Treatment('{ref}'))*dayc", g, groups=g['mouse']).fit()
    ci, pv, pr = res.conf_int(), res.pvalues, res.params
    recs = []
    for name in pr.index:
        if not name.startswith('C(cond'):
            continue
        lvl = name.split('[T.')[1].split(']')[0]
        recs.append(dict(model=tag, contrast=f'{lvl}−{ref}',
                         kind='cond×day' if ':dayc' in name else 'condition',
                         beta=float(pr[name]), lo=float(ci.loc[name, 0]),
                         hi=float(ci.loc[name, 1]), p=float(pv[name])))
    return recs


def perday_stars(ax, mask, cond, correct, ref, y_star=1.03):
    for day in DAYS:
        sub = mask & (d.day == day)
        if d.loc[sub, 'mouse'].nunique() < N_MIN:
            continue
        g = _prop(sub, cond, correct, by_day=False)
        try:
            r = smf.mixedlm(f"perf ~ C(cond, Treatment('{ref}'))", g, groups=g['mouse']).fit()
            wt = r.wald_test_terms(scalar=True).table
            ct = [i for i in wt.index if i.startswith('C(cond')][0]
            p = float(wt.loc[ct, 'pvalue'])
            if star(p):
                ax.text(day, y_star, star(p), ha='center', va='top', fontsize=PS*8,
                        fontweight='bold', color='k')
        except Exception:
            pass


# ── figure scaffold — publication grid: A full · B-E learning · F-H summary ────
fig = plt.figure(figsize=(14, 12.4))
gs = GridSpec(3, 12, height_ratios=[1.42, 1.0, 1.0], hspace=0.4, wspace=1.15,
              left=0.055, right=0.99, top=0.975, bottom=0.06)
TITLE_FS = PS*9.5              # <=7 pt at 183 mm final size (NN artwork guide)


def panel_letter(ax, L, dx=0.020, dy=0.016):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, L.lower(), fontsize=PS*11, fontweight='bold', va='top', ha='left')


# ── A: setup cartoon + task scheme + curriculum training ──────────────────────
axAm = fig.add_subplot(gs[0, 0:2])
show_scheme(axAm, MOUSE)
axA = fig.add_subplot(gs[0, 2:8])
show_scheme(axA, SCHEME, blank_tl=(0.16, 0.085))
axAt = fig.add_subplot(gs[0, 8:12])
show_scheme(axAt, TRAIN)
axAt.set_title('Curriculum training', loc='center', fontsize=TITLE_FS)

# ── B–E: learning curves ──────────────────────────────────────────────────────
axB = fig.add_subplot(gs[1, 0:3])
axC = fig.add_subplot(gs[1, 3:6])
axD = fig.add_subplot(gs[1, 6:9])
axE = fig.add_subplot(gs[1, 9:12])
axF = fig.add_subplot(gs[2, 0:4])
betas = []

plot_line(axB, per_mouse_day('performance', IS_DPA), RED, 'DPA')
plot_line(axB, per_mouse_day('odr_perf', IS_DUAL), BLUE, 'GNG')
betas += lmm(IS_DPA | IS_DUAL, COND_A, CORR_A, 'DPA', 'DPAvGNG')
perday_stars(axB, IS_DPA | IS_DUAL, COND_A, CORR_A, 'DPA')
axB.set_title('Both tasks are learned', loc='left', fontsize=TITLE_FS)

plot_line(axC, per_mouse_day('odr_perf', IS_GO), BLUE, 'Go')
plot_line(axC, per_mouse_day('odr_perf', IS_NOGO), GREEN, 'NoGo')
betas += lmm(IS_DUAL, COND_TASK, CORR_GNG, 'Go', 'GovNoGo')
perday_stars(axC, IS_DUAL, COND_TASK, CORR_GNG, 'Go')
axC.set_title('NoGo drives the GNG gain', loc='left', fontsize=TITLE_FS)

plot_line(axD, per_mouse_day('performance', IS_DPA & (d.pair == 1)), RED, 'paired', ls='-')
plot_line(axD, per_mouse_day('performance', IS_DPA & UNP), RED, 'unpaired', ls='--')
betas += lmm(IS_DPA, COND_PAIR, CORR_DPA, 'paired', 'pairedVunp')
perday_stars(axD, IS_DPA, COND_PAIR, CORR_DPA, 'paired')
axD.set_title('Unpaired trials carry learning', loc='left', fontsize=TITLE_FS)

plot_line(axE, per_mouse_day('performance', UNP & IS_DPA), RED, 'DPA only', ls='--')
plot_line(axE, per_mouse_day('performance', UNP & IS_GO), BLUE, 'Go', ls='--')
plot_line(axE, per_mouse_day('performance', UNP & IS_NOGO), GREEN, 'NoGo', ls='--')
betas += lmm(UNP, COND_TASK, CORR_DPA, 'DPA', 'unpByTask')
perday_stars(axE, UNP, COND_TASK, CORR_DPA, 'DPA')
axE.set_title('Go context lowers unpaired DPA', loc='left', fontsize=TITLE_FS)

for ax in (axB, axC, axD, axE):
    ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_ylim(0.18, 1.07); ax.set_xticks(DAYS); ax.set_xlabel('session')
    ax.legend(frameon=False, fontsize=PS*6.5, loc='lower right')
axB.set_ylabel('performance')

# no-lick thread — one subtle line tying B–E to the neural no-lick push

# ── F: LMM coefficient forest ─────────────────────────────────────────────────
cond_recs = [r for r in betas if r['kind'] == 'condition']
int_map = {(r['model'], r['contrast']): r for r in betas if r['kind'] == 'cond×day'}
xlabels = []
for i, c in enumerate(cond_recs):
    it = int_map[(c['model'], c['contrast'])]
    cc = 'k' if c['p'] < 0.05 else '0.65'
    ci_ = BLUE if it['p'] < 0.05 else '0.65'
    axF.errorbar(i - 0.16, c['beta'], yerr=[[c['beta'] - c['lo']], [c['hi'] - c['beta']]],
                 fmt='o', color=cc, ms=6, capsize=3, lw=1.5, zorder=3)
    axF.errorbar(i + 0.16, it['beta'], yerr=[[it['beta'] - it['lo']], [it['hi'] - it['beta']]],
                 fmt='s', mfc='white', mec=ci_, color=ci_, ms=5, capsize=3, lw=1.3, zorder=3)
    for r, dx in ((c, -0.16), (it, 0.16)):
        if star(r['p']):
            axF.text(i + dx, r['hi'] + 0.06, star(r['p']), ha='center', va='bottom',
                     fontsize=PS*12, fontweight='bold')
    short = c['contrast'].replace('unpaired', 'unp').replace('paired', 'pair')
    xlabels.append((i, short))
axF.axhline(0, ls='--', color='0.4', lw=1)
axF.set_xticks([x for x, _ in xlabels])
axF.set_xticklabels([lab for _, lab in xlabels], rotation=30, ha='right', fontsize=PS*8)
axF.set_xlim(-0.6, len(cond_recs) - 0.4)
axF.set_ylabel('LMM β (Δ performance)')
axF.set_title('Effect-size summary (LMM)', loc='left', fontsize=TITLE_FS)
axF.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=5, label='condition'),
                    mlines.Line2D([0], [0], marker='s', color=BLUE, mfc='white', ls='none', ms=5, label='condition×day')],
           frameon=False, fontsize=PS*6.5, loc='upper right')

# ── G: the intrusive CUE lick propagates to the test lick — NoGo trials ─────────
#   REBUILT 2026-09-01 (predictor audit): the old build split DPA performance by
#   `licks` = (cue OR test lick), which coincides with the test lick on 96.5% of
#   NoGo trials — near-circular with performance. The intended claim (the delay
#   lick leads to FALSE ALARMS, the chain the Fig-4 push suppresses) is carried by
#   the clean variables: predictor = the CUE lick (odr_choice, the genuine
#   intrusive delay lick), outcome = P(lick at test). Naive: cue lick triples the
#   odds of licking again at test (propagation -> FA on unpaired trials); Expert:
#   propagation gone, and the cue licks themselves largely vanish.
axG = fig.add_subplot(gs[2, 4:8])
ng = d[d.tasks == 'DualNoGo'].copy()
ng['licked'] = (pd.to_numeric(ng.odr_choice, errors='coerce') > 0).astype(float)
ng['testlick'] = (pd.to_numeric(ng.choice, errors='coerce') > 0).astype(float)
NOLICK_C, LICK_C = '#888888', '#1f77b4'
STAGE_X = {'Naive': (0.0, 1.0), 'Expert': (2.4, 3.4)}
for stage, (x0, x1) in STAGE_X.items():
    s = ng[ng.stage == stage].dropna(subset=['licked', 'testlick'])
    nl, lk = [], []
    for m in ALL_MICE:
        a = s[(s.mouse == m) & (s.licked == 0)].testlick
        b = s[(s.mouse == m) & (s.licked == 1)].testlick
        av, bv = (a.mean() if len(a) else np.nan), (b.mean() if len(b) >= 3 else np.nan)
        nl.append(av); lk.append(bv)
        if np.isfinite(av) and np.isfinite(bv):
            axG.plot([x0, x1], [av, bv], '-', color=MOUSE_COLOR[m], lw=0.8, alpha=0.45, zorder=2)
    nl, lk = np.array(nl), np.array(lk)
    for xx, vals, col in [(x0, nl, NOLICK_C), (x1, lk, LICK_C)]:
        mn = np.nanmean(vals); se = np.nanstd(vals, ddof=1) / np.sqrt(np.isfinite(vals).sum())
        axG.errorbar(xx, mn, yerr=se, fmt='o', color=col, ms=6, capsize=2.5, lw=1.3,
                     zorder=5, mec='k', mew=0.8)
    axG.plot([x0, x1], [np.nanmean(nl), np.nanmean(lk)], '-', color='0.3', lw=1.6, zorder=4)
    g = smf.gee('testlick ~ licked', groups=s['mouse'], data=s,
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    orr, pv = np.exp(g.params['licked']), g.pvalues['licked']
    # FA arm (unpaired trials only: the second lick IS the false alarm) -> stdout/caption
    su = s[s.odor_pair.isin([1, 3])]
    gf = smf.gee('testlick ~ licked', groups=su['mouse'], data=su,
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    # pairing-independence: propagation x pairing interaction (n.s. -> pooled OR covers the FA arm)
    si = s.assign(unp=s.odor_pair.isin([1, 3]).astype(int))
    gi = smf.gee('testlick ~ licked * unp', groups=si['mouse'], data=si,
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    print(f'G {stage}: propagation OR={orr:.2f} p={pv:.4f} | FA arm (unpaired) '
          f'OR={np.exp(gf.params["licked"]):.2f} p={gf.pvalues["licked"]:.4f} | '
          f'interaction lick x pairing p={gi.pvalues["licked:unp"]:.3f} | '
          f'cue-lick rate={s.licked.mean():.2f}')
    # significance bracket + star / ns
    ybr = 0.99
    axG.plot([x0, x0, x1, x1], [ybr - 0.012, ybr, ybr, ybr - 0.012], color='k', lw=1.3, zorder=6)
    st = star(pv) or 'ns'
    axG.text((x0 + x1) / 2, ybr + 0.003, st, ha='center', va='bottom',
             fontsize=PS*12 if star(pv) else 8, fontweight='bold', color='k')
    axG.text((x0 + x1) / 2, 1.02, stage, ha='center', va='bottom',
             transform=axG.get_xaxis_transform(), clip_on=False, fontsize=PS*8,
             fontweight='bold', color=STAGE_SHADE if stage == 'Expert' else '0.4')
    axG.text((x0 + x1) / 2, 0.04, f'OR={orr:.2f}\np={pv:.3f}',
             ha='center', va='bottom', fontsize=PS*8, color='0.3')
axG.set_xticks([0, 1, 2.4, 3.4]); axG.set_xticklabels(['no cue\nlick', 'cue\nlick', 'no cue\nlick', 'cue\nlick'])
axG.set_xlim(-0.5, 3.9); axG.set_ylim(0.0, 1.03); axG.set_ylabel('P(lick at test)')
# (legend dropped at print scale — the marker colours restate the x-axis categories)
axG.set_title('The intrusive lick propagates to the test', loc='left',
              fontsize=TITLE_FS, pad=24)   # raised above the Naive/Expert stage headers

# ── H: suboptimal expert balance — DPA vs GNG per animal ──────────────────────
axH = fig.add_subplot(gs[2, 8:12])
de = d[d.stage == 'Expert']
xd = {m: de[de.tasks == 'DPA'].loc[de.mouse == m, 'performance'].mean() for m in ALL_MICE}
yd = {m: de[de.tasks != 'DPA'].loc[de.mouse == m, 'odr_perf'].mean() for m in ALL_MICE}
lim = (0.66, 1.0)
axH.plot(lim, lim, ls='--', color='0.7', lw=0.9, zorder=1)
axH.scatter(0.99, 0.99, marker='*', s=120, color='#E8A100', edgecolor='k', linewidths=0.6, zorder=6)
axH.text(0.985, 0.955, 'optimal', ha='right', va='top', fontsize=PS*8, color='#7a5600')
for m in ALL_MICE:
    axH.scatter(xd[m], yd[m], marker=GMARKER[GROUP[m]], s=34, color=MOUSE_COLOR[m],
                edgecolors='w', linewidths=0.6, zorder=5)
# stats: across-animal DPA↔GNG correlation + mean gap below the optimal corner
xs = np.array([xd[m] for m in ALL_MICE]); ys = np.array([yd[m] for m in ALL_MICE])
ok = np.isfinite(xs) & np.isfinite(ys)
r_p, p_p = pearsonr(xs[ok], ys[ok]); r_s, p_s = spearmanr(xs[ok], ys[ok])
gap = np.mean(1.0 - np.minimum(xs[ok], ys[ok]))          # mean shortfall from ceiling
axH.text(0.035, 0.965, f'r={r_p:+.2f} p={p_p:.2f}\nρ={r_s:+.2f} p={p_s:.2f}  (n={ok.sum()})\n'
         f'gap to optimal: {gap:.2f}', transform=axH.transAxes, ha='left', va='top',
         fontsize=PS*6.5, color='0.3')
axH.set_xlim(lim); axH.set_ylim(lim); axH.set_aspect('equal')
axH.set_xlabel('DPA performance'); axH.set_ylabel('GNG performance')
axH.set_title('Experts reach a suboptimal balance', loc='left', fontsize=TITLE_FS)
axH.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=5, label='Jaws'),
                    mlines.Line2D([0], [0], marker='^', color='k', ls='none', ms=7, label='ChR'),
                    mlines.Line2D([0], [0], marker='s', color='k', ls='none', ms=7, label='ACC')],
           frameon=False, fontsize=PS*6.5, loc='lower left', handletextpad=0.2)

# ── panel letters (a on the setup cartoon, then b–h) ──────────────────────────
# bottom-row letters get extra clearance: f's significance stars, g's stage headers and h's
# tick labels reach into the default letter zone at print scale (user report 2026-09-03)
for _ax, _L, _dy in [(axAm, 'A', 0.016), (axB, 'B', 0.016), (axC, 'C', 0.016),
                     (axD, 'D', 0.016), (axE, 'E', 0.016),
                     (axF, 'F', 0.034), (axG, 'G', 0.034), (axH, 'H', 0.034)]:
    panel_letter(_ax, _L, dx=0.030, dy=_dy)

# ── CAPTION (justified, drawn below — same mechanism as Figs 2/3; replaces the old footnote) ──
CAP_PARAS = [
    'Figure 1 | Composing working memory with an embedded action is costly: the intruding lick, '
    'not the distractor odour, interferes — and it propagates to the memory response. Recorded '
    'cohort, nine mice, laser-off trials; curves show mean ± SEM across mice; ∗ p < .05, ∗∗ p < '
    '.01, ∗∗∗ p < .001 (per-day linear mixed models, uncorrected; day 6 n = 4).',
    'a, Task design. Each trial is a delayed paired-association (DPA) problem: a sample odour (A '
    'or B), a 6-s delay, then a test odour (C or D); the mouse licks if the pair matches (A→C, '
    'B→D) and withholds otherwise — the sample must be held in working memory across the delay. '
    'On two thirds of trials a Go/NoGo (GNG) discrimination is embedded inside that delay — a '
    'distractor odour, then a response cue: lick on Go, withhold on NoGo (DualGo / DualNoGo '
    'trials); the remainder are pure DPA. All three trial types are interleaved within every '
    'session, so the memory must survive both the distractor odour and the act of responding to '
    'it. Right: the training curriculum.',
    'b–e, Learning curves (per-mouse/day accuracy). b, DPA vs GNG performance. c, GNG split by '
    'distractor identity. d, DPA paired vs unpaired trials. e, DPA unpaired trials by surrounding '
    'task context. What is hard is not licking to the right odour, but not licking while a lick- '
    'demanding task runs through the middle of the memory period.',
    'f, Linear mixed model over panels b–e (fixed effects ± 95% CI; filled circles, condition '
    'offset; open squares, condition × day slope; random intercept per mouse): GNG−DPA β = +0.037 '
    '(p = 0.045), narrowing over days; NoGo−Go +0.072 (p = 0.034); unpaired−paired −0.185 (p < '
    '10⁻⁴), narrowing; Go−DPA −0.073 (p = 0.038).',
    'g, Where the interference acts — the intruding lick propagates through the response: '
    'probability of licking at the DPA test on NoGo trials, split by whether the animal licked at '
    'the distractor cue (thin lines, single mice). In naïve mice a cue lick tripled the odds of '
    'licking again at test (trial-level GEE, OR = 3.10, p = .006); the propagation is pairing- '
    'independent (lick × pairing interaction p = .61) — on unpaired trials the second lick is the '
    'false alarm (OR = 2.7, p = .09), on paired trials a hit (OR = 9.9, p = .001). In expert mice '
    'the propagation is absent (OR = 1.50, p = .42) and cue licks are rare (rate 0.24 → 0.08): '
    'this is the lick chain that the no-lick repositioning of Fig. 4 closes.',
    'h, Learned, but not jointly optimal: expert DPA vs GNG accuracy per animal (colour, mouse; '
    'marker, opsin group; star, the both-optimal corner). No animal reaches the corner (mean gap '
    '0.18) and the two accuracies are uncorrelated (r = +0.10, p = .80; ρ = +0.35, p = .36; n = '
    '9) — each animal settles its own balance. The problem the population geometry must solve is '
    'precise: shield the memory from the very action the animal is required to produce.',
]
sys.path.insert(0, '/home/leon/dual/pca')
from figcaption import draw_justified                  # shared with Figs 2/3
if '--nocap' not in sys.argv[1:]:   # submission build: legend goes below the figure
    draw_justified(fig, CAP_PARAS, fontsize=PS*7.2)

for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/behavior_main.{ext}'
    fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
plt.close(fig)
