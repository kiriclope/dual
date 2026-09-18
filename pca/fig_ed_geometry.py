"""fig_ed_geometry.py — Extended Data Fig. 4: the geometry of the population state with no axes chosen
(Leon 2026-09-18, "use UMAP/t-SNE to characterize the geometry of the data" -> "these would look great in extended
data"). Companion to Fig. 2b-d: everything Fig. 2 reads with a decoder along a named axis is read here without any
label being used to find a direction, and the two agree.

  a  t-SNE of naive AND expert pseudo-trials together at mid-delay and at the decision (no trial reuse: each real
     trial of each mouse enters at most one pseudo-trial), coloured by task, by sample and by choice
  b  k-nearest-neighbour label purity in maps of this kind, averaged over 20 independent pseudo-trial draws, against
     paired label-shuffle nulls: the task organizes the map and nothing else comes close
  c  per-mouse manifold geometry at MATCHED trial counts (20 subsamples per mouse, common per-neuron scale): the
     manifold is no larger under distraction (participation ratio), but the sample is less separated on every dual
     set, and the Go/NoGo separation is what grows with learning
  d  UMAP of the four DPA condition-mean trajectories over the whole trial, naive and expert
  e  cross-validated unsupervised axes: the basis is fitted on one half of the trials and the trajectories projected
     from the other, so both the variance and the coding are held out. The first axis is the action axis
  f  the reliable variance those axes carry: one axis on DPA trials, two on dual trials, the sample axis a 2-3% third

Reads caches only: geometry_cache.pkl (fig_geometry_main.py), manifold_purity_draws.pkl (exp_manifold_purity_draws.py),
manifold_matched.pkl (exp_manifold_matched.py), unsup_axes_cv.pkl (fig_unsup_axes_cv.py).
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_ed_geometry.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig4.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.15      # 11.5-in canvas -> 183 mm is x0.63; PS keeps every literal at >= 5 pt in print
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
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MC = dict(zip(MICE, sns.color_palette('tab10', n_colors=len(MICE))))
STAGES = ['Naive', 'Expert']
ALL12 = [(t, s, te) for t in ['DPA', 'DualGo', 'DualNoGo'] for s in (0, 1) for te in (0, 1)]
TASKC = {'DPA': '#e8000b', 'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
SAMPC = {0: '#332288', 1: '#44AA99'}
MATCHC = {True: '#4daf4a', False: '#377eb8'}
GNGC = {'DualGo': '#023eff', 'DualNoGo': '#1ac938'}
SC = {'Naive': '0.55', 'Expert': '#332288'}
SHORT = {'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'}
WLAB = {'md': 'mid-delay', 'delay': 'late delay', 'decision': 'decision'}
NB = 84; T = np.arange(NB) / 6.0
EPOCHS = [(2.0, 3.0, '#332288', 'sample'), (4.5, 5.5, '#cc3311', 'GNG'), (6.5, 7.0, '#ee7733', 'cue'), (9.0, 10.0, '#377eb8', 'test')]
EVENTS = [(12, 'sample', '#332288'), (39, 'cue', '#ee7733'), (54, 'test', '#377eb8')]
DPA4 = [c for c in ALL12 if c[0] == 'DPA']
METH = 'tsne'                                              # the maps in a are t-SNE, so quote the t-SNE draw statistic

P = 'figures/pseudo/dimensionality'
G = pickle.load(open(f'{P}/geometry_cache.pkl', 'rb'))
DR = pickle.load(open(f'{P}/manifold_purity_draws.pkl', 'rb'))
MM = pickle.load(open(f'{P}/manifold_matched.pkl', 'rb'))
UA = pickle.load(open(f'{P}/unsup_axes_cv.pkl', 'rb'))
MR, MS, MCc = MM['R'], MM['S'], MM['C']


def plabel(ax, s, dx=-0.10, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


def mark(ax, x, y, p):
    ax.text(x, y, '∗' if p < .05 else 'n.s.', ha='center', va='bottom',
            fontsize=PS*(11 if p < .05 else 6.0), fontweight='bold', color='k' if p < .05 else '0.55')


fig = plt.figure(figsize=(11.5, 9.6))
outer = fig.add_gridspec(3, 24, height_ratios=[1.00, 0.92, 1.00], hspace=0.52, wspace=1.2,
                         left=0.045, right=0.985, top=0.965, bottom=0.05)

# ══ a — the maps ══════════════════════════════════════════════════════════════════════════════
gsa = outer[0, 0:15].subgridspec(1, 4, wspace=0.10)
COLS = [('md', 'task'), ('md', 'sample'), ('decision', 'task'), ('decision', 'match')]
axA = None
for c, (wn, nm) in enumerate(COLS):
    ax = fig.add_subplot(gsa[0, c]); axA = axA or ax
    Mp = G['maps'][wn]; E = Mp['E']; lab = Mp['labs'][nm]
    cmap = TASKC if nm == 'task' else SAMPC if nm == 'sample' else MATCHC
    ax.scatter(E[:, 0], E[:, 1], s=PS*11, c=[cmap[v] for v in lab], alpha=0.8, lw=0)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_title(f'{WLAB[wn]} · by {"choice" if nm == "match" else nm}', loc='left', fontsize=TITLE_FS)
    hs = ([mlines.Line2D([0], [0], marker='o', ls='none', ms=3.4, color=TASKC[t], label=SHORT[t]) for t in TASKC] if nm == 'task' else
          [mlines.Line2D([0], [0], marker='o', ls='none', ms=3.4, color=SAMPC[x], label=f'sample {"A" if x == 0 else "B"}') for x in SAMPC] if nm == 'sample' else
          [mlines.Line2D([0], [0], marker='o', ls='none', ms=3.4, color=MATCHC[x], label='match (lick)' if x else 'nonmatch (no lick)') for x in MATCHC])
    ax.legend(handles=hs, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3,
              handletextpad=0.12, columnspacing=0.55, borderaxespad=0.0, fontsize=PS*5.8)

# ══ b — draw-averaged kNN purity ═══════════════════════════════════════════════════════════════
axB = fig.add_subplot(outer[0, 17:24])
D = DR[(DR.set == 'all') & (DR.method == METH)]
NAMES = [('task', 'task'), ('sample', 'sample'), ('match', 'choice'), ('stage', 'stage')]
for i_, (nm, lb) in enumerate(NAMES):
    for k_, wn in enumerate(['md', 'decision']):
        r_ = D[(D.win == wn) & (D['var'] == nm)].iloc[0]; x = i_ * 2.3 + k_ * 0.88
        sig = r_.p < .05
        axB.bar(x, r_.purity, width=0.80, color=('0.45' if k_ == 0 else '#332288') if sig else ('0.82' if k_ == 0 else '#b9b2d6'))
        axB.errorbar(x, r_.purity, yerr=r_.sd_draws, color='k', lw=0.9, capsize=2, zorder=5)
        axB.plot([x - 0.40, x + 0.40], [r_.null, r_.null], color='#cc3311', lw=1.1, zorder=6)
        mark(axB, x, r_.purity + r_.sd_draws + 0.025, r_.p)
axB.set_xticks([i_ * 2.3 + 0.44 for i_ in range(len(NAMES))]); axB.set_xticklabels([lb for _, lb in NAMES], fontsize=PS*7)
axB.set_ylim(0, 1.15); axB.set_yticks([0, 0.5, 1.0]); axB.set_ylabel('kNN label purity')
axB.set_title('20 draws (grey mid-delay, indigo decision; red, shuffle null)', loc='left', fontsize=TITLE_FS)

# ══ c — per-mouse geometry at matched trial counts ═════════════════════════════════════════════
gsc = outer[1, 0:14].subgridspec(1, 3, wspace=0.62)
SETX = ['Go', 'NoGo', 'dual']
axC1 = fig.add_subplot(gsc[0, 0]); axC2 = fig.add_subplot(gsc[0, 1]); axC3 = fig.add_subplot(gsc[0, 2])
for ax, mt, ylab in [(axC1, 'PR', 'Δ participation ratio\n(set − DPA)'), (axC2, 'SEP_sample', 'Δ sample separation\n(set − DPA)')]:
    sub = MR[(MR.win == 'md')]
    D_, PV_ = {}, {}
    for i_, st in enumerate(SETX):
        for k_, stage in enumerate(STAGES):
            p_ = sub[sub.stage == stage].pivot_table(index='mouse', columns='set', values=mt).dropna()
            D_[(i_, k_)] = (p_[st] - p_['DPA']).reindex(MICE).values
            row = MCc[(MCc.a == 'DPA') & (MCc.b == st) & (MCc.win == 'md') & (MCc.stage == stage) & (MCc.metric == mt)]
            PV_[(i_, k_)] = float(row.iloc[0].p) if len(row) else np.nan
    lo = min(np.nanmin(v) for v in D_.values()); hi = max(np.nanmax(v) for v in D_.values()); rng = hi - lo + 1e-12
    for (i_, k_), d in D_.items():
        x = i_ * 2.3 + k_ * 0.88; stage = STAGES[k_]
        jit = x + np.random.RandomState(i_ * 3 + k_).uniform(-0.17, 0.17, len(d))
        if stage == 'Naive':
            ax.scatter(jit, d, s=PS*17, facecolors='none', edgecolors=[MC[m] for m in MICE], linewidths=0.9, zorder=3)
        else:
            ax.scatter(jit, d, s=PS*17, color=[MC[m] for m in MICE], alpha=0.9, lw=0, zorder=3)
        ax.plot([x - 0.36, x + 0.36], [np.median(d)] * 2, color='k', lw=1.5, zorder=4)
        mark(ax, x, hi + (0.07 if k_ == 0 else 0.21) * rng, PV_[(i_, k_)])
    ax.set_ylim(lo - 0.08 * rng, hi + 0.40 * rng)
    ax.axhline(0, ls=':', color='0.55', lw=0.9)
    ax.set_xticks([i_ * 2.3 + 0.44 for i_ in range(3)]); ax.set_xticklabels(SETX, fontsize=PS*7)
    ax.set_ylabel(ylab); ax.set_title('mid-delay, matched n', loc='left', fontsize=PS*7)
sub = MR[(MR.set == 'dual')]
for i_, wn in enumerate(['md', 'delay', 'decision']):
    p_ = sub[sub.win == wn].pivot_table(index='mouse', columns='stage', values='SEP_gng').dropna().reindex(MICE)
    for m in MICE:
        axC3.plot([i_ * 1.0 - 0.17, i_ * 1.0 + 0.17], [p_.loc[m, 'Naive'], p_.loc[m, 'Expert']], '-', color=MC[m], lw=0.8, alpha=0.8, zorder=3)
        axC3.scatter([i_ - 0.17, i_ + 0.17], [p_.loc[m, 'Naive'], p_.loc[m, 'Expert']], s=PS*14, color=MC[m], lw=0, zorder=4)
    row = MS[(MS.set == 'dual') & (MS.win == wn) & (MS.metric == 'SEP_gng')].iloc[0]
    mark(axC3, i_, p_.values.max() + 0.02, float(row.p))
axC3.set_xticks(range(3)); axC3.set_xticklabels([WLAB[w] for w in ['md', 'delay', 'decision']], fontsize=PS*6.6, rotation=20, ha='right')
axC3.set_ylabel('Go/NoGo separation'); axC3.set_title('dual trials, naïve → expert', loc='left', fontsize=PS*7)
axC1.legend(handles=[mlines.Line2D([0], [0], marker='o', ls='none', ms=3.6, mfc='none', mec='0.35', mew=0.9, label='naïve'),
                     mlines.Line2D([0], [0], marker='o', ls='none', ms=3.6, color='0.35', label='expert')],
            frameon=False, loc='lower left', ncol=2, handletextpad=0.15, columnspacing=0.6,
            borderaxespad=0.1, fontsize=PS*6.0)

# ══ e — UMAP trajectories on DPA trials ════════════════════════════════════════════════════════
gse = outer[1, 16:24].subgridspec(1, 2, wspace=0.10)
axE = None
for c, st in enumerate(STAGES):
    ax = fig.add_subplot(gse[0, c]); axE = axE or ax
    E = G['traj'][st]['E']
    for ci, cd in enumerate(DPA4):
        tr = E[ci]
        ax.plot(tr[:, 0], tr[:, 1], '-' if cd[1] == cd[2] else '--', color=SAMPC[cd[1]], lw=1.2, alpha=0.9, zorder=2)
        ax.scatter(tr[0, 0], tr[0, 1], s=PS*13, color=SAMPC[cd[1]], marker='o', lw=0, zorder=4)
        for b, nm, col in EVENTS:
            ax.scatter(tr[b, 0], tr[b, 1], s=PS*11, color=col, marker='s', lw=0, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_title(f'{"naïve" if st == "Naive" else "expert"} · DPA (UMAP)', loc='left', fontsize=TITLE_FS)
    if c == 0:
        ax.legend(handles=[mlines.Line2D([0], [0], color=SAMPC[x], label=f'sample {"A" if x == 0 else "B"}') for x in SAMPC] +
                          [mlines.Line2D([0], [0], color='0.4', ls='-', label='match'), mlines.Line2D([0], [0], color='0.4', ls='--', label='nonmatch')] +
                          [mlines.Line2D([0], [0], marker='s', ls='none', ms=3.2, color=col, label=nm) for _, nm, col in EVENTS],
                  frameon=False, loc='upper left', bbox_to_anchor=(0.0, -0.005), ncol=4, handletextpad=0.2,
                  columnspacing=0.6, fontsize=PS*5.5, borderaxespad=0.0)

# ══ d — cross-validated unsupervised axes ══════════════════════════════════════════════════════
gsd = outer[2, 0:17].subgridspec(1, 4, wspace=0.30)
CASES = [('DPA', 'Naive'), ('DPA', 'Expert'), ('dual', 'Naive'), ('dual', 'Expert')]
axD = None
for c, (sn, stage) in enumerate(CASES):
    ax = fig.add_subplot(gsd[0, c]); axD = axD or ax
    U = UA[(sn, stage)]; Y, Ysd, conds = U['Y'], U['Ysd'], U['conds']
    for t0, t1, col, _ in EPOCHS:
        ax.axvspan(t0, t1, color=col, alpha=0.10, lw=0)
    for ci, cd in enumerate(conds):
        ax.plot(T, Y[ci, :, 0], '-' if cd[1] == cd[2] else '--', color=SAMPC[cd[1]], lw=1.1, alpha=0.95, zorder=3)
        ax.fill_between(T, Y[ci, :, 0] - Ysd[ci, :, 0], Y[ci, :, 0] + Ysd[ci, :, 0], color=SAMPC[cd[1]], alpha=0.10, lw=0, zorder=2)
        if sn == 'dual':
            ax.scatter(T[-1], Y[ci, -1, 0], s=PS*34, color=GNGC[cd[0]], marker='*', edgecolors='k', linewidths=0.35, zorder=6)
    best = max(U['ident'], key=lambda nm: U['ident'][nm][0])
    ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.set_xlim(0, 14.2)
    ax.set_title(f'{sn} · {"naïve" if stage == "Naive" else "expert"} · PC1', loc='left', fontsize=TITLE_FS, pad=13)
    ax.text(0.0, 1.012, f'{100*U["rel"][0]:.0f}% reliable · codes {"Go/NoGo" if best == "gng" else best} (η² {U["ident"][best][0]:.2f})',
            transform=ax.transAxes, va='bottom', ha='left', fontsize=PS*6.0, color='0.3')
    ax.set_xlabel('time (s)')
    if c == 0:
        ax.set_ylabel('held-out projection (z)')
axDb = fig.add_subplot(outer[2, 19:24])
for c, (sn, stage) in enumerate(CASES):
    U = UA[(sn, stage)]
    for j in range(3):
        x = c * 3.5 + j * 0.9
        axDb.bar(x, 100 * U['rel'][j], width=0.80, color=SC[stage], alpha=1.0 if sn == 'DPA' else 0.55)
        nm = max(U['ident'], key=lambda k: U['ident'][k][j])
        lb = 'GNG' if nm == 'gng' else nm
        axDb.text(x, 100 * U['rel'][j] + 0.9, f'{lb} {U["ident"][nm][j]:.2f}', ha='left', va='bottom',
                  rotation=90, fontsize=PS*5.4, color='0.35')
axDb.set_xticks([c * 3.5 + 0.9 for c in range(4)])
axDb.set_xticklabels([f'{s}\n{"naïve" if st == "Naive" else "expert"}' for s, st in CASES], fontsize=PS*6.2)
axDb.set_ylim(0, 46); axDb.set_ylabel('reliable variance (%)')
axDb.set_title('PC1–PC3, held out (label above: best-coded factor, η²)', loc='left', fontsize=PS*6.6)

plabel(axA, 'a', dx=-0.06); plabel(axB, 'b', dx=-0.20); plabel(axC1, 'c', dx=-0.34)
plabel(axE, 'd', dx=-0.06); plabel(axD, 'e', dx=-0.26); plabel(axDb, 'f', dx=-0.26)

# ══ caption ════════════════════════════════════════════════════════════════════════════════════
def pu(wn, nm):
    r_ = D[(D.win == wn) & (D['var'] == nm)].iloc[0]; return r_.purity, r_.null, r_.p


def cst(a, b, wn, stage, mt):
    r_ = MCc[(MCc.a == a) & (MCc.b == b) & (MCc.win == wn) & (MCc.stage == stage) & (MCc.metric == mt)].iloc[0]
    return r_.va, r_.vb, r_.p, r_.nup


def dmed(mt, wn='md'):
    """Range of the six per-mouse median differences (Go/NoGo/dual x naive/expert) and the least significant p."""
    sub = MR[MR.win == wn]; ds, ps = [], []
    for st in SETX:
        for stage in STAGES:
            p_ = sub[sub.stage == stage].pivot_table(index='mouse', columns='set', values=mt).dropna()
            ds.append(float(np.median(p_[st] - p_['DPA'])))
            ps.append(float(MCc[(MCc.a == 'DPA') & (MCc.b == st) & (MCc.win == wn) & (MCc.stage == stage) & (MCc.metric == mt)].iloc[0].p))
    return min(ds), max(ds), min(ps)


def sst(sn, wn, mt):
    r_ = MS[(MS.set == sn) & (MS.win == wn) & (MS.metric == mt)].iloc[0]
    return r_.naive, r_.expert, r_.p, r_.nup


CAP = [
    'Extended Data Fig. 4 | The same geometry with no axes chosen (companion to Fig. 2b–d). Every panel of Fig. 2 '
    'reads the population along a direction found with the labels; here nothing does, and the two agree: the state '
    'space is organized by the task context, the action variables carry the variance, and the memory occupies a small '
    'reliable axis that a variance-ordered decomposition finds last. Pseudo-population of 3,319 neurons, correct '
    'laser-off trials of the twelve odor-defined conditions, mid-delay 5.5–6.5 s and decision 9.0–10.5 s.',
    'a, One t-SNE of naïve and expert pseudo-trials together at each window, coloured by task, by sample and by '
    f'choice ({G["maps"]["md"]["kp"][0]} naïve and {G["maps"]["md"]["kp"][1]} expert pseudo-trials per condition; each real trial enters at '
    'most one pseudo-trial, so neighbourhoods are trial-to-trial population states and not resampled duplicates; '
    'neurons scaled and centered per stage). Axes are arbitrary and between-cluster distances are not metric.',
    'b, k-nearest-neighbour label purity (k = 7) in maps of this kind, against paired label-shuffle nulls (red lines). '
    'A single embedding of ~150 pseudo-trials is itself a random variable — the same data give purity 0.48 on one draw '
    'and 0.61 on the next — so the statistic is the mean over 20 independent draws (bars) and its spread across them '
    f'(error bars, s.d.); ∗, p < .05 on the draw-averaged statistic. Mid-delay: task {pu("md","task")[0]:.2f}, sample {pu("md","sample")[0]:.2f}, choice '
    f'{pu("md","match")[0]:.2f}, stage {pu("md","stage")[0]:.2f}; decision: task {pu("decision","task")[0]:.2f}, sample {pu("decision","sample")[0]:.2f}, choice {pu("decision","match")[0]:.2f}, '
    f'stage {pu("decision","stage")[0]:.2f} (nulls 0.33 for task, 0.50 otherwise; UMAP agrees to within 0.02). The task context '
    f'stands {min(pu("md","task")[0] - pu("md","task")[1], pu("decision","task")[0] - pu("decision","task")[1]):.2f}–{max(pu("md","task")[0] - pu("md","task")[1], pu("decision","task")[0] - pu("decision","task")[1]):.2f} above its null; everything else that reaches significance does so by 0.01–0.08, so the memory '
    'and the choice are present in the geometry and far too small to shape it in two dimensions. Naïve and expert '
    f'pseudo-trials interleave (stage purity {pu("md","stage")[0] - pu("md","stage")[1]:.2f} and {pu("decision","stage")[0] - pu("decision","stage")[1]:.2f} above null), which is the unsupervised form of Fig. 2b–d.',
    'c, Per-mouse manifold geometry on each animal’s own simultaneously recorded neurons, at matched trial counts: '
    'every trial set is subsampled to the smallest set’s n in that mouse, stage and window (20 draws, medians over '
    'draws), and all sets share one condition-agnostic per-neuron scale, so neither the count nor the scaling can '
    'carry a difference (Methods). Left and middle, each set minus that mouse’s DPA value at mid-delay (points, mice, '
    'in the colours used throughout; open, naïve; filled, expert; line, median; paired Wilcoxon, two-sided, n = 9). The '
    'participation ratio of the trial cloud is the same on Go, NoGo and dual sets as on DPA '
    f'(Δ medians {dmed("PR")[0]:+.1f} to {dmed("PR")[1]:+.1f} over the six comparisons, smallest p = {dmed("PR")[2]:.2f}): composing the two tasks does not enlarge '
    'the manifold, the unsupervised counterpart of Fig. 2b. The sample is nonetheless less separated on every dual '
    f'set — Go {cst("DPA","Go","md","Naive","SEP_sample")[0]:.2f} → {cst("DPA","Go","md","Naive","SEP_sample")[1]:.2f} naïve (p = {cst("DPA","Go","md","Naive","SEP_sample")[2]:.3f}) and {cst("DPA","Go","md","Expert","SEP_sample")[0]:.2f} → {cst("DPA","Go","md","Expert","SEP_sample")[1]:.2f} expert (p = {cst("DPA","Go","md","Expert","SEP_sample")[2]:.3f}), '
    f'NoGo p = {cst("DPA","NoGo","md","Naive","SEP_sample")[2]:.3f} and {cst("DPA","NoGo","md","Expert","SEP_sample")[2]:.3f}, the eight-condition dual set p = {cst("DPA","dual","md","Naive","SEP_sample")[2]:.3f} and {cst("DPA","dual","md","Expert","SEP_sample")[2]:.3f} — with 1–2 of 9 mice '
    'moving against the effect. Separation is the distance between a variable’s two condition centroids divided by '
    'the mean within-condition spread, a decoder-free and unit-free index. The Go-only and NoGo-only sets have the '
    'DPA design exactly (four conditions, sample × test), so the cost is distraction and not the extra conditions of '
    f'the pooled dual set. Right, the Go/NoGo separation on dual trials is what learning increases, at all three '
    f'windows (mid-delay {sst("dual","md","SEP_gng")[0]:.2f} → {sst("dual","md","SEP_gng")[1]:.2f}, p = {sst("dual","md","SEP_gng")[2]:.3f}; late delay {sst("dual","delay","SEP_gng")[0]:.2f} → {sst("dual","delay","SEP_gng")[1]:.2f}, p = {sst("dual","delay","SEP_gng")[2]:.3f}; '
    f'decision {sst("dual","decision","SEP_gng")[0]:.2f} → {sst("dual","decision","SEP_gng")[1]:.2f}, p = {sst("dual","decision","SEP_gng")[2]:.3f}; {sst("dual","delay","SEP_gng")[3]}/9 mice up at the late delay), while the sample separation does '
    f'not change detectably in any set (all p ≥ .16). Lines join the same mouse, naïve on the left of each pair.',
    'd, The same states with time: UMAP of the four DPA condition-mean trajectories over the whole trial, after '
    'removing the component shared by all conditions (5-bin smoothing; n_neighbors 120, min_dist 0.05). Dots, trial '
    'start; squares, sample, cue and test onsets. The four conditions leave a common state and separate by sample, in '
    'the same arrangement before and after learning. Two properties of the method set the settings and the trial set: '
    'n_neighbors must exceed the number of time points a condition spends near itself or the graph splits into one '
    'filament per condition, and on dual trials the Go/NoGo code dominates so strongly that the map is eight filaments '
    'at every setting (Methods). Distances between arms are not metric.',
    'e, Unsupervised axes with the variance and the coding both held out: for each of six random half-splits and in '
    'both directions the principal axes of the per-bin condition means are fitted on one half of the trials, with the '
    'labels never used, and the other half’s condition means are projected on them (axes matched to the full-data '
    'reference by Hungarian assignment on |cos| and sign-aligned; band, s.d. over the twelve fits). Colour, sample; '
    'solid, match; dashed, nonmatch; star, Go or NoGo at the trial end. The first axis is the action axis in every '
    f'case: on DPA trials it codes the choice (η² {UA[("DPA","Naive")]["ident"]["choice"][0]:.2f} naïve, {UA[("DPA","Expert")]["ident"]["choice"][0]:.2f} expert, read at the decision window) and on dual '
    f'trials the Go/NoGo odor (η² {UA[("dual","Naive")]["ident"]["gng"][0]:.2f} and {UA[("dual","Expert")]["ident"]["gng"][0]:.2f}). η² is the share of a component’s held-out between-condition '
    'variance explained by one factor at that factor’s window, 1 meaning the component codes that factor alone.',
    'f, Reliable variance of the first three held-out axes (the cvPCA cross-term, fit-half score × held-out score, '
    'over the held-out total), with the factor each axis codes best and its η² above the bar. One axis carries the '
    f'reliable variance on DPA trials ({100*UA[("DPA","Expert")]["rel"][0]:.0f}% expert) and two on dual trials ({100*UA[("dual","Expert")]["rel"][0]:.0f}% and {100*UA[("dual","Expert")]["rel"][1]:.0f}%, both Go/NoGo). The sample '
    f'axis is present and almost pure but small: on DPA trials it is PC2–PC3 with η² '
    f'{min(UA[("DPA",st)]["ident"]["sample"][j] for st in STAGES for j in (1, 2)):.2f}–{max(UA[("DPA",st)]["ident"]["sample"][j] for st in STAGES for j in (1, 2)):.2f} and '
    f'{min(100*UA[("DPA",st)]["rel"][j] for st in STAGES for j in (1, 2)):.0f}–{max(100*UA[("DPA",st)]["rel"][j] for st in STAGES for j in (1, 2)):.0f}% of the '
    'reliable variance. A variance-ordered decomposition therefore finds the action variables first and the memory '
    'last, which is why the memory is read here with a decoder along its own axis rather than with PCA. Without '
    'cross-validation the second and third components look four to eight times larger and appear coded; out of sample '
    'they carry neither.',
]
if '--printcap' in sys.argv[1:]:                                # keeps the results_draft.md legend in step with CAP
    print('\n\n'.join(CAP))
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
for sub_ in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub_}', exist_ok=True)
fig.savefig(f'{OUT}/png/ed_fig4.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/ed_fig4.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig4.png')
