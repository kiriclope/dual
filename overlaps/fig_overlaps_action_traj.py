"""Action/lick code TRAJECTORIES on shared axes (draft for the main figure).

The GNG decoder is a Go/NoGo lick command; the DPA choice decoder is a match→lick command. Each is a
lick readout for its OWN task. Project BOTH tasks onto EACH single axis over time (generalizing tensor:
fix the train window at that task's ACTION/REWARD window, read the decision value across test-time):

  GNG lick axis   (trained on Dual Go vs NoGo at the GNG action window):
      GNG Go(lick) vs NoGo [within]  and  DPA lick vs no-lick [cross-projected]
  DPA choice axis (trained on DPA lick vs no-lick at the DPA response window):
      DPA lick vs no-lick [within]  and  GNG Go(lick) vs NoGo [cross-projected]

If each axis also separates the OTHER task's lick — each condition rising at its own task's action
moment — the lick command is ONE shared action code, engaged whenever the task calls for the lick.

Output: figures/overlaps/action/{png,svg}/overlaps_action_traj.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.pca.io import pkl_load

matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')

DATA = '../data/overlaps'
BDUM = 'log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
BL = np.arange(0, 12)
GNG_ACT = np.arange(42, 54)                 # GNG action/reward window (gngRwd + lDel) — the GNG lick axis
DPA_ACT = np.arange(60, 72)                 # DPA response window — the DPA lick axis
xt = np.linspace(0, 14, 84)

y = pkl_load(f'labels_{BDUM}', path=DATA)
X = np.asarray(pkl_load(f'X_{BDUM}', path=DATA))


def axis_rows(target, train):
    m = (y.target == target).to_numpy()
    D = X[m][:, 1, train, :].mean(1).astype(float)   # (n_rows, 84_test) decision on the fixed axis
    return D, y[m].reset_index(drop=True)


# ── per-axis group definitions: (label, row-mask fn, colour, lick?) ────────────────────────────
def groups_gng(yy):
    dual = yy.tasks.to_numpy() != 'DPA'
    dpa = yy.tasks.to_numpy() == 'DPA'
    gng = yy.gng.to_numpy(); ch = yy.choice.to_numpy()
    return {
        'GNG Go (lick)': (dual & (gng == 1), '#1f77b4', True),
        'GNG NoGo':      (dual & (gng == 0), '#2ca02c', False),
        'DPA lick':      (dpa & (ch == 1),   '#d62728', True),
        'DPA no-lick':   (dpa & (ch == 0),   '#9467bd', False),
    }


def groups_choice(yy):
    tsk = yy.tasks.to_numpy(); ch = yy.choice.to_numpy()
    return {
        'GNG Go (lick)': (tsk == 'DualGo',   '#1f77b4', True),
        'GNG NoGo':      (tsk == 'DualNoGo', '#2ca02c', False),
        'DPA lick':      ((tsk == 'DPA') & (ch == 1), '#d62728', True),
        'DPA no-lick':   ((tsk == 'DPA') & (ch == 0), '#9467bd', False),
    }


AXES = [
    ('GNG lick axis\n(shared action code)', 'gng',    GNG_ACT, groups_gng, ('GNG Go (lick)', 'GNG NoGo')),
    ('DPA choice axis\n(shared action code)', 'choice', DPA_ACT, groups_choice, ('DPA lick', 'DPA no-lick')),
]

fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.2), sharex=True)
for ri, (ylab, target, train, gfun, orient_pair) in enumerate(AXES):
    D, yy = axis_rows(target, train)
    mo = yy.mouse.to_numpy(); lz = yy.laser.to_numpy() == 0; st = yy.learning.to_numpy(); perf = yy.performance.to_numpy()
    op, on = orient_pair                                             # positive/negative labels for per-mouse sign
    for ci, stage in enumerate(STAGES := ('Naive', 'Expert')):
        ax = axs[ri, ci]
        G = gfun(yy)
        base = lz & (st == stage) & (perf == 1)
        curves = {}
        for lab, (msk, col, lick) in G.items():
            permouse = []
            for m in MICE:
                sel = base & msk & (mo == m)
                if sel.sum() < 5:
                    continue
                # orient this mouse's axis so its OWN trained contrast (orient_pair) is positive at the action window
                pos = D[base & G[op][0] & (mo == m)][:, train].mean()
                neg = D[base & G[on][0] & (mo == m)][:, train].mean()
                s = 1.0 if pos >= neg else -1.0
                v = s * D[sel].mean(0)
                v = (v - v[BL].mean()) / (v.std() + 1e-9)
                permouse.append(v)
            if permouse:
                A = np.stack(permouse); curves[lab] = (A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A)), col, lick)
        for lab, (mu, se, col, lick) in curves.items():
            ax.plot(xt, mu, color=col, lw=2 if lick else 1.6, ls='-' if lick else '--', label=lab)
            ax.fill_between(xt, mu - se, mu + se, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color='0.8', lw=0.6)
        ax.axvspan(xt[42], xt[45], color='0.85', alpha=0.4, lw=0)     # GNG reward
        ax.axvspan(xt[60], xt[72], color='0.9', alpha=0.4, lw=0)      # DPA response
        if ri == 0:
            ax.text(xt[43], ax.get_ylim()[1], 'GNG\naction', fontsize=6.5, va='top', color='0.4')
            ax.text(xt[64], ax.get_ylim()[1], 'DPA\naction', fontsize=6.5, va='top', color='0.4')
            ax.set_title(stage, fontweight='bold')
        if ri == 1:
            ax.set_xlabel('time (s)')
        if ci == 0:
            ax.set_ylabel(f'projection on\n{ylab}')
        if ri == 0 and ci == 0:
            ax.legend(frameon=False, fontsize=7, loc='upper left')
fig.suptitle('Lick command is a shared action code — each axis carries BOTH tasks, each rising at its own action moment',
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT = 'figures/overlaps/action'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
p = f'{OUT}/png/overlaps_action_traj.png'
fig.savefig(p, dpi=300, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
