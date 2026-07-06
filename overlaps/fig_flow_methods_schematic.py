"""DESCRIPTIVE methods figure (SCHEMATIC) — "how we get the flow fields", idealized for a talk.

Not literal data — a clean cartoon of the model we fit, in three steps on the sample×choice plane:
  ① Recurrent landscape — the memory has two stable states (sample A / sample B). The recurrent overlap A
     carves a double well; dark = slow (near a fixed point).
  ② Gain-modulated velocity — the rank-2 flow assigns a velocity to every point,
        ż = −z + S(z)(A z + b),   S the mean gain (Gauss–Hermite), b the stimulus drive.
  ③ Field + attractors — the streamlines and fixed points (○); trajectories settle into the wells.

We FIT A and b so this field reproduces the measured code velocity (see fig_overlaps_story_main.py).
Self-contained (a smooth synthetic rank-2 flow). Run:
  cd /home/leon/dual/overlaps && python fig_flow_methods_schematic.py
Saves figures/overlaps/methods/{png,svg}/flow_methods_schematic.{png,svg}.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from src.pca.dynamics import flow_fixed_points

HALO = [pe.withStroke(linewidth=2.5, foreground='white')]     # white outline so coloured labels read on dark

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'

COL_A, COL_B = '#332288', '#44AA99'                     # sample A indigo, sample B teal
A_REC = np.array([[2.05, 0.0], [0.0, 0.72]])           # recurrent overlap → double well along sample axis
A_GAIN, DELTA = 0.55, 0.15                             # gain sharpness a, offset δ
B_DRIVE = np.array([0.0, 0.0])                         # autonomous (no stimulus drive) for the schematic

NODES, WK = np.polynomial.hermite_e.hermegauss(20); WK = WK / np.sqrt(2 * np.pi)
def gd(D, h):                                          # mean gain S = ⟨φ'(√D ξ + h)⟩, φ = tanh
    t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * NODES[None, :] + h[:, None])
    return (WK * (1 - t ** 2)).sum(1)


def flow(P):                                           # rank-2 low-rank RNN: ż = -z + S(z)(A z + b)
    P = np.atleast_2d(P); D = A_GAIN ** 2 * (P ** 2).sum(0) + DELTA
    S = gd(D, np.zeros(P.shape[1])); AP = A_REC @ P
    return np.vstack([-P[0] + S * (AP[0] + B_DRIVE[0]), -P[1] + S * (AP[1] + B_DRIVE[1])])


def integrate(z0, n=60, dt=0.35):                      # smooth trajectory into a well
    z = np.asarray(z0, float); out = [z.copy()]
    for _ in range(n):
        z = z + dt * flow(z[:, None]).ravel(); out.append(z.copy())
    return np.array(out).T


FP = [(np.asarray(p, float), k) for p, k, _ in flow_fixed_points(flow, [(-3, 3), (-3, 3)], n_seed=24)]
ATTR = [p for p, k in FP if k == 'attractor']
L = 1.45 * max(float(np.abs(np.array(ATTR)).max()), 1.0)

# smooth demo trajectories: start near the ridge, settle into each well
trajA = integrate([-0.35, 0.85 * L])                    # → left well  (sample A)
trajB = integrate([0.35, 0.85 * L])                     # → right well (sample B)

# ── figure ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 5.9))
gs = fig.add_gridspec(1, 3, wspace=0.13, left=0.035, right=0.985, top=0.80, bottom=0.30)
ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
for ax in (ax1, ax2, ax3):
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('sample code', fontsize=11)
ax1.set_ylabel('choice code', fontsize=11)

gl = np.linspace(-L, L, 220); Xg, Yg = np.meshgrid(gl, gl)
Fh = flow(np.vstack([Xg.ravel(), Yg.ravel()]))
U, V = Fh[0].reshape(Xg.shape), Fh[1].reshape(Xg.shape); SPD = np.hypot(U, V)


def draw_fp(ax, big=13):
    for p, k in FP:
        mec = {'attractor': 'k', 'saddle': '0.5', 'repeller': 'r'}[k]
        ax.plot(p[0], p[1], 'o', mfc='white', mec=mec, ms=big if k == 'attractor' else big - 3,
                mew=1.7, zorder=9)


# ① recurrent landscape — speed heatmap (dark wells) + faint streamlines + state labels
ax1.pcolormesh(Xg, Yg, SPD, cmap='magma', shading='auto')
ax1.streamplot(Xg, Yg, U, V, color='w', density=0.7, linewidth=0.5, arrowsize=0.6)
draw_fp(ax1)
for p, k in FP:
    if k == 'attractor':
        lab, c = ('sample A\nstate', COL_A) if p[0] < 0 else ('sample B\nstate', COL_B)
        ax1.annotate(lab, (p[0], p[1]), (p[0], p[1] - 0.42 * L), ha='center', va='top', fontsize=10.5,
                     fontweight='bold', color=c, path_effects=HALO,
                     arrowprops=dict(arrowstyle='-', color=c, lw=1.2))
ax1.set_title('① recurrent memory landscape', fontsize=13, fontweight='bold', pad=8)

# ② gain-modulated velocity — big clean uniform arrows, coloured by speed
gc = np.linspace(-L * 0.85, L * 0.85, 8); Xc, Yc = np.meshgrid(gc, gc)
Fc = flow(np.vstack([Xc.ravel(), Yc.ravel()]))
Uc, Vc = Fc[0].reshape(Xc.shape), Fc[1].reshape(Xc.shape); Mc = np.hypot(Uc, Vc) + 1e-9
q = ax2.quiver(Xc, Yc, Uc / Mc, Vc / Mc, Mc, cmap='magma', angles='xy', pivot='mid',
               scale=15, width=0.013, headwidth=4.5, headlength=5)
draw_fp(ax2)
axins = ax2.inset_axes([0.63, 0.74, 0.33, 0.22])       # gain curve S(‖z‖)
rr = np.linspace(0, L, 60); axins.plot(rr, gd(A_GAIN ** 2 * rr ** 2 + DELTA, np.zeros(len(rr))), 'k', lw=1.6)
axins.set_title('gain S(‖z‖)', fontsize=8); axins.tick_params(labelsize=6.5); axins.set_xticks([0, round(L)])
ax2.set_title('② gain-modulated velocity  ż', fontsize=13, fontweight='bold', pad=8)

# ③ field + attractors + settling trajectories
ax3.pcolormesh(Xg, Yg, SPD, cmap='magma', shading='auto')
ax3.streamplot(Xg, Yg, U, V, color='w', density=0.75, linewidth=0.5, arrowsize=0.7)
for tr, c in [(trajA, COL_A), (trajB, COL_B)]:
    ax3.plot(tr[0], tr[1], '-', color=c, lw=3.0, zorder=6)
    ax3.plot(tr[0, 0], tr[1, 0], 'o', color=c, mfc='w', ms=8, mew=2, zorder=7)
draw_fp(ax3)
ax3.set_title('③ field + attractors', fontsize=13, fontweight='bold', pad=8)

# arrows between panels
for a, b in [(ax1, ax2), (ax2, ax3)]:
    x = 0.5 * (a.get_position().x1 + b.get_position().x0)
    fig.text(x, 0.55, '→', ha='center', va='center', fontsize=30, fontweight='bold')

# title + equation band + legend
fig.text(0.5, 0.955, 'How the flow fields are built', ha='center', va='center',
         fontsize=15, fontweight='bold')
fig.text(0.5, 0.185, r'$\dot{z} \;=\; -z \;+\; S(z)\,(A\,z \;+\; b)$', ha='center', va='center', fontsize=19)
fig.text(0.5, 0.115,
         'leak −z   ·   gain S(z)   ·   recurrence A z  (memory)   ·   input drive b  (stimulus)          '
         'fit A, b to the measured code velocity',
         ha='center', va='center', fontsize=10, color='0.30')
leg = [Line2D([0], [0], color=COL_A, lw=3, label='sample A trajectory'),
       Line2D([0], [0], color=COL_B, lw=3, label='sample B trajectory'),
       Line2D([0], [0], ls='', marker='o', mfc='w', mec='k', ms=7, mew=2, label='start'),
       Line2D([0], [0], ls='', marker='o', mfc='white', mec='k', ms=9, mew=1.7, label='attractor'),
       Line2D([0], [0], ls='', marker='o', mfc='white', mec='0.5', ms=7, mew=1.7, label='saddle')]
fig.legend(handles=leg, loc='lower center', ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))

OUT = 'figures/overlaps/methods'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/flow_methods_schematic.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/svg/flow_methods_schematic.svg', bbox_inches='tight')
print('fixed points:', [(round(float(p[0]), 2), round(float(p[1]), 2), k) for p, k in FP])
print(f'saved {OUT}/png/flow_methods_schematic.png')
