"""Presentation explainer — how the rank-2 (low-rank RNN) flow fields are derived.

A self-contained METHODS slide (no data needed). The model is the mean-field reduction of a rank-2 low-rank
recurrent network (Mastrogiuseppe & Ostojic, Neuron 2018; Dubreuil, Valente, Beiran, Mastrogiuseppe &
Ostojic, Nat. Neurosci. 2022):

  full net      tau x' = -x + J phi(x) + u s,     J = sum_r m_r n_r^T / N   (rank 2)
  collective    kappa_r = <n_r phi(x)>            (the two latents = sample & choice codes)
  reduced       kappa' = -kappa + S(kappa) (A kappa + B s)
                S(kappa) = <phi'(sqrt(Delta) xi)>,  Delta = a^2||kappa||^2 + beta_in + delta

Key point (vs a rate-current model): the INPUT enters INSIDE the nonlinearity — the drive B s is multiplied
by the same gain S and its variance adds to Delta — NOT as an external additive current.

  cd /home/leon/dual/overlaps
  python fig_flow_methods_explainer.py
Saves figures/overlaps/methods/{png,svg}/flow_methods_explainer.{png,svg}.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.pca.dynamics import flow_fixed_points

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

COL_A, COL_B = '#332288', '#44AA99'
C_LEAK, C_REC, C_INP, C_GAIN = '#666666', '#AA3377', '#4477AA', '#DDAA33'

# ── rank-2 mean-field model (the actual low-rank reduction) ─────────────────────
NODES, WK = np.polynomial.hermite_e.hermegauss(20); WK = WK / np.sqrt(2 * np.pi)
def gd(D, h):                                            # <phi'(sqrt(D) xi + h)>,  phi = tanh  — mean-field gain
    t = np.tanh(np.sqrt(np.maximum(D, 0))[:, None] * NODES[None, :] + h[:, None])
    return (WK * (1 - t ** 2)).sum(1)


def flow_lr(A, b, a, dd, beta=0.0):
    """Rank-2 reduced flow  kappa' = -kappa + S(kappa)*(A kappa + b),  b = B s the input drive INSIDE the
    gain; the input variance beta adds to Delta = a^2||kappa||^2 + beta + delta."""
    def fl(P):
        P = np.atleast_2d(P); D = a ** 2 * (P ** 2).sum(0) + beta + dd
        S = gd(D, np.zeros(P.shape[1])); AP = A @ P
        return np.vstack([-P[0] + S * (AP[0] + b[0]), -P[1] + S * (AP[1] + b[1])])
    return fl


def sim(fl, z0, n):
    z = np.asarray(z0, float).copy(); out = [z.copy()]
    for _ in range(n - 1):
        z = z + fl(z[:, None]).ravel(); out.append(z.copy())
    return np.array(out).T


A_DEMO = np.array([[2.05, 0.0], [0.0, 0.75]]); A_GAIN, A_DELTA = 0.55, 0.15
B_S = np.array([0.7, -3.0])                              # input drive B*s (enters INSIDE the gain)
fl_auto = flow_lr(A_DEMO, np.zeros(2), A_GAIN, A_DELTA)                 # autonomous (s = 0): symmetric wells
fl_input = flow_lr(A_DEMO, B_S, A_GAIN, A_DELTA, beta=float((A_GAIN * np.linalg.norm(B_S)) ** 2))


def draw_flow(ax, fl, LIM=3.2):
    gl = np.linspace(-LIM, LIM, 64); Xg, Yg = np.meshgrid(gl, gl); P = np.vstack([Xg.ravel(), Yg.ravel()])
    F = fl(P); U, V = F[0].reshape(Xg.shape), F[1].reshape(Xg.shape)
    ax.pcolormesh(Xg, Yg, np.hypot(U, V), cmap='magma', shading='auto')
    ax.streamplot(Xg, Yg, U, V, color='w', density=0.9, linewidth=0.6, arrowsize=0.8)
    for pt, kind, _ in flow_fixed_points(fl, [(-LIM, LIM), (-LIM, LIM)], n_seed=20):
        mk = {'attractor': ('*', 'yellow', 17), 'saddle': ('s', 'w', 9),
              'repeller': ('X', 'r', 10)}.get(kind, ('*', 'y', 12))
        ax.plot(pt[0], pt[1], mk[0], mfc=mk[1], mec='k', ms=mk[2], mew=1.1, zorder=7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


# ── figure ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10.2))
gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.0, 1.0], hspace=0.46, wspace=0.26,
                      left=0.045, right=0.965, top=0.905, bottom=0.055)

fig.suptitle('Rank-2 low-rank RNN flow fields on the sample × choice plane',
             fontsize=19, fontweight='bold', y=0.978)

# ── equation band: full network  →  reduced latent dynamics (inputs inside the gain) ──
axEq = fig.add_subplot(gs[0, :]); axEq.axis('off')
axEq.text(0.5, 0.97, r'low-rank RNN  (Mastrogiuseppe & Ostojic 2018; Dubreuil et al. 2022):'
          r'    $\tau\,\dot{x}=-x+J\,\phi(x)+u\,s,\quad J=\sum_{r=1}^{2} m_r\, n_r^{\top}/N$',
          ha='center', va='top', fontsize=13.5, color='0.15', transform=axEq.transAxes)
# reduced latent dynamics — colored inline terms
EQ = [(0.318, r'$\dot{\kappa}\;=$', 'k'), (0.378, r'$-\,\kappa$', C_LEAK), (0.420, r'$+$', 'k'),
      (0.492, r'$S(\kappa)\,($', C_GAIN), (0.560, r'$A\,\kappa$', C_REC), (0.605, r'$+$', 'k'),
      (0.655, r'$B\,s$', C_INP), (0.690, r'$)$', C_GAIN)]
for x, s, cc in EQ:
    axEq.text(x, 0.60, s, ha='center', va='center', fontsize=26, color=cc, transform=axEq.transAxes)
for x, lab, cc in [(0.378, 'leak', C_LEAK), (0.492, 'gain', C_GAIN),
                   (0.560, 'recurrence', C_REC), (0.655, 'input (inside!)', C_INP)]:
    axEq.text(x, 0.30, lab, ha='center', va='center', color=cc, fontsize=11, fontweight='bold',
              transform=axEq.transAxes)
axEq.text(0.5, 0.03, r'latents  $\kappa=\langle n\,\phi(x)\rangle$  (= the two CCGD codes);   '
          r'$A=\langle n\,m\rangle$ recurrent,  $B=\langle n\,u\rangle$ input overlaps;   '
          r'gain  $S=\langle \phi^{\prime}\rangle$,   $\Delta=a^{2}\|\kappa\|^{2}+\beta_{\mathrm{in}}+\delta$',
          ha='center', va='center', fontsize=11.5, color='0.3', transform=axEq.transAxes)

# ── 1 · State ───────────────────────────────────────────────────────────────────
axS = fig.add_subplot(gs[1, 0])
tt = np.linspace(0, 1, 40)
trajA = np.vstack([-2.3 + 0.5 * tt, 1.6 - 2.2 * tt + 0.6 * tt ** 2])
trajB = np.vstack([2.3 - 0.4 * tt, 1.5 - 3.0 * tt + 0.9 * tt ** 2])
for tr, c in [(trajA, COL_A), (trajB, COL_B)]:
    axS.plot(tr[0], tr[1], '-', color=c, lw=2.6, zorder=4)
    axS.plot(tr[0, 0], tr[1, 0], 'o', color=c, mfc='w', mew=1.6, ms=8, zorder=5)
    axS.plot(tr[0, -1], tr[1, -1], '*', color=c, ms=16, mec='k', mew=0.8, zorder=6)
    axS.annotate('', xy=(tr[0, 30], tr[1, 30]), xytext=(tr[0, 26], tr[1, 26]),
                 arrowprops=dict(arrowstyle='-|>', color=c, lw=2.4))
axS.axhline(0, color='0.7', lw=0.8, ls=':'); axS.axvline(0, color='0.7', lw=0.8, ls=':')
axS.set_xlim(-3.2, 3.2); axS.set_ylim(-3.2, 3.2); axS.set_aspect('equal')
axS.set_xticks([]); axS.set_yticks([])
axS.set_xlabel(r'sample code  $\kappa_{\mathrm{sample}}$', fontsize=11)
axS.set_ylabel(r'choice code  $\kappa_{\mathrm{choice}}$', fontsize=11)
axS.set_title('1 · State — each condition gives a\nmean trajectory  $\\mu(t)$  of the latents',
              fontsize=12.5, fontweight='bold')
axS.legend(handles=[Line2D([0], [0], color=COL_A, lw=2.6, label='sample A'),
                    Line2D([0], [0], color=COL_B, lw=2.6, label='sample B')],
           loc='lower left', fontsize=9, frameon=False)

# ── 2 · Gain ────────────────────────────────────────────────────────────────────
axG = fig.add_subplot(gs[1, 1])
rn = np.linspace(0, 4, 200)
for a, dd, cc, ls in [(0.55, 0.15, C_REC, '-'), (0.9, 0.15, '#CC6677', '--'), (0.3, 0.8, C_INP, ':')]:
    axG.plot(rn, gd(a ** 2 * rn ** 2 + dd, np.zeros_like(rn)), ls, color=cc, lw=2.6,
             label=fr'$a={a},\ \delta={dd}$')
axG.set_xlabel(r'$\|\kappa\|$  (distance from origin)', fontsize=11)
axG.set_ylabel(r'gain  $S(\kappa)$', fontsize=11)
axG.set_title('2 · Gain drops as the state grows\n$\\Rightarrow$ saturation enables two stable wells',
              fontsize=12.5, fontweight='bold')
axG.legend(fontsize=9.5, frameon=False, loc='lower left')
axG.set_ylim(0, 1.22); axG.set_yticks([0, 0.5, 1.0]); axG.grid(alpha=0.25)
axG.text(0.5, 0.955, r'$S(\kappa)=\langle\, \tanh^{\prime}(\sqrt{\Delta}\,\xi)\,\rangle'
         r'_{\xi\sim\mathcal{N}(0,1)},\ \ \Delta=a^{2}\|\kappa\|^{2}+\beta_{\mathrm{in}}+\delta$',
         ha='center', va='top', fontsize=11, transform=axG.transAxes,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.75', alpha=0.95))

# ── 3 · Shared recurrent landscape + pooling ────────────────────────────────────
axP = fig.add_subplot(gs[1, 2]); axP.axis('off')
axP.text(0.5, 1.02, '3 · Shared recurrent landscape + pooling', ha='center', va='top', fontsize=12.5,
         fontweight='bold', transform=axP.transAxes)
axP.text(0.5, 0.83, r'$A_r \;=\; A_{\mathrm{sh}} \;+\; \Delta A_r$', ha='center', va='top',
         fontsize=17, transform=axP.transAxes)
axP.text(0.5, 0.66, r'ridge penalty  $\lambda\,\|\Delta A_r\|^{2}$', ha='center', va='top',
         fontsize=13, color='0.25', transform=axP.transAxes)
axP.text(0.5, 0.52, 'regimes sharing a decoder pool ONE\nrecurrent landscape $A_{\\mathrm{sh}}$',
         ha='center', va='top', fontsize=11, color='0.25', transform=axP.transAxes)
modes = [('partial', r'$A_{\mathrm{sh}}+\Delta A_r$  (ridge $\lambda$)', C_REC),
         ('shared', r'one $A_{\mathrm{sh}}$  ($\Delta A_r\equiv 0$)', C_INP),
         ('independent', r'per-regime $A_r$  (no pooling)', C_LEAK)]
for i, (nm, desc, cc) in enumerate(modes):
    yy = 0.33 - i * 0.10
    axP.text(0.05, yy, '•', color=cc, fontsize=16, fontweight='bold', transform=axP.transAxes)
    axP.text(0.12, yy, nm, color=cc, fontsize=12, fontweight='bold', va='center', transform=axP.transAxes)
    axP.text(0.42, yy, desc, color='0.2', fontsize=10.5, va='center', transform=axP.transAxes)
axP.text(0.5, 0.005, r'all modes: input enters as  $S\,(B_r\,s)$,  inside the gain', ha='center', va='top',
         fontsize=10.5, color=C_INP, style='italic', transform=axP.transAxes)

# ── 4 · Fit objective ───────────────────────────────────────────────────────────
axF = fig.add_subplot(gs[2, 0])
mu = sim(fl_auto, np.array([-1.9, 1.7]), 22)
rng = np.random.default_rng(1)
obs = mu + rng.normal(0, 0.16, mu.shape)
for k in range(mu.shape[1]):
    axF.plot([mu[0, k], obs[0, k]], [mu[1, k], obs[1, k]], '-', color='0.7', lw=0.9, zorder=2)
axF.plot(mu[0], mu[1], '-', color=C_REC, lw=2.8, zorder=4, label=r'fit $\hat{\kappa}(t)$ (integrated flow)')
axF.plot(obs[0], obs[1], 'o', color='0.25', ms=4.5, zorder=5, label=r'data $\mu(t)$ (condition mean)')
axF.plot(mu[0, -1], mu[1, -1], '*', mfc='yellow', mec='k', ms=16, zorder=6)
axF.set_xlim(-3.0, 1.0); axF.set_ylim(-2.2, 2.2)
axF.set_xticks([]); axF.set_yticks([])
axF.set_xlabel('sample code', fontsize=11); axF.set_ylabel('choice code', fontsize=11)
axF.set_title('4 · Fit — integrate the flow, match the\nTRAJECTORY (position), not the noisy velocity',
              fontsize=12.5, fontweight='bold')
axF.legend(fontsize=9, frameon=False, loc='lower right')
axF.text(0.03, 0.03, r'$\min_{A,\,B}\ \sum_t\|\hat{\kappa}(t)-\mu(t)\|^{2}$'
         '\n' r'$\hat{\kappa}(t{+}1)=\hat{\kappa}(t)+\dot\kappa(\hat{\kappa}(t))$',
         ha='left', va='bottom', fontsize=11.5, transform=axF.transAxes,
         bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.6', alpha=0.9))

# ── 5 · Result (two example fitted flows) ───────────────────────────────────────
axR1 = fig.add_subplot(gs[2, 1]); draw_flow(axR1, fl_auto)
axR1.set_xlabel('sample code', fontsize=11); axR1.set_ylabel('choice code', fontsize=11)
axR1.set_title('5 · Autonomous ($s=0$) — two wells\n(sample A / sample B memory)', fontsize=12.5,
               fontweight='bold')
axR2 = fig.add_subplot(gs[2, 2]); draw_flow(axR2, fl_input)
axR2.annotate('gain-modulated input\n$S\\,(B s)$  moves the well', xy=(0.85, -1.45), xytext=(-2.9, 1.7),
              fontsize=10.5, color='w', ha='left', va='center',
              bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.5),
              arrowprops=dict(arrowstyle='-|>', color='w', lw=2.0))
axR2.set_xlabel('sample code', fontsize=11); axR2.set_ylabel('choice code', fontsize=11)
axR2.set_title('input-driven — the drive enters\nINSIDE the gain  ($S\\,(A\\kappa+Bs)$)', fontsize=12.5,
               fontweight='bold')

leg = [Line2D([0], [0], ls='', marker='*', mfc='yellow', mec='k', ms=14, label='attractor'),
       Line2D([0], [0], ls='', marker='s', mfc='w', mec='k', ms=9, label='saddle'),
       Line2D([0], [0], ls='', marker='X', mfc='r', mec='k', ms=10, label='repeller')]
fig.legend(handles=leg, loc='lower center', ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.004))

OUT = 'figures/overlaps/methods'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
fig.savefig(f'{OUT}/png/flow_methods_explainer.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/svg/flow_methods_explainer.svg', bbox_inches='tight')
print('autonomous fixed points:', [(k, np.round(p, 2)) for p, k, _ in
      flow_fixed_points(fl_auto, [(-3.2, 3.2), (-3.2, 3.2)], n_seed=20)])
print('input fixed points     :', [(k, np.round(p, 2)) for p, k, _ in
      flow_fixed_points(fl_input, [(-3.2, 3.2), (-3.2, 3.2)], n_seed=20)])
print(f'saved {OUT}/png/flow_methods_explainer.png')
