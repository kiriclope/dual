"""
Linear latent dynamical system (LDS) on the pseudo-population latents.

Fits a first-order linear flow  z_{t+1} = A z_t + b  to the latent trajectories
(the PCA or dPCA components of a saved `pseudo_traj`), to characterise the
population dynamics and test how low-dimensional they are:

- the eigenvalues of `A` are the dynamical modes — a time constant from |λ|
  (|λ|<1 decays, =1 is marginally stable / line-attractor-like, >1 grows) and a
  rotation frequency from its phase (complex λ → oscillation);
- the held-out one-step predictive R² as a function of the number of latent
  dimensions shows how many dimensions the flow actually needs (an early plateau
  ⇒ low-dimensional dynamics).

This is a two-stage (latents-then-dynamics) LINEAR model fit on condition-resolved
single trials — it captures the average linear flow, not a full single-trial
latent-inference model (GPFA / LDS-EM), which is the heavier alternative.

Public API
----------
fit_lds            : ridge least-squares fit of z_{t+1} = A z_t (+ B u_t) + b
lds_modes          : eigenmodes of A → time constant (s) and frequency (Hz)
cv_predict_r2      : held-out h-step predictive R² of the linear flow
dimensionality_curve : predictive R² vs number of leading latent dimensions
boxcar_inputs      : build stimulus-onset boxcar regressors u_t from bin windows
fit_poly_flow      : nonlinear flow Δz = W φ(z) with polynomial features
fit_rnn_flow       : neural (rate-network) flow Δz = L z + W φ(g·z) + b with a
                     tanh/relu/logistic nonlinearity — admits multiple attractors
                     (a linear A has exactly one fixed point)
flow_fixed_points  : locate + classify fixed points of an arbitrary flow field
cv_flow_r2         : held-out one-step velocity R² of the rnn flow (KFold over trials)
bootstrap_fixed_points : fixed-point stability of the rnn flow under a resampler
"""

from itertools import combinations_with_replacement

import numpy as np


# ── inputs ──────────────────────────────────────────────────────────────────────

def boxcar_inputs(n_time, windows):
    """Stimulus design matrix: one boxcar regressor per time window.

    windows : list of array-like bin indices (e.g. [bins_STIM, bins_DIST, bins_TEST]).
    Returns U (n_inputs, n_time), 1.0 inside each window, 0 elsewhere.  Inputs are
    timing-locked (condition-independent), so U is shared across trials.
    """
    U = np.zeros((len(windows), n_time))
    for i, w in enumerate(windows):
        U[i, np.asarray(w)] = 1.0
    return U


# ── fit ─────────────────────────────────────────────────────────────────────────

def fit_lds(Z, U=None, ridge=1e-2):
    """Fit z_{t+1} = A z_t + B u_t + b by ridge least squares over trials & steps.

    Z : (n_trials, n_latent, n_time)
    U : (n_inputs, n_time) timing-locked stimulus regressors, or None for the
        autonomous model.  u_t is taken at the SOURCE step t (drives z_{t+1}).
    Returns A (nl, nl), B (nl, n_inputs) or None, b (nl,)
    """
    nt, nl, T = Z.shape
    Z0 = Z[:, :, :-1].transpose(0, 2, 1).reshape(-1, nl)     # (M, nl)  state
    Z1 = Z[:, :, 1:].transpose(0, 2, 1).reshape(-1, nl)      # (M, nl)  next state
    cols = [Z0]
    ni = 0
    if U is not None:
        if U.ndim == 2:                                      # (n_inputs, n_time) shared
            ni = U.shape[0]
            U0 = np.broadcast_to(U[:, :-1].T, (nt, T - 1, ni)).reshape(-1, ni)
        else:                                                # (n_trials, n_inputs, n_time)
            ni = U.shape[1]
            U0 = U[:, :, :-1].transpose(0, 2, 1).reshape(-1, ni)
        cols.append(U0)
    cols.append(np.ones((Z0.shape[0], 1)))                   # bias
    X = np.hstack(cols)
    reg = ridge * np.eye(nl + ni + 1)
    reg[-1, -1] = 0.0                                         # don't regularise bias
    W = np.linalg.solve(X.T @ X + reg, X.T @ Z1).T           # (nl, nl+ni+1)
    A = W[:, :nl]
    B = W[:, nl:nl + ni] if U is not None else None
    b = W[:, -1]
    return A, B, b


# ── analysis ────────────────────────────────────────────────────────────────────

def lds_modes(A, dt):
    """Dynamical modes of A.

    Returns
    -------
    ev   : (n,) discrete-time eigenvalues
    tau  : (n,) time constant (s); +inf if marginally stable, <0 if growing
    freq : (n,) oscillation frequency (Hz); 0 for non-rotational modes
    """
    ev = np.linalg.eigvals(A)
    lc = np.log(ev.astype(complex)) / dt                     # continuous-time
    with np.errstate(divide='ignore'):
        tau = np.where(lc.real != 0, -1.0 / lc.real, np.inf)
    freq = np.abs(lc.imag) / (2 * np.pi)
    return ev, tau, freq


def cv_predict_r2(Z, U=None, ridge=1e-2, n_splits=5, horizon=1, random_state=0):
    """Held-out h-step predictive R² of the linear flow (trials split for CV).

    If U (n_inputs, n_time) is given, the known stimulus inputs are fed in at each
    rolled step — so the score reflects how well the flow predicts GIVEN the
    external drive, isolating the autonomous-dynamics error from stimulus jumps.
    """
    from sklearn.model_selection import KFold
    T = Z.shape[2]
    per_trial = U is not None and U.ndim == 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    num = den = 0.0
    for tr, te in kf.split(np.arange(Z.shape[0])):
        Utr = U[tr] if per_trial else U
        A, B, b = fit_lds(Z[tr], U=Utr, ridge=ridge)
        Ute = U[te] if per_trial else U
        zt = Z[te][:, :, :T - horizon]
        for k in range(horizon):                             # roll the flow forward h steps
            pred = np.einsum('ij,njt->nit', A, zt) + b[None, :, None]
            if U is not None:
                sl = slice(k, k + (T - horizon))
                if per_trial:                                # (n_te, n_inputs, T-h)
                    pred = pred + np.einsum('il,nlt->nit', B, Ute[:, :, sl])
                else:                                        # (n_inputs, T-h) shared
                    pred = pred + (B @ Ute[:, sl])[None]
            zt = pred
        target = Z[te][:, :, horizon:]
        num += float(((target - zt) ** 2).sum())
        den += float(((target - target.mean()) ** 2).sum())
    return 1.0 - num / den


def dimensionality_curve(Z, dims, U=None, ridge=1e-2, **kw):
    """Predictive R² as a function of the number of leading latent dimensions.

    Z is assumed ordered so that Z[:, :k, :] are the k leading dimensions
    (variance-ordered for a PCA run).
    """
    return np.array([cv_predict_r2(Z[:, :k, :], U=U, ridge=ridge, **kw) for k in dims])


# ── nonlinear flow ────────────────────────────────────────────────────────────────
# A linear LDS (z_{t+1}=Az_t+b) has exactly ONE fixed point, so it can never show
# multistability.  Fitting the velocity Δz as a polynomial of z lifts that: a cubic
# flow admits several fixed points, so two condition-dependent attractors (e.g. the
# A and B sample states) appear on a SINGLE pooled field as two stable nodes with a
# saddle between them.

def _poly_features(P, deg):
    """Monomials of P (n_dim, N) up to total degree `deg`, incl. bias. → (n_feat, N)."""
    d, N = P.shape
    feats = [np.ones(N)]
    for o in range(1, deg + 1):
        for combo in combinations_with_replacement(range(d), o):
            f = np.ones(N)
            for k in combo:
                f = f * P[k]
            feats.append(f)
    return np.asarray(feats)


def fit_poly_flow(Z, deg=3, ridge=1e-2):
    """Fit the velocity field Δz = W φ(z) by ridge LS (φ = polynomials up to `deg`).

    Z : (n_trials, n_dim, n_time).  Returns a function `flow(P)` mapping states
    P (n_dim, N) → velocities (n_dim, N), plus the weight matrix W.
    """
    nl = Z.shape[1]
    Z0 = Z[:, :, :-1].transpose(0, 2, 1).reshape(-1, nl).T        # (nl, M) state
    dZ = (Z[:, :, 1:] - Z[:, :, :-1]).transpose(0, 2, 1).reshape(-1, nl).T  # (nl, M) Δz
    Phi = _poly_features(Z0, deg)                                 # (F, M)
    reg = ridge * np.eye(Phi.shape[0]); reg[0, 0] = 0.0          # don't shrink bias
    W = np.linalg.solve(Phi @ Phi.T + reg, Phi @ dZ.T).T         # (nl, F)

    def flow(P):
        return W @ _poly_features(np.atleast_2d(P), deg)

    return flow, W


_ACT = {
    'tanh':     np.tanh,
    'relu':     lambda x: np.maximum(x, 0.0),
    'logistic': lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def fit_rnn_flow(Z, act='tanh', gain=2.0, ridge=1e-2):
    """Neural (rate-network) flow  Δz = L z + W φ(g·z) + b,  fit by ridge LS.

    This is the canonical attractor-memory form: a linear **leak** (`L`) plus
    **recurrent feedback** through a saturating neural nonlinearity `φ` (tanh /
    relu / logistic).  Leak + saturating positive feedback is exactly what creates
    multiple stable states, so two condition-dependent attractors appear on one
    pooled field — unlike a linear `A` (one fixed point only).  φ acts elementwise
    on the current state (the "rate"), so the model is **linear in the parameters**
    `L, W, b` and fits in closed form; `gain` sets where the nonlinearity saturates
    relative to the data scale.

    Z : (n_trials, n_dim, n_time).  Returns `flow(P): (n_dim,N)→(n_dim,N)` and the
    stacked weights M = [b | L | W].
    """
    phi = _ACT[act]
    nl = Z.shape[1]
    Z0 = Z[:, :, :-1].transpose(0, 2, 1).reshape(-1, nl).T        # (nl, M)
    dZ = (Z[:, :, 1:] - Z[:, :, :-1]).transpose(0, 2, 1).reshape(-1, nl).T

    def feats(P):
        return np.vstack([np.ones((1, P.shape[1])), P, phi(gain * P)])  # (1+2nl, F)

    Phi = feats(Z0)
    reg = ridge * np.eye(Phi.shape[0]); reg[0, 0] = 0.0          # don't shrink bias
    M = np.linalg.solve(Phi @ Phi.T + reg, Phi @ dZ.T).T        # (nl, 1+2nl)

    def flow(P):
        return M @ feats(np.atleast_2d(P))

    return flow, M


def flow_fixed_points(flow, bounds, n_dim=2, n_seed=12, tol=1e-4, dedup=0.05):
    """Locate + classify fixed points of `flow` inside `bounds` = [(lo,hi),…].

    Roots are found with fsolve from a grid of seeds, kept if inside bounds and
    |flow|<tol, de-duplicated, then classified by the Jacobian of the **discrete map**
    z⁺ = z + flow(z) (since `flow` is the one-step increment Δz) — matching the RNN's
    map-Jacobian convention.  With J = ∂flow/∂z, the map Jacobian is I+J:
        all |eig(I+J)|<1 → 'attractor' ; all >1 → 'repeller' ; mixed → 'saddle'.
    (For slow flows this coincides with the continuous Re(eig J)<0 test.)
    Returns a list of (point (n_dim,), kind, map-eigenvalues).
    """
    from scipy.optimize import fsolve

    def f(z):
        return flow(np.asarray(z).reshape(n_dim, 1))[:, 0]

    seeds = np.array(np.meshgrid(*[np.linspace(lo, hi, n_seed) for lo, hi in bounds])
                     ).reshape(n_dim, -1).T
    found = []
    for s in seeds:
        z, info, ier, _ = fsolve(f, s, full_output=True)
        if ier != 1 or np.linalg.norm(info['fvec']) > tol:
            continue
        if any(not (lo <= z[k] <= hi) for k, (lo, hi) in enumerate(bounds)):
            continue
        if any(np.linalg.norm(z - p) < dedup for p, _, _ in found):
            continue
        eps = 1e-4
        J = np.zeros((n_dim, n_dim))
        for k in range(n_dim):
            dz = np.zeros(n_dim); dz[k] = eps
            J[:, k] = (f(z + dz) - f(z - dz)) / (2 * eps)
        map_eig = np.linalg.eigvals(np.eye(n_dim) + J)       # discrete map z⁺=z+flow(z)
        mag = np.abs(map_eig)
        kind = ('attractor' if (mag < 1).all() else
                'repeller' if (mag > 1).all() else 'saddle')
        found.append((z, kind, map_eig))
    return found


# ── model selection (cross-validation + fixed-point stability) ─────────────────────

def cv_flow_r2(Z, act='tanh', gain=0.7, ridge=1e-2, n_splits=5, random_state=0):
    """Held-out one-step velocity R² of the rnn flow Δz=Lz+Wφ(g·z)+b (KFold over
    trials).  Predictive accuracy — fit on train trials' (z,Δz) pairs, score on the
    held-out trials'.  NB high R² is necessary but not sufficient: a flexible fit can
    predict Δz well yet invent spurious fixed points — pair this with
    `bootstrap_fixed_points`.  Z : (n_trials, n_dim, n_time)."""
    from sklearn.model_selection import KFold
    nl = Z.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    num = den = 0.0
    for tr, te in kf.split(np.arange(Z.shape[0])):
        flow, _ = fit_rnn_flow(Z[tr], act=act, gain=gain, ridge=ridge)
        z0 = Z[te][:, :, :-1].transpose(0, 2, 1).reshape(-1, nl).T
        dz = (Z[te][:, :, 1:] - Z[te][:, :, :-1]).transpose(0, 2, 1).reshape(-1, nl).T
        pred = flow(z0)
        num += float(((dz - pred) ** 2).sum())
        den += float(((dz - dz.mean()) ** 2).sum())
    return 1.0 - num / den


def cv_condmean_flow_r2(trials, groups, act='tanh', gain=1.0, ridge=0.2,
                        n_splits=5, random_state=0):
    """Held-out CONDITION-MEAN velocity R² — the right metric for a flow we fit on
    condition means (single-trial `cv_flow_r2` is the ~1% noise floor and tests the
    wrong object). Stratified split over trials by `groups`; build condition means from
    the train trials, fit the rnn flow, and score how well it predicts the **held-out**
    trials' condition-mean velocity.  trials:(n,n_dim,n_time), groups:(n,) condition id."""
    from sklearn.model_selection import StratifiedKFold
    groups = np.asarray(groups)
    ug = np.unique(groups)
    if len(ug) < 2 or min(int((groups == k).sum()) for k in ug) < n_splits:
        return float('nan')                       # too few trials/condition to split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    nl = trials.shape[1]
    num = den = 0.0
    for tr, te in skf.split(np.arange(len(trials)), groups):
        gtr, gte = groups[tr], groups[te]
        train = np.stack([trials[tr][gtr == k].mean(0) for k in ug])   # (n_cond,dim,time)
        test = np.stack([trials[te][gte == k].mean(0) for k in ug])
        flow, _ = fit_rnn_flow(train, act=act, gain=gain, ridge=ridge)
        z0 = test[:, :, :-1].transpose(0, 2, 1).reshape(-1, nl).T
        dz = (test[:, :, 1:] - test[:, :, :-1]).transpose(0, 2, 1).reshape(-1, nl).T
        pred = flow(z0)
        num += float(((dz - pred) ** 2).sum())
        den += float(((dz - dz.mean()) ** 2).sum())
    return 1.0 - num / den


def bootstrap_fixed_points(resample, bounds, act='tanh', gain=0.7, ridge=1e-2,
                           n_boot=50, n_seed=21):
    """Bootstrap the rnn-flow fit to gauge how REPRODUCIBLE the fixed-point structure
    is — the scientific object, which CV velocity-R² does not directly measure.

    `resample` is a 0-arg callable returning ONE bootstrap fit tensor
    (n, n_dim, n_time); the caller decides the resampling.  For a landscape estimated
    from **condition means** (the meaningful case here) `resample` should bootstrap
    trials *within each condition* and return the rebuilt condition means — bootstrapping
    raw single trials instead tests the wrong thing (single-trial noise collapses the
    well).  Returns a dict: `n_attractors` (per-bootstrap counts), `p2` (fraction with
    exactly 2 attractors), `wells` (all attractor locations stacked, for a scatter)."""
    n_dim = len(bounds)
    n_att, wells = [], []
    for _ in range(n_boot):
        flow, _ = fit_rnn_flow(resample(), act=act, gain=gain, ridge=ridge)
        att = [p for p, k, _ in flow_fixed_points(flow, bounds, n_seed=n_seed)
               if k == 'attractor']
        n_att.append(len(att))
        wells.extend(att)
    n_att = np.asarray(n_att)
    return {'n_attractors': n_att, 'p2': float((n_att == 2).mean()),
            'wells': np.asarray(wells) if wells else np.empty((0, n_dim))}
