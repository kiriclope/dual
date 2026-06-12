# Overlaps Analysis Decisions & Feedback

## Flow field (`plot_flow2d.py`)

**Bin velocity per sample (A/B) separately, combine winner-take-all — NOT pooled.**
Why: A and B leave origin in opposite sample-code directions; pooling cancels them → spurious central fixed point, real attractors vanish.
How: compute `(U_A, count_A)` and `(U_B, count_B)` separately; per cell keep velocity of whichever has higher raw count (`np.argmax(cnt_stack, axis=0)`).

**Use per-(mouse,odor_pair) MEAN trajectories, NOT individual trials.**
Why: Individual-trial noise σ≈5 BL σ/bin swamps the 0.1–0.5 BL σ/bin true drift → field ~9× weaker, noise-dominated. Means denoise positions so field is tangent to average trajectory.
How: `group_mean_trajs` returns `X_ep[mask].mean(0)` per (mouse,odor_pair).

**Centred finite difference, NOT forward difference.**
Why: Forward diff `(x[t+k]-x[t])/k` lags field by ~k/2 bins → streamlines don't align with trajectory.
How: `h = VEL_STEP//2`; `(x[c+h]-x[c-h])/(2h)` attributed to position `x[c]`.

**Fixed points = grand_mean_traj(BINS_LATE).mean(), NOT speed minimum.**
Why: NW speed minimum is displaced from true attractor in noisy field. Trajectory endpoint is correct by construction.
How: `cx = sx_full[BINS_LATE].mean(); cy = cy_full[BINS_LATE].mean()`. Draw as `marker='*', s=220`.

**Speed heatmap: hybrid BINS_LATE / all-delay, masked to all-delay support.**
Why: All-delay speed at fixed-point cells contaminated by fast early-delay transients → fixed points look fast. BINS_LATE-only support too small (misses approach path).
How: `speed_hybrid = np.where(supported_late, speed_late, speed)`; mask with `supported` (all-delay).
Colormap: `magma` (slow=dark). `set_bad='#f5f5f5'`.

**Dynamic axis limits from BINS_LATE fixed-point positions + AXIS_PAD=0.4.**
Why: Fixed AXIS_LIM=(-2,2) clipped sample A's endpoint.
How: `lim = max(abs(fp_x+fp_y)) + AXIS_PAD`; `AXIS_LIM = (-lim, lim)`.

**Baseline bins (`BINS_BL`) in velocity field create dominant slow basin at origin — excluded.**
Why: Baseline points cluster near (0,0) with ~0 velocity → overwhelms delay attractors.
How: Default variant `all_trials` uses `[BINS_DELAY]` only.

---

## `plot_traj2d.py` — histogram strip

**KDE strip added to right of each 2D panel** replaces occupancy plots; directly shows A/B trajectories occupy opposite halves of y=0.
Why: 2D occupancy heatmap was harder to read; 1D marginal on choice axis shows the separation more clearly.
How: `gaussian_kde(choice_vals, bw_method=0.4)` over all `BINS_DELAY` values. y-range matches 2D panel via `sharey`. Dashed line at each distribution mean. "choice dist." column title; A/B legend in top-left strip.

---

## `plot_scatter_perf.py`

**x-axis uses correct trials only; y-axis (Δ performance) uses all laser-off.**
Why: Correct trials give cleaner choice code signal. Δ performance must use all trials — filtering to correct would trivially give ~1 for everyone.

**Per-panel xlim centred on 0 (`half = max(abs(min), abs(max))`).**
Why: Shared xlim hid per-condition range; off-centre xlim made correlations hard to read.

**Stats text placed above panels (`(0.5, 1.01)`, `va='bottom'`).**
Why: Below-panel placement overlapped with xlabel. Inside-panel placement obscured data.

**1-sample t-test tests Δ performance ≠ 0, NOT the correlation.**
Why: It asks "did performance change at all?" — independent of r/ρ. Label it `1-samp t (Δperf≠0)` to avoid confusion.
