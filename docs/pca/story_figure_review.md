# Story figure — review log & caveats

Review of `pca/fig_dpca_story_main.py` (the full-arc dPCA main figure). Records the bugs found and
fixed, and the standing caveats to remember before making claims from this figure. Companion:
`story_figure_methods.md` (the how-to). Deep design notes live in `flows_handoff.md`.

Dates are absolute. Current as of 2026-07-03.

---

## Bugs found & fixed

### Section 4 push — was a rigid translation, now a gated deformation  *(fixed)*
- **Symptom (user-reported):** "the whole manifold goes down instead of just the wells."
- **Cause:** the no-lick drive was `ẏ = −(y − y_tgt)` — a linear pull of the *entire* plane to one
  level. The input sat **outside** the gain nonlinearity, so it slid every streamline down uniformly.
- **Fix:** the input now enters **inside** the nonlinearity — `ẏ = −y − h·r(z)`,
  `r(z) = 1 − S(z) = ⟨tanh²(√(a²‖z‖²+δ)·ξ)⟩`. `r≈0` near the origin (linear regime, `S≈1`) and `≈1` at
  the wells (saturated, `S≈0`), so the drive pushes **only the wells** (high ‖z‖) down while the
  centre/saddle stays pinned at 0. That is a manifold deformation, not a translation.
- **Verify:** the saddle sits at `y=0` in BOTH the Naive and Expert panels; only the wells dip in Expert.

### Panel J — trivial control + inconsistent with the flow  *(fixed)*
- **Bug (a):** the "sample" control averaged sample A (−) and B (+) on the sample axis → **≈0 by
  symmetry** for both stages (Δ−0.03). It could not show anything; it was a symmetry identity, not a
  control.
- **Bug (b):** J plotted **absolute** depths (Naive ≈ −1.2), contradicting the flow, which shows Naive
  at the reference (0).
- **Bug (c):** `load_mouse` only oriented the *tasks* axis, so the sign of `B−A` on the sample axis was
  arbitrary per mouse (some mice showed negative "separation").
- **Bug (d):** `y.sample` silently resolved to the pandas `DataFrame.sample()` **method**, not the
  column → `AttributeError`. Must be `y['sample']`.
- **Fixes:**
  - **J1 "no-lick push"** — per-mouse push **anchored to each mouse's own Naive well (0)** so it reads
    like the flow (mean −0.59, **p=0.012**; matches the flow's −0.53). Anchoring does not change the
    paired Wilcoxon p.
  - **J2 "sample memory"** — replaced the trivial centroid with the real **separation |B−A|**;
    `load_mouse` now orients the sample axis B>A. Preserved/sharpened (N +1.65 → E +2.33, p=0.10 correct
    / 0.02 all).
  - Depth window aligned to `LATE = bins 39–53` in both the pooled flow (`stage_delay`) and the
    per-mouse stats (`depth_of`).

---

## Standing caveats (do NOT over-claim)

1. **Section-3 flows now cross-validate (partial pooling).** Model switched from a free per-regime fit
   ("independent", CV velocity-R² ≈ **−0.13** — *overfit*, and it looked artificial) to **partial pooling**:
   a shared recurrent `A_sh` per epoch + a ridge-penalized per-regime deviation `ΔA_r` + input `c_r`
   (`ż = −z + S(z)·(A_sh+ΔA_r)·z + c_r`; `--partial` mode of `fig_dpca_flow_lowrank_shared.py`). CV vel-R²
   is now **+0.077** (correct) / +0.104 (all) — *positive*, the per-regime velocity fields **generalize**
   out-of-sample. This is still a **rank-2 reduced** description; the standing "rank-2 not validated as the
   *full* latent dynamics" note holds (dPCA reduced-rank test: rank-2 = 62–67% of full, no plateau). So:
   the per-regime flows cross-validate, but do not claim the full latent dynamics are rank-2.

   **Two shared landscapes (why).** A single shared `A_sh` can't be both sample-bistable (delay) and
   choice-bistable (cue/test); the 3 choice-bistable regimes outvote the 1 sample-bistable regime, so a
   single `A_sh` comes out choice-dominated and the sample-memory panels (autonomous, A, B) put the memory
   states at **saddles** — scientifically wrong. Fixed by pooling **within** each epoch: a delay landscape
   `{autonomous, A, B}` (sample-bistable) and a choice landscape `{Go, NoGo, cue, C, D}` (choice-bistable).
   Model selection is restricted to gains whose shared autonomous flow keeps 2 wells (WM bistability is an
   established result; the raw CV-optimal gain is often monostable).

   **What the CV score is.** `section3.cv()` = held-out velocity R² over 5 folds: fit each flow to train
   condition-mean velocities (positions `z=μ(t)`, one-step `v=μ(t+1)−μ(t)`), predict held-out velocities.
   `R²=1` perfect; `R²=0` = the mean-velocity baseline; `R²<0` = worse than that (what the old independent
   fit did). Positive now = the regularized shared-landscape fit transfers across trial splits. The
   *geometry* (wells/attractors/saddles = where the mean trajectories sit) was always real; partial pooling
   additionally makes the *fitted vector field* validated out-of-sample.

2. **Per-task variance is a proxy.** `marginal_variance` = variance of the *demixed condition-mean*
   components per marginal, **not** the exact dPCA encoder/decoder marginal-EVR. Numbers: **time 54%,
   tasks 31%, sample 7%, choice 7%, test 1%**. Time dominates (the temporal ramp); the WM axes
   (sample/choice) are genuinely low-variance but decodable — state that plainly.

3. **Section 4 is a data-constrained model, not a free-hand cartoon.** *Data-driven:* the sample
   bistability `fx` (least-squares fit to the A/B delay means); the push depth `push = dE − dN` (the
   *measured* Naive→Expert deepening, the p=0.012 effect); and **`hE`, which is fit** — grid-searched so
   the gated flow's wells sit exactly at `push`. So the Expert well depth IS the data value.
   *Modeling choices (not fit):* the **form** of the gate (input as `−h·r(z)`, `r=1−S` → deforms rather
   than translates — the structural point); the two gate hyperparameters `GATE_A=0.9, GATE_D=0.12`,
   which set how sharply the deformation localizes to the wells (they change the *look*, not the depth,
   since `hE` re-tunes); the fixed bistability shape (`α=0.42, δ=0.4`); and the naive=0 display anchor.
   Absolute depths: dN=−0.86, dE=−1.39. Bottom line: the push magnitude and landscape come from data;
   the gate's spatial profile is illustrative.

4. **Flows are POOLED** across the 9 mice (not per-mouse fits). State geometry is ~2-D (top-2 94% wm /
   82% all, PR 2.2) but the full latent dynamics is higher-rank.

---

## Open judgment call
- J2 says "sample memory **preserved**", but on **all-trials it is a significant increase** (p=0.02) —
  the memory *sharpens* with training. If that should be foregrounded as a finding ("memory sharpens")
  rather than a control ("preserved"), reword J2.

## Cosmetic / non-blocking
- Unused names: `ncol` arg in `section3`, `LIM`, a few loop vars. Pyright flags `src.pca` imports and
  `.pvalue` — all false positives at runtime (sys.path set at import; scipy namedtuple).
