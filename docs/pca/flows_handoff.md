# Latent dynamics & flow fields — work log / handoff

Where we are with the **pseudo-population latent dynamics** work and the
**2-D flow-field figures**, so a future session can resume without re-deriving it.
Companion to `docs/pca/overview.md` (which has the user-facing reference); this file
is the *why/how-we-got-here* and the open issues.

---

## Goal

Characterise the population dynamics of the dPCA latent space and, ultimately,
draw **input-driven flow fields** (RNN-style: speed heatmap + streamlines + fixed
points) in the **sample × choice** plane showing how each stimulus reshapes a
bistable working-memory landscape.

## Data / state space

- Canonical run (DUM): `pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca`
  - whole-timeline (`ALL`), blcenter, dPCA with factors sample, test.
  - `pseudo_marglabels_<dum>.pkl` → axis labels `['sample','sample','test','test','sample:test','sample:test','time','time']`.
- The 2-D plane used everywhere: **x = `sample` axis, y = `choice` = the `sample:test` interaction axis.**
- Codings: sample 0=A / 1=B; test 0=C / 1=D; tasks DPA / DualGo / DualNoGo.
  **choice + = lick.** Lick pairs: AC, BD; no-lick: AD, BC.
- Bins (84 total, 6 Hz): BL 0–11, STIM(sample) 15–17, DIST(distractor) 30–32,
  DELAY 21–53, TEST 57–59. Go/NoGo = the DualGo/DualNoGo distractor (DPA has none).
- Empirical landmarks (Expert):
  - autonomous DPA-delay wells: **A ≈ (−1.37, −0.1)**, **B ≈ (1.07, 0.0)**, saddle ≈ (0.24, 0).
  - Go choice → **+2.2**, NoGo → **−2.3** (they **diverge from the first distractor bin**; no shared upward phase).
  - test endpoints (bins 57–66): C: A→+1.37, B→−1.30 ; D: A→−1.21, B→+1.17.

## Files

- `src/pca/dynamics.py` — the dynamics/flow model library.
- `pca/plot_pseudo_dynamics.py` — LDS eigenvalues + dimensionality + input term.
- `pca/plot_pseudo_flow.py` — the 2-D flow-field figures (all modes incl. `--inputs`).
- `pca/run_pseudo.py` — has `--epoch ALL` (whole-timeline PCA basis) added for this.
- `pca/build_mouse_dpca.py` — builds a per-mouse dPCA DUM (one real co-recorded
  population) reusing `X_all` without clobbering it; DUM = `..._f-<factors>_dpca_<MOUSE>[_ci<q>]`,
  plugs straight into `plot_pseudo_flow.py --dum`. `--factors`, `--remove-ci`.
- `pca/build_subpool_dpca.py` — pseudo-population dPCA over a SUBSET of mice (reuses `X_all`,
  no clobber); `--exclude`/`--include` + `--tag`. DUM `..._dpca_<tag>` (e.g. `pool7`).
- `pca/exp_single_trial_bias.py`, `pca/exp_single_trial_sep.py` — diagnostics behind the
  "condition means are required" conclusion (single-trial section). `pca/exp_tune_flow.py`,
  `pca/exp_regime_r2.py`, `pca/exp_tune_global.py` — gain×ridge / per-regime CV-R² tuning.
  `pca/exp_manifold.py` (cond means in the dPCA sample×choice plane), `pca/exp_manifold_pca.py`
  (same in raw / ramp-removed PCA, role-based planes) — the "slow circular manifold" test.
  `pca/exp_eig_nonlinear.py` (per-fixed-point eigenvalue modes of the nonlinear flow),
  `pca/analyze_bistability.py` (9-mouse barrier/well/d′/P2 survey + summary figure),
  `pca/exp_bootstrap_polar.py` (pooled radial/angular geometry: bootstrap-CI on the
  condition-mean trajectory — preferred radius + no-rotation with CIs),
  `pca/exp_polar_bug.py` + `pca/exp_polar_resample.py` + `pca/fig_polar_methods.py` (proof that
  bundling ≡ bootstrap ≡ CV at matched k; only small-k EIV differs),
  `pca/fig_polar_radang_permouse.py [boot|bundle|cv] [sample|choice]` (radial+angular v(r) from the
  trajectory, bootstrap-of-condmean ±CI) + `pca/fig_polar_radang_condmean.py` (raw point estimate)
  + `pca/fig_polar_wellfit_permouse.py` (**per-mouse well rescue**: restoring-slope fit, 4/9 sig)
  + `pca/fig_polar_flowfield_permouse.py [sample|choice]` & `pca/fig_polar_flowfield_pooled.py`
  (**cleanest polar view**: v_rad/v_ang read from the fitted flow field along the axis, ±CI),
  `pca/fig_flowfield_panels.py [mouse|pooled]` (3-panel field: speed/radial/angular heatmaps) &
  `pca/fig_flowfield_binned.py [mouse|pooled] [k]` (model-free binned field, k=bundle size),
  `pca/exp_slowmanifold_test.py [mouse|pooled]` (slow-manifold vs 2D-double-well discriminator),
  `pca/exp_ceiling.py` (proves the denoising ceiling = condition mean: bundling vs smoothing/Kalman),
  `pca/fig_manifold.py` (the defensible single-trial-blob vs condition-mean-2-D figure).

## HOW TO RUN — analyses & plots (cookbook; read this before plotting)

**Environment (always):** `cd /home/leon/dual/pca` first (scripts use relative `../data/pca`), then run with
`/home/leon/mambaforge/envs/dual/bin/python <script> ...`. Figures save to `pca/figures/pseudo/...` as PNG
(dpi 300) + SVG. Every flow script auto-saves; just run it and open the printed path.

**DUM cheat-sheet (the #1 source of confusion).** A DUM is the saved-run id; the flow scripts take `--dum`.
Build pattern: `pseudo_ALL_<STAGE>_zscore_5x1_scale_blcenter_f-<factors>_dpca[_<MOUSE>][_ci<q>]`.
| you want | DUM |
|---|---|
| pooled Expert, sample×choice plane | `pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca` |
| pooled Naive | `…_ALL_Naive_…_f-sample-test_dpca` |
| **+ tasks marginal** (needed for the no-lick push / `--push`) | `…_f-sample-test-tasks_dpca` |
| per-mouse | append `_<MOUSE>` e.g. `…_f-sample-test_dpca_JawsM15` (build first, see below) |
| CI-removed | append `_ci1`/`_ci2` (pooled: before `_dpca`; per-mouse: after `_<MOUSE>`) |
`<STAGE>` in the DUM = the dPCA **basis**; the scripts also pick the matching trials. Mice: JawsM01/06/12/15/18,
ChRM04/23, ACCM03/04. Build a per-mouse/CI DUM (one-off) before plotting it: `python build_mouse_dpca.py
--mice JawsM15 [--factors sample test tasks] [--remove-ci 1]`. Build a Naive/tasks pooled basis (one-off):
`python run_pseudo.py --rebuild --dpca --stage Naive --norm zscore --scale blcenter --factors sample test tasks --n-splits 5 --n-repeats 1`.

**"I want to plot …" → run this:**
| goal | command |
|---|---|
| **autonomous + input-driven flows** (the canonical Fig-2E-style grid, generic rate-net) | `python plot_pseudo_flow.py --dum <DUM> --dims sample choice --inputs --cond-mean [--stage Expert\|Naive]` |
| same, **per-mouse** (rescale needed; smooth helps thin cells) | `python plot_pseudo_flow.py --dum <DUM>_JawsM15 --dims sample choice --inputs --cond-mean --rescale 2.8 --smooth 1.5` |
| autonomous **WM bistability only** (DPA delay) | `python plot_pseudo_flow.py --dum <DUM> --dims sample choice --nonlinear --cond-mean` |
| simplest **linear** / **per-sample** flow | `python plot_pseudo_flow.py --dum <DUM> --dims sample choice [--per-sample]` |
| **rank-2 low-rank** flows (autonomous + inputs grid; the project's RNN-form) | `python fig_dpca_flow_lowrank_shared.py [--dum <DUM>]` (pooled→independent; per-mouse→partial; override `--shared`/`--independent`/`--partial`) |
| rank-2 flows **with the no-lick push** (wells pushed into no-lick, autonomous = bistable+gated deformation) | `python fig_dpca_flow_lowrank_shared.py --push [--dum …_f-sample-test-tasks_dpca]` ← **needs the tasks DUM** |
| rank-2 **Naive** (own basis) | add `--dum pseudo_ALL_Naive_zscore_5x1_scale_blcenter_f-sample-test[-tasks]_dpca` |
| **LDS eigenvalues / dimensionality / input term** | `python plot_pseudo_dynamics.py --dum <DUM>` |
| **no-lick push magnitude / stats** (tasks-axis depth, raw-ΔF/F check, CI q-sweep) | `fig_dpca_descent_rawdff.py`, `fig_dpca_nolick_ci_qsweep.py` (analysis figs; see SETTLED section) |

**Notes that trip me up:** `--inputs` is enforced on the **sample × choice** plane only and needs `--cond-mean`
(single-trial collapses). `--stage` selects which trials overlay/fit; the DUM sets the basis — keep them matched
(Naive DUM → `--stage Naive`). `fig_dpca_flow_lowrank_shared.py` is **stage-aware from the DUM** and tags output
`dpca_lowrank_<mode>_<STAGE>[_push].png`, so Naive/Expert don't overwrite.

**Do NOT use (retracted — see SETTLED CONCLUSIONS):** `fig_dpca_flow_autonomous_choice.py`,
`fig_dpca_choice_ci_qsweep.py`, `fig_dpca_flow_lowrank_shared.py --choice-auto`,
`fig_dpca_flow_learning_ingain*.py` (Naive=0 / combined-axis claims). These depict orientation/leakage artifacts.

## How we got here (progression — each step's finding)

1. **Linear LDS** (`fit_lds`): `z_{t+1}=Az+b`. Eigenvalues near unit circle, slow,
   non-rotational → integrator/memory. Dimensionality (held-out multi-step R²)
   plateaus ~2–3 D. *Basis matters:* must use a **whole-timeline blcenter** basis
   (delay-fit/center bias the answer); added `--epoch ALL`.
2. **dPCA task vs time subspace** (in `plot_pseudo_dynamics.py`): time subspace
   slowest (τ≈22 s, most predictable); task subspace faster (τ 3–6 s).
3. **Input term** `z_{t+1}=Az+Bu+b`: stimulus inputs **do not** recover the task
   subspace's long-horizon predictability → that gap is genuine dynamics, not
   unmodelled stimulus drive. (Time subspace mildly helped.)
4. **2-D flow fields** (`plot_pseudo_flow.py`): linear (one fp), `--per-sample`
   (A/B displaced fps — the bistability is condition-dependent), then **nonlinear**.
5. **Why two attractors need nonlinearity:** a linear A has exactly one fixed
   point. Added `fit_rnn_flow` (neural `Δz=Lz+Wφ(g·z)+b`) and `fit_poly_flow`.
   Two requirements for the clean **double-well**: fit on **condition means** (single
   trials too noisy → collapse to one well) over the **maintenance window**
   (whole-trial makes the wells saddles because test drives choice).
   → **DPA delay autonomous = bistable A/B**: `--nonlinear --task DPA --fit-win 21 54`.
6. **RNN aesthetic**: magma `‖Δz‖` heatmap + white streamlines + fixed-point markers
   (white ● attractor / white ✕ saddle / red ○ repeller), matching `~/rnn/src/dynamics.py`.
7. **Input-driven panels** (`--inputs`): many approaches tried — see
   "Input-driven model: what we tried" below. Current = simple per-regime fit.

## Input-driven model: what we tried (and why each was dropped)

Goal: per-stimulus flow fields to set beside an RNN's autonomous + input-driven flows.
The hard case throughout was **C/D bimodality** (test resolves choice oppositely by
sample: AC lick / BC no-lick, AD/BD mirror — verified in data: AC +1.08 / BC −1.00,
AD −0.92 / BD +0.88). Chronological attempts:

1. **Hand-calibrated drives** (`ct_drive`/`st_drive`): construct a drive that cancels the
   autonomous velocity at a data-derived target, so a fixed point lands there; C/D
   sample-gated via `np.interp`. *Worked* (bimodal) but **imposed**, not fit. Dropped
   when we moved to fitting.
2. **Fitted current inside φ, autonomous frozen** (`fit_input_current`, `Δz=Lz+Wφ(g·z+h)`):
   a constant current can't gate choice by sample (AC↑/BC↓ cancel) → C≈0, no bimodality.
3. **Affine current `h=a+Mz`**: full `M` destabilised (D→0 attractors / rotational);
   restricting to the cross term collapsed a well. No setting gave both C and D clean.
4. **Decoupled construction** (sample bistable at choice=0, choice relaxes to a
   sample-gated target): *reliably* bimodal, but a construction; superseded as too imposed.
5. **Outside-φ joint fit** (`Δz=Lz+Wφ(g·z)+Bu+D(u⊗z)+b`): u=0 stayed bistable, but a
   held-on input still collapsed the second well; the gating interaction fit to ≈0
   (initially because segments were pooled — a bug; even sample-split it didn't keep two
   attractors of a held-on input).
6. **RNN-form inside-φ, recurrent frozen** (`fit_input_inside`, `Δz=Lz+Wφ(g·z+Bu)+b`):
   matched the RNN's `φ(g(Mκ+drive))`; u=0 = autonomous; but a *constant* inside-φ current
   (as the RNN's constant `ff_input`) can't gate choice → input fields ≈ autonomous.
7. **CURRENT — simple per-regime fit:** drop the input model entirely. Each panel
   independently fits one `fit_rnn_flow` to the data **in that regime** (autonomous = DPA
   delay; each input = data while present, window through the response peak/settling).
   Attractors land in the data for autonomous/A/B/NoGo/C/D, and **C/D come out bimodal at
   the lick/no-lick endpoints** naturally. Go keeps one spurious attractor (transient,
   asymmetric response). This is the kept version — simplest and most faithful.

**Key empirical lesson:** a *held-on constant input* over all of state space cannot keep
two attractors (verified for every fitted form); C/D bimodality is a **transient
sample-gated readout**. Fitting each regime's data over the full response window recovers
two attractors *because the data settles at the two endpoints within the window* — i.e.
it's the trajectory endpoints, not a property of a constant-input field.

## Current `--inputs` model (the important part)

> **GOAL.** Produce a **flow-field approximation of the data's dynamics**, both autonomous
> and input-driven, to compare with an RNN model trained on the task (which plots the same
> two objects). Kept deliberately **simple**: each panel is just a flow fit to the data in
> that regime — no input model. (History/dead ends: hand-calibrated drives → frozen current
> inside φ → decoupled construction → outside-φ joint fit → RNN-form inside-φ joint →
> **per-regime fit** (current, simplest). The earlier ones over-engineered the input term.)

8-panel grid: **autonomous + A, B, Go, NoGo, Cue, C, D**, in `input_panels()`.

- **Each panel independently fits one nonlinear rate flow** `Δz = Lz + Wφ(g·z) + b`
  (`fit_rnn_flow`, the same call as `--nonlinear`) to the data in that regime, locates its
  fixed points (discrete-map Jacobian), and overlays the condition trajectories. No input
  encoding / no `B u` term — the "input-driven flow" is literally the flow fit to the data
  while that input is present.
  - **autonomous** — DPA, bins 21–53 (delay, no input) → bistable WM landscape (=`--nonlinear`).
  - **A/B** — sample 0/1, bins 15–29; **Go/NoGo/Cue** — DualGo/DualNoGo/both, bins 30–51;
    **C/D** — test 0/1, bins 57–83. **Windows run through the response PEAK/SETTLING** (not
    just onset) so the data reaches Δz≈0 and the fitted attractor lands IN the data — a
    too-short window cuts off mid-transient and the fit places fps by extrapolation.
  - result: autonomous 2 wells; A/B at the sample wells; NoGo at no-lick; **C/D give two
    attractors at the lick/no-lick endpoints (bimodal, in-support)**. Go keeps one spurious
    extrapolated attractor (its response is transient + asymmetric — peaks then decays).
  - fits use condition means split by sample×test (denoise); shared 92nd-pct `vmax`; fixed
    view `(-2.5,2.5)×(-3.2,3.2)`; trajectories truncated at each window's end.
- **To compare with the RNN:** read off, per panel, the fixed-point structure (number/
  location/stability) and how each input reshapes the field vs the autonomous — against the
  RNN's autonomous and input-driven flows. (Note this is the data's *average* flow per
  regime; single-trial fits are pure noise, R²≈0.01.)

Run: `python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --inputs --correct`
(`--inputs` requires `--task all`; `--stage Naive` needs a Naive `<dum>`). **`--correct`**
restricts to correct trials (default ALL) → `_correct`.

**Flags added 2026-06-16 (denoising the condition means — the thing that actually limits
the fields):**
- **Regime-aware condition averaging** (default, in both `fit_data` for the fields and
  `cond_trajs` for the overlays): group the means by only the factor whose stimulus has
  occurred in the fit window — **sample alone pre-test** (delay/distractor: the test odor
  isn't present yet, so splitting by test just halves N), **sample×test once the window
  reaches the test epoch** (`TEST_ON = 54`). → ~2× trials/mean for autonomous/A/B/Go/NoGo.
  The view-box call `cond_trajs()` (no window) keeps full sample×test so axes aren't clipped.
- **`--smooth SIGMA`** — Gaussian temporal denoise of the latents at the source (commutes
  with averaging); tags `_sm<σ>`. **Cosmetic for the pooled DUM** (n≈2053 means are already
  smooth); **matters only for thin per-mouse DUMs**. Try 1.0–1.5; do NOT chase bistability
  with it on single trials (it overshoots: collapse→repeller, never a clean saddle).
- **`--rescale STD`** — normalise each latent axis to this std. **Required for per-mouse
  DUMs**, whose dPCA latents come out at std≈6–15 vs the pooled ≈2.8, so the FP search box
  `(−2.5,2.5)` and `gain=1` (tuned to the pooled scale) miss the data. Use `--rescale 2.8`.
  With `--rescale`, `--nonlinear`/`--inputs` use a box floored at `±2.5/±3.2` but grown to
  the data so per-mouse wells aren't clipped (`robust_box`).
- **`--mask-support`** — fade the field (heatmap + streamlines) where it is far from the
  condition-mean trajectories it was fit to (KDE over the fit curves; `--mask-bw` fraction
  of box, `--mask-alpha` max fog). Flags **off-manifold extrapolation** so fixed points in
  no-data regions are visibly greyed — the field is fit to a few 1-D curves, so its structure
  away from them is unconstrained. Tags `_msk`. Opt-in (existing figures unaffected).

## `src/pca/dynamics.py` API

`fit_lds` (linear, optional inputs) · `lds_modes` · `cv_predict_r2` ·
`dimensionality_curve` · `boxcar_inputs` · `fit_poly_flow` · `fit_rnn_flow`
(neural `Δz=Lz+Wφ(g·z)+b`, returns `flow, M=[b|L|W]` — fit per regime for both autonomous
and input-driven panels) · `cv_flow_r2` (single-trial velocity R² = noise floor) ·
`cv_condmean_flow_r2` (held-out **condition-mean** velocity R² — the correct metric) ·
`bootstrap_fixed_points` (FP-stability via a caller-supplied resampler) · `flow_fixed_points`
(locate + classify by the **discrete map** Jacobian `|eig(I+J)|<1`).

*Removed 2026-06-15:* `fit_input_current`, `fit_input_rnn_flow`, `fit_input_joint`,
`fit_input_inside` — the input-term models, all superseded by the simpler per-regime fit.

## Single-trial flow fields: why condition means are *required*, not a shortcut (2026-06-16)

Re-examined the single-trial collapse (asked: "if single trials are dPCA projections they
should behave like their mean"). They do for the **linear** dynamics; the **nonlinear
bistable** field is genuinely unrecoverable per-trial. Settled, with diagnostics:

- **Not an estimator (errors-in-variables) bias.** A velocity regression `Δz=Lz` is biased
  when state noise ε is white (ε in `+z_t` and `−z_t` of `Δz`), but the per-bin noise here is
  **temporally autocorrelated** (calcium), so the bias mostly cancels: OLS, an IV (lagged
  instrument), and the condition-mean fit all give the **same slow eigenvalues**
  `|eig(I+L)|≈0.97` (the white-noise EIV "correction" blows up, confirming non-white noise).
  → the **linear/slow dynamics transfer to single trials**. (`exp_single_trial_bias.py`)
- **The limiter is state SNR + a slow per-trial offset.** Per-bin within-condition noise
  std ≈ **3.0** vs sample-well separation ≈ **2.2** (d′≈0.7, 65% single-bin well classify).
  Crucially the noise is a **persistent per-trial offset**: averaging the whole 33-bin delay
  drops its variance only **1.3×** (white noise → 33×) → ≈**1 effective independent sample
  per trial**, so per-trial time-averaging barely helps (d′ 0.72→0.82). (`exp_single_trial_sep.py`)
- **Why the mean works and a single-trial *flow fit* can't.** A flow fit conditions on each
  trial's own **noisy position**; at this SNR position ≠ well, so at a given `z` the field
  mixes A- and B-well trials and the two wells wash out (→ one central attractor). The
  condition mean conditions on the **trial label** and averages ~350 trials whose offsets are
  independent **across** trials → clean wells. So the condition mean is the **minimal
  sufficient statistic** for well identity; the bistable field is an **across-trial ensemble
  property**, not a property of any single trajectory. No smoothing window recovers it
  (collapse→repeller, never a clean saddle).

**Per-mouse follow-up (real co-recorded populations):** single mice carry more single-trial
signal than the pseudo-pop (JawsM15 autonomous CV R² 6.6% vs ~1%) but still noise-dominated.
Built per-mouse dPCA DUMs (`build_mouse_dpca.py`); their latents are ~3× larger (std≈9), so
the FP box/gain miss them → use `--rescale 2.8`. **JawsM15 cond-mean autonomous is then
bistable** (attractors ≈ ±2, saddle ≈ 0 — the "no attractors" earlier was a scale artifact).
**ChRM04 is genuinely NOT bistable** (after fixing the box + using `_ci2` to remove choice-axis
ramp leakage): its sample A/B delay separation is weak (d′≈0.75 vs JawsM15 1.31), so the two
states sit near the origin — per-mouse bistability tracks sample-memory strength.

## Validating & tuning the flow fits (2026-06-16)

The displayed flow is a **condition-mean** object, so the right validation metric is a
condition-mean CV — **`cv_condmean_flow_r2`** (`src/pca/dynamics.py`): stratified split over
trials, build condition means from the train half, fit, and score the **held-out** half's
condition-mean velocity R². (The older `cv_flow_r2` scores single trials = the ~1% noise
floor → tests the wrong object.) `plot_pseudo_flow.py` now annotates every `--inputs` panel:

- **High-velocity input regimes → held-out cond-mean velocity R²** (`panel_cv_r2`,
  shares `regime_group_ids` with `fit_data`). Pooled: **C +0.22, NoGo +0.19, Go +0.16,
  D +0.11, Cue +0.23** — the input-driven flows are genuinely predictive. A/B are single-
  condition (no CV). R² scales with regime speed (|Δz|/bin).
  - **Cue grouping fix (2026-06-16):** the Cue panel's defining variable is the distractor
    (Go vs NoGo), not sample. It was being grouped by sample (regime-aware default), which
    averaged Go+NoGo together → the field missed the cue divergence while the overlay showed
    the Go/NoGo means (mismatch). REG entries now carry an optional `groupby`; Cue uses
    `['tasks']`. Result: Cue → 2 attractors, CV R² **+0.06→+0.23** (pool7 −0.00→+0.29).
- **Autonomous = slow attractor regime → velocity R² is uninformative** (Δz≈0, negative
  regardless of fit). Validate by **FP reproducibility instead**: bootstrap
  `P(2 attractors)` (`bootstrap_fixed_points`, n_boot=40), shown on the panel. **Pooled
  P(2)=0.72.**

**Tuning (gain×ridge sweep with the new metric, `exp_tune_global.py`):** the **current
defaults gain=1.0, ridge=0.2 are already near-optimal** — mean input R² +0.245 vs best
+0.271 (gain 2.0, ridge 0.5), within noise, all settings bistable. No change made.

**Per-mouse validation (the metric earns its keep):**
- **JawsM15 autonomous P(2)=0.93** — bistability *more* robust than pooled. But its **input
  panels are NOT velocity-validated** (R²≈0 to −0.5): only ~288 trials → held-out cond-mean
  velocity is noise.
- **ChRM04 P(2)=0.05** (ci0 and ci2) — **reproducibly monostable**, quantitatively confirming
  the weak sample memory (d′ 0.75). Input R² ≤ 0 too.
- **Verdict: per-mouse, only the autonomous FP-reproducibility is trustworthy; the per-mouse
  input-driven panels are not** (thin trials). The pooled population is the object for the
  input flows.

**9-mouse autonomous bistability survey** (correct trials, `--inputs --cond-mean --rescale 2.8
--correct`, P = bootstrap P(2 attractors)):

| mouse | P(2 att) | | mouse | P(2 att) |
|---|---|---|---|---|
| JawsM18 | **0.97** | | JawsM06 | 0.07 |
| JawsM15 | **0.95** | | ACCM04 | 0.05 |
| ChRM23 | **0.90** | | JawsM01 | 0.05 |
| JawsM12 | 0.45 (borderline) | | ChRM04 | 0.03 |
| ACCM03 | 0.10 | | | |

→ **the WM double-well is reproducible in only 3/9 mice** (JawsM18, JawsM15, ChRM23), borderline
in 1, monostable in 5. Per-mouse bistability tracks **sample-memory strength** (JawsM15 d′1.31→
bistable; ChRM04 d′0.75→monostable). The pooled bistability is carried by the strong animals.

**Sub-pool (drop JawsM01+JawsM06 → `..._dpca_pool7`, `build_subpool_dpca.py`):** pooled autonomous
**P(2) 0.72 → 0.95** (all-trials; 0.93 correct), and input flows sharpen (C R² 0.22→0.34, D
0.11→0.36 all-trials; C/D both bistable). **CAVEAT — partly circular:** the mice were dropped
*because* they were monostable, so a stronger pooled bistability is partly outcome-selection,
not a free improvement. JawsM01 has an independent reason (least data, 4 days); JawsM06 was
outcome-selected. Honest framing: "the pooled structure is carried by the animals that have it
and isn't an averaging artifact," **not** "removing 2 arbitrary mice improves the pool."

Diagnostics: `exp_tune_flow.py` (autonomous gain×ridge), `exp_regime_r2.py` (per-regime R²
vs speed), `exp_tune_global.py` (global tune).

## Method review & the "slow circular manifold" question (2026-06-16)

A calm re-read of the model + a test of the hypothesis that **the autonomous dynamics is a
slow circular manifold with two wells.**

**Model critique (the rate network `Δz=Lz+Wφ(g·z)+b`, fit per regime).** Sound and now
validated, but three caveats: (1) **off-manifold extrapolation** — ~10 params fit to 2–4
near-1-D condition-mean curves, so the field (esp. the saddle and any off-path attractor) is
unconstrained away from the data → added `--mask-support` to show it; (2) **`gain` partly
sets the multistability** (φ saturation vs data scale) — bistability is "supported at gain≈1,"
not unconditional; (3) the velocity in the slow autonomous is ~noise (its cond-mean CV R² is
negative), so attractor *locations* (trajectory endpoints) are robust but the *arrows* are
weakly constrained.

**Why a gradient/potential flow `Δz=−∇V` is the WRONG fix (user's key point).** At
|λ|≈0.96–0.98 the dynamics are a **slow manifold**, not steep wells. A gradient flow can
encode the manifold's *geometry* (shallow valley) but **cannot carry slow flow *along* it**
(gradient systems only descend V and stop) — so it would erase exactly the slow-manifold
transport. The non-curl-free rate network is the more faithful form. Don't constrain to a
potential.

**Empirical test of the ring (pure data, no model — `exp_manifold.py`, `exp_manifold_pca.py`):**
the same 4 sample×test condition means, viewed three ways:
- **dPCA sample×choice** → a clean **cross/H**: two wells on the sample axis (delay, choice≈0),
  choice an orthogonal branch at test. But dPCA **orthogonalises sample⊥choice**, so curvature
  is removed *by construction* — this plane *cannot* show a ring.
- **raw PCA (top PCs)** → one shared **curved arc = the condition-independent timing/ramp**;
  task structure is tiny (Sample demoted to PC6). The dominant slow manifold is *timing*, not WM.
- **ramp-removed PCA (role planes)** → task geometry now has **real curvature** (the dPCA cross
  is partly a demixing artifact), but it's a curved **fan** to the test corners, **not a closed
  ring with two defined wells**; the wells are a weak sample modulation.

**Verdict (defensible claims):** (1) a slow low-D **timing manifold** (condition-independent);
(2) **two-well sample bistability** in the demixed memory axis (3/9 mice, tracks sample-memory
strength); (3) choice resolved orthogonally at test. A **"slow circular manifold with two
wells" as one object is over-reading** — it merges the big timing signal with the faint memory
signal, and at this SNR ring-vs-cross-vs-noise is **underdetermined**. Only claim the ring if
the **RNN** predicts it *and* the data trajectories are shown to traverse it (they don't here).

### Geometry resolution — radial/angular + the defensible figure (2026-06-17)

Pushed the "is it a ring / 2-D / line?" question with **model-free** tools (no fitted flow):

- **Radial/angular geometry — canonical: `exp_bootstrap_polar.py` → `polar_bootstrap.png`.**
  Bootstrap-CI on the **condition-mean trajectory** (unbiased, no bundling, no arbitrary k): the
  sample-axis radial profile `v_rad(r)` goes **outward below r₀, crosses zero, inward above** →
  a **significant preferred radius (well) r₀ = 1.02 [95% CI 0.59–1.36]** (100% of bootstraps
  cross zero); **angular speed brackets 0 at all radii → no rotation**. ⇒ a **two-well / line-like,
  non-rotational** geometry along sample, **not a ring**. (NB an earlier *raw single-trial* sector
  decomposition gave a misleadingly all-inward, no-crossing profile — that was the EIV/offset bias;
  see the ceiling demo. The bootstrap condition-mean profile is the clean version.)
- **Reconciliation of the gradient-flow point:** the autonomous flow is **non-rotational**
  (angular speed ≈0, all FP eigenvalues real) → it *is* gradient/potential-compatible after all;
  the earlier worry that a potential would miss "slow flow along the manifold" isn't borne out
  (there's no angular drift, just contraction to wells). We keep the rate-network because it
  already works, not because a potential would fail.
- **Single-trial vs condition-mean (the noise floor; computed in `fig_manifold.py`):**
  single-trial delay occupancy is **one contractive 2-D blob** — the A/B
  wells are *sub-noise* (std≈3 ≫ separation≈1.4), so single trials can't resolve the geometry;
  only condition-averaging does.

**Per-mouse polar — reworked 2026-06-18.** The earlier per-mouse claim ("outward→0→inward for
most mice", `analyze_polar_permouse_profile.py`) was **wrong: it used k=6 trial-bundling, which
sits in the EIV/regression-to-mean regime** and *manufactures* a consistent inward branch.
Investigated thoroughly (`exp_polar_bug.py`, `exp_polar_resample.py`):

- **The three resampling methods are equivalent.** *Bundling* (disjoint groups of k), *bootstrap-
  of-condmean* (resample trials → full condition mean per resample), and *CV-of-condmean* (fold
  means) give the **same radial profile at matched averaging unit** — they overlap where they share
  support, and all agree on the well crossing. The only real axis of variation is **k**: small-k
  bundling (k=6) is EIV-biased (whole curve pulled negative, spurious inward branch); as k grows it
  lifts onto the condition-mean profile. So "bundle vs bootstrap" was a non-issue — the discrepancy
  I kept hitting was purely small-k EIV. **Figures:** `polar_methods_compare.png` (pooled, 3 methods
  + individual traces), `polar_methods_permouse.png` (per-mouse, 3 methods overlaid).
- **Canonical per-mouse estimator = bootstrap-of-condmean** (lowest EIV: each replicate is a full
  k=all condition mean; honest sampling-distribution CI; no tuning knob). CV is an equivalent cross-
  check; **k=6 bundling is retired as EIV-biased.** `fig_polar_radang_permouse.py [boot|bundle|cv]`
  → `polar_radang_permouse_{boot,bundle,cv}.png` plots **radial v_rad(r) AND angular v_ang(r)** per
  mouse, ±95% CI; `fig_polar_radang_condmean.py` → `..._condmean.png` is the raw point-estimate
  (two condition means, no resampling) showing the per-mouse scatter directly.
- **Honest per-mouse result:** with the unbiased methods the **+→0→− shape is NOT reliably present
  per animal** — bands are wide (15–60 DPA trials/sample), crossings unstable/absent. The well is
  statistically solid **only pooled** (r₀ ≈ 1.0–1.1). **Non-rotation IS robust in every mouse**
  (angular CI brackets 0) — the one per-animal feature that replicates.
- **Rescue of the per-mouse well (`fig_polar_wellfit_permouse.py` → `polar_wellfit_permouse.png`):**
  don't hunt the noisy empirical crossing — **fit the restoring flow** `v_rad = b·(r₀−r)` (linear in
  r) to the condition-mean delay trajectory; the **outward approach alone determines r₀** (no need
  for beyond-well data), and every timepoint contributes. Test = **slope b significantly < 0**
  (restoring → settles at a finite preferred radius). Theil-Sen + bootstrap CI. Result: **4/9 mice
  individually significant** (JawsM15, JawsM18, ChRM23, ChRM04 — incl. 3 of the pooled-bistable set);
  3 more (JawsM12, JawsM06, ACCM04) negative-leaning but under-powered; only JawsM01 (n=15) & ACCM03
  genuinely flat. r₀ *location* stays poorly pinned per mouse (wide CI) but well *existence* is
  detectable. Does **not** cleanly track bistability (that's the P2/barrier survey).

NB pitfalls caught earlier: an **RNG-reuse bug** (re-bundling with an advanced RNG faked a clean
split) and a **fixed r-range** that mislocated r₀ onto baseline jitter when a mouse's well sat
beyond range. (Why temporal/Kalman denoisers can't substitute for averaging is the ceiling demo
below — and why the k→∞ / condition-mean limit is the right target.)

**Polar profile from the FITTED FLOW FIELD — cleanest view (2026-06-18).** All the above bins
the velocity *along the trajectory* (only samples r where the data goes — the condition mean stops
at the well, so the inward branch is missing/EIV-prone). Better: **fit the rate-net flow**
`Δz=Lz+Wφ(g·z)+b` (`fit_rnn_flow`, gain 1.0, ridge 0.2) to the 2 DPA-delay condition means, then
**evaluate the field along the axis** at every radius: `v_rad(r)=flow(r·û)·r̂`, `v_ang(r)=flow(r·û)·t̂`
(both wells averaged). Bootstrap over trials → 95% CI. Gives a **smooth profile defined at every r,
including the inward branch the mean never reaches**, denoised (one field fit to all timepoints).
Scripts: `fig_polar_flowfield_permouse.py [sample|choice]` (3×3), `fig_polar_flowfield_pooled.py`
(pooled, sample|choice side-by-side) → `polar_flowfield_{permouse_sample,permouse_choice,pooled}.png`.
- **Pooled:** sample axis = clean well **r₀=1.15** (radial +→0→−), angular ≈0; **choice axis flat,
  no well, extent only r≲1.0** (choice is barely coded in the delay) → confirms **sample ⟂ choice**,
  the well is sample-specific.
- **Per-mouse:** clear restoring wells in **JawsM15, JawsM18, ChRM23, JawsM12, ChRM04**; weak/flat
  in ACCM03, ACCM04, JawsM06, JawsM01. (More mice read as wells here than via the noisy trajectory
  bins, because the field interpolates — but it's the same data, just smoothed.)
- **Caveats:** the field is fit from only 2 condition-mean trajectories, so the small-r/large-r ends
  lean on the rate-net form; **beyond the max data radius (~2.3 sample / ~0.9 choice) it is pure
  extrapolation — flagged grey** in the figures. Existence of the sample well is robust; exact r₀
  per mouse keeps the wide CI.

**The denoising ceiling = the condition mean** (`exp_ceiling.py` → `denoising_ceiling.png`,
*demonstrated*): the dominant noise is a **slow per-trial offset** (RMS 3.5 ≫ well sep 1.4),
and it is reduced **only by averaging across trials** — bundling drops it exactly as raw/√k
(3.5→0.4 at k=64), while **temporal Gaussian smoothing (any σ, ~4% at σ=8) and a Kalman/RTS
smoother (~0%) leave it intact** (a slow offset is "signal" to any temporal/dynamics filter).
And the bundled flow metric **converges to the condition-mean value as k→∞** (−0.064 → +0.014).
⇒ when per-trial variability is noise, the **MMSE-optimal per-trial estimate shrinks to the
condition mean**, so **no single-trial denoiser (GPFA, Kalman, shrinkage) can beat it here** —
the condition mean is the ceiling, and bundling just approaches it. This is the quantitative
justification for "condition means are required, not a shortcut."

**Defensible figure** (`fig_manifold.py` → `figures/pseudo/flow/wm_manifold_defensible.png`):
two panels separating *measured* from *modelled* — (A) single-trial delay occupancy (one blob,
wells sub-noise) and (B) condition-mean state geometry over the full trial, mean ± SEM
(sample-coded delay @ choice≈0 → choice-resolved corners). **This is the headline manifold
result.** Honest scope: the data supports a **2-D condition-mean geometry** (sample memory ⟂
choice), **non-rotational, elongated/line-like, with sub-noise bistability**; finer claims
(line vs circle vs deep wells) are under-determined at this SNR and are model extrapolation.

## Nonlinear fixed-point eigenvalue analysis (2026-06-17)

The right eigenvalue analysis for the **nonlinear** flow (not the global LDS): linearise at
each fixed point — `flow_fixed_points` already returns the discrete-map Jacobian eigenvalues
`eig(I+J)`; convert to τ (s) and freq (Hz) like `lds_modes` (dt=1/6). Scripts:
`exp_eig_nonlinear.py` (per-fixed-point modes), `analyze_bistability.py` (9-mouse survey +
summary figure `figures/pseudo/flow/bistability_summary.{png,svg}`).

- **Every fixed point has explicit fast/slow structure** — a **slow** direction along the
  sample/memory axis (pooled τ≈5–8 s, |λ|≈0.97–0.98) and a **fast** one along choice / off the
  manifold (τ≈0.9 s, |λ|≈0.84). That *is* fast contraction onto a slow manifold, quantified;
  the slow τ matches the LDS task timescale (3–6 s).
- **All wells are non-rotational nodes (freq=0)** → **reconciles the "non-rotational" claim**:
  the **task/WM dynamics are non-rotational** (correct), and the slow rotational mode (0.038 Hz)
  in the global top-6 LDS is the **condition-independent timing**, not the memory. Update the
  loose "f≈0 everywhere" wording accordingly (WM yes; global timing has weak rotation).
- **Barrier height = saddle's unstable |eig(I+J)|** quantifies bistability *depth*: pooled
  **1.018** (nearly flat), bistable mice **1.15–1.24** (ChRM23 deepest 1.236). Even the robust
  mice have **shallow** barriers — bistability is marginal throughout (a near-flat slow manifold
  with weak wells, not deep attractors).
- **9-mouse survey (summary figure):** 4 bistable — ChRM23 (P2 0.90), JawsM18 (0.88), JawsM12
  (0.63), JawsM15 (0.63) — 5 monostable. P(2) tracks sample-memory **d′ positively but
  imperfectly**: **ACCM03 is the outlier — highest d′ (1.43) yet monostable (P2 0.06)**. So
  strong sample memory is necessary-ish but **not sufficient** for a bistable flow.
- **Caveat:** P(2) is **box/bootstrap-sensitive** (JawsM15 0.63 here with the fixed box vs 0.95
  in the robust-box `--inputs` survey). The bistable *set* (ChRM23/JawsM18 robust; JawsM12/M15
  borderline) is stable; the exact P(2) value is not — report it as a range, not a point.

## Slow manifold vs genuine 2D double-well (2026-06-18)

**Question:** is the geometry (A) a real 2-D double-well (two isolated point attractors + saddle)
or (B) a slow 1-D manifold carrying two shallow wells? **Discriminator = the dynamics BETWEEN the
wells, not at them** (`exp_slowmanifold_test.py`): walk the inter-well (sample) axis; at each point
take the local Jacobian, split eigen-rates into **longitudinal** (along axis) vs **transverse**
(onto axis), `rate=-ln|eig(I+J)|/dt`, and read the flow speed.
- **(B) slow manifold** ⇒ *persistent* fast-transverse / slow-longitudinal gap ALL ALONG the axis +
  low speed everywhere along it + near-flat saddle.
- **(A) 2D double well** ⇒ gap only near the fixed points; speed rises (fast roll) between them.

**Pooled = decisively (B) slow manifold + shallow wells:** mean inter-well speed only **8% of max**;
transverse contraction fast & ~constant (**rate ≈+0.95/s, τ≈1.0 s**), longitudinal near-marginal
(**τ≈26 s**, expanding only at the saddle); persistent gap **≈4×**; near-flat barrier (matches the
saddle |λ|≈1.018). I.e. the **sample axis is a slow 1-D manifold** (fast ~1 s contraction onto it,
slow ~26 s flow along it) with two shallow wells.

**Honest scope — what is actually constrained:** the *longitudinal* facts (low along-axis speed,
shallow wells, flat saddle) are **data-supported** (the trajectory's own speed). The *transverse*
fast-contraction — the thing that makes it a "manifold" not just a slow 1-D system — **leans on the
rate-net form + ridge**, because the **choice/transverse direction is barely explored in the delay**
(choice extent ≈0). So "slow + shallow wells along sample" is robust; "fast transverse contraction
onto a manifold" is consistent but weakly constrained.

**Per-mouse (well mice, `slowmanifold_test_<mouse>.png`): underdetermined.** Only **ChRM23** cleanly
replicates (gap 3.3×, speed 0.40 → slow manifold); JawsM15 (gap 3.1× but speed 0.62) and JawsM12
(gap 2.5×, speed 0.18) are **intermediate**; JawsM18 (gap 1.0×) and ChRM04 (monostable, test
ill-defined) read 2D-double-well but are the noisiest fits. ⇒ **the slow-manifold verdict is a
POOLED result**; per mouse the transverse quantity is too SNR-limited and well placement too
asymmetric to decide. (Same pooled-clean / per-mouse-underdetermined pattern as the polar well.)

**CI-removal robustness — the slow-manifold reading is ramp-driven (important, 2026-06-18).** The
flow DUM is q=0 (CI not removed; flow plane = sample × sample:test, see "what dpca" below). But the
condition-independent timing ramp **leaks into the plane and inflates the transverse contraction**.
Re-running the test on CI-removed DUMs (`exp_slowmanifold_test.py pooled {1,2}`, using the existing
`…f-sample-test_ci{q}_dpca` pooled DUMs):

| q (CI removed) | transverse τ | longitudinal τ | gap | inter-well speed | verdict |
|---|---|---|---|---|---|
| 0 (raw) | **1.05 s** | 25.6 s | 4.0× | 0.08 | SLOW MANIFOLD |
| 1 | 1.27 s | 31.3 s | 2.9× | 0.10 | intermediate |
| 2 | **12.3 s** | 9.1 s | 0.4× | 0.18 | 2D double well |

As the ramp is stripped, the **fast transverse contraction collapses (τ 1 s → 12 s)** and the
manifold separation vanishes. ⇒ **What's robust to CI removal: bistability** (two attractors + saddle,
low-ish inter-well speed at all q). **What's NOT robust: the "slow manifold" characterization** — it
was substantially the condition-independent timing ramp, not the sample-memory dynamics. After
removing the ramp the condition-dependent geometry is **two wells without a strong fast-transverse
manifold** (≈ a shallow 2-D double-well). This confirms the earlier caveat (transverse = least
data-constrained) with a concrete cause. **Bottom line:** report the bistable two-well structure as
the solid result; treat "slow manifold" as ramp-dependent, not a property of the memory subspace.

**Per-mouse well mice across q (`exp_slowmanifold_test.py <mouse> {1,2}`,
`fig_polar_flowfield_permouse.py sample {1,2}` → `polar_flowfield_permouse_sample_ci{q}.png`):**
- **Radial well (restoring flow along sample) is CI-robust** — the polar flow-field still shows
  radial +→0→− for all five well mice at q1 and q2; angular ≈0 throughout.
- **Bistability mostly CI-robust** — JawsM18/ChRM23/JawsM12 keep 2 attractors at all q; JawsM15 loses
  its 2nd well at q2; ChRM04 monostable throughout.
- **Slow-manifold gap is NOT determinable per mouse** — the transverse/longitudinal gap jumps
  erratically across q (ChRM23 3.3→1.1→3.7×; JawsM15 q2 reads 15.5× but is monostable ⇒ inter-well
  metric ill-defined). Underdetermined at every q.
⇒ **CI-robust defensible result = non-rotational two-well restoring structure along sample** (radial
well + bistability); the **slow-manifold framing is ramp-dependent and not stable per mouse**.

## Low-rank (rank-2) reduced flows + input model (2026-06-22)

A reduced **rank-2 RNN** flow as an alternative to the generic rate-net, to compare the data flow to
the project's rank-2 RNN. Script: **`pca/fig_dpca_flow_lowrank_shared.py [--independent] [--dum DUM]`**
→ `figures/pseudo/flow/lowrank/dpca_lowrank_{shared,independent}.{png,svg}`.

**Model (gain-modulated LINEAR form — NOT elementwise tanh).** Per latent dim d:
`ż_d = −z_d + S(z)·(A z)_d + c_r`, with **average gain** `S(z)=⟨φ′(√Δ·ξ)⟩_{ξ~N(0,1)}` (Gauss–Hermite,
20 nodes), `Δ = a²‖z‖²+δ`. This is the mean-field reduction of a rank-2 tanh RNN (z≈κ overlaps,
A≈loading-overlap matrix, S≈⟨φ′⟩). **Why not `Wφ(gz)`:** the latents are rescaled to std 2.8, so
elementwise `tanh(gz)` SATURATES (|gz|~3 → ±0.99) → recurrent ≈ const → bad fit; the gain-modulated
*linear* `S(z)·Az` never saturates (Az linear, S∈(0,1] shrinks as ‖z‖ grows).

**Three modes. Default auto-selects: pooled → independent (best descriptive geometry); per-mouse →
partial (regularized).** Override with `--shared`/`--independent`/`--partial`.
- **partial**: **A_r = A_shared + ΔA_r**, with the per-regime deviation `ΔA_r`
  **ridge-penalized toward 0** (CV-tuned `λ`). Interpolates shared↔independent: large λ → shared,
  small λ → independent. The ridge is **safe** (shrinks ΔA toward the shared A, which stays free to
  balance the leak — unlike ridging A toward 0).
- **`--shared`**: ONE recurrent A + per-input current (additive `c_r` + per-mode in-gain `h_r` that tilts).
- **`--independent`**: each regime its OWN `A_r + c_r`.

**Fit:** given `(a,δ[,λ])` the model is **linear in A and currents** (closed-form LS; partial = ridge on
the ΔA block only). `(a,δ,λ)` **CV-tuned** (5-fold over trials, pooled held-out condition-mean velocity
R²; best **a≈0.2, δ≈0.3–2, λ≈1 pooled**). Regimes/windows = same as `--inputs` (autonomous DPA 21–54
by sample; A/B 15–30; Go/NoGo/Cue 30–52, Cue by `tasks`; C/D 57–84).

**Result — partial pooling wins (bias/variance):**
- **partial (default): pooled CV vel-R² +0.14** — beats shared (+0.07) AND independent (−0.08) — with
  high per-regime R² (C +0.50, D +0.51, Cue +0.47): **captures the C/D diagonals AND generalizes.**
- **shared**: CV +0.07, currents on the right axes (A/B↔sample, Go/NoGo↔choice), but **can't tilt C/D**
  (a shared A only translates; the C/D diagonal is a sample×test interaction).
- **independent**: high in-sample R² (A +0.73,…,D +0.70) and C/D tilt, but **CV −0.08** (mild overfit).
⇒ **pooled: use independent** — it's the most faithful *descriptive* geometry (C/D diagonals, Go/NoGo
choice push, bistable autonomous); its −0.08 CV *overstates* the issue (velocity R² is weak/low-ceiling,
autocorrelated, autonomous≈0 by construction; the pooled per-regime geometry is well-estimated & stable).
**per-mouse: use partial** — there independent genuinely overfits (CV −0.1…−0.9). Legend on figure
(★ attractor, □ saddle, ✖ repeller; A/B/Go/NoGo trajectories).

**Key model lessons (don't re-derive):** (1) use the **gain-modulated LINEAR** form `S(z)·Az`, NOT
elementwise `Wφ(gz)` — at std 2.8 tanh saturates → recurrent ≈ const → fit fails. (2) **never ridge A
toward 0** (the −z leak must be balanced; ridge → leak dominates → R²→−40); ridge only the *deviation*
ΔA toward the shared A (partial mode). (3) a **constant current only translates**; tilt needs the
recurrent (ΔA) or in-gain `h` to change — a shared A alone can't produce the C/D diagonal.

**Per-mouse** (`--dum …_dpca_<MOUSE>`): independent **overfits** (CV −0.1 to −0.9); shared and partial
sit at the noise floor (−0.09 to +0.03). **Partial auto-adapts λ per mouse** — λ≈100 (≈shared) for
thin/noisy mice (JawsM01/06/12, ChRM23), λ≈5 (allows deviation) for cleaner ones (JawsM15/18, ACCM03/04,
now positive CV). Smoothing the per-mouse means does NOT help (shrinks the velocity signal scored by CV).

**On the overlaps (for contrast):** the same rank-2 fit is **data-limited** (held-out vel-R²≈0, gain
unconstrained → linear) — the overlaps CCGD velocity SNR is the ceiling, not the model. The rank-2
model belongs on the **dPCA** latents (true-ish κ, high SNR). See `docs/overlaps/overview.md`.
**Verbatim port done 2026-06-30** (`overlaps/fig_overlaps_flow_lowrank_shared.py`, dPCA script with
only the data section swapped): confirms this — pooled Expert CV vel-R² ≈0 (partial +0.035 > indep
−0.015, same mode-ordering as dPCA), and **only the sample-input regimes validate** (A +0.93, B +0.78,
Cue +0.30); **autonomous weak (+0.02–0.11) and C/D do NOT validate** (overlaps test code is weak).
So the fields render but stay SNR-limited; the dPCA latents remain the right object for the rank-2 model.
**"Improve the fits?" tested 2026-07-01 — velocity R² is NOT improvable** (temporal `--smooth`
0.035→0.008 and multi-bin `--vstep` 0.035→−0.13 both fail; the chord velocity ≠ the model's tangent →
`--vstep` default 1). **FP placement (corrected 2026-07-01):** mark fixed points at the condition-mean **trajectory
endpoints** (subproject `plot_flow2d` convention) so trajectories terminate at them — root-finding is
displaced at this SNR (at CV gain the field's fp sits on the A endpoint; B's shallow well unclassified).
**RETRACTED:** forcing autonomous bistability by raising the gain to hit bootstrap P₂≥0.5 ("P₂=0.62 at
gain 0.7") was WRONG — the raised gain put a 2nd attractor OFF the B endpoint (the "fp not where the
data goes" bug). Honest: the overlaps autonomous is monostable-leaning **on the TEST-trained plane** (B buried near the
saddle at +0.46). **IMPROVED 2026-07-01**: read the codes on the **DELAY-trained axis** (draw the delay
flow on the delay axis — CV-honest, smoother, better-separated) → B moves to +1.25 (a real well) and the
autonomous is **faithfully bistable, 2 attractors at the A/B endpoints**, pinned there by endpoint
anchoring (v=0 anchors in the LS, `--anchor`). The plane change (not gain-raising) is what makes B a
well; anchoring alone on the TEST plane can't (B too near the saddle). Defaults now `--train delay
--anchor 8`.

## "Is rank-2 good enough?" — reduced-rank sufficiency test (2026-07-01) — ANSWER: NO by prediction

`pca/exp_rank_dpca.py [Naive|Expert] [horizon]` → `figures/pseudo/flow/rank_sufficiency_<STAGE>.png`.
With dt=1 bin the discrete transition `z_{t+1}=A z_t+b` has **A ≈ the recurrent connectivity**, so
constraining **rank(A)≤R** (reduced-rank regression) tests the connectivity rank directly. Sweep R,
score **held-out `horizon`-step predictive R²** (CV over trials) on the dPCA latents (D=8).

**Result (Expert, 3-step):** the curve **rises smoothly to full rank — NO elbow at 2.** Full 8-D:
rank1 +0.49, **rank2 +0.62**, rank3 +0.71 … rank8 +0.92 (full); rank-2 = **67% of full**, ≥95% only at
rank 7. Task 4-D (sample+choice): rank2 +0.52 = **62% of full-4D**, ≥95% at rank 4. ⇒ **rank-2 is NOT
sufficient to predict the latent dynamics** — it is genuinely higher-D (each dPCA marginal carries its
own timescale). **Reviewer verdict:** "rank-2 is good enough" **fails the predictive-sufficiency test**;
rank-2 is a **modeling choice** (to match the task-trained rank-2 RNN / to draw the sample×choice
portrait), NOT a validated property. If the claim is to survive it must be on a **task-computation**
criterion (does rank-2 reproduce the sample×choice fixed-point structure / the WM-subspace flow after
stripping the CI time-ramp & test dims), not "predict all latent variance." NB the slow trajectories are
also fit equally well by a LINEAR flow, so the trajectory fit can't distinguish bistable from monostable
either — see `docs/overlaps/overview.md` (`--compare` caveat).

**Task-computation version → RECONCILED (2026-07-01, `pca/exp_rank_task.py`).** The velocity/dynamics
criterion FAILS (fit a fixed connectivity to condition-mean velocity: negative R² even at full rank —
the delay is near-stationary → velocity R² at the noise floor; over the trial it's input-driven). So ask
the well-posed GEOMETRIC question instead: the **dimensionality of the condition-mean (sample×test)
trajectory manifold** (variance by PC, top-2 %, participation ratio PR). Result: **the task states live
on a ~2-D manifold** — WM/choice subspace **top-2 PCs = 94% (Expert) / 92% (Naive), PR ≈ 2.2**; even the
full 8-D condition-mean states have PR ≈ 2.2, top-2 ≈ 82%. ⇒ **RECONCILIATION: rank-2 IS good enough for
the task-computation STATE GEOMETRY** (what the sample×choice flow portrays) — the WM/choice states are
2-D — even though rank-2 is NOT sufficient to predict the full temporal DYNAMICS (test coding + CI ramp +
transients add dims). So claim: "the WM/choice computation occupies a 2-D manifold (rank-2 geometry)",
NOT "the latent dynamics is rank-2." `exp_rank_task.py` → `figures/pseudo/flow/rank_task_manifold.png`.

## No-lick push & choice readiness — SETTLED CONCLUSIONS (2026-06-23)

> **READ THIS FIRST. It supersedes the detailed audit-trail log further down in this section**, which
> records the iterative path (including claims that were later corrected/retracted — kept only for the
> reasoning and the "don't re-try" warnings).

**Two real effects + one retracted artifact:**

1. **HEADLINE (robust): the WM delay state is pushed into no-lick, deepening with learning.** The `tasks`
   marginal **is a lick/no-lick ACTION axis** (DualGo=lick +0.96; DualNoGo/DPA=no-lick negative; tracks the
   actual lick; not ⊥ to choice, decoder cos +0.22). The DPA WM delay state sits at the no-lick end and is
   pushed deeper with learning: signed tasks depth **ALL trials (DEFAULT) Naive −1.28 → Expert −1.84,
   Wilcoxon p=0.027, 7/9 mice**; **correct-only (`--correct`) Naive −1.30 → Expert −1.88, p=0.012, 8/9**
   (mouse-bootstrap CI [+0.29,+0.87]). **Default switched to ALL trials 2026-06-23** (no outcome selection;
   consistent with the all-trials decoders) — the effect is robust to trial selection (≈0.55 deepening
   either way). **sample-memory axis FLAT (Δ−0.04) = specificity control**; confirmed in raw ΔF/F (r=0.997,
   not a normalisation artifact); robust to CI removal (q0/q1/q2). SUPPORTS the paper hypothesis. Stat figure:
   **`fig_dpca_flow_learning_ingain_permouse.py [--correct]`** (`..._summary[_correct].png`); flow figure:
   **`fig_dpca_flow_lowrank_shared.py --push`**.

   **BASIS-ROBUSTNESS LADDER (2026-06-23) — answers "is the Naive offset just borrowed from the Expert
   axis?".** The canonical stat reads BOTH stages through the per-mouse **Expert** basis. Re-tested with
   stage-specific axes, orienting each by an EXTERNAL lick anchor (DualGo>DualNoGo), NOT by the DPA-delay
   sign (orienting by the tested quantity is circular):
   | design | Naive→Expert | Wilcoxon | mouse-boot CI | both no-lick |
   |---|---|---|---|---|
   | shared per-mouse **Expert** basis (canonical, all trials) | −1.28 → −1.84 | 0.027 | — | — |
   | **own** per-mouse refit per stage (Naive-fit Naive, Expert-fit Expert) | −0.35 → −0.63 | **0.73** | — | 5/9 |
   | **pooled** stage-axis (pooled Naive-fit vs pooled Expert-fit) | −1.19 → −1.50 | 0.074 | **[−0.56,−0.08]** | **8/9** |
   - **Own per-mouse refit FAILS — axis-sign noise, not absence of effect.** For weak-lick mice (ChRM23,
     ACCM03, ACCM04) the per-mouse tasks axis has ~0 Go−NoGo separation (anchor 0.01–0.42) → its sign is
     random → depths scatter (ChRM23 −1.10→+2.50). ChRM04 is a genuine exception (strong anchor but DPA-delay
     on the LICK side, weak sample memory). Where the per-mouse lick axis IS well-estimated (5 Jaws mice), all
     5 land no-lick and match the shared-basis depths. ⇒ single-mouse own-basis is too noisy to carry this
     comparison — **the canonical analysis uses a shared/pooled axis for exactly this reason.** **Do NOT use
     `fig_dpca_nolick_ownbasis.py` for a claim** (kept as the demonstration of the failure).
   - **Pooled stage-axis CONFIRMS the Naive offset is NOT an Expert-basis artifact.** Read through a
     Naive-DEFINED pooled axis (estimated from all 9 mice → stable direction, ONE sign), Naive sits at −1.19,
     **8/9 mice no-lick** (anchors strong: Naive +1.07 / Expert +2.18). The learning deepening survives
     (change −0.31, **mouse-bootstrap CI [−0.56,−0.08] excludes 0**) but is only marginal on per-mouse Wilcoxon
     (p=0.074) — expected, since reading Naive through a less-developed Naive axis dilutes vs the cleaner shared
     Expert axis. So the canonical shared-basis figure (p=0.027) is the *powerful* estimate; the pooled
     stage-axis is the *conservative* one proving the Naive offset isn't borrowed. Script:
     **`fig_dpca_nolick_pooledbasis.py [--correct]`** → `dpca_nolick_pooledbasis[_correct].png`. Per-mouse
     Naive-fit DUMs built via `build_mouse_dpca.py --stage Naive` (now supports `--stage`).
   - **SPECIFICITY: the delay push is on the TASKS (action) axis, NOT the CHOICE (sample:test) axis (2026-06-23).**
     Re-ran the ladder on the choice axis (`--axis choice` on both basis scripts; external anchor =
     match(lick)>nonmatch at the TEST window, never the delay sign). **Choice-axis DPA-delay condition-mean
     depth ≈ 0 in BOTH bases** (pooled +0.14→−0.04, mouse-boot CI [−0.68,+0.32], 2/9 no-lick; own +0.25→−0.07)
     — and the choice axis is WELL-anchored (match−nonmatch +1.8/+2.1), so the ≈0 is a real absence of a delay
     displacement, not an estimation failure. Contrast tasks-axis CI [−0.56,−0.08]. ⇒ the delay no-lick push is
     **action-axis-specific**; choice is resolved at test, not held in the delay (consistent with sample⟂choice).
   - **DEPTH ↔ individual PERFORMANCE = essentially NULL at n=9 (2026-06-24, `fig_dpca_depth_vs_perf.py`).**
     Per-mouse |Expert tasks-depth| vs accuracy: DPA `performance`, Go/NoGo `odr_perf` (the distractor; NaN on
     DPA). **Levels:** depth vs DPA r=+0.46 p=0.21 (weak positive TREND, n.s.); Go r=+0.10, NoGo r=−0.22 (null).
     **Changes (Δ|depth| deepening vs Δaccuracy):** ΔDPA r=−0.06, ΔGo +0.16, ΔNoGo −0.33 — **all null**. So the
     cross-sectional DPA trend does NOT carry into the learning change (biggest deepeners ≠ biggest improvers:
     ACCM03 Δd+1.31/ΔDPA+0.26 vs ACCM04 Δd−0.02/ΔDPA+0.17), and splitting GNG into Go/NoGo reveals no hidden
     effect. ⇒ the push's link to behaviour is at the **population/average** level (robust paired deepening
     p=0.027), **NOT the individual-differences level**. n=9 underpowered throughout; only the DPA-level trend
     is even suggestive. Don't claim a depth↔performance correlation.

2. **SECONDARY (real but small): choice-axis lick-readiness.** During the delay the choice (sample:test)
   state predicts WHETHER the animal licks, controlling for the stimulus — **lick-readiness, NOT anticipation**
   (the choice depends on the future test). Clean causal test (fix sample&test, split by behaviour): Expert
   licked−withheld +0.176; within-cell permutation **p=0.036**, per-mouse Wilcoxon **p=0.031** (6/7 mice);
   marginal under mouse-bootstrap; ~0.18 σ. Naive null. (`lick = match (sample==test) = choice==1`.)
   - **Basis-robustness of the lick-readiness is NOT assessable per-mouse/per-basis — UNDERPOWERED (2026-06-23,
     `fig_dpca_lickready_basis.py`).** A per-mouse within-(sample,test) licked−withheld contrast rests on Expert
     ERROR trials, which are far too rare: **JawsM01 & JawsM18 have 0 usable cells** (FA≤1, miss=0); per-mouse
     estimates swing **+0.17 to −3.09** driven by 2–3 error trials (ACCM04 −3.09 from 2 miss trials); pooled
     (−0.93, Wilcoxon p=0.031) and own (−0.38, n.s.) DISAGREE — both noise. ⇒ the only powered estimate is the
     canonical **pooled within-(sample,test) permutation** (pools all mice, +0.176, already marginal). **Do NOT
     read `fig_dpca_lickready_basis.py`'s p=0.031 as a real per-basis effect** — it is rare-error-trial noise.
     (The CONDITION-MEAN choice depth IS basis-robustly ≈0; only the behaviour-split readiness is under-powered.)

3. **RETRACTED ARTIFACT — do NOT re-use or cite.** The "choice-axis polarization / Naive≈0 / +0.5 p=.02 /
   deep choice push" was a **circular-orientation + future-test-leakage artifact** (per-mouse REFIT DUMs have
   arbitrary signs; orienting per-mouse by the statistic being tested rectifies noise). The pooled "Naive→0
   after CI removal" was cross-mouse sign cancellation (JawsM18 +4.4 / ChRM04 −4.0). "Naive wells at zero" is
   FALSE for the tasks push (already −1.3 in Naive; learning DEEPENS an existing push). **Bad outputs:**
   `fig_dpca_flow_autonomous_choice.py`, `fig_dpca_choice_ci_qsweep.py`, `--choice-auto`,
   `fig_dpca_flow_learning_ingain*.py`'s Naive=0 / combined-axis claims.

**CI/time caveat:** the condition-independent `time` ramp also deepens but is GLOBAL (not lick-specific) →
keep UNBUNDLED. The corrected `--push` uses the TASKS axis only (CI/time excluded).

**`--push` construction (corrected):** action axis = choice + tasks (lick/no-lick). INPUT panels refit on
choice+tasks. **AUTONOMOUS panel special-cased** (the tasks ramp is input-driven → refitting it collapses the
wells): bistability fit on the choice axis (gain auto-raised to the smallest `a` giving 2 wells) + a no-lick
drive **gated by recurrent activity** `r(z)=1−S(z)`, `h` tuned numerically to keep 2 attractors → wells pushed
slightly into no-lick, **slow manifold DEFORMS**, bistability preserved (Expert −0.48 deeper than Naive −0.19).

**Methodological lessons (carry forward):** (a) the choice-axis sign is FIXED on the POOLED dPCA
(lick=match=choice1); per-mouse REFIT DUMs have ARBITRARY sign — **never per-mouse-flip a sign by the statistic
you're testing**. (b) To test a delay→choice signal, **fix the stimulus (sample AND test) and split by
behaviour** (else future-test leakage / correct-trial selection confound). (c) **Never fit an AUTONOMOUS flow
on an input-driven (ramping) trajectory** — it misreads the ramp as dynamics and collapses the wells.
(d) Naive-native DUM `pseudo_ALL_Naive_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca`; raw un-z-scored
array `X_all_no_scale` at `/home/leon/dual_task/dual_data/data/pca/`.

---

### Detailed audit trail (the iterative path — kept for reasoning + "don't re-try" warnings)

## No-lick push as a DEFORMATION, strengthens with learning (2026-06-22)

**Claim:** the WM (sample) attractors are **pushed into no-lick WITH LEARNING**, and the push is a
**deformation of the landscape** (only the wells move), **not a rigid translation**. Scripts:
**`pca/fig_dpca_flow_learning_ingain.py`** (pooled Naive-vs-Expert) and
**`pca/fig_dpca_flow_learning_ingain_permouse.py`** (3×3 grid + paired summary).
Plane = **[sample, tasks]** on the `…f-sample-test-tasks_dpca` DUM (the **tasks** marginal carries the
absolute no-lick/action offset; the sample:test contrast is demixed condition-mean-zero so it can't).

**Why the tasks marginal (not sample:test):** dPCA demixing makes each marginal a condition-mean-zero
contrast, so the *absolute* no-lick offset lives in the **tasks / time (CI)** marginals — that's exactly
the property "dPCA misses" on the sample×(sample:test) plane. So model the push on the **tasks** axis.

**Model (the one that works):**
```
ẋ = xflow(x)              # 1-D sample bistability, autonomous, fit on approach window 12–54 (fit_rnn_flow);
                          #   SHARED across stages — learning does NOT change the recurrent
ẏ = −y − h·r(z)           # no-lick drive GATED by recurrent activity
   r(z) = 1 − S(z) = ⟨tanh²(√Δ·ξ)⟩,  Δ = a²‖z‖² + δ     # gate: ~0 at baseline (quiet, S≈1), →1 at wells
```
`a=0.6, δ=0.05, KY=1` fixed; `h` **fit per stage** to the observed late-delay (39–54) tasks depth.
`r(z)` is the **gain-drop** quantity from the rank-2 framework (1−S, S=⟨φ′⟩) — a readout of the recurrent
activity from *inside* φ. So the drive only fires when the network holds a memory → **pushes just the
wells, baseline barely moves**. Verified (pooled): drive `h·r` at baseline x=0 ≈ +0.13/+0.15
(Naive/Expert) vs at the well x=2 ≈ +1.28/+1.58 — ~10× → genuine deformation.

**Result.** Pooled: `h` 2.75→3.37, well depth −0.97→−1.57 (Naive→Expert). **Per-mouse (n=9, after
tasks-axis sign alignment — see below): well depth −1.30→−1.88 (Wilcoxon p=0.012, 8/9 deepen); drive
h 3.33→3.90 (p=0.074, trend).** Only JawsM06 fails to deepen.

**Per-mouse sign alignment (REQUIRED, not optional):** the dPCA tasks component sign is **arbitrary per
mouse** — 2/9 (JawsM01, ACCM03) came out with the no-lick state on the *positive* side, washing out the
signed mean (p=0.50 → 0.012 after fixing). Fix: flip each mouse's tasks axis so the **Expert DPA-delay
state is negative**, applied to BOTH stages (within-mouse Naive→Expert comparison untouched; only the
cross-mouse orientation is fixed). This is legitimate dPCA sign-ambiguity handling, not p-hacking.

**The push is SPLIT across tasks + CI/time (the original "dPCA misses it" reason).** dPCA demixing
divides the single absolute no-lick displacement the raw population shows into two marginals — the
condition-specific **tasks** axis AND the condition-independent **time/CI** axis — and **CI is the larger
share** and grows more with learning (per-mouse, sign-oriented, n=9):
| component | Naive | Expert | Δ | p |
|---|---|---|---|---|
| tasks | −1.28 | −1.84 | −0.56 | 0.027 |
| **time / CI** | **−2.27** | **−3.02** | **−0.75** | 0.012 |
| combined (tasks+time direction) | 2.59 | 3.54 | +0.95 | 0.012 |
⇒ the tasks axis alone **undercounts** the push; the sample:test plane can't see it at all. **Best flow =
`fig_dpca_flow_learning_ingain_combined.py`**, whose vertical axis is the **combined no-lick direction
= 0.53·tasks + 0.85·time** (the direction the DPA delay state actually travels from baseline). Full push
deepens Naive −1.44 → Expert −2.25; drive h 3.47 → 4.07. **All trials (no `performance` filter)** —
consistent with the decision-function decoders.

**"Naive wells at zero" — TESTED, FALSE (do not re-run).** Expectation was Naive near baseline, pushed
into no-lick only with learning. The DPA delay state is **already in no-lick in Naive** (−1.3), robustly:
(a) 3 baseline references agree (pre-stim / demixed-zero / own-baseline); (b) **DPA−Go** (removes the
common ramp) is still −2.23 in Naive; (c) a **Naive-native dPCA basis** (`…_Naive_…f-sample-test-tasks_dpca`,
built 2026-06-22) gives the same −1.30 — not a projection artifact; (d) **including incorrect trials**
shifts pooled Naive only −0.97→−0.82 (errors are shallow: incorrect −0.50 vs correct −0.97 — the axis
tracks the decision), per-mouse essentially unchanged. Learning **deepens** an already-present push; it
does not create it from zero. NB the tasks-axis delay value does **not** separate by the trial's lick
decision (no-lick −0.86 vs lick −0.81 in Naive) — the decision is at *test*, so the delay displacement is
the DPA **withholding context** (task identity), not the trial-by-trial decision function (that lives on
the choice/sample:test axis, ≈0 during the delay).

**CI-removal robustness q0/q1/q2 (CLOSED).** DUMs: `…f-sample-test-tasks_dpca` (q0),
`…f-sample-test-tasks_ci1_dpca` / `_ci2_dpca` (pooled); `…_dpca_<MOUSE>_ci1` / `_ci2` (per-mouse).
Tasks-axis DPA late-delay depth (oriented no-lick<0), Naive→Expert:
| q | pooled Naive | pooled Expert | per-mouse Naive | per-mouse Expert | per-mouse Δ, p(N≠E) |
|---|---|---|---|---|---|
| q0 | −0.97 | −1.57 | −1.29 | −1.88 | −0.59, p=.012 |
| q1 | **−0.08** | −0.58 | −1.01 | −1.61 | −0.60, p=.008 |
| q2 | **−0.05** | −0.47 | −0.99 | −1.60 | −0.62, p=.012 |
**Robust (pooled AND per-mouse): the LEARNING deepening** (Expert ~0.6 deeper than Naive) is unchanged
by CI removal — Δ ≈ −0.6 and p≤.012 at every q. So the Naive→Expert effect is genuinely condition-dependent,
not a CI/ramp artifact. **This is the result to lean on.**
**The pooled "Naive→0 at q1/q2" is an AVERAGING ARTIFACT (DOWNGRADED — `fig_dpca_nolick_ci_pooledaxis.py`).**
Projecting each mouse onto the SINGLE pooled tasks axis (no per-mouse flip): at q1/q2 most mice shrink toward
0 (the shared direction isn't each mouse's optimal axis) AND two high-variance mice with LARGE OPPOSITE signs
cancel — JawsM18 +4.4 / ChRM04 −4.0 at q1 → pooled mean ≈ 0. The per-mouse "all below zero (−1.0)" uses each
mouse's OWN refit axis + sign-flip (Expert<0), which removes both the shrinkage and the cancellation. So
pooled≈0 is NOT evidence Naive lacks a push; it's signed cross-mouse cancellation on a noisy shared axis.
**Conclusion: the Naive push persists per-mouse; the "Naive at zero" idea is not supported once you account
for the cancellation.** Trust the per-mouse refit + the deepening; do not cite pooled Naive≈0.

**Choice-axis delay signal = weak Expert LICK-READINESS, not anticipation (FINAL, 2026-06-23).**
KEY FACTS (verified): lick = match (sample==test) = `choice==1` (match→`correct_hit`); the POOLED sample:test
axis is ONE consistent direction (sign NOT arbitrary); the per-mouse REFIT DUMs DO have arbitrary signs.
**Two earlier numbers were BOTH wrong:** "+0.5 p=.02 polarization" (per-mouse refit + flip-by-delay = circular,
inflated) and "flat zero, retracted" (flip-by-noisy-response = scrambled). **Correct, causally-clean test**
(fix BOTH sample & test, split by the animal's actual choice → same stimulus, so no test-leakage): Expert
licked−withheld delay sep positive in ALL 4 stimulus cells (mean +0.176). **Stats (stimulus-controlled):
within-cell PERMUTATION (shuffle choice within (sample,test)) p=0.036; PER-MOUSE Wilcoxon p=0.031 (mean +0.25,
6/7 mice +); logit `choice~delay+sample+test+match` coef +0.24 p=0.064; hierarchical bootstrap (resample
mice→trials) 95%CI [−0.065,+0.395].** Sharpest dissociation FA-vs-CR (both non-match, identical correct
answer): Expert FA(+0.143,n=68) > CR(−0.007,n=332)=+0.150; mirror hit>miss(+0.203); each underpowered alone
(MWU p≈0.11–0.13, few errors). Naive: NULL on all (perm p=0.50, per-mouse 2/9+, FA−CR −0.07). **Verdict: a
REAL, small, Expert-only delay LICK-READINESS signal — the delay choice-axis state predicts WHETHER the animal
licks, controlling for stimulus (the bias that drives FA/miss errors); NOT anticipation of the match (test is
future). Significant on within-stimulus permutation & per-mouse (p≈0.03), MARGINAL under the strictest
mouse-resampling bootstrap (CI edge just <0; 2 mice too few errors → n=7); ~0.18 std vs ~1.0 for the decision
at response.** `fig_dpca_flow_autonomous_choice.py` & `--choice-auto` used
refit+flip → unreliable; if needed, use the fixed pooled axis / the within-(sample,test) test.
LESSONS: (1) choice axis = pooled fixed sign, never per-mouse-flip by the stat tested; (2) to test a delay→
choice signal, FIX the stimulus and split by behavior (else test-leakage/selection confounds it).

**What is REAL: the no-lick push on the TASKS axis, deepening with learning.** The `tasks` marginal **IS a
lick/no-lick (action) axis** (verified): DualGo(lick)=+0.96, DualNoGo/DPA(no-lick)=negative; it tracks the
actual lick (DPA lick −0.31 > no-lick −0.46 @resp); and it is **NOT orthogonal to the choice axis** (decoder
cos +0.22, 77°) — dPCA demixes the components but the readout directions still share lick/no-lick content. So
the DPA WM delay state sits at the no-lick (DPA/NoGo) end and the "push down in tasks" = **push into no-lick**.
Sign-free `|tasks-axis depth|`, per-mouse: **Naive 1.29 → Expert 1.88, Wilcoxon p=0.012, 8/9 mice, same raw
sign 9/9, mouse-bootstrap 95% CI [+0.29,+0.87]; sample-memory axis FLAT (Δ−0.04, CI[−0.11,+0.02]) as a
specificity control.** (Those are **correct-only** numbers; the stat figure default is now **ALL trials**:
−1.28→−1.84, p=0.027, 7/9 — robust to trial selection. `--correct` reproduces the correct-only version.)
Confirmed in raw ΔF/F (r=0.997). Present in Naive AND grows with learning (no "absent
in Naive" — that premise was an artifact). **The correct flow depiction is `--push`.** Earlier framing errors
(now fixed): "--push not learning-specific" (WRONG — deepening is significant) and "tasks = task-context not
no-lick" (WRONG — tasks IS the Go↔DPA/NoGo lick/no-lick axis). **One caveat survives:** the condition-
independent `time`/CI ramp also deepens but is global (not lick-specific) → keep unbundled from the tasks push.

**Normalization artifact ruled out — the descent IS in raw ΔF/F (CLOSED, do not re-litigate).** Concern:
is the no-lick descent created by z-scoring across trials at each timepoint? No, three ways:
(1) **Code** — every step is per-trial or a single per-neuron scalar, none demean across trials per
timepoint: raw `standard_scaler_BL` centers/scales per `(trial,neuron)` over the *baseline time window*
(`axis=-1`); `blcenter` (`_scale_per_day`) subtracts a per-neuron baseline-mean scalar (`std=1`, comment:
"not the full per-time mean PSTH … avoids the cross-condition per-timepoint demeaning artifact"); the dPCA
`_neuron_norm` zscore is one per-neuron mean + one per-neuron std over `(cond,time)`. The only per-timepoint
demeaner (`standard_scaler`, `axis=0`) is NOT on this path.
(2) **Logical** — per-timepoint across-trial demeaning would zero the CI/time marginal by construction, yet
that's the *largest* component (−2.3→−3.0).
(3) **Empirical** — `fig_dpca_descent_rawdff.py`: apply the dPCA no-lick decoder W to RAW un-z-scored ΔF/F
(`X_all_no_scale`, baseline-centred only) vs the z-scored pipeline → trajectories overlap, **r=0.997
(Naive) / 0.995 (Expert)**; both descend into no-lick across the delay and return at test. z-scoring is a
per-neuron 1/s reweight (constant in time) → cannot bend a time course. The push is real population structure.

**Rejected alternatives (do NOT re-try):**
- **#3 tonic input current** (`ż=f(z)+u`, `fig_dpca_flow_nolick_input.py`, `fig_dpca_flow_nolick.py`,
  `fig_dpca_flow_learning.py`): adds the SAME velocity everywhere → **rigidly translates the whole field**
  (baseline included). User rejected: "shifts the whole flow instead of deforming the manifold." The fix
  is exactly the gated/in-gain drive above.
- **#2 autonomous fit on [sample, tasks]** (`fig_dpca_flow_tasks.py`): the descent into no-lick is
  input-driven, so an autonomous fit **extrapolates wrongly** ("the whole flow is wrong").
- **in-gain drive via `⟨tanh(√Δ·ξ+h)⟩`** (mean-shift inside φ): deforms but in the **wrong direction**
  (averaging over the spread shrinks the drive at the wells where Δ is large → pushes the *centre* more
  than the wells) and **saturates** to a uniform translation at the h needed for depth>1. Use `1−S(z)`
  (grows with activity) instead.

## Open issues / where to pick up

- **SESSION CLOSE 2026-06-23 — no-lick-push arc DONE & SETTLED** (see "No-lick push & choice readiness —
  SETTLED CONCLUSIONS" above; the arc below it is the audit trail). Net result: (1) robust no-lick push on
  the tasks (lick/no-lick action) axis, deepens with learning, supports the hypothesis — flow = `--push`;
  (2) small Expert-only choice-axis lick-readiness; (3) the dramatic choice-polarization / Naive=0 claims
  were retracted artifacts. **Genuinely open next:** reconcile with the **decoders** subproject (Fig 3,
  distance-to-lick-boundary) — does the single-neuron lick axis show the same push & deepening? And set the
  `--push` flows beside the **RNN** model's autonomous+input flows.
- **Single-trial flow fields: closed** — SNR-limited (per-trial offset), condition means are
  the minimal sufficient statistic; see the single-trial section above. Don't re-litigate.
- **Canonical figures regenerated — done (2026-06-16).** All `--inputs` grids re-fit with the
  regime-aware grouping + corrected Cue (pooled, pool7, 9 per-mouse, ChRM04_ci2, smoothed
  variants, Naive). `--smooth 1.5` is worth it only for per-mouse DUMs (cosmetic on the pooled);
  `_cm` vs `_cm_sm1.5` coexist. **Figure dir pruned** to the canonical **cond-mean** set
  (`figures/pseudo/flow`, ~59 MB): removed single-trial collapses (`_inputs`/`_inputs_correct`
  with no `_cm`, single-trial `nl-tanh`), the `sm0.5…sm5` sweep, and `_DPA`/`sm2.5` one-offs.
  `linear`/`persample`/`select` kept (1 each).
- **ChRM04 per-mouse autonomous: resolved — genuinely not bistable.** The edge-attractor was
  two artifacts (now fixed): the data-driven box blew up on choice-axis test excursions
  (`--nonlinear` now uses the calibrated box when `--rescale` is set), and choice-axis ramp
  leakage during the delay (use the `_ci2` DUM — cleans it). After both, 0 attractors, because
  ChRM04's **sample A/B delay separation is weak (d′≈0.75 vs JawsM15 1.31)** — the two states
  sit near the origin. Per-mouse bistability tracks sample-memory strength.
- **The comparison to make:** read each per-regime flow field (autonomous + each input)
  against the RNN's autonomous + input-driven flows — fixed-point count/location/stability
  and how each input reshapes the field. The per-regime fit captures whatever the data does
  in that window (e.g. the C/D panels now show the real test-driven choice movement), so
  it's a faithful data-side flow field to set beside the RNN's.
- **Naive renders now** (the current fit needs no bistability) on the **genuine Naive
  DUM** `pseudo_ALL_Naive_zscore_5x1_scale_blcenter_f-sample-test_dpca`. NB its
  autonomous flow is **monostable** (verified across gain 0.7–2.0) — a real
  not-yet-learned finding, not a basis artifact.
- Only tested on the **sample × choice** plane (enforced by a guard for `--inputs`).
- `ask-kimi` delegation tool is **broken** (no API key) — docs edited directly.

## Key commands

```bash
# build whole-timeline blcenter basis (one-off)
python run_pseudo.py --scale blcenter --epoch ALL --n-comp 12

# dynamics: eigenvalues + dimensionality + input term
python plot_pseudo_dynamics.py --dum pseudo_ALL_Expert_zscore_5x1_scale_blcenter
python plot_pseudo_dynamics.py --dum <dpca_dum>          # task vs time subspace

# flow fields  (--cond-mean = the real result; single-trial collapses, see single-trial section)
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice              # linear
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --per-sample
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --nonlinear --cond-mean  # DPA-WM bistable (autonomous, DPA delay)
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --inputs --cond-mean      # autonomous + per-regime input-driven fields
python plot_pseudo_flow.py --dum <naive_dpca_dum> --dims sample choice --inputs --cond-mean --stage Naive

# per-mouse flow (build the DUM first, then rescale to the pooled scale; smooth helps thin cells)
python build_mouse_dpca.py --mice JawsM15
python plot_pseudo_flow.py --dum pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca_JawsM15 \
       --dims sample choice --inputs --cond-mean --rescale 2.8 --smooth 1.5

# rank-2 reduced flows (autonomous + inputs, gain-modulated). Mode auto-selects by DUM:
python fig_dpca_flow_lowrank_shared.py                       # pooled Expert -> INDEPENDENT (descriptive geometry, C/D diagonals)
python fig_dpca_flow_lowrank_shared.py --dum pseudo_ALL_Naive_zscore_5x1_scale_blcenter_f-sample-test_dpca   # Naive (own basis)
python fig_dpca_flow_lowrank_shared.py --dum ...f-sample-test_dpca_JawsM18  # per-mouse -> PARTIAL (regularized, auto-λ)
#   override: --shared (parsimonious) | --independent | --partial
#   STAGE is auto-derived from the DUM ('Naive' if in DUM else 'Expert') -> trials match the basis;
#   output tagged dpca_lowrank_<mode>_<STAGE>[_<mouse>].png (Naive/Expert no longer collide).
#   Naive vs Expert: autonomous bistable both; Expert C/D vel-R² 0.67/0.70 & Cue 0.64 vs Naive 0.58/0.63 & Cue 0.32
#   (choice/test-driven structure less developed in Naive — matches the choice-axis q-sweep).

# no-lick push as DEFORMATION (gated drive), strengthens with learning.
python fig_dpca_flow_learning_ingain_combined.py   # BEST: no-lick axis = tasks+CI/time, all trials (decision-consistent)
python fig_dpca_flow_learning_ingain.py            # tasks-axis only, pooled Naive vs Expert (undercounts the push)
python fig_dpca_flow_learning_ingain_permouse.py   # 3x3 per-mouse grid + paired Naive->Expert summary (sign-aligned)
# build the Naive-native dPCA basis (one-off; used to show Naive push is not a basis artifact):
python run_pseudo.py --rebuild --dpca --stage Naive --norm zscore --scale blcenter --factors sample test tasks --n-splits 5 --n-repeats 1
# raw-ΔF/F check: descent is NOT a z-score artifact (raw vs z-scored r=0.997)
python fig_dpca_descent_rawdff.py
# CI-removal robustness q0/q1/q2 of the no-lick push (tasks axis) + why pooled Naive≈0 (cancellation)
python fig_dpca_nolick_ci_qsweep.py
python fig_dpca_nolick_ci_pooledaxis.py
# ⚠️ RETRACTED (circular-orientation artifact, see retraction note above) — do NOT use for claims:
#   fig_dpca_choice_ci_qsweep.py ; fig_dpca_flow_autonomous_choice.py ; fig_dpca_flow_lowrank_shared.py --choice-auto
# CORRECT no-lick-push flow = --push: action axis = choice + TASKS (lick/no-lick action; CI/time EXCLUDED).
#   INPUT panels: action=choice+tasks data, refit. AUTONOMOUS panel: special-cased (input-driven tasks RAMP
#   would collapse the 2 wells if refit) — bistability fit on the choice axis (gain auto-raised to the
#   smallest a giving 2 wells) + a no-lick drive GATED by recurrent activity r(z)=1−S(z), h tuned numerically
#   to the largest value keeping 2 attractors → wells pushed slightly into no-lick, slow manifold DEFORMS,
#   bistability preserved. Expert wells deeper (−0.48) than Naive (−0.19) = the learning effect.
python fig_dpca_flow_lowrank_shared.py --push           # Expert (tasks DUM auto)
python fig_dpca_flow_lowrank_shared.py --push --dum pseudo_ALL_Naive_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca
```
Env: `/home/leon/mambaforge/envs/dual/bin/python`, run from `pca/`.
Figures → `figures/pseudo/{dynamics,flow}/<mode>/<scale>/<factor>/<stage>/{png,svg}/`.
- flow `<mode>` = `linear` / `persample` / `nonlinear` / `inputs` (from args);
  dynamics `<mode>` = `pca` / `dpca` (auto-detected). `<stage>` = `Expert` / `Naive`.
