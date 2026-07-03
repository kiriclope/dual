# Story figure — methods / how-to

How `pca/fig_dpca_story_main.py` builds the full-arc dPCA main figure, section by section, with the
exact math and windows. Companion: `story_figure_review.md` (bugs & caveats). Read `flows_handoff.md`
for the design history.

```bash
cd /home/leon/dual/pca
/home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py            # correct trials
/home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py --all-trials
/home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py --panels 4 # sec-3 reduced set
```
Outputs: `figures/pseudo/story/{png,svg}/fig_dpca_story_main[_all].{png,svg}`.

## Conventions
- **Env:** `/home/leon/mambaforge/envs/dual/bin/python`; run from `pca/` (scripts use `../data/pca`).
- **Imports:** only the stable library — `src.pca.io.pkl_load`, `src.pca.dynamics.flow_fixed_points`.
  Everything else is thin glue COPIED in (source scripts `savefig` at import → not import-safe).
- **DUMs (saved dPCA runs):**
  - `DUM_ST = pseudo_ALL_{Naive,Expert}_..._f-sample-test_dpca` — 8-comp (sample×choice). Sec 1B, sec 3.
  - `BASE = pseudo_ALL_Expert_..._f-sample-test-tasks_dpca` — 16-comp (adds tasks marginal). Sec 1C/2/4.
    Per-mouse suffix `_{MOUSE}` for the 9 mice (sec-4 stats).
- **Data tensor:** `pseudo_traj_<DUM>` = (trials, components, time); `pseudo_marglabels_<DUM>` gives each
  component's marginal → resolve indices by `lab.index('sample')` etc. (order-robust).
- **Bins:** 6 Hz (`FS`), 84 bins/trial. Windows used: pre-trial `0:12`, sample window for the bistability
  fit `SX = 21:54`, late delay `LATE = 39:54`, condition-mean window `12:72`.
- **Trial mask:** `(laser==0) & (learning==STAGE)` and, unless `--all-trials`, `& (performance==1)`.
- **Palette:** sample A `#332288` / B `#44AA99`; test C `#377eb8` / D `#4daf4a`; DPA/Go/NoGo bright-3.
- **Save:** PNG `dpi=300` + SVG `svg.fonttype='none'`. Final publication assembly = vector-edit the SVG.

Layout: `figure(9.2, 12.9)`, 4-band GridSpec `[0.95, 0.9, 2.4, 1.05]`. Section headers are placed from
each band's computed top edge (`_top()`), left-aligned so they clear the panel letters.

---

## Section 1 — low-dimensional geometry & per-task variance  (A schematic, B EVR, C contrasts)

**B — EVR scree** (`section1_evr`, glue from `exp_rank_task.py`). For each stage, build the 4 sample×test
condition-mean trajectories over bins `12:72`, stack (time × components), centre, SVD → eigenvalue
spectrum `ev = s²/Σs²`. Report:
- `top-2 (all)` = `ev[:2].sum()` → **82%**; `top-2 (wm)` = same on the `{sample, sample:test}` component
  subset → **94%**; participation ratio `PR = (Σev)²/Σev²` → **2.2**.
Curve decays (EVR, not cumulative). Naive vs Expert overplotted.

**C — marginal contrasts** (`section1_contrast`, glue from `plot_mouse_dpca_traj.py`). z-score each
component over (trials, time); per-marginal **sign orientation** (B>A at delay; D>C, lick>no-lick,
Go>NoGo at test). Plot the 4 contrasts (sample B−A, test D−C, choice lick−no-lick, tasks Go−NoGo),
mean±SEM. Epoch shading via `EP_SHADE` (STIM 2–3 s, DIST 4.5–5.5, GNG cue 6.5–7, rwd 7–7.5, TEST 9–10,
rwd2 11–12).

**Per-task variance** (`marginal_variance`) — a **proxy** for dPCA marginal-EVR: build condition means
over `sample×test×tasks`, centre per component, `var_c = Var over (cond, time)`; per-marginal fraction =
`Σ var_c[marginal] / Σ var_c`. Shown in C's legend: **time 54%, tasks 31%, sample 7%, choice 7%,
test 1%**. (Proxy — uses demixed trajectories, not encoder/decoder reconstruction. See review caveat 2.)

---

## Section 2 — condition-mean trajectories  (D–G)

`section2_traj`. Same loader/orientation as C. Four panels of per-condition time courses (mean±SEM):
sample (A/B), test (C/D), sample:test=choice (lick/no-lick), tasks (DPA/Go/NoGo). Epoch shading only
(labels omitted — they live on C).

---

## Section 3 — the computation: rank-2 gain-modulated flows  (H–M, sample × choice, pooled)

`section3`, glue from `fig_dpca_flow_lowrank_shared.py` (INDEPENDENT per-regime model). Plane = sample ×
choice (`sample`, `sample:test`), pooled over mice, **each axis normalized to std 2.8 over the FULL
trial** (this is why the choice axis doesn't inflate — a delay-only rescale would).

**Gain-modulated low-rank flow, PARTIAL POOLING.** With `φ = tanh`, the mean-field gain is
`S(z) = ⟨φ'(√Δ · ξ)⟩`, `Δ = a²‖z‖² + δ`, `ξ ~ N(0,1)` (20-node Gauss–Hermite; `gd()` implements this).
Each regime flow is
```
ż = −z + S(z)·(A_sh + ΔA_r)·z + c_r
```
a **shared recurrent `A_sh`** + a **ridge-penalized per-regime deviation `ΔA_r`** (λ shrinks it toward
`A_sh`) + a **per-regime input current `c_r`** — the `--partial` mode of `fig_dpca_flow_lowrank_shared.py`.
Closed-form ridge LS with `λ·I` penalizing only the `ΔA` block. This is the fix for the earlier
**overfit**: a free per-regime `A_r` (the old "independent" fit) had CV vel-R² ≈ −0.13 (didn't generalize,
looked artificial); partial pooling gives **CV ≈ +0.08–0.10** (generalizes).
`(a, δ, λ)` chosen by 5-fold CV over `a∈{0.2,0.4,0.7,1.0} × δ∈{0.3,0.8,2.0} × λ∈{0.2,1,5,20,100}`,
maximising velocity-R² **restricted to configs whose shared autonomous flow stays bistable** (the raw
CV-optimal gain is often monostable; WM bistability is an established result).

**Two shared landscapes, one per epoch** (`GROUPS`): a single `A_sh` can't hold both bistabilities —
the choice-bistable regimes (cue, test C/D) outvote the one sample-bistable regime, so it comes out
choice-dominated and the sample memory lands at **saddles** (wrong). So pool **within** each epoch:
`{autonomous, sample A, sample B}` → sample-bistable **delay landscape** (autonomous = both wells on the
sample axis; A/B settle in one), and `{Go, NoGo, cue, test C, test D}` → choice-bistable **choice
landscape** (Go = push ↑ lick, NoGo = push ↓ no-lick, cue = the two splitting, test C/D = choice
resolution). Fixed points via `flow_fixed_points` on `±L` (`L = 1.3·max|mean|`): ★ attractor / □ saddle /
✖ repeller. Autonomous/cue/test C/D = 2 attractors; sample A/B, Go/NoGo = 1. Equation printed above the
grid; `--panels 4` keeps autonomous / sample A / cue / test C.

> **CV is positive** (≈ +0.08 correct / +0.10 all) under partial pooling — the per-regime flows now
> generalize (the old independent per-regime fit was −0.13, overfit). Still a rank-2 *reduced* portrait,
> not a claim that the full latent dynamics are rank-2. See review caveat 1.

---

## Section 4 — learning pushes the memory into no-lick  (H Naive, I Expert, J stats)

Plane = **sample × tasks(no-lick)**, `load_st`: normalize each axis to std 2.8 by the pre-trial DPA ref
(`0:12`), orient the tasks axis so no-lick is negative.

**Sample bistability (shared).** `fit_sample_bistab` fits a **1-D gain-modulated** flow to the pooled
A/B delay means on the sample axis:
```
ẋ = −x + S(x)·a·x + c,   S(x) = ⟨φ'(√(α²x² + δ)·ξ)⟩   (α=0.42, δ=0.4)
```
→ two wells at ±x_w. The SAME `fx` is used in both panels, so only the vertical push differs.

**No-lick push = a GATED input (the key point).** The drive enters INSIDE the nonlinearity:
```
ẏ = −y − h · r(z),   r(z) = 1 − S(z) = ⟨tanh²(√(a²‖z‖² + δ)·ξ)⟩   (GATE_A=0.9, GATE_D=0.12)
```
`r(z) ≈ 0` near the origin (linear regime) and `≈ 1` at the wells (saturated) ⇒ the input **deforms** the
manifold — it pushes the wells (high ‖z‖) down while the centre/saddle stays at 0 — instead of
translating the whole plane. `h=0` → flat reference (Naive); `h=hE` → gated push (Expert). `hE` is tuned
numerically so the deformed wells reach the measured learned push.

**Naive-anchoring (visualization).** Trajectories/wells are drawn relative to the naive well: shift by
`−dN` so the Naive memory sits at 0 (cyan "naive level") and the Expert wells are pushed to
`push = dE − dN` below it. Absolute depths: dN ≈ −0.86, dE ≈ −1.39; push ≈ −0.53. Ghost stars at the
naive level + white "learning push" arrows mark the deformation.

> `dN, dE = stage_delay(...)` measure the DPA no-lick depth over `LATE` on the pooled common frame.

**Panel J — per-mouse quantification** (`load_mouse`, `depth_of`; the 9 per-mouse DUMs). Per mouse,
normalize sample×tasks to std 2.8 by the pre-trial DPA ref; **orient BOTH axes** (tasks no-lick negative;
sample B>A). `depth_of` returns, over `LATE`:
- `uy` = no-lick depth = mean over A,B of the tasks-axis late-delay value (A/B centroid).
- `sep` = **sample memory** = `mean(B) − mean(A)` on the sample axis (separation |B−A|).
- **J1 "no-lick push":** per-mouse push `uy_E − uy_N`, anchored to naive (0). Paired Wilcoxon
  **p=0.012** (0.027 all); mean −0.59 (matches the flow's −0.53); 8/9 deepen.
- **J2 "sample memory":** `sep` Naive vs Expert — preserved/sharpened (N +1.65 → E +2.33, p=0.10 correct
  / 0.02 all). Confirms the two wells stay separated while pushed down.

> Anchoring J1 to naive does **not** change the paired p (it subtracts a per-mouse constant). It only
> makes J read consistently with the naive=0 flow.

---

## Gotchas (learned the hard way)
- **`y['sample']`, never `y.sample`** — the attribute form hits `DataFrame.sample()` (a method), giving
  a silent wrong result / `AttributeError`.
- **Orient every axis you take a signed quantity on.** The sample-axis B−A sign is arbitrary per mouse
  until you orient B>A.
- **Never fit an autonomous flow on an input-driven (ramping) trajectory** — it misreads the ramp as
  dynamics. The tasks axis is input-driven, which is why section 4 fits the bistability on the sample
  axis and adds the no-lick drive separately (gated), rather than fitting a 2-D autonomous flow on
  sample × tasks.
- **Gate the input, don't translate.** A push added *outside* the nonlinearity slides the whole
  manifold; gating it by `r(z)=1−S(z)` deforms only the wells (see section 4 + review).
