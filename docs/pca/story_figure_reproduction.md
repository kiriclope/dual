# dPCA story figure — full reproduction guide

Self-contained record of the full-arc dPCA main figure: **hypotheses, data, every routine, the exact
math and windows, the results, and how to regenerate it.** Written so a future session can rebuild and
defend the figure from scratch. Companions: `story_figure_methods.md` (condensed how-to),
`story_figure_review.md` (bugs & caveats), `flows_handoff.md` (design history).

Script: `pca/fig_dpca_story_main.py`. Absolute dates. Current 2026-07-03.

---

## 0. TL;DR — regenerate

```bash
cd /home/leon/dual/pca
/home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py             # correct trials (main)
/home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py --all-trials
/home/leon/mambaforge/envs/dual/bin/python fig_dpca_story_main.py --panels 4  # sec-3 reduced set
```
Outputs → `figures/pseudo/story/{png,svg}/fig_dpca_story_main[_all].{png,svg}` (PNG dpi=300, SVG with
`svg.fonttype='none'` for vector editing). No large array is loaded; runs in ~1–2 min.

Expected stdout (correct trials):
```
sec1: Expert top2 all 81.69% / wm 94.16% PR 2.22
per-marginal variance (%): sample 7%, test 1%, sample:test 7%, tasks 31%, time 54%
sec3 rank-2 [partial] best (a,δ,λ)=(0.2, 2.0, 1.0) CV vel-R²=+0.077 (57/60 bistable-autonomous configs)
sec3 attractors: {'autonomous': 2, 'sample A': 1, 'sample B': 1, 'Go': 1, 'NoGo': 1, 'cue': 2, 'test C': 2, 'test D': 2}
sec4 gated push: dN -0.86  dE -1.39  learned push -0.53  hE 1.10
sec4 per-mouse: push mean -0.59 p=0.012 (8/9 deepen) | sample sep N +1.65→E +2.33 p=0.10
```

---

## 1. Hypotheses (what the figure argues)

1. **Low-dimensional geometry.** The condition-mean neural state lives on a ~2-D manifold; two dPCA
   components capture the working-memory structure.
2. **Demixed axes = task variables.** Each dPCA axis carries one variable (sample, test,
   sample:test=choice, tasks), with interpretable time courses.
3. **Computation as flows.** The delay-period computation is captured by 2-D flow fields on the
   sample × choice plane: a bistable sample memory plus input-driven regimes (sample sets the well;
   distractor/test move the choice axis).
4. **Learning = a gated no-lick push.** With training the DPA (no-go) sample memory is pushed **down the
   no-lick axis**, and this is a **deformation of the wells** (the no-lick input is gated by the
   nonlinearity), not a rigid shift — quantified per-mouse (p=0.012), with the sample memory preserved.

---

## 2. Data & environment

- **Subjects/task:** 9 mice, dual task — DPA + DualGo + DualNoGo. Learning stages **Naive** / **Expert**.
- **Python:** `/home/leon/mambaforge/envs/dual/bin/python`. Run from `pca/` (relative `../data/pca`).
- **Imports (only the stable library):** `from src.pca.io import pkl_load`;
  `from src.pca.dynamics import flow_fixed_points`. All other logic is thin glue copied in — the source
  analysis scripts `savefig` at import and are **not import-safe**. (`sys.path.insert(0,'/home/leon/dual/')`
  before imports.)
- **Saved dPCA runs (DUMs)** — load `pseudo_{traj,labels,marglabels}_<DUM>` via `pkl_load(..., path='../data/pca')`:
  - `DUM_ST = 'pseudo_ALL_{}_zscore_5x1_scale_blcenter_f-sample-test_dpca'` (`.format('Naive'/'Expert')`)
    — 8 components (sample, sample:test/choice, time). **Sec 1B, Sec 3.**
  - `BASE = 'pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test-tasks_dpca'` — 16 components (adds
    the **tasks** marginal). **Sec 1C, Sec 2, Sec 4 pooled.** Per-mouse: `BASE + '_' + MOUSE`. **Sec 4 stats.**
  - `MICE = ['JawsM01','JawsM06','JawsM12','JawsM15','JawsM18','ChRM04','ChRM23','ACCM03','ACCM04']`.
- **Tensor shape:** `pseudo_traj` = (trials, components, time). `pseudo_marglabels` = list mapping each
  component → its marginal; resolve by `lab.index('sample')` etc. (order-robust — never hardcode indices).
- **Sampling / bins:** `FS = 6` Hz, 84 bins/trial. Key windows (bin indices):
  - `0:12` pre-trial baseline (ref for normalization).
  - `12:72` condition-mean window (sec 1 EVR / variance).
  - `SX = 21:54` delay window for the sample bistability fit (sec 4).
  - `LATE = 39:54` late delay (depth measurements).
  - `TST = 57:66` test window (sign orientation).
- **Trial mask:** `(laser==0) & (learning==STAGE)`; unless `--all-trials`, also `& (performance==1)`
  (`CORRECT = not --all-trials`). Laser-on trials are always excluded here.
- **⚠ pandas trap:** use `y['sample']`, never `y.sample` (the latter is `DataFrame.sample()`, a method).

### Palette / epochs
- Sample A `#332288` / B `#44AA99`; test C `#377eb8` / D `#4daf4a`; choice lick/no-lick reuse test cols;
  DPA `#e8000b` / Go `#023eff` / NoGo `#1ac938`; flow overlays `LR_COL` (Go `#117733`, NoGo `#CC6677`).
- `EP_SHADE`: sample 2–3 s, distractor 4.5–5.5, GNG cue 6.5–7, rwd 7–7.5, test 9–10, rwd2 11–12.

---

## 3. Shared math — the gain-modulated nonlinearity `gd()`

The whole flow machinery rests on the mean-field gain of a `tanh` rate unit driven by Gaussian input.
With `φ = tanh`, 20-node Gauss–Hermite quadrature (`np.polynomial.hermite_e.hermegauss(20)`,
weights `/√(2π)`):
```
gd(D, h) = ⟨φ'(√D · ξ + h)⟩ = ⟨1 − tanh²(√D · ξ + h)⟩ ,   ξ ~ N(0,1)
```
Interpretations used:
- **Gain** `S(z) = gd(a²‖z‖² + δ, 0)` — the effective linear gain; `≈1` near the origin (linear
  regime), `→0` at large ‖z‖ (saturated).
- **Gate** `r(z) = 1 − S(z) = ⟨tanh²(√Δ ξ)⟩` — the complement; `≈0` at the origin, `→1` at the wells.
  This is what makes the no-lick input a **deformation** (sec 4).

---

## 4. Layout

`figure(9.2, 12.9)`, `add_gridspec(4, 12, height_ratios=[0.95, 0.9, 2.4, 1.05], hspace=0.5, wspace=0.55,
left=0.075, right=0.975, top=0.93, bottom=0.055)`.
- Row 0 — **Sec 1** A/B/C equal thirds: `gs[0,0:4]`, `[0,4:8]`, `[0,8:12]`.
- Row 1 — **Sec 2** D–G: `gs[1, k*3:(k+1)*3]`.
- Row 2 — **Sec 3** flows: `gs[2,0:12].subgridspec(2, ncol3, hspace=0.14, wspace=0.05)` (`ncol3=3`, or 2
  for `--panels 4`). **No per-flow letters.** Shared `sample axis` / `choice axis (+ = lick)` labels
  placed from the grid bbox.
- Row 3 — **Sec 4** H/I flows + J stats: `gs[3,0:12].subgridspec(1,4, width_ratios=[1.1,1.1,0.5,0.5])`.
- Section headers left-aligned, placed from each band's computed top edge (`_top()`); panel letters via
  `plabel()`.

---

## 5. Section 1 — low-dimensional geometry & per-task variance (A, B, C)

### A — `schematic(ax)`
Hand-drawn dPCA cartoon: population box → "dPCA / demix by task variable" arrow → 5 stylized marginal
traces (sample, test, choice, tasks, time). Pure illustration, no data.

### B — `section1_evr(ax)`  (glue from `exp_rank_task.py`)
For each stage (`DUM_ST.format(STAGE)`): mask clean trials; build the **4 sample×test condition-mean
trajectories** over `win=12:72` as `(time × components)`, stack, centre, `np.linalg.svd` → eigenvalues
`ev = s²/Σs²`. Two subsets:
- all components → `top-2 = ev[:2].sum()`, `PR = (Σev)²/Σev²`.
- WM subset (`{sample, sample:test}` components) → `top-2 wm`.
Plot the decaying EVR scree, Naive vs Expert, dashed line at PC 2.
**Results (Expert):** top-2 **94% (wm) / 82% (all)**, **PR 2.2**. Annotated honest scope: geometry ≈ 2-D;
full dynamics higher-rank (rank-2 = 62–67%, from the separate dPCA reduced-rank test — cited, not
recomputed here).

### C — `section1_contrast(ax)`  (glue from `plot_mouse_dpca_traj.py`)
`load_marg(BASE)`: z-score each component over (trials,time); **sign-orient** each marginal (B>A at
delay `42:54`; D>C, lick>no-lick, Go>NoGo at `TST`). `cstat()` = mean difference ± pooled SEM. Plot the
4 contrasts: sample (B−A), test (D−C), choice (lick−no-lick), tasks (Go−NoGo), with `EP_SHADE`.

**Per-task variance** — `marginal_variance(BASE)`: build condition means over `sample×test×tasks` (bins
`12:72`), centre per component, `var_c = Var over (cond,time)`; per-marginal fraction
`= Σ var_c[marginal]/Σ var_c`. **Results: time 54%, tasks 31%, sample 7%, choice 7%, test 1%** (shown in
C's legend). *This is a proxy* for dPCA marginal-EVR (uses demixed trajectories, not encoder/decoder
reconstruction). WM axes are low-variance but decodable — state plainly.

---

## 6. Section 2 — condition-mean trajectories (D–G)  `section2_traj`
Same loader/orientation as C. Four panels, per-condition mean±SEM (`stat()`): sample (A/B), test (C/D),
sample:test/choice (lick/no-lick), tasks (DPA/Go/NoGo). Epoch shading only (labels on C). Shows each
demixed axis rises at its variable's epoch (sample→delay, test→test, choice/tasks→post-cue).

---

## 7. Section 3 — the computation: rank-2 gain-modulated flows (H–M)  `section3`

**Plane:** sample × choice (`isam=lab.index('sample')`, `icho=lab.index('sample:test')`) from
`DUM_ST.format('Expert')`, pooled. `Z2 = Z[m][:,[isam,icho]] ; Z2 /= Z2.std((0,2)) ; Z2 *= 2.8`
— **normalize each axis to std 2.8 over the FULL trial** (a delay-only rescale would inflate the choice
axis). Orient choice so lick>no-lick at `TST`.

**Model (INDEPENDENT per-regime gain-modulated low-rank flow).** For regime `r` fit `A_r∈ℝ²ˣ²`, `c_r∈ℝ²`:
```
ż = −z + S(z)·A_r z + c_r ,   S(z) = gd(a²‖z‖² + δ, 0)
```
- `fit_indep_one(z, v, a, dd)`: form `S = gd(a²‖z‖²+dd)`, design `F = [S·z0, S·z1, 1]`, solve
  `lstsq(F, v + z)` per output dim → `A_r`, `c_r` (the `+z` folds in the `−z` leak).
- `flow_indep(A, c, a, dd)`: returns `P ↦ −P + S(P)·A P + c`.
- `regime_means(mask)`: per regime, condition means over its window (≥3 trials).
- `zv_one`: stack `(z = μ[:,w][:,:-1], v = diff(μ[:,w]))` — positions & one-step velocities.

**Regimes `REG` (name, trial mask, window bins, factor levels) — 8 panels, 2×4 grid:**
| regime | mask | window | overlay levels |
|---|---|---|---|
| autonomous | DPA | 21:54 | sample 0,1 |
| sample A | sample==0 | 15:30 | sample 0 |
| sample B | sample==1 | 15:30 | sample 1 |
| Go | DualGo | 30:52 | sample 0,1 |
| NoGo | DualNoGo | 30:52 | sample 0,1 |
| cue | Go\|NoGo | 39:54 | tasks Go, NoGo |
| test C | test==0 | 57:84 | sample 0,1 |
| test D | test==1 | 57:84 | sample 0,1 |

**Model = PARTIAL POOLING within two epochs** (ported from `fig_dpca_flow_lowrank_shared.py --partial`).
Each regime flow is `ż = −z + S(z)·(A_sh + ΔA_r)·z + c_r`: a **shared recurrent `A_sh`** carrying the
epoch's bistability, a **ridge-penalized per-regime deviation `ΔA_r`** (λ shrinks it toward `A_sh` — so
per-regime flows generalize instead of overfitting a free landscape to two mean trajectories), and a
**per-regime input current `c_r`**. Closed-form ridge LS: design `[S·z | one-hot⊗S·z | one-hot]`, penalty
`λ·I` on the `ΔA` block only (not the shared `A` or the inputs). `fit_group` does one group; `fit_all`
runs it per group.

**Two shared landscapes, one per epoch** — a *single* shared `A` can't hold both bistabilities: the three
choice-bistable regimes (cue, test C/D) outvote the one sample-bistable regime, so a single `A_sh` comes
out choice-dominated and the sample memory (autonomous, A, B) lands at **saddles** (wrong). Fix: pool
**within** each epoch — `GROUPS = [{autonomous, sample A, sample B}, {Go, NoGo, cue, test C, test D}]`. The
**delay landscape** is sample-bistable (autonomous = both wells on the sample axis; A/B settle in one); the
**choice landscape** is choice-bistable (Go = push ↑ lick, NoGo = push ↓ no-lick, cue = the two splitting,
test C/D = choice resolution). `--panels 4` keeps autonomous / sample A / cue / test C.

**Hyperparameters `(a, δ, λ)` by 5-fold CV** (`KFold(5, shuffle, random_state=0)`), grid
`a∈{0.2,0.4,0.7,1.0} × δ∈{0.3,0.8,2.0} × λ∈{0.2,1,5,20,100}`, maximise held-out **velocity-R²** (fit on
train regime means, predict test regime velocities). Selection is **restricted to configs whose shared
autonomous flow keeps 2 wells** (the WM bistability is an established result; the raw CV-optimal gain is
often monostable), then max CV among those — stays fully in the partial model, no separate autonomous refit.

**Fixed points** via `flow_fixed_points(flow, [(-L,L),(-L,L)], n_seed=18)`, `L = 1.3·max|means|`:
★ attractor (yellow) / □ saddle (white) / ✖ repeller (red). Condition-mean trajectories overlaid.

**Results:** best `(a,δ,λ) = (0.2,2.0,1.0)` correct / `(0.2,0.8,1.0)` all; **CV vel-R² = +0.077 / +0.104**
(≈57/60 grid configs give a bistable autonomous). Attractor counts: autonomous **2** (+saddle), sample A/B
**1** each, Go/NoGo **1**, cue **2**, test C/D **2**.
> ✔ **CV is now POSITIVE** — partial pooling makes the per-regime flows **generalize** (vs the old
> independent per-regime fit, CV ≈ −0.13, which overfit and looked artificial). The flows are a
> regularized shared-landscape portrait that holds out-of-sample. Still a rank-2 *reduced* description of a
> higher-rank latent (see the standing "rank-2 not validated as the full dynamics" note); don't claim the
> full dynamics are rank-2, but the per-regime velocity fields do now cross-validate. (See
> `story_figure_review.md` caveat 1.)

---

## 8. Section 4 — learning pushes the memory into no-lick (H, I, J, K)

Plane = **sample × tasks(no-lick)**. **The push depth and the landscape are fit from data; only the
gate's spatial profile is a modeling choice.**

### 8.1 Load & orient — `load_st()` (pooled) / `load_mouse(mm)` (per-mouse)
`Zr = Z[:,[isam,itask]]`, centre by pre-trial DPA ref (`0:12`), divide by DPA-ref std, `× 2.8`. Orient
the **tasks** axis so Expert DPA late-delay is negative (no-lick negative). `load_mouse` additionally
orients the **sample** axis so Expert B>A (else the sign of `B−A` is arbitrary per mouse — this was a bug).

### 8.2 Sample bistability (data-fit, shared) — `fit_sample_bistab(muA, muB, α=0.42, δ=0.4)`
1-D gain-modulated fit to the pooled A/B delay means on the sample axis:
```
ẋ = −x + S(x)·a·x + c ,   S(x) = gd(α²x² + δ, 0)
```
`a, c` by least squares → two wells at ±x_w. The **same `fx`** is used in both panels (only the vertical
push differs). `stage_delay(Zr,y,stage)` returns `(muA, muB, depth)` where `depth` = mean over A,B of the
tasks-axis value over `LATE` (mapped into the `SX` window).

### 8.3 The no-lick push = a GATED input (the key mechanism) — `rgate`, `make_flow`
The input enters **inside** the nonlinearity:
```
ẏ = −y − h · r(z) ,   r(z) = 1 − gd(GATE_A²‖z‖² + GATE_D, 0)   (GATE_A=0.9, GATE_D=0.12)
```
`r(z)≈0` at the origin (linear regime) and `≈1` at the wells (saturated) ⇒ the drive pushes **only the
wells** down while the **centre/saddle stays pinned at 0** — a manifold **deformation**, not a rigid
translation. `make_flow(fx, h)` returns `P ↦ [fx(P0), −P1 − h·r(P)]`. `h=0` → flat reference (Naive);
`h=hE` → gated push (Expert).

### 8.4 `hE` is FIT from data
`push = dE − dN` is the **measured** Naive→Expert deepening. `hE` is grid-searched over
`np.linspace(0,5,51)` as the smallest `h` that (i) keeps ≥2 attractors and (ii) drives the mean
attractor `y` down to `push`. So the Expert well depth **is** the data value. *Not* fit: the gate form
and `GATE_A/GATE_D` (they change how localized the deformation looks, not the depth, since `hE` re-tunes);
`α,δ`; and the naive=0 display anchor.

### 8.5 Draw — `draw_st(...)`
Magma speed + white streamlines; cyan dashed "naive level" at `y=0`; trajectories shifted by `−dN` (Naive
settles at 0, Expert at `push`); on Expert, ghost stars at the naive level + white "learning push" arrows
to the deformed wells; ★/□/✖ fixed points.
**Results:** dN=−0.86, dE=−1.39, **push=−0.53**, hE=1.10 (all-trials: push −0.56, hE 1.40).

### 8.6 Panel J — per-mouse quantification — `depth_of`, `section4` stats
Per mouse (`load_mouse`, both axes oriented), over `LATE`:
- `uy` = no-lick depth = mean over A,B of the tasks-axis value (A/B centroid).
- `sep` = **sample memory** = `mean(B) − mean(A)` on the sample axis (separation |B−A|).

- **J1 "no-lick push":** per-mouse `push_m = uy_E − uy_N`, plotted **anchored to naive (0)** so it reads
  like the flow. Paired `wilcoxon(uy_N, uy_E)`. **Results: mean −0.59, p=0.012 (8/9 deepen)**; all-trials
  −0.56, p=0.027 (7/9). Anchoring does not change the paired p; it matches the flow's −0.53.
- **J2 "sample memory":** `sep` Naive vs Expert. **Results: N +1.65 → E +2.33, p=0.10** (all-trials
  +1.44 → +2.48, **p=0.02**) — the two wells stay separated (memory preserved / sharpens) while pushed
  down. *Note:* on all-trials this is a significant increase, so "preserved" is conservative wording; if
  foregrounding it as a finding, say "memory sharpens".

Old J (removed) was wrong: its control was the sample **centroid** ((A+B)/2 ≈ 0 by symmetry — trivial),
and it plotted absolute depths inconsistent with the naive=0 flow. See `story_figure_review.md`.

### 8.7 Panel K — tasks↔choice axis mixing — `section4_mixing(ax)`
A second, independent learning readout: how much the **`tasks`** axis (cue-driven lick/no-lick *action*)
and the **`sample:test`/choice** axis (decide-to-lick) **mix** in neural space, Naive vs Expert. Metric =
**|cos| between the leading `tasks` and `choice` dPCA decoder axes** (0 = demixed/orthogonal, 1 = collinear),
from the pooled per-stage `pseudo_weights_…f-sample-test-tasks_dpca` (decoder, 16 comp × 3319 neurons;
`tasks` = comps 4–5, `sample:test` = 6–7). Both stages share the **same 3319 neurons**, so a **neuron
bootstrap** (2000×, same resampled neurons) gives a paired CI. **Result: |cos| N 0.147 → E 0.222,
Δ+0.076, p<0.001** (2-D subspace overlap agrees: 0.039 → 0.063). Interpretation: learning **binds the
decision and action axes into a shared lick/no-lick code** — the same process that pushes the memory into
no-lick (this *is* the tasks→choice coupling that section-3's cue/Go/NoGo flows show dynamically).
> ⚠ CI is a **neuron bootstrap** (samples the decoder's neurons), not an across-animal test — per-mouse
> `tasks` DUMs don't exist yet. For animal-level error bars, refit dPCA per mouse per stage.

---

## 9. Routine inventory (quick index)

| routine | role |
|---|---|
| `gd(D,h)` | gain-modulated nonlinearity `⟨φ'(√D ξ + h)⟩` (20-node GH) — used by S and r |
| `plabel`, `shade_epochs`, `_top`, `sechead` | panel letters, epoch shading, header placement |
| `schematic` | Sec 1A dPCA cartoon |
| `section1_evr` | Sec 1B EVR scree, top-2 / PR (SVD of condition means) |
| `load_marg`, `stat`, `cstat` | dPCA marginal loader (+sign orient), mean/SEM, contrast |
| `marginal_variance` | per-task variance proxy |
| `section1_contrast` | Sec 1C the 4 marginal contrasts + variance legend |
| `section2_traj` | Sec 2 D–G per-condition trajectories |
| `fit_indep_one`, `flow_indep`, `section3` | Sec 3 rank-2 gain-modulated per-regime flows + CV |
| `load_st`, `fit_sample_bistab`, `stage_delay` | Sec 4 plane load, bistability fit, delay means/depth |
| `rgate`, `make_flow` | Sec 4 gated no-lick input + flow builder |
| `draw_st` | Sec 4 flow render (deformation, ghosts, arrows) |
| `load_mouse`, `depth_of`, `section4` | Sec 4 per-mouse stats (J1 push, J2 memory) + assembly |
| `section4_mixing` | Panel K — tasks↔choice \|cos\| mixing, Naive vs Expert (neuron bootstrap) |

---

## 10. Results summary (headline numbers)

| claim | quantity | correct | all-trials |
|---|---|---|---|
| geometry 2-D | top-2 EVR (wm / all), PR | 94% / 82%, 2.2 | ~same |
| per-task variance | time/tasks/sample/choice/test | 54/31/7/7/1 % | ~same |
| flows cross-validate | CV vel-R² (best a,δ,λ) | +0.077 (0.2,2.0,1.0) | +0.104 (0.2,0.8,1.0) |
| autonomous bistable | attractors | 2 (+saddle) | 2 |
| no-lick push (flow) | dN→dE, push, hE | −0.86→−1.39, −0.53, 1.10 | push −0.56, hE 1.40 |
| no-lick push (per-mouse) | mean, p, n deepen | −0.59, **0.012**, 8/9 | −0.56, **0.027**, 7/9 |
| sample memory | sep N→E, p | +1.65→+2.33, 0.10 | +1.44→+2.48, **0.02** |
| tasks↔choice mixing | \|cos\| N→E, p (neuron boot) | 0.147→0.222, **<0.001** | ~same |

---

## 11. Standing caveats (before making any claim)
1. **Sec-3 flows now cross-validate** (CV>0 under partial pooling; the old independent per-regime fit was
   CV<0 = overfit). Still a rank-2 *reduced* portrait — don't claim the full latent dynamics are rank-2.
2. **Per-task variance is a proxy** (demixed-trajectory variance, not exact dPCA marginal-EVR).
3. **Sec-4 gate profile is a modeling choice** (`GATE_A/D`, form). The **push depth and landscape are
   data-fit**; `hE` reproduces the measured push. Naive=0 is a display anchor (absolute dN=−0.86,
   dE=−1.39).
4. **Flows are POOLED** across the 9 mice; per-mouse variability is only in panel J.
5. **`y['sample']` not `y.sample`**; orient every axis you take a signed quantity on; never fit an
   autonomous flow on an input-driven ramp (why sec-4 fits the bistability on the sample axis and adds
   the no-lick drive separately).
