# Overlaps Subproject Overview

## Results so far (2026-07-01) — consolidated

The dual-coding hypothesis, now with a measured mechanism, from two independent angles
(decoder-weight cosines + latent rank-2 flows). Detailed sections + scripts below.

1. **Three independent, stable codes (geometry).** sample / choice / test are **mutually orthogonal**
   (between-code weight cosine at the ±1/√N chance floor, every epoch, both stages) and **temporally
   stable** (within-code cosine 0.4–0.9). → `fig_overlaps_cosine.py`; see *Interpretation & synthesis*.
2. **No-lick push + its mechanism (learning).** Expert DPA delay states are pulled into the no-lick
   half of the choice axis (asymmetric: A strong ≈−0.66, B weak). Robust at the **population level**
   (paired deepening; cross-method); the **individual depth↔performance correlation is borderline/fragile
   — see Caveats** (`plot_scatter_perf.py` ρ=−0.67 p=0.050, but null on the dPCA side). Proposed mechanism:
   **sample×choice orthogonalize with learning** (small, suggestive: |cos| 0.083→0.029, Wilcoxon p=0.020,
   7/9 mice, reliability-controlled; *only* that pair) → `fig_overlaps_orthogonalization.py`.
3. **Dual coding is live in the delay.** The 9-epoch code sweep shows the choice/lick code **non-flat
   already in the delay** (ED/MD) — a maintained no-lick action set coexisting with the held sample
   memory (NOT foreknowledge; the correct lick needs the future test). → `fig_overlaps_codes_1d.py`.
4. **Rank-2 latent flows (dPCA port).** `fig_overlaps_flow_lowrank_shared.py` — faithful **bistable
   autonomous** WM flow (delay axis, endpoint-anchored) + input-driven panels (odor-timed windows).
   Corroborates the geometry from the flow angle. CCGD velocity-SNR-limited; see *Rank-2 low-rank port*.

**Infrastructure added:** decoder weights persist (`run_overlaps.py --save-weights`); traj2d all-trials
variant; the cosine/orthogonalization/1D-codes/flow scripts above. **Headline figures:** the sample×choice
trajectories (`plot_traj2d.py`) and the depth↔performance scatter (`plot_scatter_perf.py`).

### Caveats — how much to trust each (2026-07-01, critical review)

All n=9 mice → treat every p near 0.05 as suggestive; several quantitative hooks are softer than they look.

- **STRONG.** (i) Within-code temporal stability (cosine 0.4–0.9 ≫ floor). (ii) The **no-lick push
  deepening with learning at the POPULATION level** (robust, cross-method: dPCA + overlaps + raw ΔF/F).
- **"Orthogonal" = at the ±1/√N chance floor** = statistically *independent*, NOT actively orthogonalised
  beyond chance (random axes in ~369-D are near-orthogonal anyway). Say "independent," not "orthogonalised."
- **Orthogonalisation with learning is SUGGESTIVE, not established.** Real & reliability-controlled
  (p=0.020, 7/9) but SMALL — |cos| 0.083→0.029, i.e. Naive only marginally above the 0.05 floor → Expert at
  it; and it is **1 of 3 pairs tested** (borderline under multiple-comparison correction), n=9.
- **Depth↔individual-performance — DROP as a headline.** Overlaps ρ=−0.67 is exactly at p=0.050 and
  **disagrees with the dPCA side (null: r=+0.46, p=0.21)**. The push↔behaviour link is at the
  population/average level, NOT individual differences (see `docs/pca/flows_handoff.md`).
- **Dual coding "in the delay" = the GEOMETRY (stable action axis), not a strong trial-by-trial signal.**
  The codes1d choice-in-delay separation may be **sample leakage** (on DPA, choice = match = sample==test);
  the stimulus-controlled test found only a small **Expert-only lick-readiness** (~0.18σ, marginal). Don't
  assert a strong maintained action *signal* — only the coexisting axis.
- **Rank-2 flows are DESCRIPTIVE, not predictive.** The per-regime R² quoted (A +0.96, C +0.70…) are
  **IN-SAMPLE**; the honest **pooled CV vel-R² ≈ 0** (best +0.17 for diag). Bistability is **gain/anchor-
  dependent** (monostable at the CV gain 0.2) and **shallow (robust in ~3/9 mice)** — an illustration of the
  WM landscape, not a validated dynamical model. Researcher DOF (train epoch/window/plane/anchor/gain) were
  tuned this session (failures retracted + documented, but not pre-registered).

**Bottom line to claim:** independent, temporally-stable codes + a population-level no-lick push that
sharpens the memory/action geometry with learning. Suggestive: the sample×choice orthogonalisation.
Descriptive only: the rank-2 flows. De-emphasise: individual depth↔perf and any strong delay action signal.

---

## What overlaps measures

Cross-generalising decision codes (CCGD): a logistic decoder is trained for each target variable —
**sample, choice, and test** — and cross-generalises across training and test time, giving a
**train × test** decision-function matrix per trial. The three targets are stacked along the trial
axis (`y['target']`); the decision function is `X[:, 1]` (the `X[:, 0]` probability is ~0.5 and unused).
To get a time-resolved scalar code we **average the decoder over a chosen train epoch and read it
across test time** (a fixed axis) — see the method section below. (Earlier docs said "diagonal"; the
scripts all use train-epoch averaging.)

Projecting two codes at once gives a 2D state space, e.g. **sample × choice**:
- **x** = sample code (separates A from B)
- **y** = choice code (separates lick from no-lick)

(also `sample × test`, `choice × test`). The paper hypothesis predicts that Expert delay-period
states occupy the **lower (no-lick)** half of the choice axis for DPA, with A/B separating along the
sample axis.

---

## Active figures

| Script | Shows | Trials |
|---|---|---|
| `plot_traj2d.py` | 2D trajectory over time + KDE strip | correct (default) or `all` laser-off (`all`/`--all` arg) |
| `plot_flow2d.py` | Empirical flow field (speed + streamlines) | all laser-off |
| `plot_scatter_perf.py` | Δ choice loc. vs Δ performance | x: correct; y: all |
| `plot_geometry.py` | Late-delay positions per (mouse, odor_pair) | correct only |
| `plot_marginal.py` | 1D code vs time (Naive/Expert) | all laser-off |
| `plot_occupancy.py` | 2D KDE occupancy at BINS_LATE | all laser-off |
| `plot_scatter_ab.py` | Per-animal A/B endpoints | all laser-off |
| `fig_overlaps_flow_empirical.py [mouse\|pooled] [--pooled]` | **Autonomous flow (recommended): empirical binned, sample×choice, trainTEST, DELAY-only; A/B winner-take-all (default), `--pooled` to average** | DPA Expert correct |
| `fig_overlaps_flow_planes.py [mouse\|pooled]` | Rate-net flow on the same plane (reference; CV-tunes ridge, held-out vel-R²≈0 → noise floor, see below) | DPA Expert correct |
| `fig_overlaps_flow_inputs.py [mouse\|pooled]` | Autonomous + input-driven flow grid (8 panels: A/B/Go/NoGo/Cue/C/D), **empirical binned** | DPA+Dual Expert correct |
| `fig_overlaps_traj.py` | sample×choice condition-mean trajectories (per-mouse+pooled) + sample-memory(t) | DPA Expert correct |
| `fig_overlaps_cosine.py` | cosine similarity of the decoder **weight axes** (neuron space): within-code epoch×epoch stability + between-code alignment vs chance ±1/√N. Result: codes mutually **orthogonal** all trial (sharpen with learning); axes temporally stable. Needs `run_overlaps.py --save-weights` (→ `weights_*_raw.pkl`) | all laser-off |
| `fig_overlaps_codes_1d.py` | 1D codes: sample/choice/test each on its own code (**DPA-only**) + task on choice code (all tasks); **nine train epochs sweeping the trial (stim/ed/md/gng_rwd/delay/ld/test/choice/dpa_rwd) × both stages**; grand-mean over mice + per-mouse 9×4 grid + pooled reference; pre-test anchors flag the test panel ⚠ pre-test confound. NB the Go/NoGo distractor + GNG reward are **Dual-only** — on the DPA-only panels ed/md/gng_rwd/ld are uninterrupted maintenance timepoints (events live in the task panel). Key: sample epoch-invariant (stable memory), **choice/lick code non-flat in the delay = maintained no-lick action set (dual-coding), not foreknowledge** | correct |

See `docs/overlaps/routines.md` for run commands and output paths.

---

## Flows from the overlaps (CCGD) codes — method & rationale (2026-06-22)

**Data shape.** The overlaps result is a **train × test** generalisation tensor
`X[trial, {proba, decision-fn}, T_train, T_test]`, with the decoded variables (sample / choice / test)
**stacked along the trial axis** (`y['target']`) — the same physical trials decoded once per target.
The usable code is the **decision function** (`X[:, 1]`); the `X[:, 0]` probability sits ~0.5 and is
uninformative. (NB: this supersedes the earlier "diagonal" reading — code 0/1 are proba/df, *not*
sample/choice; the codes are split by `y['target']`, not by the second axis.)

**Method — collapse the train axis, then run the dPCA routines.** A flow needs a **fixed measurement
axis across time**, so for each code we **average the decoder over a chosen TRAIN epoch** and read the
resulting fixed axis across **test time**:
`code(t) = mean_{train ∈ epoch} decision_fn[:, train, t]`.
This reduces the train dimension to one decoder → a per-trial time trace exactly analogous to a dPCA
latent. We then run **the same routines as the dPCA flows** (condition means by sample, delay window,
flow field, trajectories). Using the *diagonal* (train=test at each t) would change the axis every
bin and a flow/phase-portrait would be ill-defined — hence a fixed train epoch.

**Why averaging over a train epoch is justified (the key assumption).** The codes are *dynamic* (the
decoder weights drift across training time), **but the sample-memory subspace is approximately stable
through the delay** — the early-delay (ED) sample code ≈ late-delay (LD) sample code ≈ the **dPCA
sample axis**. Under that premise, averaging the decoder over a delay/test epoch yields one
well-defined sample axis, and the flow built on it is comparable to the dPCA flow. (This is exactly
the cross-temporal-generalisation premise of CCGD: a code that generalises across train/test times
*is* a fixed axis.)

**Validation of the stability assumption** (`fig_overlaps_traincmp.py [sample|choice|test]` →
`figures/overlaps/traincmp/`). Each code was recomputed under every train reduction — `trainED`,
`trainLD` (late delay 39–53), `trainDELAY` (full), `trainCHOICE`, `trainTEST`, and the **diagonal**
(train=test) — pooled, Expert DPA; reported is the corr of `code(level1−level0)` over the variable's
active window. The three codes behave differently:

- **sample (delay window): stable across the whole trial.** corr(LD,TEST)=0.98, corr(CHOICE,TEST)=1.0,
  DELAY↔all=0.91–0.96, diagonal↔others=0.85–0.97; only **`trainED` is an outlier** (0.75–0.82, the
  early-delay encoding transient, still 0.95 with full-delay). ⇒ ED ≈ LD ≈ test ≈ diagonal — the
  memory subspace is stable, so train-epoch averaging is valid and `trainTEST` (or any late-delay/test
  epoch) is a sound default for the sample axis.
- **choice (response window): response-locked, generalises back to late delay.** CHOICE↔TEST=0.99,
  LD=0.88, DELAY=0.84, diagonal=0.86, **ED=0.49** (choice not yet present early). Fine to read on the
  delay if trained late, but not from early delay.
- **test (response window): meaningful ONLY at the response.** The corr matrix splits into two
  **anti-correlated** blocks — delay-trained (ED/LD/DELAY) vs response-trained (CHOICE/TEST/diagonal)
  correlate **−0.6 to −0.92**. The test odor isn't present during the delay, so the delay-trained
  "test decoder" is reading a confound (sample/distractor) that is *anti-aligned* with the true test
  code. ⇒ **use the test code only in the response window**; a delay-trained test axis is spurious.

Net: train-epoch averaging is legitimate **within a variable's active period** — trial-wide for
sample, late-delay→response for choice, response-only for test.

**The sample × choice plane.** x = sample code (`target=='sample'` df, train-epoch-averaged),
y = choice code (`target=='choice'` df). Trained on the **TEST epoch** this is the condition closest
to the *memory pushed into the no-lick region* (`plot_traj2d_planes.py nofold trainTEST sample`,
Expert DPA). Frame = raw per-mouse BL-std units, per-axis limits from the cross-mice mean (as in
`plot_traj2d_planes`); per-mouse mean → cross-mice mean for the overlaid trajectory.

**Autonomous flow = DELAY only.** Like dPCA, the autonomous flow uses the **delay window**
(`bins_DELAY` ≈ 21–53) — memory dynamics, not the test/response. (Running it into the test epoch was
a mistake; corrected.)

**Empirical (binned), NOT rate-net.** The dPCA-style rate-net fit does **not** transfer here: the
CCGD condition means are noise-dominated (jaggedness ≈ 1) and don't settle within the delay, so the
fitted flow places its fixed points by **extrapolation far off the data** (attractors outside the
trajectory bbox). **Quantified by CV** (`fig_overlaps_flow_planes.py` 5-fold CV-tunes the ridge on
held-out condition-mean velocity R²): the **held-out vel-R² is ≈ 0 at every ridge** — CV picks
ridge≈2.0 with R²≈0.01, vs −0.03 at the dPCA default 0.2. So **ridge is not the bottleneck, the
velocity SNR is**: the rate-net flow is at the **noise floor** and can't be meaningfully improved by
regularization. (In-sample vel-R²≈0.2, ~0.45 with temporal smoothing, is optimistic — held-out ≈0.)
So use
the subproject's **empirical binned flow** (`collect_flow_points` + `build_smooth_field`:
per-(mouse, odor_pair) **mean-trajectory** velocities — which cancel the per-trial CCGD noise,
σ≈5 BLσ/bin — then Nadaraya–Watson smoothed). Faithful to the trajectories, no extrapolation.
**A and B are binned separately and combined winner-take-all** (default; per cell keep the velocity
of whichever sample has more data — avoids A/B cancellation near the origin, per `feedback.md`);
`--pooled` reverts to A+B averaging. **Pooled (~1150 vel pts) is reliable; per-mouse (~128 pts) is
sparse.** Canonical script: `fig_overlaps_flow_empirical.py [mouse|pooled] [--pooled]`.

**Rank-2 low-rank port (2026-06-30 — `fig_overlaps_flow_lowrank_shared.py`).** The dPCA rank-2
gain-modulated flow (`pca/fig_dpca_flow_lowrank_shared.py`) ported verbatim onto the CCGD plane
(model/regimes/CV/plot unchanged; only the data section swapped — sample×choice from the codes via
the matched `target` blocks). It renders the same 8-panel autonomous+inputs grid, but is **velocity-
SNR-limited**, consistent with the rate-net verdict above: **pooled CV vel-R² ≈ 0** (Expert: partial
+0.035 > independent −0.015 > shared, same ordering as dPCA). **The stimulus (sample) input regimes
DO validate** — A +0.93 / B +0.78 (independent), Cue +0.30, Go +0.13 — but the **autonomous field is
weakly constrained (+0.02–0.11) and C/D do NOT validate (−0.05/−0.11)** (the overlaps *test* code is
the weak, response-locked one — unlike dPCA where C/D were the best regimes). So: the fields render
and the sample-driven flows are genuinely predictive, but don't over-read the autonomous wells or
C/D — the CCGD velocity SNR is the ceiling (same conclusion as the empirical/rate-net analyses).

**"Can the fits be improved?" — NO on velocity R² (data ceiling); and fixed points must sit at the
trajectory endpoints (2026-07-01).** Both principled velocity denoisers were tested and FAIL:
**temporal `--smooth`** (0.035→0.008: helps clean regimes, hurts noisy C/D → flat-to-down) and
**multi-bin `--vstep`** (the empirical-flow VEL_STEP trick: WORSE at every step, 0.035→−0.07→−0.13,
because a chord velocity ≠ the model's instantaneous tangent in fast/curved regimes). ⇒ the pooled
velocity R² is genuinely at the CCGD velocity-SNR ceiling; `--vstep` default stays 1.

**Improving the flows (2026-07-01) — the delay plane + endpoint anchoring.** Two levers fix the
autonomous. **(#2, the big one) read the codes on the DELAY-trained axis** (`--train delay`, now the
default) — we draw the *delay* flow, so use the delay axis: CV-honest, smoother (sample-code
jaggedness 0.045 vs 0.064 on TEST), better A/B separation (3.0 vs 2.55), and it moves **B from +0.46
(buried near the saddle on the TEST axis) to +1.25 — a real well** (both states in no-lick). (The
weight-axis projection showed even bigger separation but is in-sample-confounded — only the fold-avg
weight was saved, no CV; the delay-df is its CV-honest equivalent since the df *is* the weight-axis
projection.) **(#1) endpoint anchoring** (`--anchor 8`, independent mode): add the settled trajectory
endpoints as v=0 anchors in the LS so the field's attractors land ON the endpoints. Together: the
autonomous is now **faithfully bistable — two attractors AT the A/B endpoints** (A≈−1.8 on its −1.76
endpoint, B≈+1.6), streamlines converge to both, trajectories terminate there. Genuine (delay axis
separates B), not forced. NB pooled CV drops (−0.17) because C/D are *more* noise on the delay axis
(delay-trained test = the documented delay-confound) — expected; don't read response regimes on a
delay axis. What FAILED: **#1 alone on the TEST plane** can't place B (too near the saddle → 2nd
attractor lands at ~1.0, offset/fragile) — the plane change (#2) is what makes B a real well.

**Input-driven windows = [odor onset, odor offset + calcium margin] (2026-07-01).** Each input regime
is fit over the odor's presentation plus a margin for the slow calcium tail (GCaMP decay ~1.5–2 s),
NOT the long arbitrary tail from the dPCA port. Odors: sample/STIM 15–17, distractor/DIST 30–32, test
57–59; `--margin` (default 12 bins ≈2 s) → A/B 15–30, Go/NoGo/Cue 30–45, **C/D 57–72 (was 57–84)**.
Cutting the post-response relaxation tail **fixed C/D**: in-sample vel-R² **C −0.05→+0.65, D −0.10→+0.48**,
and they now show the proper **AC→lick / BC→no-lick bimodal diagonal** with trajectories reaching both
fixed points (the dPCA C/D signature). Go/Cue also improved. The autonomous is NOT input-driven → keeps
the delay maintenance window.

**Single train axis + per-regime input windows (2026-07-01, final design).** ONE train axis for the
whole figure (`--train`, default **delay** — draws the delay flow, faithful bistable autonomous). The
only thing that varies per regime is the READ WINDOW = the odor±margin window above. The **window is
what fixes C/D**, not the axis: on the single delay axis, moving C/D from the long 57–84 window to the
odor window 57–72 takes them from −0.25 to **+0.46/+0.30** (the post-response relaxation tail was
noise). So a per-regime *plane* is NOT needed — one axis suffices once the windows are right. (An
earlier per-regime-plane version, delay for memory / test for C/D, was tried and reverted per the
single-axis decision; test does make the C/D bimodal diagonal slightly cleaner, +0.65/+0.48, but a
single delay axis keeps the panels comparable and the autonomous faithful.) **Anchor only the
autonomous** (the true attractor regime; inputs are driven). Residual: Cue is the lowest-SNR panel.

**Train-axis options (`--train`), pooled CV vel-R² and character:**
- **`delay` (default)** — the principled fixed-axis; faithful bistable autonomous + strong Go/NoGo; C/D
  softer. CV −0.23.
- **`test`** — clean C/D bimodal (+0.65/+0.48) but weak autonomous B-well; Go/NoGo weak. CV −0.05.
- **`ld`** — like delay, slightly noisier (fewer bins). CV −0.39.
- **`diag`** — the generalization-matrix **diagonal** (train==test per bin) → each regime auto-read on
  its contemporaneous decoder: **bistable autonomous AND clean C/D (+0.70/+0.57), best CV +0.17**. BUT the
  axis **rotates per bin** so the plane isn't a fixed subspace → a flow/phase-portrait is looser/ill-defined
  (works here only because adjacent-bin axes are stable, cosine 0.4–0.9); A/B fits also drop (+0.63 vs
  +0.96, the stable sample regime prefers one fixed axis). Best-fitting but conceptually looser than `delay`.

**FP placement — the key correction.** A flow is wrong if the trajectories don't terminate at the
marked fixed points. Root-finding on the fitted field violates this at low SNR: verified at the CV
gain (0.2) the field's one attractor sits on the **A** endpoint (−2.29,−0.30 ≈ data −2.09,−0.35;
speed 0.02), while **B**'s shallow well is not classified though the data settles there (speed 0.03).
So mark fixed points at the condition-mean **trajectory endpoints** (last-5-bin mean) — the overlaps
subproject convention (`plot_flow2d.py`: "trajectory endpoint is correct by construction; root-found /
speed-min fp is displaced in a noisy field"). Then trajectories terminate at the fps by construction,
for every panel. **RETRACTED:** an earlier attempt to force autonomous bistability by *raising the
gain* to hit bootstrap P₂≥0.5 (reported "P₂=0.62 at gain 0.7") was WRONG — the raised gain manufactured
a 2nd attractor **off** the B endpoint (B endpoint speed rose 0.03→0.09), i.e. exactly the "fp not where
the data goes" bug; removed. Honest reading: at the data-faithful CV gain the autonomous field is
**monostable-leaning** (A well; B a shallow shoulder the data holds but the field doesn't) — consistent
with the empirical-flow "A strong / B weak, don't over-read one well vs two."

**Result.** Sample memory is encoded and **maintained** through the delay (A/B separate along the
sample axis), and **both A and B Expert-DPA delay states are pulled into the no-lick (lower) half**
of the choice axis — the predicted geometry — though **asymmetrically: A strongly (≈−0.66 BLσ below
baseline), B weakly (≈−0.11)**. Whether the residual delay flow reads as one well or two is shallow
and representation-dependent — don't over-read it (the dPCA bistability set does not transfer; see
`docs/pca/flows_handoff.md`).

---

## Interpretation & synthesis (2026-06-24) — the dual-coding geometry, with a mechanism

The codes (1D sweep), the weight-based cosines, and the cross-stage test tell one story.

**1. Three stable, independent codes.** mPFC holds `sample`, `choice`, `test` on **mutually
orthogonal axes** (between-code cosine at the ±1/√N chance floor at every epoch, both stages),
each **temporally stable** within its window (within-code cosine 0.4–0.9 ≫ floor; sample most
stable across the whole delay). Not a shared rotating axis — separate fixed subspaces. That is the
structural precondition for holding S1 *and* a decision state without the two reading each other out.

**2. Dual coding is live during the delay.** The epoch sweep shows the **choice/lick code is
non-flat already in the delay** (ed/md), coexisting with the held sample code. Read correctly this
is a **maintained no-lick action set** (the DPA default), NOT foreknowledge of the answer (the
correct lick is undetermined until S2). So the delay simultaneously holds *what S1 was* (sample
axis) and *the current action disposition* (choice axis, defaulted to no-lick).

**3. Learning orthogonalizes memory×action, which makes the no-lick push "free".** Two things
change Naive→Expert: (i) the **no-lick push** — the DPA delay state moves deeper into no-lick along
the choice axis (deepens with learning; correlates with DPA accuracy across mice — the
depth↔performance scatter); (ii) **orthogonalization** — `sample` and `choice` axes are **mildly
anti-aligned in Naive (cos −0.07) and become orthogonal in Expert** (p=0.020, 7/9 mice,
reliability-controlled), and *only* this memory×action pair. The Naive anti-alignment means the
sample-B direction partially overlaps the action axis, so pushing along choice would **drag the
sample code with it**. Once orthogonal, the state can be pushed far into no-lick **without touching
sample discriminability**. So orthogonalization is the **geometric mechanism that licenses the
no-lick push** — exactly the paper's "delay activity moves along the lick axis without disrupting
the sample axis," now with a measured cause. The two learning effects are one mechanism, not two.

**Why it matters.** The dual task is adversarial to WM (hold S1 across a delay containing a
distractor odor + a competing GNG lick). Independent, stable subspaces are what let the memory
survive interference while a decision is staged in parallel; learning sharpens the most
task-critical separation (memory vs action) selectively — `sample–test` and `choice–test` do not
significantly change.

**Honest limits.** "Orthogonal" = at the chance floor = statistically *independent*, not pushed
beyond chance; the orthogonalization is a **small shift** (Naive ~0.08 just above floor → Expert at
floor) — real and controlled, but modest. n=9 mice, single paired test; cosines from fold-averaged
weights (mild shrinkage). The delay choice signal is a maintained action set, not anticipation.
Corroborates the dPCA no-lick push (`docs/pca/flows_handoff.md`) from an independent weight-based
angle — convergent, same conclusion.

---

## Plotting the codes — 1D and 2D trajectories

The CCGD has **three target codes** — `sample`, `choice`, **and `test`** (selected by `y['target']`;
the same trials decoded once per target). Each is a **decision-function axis** (`X[:,1]`,
train-epoch-averaged, read across test time, per-mouse BL-std normalised — see the method section
above). `task` is *not* a decoded target — it is obtained by **averaging trials per task**
(DPA / Go / NoGo) on a chosen code.

**1D trajectories (code vs time).** Each variable on **its own code**, split by the variable's label,
mean ± SEM (project conventions — `plot_mean_sem`, `add_vlines`, colours; see `routines.md`). Each
code carries its variable in its natural epoch (sample in the delay; choice/test at the response):

| Panel | code (`y['target']`) | split by | colours |
|---|---|---|---|
| sample | `sample` | `sample_odor` | A `#332288` / B `#44AA99` |
| choice | `choice` | `choice` | No-lick `#377eb8` / Lick `#4daf4a` |
| test | `test` | `test_odor` | C `#377eb8` / D `#4daf4a` |
| task | `choice` (or any) | `tasks` | DPA/Go/NoGo = `muted[3]/[0]/[2]` |

Scripts: `fig_overlaps_codes_1d.py` (per-mouse 9×4 grid + grand-mean over mice; sample/choice/test
DPA-only, task all-tasks; both stages), `plot_marginal.py` (Naive vs Expert).

**2D trajectories (code pairs).** Project two codes at once → a 2D state path over time:
- **`sample × choice`, `sample × test`, `choice × test`** — `plot_traj2d_planes.py` (3 panels;
  time-gradient paths + direction arrows + SEM-over-mice band; per (stage, condition); `--no-fold`
  = raw pooled axes, default folds each code by its label to keep A/B from cancelling).
- **`sample × choice` only** — `plot_traj2d.py` (2D trajectory + KDE strip).
Frame: raw per-mouse BL-std units, **per-axis** limits from the cross-mice mean (not square/equal-
aspect); per-mouse mean → cross-mice mean for the path, SEM over mice. Train epoch set via
`TRAIN_EPOCHS` (`trainTEST` / `trainDELAY` / `trainCHOICE` / `trainED`).

**Flows** use the same plane + train-epoch averaging, **delay-only**, empirical-binned — see the
method section above (`fig_overlaps_flow_empirical.py`).

---

## Locked-in design choices

- **Colors**: sample A = `#332288` (indigo), sample B = `#44AA99` (teal)
- **Condition titles**: DPA / Go / NoGo (strip "Dual" prefix)
- **Trajectory limits**: xlim=(-4,4), ylim=(-2,6), yticks=[-2,0,2,4,6]
- **BINS_LATE** = `BINS_DELAY[int(0.6*len):]` ≈ bins 39–53 (last 40% of delay)
- **Stage labels**: Naive / Expert (from `days=['first','last']`)
- **Fixed points**: mean of grand_mean_traj over BINS_LATE (NOT speed minimum)
- **Flow field heatmap**: magma colormap, hybrid speed (BINS_LATE at fixed points, all-delay elsewhere)
- **Condition naming**: `cond.replace('DualGo','Go').replace('DualNoGo','NoGo')`
