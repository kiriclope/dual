# Overlaps Subproject Overview

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
