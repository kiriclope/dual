# Compositional learning by geometric editing — main paper (draft v4)

> **Thesis:** cortical computation is low-dimensional and carried by the *geometry* of population
> activity; the brain composes a new task by **editing that geometry** — reusing a pre-existing neural
> manifold and repositioning the memory state along its existing axes — rather than by building new
> coding dimensions.
> Working title: *"Compositional learning by geometric editing: mPFC reuses a low-dimensional manifold
> and repositions the working-memory state to interleave memory and action."*
>
> **Flow fields / attractor-dynamics are OUT of the paper (2026-08-03) — kept as "extra"** (gallery pca/
> overlaps tabs), so Fig 2 is dPCA GEOMETRY + the no-lick push (no flow model), and Supp Fig 7 is dropped.
>
> Figure order: **1 Behaviour · 2 dPCA (low-D geometry) · 3 Overlaps (reused geometry) · 4 Model
> (deferred) · 5 Opto (causal).** Publication-ready standard: **every panel is referenced and described.**
> Panel letters/stats verified against the RENDERED figures (Fig 1 `behavior_main.png`;
> Fig 2 `fig_dpca_story_main.png`; Fig 3 `fig_overlaps_main_ab_dpaact.png` — the CURRENT build, not the
> stale `fig_overlaps_main_ab.png`; Fig 5 `behavior_opto_main.png`). Caveats to carry are flagged inline
> as _(caveat: …)_; guardrails and to-reconcile items at the bottom.

---

A hallmark of flexible behaviour is the ability to compose a new task out of computations the brain
already performs. Rather than learn each new demand from scratch, cortex may reuse pre-existing
representational structure and adapt it — an economical strategy if computation is implemented, as
increasingly argued, in the low-dimensional geometry of population activity. We asked whether prefrontal
cortex acquires a two-component task in exactly this way: by reusing an existing population manifold and
making a *targeted edit* to its geometry, rather than by forming a new representation.

We trained head-fixed mice (n = 9) on a delayed paired-association (DPA) working-memory task — a sample
odour (A or B), after a 6 s delay, matched to a test odour (C or D) with a lick-for-reward response. On a
subset of sessions a Go/NoGo (GNG) discrimination was embedded in the delay as a distractor (DualGo /
DualNoGo trials), interleaved with distractor-free DPA-only trials, so that the same delay period
alternately did and did not demand an intervening action. This makes the dual task an explicit
*composition* of two computations — hold a memory, and act on an interposed cue — that the animal must run
concurrently without letting one corrupt the other. Mice were brought through a fixed curriculum
(DPA → GNG → Dual task) and imaged with two-photon microscopy in prelimbic mPFC across learning, yielding
a pseudo-population of 3,319 neurons spanning naïve and expert stages.

We find that mPFC does not construct a new representation for the dual task. The memory and the action are
carried on near-orthogonal axes of a low-dimensional manifold that is already present in naïve animals;
learning leaves this factorized geometry in place and instead repositions the working-memory state along
a pre-existing action axis into the no-lick region — an edit that protects the memory, predicts performance
across animals, and is set by top-down input
from anterior cingulate cortex (ACC). Computation here is low-dimensional and geometric, and composition is
an edit to that geometry.

## 1. Composing the two tasks is costly: an intruding action corrupts the working memory (Fig. 1)

Mice learned the two component tasks under a fixed curriculum (task structure and training schedule in
Fig. 1a). Across six dual-task sessions DPA and GNG accuracy both rose and converged (Fig. 1b;
mixed-effects model, GNG−DPA condition β = +0.037, p = 0.045; condition × day β = −0.039, p = 8 × 10⁻⁴).
The GNG gain was carried by the NoGo condition — learning to *withhold* rather than to lick (Fig. 1c;
NoGo−Go β = +0.072, p = 0.034) — and, on the DPA side, by the unpaired trials, which began far below the
paired trials and improved fastest (Fig. 1d; unpaired−paired β = −0.185, p < 10⁻⁴; condition × day
β = +0.088, p < 10⁻⁴). The full set of condition and condition × day effect sizes is summarised in
Fig. 1f.

Composing the two tasks was not free: the interposed action interfered with the memory. Unpaired DPA
accuracy was selectively depressed when the distractor required a Go response (Fig. 1e; Go−DPA
β = −0.073, p = 0.038) but not a NoGo response (NoGo−DPA p = 0.78), identifying the *intrusive lick* — the
action itself — as the source of interference. Accordingly, on NoGo trials a spurious delay-period lick
predicted a subsequent DPA error, strongly in naïve animals and not once expert (Fig. 1g; GEE logistic on
NoGo trials, naïve odds ratio = 0.56, p = 0.006; expert OR = 0.76, p = 0.50): the cost of an intruding
action on the memory is largest before the composition is consolidated. _(caveat: Fig. 1g is a
within-trial lick↔accuracy association, correlational rather than causal.)_

Even at expertise the composition was imperfect. Across animals, expert DPA and GNG performance were
decoupled (Fig. 1h; Pearson r = +0.10, p = 0.80; Spearman ρ = +0.35, p = 0.36; n = 9), with no animal
reaching the both-tasks-optimal corner (mean shortfall ≈ 0.18; mean DPA 0.88, GNG 0.87), whereas in naïve
mice the two accuracies still co-varied (r ≈ 0.66–0.74). The behavioural problem is therefore precise:
the memory must be shielded from the action the animal is simultaneously required to produce. We next
asked how the population solves it. _(caveat: n = 9, underpowered; Fig. 1h supports "decoupled / no
trade-off correlation", not a significant trade-off. LMM random-effect variance is near its boundary, so
per-day curve p-values are mildly anti-conservative.)_

## 2. Low-dimensional dPCA geometry: near-orthogonal factors, and learning pushes the delay memory into no-lick (Fig. 2)

To see how mPFC arranges the memory and the action, we applied demixed PCA (dPCA) to the
condition-averaged pseudo-population, factorising activity into marginal axes for sample identity, test
identity, choice (lick / no-lick), the task/action context (DualGo vs DualNoGo vs DPA), and a
condition-independent time component (schematic, Fig. 2a). The computation was low-dimensional: the
sample × choice condition means lived on an approximately two-dimensional manifold (Fig. 2b; expert top-2
explained variance ≈ 94%; participation ratio ≈ 2.2). Time dominated the demixed variance (≈ 54%),
followed by the task/action axis (≈ 31%), with the pure sample and choice axes low-variance (≈ 7% each)
and the test axis ≈ 1% — low-variance but reliably time-locked to their epochs (Fig. 2c). The code is
thus factorised: what-to-remember and what-to-do are separable, near-orthogonal directions of one
population geometry. _(caveat: the per-task figures are a variance proxy from demixed condition-mean
components, not exact dPCA marginal EVR; the ~2-D figure is the state *geometry* — full dynamics are
higher-rank (~62–67% captured at rank 2, no elbow), so we describe a "rank-2 geometry", not rank-2
dynamics.)_

The single-axis trajectories sharpened with learning without reorganising (Fig. 2d, naïve top vs expert
bottom for the sample, test, choice and task axes), and the factorisation was *refined* rather than
replaced (Fig. 2e): the choice and task/action axes became more aligned (|cos| 0.147 → 0.222, Δ = +0.076,
p < 0.001), binding the lick decision to the action context, while the sample and test axes de-mixed
(|cos| 0.098 → 0.033, Δ = −0.065, p = 0.008), sharpening the separation of memory from match. The scaffold
is retained; its factors are tuned. _(caveat: axis-mixing significance is a neuron bootstrap over the
shared 3,319-neuron pool, not an across-animal test.)_

The decisive change was confined to a single pre-existing axis. The task/action axis orders the conditions
by their required action (DualGo/lick positive; DualNoGo and DPA/no-lick negative), and along it the DPA
delay state was pushed deeper into the no-lick region with learning — visible in the tasks-axis trajectory
(Fig. 2d, rightmost column) and quantified per mouse (mean push = −0.59, paired Wilcoxon p = 0.012, 8/9
mice; Fig. 2f). The shift was protective — the two sample states (A vs B) stayed separated across learning
(Fig. 2g; separation |B−A| 1.65 → 2.33, p = 0.10; significantly sharpening on all-trials, p = 0.02) — and
specific: the deepening survived removal of the condition-independent time ramp (−0.59 / −0.58 / −0.61
after projecting out the top 0/1/2 time directions, p ≤ 0.012, 8/9). Learning thus repositions the delay
memory along a pre-existing action axis, into no-lick, without disturbing what is remembered. _(caveats:
the naïve level marked in Fig. 2d,f is a display anchor — naïve mice already sit at a non-zero no-lick
depth (≈ −0.9); learning *deepens* an existing push, the clearest evidence that the structure pre-exists.
The delay push is specific to the task/action axis and does not appear on the choice axis, which is
resolved only at test; earlier "choice-polarization" claims were retracted as an orientation/leakage
artifact. A dynamical (low-rank flow / attractor-landscape) account of this push is NOT claimed from the
dPCA geometry — that analysis is retained as "extra", outside the paper.)_

## 3. A reused geometry: memory and action share one manifold, and learning repositions the state on it (Fig. 3)

The dPCA picture is pooled and condition-averaged. To establish that the same geometry holds at the
single-trial, single-animal level — and to connect the edit to behaviour — we trained cross-generalising
decision-code decoders (CCGD): a balanced logistic decoder per mouse and stage for each of the sample,
GNG, test and choice(lick) variables, read across the whole trial (Fig. 3a, naïve top / expert bottom).
The geometry matched the dPCA account: the sample memory code was stable and epoch-invariant, the GNG code
diverged sharply around the cue, the test code appeared only at test, and the choice/action code separated
only at the response — four near-orthogonal, temporally stable directions of one manifold. The action code
was decodable and, importantly, its *decodability was stable across learning* (Fig. 3b1; action-code d′
naïve vs expert, Δ = +0.27, p = 0.246, n.s.) — so what learning changes is the state's *position*, not the
code's fidelity. The two tasks even shared a *single* action command: the DPA-lick and GNG-lick axes were
aligned above chance at both stages (Fig. 3b2; signed cosine naïve +0.12, p < 0.001; expert +0.18,
p = 0.001; label-permutation p = 0.002), so composing the tasks reuses the same lick direction rather than
building a second one.

Within this fixed geometry, learning repositioned the state exactly as the dPCA analysis predicted. The expert
DPA delay state sat further into the no-lick half of the action axis than the naïve state (Fig. 3c;
sample × choice planes for naïve and expert, and the per-mouse choice-code depth Naïve → Expert; mixed
model 9 mice/36 obs, β = −0.744, p = 0.046), specifically for sample A (A ≈ −1.45, p = 0.054; B ≈ −0.02,
p = 0.95). And the size of this geometric edit was behaviourally meaningful across animals: the more a
mouse pushed its delay state toward no-lick, the more its DPA accuracy improved (Fig. 3d left; between-
mouse per-mouse Spearman ρ = −0.83, p = 0.005, n = 9), with no relationship to Go/NoGo accuracy (Fig. 3d
right; ρ = +0.20, p = 0.61) — repositioning the state on the pre-existing manifold specifically buys memory
performance. The coupling held across every normalisation (ρ ≈ −0.83 to −0.90, all p ≤ 0.005) and a
resampling battery (jackknife; bootstrap CI excluding 0; permutation p = 0.008). As an internal
within-trial validation that this axis is the action code, on naïve error trials the choice code sat closer
to lick before false alarms than before correct rejections, though on the current build this did not reach
significance (Fig. 3e; sample-A/AD p = 0.204; sample-B/BC p = 0.833, n.s.). _(caveats: the push is
directional rather than a precise magnitude — on a fixed common axis it attenuates to a trend, part of the
per-stage change being decoder-axis reorganisation, so it is a re-sculpting of the state on a retained
scaffold, not a rigid translation; the behavioural coupling is an n = 9 individual-difference correlation,
robust within the overlaps analysis but null on the dPCA-derived depth (r = +0.46, p = 0.21); the
false-alarm/correct-rejection panel (Fig. 3e) is the weakest element and is currently n.s.)_

That the code is reusable and compositional was borne out at the single-neuron level: the near-
orthogonality did not arise from conjunctive mixed selectivity but from largely separate, independently
tuned populations — per-neuron permutation tests placed cross-variable co-tuning at chance (fraction tuned
to two variables ≈ product of the marginals), with more neurons selective for the action-relevant GNG
variable (≈ 39%) than for the purely mnemonic sample identity (≈ 10%) (Supplementary Fig. X). A factorised
substrate of this kind is what makes the two tasks composable without cross-talk.

## 4. A low-rank circuit model of the gated no-lick edit (Fig. 4)

_[Deferred — modelling figure to be drafted. Placeholder to preserve final figure numbering.]_

## 5. ACC→mPFC input sets the state's position on the manifold without changing its content (Fig. 5)

If composition is implemented as an edit to this geometry, some input must supply the edit. We tested the
projection from ACC to the recorded prelimbic mPFC (viral/optogenetic strategy and trial timeline,
Fig. 5a; hSyn-GCaMP6s in mPFC, CaMKII-Jaws-tdTomato in ACC, 635 nm on a pseudo-random 50% of delay
periods). Chronic, every-trial silencing during training (between-group, opto vs control, 9 vs 9 mice)
impaired DPA learning (Fig. 5b; learning curves), an effect that a mixed model localised to DPA and, most
strongly, to its unpaired trials, sparing Go/NoGo (Fig. 5c; group and group × day contrasts: DPA
β = −0.06, p = 0.009; DPA-unpaired β = −0.12, p = 0.014; GNG n.s.) — the same DPA-selective vulnerability
seen behaviourally in Fig. 1.

In the imaged cohort we silenced ACC→mPFC transiently, on half of the delay periods (within-mouse laser
ON vs OFF, Jaws, n = 5), and projected both trial types through the fixed, laser-OFF-trained choice(lick)
axis to read the state's *position* on the learned geometry. Transient silencing produced no gross
behavioural change (Fig. 5d, DPA spared, p = 0.40; Fig. 5e, GNG spared, p = 0.24) but moved the delay
choice code, per mouse (Fig. 5f; DPA choice-code depth OFF vs ON, 5 Jaws mice). Across animals this
displacement traded the two tasks off against one another (Fig. 5g; Δdepth vs ΔDPA−ΔGNG, r = +0.53,
p = 0.016, n = 20), an effect carried by the Go/NoGo arm (Fig. 5i; Δdepth vs ΔGNG, r = −0.65, p = 0.002)
rather than the DPA arm (Fig. 5h; Δdepth vs ΔDPA, r = +0.34, p = 0.146, n.s.). In laser-ON trials the mice
still occupied the same suboptimal DPA–GNG balance as at baseline (Fig. 5j; r = +0.44, p = 0.20). Critically,
the input moved the state without altering the manifold's content: the discriminability (d′) of both the
sample-memory axis (Fig. 5k; A vs B, late delay) and the Go/NoGo choice axis (Fig. 5l; Go vs NoGo,
mid-delay) was spared under laser (LMM laser effect sample p = 0.34, GNG p = 0.74; n = 10). ACC→mPFC input
therefore acts as the knob that sets *where* the delay state sits along the shared action axis — the very
edit that composes memory with action — while leaving the coded content intact.

_(caveats to carry: this figure combines two designs — a chronic between-group cohort (Fig. 5b,c; 9 vs 9)
and a within-mouse transient-silencing cohort (Fig. 5d–l; Jaws, n = 5) — justified only because both
target the same ACC→mPFC projection; state this explicitly. The trade-off in Fig. 5g–i is a between-animal
coupling; the n = 20 points are 5 mice × stage × sample and thus pseudoreplicated, so the raw r is
anti-conservative — under a mouse-clustered model the joint trade-off (g) is a trend (p ≈ 0.11) and only
the ΔGNG arm (i) survives (β = −0.013, p = 0.018); frame i as the robust arm and g as the trend. The axis
and window are pre-committed (the locked main-overlaps axis), with the window sweep reported to avoid
cherry-picking. Do not headline the trial-level signal × laser interaction (pseudoreplicated); the
mouse-level d′ result (Fig. 5k,l) is the honest read. An alternative build of the coupling over all 7 laser
mice (5 Jaws + 2 ChR, Spearman) gives GNG ρ ≈ −0.90, p ≈ 0.007, DPA null — do not mix the two n's in one
sentence.)_

## Synthesis

Together these results describe compositional learning as **geometric editing**. mPFC represents the memory
and the action on near-orthogonal, reusable axes of a low-dimensional manifold that predates expertise
(Fig. 2, 3); learning does not add representational dimensions but repositions the working-memory state
along a pre-existing action axis into the no-lick region
(Fig. 2), an edit that shields the memory and, animal by animal, predicts how well the two tasks are
composed (Fig. 3); and top-down ACC→mPFC input supplies this edit — setting the state's position on the
manifold without changing what the population encodes (Fig. 5). Computation is low-dimensional and
geometric, and to compose is to edit that geometry.

---

## Extended Data Figures

> Consolidated from the granular supplement set into **8 multi-panel Extended Data (ED) figures** (Nature
> Neuroscience caps ED at 10); the trial-count reporting figure → Supplementary Information. Each ED figure
> backs specific main-figure claims/_(caveats)_; all panels are in the shared house style (gallery Supp
> tab). Stats are current run values (verified 2026-08-03). **Trims applied 2026-08-03:** former S5
> (demixed axes) and S17 (d′ standalone) cut as redundant; former S13 folded into ED 6; flows (former S7)
> removed → "extra".

**ED Fig. 1 | Behaviour: learning curves (Fig. 1).** DPA/GNG, Go/NoGo, paired/unpaired and
unpaired-by-context curves over six sessions + the LMM effect-size forest, pooled and split by opsin/target
(Jaws, ChR, ACC) and for laser-ON trials. Condition effects reproduce (GNG−DPA β=+0.037 p=0.045; NoGo−Go
+0.072; unpaired−paired −0.185; Go−DPA −0.073). Learning is comparable across cohorts.

**ED Fig. 2 | Behaviour: the DPA↔GNG balance is not a trade-off (Fig. 1e/g/h).** (a–c) per-animal
DPA-vs-GNG scatter (Naïve co-vary r≈0.67 → Expert decouple r=+0.10), Pareto front (no animal on the
both-optimal corner), a small fixed dual cost (Δ≈−0.03), and a *positive* within-trial DPA|GNG-correct
coupling (GEE OR=2.03, p<0.001). (d–e) trial-history: a preceding dual trial lowers current-Go DPA accuracy
(OR=0.81, p=0.047; GNG history-independent), and the blocked-design switch-cost mirrors it (into-dual
OR=0.90, p<0.001).

**ED Fig. 3 | dPCA dimensionality (Fig. 2b).** Reduced-rank test — held-out fit rises smoothly with no
elbow at 2 (rank-2 = 62–67% of full); the condition-mean task manifold is ~2-D (top-2 ≈94% Expert/92%
Naïve, PR≈2.2). Backs the "rank-2 geometry, not rank-2 dynamics" caveat.

**ED Fig. 4 | dPCA no-lick push robustness (Fig. 2f).** The Naïve→Expert deepening reproduces in raw ΔF/F
(r≈0.997, not a z-score artifact), survives condition-independent time-ramp removal (q0/1/2 =
−0.59/−0.60/−0.61), holds on a Naïve-defined pooled basis (8/9; bootstrap CI [−0.56,−0.08]), and is
population- not individual-level (depth↔accuracy null, r=+0.46, p=0.21).

**ED Fig. 5 | Overlaps: coupling/push robustness + movement control (Fig. 3c,d).** (a–b) the Δdepth↔ΔDPA
coupling is ★ under all six normalisations (ρ=−0.83 to −0.90) and survives a fixed common axis (ρ=−0.72)
where the push attenuates to a trend; (c) a resampling battery (Mundlak β=−0.041 p=0.006; jackknife 9/9;
bootstrap CI [−1.00,−0.26]; permutation p=0.008), ΔGNG null throughout; (d) movement control — late-delay
licking is rare, the choice-code depth does not track it (ρ=+0.07), and the push/coupling are unchanged
with a lick covariate.

**ED Fig. 6 | Overlaps: the factorised geometry is robust (Fig. 3a,b).** (a) cross-temporal cosine matrices
— cross-code |cos| ≈ the 0.05 chance floor at all time-pairs, within-code diagonals 0.4–0.9, choice×GNG the
one least-orthogonal pair (~0.29); (b) modular, not mixed, selectivity — per-neuron permutation tuning
(sample 10 / GNG 39 / test 3 / choice 10 %, cross-variable co-tuning at chance); (c) decoder-variant
robustness — the main figure under L1 and LDA decoders (geometry/orthogonality decoder-invariant;
push/coupling clearest under logistic); (d) codes robust to the Go/NoGo distractor — panel-A codes split by
Go vs NoGo (sample/test unperturbed; the action code carries the distractor lick).

**ED Fig. 7 | Opto: chronic silencing + transient behaviour (Fig. 5b–e).** (a–c) control-vs-opto learning
curves for the ACC, ACC→Prl and Prl→ACC batches — ACC null; ACC→Prl impairs DPA (β=−0.06 p=0.009) and its
unpaired trials (β=−0.12 p=0.014); Prl→ACC impairs GNG; (d–e) transient within-mouse laser OFF-vs-ON curves
(Jaws n=5): DPA p=0.40, GNG p=0.24 — geometric, not a behavioural knock-down.

**ED Fig. 8 | Opto: laser ON−OFF coupling, 7 mice (Fig. 5g–i).** The acute causal analog of the learning
coupling over all 7 laser mice (5 Jaws + 2 ChR): GNG ρ≈−0.90 (p≈0.007), DPA null. Backs the axis choice and
the alternative-n disclosure.

**Supplementary Information**
- **Trial counts per mouse** — per-mouse × stage × task counts entering the pseudo-population (balanced by
  design; 5,568 laser-OFF trials total). _(Analysis-balanced counts, not raw behavioural trial numbers —
  see Methods.)_

**Omitted:** retracted dPCA choice-polarization figures (`dpca_flow_learning_ingain*`,
`dpca_flow_autonomous_choice`, `dpca_choice_ci_qsweep`); the flow-field / bistability analysis (former S7 →
"extra"); the standalone d′ figure (former S17 → already main Fig. 5k,l); demixed-axes loadings/mixing
(former S5 → covered by Fig. 2e + ED 6). **Author-supplied gaps still needed:** histology / viral
expression, imaging FOV + per-mouse cell counts, laser-power / opsin titration.

---

### To reconcile before submission (figure ↔ text integrity)
- **Fig. 3 file:** the current figure is `fig_overlaps_main_ab_dpaact.png` (5-panel A–E, n=9 Spearman in D,
  FA/CR n.s. in E). A stale `fig_overlaps_main_ab.png` (4-panel, 18-obs LMM coupling) still exists in the
  repo — do NOT ship it; regenerate/replace so only the current build is used (the gallery Main tab was
  repointed to the `_dpaact` file).
- **Fig. 2 flow-free — DONE (2026-08-03):** `fig_dpca_story_main.py` regenerated without the section-3 flow
  grid or the section-4 flow panels → 7-panel **a** schematic, **b** scree, **c** per-task variance, **d**
  trajectory grid, **e** axis-mixing, **f** no-lick push (p=0.012), **g** sample-memory preserved (p=0.10);
  title "The dPCA **geometry**…". Push stats reproduce (−0.59, 8/9). The Main-tab `fig_dpca_story_main.png`
  now shows the flow-free build.
- **Centered vs left titles:** the two newly-built supplements (Supp. Fig. 8 battery, Supp. Fig. 17 d′)
  use centered subplot titles; the house convention is left-aligned (`loc='left'`). Minor consistency
  sweep pending (the restyled existing supplements already use `loc='left'`).

### Open drafting decisions
- **Fig. 1h framing:** "decoupled / suboptimal balance" (defensible, n.s. correlation) vs a stronger
  "trade-off" (not supported at n = 9). Draft uses the conservative version.
- **Fig. 2 sample-coding wording:** "preserved" (correct-trials, p = 0.10) vs "preserved and sharpens"
  (all-trials, p = 0.02). Draft uses "preserved".
- **"reuse the same manifold"** is carried by the near-orthogonal, temporally stable axes present in naïve
  + the pre-existing no-lick well; phrased as "factorised scaffold retained; state re-sculpted", NOT
  "identical manifold, pure translation" (decoder-axis reorganisation is part of the push).
- **Supplementary Fig. X** (single-neuron selectivity) number to be assigned.
- **Fig. 5 numbering:** kept as 5 to reserve 4 for the deferred model; renumber if the model is cut.

### Framing guardrails (keep the thesis defensible)
- "low-dimensional **geometry** / rank-2 portrait", never "the dynamics are rank-2". Flow-field /
  attractor-dynamics claims are OUT of the paper (kept as "extra") — don't reintroduce "landscape",
  "gated deformation", "bistable" into the main text.
- near-orthogonality = the representational basis of compositionality (factorised code → combine without
  cross-talk) — the load-bearing, solid claim.
- the pre-existing naïve no-lick well is the strongest "structure already there, fine-tuned" evidence —
  lean on it for the reuse claim.
