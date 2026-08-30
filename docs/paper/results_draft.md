# Compositional learning by geometric editing — main paper (draft v4)

> Methods paragraphs are staged per-figure in **`methods_notes.md`** (started 2026-08-10 with the Fig 2
> dimensionality block: cvPCA incl. the repeated-2-fold-vs-k-fold justification, jackknife CIs, η²
> decomposition, shattering).

> **Thesis:** cortical computation is low-dimensional and carried by the *geometry* of population
> activity; the brain composes a new task by **editing that geometry** — reusing a pre-existing neural
> manifold and repositioning the memory state along its existing axes — rather than by building new
> coding dimensions.
> Working title: *"Compositional learning by geometric editing: mPFC reuses a low-dimensional manifold
> and repositions the working-memory state to interleave memory and action."*
>
> **Flow fields / attractor-dynamics are OUT of the paper (2026-08-03) — kept as "extra"** (gallery pca/
> overlaps tabs), and Supp Fig 7 is dropped. **Fig 2 (final form 2026-08-10) = the MESSAGE figure
> "one dedicated axis per task variable — the working memory is a line"**: a schematic · b cvPCA spectrum ·
> c PR 1→2→3 (Naïve+Expert overlaid) · d η² PC-coding matrices — `fig_dimensionality_main.py`. The dPCA
> display panels (trajectories, mixing, linking plane) are ED Fig. 9; shattering is ED Fig. 3 / §3.
>
> Figure order: **1 Behaviour · 2 dPCA (low-D geometry) · 3 Manifold (abstract / reused geometry) ·
> 4 Overlaps (push / repositioning) · 5 Opto (causal).** The single overlaps figure was SPLIT
> (2026-08-04): geometry/abstraction → the new Fig 3 (`fig_overlaps_manifold.py`); push/repositioning →
> Fig 4 (`fig_overlaps_main_native.py`, trimmed to push panels). The old reserved "Fig 4 — low-rank
> circuit model" placeholder is dropped; its number is reused by the push figure. Publication-ready
> standard: **every panel is referenced and described.** Panel letters/stats verified against the RENDERED
> figures (Fig 1 `behavior_main.png`; Fig 2 `fig_dimensionality_main.png` — the dimensionality composite that REPLACED
`fig_dpca_story_main.png` on 2026-08-10; Fig 3 `fig_overlaps_manifold.png`;
> Fig 4 `fig_overlaps_main_ab_dpaact.png` — the CURRENT build, not the stale `fig_overlaps_main_ab.png`;
> Fig 5 `behavior_opto_main.png`). Caveats to carry are flagged inline as _(caveat: …)_; guardrails and
> to-reconcile items at the bottom.

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

## 2. The dual-task computation is low-dimensional and factorised (Fig. 2)

To see how mPFC arranges the memory and the action, we measured the dimensionality and composition of the
pseudo-population geometry directly, with cross-validated estimators that cannot inherit structure from
the analysis itself (task variables and the 12-condition space, Fig. 2a), comparing the pure memory task
(DPA) with the dual tasks at two states: mid-delay (post-distractor but before any cue or lick — clean
maintenance) and the post-test decision. Cross-validated PCA — the basis fit on one half of the trials,
variance evaluated on the independent held-out half, so only variance that *replicates* counts — shows
that the maintained memory state is **one-dimensional**: the DPA mid-delay reliable spectrum is a single
component (fraction 1.00, 95% CI [0.98, 1.00], jackknife across mice; Fig. 2b), the dual mid-delay adds
exactly one further reliable component (0.92 + 0.07 [0.01, 0.13]) — the distractor axis, whose
identity and reliability are carried by the decoding and factor tests (Fig. 2c,d) rather than by the
small variance fraction alone — and the
decision state spreads to ~3 components (DPA 0.66/0.17/0.17; dual 0.61/0.30/0.05). The working memory is
a line. Decoding makes the same point amplitude-free (Fig. 2c, balanced accuracy of held-out
pseudo-trials along each variable's demixed axis against a shuffle null): **a variable decodes only when
in play** — at mid-delay only sample (DPA 0.89; dual 0.81) and, in dual, the distractor (1.00) exceed
their nulls, with test and choice at chance until the test arrives, whereupon every variable decodes
(choice 0.96–0.97, test 0.74–0.77). And the reliable dimensions *are* the task variables (Fig. 2d, η² of
each condition-mean PC on the orthogonal factor contrasts, same mid-delay/decision windows): the memory
line is the sample axis
(η² = 0.93); the dual mid-delay holds one large distractor axis (η² = 0.98, 37% of condition-mean
variance) and the compact sample axis (0.91, 14%); the decision adds choice and test (0.92, 0.71; DPA
decision = choice, 0.99) — one dedicated axis per engaged variable, appearing exactly when its variable
comes online. The distractor code, moreover, barely enters the DPA geometry: Go-vs-NoGo cross-decoded from the
DPA-state subspace ("gng ×") reaches only 0.61 at mid-delay (Naïve n.s.; per-PC values in Fig. 2d) —
the memory geometry is close to orthogonal to the distractor code it must resist. The sample-memory axis
is thus compact but reliable — little variance, cleanly decoded — orthogonal to the larger
distractor/action dimensions. None of these measures changed with learning (Naïve ≈ Expert throughout
Fig. 2b–d, spectra, decodability and coding pattern alike; leave-one-mouse-out jackknife
Δ(Naïve − Expert) CIs span zero for every spectrum component and every decodable variable — Methods):
learning neither adds nor removes
representational dimensions. The one delay-period signal learning does remove is a premature choice
signal in the naïve dual-task delay — in naïve mice the upcoming choice is decodable from the delay
state well before the test (0.64–0.66 from early through late delay), whereas in trained mice it sits
at chance until the test arrives (Extended Data Fig. 3h): training holds the delay clean of the
decision. _(caveats: the DPA mid-delay 1-D is partly definitional — only one binary
variable is encoded during maintenance; variance-weighted summaries (participation ratio: memory 1.0
[1.0, 1.1] → full-state delay 2.0 [1.6, 2.5] → decision 3.3 [2.8, 3.8], and the full 12-condition
"all-tasks" spectra whose two large delay dimensions are context contrasts) are in Extended Data Fig. 3;
the time-resolved trajectory dimensionality is excluded because its shuffle null retains ~half the
variance via the condition-independent time ramp; the ~2-D figure is the state *geometry* — full
dynamics are higher-rank (~62–67% captured at rank 2, no elbow), so we describe a "rank-2 geometry",
not rank-2 dynamics.)_

The demixed (dPCA) axes show the same factorisation time-resolved (Extended Data Fig. 9): the single-axis
trajectories sharpened with learning without reorganising, and the one learning change was a refinement —
the choice and task/action axes became more aligned (|cos| 0.147 → 0.222, p < 0.001), binding the lick
decision to the action context, while the sample and test axes de-mixed (0.098 → 0.033, p = 0.008); the
scaffold is retained, its factors tuned. Notably, the demixed task/action axis orders the conditions by
their required action (DualGo/lick positive; DualNoGo and DPA/no-lick negative) — a pre-existing action
direction within the geometry (Extended Data Fig. 9). This near-orthogonal, factorised, low-dimensional geometry is the
representational substrate on which the two tasks are composed. We next asked whether this substrate is a
single *abstract* manifold the animal reuses across the two task contexts (Fig. 3), and then how learning
repositions the working-memory state on it to shield the memory from the intervening action (Fig. 4).
_(The dPCA population also shows the delay state deepening into no-lick with learning; because that is the
same phenomenon quantified single-trial and behaviourally in Fig. 4, we present it there, with the dPCA
corroboration — raw-ΔF/F, time-ramp-removal and pooled-basis robustness — in Extended Data. Earlier
flow-field / attractor-dynamics analyses of this deepening are retained as "extra", outside the paper.)_

## 3. The low-dimensional manifold is abstract and reused across tasks (Fig. 3)

A low-dimensional code is not automatically a reusable one. Having shown that the computation occupies only
~two demixed dimensions (Fig. 2), we asked whether those dimensions form a single abstract manifold shared
across the two task contexts — one the animal can reuse — rather than task-specific structure. Decoders
trained in one task generalised to the others (cross-task / off-diagonal decoding above chance, approaching
within-task / diagonal decoding; a single shared action axis), and — because the animals already know the
two component tasks (Naïve = the first three sessions) — this abstract, factorised format was already
present in naïve animals and preserved across learning. Establishing one reusable manifold, we then asked
how learning repositions the working-memory state on it (Fig. 4).

The manifold is inherited directly from the dPCA geometry. Re-plotting the dPCA sample × action plane,
sample A and B separated along a *single shared* sample axis, essentially identically across DPA, DualGo
and DualNoGo trials, orthogonal to the pre-existing task/action axis (Extended Data Fig. 9) — one memory dimension,
reused whichever action context it is embedded in. To confirm this at the single-trial, single-animal
level we trained cross-generalising decision-code decoders (CCGD): a balanced logistic decoder per mouse
and stage for each of the sample, GNG, test and choice(lick) variables, read across the whole trial
(Fig. 3a, naïve top / expert bottom). The four codes matched the dPCA account — the sample memory code was
stable and epoch-invariant, the GNG code diverged sharply around the cue, the test code appeared only at
test, and the choice/action code separated only at the response — four near-orthogonal, temporally stable
directions of one manifold.

That this manifold is *abstract*, not a collection of task-specific codes, was shown by cross-task
generalisation: decoders trained within one task and tested on another decoded above chance, the cross-task
(off-diagonal) balanced accuracy approaching the within-task (diagonal) values (Fig. 3b, Naïve and Expert)
— the geometry is shared across contexts rather than re-learned in each. Aggregated across the four codes,
every code read out above chance across the context that challenges it, with the choice and GNG action
codes the strongest cross-context generalisers (Fig. 3d). Critically, this abstraction was not built by
learning. Per-mouse cross-condition generalisation performance (CCGP) sat on the Naïve = Expert unity line
for the sample and choice codes (n.s.), with only the test code nudging up slightly (p ≈ 0.04) (Fig. 3e) —
the factorised, abstract format is present already in naïve animals and preserved, consistent with the two
component tasks being pre-learned before the dual composition. The same signature appears in the geometry's
functional expressivity: decoding all 462 balanced dichotomies of the 12 conditions (the shattering
dimension) gives 0.69–0.70 against a shuffle floor of 0.50 and an unstructured ceiling of 1 (Extended Data
Fig. 3) — high cross-condition generalisation with a moderate shattering dimension is precisely the
*abstract, compressed* regime, and it too is unchanged by learning (Δ = +0.01).

The action dimension in particular was a *single shared command*. Because the go/no-go decision and the DPA
match-lick are two expressions of the same lick action, we tested directly whether they share a neural axis
by cross-decoding between them — training a decoder on Go-vs-NoGo (at the distractor) and testing it on DPA
lick-vs-no-lick (at test), and vice versa; because train and test draw on different trials, tasks *and*
epochs, above-chance transfer cannot arise from leakage. In expert animals the two action codes cross-
decoded well above chance (Fig. 3c; cross-decoded balanced accuracy 0.60 and 0.84; off-/within-diagonal
ratio 0.56, 95% CI [0.36, 0.78]) — composing the tasks reuses one lick direction rather than building a
second. This shared axis was weaker and not clearly above chance in naïve animals (off/diagonal 0.32, CI
including zero), a positive but non-significant consolidation with learning (Δ = +0.24, p = 0.30). (This
generalisation-based test supersedes a raw weight-space cosine between the two action axes, which is only
~0.18 — statistically above its ±1/√N floor but numerically small, because independently fit high-
dimensional decoders share little raw direction even when they cross-decode; cross-decoding is the robust
measure of a shared axis.) What learning changes is therefore not the abstraction — largely inherited from
the pre-learned component tasks — but the state's *position* on the manifold, the repositioning we quantify
next (Fig. 4).

That the code is reusable and compositional was borne out at the single-neuron level: the near-
orthogonality did not arise from conjunctive mixed selectivity but from largely separate, independently
tuned populations — per-neuron permutation tests placed cross-variable co-tuning at chance (fraction tuned
to two variables ≈ product of the marginals), with more neurons selective for the action-relevant GNG
variable (≈ 39%) than for the purely mnemonic sample identity (≈ 10%) (Supplementary Fig. X). A factorised
substrate of this kind is what makes the two tasks composable without cross-talk.

## 4. Learning repositions the working-memory state on the manifold, and this predicts DPA accuracy (Fig. 4)

Having established one abstract manifold whose coding axes are fixed across learning, we asked how learning
*uses* it. The answer is a targeted geometric edit: within the fixed geometry, learning repositioned the
working-memory state exactly as the dPCA analysis predicted. The expert DPA delay state sat further into
the no-lick half of the action axis than the naïve state (Fig. 4a; sample × choice planes for naïve and
expert, and the per-mouse choice-code depth Naïve → Expert; mixed model 9 mice/36 obs, β = −0.744,
p = 0.046), specifically for sample A (A ≈ −1.45, p = 0.054; B ≈ −0.02, p = 0.95). And the size of this
geometric edit was behaviourally meaningful across animals: the more a mouse pushed its delay state toward
no-lick, the more its DPA accuracy improved (Fig. 4b left; between-mouse per-mouse Spearman ρ = −0.83,
p = 0.005, n = 9), with no relationship to Go/NoGo accuracy (Fig. 4b right; ρ = +0.20, p = 0.61) —
repositioning the state on the pre-existing manifold specifically buys memory performance. The coupling
held across every normalisation (ρ ≈ −0.83 to −0.90, all p ≤ 0.005) and a resampling battery (jackknife;
bootstrap CI excluding 0; permutation p = 0.008). Crucially this was a change in the state's *position*, not
in the code itself: the within-task decodability of the action code was statistically unchanged across
learning (Fig. 4d; action-code d′ Δ = +0.27, p = 0.25, n.s.) — the manifold's coding fidelity is fixed while
the state moves on it. As an internal within-trial validation that this axis is
the action code, on naïve error trials the choice code sat closer to lick before false alarms than before
correct rejections, though on the current build this did not reach significance (Fig. 4c; sample-A/AD
p = 0.204; sample-B/BC p = 0.833, n.s.). _(caveats: the push is directional rather than a precise magnitude
— on a fixed common axis it attenuates to a trend, part of the per-stage change being decoder-axis
reorganisation, so it is a re-sculpting of the state on a retained scaffold, not a rigid translation; the
behavioural coupling is an n = 9 individual-difference correlation, robust within the overlaps analysis but
null on the dPCA-derived depth (r = +0.46, p = 0.21); the false-alarm/correct-rejection panel (Fig. 4c) is
the weakest element and is currently n.s.)_

_(A mechanistic low-rank circuit model of this gated no-lick edit is deferred to future work; no modelling
figure is included here.)_

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
(Fig. 4), an edit that shields the memory and, animal by animal, predicts how well the two tasks are
composed (Fig. 4); and top-down ACC→mPFC input supplies this edit — setting the state's position on the
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

**ED Fig. 3 | Dimensionality: provenance & robustness (Fig. 2b–d).** (a0) the shattering dimension —
all 462 balanced dichotomies decoded at the decision window, 0.69 (Naïve) → 0.70 (Expert) vs shuffle 0.50
and ceiling 1 (cited from §3: abstract + compressed with the CCGP); (a1) the variance-weighted summary
of Fig. 2 — the previous PR build (`fig_dimensionality_main_pr.png`): full 12-condition "all-tasks"
spectra + the PR ladder memory 1.0 [1.0, 1.1] → delay 2.0 [1.6, 2.5] → decision 3.3 [2.8, 3.8]
(jackknife CIs); the full-state delay's two large dimensions are its context contrasts (distractor
presence and identity); (a) the descriptive dPCA scree
(top-2 ≈94%, PR≈2.2) with its circularity caveat — computed on 4 condition-means inside the demixed
sample/sample:test subspace, hence retired from the main figure in favour of the cross-validated
estimators; (b) per-marginal demixed variance (time 54% / tasks 31% / sample 7% / choice 7% / test 1%);
(c) reduced-rank test — held-out fit rises smoothly with no elbow at 2 (rank-2 = 62–67% of full), backing
the "rank-2 geometry, not rank-2 dynamics" caveat; (d) window robustness — on full-delay / test windows
the DPA-delay PR stays 1.0–1.1, delay 2.3–2.6, decision ≈2.5; (e) the FULL PC×factor η² grid behind Fig. 2d — Naïve & Expert × {dual, DPA} × {delay, decision}
(`plot_dimensionality_main.py`) — plus the per-task-set fits (DPA / dual / all, incl. the delay+dec
window and the 12-cond presence/identity task split); the DPA-delay condition-mean PCs beyond PC1 carry
apparent test/choice η² — variables undetermined at that point by design (the test is drawn independently
of the sample) and at chance in held-out decoding — i.e. sampling noise stripped by cvPCA (PR = 1), not
anticipatory coding: the noise-dimension gotcha flagged in Fig. 2d's footnote; (f) the shared-memory dPCA
d′ scatter (Naïve +0.61 → Expert +0.54, Δ = −0.07, p = 0.91 — memory code present in naïve and preserved);
(g) the Go/NoGo cross-decode from the DPA subspace, per window (main Fig. 2c/d shows the clean mid-delay
value 0.61; the late-delay ~0.7 figure is consummatory-inflated — the DPA geometry is close to, but not
fully, orthogonal to the distractor); (h) **learning removes the premature choice signal from the dual
delay** (`fig_bias_cleanup_ed.png`): in naïve mice the upcoming match/nonmatch choice is decodable from
the dual delay state from ED through LD (0.64–0.66 vs shuffle-null ≈0.59, demixed-axis held-out
decoding), and the held-out future-choice separation on a late-delay-defined — hence reward-free — axis
climbs to ~+2 z by LD; in Expert the same signal sits at chance throughout the delay (0.47–0.49) while
post-test decoding is intact (0.96). DPA shows no such signal at either stage (control). Decodability
already at ED (post-sample, pre-distractor) marks it as a trial-history/bias state rather than premature
deliberation. Caveats: on correct trials choice ≡ trial completion, so state-dependent selection
contributes to the naïve separation (the naïve/expert contrast may partly reflect stronger selection at
lower naïve accuracy); and at the mouse level the learning difference is not individually resolved
(Δ accuracy +0.19, leave-one-mouse-out jackknife CI [−0.09, +0.46], n = 9) — the effect is established
at the pooled-population level, where it replicates across three independent pipelines.

**ED Fig. 4 | dPCA no-lick push robustness (corroborates Fig. 4).** The Naïve→Expert deepening reproduces in raw ΔF/F
(r≈0.997, not a z-score artifact), survives condition-independent time-ramp removal (q0/1/2 =
−0.59/−0.60/−0.61), holds on a Naïve-defined pooled basis (8/9; bootstrap CI [−0.56,−0.08]), and is
population- not individual-level (depth↔accuracy null, r=+0.46, p=0.21).

**ED Fig. 5 | Overlaps: coupling/push robustness + movement control (Fig. 4a,b).** (a–b) the Δdepth↔ΔDPA
coupling is ★ under all six normalisations (ρ=−0.83 to −0.90) and survives a fixed common axis (ρ=−0.72)
where the push attenuates to a trend; (c) a resampling battery (Mundlak β=−0.041 p=0.006; jackknife 9/9;
bootstrap CI [−1.00,−0.26]; permutation p=0.008), ΔGNG null throughout; (d) movement control — late-delay
licking is rare, the choice-code depth does not track it (ρ=+0.07), and the push/coupling are unchanged
with a lick covariate.

**ED Fig. 6 | Overlaps: the factorised geometry is robust (Fig. 3b,c).** (a) cross-temporal cosine matrices
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

**ED Fig. 9 | dPCA demixed axes: trajectories, mixing, and the shared plane (Fig. 2/3).** The dPCA story
build (`fig_dpca_story_main.py`): demixing schematic + descriptive scree + marginal contrasts; the 2×4
Naïve/Expert trajectory grid (single-axis time courses sharpen without reorganising); the full pairwise
axis-mixing slopegraph (choice–task binds, 0.147→0.222 p<0.001; sample–test demixes, 0.098→0.033 p=0.008 —
neuron-bootstrap, not across-animal); the per-mouse shared-memory d′ scatter (Δ=−0.07, p=0.91, flat); and
the sample × action linking plane (one shared sample axis across DPA/Go/NoGo, ⊥ the pre-existing action
axis — the bridge cited in §3).

**Supplementary Information**
- **Trial counts per mouse** — per-mouse × stage × task counts entering the pseudo-population (balanced by
  design; 5,568 laser-OFF trials total). _(Analysis-balanced counts, not raw behavioural trial numbers —
  see Methods.)_

**Omitted:** retracted dPCA choice-polarization figures (`dpca_flow_learning_ingain*`,
`dpca_flow_autonomous_choice`, `dpca_choice_ci_qsweep`); the flow-field / bistability analysis (former S7 →
"extra"); the standalone d′ figure (former S17 → already main Fig. 5k,l); demixed-axes loadings/mixing
(former S5 → covered by Fig. 2g + ED 6). **Author-supplied gaps still needed:** histology / viral
expression, imaging FOV + per-mouse cell counts, laser-power / opsin titration.

---

### To reconcile before submission (figure ↔ text integrity)
- **Fig. 2 DECODE BUILD ADOPTED (2026-08-10, supersedes the PR build below):** panels B/C/D now share
  one grid, DPA vs dual × mid-delay vs decision. **b** cvPCA reliable spectra per set (leave-one-mouse-
  out jackknife 95% CIs; common 6-component axis), **c** per-variable decoding power (held-out
  pseudo-trials along each variable's demixed axis vs shuffle nulls, Kobak-style; incl. the hatched
  "gng ×" = Go/NoGo cross-decoded from the DPA-state subspace, 0.61 at mid-delay), **d** η² matrices
  DPA-first, PC1–4 in both sets, with the boxed gng × column on DPA. Mid-delay (bins 36–38,
  pre-cue/pre-lick) replaces late delay in b/c; d stays late-delay (footnoted). The PR bars +
  all-tasks spectra moved to **ED 3(a1)** and render via `fig_dimensionality_main.py --pr` →
  `fig_dimensionality_main_pr.png`. §2 rewritten accordingly (spectra + decoding numbers headline;
  PR quoted in the caveat with ED pointer). Metric rationale (PR variance-weighting challenged →
  decoding adopted; dot-strip variant rejected) and all numbers are logged in memory
  (`project_dimensionality`).
- **Fig. 2 REPLACED (2026-08-10, message-first FINAL):** the dPCA story build (`fig_dpca_story_main.py`)
  is superseded by `pca/fig_dimensionality_main.py` → `figures/pseudo/dimensionality/
  fig_dimensionality_main.png`, built around ONE message ("one dedicated axis per task variable — the
  working memory is a line"): **a** trial-timeline + split-half cvPCA schematic, **b** cvPCA reliable
  spectrum (delay, Naïve/Expert + shuffled null), **c** PR bars with split-level CIs, Naïve+Expert
  overlaid (memory ≈1 · delay ≈2 · decision ≈3.3), **d** the η² PC-coding matrices, Expert row of four
  (dual-delay · dual-decision · DPA-delay · DPA-decision). Everything off-message moved out: shattering →
  ED 3(a0) + cited from §3; dPCA trajectory grid, axis-mixing, linking plane, shared-memory scatter →
  **ED 9** (= `fig_dpca_story_main.png`, kept rendering); old scree + marginal variance + Naïve η²
  matrices + per-task-set fits → ED 3. Design iterations that got here (matrices are the real-data
  panel; a derived chips graphic was rejected) are logged in memory. Data:
  `figures/pseudo/dimensionality/results.pkl` (merged caches; CIs from `exp_dimensionality_ci.py`).
  Earlier "Fig. 2 flow-free — DONE" and "Fig 3 panels updated" notes below describe superseded builds.
- **Overlaps split (2026-08-04; Fig 3 panels updated 2026-08-05):** the single overlaps figure was split
  into **Fig. 3** (manifold / abstraction, `fig_overlaps_manifold.py` → `fig_overlaps_manifold.png`,
  panels a–e = code traces / within-vs-cross-task matrix / **shared action axis (Go/NoGo↔DPA-lick cross-
  decode)** / **cross-context generalisation summary** / CCGP-across-learning; the dPCA linking plane moved
  to Fig 2g and the weak weight-cosine panel was dropped) and **Fig. 4** (push / repositioning,
  `fig_overlaps_main_native.py` → `fig_overlaps_main_ab_dpaact.png`, push panels a–d = push planes+depth /
  Δdepth↔Δacc coupling / Naïve FA-CR / action-code d′). A stale `fig_overlaps_main_ab.png`
  (4-panel, 18-obs LMM coupling) still exists in the repo — do NOT ship it; use only the current builds
  (the gallery Main tab was repointed to the `_dpaact` file).
- **Fig. 2 flow-free — DONE (2026-08-03):** `fig_dpca_story_main.py` regenerated without the section-3 flow
  grid or the section-4 flow panels → 7-panel **a** schematic, **b** scree, **c** per-task variance, **d**
  trajectory grid, **e** axis-mixing, **f** no-lick push (p=0.012), **g** sample-memory preserved (p=0.10);
  title "The dPCA **geometry**…". Push stats reproduce (−0.59, 8/9). The Main-tab `fig_dpca_story_main.png`
  now shows the flow-free build.
- **Titles left-aligned — DONE (2026-08-03):** the subplot titles in both new supplements were already
  `loc='left'`; the coupling-battery's one centered element (a full-width suptitle banner) resisted
  left-alignment under the tight-bbox save and was removed (its message lives in the two left panel titles
  + the ED 5 legend). d′ figure unchanged (no centered element).

### Open drafting decisions
- **Fig. 1h framing:** "decoupled / suboptimal balance" (defensible, n.s. correlation) vs a stronger
  "trade-off" (not supported at n = 9). Draft uses the conservative version.
- **Fig. 2 sample-coding wording:** "preserved" (correct-trials, p = 0.10) vs "preserved and sharpens"
  (all-trials, p = 0.02). Draft uses "preserved".
- **"reuse the same manifold"** is carried by the near-orthogonal, temporally stable axes present in naïve
  + the pre-existing no-lick well; phrased as "factorised scaffold retained; state re-sculpted", NOT
  "identical manifold, pure translation" (decoder-axis reorganisation is part of the push).
- **Supplementary Fig. X** (single-neuron selectivity) number to be assigned.
- **Fig. 4/5 numbering (resolved 2026-08-04):** the deferred low-rank circuit model is cut; Fig. 4 is now
  the overlaps push/repositioning figure and the opto figure stays Fig. 5. If a modelling figure is added
  later it takes a new number rather than reclaiming 4.

### Framing guardrails (keep the thesis defensible)
- "low-dimensional **geometry** / rank-2 portrait", never "the dynamics are rank-2". Flow-field /
  attractor-dynamics claims are OUT of the paper (kept as "extra") — don't reintroduce "landscape",
  "gated deformation", "bistable" into the main text.
- near-orthogonality = the representational basis of compositionality (factorised code → combine without
  cross-talk) — the load-bearing, solid claim.
- the pre-existing naïve no-lick well is the strongest "structure already there, fine-tuned" evidence —
  lean on it for the reuse claim.
