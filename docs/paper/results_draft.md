# Compositional learning by geometric editing — main paper (draft v7)

> **v7 (2026-09-03): NATURE-NEUROSCIENCE REGISTER & STRUCTURE PASS** (user: "make the draft more
> nature neuro oriented... check the references for inspiration" / "I meant the writing style and
> structure"). Modelled on the cited NatNeuro corpus (Golub 2018; Libby & Buschman 2021;
> Parthasarathy 2017; Kaufman 2014; Driscoll 2024). Changes: (1) abstract rewritten to one
> ~175-word paragraph (setup → "Here we show" → findings → significance; keeps n = 9 / n = 5 /
> 3,319 / ρ = −0.83 per the v6.1 transparency decision); (2) section headings now SHORT,
> UNNUMBERED, DECLARATIVE, without "(Fig. N)"; (3) Results prose rebuilt sentence-level — question
> → result (stat in parentheses) → conclusion; past tense for observations; meta-commentary
> ("the panels build the claim", "not merely X but Y", "the honest boundary") removed; aphorisms
> sobered; (4) every inline "_(caveat: …)_" block ABSORBED into measured prose sentences at
> paragraph ends (content preserved — nothing dropped, verify against v6.1 in git); (5) the
> Synthesis section DELETED — its job is the Discussion opener (which now carries "geometric
> editing"); (6) Discussion bold paragraph labels stripped (continuous prose). **ALL statistics
> verbatim from v6.1 — no number changed.** Old section names for cross-reference: §1 cost →
> "The dual-task cost arises from the intruding action"; §2 low-D → "One coding dimension per
> task variable, shared across tasks"; §3 plane → "A single sample × choice plane is necessary,
> sufficient and stable"; §4 learning → "Learning repositions the memory state along the choice
> axis"; §5 opto → "ACC input shifts the state's position but not the code".

> **v6.1 (2026-09-01): CODEX-REVIEW CALIBRATION PASS** (adjudicated — see the to-reconcile log).
> Claims recalibrated to evidence (no *detectable* added dimensions; predicts/associated-with for
> Fig 4; shifts + is-required-during-learning for Fig 6; scaffold-vs-refinement sentence in §2);
> §5 emphasis flipped to the mouse-clustered ΔGNG arm; abstract carries n = 9 and n = 5. TWO NEW
> STATISTICS strengthen contested claims rather than softening them: Fig 1g's lick × pairing
> interaction (p = .61 — the propagation is pairing-independent, so the pooled OR = 3.10 covers
> the false-alarm arm; now printed by the script and in the caption) and the plane-vs-full
> BOUNDED EQUIVALENCE (|Δ| ≤ 0.012 accuracy in every mouse × stage; sample ≤ 0.003 — §3 keeps
> "sufficient" with formal backing). ED count fixed 8→9. Rejected: days-4–6 wording (day counts
> vary), push-star re-litigation (settled policy), Fig 5 placeholder removal (model in prep).

> **v6 (2026-09-01): HIGH-IMPACT REWRITE.** Abstract added; Introduction rebuilt (stakes →
> construction-vs-editing dichotomy → task lineage → here-we-show); every Results section re-led
> with a claim-first topic sentence; field vocabulary applied throughout (fixed shared *subspace*,
> output-suppressing *set-point*, *reassociation*, *code morphing* — per memory
> `reference_literature_positioning`); the internal note-to-self phrasings of v5 rewritten as
> manuscript prose; the single-neuron passage repointed from "Supplementary Fig. X" to Fig. 2g
> (the biplot is now a main panel). **ALL statistics are carried over verbatim from v5** (verified
> against the rendered builds; no number changed). Discussion is v2 in `discussion_draft.md`.
> Citations are [Author Year] placeholders for the reference manager.

> Methods paragraphs are staged per-figure in **`methods_notes.md`** (started 2026-08-10 with the Fig 2
> dimensionality block: cvPCA incl. the repeated-2-fold-vs-k-fold justification, jackknife CIs, η²
> decomposition, shattering).

> **Thesis:** cortical computation is low-dimensional and carried by the *geometry* of population
> activity; the brain composes a new task by **editing that geometry** — reusing a pre-existing neural
> subspace and repositioning the memory state along its existing axes — rather than by building new
> coding dimensions.
> Working title (NatNeuro declarative style, v7): *"Learning repositions working memory within a
> fixed prefrontal population geometry."* Alternatives kept: *"Prefrontal cortex learns a dual task
> by repositioning states in a fixed population subspace"*; *"A fixed population geometry constrains
> and enables dual-task learning"*. (The v6 colon title is retired — the cited NatNeuro corpus
> [Golub 2018; Libby 2021; Parthasarathy 2017; Driscoll 2024] titles are single declarative claims.)
>
> Figure order: **1 Behaviour · 2 Geometry (low-D, factorised, shared) · 3 One subspace (the
> sample × choice plane: sufficiency + stability) · 4 Learning (alignment + push + coupling) ·
> 5 Modelling (TBD) · 6 Opto (causal).** Publication-ready standard: **every panel is referenced and
> described.** Panel letters/stats verified against the RENDERED figures (Fig 1 `behavior_main.png`;
> Fig 2 `fig_dimensionality_main.png`; Fig 3 `fig_manifold_main.png` — canonical no-PCA build;
> Fig 4 `fig_overlaps_main_ab_dpaact.png`; Fig 6 `behavior_opto_main.png`). Caveats to carry are
> flagged inline as _(caveat: …)_; guardrails and to-reconcile items at the bottom.

---

## Abstract

Behaviour often requires doing two things at once: holding a memory while acting on an
unrelated cue. The evoked action risks capturing the response that the memory should control,
yet with practice both computations run through the same cortex without interference. Here we
show how mouse medial prefrontal cortex learns this composition. We imaged 3,319 prelimbic
neurons in nine mice learning an olfactory working-memory task with an embedded Go/NoGo
distractor. The maintained memory occupied a single dimension, held nearly orthogonal to the
axes coding the distractor and the lick decision, within a low-dimensional subspace whose axes
were shared across task variants and present before learning. Learning added no detectable
coding dimensions. Instead, the distractor code rotated onto the choice axis, and the memory
state was repositioned along that axis toward an output-suppressing, no-lick set-point; the
deeper an animal pushed its memory state, the more its memory improved (ρ = −0.83). Anterior
cingulate input was required for learning the composition, and silencing it acutely (five mice)
shifted the state's position without degrading the code. Learning and top-down control thus
operate on state positions within a fixed population geometry.

---

Doing two things at once is a signature of flexible behaviour and a hard computational problem.
When an action must be selected and executed in the middle of a memory delay, the network that
holds the memory is also the network being driven to act, and the intruding activity can corrupt
the stored content — a cost documented from human dual-task interference to the degradation of
prefrontal memory codes by distractors [Parthasarathy 2017; Libby 2021]. Composing two behaviours
into one is therefore not simply a matter of learning each: the brain must arrange for two
computations to share a circuit without overwriting one another.

Population recordings suggest where such an arrangement could live. Cortical computation is
increasingly understood through the low-dimensional geometry of population activity — the
manifolds, subspaces and coding axes along which neural states move [Mante 2013; Kaufman 2014;
Vyas 2020] — and the format of that geometry determines what a downstream reader can do with it:
abstract, factorised representations, with one near-orthogonal axis per variable, generalise to
new conditions and let variables be read and combined without interference [Rigotti 2013;
Bernardi 2020]. On this view there are two qualitatively different ways to acquire a composite
task. The cortex could *construct*: add new coding dimensions, or a new configuration of them, to
accommodate the second task — the analogue of outside-manifold learning, which is slow and
effortful in brain–computer-interface (BCI) experiments [Oby 2019]. Or it could *edit*: keep the
existing geometry fixed and learn only where states sit within it — the analogue of within-manifold
reassociation, the fast and natural mode of BCI learning [Sadtler 2014; Golub 2018], and the mode
predicted if the geometry is shared infrastructure for a family of tasks [Yang 2019;
Driscoll 2024]. Which of these describes the natural acquisition of a composite task by a cortical
population is unknown.

We addressed this in a task built to force the composition. Head-fixed mice (n = 9) learned a
delayed paired-association (DPA) working-memory task — a sample odour (A or B) matched, after a
6-s delay, to a test odour (C or D), with a lick-for-reward response — in which, on dual-task
sessions, a Go/NoGo (GNG) discrimination is embedded in the middle of the delay (DualGo /
DualNoGo trials, interleaved with distractor-free DPA trials). The same delay period thus
alternately does and does not demand an intervening action, making the dual task an explicit
composition of two computations that must run concurrently without one corrupting the other.
mPFC delay activity is required for learning exactly this class of olfactory working-memory task
[Liu 2014]. Mice progressed through a fixed curriculum (DPA → GNG → Dual), and we imaged
prelimbic mPFC with two-photon microscopy at both ends of learning, yielding a pseudo-population
of 3,319 neurons spanning naïve and expert stages. Throughout, we measured the population
geometry with cross-validated measures that count only structure replicating across independent
trial halves, paired every pooled claim with a per-animal (n = 9) companion statistic, and closed
with a causal test of the circuit input that supplies the learned change.

Here we show that mPFC does not construct a new representation for the dual task. The memory and
the action are carried on near-orthogonal axes of a low-dimensional subspace that is already
present, with the same axes, in naïve animals. Learning leaves this factorised scaffold in place,
refining axis alignments without adding detectable dimensions, and repositions the working-memory
state along a pre-existing action axis into its output-suppressing, no-lick half. This edit
tracks the removal of the lick interference, predicts memory performance animal by animal, and
its position is shifted by top-down input from the anterior cingulate cortex (ACC), a projection
required while the composition is learned. The code's geometry is a fixed constraint; learning
and top-down input both act on the state's position within it.

## The dual-task cost arises from the intruding action

Mice learned the two component tasks under a fixed curriculum (Fig. 1a). Across six dual-task
sessions, DPA and GNG accuracy rose and converged (Fig. 1b; mixed-effects model, GNG−DPA
condition β = +0.037, p = 0.045; condition × day β = −0.039, p = 8 × 10⁻⁴). The GNG gain was
carried by the NoGo condition — learning to withhold rather than to lick (Fig. 1c; NoGo−Go
β = +0.072, p = 0.034) — and the DPA gain by the unpaired trials, which began far below the
paired trials and improved fastest (Fig. 1d; unpaired−paired β = −0.185, p < 10⁻⁴;
condition × day β = +0.088, p < 10⁻⁴; the full set of effect sizes is summarised in Fig. 1f).

The cost of composition had a specific signature. Unpaired DPA accuracy was selectively
depressed when the distractor required a Go response (Fig. 1e; Go−DPA β = −0.073, p = 0.038) but
not a NoGo response (NoGo−DPA p = 0.78), indicating that the intrusive lick, rather than the
distractor stimulus, was the source of interference. The interference propagated through the
response itself: a naïve animal that intruded a lick at the NoGo distractor cue was three times
as likely to lick again at the test (Fig. 1g; trial-level regression, odds ratio OR = 3.10,
p = 0.006). This propagation was pairing-independent (lick × pairing interaction p = 0.61),
applying equally to unpaired trials, where the second lick is by definition the false alarm
(direct unpaired estimate OR = 2.7, p = 0.09), and to paired trials, where it lands on a hit
(OR = 9.9, p = 0.001) — response propagation rather than memory loss. In expert animals the
propagation was gone (OR = 1.50, p = 0.42) and the intrusive licks themselves largely
disappeared (cue-lick rate 0.24 → 0.08): learning removed both the intruding action and its grip
on the subsequent response. Because this is a within-trial association, Fig. 1g establishes the
route of the interference, not its cause.

Even at expertise the composition remained imperfect. Across animals, expert DPA and GNG
performance were decoupled (Fig. 1h; Pearson r = +0.10, p = 0.80; Spearman ρ = +0.35, p = 0.36;
n = 9), with no animal reaching the both-tasks-optimal corner (mean shortfall ≈ 0.18; mean DPA
0.88, GNG 0.87), whereas in naïve mice the two accuracies still co-varied (r ≈ 0.66–0.74). With
nine animals this supports decoupling — the absence of a joint optimum — rather than a
significant trade-off. The behavioural problem is therefore precise: the memory must be shielded
from the very action the animal is simultaneously required to produce. We next asked how the
population solves it.

## One coding dimension per task variable, shared across tasks

We measured the dimensionality and composition of the pseudo-population geometry with
cross-validated measures that cannot inherit structure from the analysis itself (task variables
and the 12-condition space, Fig. 2a), comparing the pure memory task with the dual tasks at two
states: mid-delay (after the distractor but before any cue or lick) and the post-test decision.
Cross-validated PCA, in which the basis is fit on one half of the trials and variance is
evaluated on the other, counts only variance that replicates. By this measure the maintained
memory state was one-dimensional: the DPA mid-delay reliable spectrum comprised a single
component (fraction 1.00, 95% CI [0.98, 1.00], jackknife across mice; Fig. 2b). The dual
mid-delay added exactly one further reliable component (0.92 + 0.07 [0.01, 0.13]) — the
distractor axis, whose identity is established by the decoding and factor tests below rather
than by the small variance fraction alone — and the decision state spread to approximately three
components (DPA 0.66/0.17/0.17; dual 0.61/0.30/0.05). The same picture held animal by animal on
each mouse's own simultaneously recorded population: wherever the per-mouse reliable variance
was resolvable, the mid-delay spectrum was dominated by a single component (per-mouse cvPCA,
median top-1 fraction 0.90 naïve / 0.93 expert, n = 7 resolvable mice per stage), and the
within-mouse memory-versus-decision contrast spread as predicted (expert 0.93 vs 0.61, Wilcoxon
p = 0.047, 6/7 mice; naïve directional, p = 0.22) (Extended Data Fig. 3c).

Each reliable dimension corresponded to a task variable, engaged exactly when that variable was
in play. Decoding made this point independently of signal size (Fig. 2c, balanced decoding
accuracy of withheld pseudo-trials along each variable's demixed axis, against a shuffle null):
at mid-delay only sample (DPA 0.89; dual 0.81) and, in the dual tasks, the distractor (1.00)
exceeded their nulls, with test and choice at chance until the test arrived, whereupon every
variable decoded (choice 0.96–0.97, test 0.74–0.77). And the reliable dimensions were the task
variables themselves (Fig. 2d, η² of each condition-mean PC on the orthogonal factor contrasts):
the memory line was the sample axis (η² = 0.93); the dual mid-delay held one large distractor
axis (0.98, 37% of condition-mean variance) and the compact sample axis (0.91, 14%); and the
decision state added choice and test axes (0.92, 0.71; DPA decision = choice, 0.99). The
distractor code barely entered the memory geometry: Go-versus-NoGo read out from the DPA-state
subspace reached only 0.61 at mid-delay (naïve n.s.; per-PC values in Fig. 2d). The
sample-memory axis was thus compact but reliable, and close to orthogonal to the larger
distractor and action dimensions it must resist.

None of these measures changed detectably with learning: naïve and expert spectra, decodability
and coding patterns were near-identical (Fig. 2b–d), and leave-one-mouse-out jackknife CIs on
the naïve−expert difference spanned zero for every spectrum component and every decodable
variable (Methods). We find no evidence that learning adds or removes representational
dimensions. The claim throughout is a stable scaffold — the dimensions and their identities —
while learning tunes alignments within it (Fig. 4).

The memory and choice axes were, moreover, shared across the three tasks, and the sharing
preceded learning. Decoders trained on one task transferred most of their decodable signal to
the others (Fig. 2e; cells show the transferred fraction (cross − 0.5)/(within − 0.5); the test
code's matrix has weak within-task ceilings at this window and bounds the claim). Per mouse,
cross-task accuracy sat on the naïve = expert unity line for every code, with the 95% CIs on the
change confined to ±0.05 accuracy (Fig. 2f; sample [−.03, +.02]; test [−.01, +.04]; choice
[−.03, +.05]; Wilcoxon n = 9, robust across decoder variants). At the single-neuron level this
abstract, factorised format was built from largely separate, independently tuned populations
rather than conjunctive mixed selectivity: per-neuron sample and decision d′ were uncorrelated
(r = −0.03), and the fraction of double-selective neurons matched the independence expectation
(6.2% vs 6.4%; Fig. 2g; per-neuron permutation tuning in Extended Data Fig. 6b). A factorised,
shared subspace of this kind is the geometry a memory needs to survive an embedded action task —
variables coded on separate axes cannot overwrite one another — and it was in place before the
composition was learned.

The one delay-period signal that learning did remove was a premature choice signal in the naïve
dual-task delay: in naïve mice the upcoming choice was decodable from the delay state well
before the test (0.64–0.66 from early through late delay), whereas in trained mice it sat at
chance until the test arrived (Extended Data Fig. 3g) — training holds the delay clean of the
decision. The demixed (dPCA) axes showed the same factorisation time-resolved (Extended Data
Fig. 9): single-axis trajectories sharpened with learning without reorganising, the choice and
task/action axes became more aligned (|cos| 0.147 → 0.222, p < 0.001) while the sample and test
axes de-mixed (0.098 → 0.033, p = 0.008), and the demixed task/action axis ordered the
conditions by their required action — a pre-existing action direction within the geometry. Three
qualifications bound this section. The DPA mid-delay one-dimensionality is partly definitional,
since only one binary variable is encoded during maintenance. Variance-weighted summaries
(participation ratio: memory 1.0 [1.0, 1.1] → full-state delay 2.0 [1.6, 2.5] → decision 3.3
[2.8, 3.8], and the full 12-condition spectra, whose two large delay dimensions are context
contrasts) are given in Extended Data Fig. 3; time-resolved trajectory dimensionality is
excluded because its shuffle null retains the condition-independent time ramp. And the low
dimensionality describes the state geometry, not the dynamics, which are higher-rank (62–67%
captured at rank 2, with no elbow).

## A single sample × choice plane is necessary, sufficient and stable

A low-dimensional, shared code is not automatically a single reusable structure. We therefore
asked whether the memory and the action live in one two-dimensional subspace — the
sample × choice plane; whether that plane is sufficient and necessary for the codes it should
carry; and whether it is the same plane before and after learning.

We first read the two frame axes in each task separately (Fig. 3a; per-mouse cross-validated
decoder projections, baseline-zeroed, in one shared per-mouse unit). The DPA sample code was
maintained across the entire delay, whereas on dual trials the same readout decayed after the
distractor arrived (lower in 9/9 naïve and 8/9 expert mice) — the code-morphing signature
described in primate PFC after a distractor [Parthasarathy 2017], read here on a fixed axis;
whether the memory itself survives, or only this readout, is answered by the cross-task transfer
of Fig. 2e. On the choice axis, the Go trace rose at the cue in both trial classes — every
correct Go trial licks the cue, a motor/reward transient rather than choice coding — and the
lick/no-lick split opened only at the test; the expert NoGo trace ran below baseline through the
late delay (7/9 mice), consistent with active withholding. Snapshots of the same projections at
three trial moments (Fig. 3b; each window re-centred per mouse on its mean state, so the panels
show condition geometry rather than absolute position) showed every separation, in every task,
at every moment, falling along the same two axes.

To test whether the plane is genuinely the code's home — and not an artefact of plotting in
it — we decoded each variable from only the two coordinates of each mouse's sample × choice
plane, from the residual population left when the plane was removed, and from the full
population (trials withheld from the fit; paired Wilcoxons, n = 9; Fig. 3c,d). The result was a
double dissociation. For sample and choice, the two-dimensional plane did as well as the whole
population — agreeing to within 0.012 accuracy in every mouse and stage (sample within
0.003) — and removing it collapsed decoding (p = .004; the collapse is expected by construction,
the informative half of the dissociation being the plane–full equivalence). The test code was
the mirror image — at chance from the plane, untouched without it (p = .004): that code lives
elsewhere. The distractor sat in between, its plane share real but partial (p = .004). The
dissociation held animal by animal (Fig. 3d) and carried the section's one learning effect: the
distractor's plane-only accuracy grew with learning (0.57 → 0.65, p = .020/.027 across both
decoder variants, 8/9 and 7/9 mice) — per animal, learning pulled the distractor code into the
plane (quantified further in Fig. 4a). The axes themselves were near-orthogonal — the sample
axis at |cos| ≈ 0.07–0.09 to both action codes, with a partial, growing choice × distractor
overlap (0.32 → 0.47; Fig. 3e, attenuation-corrected with disclosed split-half reliabilities;
the per-animal test is in Fig. 4a).

Finally, it was the same plane across learning (Fig. 3f): decoders trained in one stage read the
other stage's withheld activity at ~90% of the within-stage ceiling (transfer/within 0.90 for
sample, 0.87 for choice; robust across decoder variants and to scoring both stages in one common
unit scaling), and the same transfer held inside each animal. In the vocabulary of the
brain–computer-interface learning literature [Sadtler 2014; Golub 2018], what follows is
within-manifold learning: the subspace is a fixed constraint, and learning reassociates states
within it rather than building new dimensions. The abstraction of this format — its
cross-condition generalisation [Bernardi 2020] — was present in naïve animals and preserved
across learning: per-mouse cross-condition generalisation performance (CCGP) sat on the
naïve = expert unity line for the sample and choice codes, with only the test code nudging up
(significant under one decoder variant but not the other, p = .04/.73 — reported without a
verdict) (Extended Data). Decoding all 462 balanced dichotomies of the 12 conditions (the
shattering dimension) gave 0.69–0.70 against a shuffle floor of 0.50 and an unstructured ceiling
of 1 (Extended Data Fig. 3), and was likewise unchanged by learning (Δ = +0.01) — high
generalisation with moderate shattering, the abstract, compressed regime [Bernardi 2020].

## Learning repositions the memory state along the choice axis

Having established one fixed shared subspace, we asked what learning changes within it. Two
things changed, and they are two faces of a single reorganisation of the action dimension.
First, the distractor code aligned onto the choice axis (Fig. 4a): cross-decoding between the
two codes rose from chance-referenced transfer 0.33 [−0.03, 0.60] in naïve to 0.57 [0.36, 0.75]
in expert animals, and per animal both the raw axis cosine (0.073 → 0.114, p = .008) and the
cross-decode (0.53 → 0.61, p = .004) increased, robustly across decoder variants. The
distractor's demand became readable as what it is behaviourally: a lick decision.

Second, learning repositioned the working-memory state along that same axis. The expert DPA
delay state sat further into the no-lick half of the choice axis than the naïve state (Fig. 4b;
mixed model, 9 mice/36 observations, β = −0.744, p = 0.046; per-animal Wilcoxon p = 0.098) — a
set-point shift that moves the delay state away from the lick boundary, suppressing precisely
the intrusive-lick → false-alarm chain that learning removes behaviourally (Fig. 1g). The shift
was numerically larger for sample A than for sample B (ΔA ≈ −1.42, p = 0.098; ΔB ≈ −0.07,
p = 0.91), but the direct paired comparison across the same nine mice was not significant in any
of six axis × normalisation builds (p = 0.055–0.203), and the difference was not a decoder
artefact: the sample and choice axes are orthogonal (per-mouse |cos| = 0.04), and axis leakage
would displace A and B in opposite directions, which only 3/9 mice showed. We therefore report
the repositioning itself and make no claim of sample specificity.

The size of the repositioning predicted behaviour across animals: the more a mouse pushed its
delay state toward no-lick, the more its DPA accuracy improved (Fig. 4c left; per-mouse Spearman
ρ = −0.83, p = 0.005, n = 9), with no relationship to Go/NoGo accuracy (Fig. 4c right;
ρ = +0.20, p = 0.61) — an association specific to memory performance, at no detectable
distractor-task cost. The coupling held across every normalisation (ρ = −0.83 to −0.90, all
p ≤ 0.005) and every resampling check (leave-one-out jackknife; bootstrap CI excluding 0;
permutation p = 0.008). Two controls close the section. Within naïve nonpaired trials,
trial-by-trial state depth did not separate correct rejections from false alarms on this axis
(Fig. 4d; sample-A Δ(CR−FA) = −1.16, p = 0.27; sample-B +0.73, p = 0.47): the push is a
between-animal learning effect, not a within-stage trial-level accuracy readout. And the change
was in the state's position, not the code itself: the choice code's discriminability was
statistically unchanged across learning (Fig. 4e; d′ 0.80 → 1.07, Δ = +0.27, p = 0.25). The push
is directional rather than a precise magnitude — on a fixed common axis it attenuates to a
trend, part of the per-stage change being decoder-axis reorganisation — and the behavioural
coupling is an n = 9 individual-difference correlation, robust within this analysis but null on
the dPCA-derived depth (r = +0.46, p = 0.21). (A circuit model of this gated no-lick edit is in
preparation as Fig. 5.)

## ACC input shifts the state's position but not the code

If composition is implemented as an edit to this geometry, some input must supply the edit. We
tested the projection from ACC to the recorded prelimbic mPFC (Fig. 6a; hSyn-GCaMP6s in mPFC,
CaMKII-Jaws-tdTomato in ACC, 635 nm on a pseudo-random 50% of delay periods). Chronic,
every-trial silencing during training (between-group, 9 opto vs 9 control mice, a separate
cohort) impaired DPA learning (Fig. 6b), an effect localised to DPA and most strongly its
unpaired trials, sparing Go/NoGo (Fig. 6c; DPA β = −0.06, p = 0.009; DPA-unpaired β = −0.12,
p = 0.014; GNG n.s.) — the same DPA-selective vulnerability seen behaviourally in Fig. 1, and
the same learning-phase dependence reported for mPFC delay activity itself [Liu 2014].

In the imaged cohort we silenced ACC→mPFC transiently, on half of the delay periods
(within-mouse laser ON vs OFF, Jaws, n = 5), and projected both trial types through the fixed,
laser-OFF-trained choice axis to read the state's position on the learned geometry. Transient
silencing produced no gross behavioural change (Fig. 6d,e; DPA p = 0.40, GNG p = 0.24) but moved
the delay choice code per mouse (Fig. 6f). Across animals, the displacement's robust behavioural
coupling was on the distractor side: Δdepth predicted ΔGNG accuracy (Fig. 6i; r = −0.65,
p = 0.002; ρ = −0.62, p = 0.003 — the one arm that survives a mouse-clustered model, β = −0.013,
p = 0.018), whereas the joint DPA−GNG trade-off was a raw-level trend only (Fig. 6g; r = +0.53,
p = 0.016 over n = 20 points that cluster within five mice; clustered p = 0.108) and the DPA arm
was not significant (Fig. 6h). In laser-ON trials the mice still occupied the same suboptimal
DPA–GNG balance as at baseline (Fig. 6j; r = +0.44, p = 0.20).

Critically, the input moved the state without altering the code's content: the discriminability
of both the sample-memory axis (Fig. 6k; A vs B, late delay) and the Go/NoGo choice axis
(Fig. 6l; Go vs NoGo, mid-delay) was spared under laser (LMM over 20 mouse × stage × laser
observations from the five mice: sample p = 0.34, GNG p = 0.74). Acute ACC→mPFC input therefore
shifts where the delay state sits along the shared action axis — the same variable that learning
acts on — while leaving the coded content intact; and the chronic experiment shows the
projection is required for the composition to be learned (Fig. 6b,c). The mirror with learning
is informative: learning's repositioning couples to the DPA arm with no GNG cost (Fig. 4c),
whereas the acute displacement couples to the GNG arm — the fixed subspace permits a factorised,
memory-specific improvement that a momentary perturbation does not reproduce. Because this
figure combines a chronic between-group cohort with a within-mouse transient cohort, and its
n = 20 trial-type points cluster within five animals, we frame Fig. 6i as the robust arm and
Fig. 6g as a raw-level trend (Methods); an alternative build of the coupling over all seven
laser mice gives GNG ρ = −0.90, p = 0.006, with the DPA arm null (Extended Data Fig. 8).

---

## Methods

> Assembled 2026-09-01 and **audited against the producing code the same day** (three-agent
> code audit; every parameter below carries file-level evidence — discrepancies found in the
> first assembly are fixed here and listed under "To reconcile"). Author-supplied experimental
> details still needed are marked **[AUTHOR: …]**.

### Animals and behavioural task

Nine adult mice were used for imaging (five expressing Jaws for ACC→mPFC silencing, two ChR2, two
with ACC-targeted controls); a separate behavioural cohort (9 opto vs 9 control) was used for the
chronic-silencing training experiment (Fig. 6b,c). **[AUTHOR: strain, sex, age, housing, water
restriction, licence/ethics statement.]** Mice learned a delayed paired-association (DPA) task:
a sample odour (A or B, 2–3 s) followed after a 6-s delay by a test odour (C or D, 9–10 s), with
a lick response in the 10–11-s window rewarded on matching sample–test pairs and unrewarded
(false alarm) otherwise. On dual-task trials a Go/NoGo (GNG) discrimination was embedded in the
delay: a distractor odour at 4.5–5.5 s, a response cue at 6.5–7.0 s, lick-for-reward on Go.
DualGo, DualNoGo and distractor-free DPA trials were interleaved within sessions. Mice progressed
through a fixed curriculum (DPA → GNG → Dual, six dual-task sessions); days 1–3 are analysed as
"naïve" and day 4 to the last day as "expert" (mice contribute 4–6 recorded days). Trials are analysed in 84 bins over 14 s
(nominal 6 Hz; bin b ≈ [b/6, (b+1)/6) s). Two window conventions coexist in the codebase and are
stated per analysis below: the single-trial (overlaps) pipeline indexes epochs directly
(baseline bins 0–11; mid-delay 33–38; late delay 45–53; test 54–59; DPA lick window 57–62),
whereas the pseudo-population pipeline offsets each epoch onset by 0.5 s (mid-delay bins 36–38;
late delay 48–53; decision 57–65).

### Behavioural statistics (Fig. 1)

Learning curves were modelled on per-mouse × day × condition accuracies (proportions, not
trials) with linear mixed-effects models — accuracy ~ condition × centred day with a random
intercept per mouse (REML); per-day markers are uncorrected Wald/Welch tests requiring ≥4 mice
per group. Random-effect variances near the boundary make per-day p-values mildly
anti-conservative, as noted in the text. Trial-level associations (Fig. 1g; history effects,
ED 2) used logistic GEEs clustered by mouse (exchangeable working correlation), fit separately
per stage. Fig. 1g models the probability of licking at the DPA test on NoGo trials as a
function of the intrusive cue lick (the pure delay-period lick variable); a lick × pairing
interaction term tests whether the propagation differs between paired and unpaired trials (it
does not, p = 0.61, so the pooled estimate applies to the unpaired arm, where the test lick is
the false alarm), and the paired arm is the propagation control (a cue lick there predicts a
hit — incompatible with memory corruption, diagnostic of response propagation). The panel was rebuilt 2026-09-01: the
original build's predictor pooled cue and test licks, which is near-circular with performance.
Across-animal relationships are Pearson/Spearman correlations over n = 9 mice. All tests
two-sided; p-values uncorrected and reported exactly.

### Two-photon imaging and pseudo-population

Prelimbic mPFC was imaged with two-photon microscopy through hSyn-GCaMP6s **[AUTHOR: surgery,
window/lens, rig, frame rate, ROI extraction/registration pipeline, per-mouse cell counts and
FOVs]**, yielding 3,319 neurons across the 9 mice. The pseudo-population analyses (Fig. 2 and
the per-mouse geometry analyses of Fig. 3c–f) use correct, laser-OFF trials; the single-trial
projection analyses (Figs 3a,b, 4, 6) use all laser-OFF trials, with correctness filters only
where stated (per-mouse × stage × task counts in Supplementary Information; 5,568 laser-OFF
trials in the balanced set). Neurons partition disjointly across mice, so all pseudo-population
resampling and jackknifing respects mouse identity (mice are the exchangeable unit).

### Optogenetics (Fig. 6)

CaMKII-Jaws-tdTomato was expressed in ACC and its mPFC terminals illuminated at 635 nm
**[AUTHOR: viral titres/coordinates, fibre placement, laser power, histology]**. Two designs are
combined, both targeting the same ACC→mPFC projection and analysed strictly separately: (i)
chronic training silencing — every-trial illumination throughout learning, compared between
groups (9 opto vs 9 control; Fig. 6b,c), with direction controls (ACC cell bodies; Prl→ACC) in
ED 7; (ii) transient silencing in the imaged, trained cohort — laser ON on a pseudo-random 50%
of delay periods, within-mouse ON vs OFF (Jaws, n = 5; Fig. 6d–l). In the transient design all
decoder axes are trained on laser-OFF trials only (no correctness filter on the projected ON
trials — a correct-only filter would be survivor-biased), and ON trials are projected through
the fixed OFF axis. The opto depth axis is the pre-committed main-overlaps axis trained on late
delay + test (bins 45–59; the window sweep is reported to avoid cherry-picking), with per-mouse
projections divided by the standard deviation of the baseline bins — deliberately the locked
earlier convention, which differs from Fig. 4's lick-window axis and evoked-SD unit; both
choices predate the opto analysis. Discriminability under laser (Fig. 6k,l) is per-axis d′ —
sample A vs B on the sample axis (late delay, bins 45–53) and Go vs NoGo on the choice axis
(mid-delay, bins 33–38) — modelled as d′ ~ laser + stage with a random intercept per mouse
(20 observations = 5 mice × 2 stages × laser OFF/ON).

### Cross-validated population decoders (the CCGD pipeline; Figs 3a,b, 4, 6)

All single-trial code readouts come from one pipeline. For each mouse, stage and target variable
(sample, choice/lick, test, distractor), an L2-regularised logistic-regression decoder
(class-balanced; regularisation strength selected by an inner 5-fold cross-validation over 10
log-spaced values, 10⁻⁴–10⁴) was trained in that mouse's neuron space on all laser-OFF trials
of that stage, and every trial's decision function was evaluated cross-temporally
(train bin × test bin) with stratified 5-fold cross-validation (folds stratified on
odour-pair × task × day), so that every projected trial is scored out-of-fold — trials from
other conditions or laser-ON trials are never in any training set. Decision functions
(normalised by the weight-vector norm per train bin) were averaged over the training window of
the relevant axis: sample code, train bins 16–47; test code, 58–83; choice ("action") code, the
DPA lick window, bins 57–62; distractor code, mid-delay bins 33–38. Per mouse, each code's
projections were then baseline-centred (subtracting the pooled mean over baseline bins 0–11)
and expressed in evoked-SD units — divided by the temporal standard deviation of that mouse's
baseline-centred, class-signed mean trajectory (all laser-OFF trials, both stages;
"pooled-evoked"). A "robust" variant normalising by each mouse's A–B sample separation,
alternative axis windows, and L1/LDA decoder variants all reproduce the conclusions (ED 5,
ED 6c). The delay-state depth used in Fig. 4 is the choice-axis projection averaged over late
delay (bins 45–53, pre-test), on all laser-OFF DPA trials (correct and error).

### One estimator for the geometry analyses (Figs 2e–g, 3c–f)

The pseudo-population and per-mouse geometry analyses — plane ablation (full-population and
residual arms), cross-task generalisation, cross-stage transfer, axis cosines — use one shared
estimator, defined once and imported by every script: standardisation, an *optional* PCA
compression to min(20, n_features, n_samples − 1) components, then L2-regularised logistic
regression (C = 1, class-balanced). The canonical build omits the PCA step (no-PCA); the PCA-20
build is the robustness companion, and every starred result is required to hold in both. Where a decision
direction is used as a geometric axis it is the pipeline's own decision vector mapped back to
neuron space (undoing PCA and standardisation) and unit-normalised, so decoder and axis are the
same vector and cannot disagree. (The one deliberate exception: the plane arm of the ablation
decodes from only two coordinates and uses a bare logistic regression on them.)

### Cross-validated dimensionality (Fig. 2b)

We estimated the reliable dimensionality of the pseudo-population (12 conditions = 3 tasks ×
2 samples × 2 test odours) with cross-validated PCA [Stringer 2019b]. For each of 30 random halvings (20 for the
ED participation-ratio bars), the trials of every (mouse, condition) pool were split into two
disjoint halves, yielding two independent condition-mean pseudo-populations. A PCA basis was fit
on one half and the variance of the other half evaluated by cross-projection (both directions
averaged): trial-to-trial noise averages to zero in this cross-term, so only variance that
replicates across independent halves — signal — is retained (on our data only ~9–24% of the
condition-mean variance replicates, and the naïve scree fails to separate states that the
reliable spectra separate cleanly). Because the basis is fit on a noisy half, per-component
values estimate signal variance along empirical axes: total reliable variance is unbiased but
the spectrum is flattened by basis misalignment [Pospisil 2025] — conservative for the low-dimensionality
claims made here. Neurons were scaled by a stage-level, condition-agnostic standard deviation
(scale only — no mean subtraction; condition means are centred across conditions inside the
estimator). Repeated split-half CV is used rather than k > 2 folds because the estimator is a
cross-product of two independent condition-mean estimates whose variance is minimised by equal
halves. These estimators operate on condition means: they characterise the task-conditioned
state geometry, not the single-trial state space. A per-mouse companion applies the identical
estimator within each mouse's own simultaneously recorded neurons (DPA 4-condition set, 30
halvings, ≥6 trials per condition; cells whose reliable-variance total falls below 5 are
flagged noise-limited, drawn open, and excluded from the paired memory-vs-decision Wilcoxon;
Extended Data Fig. 3c). The null is a label-shuffled realisation of
the full pipeline (condition labels permuted within mouse, trial counts preserved), normalised
by the real spectrum's positive total. Windows (pseudo-population convention): mid-delay
(bins 36–38 — the final third of the 5.5–6.5-s post-distractor epoch, closing at the Go/NoGo
cue onset, so no cue or lick has occurred) and decision (bins 57–65, post-test); the legacy PR
analyses additionally use late delay (bins 48–53). 95% CIs are leave-one-mouse-out jackknife
with a t(8) = 2.306 multiplier on the jackknife SE (fractions clipped to [0, 1]; the PR floored
at 1); the "unchanged with learning" statement applies the same jackknife to Δ(Naïve − Expert),
whose CI spans zero for every component and variable — reported as absence of detectable change
at the stated precision (CI half-widths 0.03–0.30), not strict equivalence.

### Per-variable decoding power and PC coding (Fig. 2c,d)

For each condition set and window, each variable's demixed axis was computed by applying its
orthogonal design contrast to condition means from a training half (binary factors and
window-averaged states make each dPCA marginalisation rank-1 [Kobak 2016]). Held-out pseudo-trials (one
test-half trial per mouse per pseudo-trial, 10 per condition) were projected on the axis and
classified by the training-set class midpoint; performance is balanced accuracy over 15 splits,
tested against the 95th percentile of a within-mouse label-shuffle null (100 shuffles, full
pipeline re-run per shuffle; conservative, as the null draws are single-split accuracies). The
"gng ×" entry cross-decodes Go vs NoGo from held-out dual pseudo-trials (24 per condition,
disjoint train/test halves, 8 repeats) projected into the DPA-state subspace — the top-3 PCs of
the DPA condition means, which are estimated from all DPA trials (the held-out split applies to
the decoded dual trials) — with LDA on the 3-D projection and a 1,000-shuffle within-mouse
label-permutation null. For Fig. 2d, each condition-mean PC's across-condition variance was
decomposed as η² onto mutually orthogonal factor contrasts (sample, distractor, test, choice;
plus a task contrast in the 12-condition set); the balanced design makes the shares exhaustive
(chance 1/3 per contrast, Beta(½,1) under no signal — large η² on unreliable components is
expected and is not evidence of coding, which is why decodability is tested directly in
Fig. 2c). PCs beyond the reliable rank are faded; the rank is the number of leading components
that individually exceed twice the shuffle floor, accumulated until 95% of the reliable
variance is reached (DPA mid-delay 1, dual mid-delay 2, both decisions 3). Apparent
delay-period test/choice η² in DPA cannot be anticipatory coding — the test odour is drawn
independently of the sample — and fails cross-validation; it is condition-mean sampling noise.

### Generalisation, parallelism, abstraction (Figs 2e–g, ED)

Cross-task generalisation (Fig. 2e) trains a decoder on one task and tests on held-out trials of
another, reported as (cross − 0.5)/(within − 0.5) against the test task's own within-task
ceiling; per-mouse companions and their learning equivalence (Δ 95% CIs within ±0.05) are in
Fig. 2f. The parallelism score (PS) is decoder-free: per task, a class-difference coding vector
is computed from condition means on independent trial halves (correct laser-OFF trials,
per-mouse subspaces written into the shared neuron space, unit-normalised), and PS is the
cross-half, sign-preserving cosine averaged over the three task pairs (10 half-splits); the
null permutes class labels within task (100 draws, 95th percentile), and the
reliability-corrected PS (raw ÷ split-half reliability) is reported as an estimate — it can
exceed 1 at low reliability — alongside the raw value. CCGP follows Bernardi et al. [Bernardi 2020], computed
on the pseudo-population with leakage-free matched cross-validation and label-shuffle nulls
(per-mouse companion: Wilcoxon, n = 9). The shattering dimension decodes all 462 balanced
6-vs-6 dichotomies of the 12 conditions at the decision window (disjoint train/test halves per
mouse × condition, 24 pseudo-trials per condition, scaler + PCA(30) fit on the training half
only, LDA per dichotomy, 8 resamples; null: pseudo-trial condition labels permuted, 0.50).
Per-neuron selectivity (Fig. 2g, ED 6b) uses per-neuron d′ (pooled-variance; sample at
mid-delay across tasks, choice at the decision window on correct DPA trials) with a selectivity
threshold at the 95th percentile of a within-mouse label-permutation |d′| null (computed on the
sample window and applied to both); the fraction of double-selective neurons is compared
descriptively with the product of the marginal fractions, and the cached statistic is the
correlation of |d′| across neurons.

### The sample × choice plane: ablation, cosines, cross-stage transfer (Fig. 3)

Panel-a traces replay the CCGD projections (above) on correct trials per task, mean ± SEM
across mice; panel-b snapshots read the same projections at three moments (overlaps
convention: mid-delay bins 33–38; late delay 45–53; decision read bins 60–66, a window scan
having moved the read window later than the 57–62 axis-training window) and re-centre each
window per mouse on its cross-condition mean state, so they display condition geometry, not
absolute position (ellipses = 1 SEM of the across-mouse mean). Plane ablation (Fig. 3c,d): per
mouse, the plane is the QR-orthonormalised span of that mouse's sample axis (mid-delay) and
behavioural choice axis (decision window, lick vs no-lick including errors), fit on one half of
the trials; each variable was then decoded from (i) only the two plane coordinates, (ii) the
residual after projecting the plane out, and (iii) the full population — training on the same
half and scoring the held-out half, 10 random half-splits, paired Wilcoxon across mice (n = 9);
the plane-vs-full equivalence is additionally bounded per cell (|Δ| ≤ 0.012 accuracy in every
mouse × stage for sample and choice; the plane arm decodes from two coordinates with a bare
logistic regression).
Axis-angle matrices (Fig. 3e) report attenuation-corrected cosines with the split-half
reliabilities disclosed alongside; the per-animal statistics use the raw (uncorrected)
per-mouse |cos| (Fig. 4a). Cross-stage transfer (Fig. 3f) trains the pipeline in one stage and
tests on held-out trials of the other — as a pooled pseudo-trial analysis (24–48 pseudo-trials
per class, 8 resamples) and per mouse on real trials (10 half-splits; per-mouse ratios require
a within-stage ceiling > 0.52) — summarised as the chance-referenced ratio
(cross − 0.5)/(within − 0.5); a scaling-sensitivity check re-scoring the test stage in the
training stage's per-neuron scaling changes the pooled ratios by ≤ 0.02.

### Repositioning and couplings (Fig. 4)

The distractor↔choice alignment (Fig. 4a) is quantified per mouse as the raw axis cosine and as
symmetric cross-decoding between the two codes (train on one, test held-out on the other), each
compared Naïve vs Expert by Wilcoxon (n = 9) and required to replicate across both decoder
pipelines. Delay-state depth (Fig. 4b) is the choice-axis projection of DPA delay states (late
delay bins 45–53, per-mouse evoked-SD units, all laser-OFF trials); the stage effect is a mixed
model — depth ~ stage + sample with a random intercept per mouse, over 36 mouse × stage ×
sample observations from 9 mice — with a per-animal Wilcoxon companion on the nine
Expert − Naive differences. The depth↔accuracy coupling (Fig. 4c) is a between-mouse Spearman
correlation over the nine per-mouse means of Δdepth and Δaccuracy (Expert − Naive) — the fitted
unit is the animal — with robustness established across all six axis × normalisation builds and
a resampling battery (jackknife, bootstrap, permutation; ED 5). The FA/CR control (Fig. 4d) is
a paired t-test on per-mouse median depths of correct-rejection vs false-alarm trials (naïve
nonpaired trials, ≥3 trials per cell). The d′ control (Fig. 4e) computes the choice code's
pooled-variance d′ at the axis window (bins 57–62) per mouse × stage and compares stages by
paired t-test.

### Statistical policy

All tests are two-sided; exact p-values are reported uncorrected and multiplicity is addressed
by disclosure and by the replication requirements rather than correction. Claims about
individual differences use the animal as the unit (n = 9 Spearman/Wilcoxon; mixed models with
mouse random effects); trial-level models are never used for between-animal claims
(pseudoreplication is flagged wherever a raw trial-level statistic is shown, e.g. Fig. 6g).
Every pooled pseudo-population claim is paired with a per-animal companion statistic. A result
is starred only if it replicates across both decoder pipelines (no-PCA and PCA-20); †
marks pipeline-dependent results, which are reported without a verdict.

### Data and code availability

Analysis code (Python; one shared decoder module, per-figure scripts) will be deposited at
**[AUTHOR: repository/DOI]**; imaging and behavioural data at **[AUTHOR: archive/DOI]**.
(Reproducibility note: the CCGD tensor's cross-validation partition is currently unseeded —
re-running the tensor build permutes folds; all downstream statistics average over folds, but
bit-exact tensor reproduction requires seeding, flagged for the deposition.)

---

## Figure legends

> Mirrored from the scripts’ CAP_PARAS (the single source of truth — edit BOTH together;
> Nature legend style since 2026-09-03; Fig. 5, modelling, in preparation).

Figure 1 | An embedded action task interferes with working memory through the intruding lick.
Recorded cohort, nine mice, laser-off trials; curves show mean ± SEM across mice; ∗ p < .05, ∗∗ p <
.01, ∗∗∗ p < .001 (per-day linear mixed models, uncorrected; day 6 n = 4).

a, Task design. Each trial is a delayed paired-association (DPA) problem: a sample odour (A or B), a
6-s delay, then a test odour (C or D); the mouse licks if the pair matches (A→C, B→D) and withholds
otherwise. On two thirds of trials a Go/NoGo (GNG) discrimination is embedded in the delay — a
distractor odour, then a response cue: lick on Go, withhold on NoGo (DualGo / DualNoGo trials); the
remainder are pure DPA. All three trial types are interleaved within every session. Right: the
training curriculum.

b–e, Learning curves (per-mouse/day accuracy). b, DPA vs GNG performance. c, GNG split by distractor
identity. d, DPA paired vs unpaired trials. e, DPA unpaired trials by surrounding task context.

f, Linear mixed model over panels b–e (fixed effects ± 95% CI; filled circles, condition offset;
open squares, condition × day slope; random intercept per mouse): GNG−DPA β = +0.037 (p = 0.045),
narrowing over days; NoGo−Go +0.072 (p = 0.034); unpaired−paired −0.185 (p < 10⁻⁴), narrowing;
Go−DPA −0.073 (p = 0.038).

g, Probability of licking at the DPA test on NoGo trials, split by whether the animal licked at the
distractor cue (thin lines, single mice). In naïve mice a cue lick tripled the odds of licking again
at test (trial-level GEE, OR = 3.10, p = .006); the propagation is pairing-independent (lick ×
pairing interaction p = .61) — on unpaired trials the second lick is the false alarm (OR = 2.7, p =
.09), on paired trials a hit (OR = 9.9, p = .001). In expert mice the propagation is absent (OR =
1.50, p = .42) and cue licks are rare (rate 0.24 → 0.08).

h, Expert DPA vs GNG accuracy per animal (colour, mouse; marker, opsin group; star, the both-optimal
corner). No animal reaches the corner (mean gap 0.18) and the two accuracies are uncorrelated (r =
+0.10, p = .80; ρ = +0.35, p = .36; n = 9).

Figure 2 | One dedicated coding axis per task variable, with the memory and choice axes shared
across tasks. All panels use the pseudo-population (3,319 neurons, nine mice, 12 conditions). Memory
state, mid-delay window (5.5–6.3 s; after the distractor, before any cue or lick); decision state,
test onset onward.

a, Trial timeline, the two analysed states, and the cross-validated PCA (cvPCA): condition means are
estimated on one half of the trials and evaluated on the other (30 random half-splits, both
directions averaged), so only replicating structure counts.

b, Fraction of reliable condition-mean variance per cvPCA component (error bars, leave-one-mouse-out
jackknife 95% CI, t(8); dashed grey, within-mouse label-shuffle null). The DPA mid-delay state
occupies a single reliable axis; the dual tasks add exactly one (distractor 0.92 vs sample 0.07);
the decision state needs ≈3. Naïve and expert spectra are near-identical.

c, Decoding accuracy for each variable along its own demixed axis, on withheld pseudo-trials,
against each stage’s own label-shuffle null (95th percentile; expert solid, naïve open). A variable
decodes exactly when it is in play; † marks the single exception — an anticipatory choice signal in
naïve mid-delay (0.66 vs its null) that disappears with learning. Hatched bar: the distractor
decoded from the DPA-state subspace (top-3 PCs), a weak but reliable transfer at mid-delay
(permutation p = .031, 1,000 draws) against 1.0 within the dual tasks.

d, η² of each condition-mean PC against the design contrasts (rows, PCs labelled with their % of
condition-mean variance; a cell near 1 means that PC codes that variable). Rows beyond panel b’s
reliable rank are faded; the orange box carries panel c’s distractor cross-decode per DPA PC; dual
rows show 4 of the 7 centred contrasts.

e, Decoders trained on one task, tested on the others (expert). Cells, the transferred fraction of
decodable signal, (cross − 0.5)/(within − 0.5); column labels print each test task’s within-task
ceiling; hatched cells, ceiling ≈ chance or ratio > 1 (not interpretable). Below each matrix, the
parallelism score against a label-shuffle null; corrected for split-half reliability, the sample and
choice directions are essentially parallel across tasks (≈0.96–1.0).

f, Per-mouse mean cross-task accuracy, naïve vs expert (points on the unity line = no change). All
changes are n.s. (Wilcoxon, n = 9; both decoder variants) and bounded: the Δ 95% CIs fall within
±0.05 accuracy (sample [−.03, +.02]; test [−.01, +.04]; choice [−.03, +.05]).

g, Per-neuron discriminability (d′, within-mouse) for sample at mid-delay vs choice at decision (n =
3,319; grey square, label-shuffle floor). |d′| across the two variables is uncorrelated (r = −0.03)
and the both-selective fraction (6.2%) equals the independence prediction (6.4%).

Figure 3 | A single sample × choice plane is necessary and sufficient for the memory and choice
codes, and learning pulls the distractor code into it. A code is the projection of population
activity on a per-mouse cross-validated decoder axis (the decoder never sees the projected trials),
baseline-zeroed, in units of that mouse’s evoked s.d.; correct laser-off trials; mean ± SEM across
nine mice; p values uncorrected.

a, The two frame axes read in each task (columns, DPA | Go | NoGo × sample / choice; rows, naïve |
expert). The DPA sample code is maintained across the delay, whereas on dual trials the same readout
decays after the distractor (lower in 9/9 naïve and 8/9 expert mice). On the choice axis the Go
trace rises at the cue in both trial classes (every correct Go trial licks the cue) and the lick/no-
lick split opens only at the test; the expert NoGo trace runs below baseline through late delay (7/9
mice). Distractor and test codes are shown in Extended Data.

b, The same projections as snapshots of the sample × choice plane (windows: mid-delay 5.5–6.3 s,
late delay 7.5–8.8 s, decision 10–11 s; the choice axis is trained at the lick moment, 9.5–10.5 s).
Each panel is re-centred per mouse on that window’s mean state and therefore shows condition
geometry, not absolute position (the shared ramp and the push are carried by a and Fig. 4b). Dots,
per-mouse condition means (≥3 correct trials); ellipses, SEM across mice; large marker, grand mean;
fill = lick, open = no-lick; circle / triangle / square = DPA / Go / NoGo; colour = sample; scale
bar, 2 z.

c, Decoding each variable from the plane’s two coordinates, from the residual after removing the
plane, and from the full population (mean ± SEM, n = 9, stages averaged; withheld trial halves;
paired Wilcoxons, all comparisons drawn). Sample and choice decode as well from the plane as from
the full population and collapse without it (p = .004); test is at chance from the plane and
untouched without it (p = .004); the distractor is partial (p = .004). † marks the one decoder-
variant-dependent comparison.

d, The same comparison per animal (naïve x vs expert y; rows, spaces; columns, variables). Learning
changes are n.s. everywhere except the distractor’s plane-only accuracy (0.57 → 0.65, p = .020/.027
across decoder variants, 8/9 and 7/9 mice up), starred.

e, |cos| between the decoder axes, attenuation-corrected by split-half reliabilities (printed above
each matrix; 0 = orthogonal). The sample axis is orthogonal to both action codes (≈0.07–0.09, both
stages); the choice × dist overlap is partial and grows (0.32 → 0.47). Right, the raw within-mouse
|cos|, naïve vs expert; the choice × dist increase is the starred per-animal test of Fig. 4a; no
tests are drawn here.

f, Cross-stage decoding (registered neurons): axes trained in one stage read the other stage’s
withheld activity at ~90% of the within-stage ceiling (transfer/within 0.90 sample, 0.87 choice;
cross-stage accuracy 0.88 ± 0.03–0.05; robust across decoder variants, resampling, and a common-
scaling check, ratios shifting ≤ 0.02). Right, the same test within each animal. Learning moves the
state within the frame (Fig. 4b); it does not rotate the frame.

Figure 4 | Learning rotates the distractor code onto the choice axis and repositions the memory
state along it, predicting each animal’s memory improvement. Code depth, the projection on the
choice (lick) decoder axis, per mouse, baseline-zeroed, in evoked-s.d. units; negative = toward no-
lick.

a, Cross-decoding between the distractor and choice codes (balanced accuracy; diagonal, within-code;
off-diagonal, transfer). The chance-referenced transfer grows from 0.33 [−0.03, 0.60] (naïve) to
0.57 [0.36, 0.75] (expert). Right, the same convergence within each animal, naïve vs expert: per-
mouse |cos| 0.073 → 0.114 (∗ p = .008) and cross-decode 0.53 → 0.61 (∗ p = .004); both robust across
decoder variants; drawn from fixed canonical caches in every build.

b, DPA delay trajectories in the sample × choice plane (naïve | expert; strips, late-delay depth
distributions) and per-mouse late-delay depth. The deepening is significant in the mixed model (β =
−0.74, p = .046 ∗; 9 mice, 36 observations) and a per-animal trend (Wilcoxon p = .098), carried by
sample A (Δ = −1.42, p = .098; sample B ≈ 0; the A-vs-B difference itself n.s., p = .055).

c, Each mouse’s change in depth vs its change in accuracy (circles, the two sample classes per
mouse, joined; the regression band, ρ and p are computed on the nine per-mouse means). A deeper push
predicts DPA improvement (ρ = −0.83, p = .005 ∗); the same change predicts nothing for GNG (ρ =
+0.20, p = .61).

d, Trial-level control (naïve nonpaired trials): within a stage, single-trial depth does not
separate correct rejections from false alarms (sample A, Δ(CR−FA) = −1.16, p = .27; sample B, +0.73,
p = .47).

e, The choice code’s discriminability (d′, lick vs no-lick) is unchanged with learning (0.80 → 1.07,
p = .25): learning moves where the state sits (b), not how well the axis reads out.

Figure 6 | ACC→mPFC input is required for learning the composition and acutely sets the state’s
position without degrading the code. Code depth as in Fig. 4.

a, Design: hSyn-GCaMP6s imaging in mPFC with CaMKII-Jaws-tdTomato in ACC; 635-nm laser on a pseudo-
random 50% of trials, delay period only — every comparison in d–l is within-mouse, laser ON vs OFF.

b, c, Chronic every-trial silencing during training (separate cohort; 9 opto vs 9 control mice,
between-group): DPA acquisition is impaired (b); the mixed-model summary (c; group ● and group × day
□ fixed effects ± 95% CI) loads the deficit on DPA (β = −0.06, p = 0.009) and its unpaired trials (β
= −0.12, p = 0.014), sparing GNG.

d–f, Transient silencing in the recorded cohort (Jaws, n = 5): DPA (d) and GNG (e) accuracy are
unchanged ON vs OFF, yet the same manipulation displaces each animal’s choice-code depth (f; per-
mouse depth, samples pooled; directions differ across mice, so the group mean is flat).

g–i, The displacement against the accuracy change it accompanies (Δ, ON−OFF; 20 points = 5 mice ×
naïve/expert × sample A/B; depth read on the trainLD_TEST axis at late delay). The joint trade-off
(g) is a raw-level trend (r = +0.53, p = .016; the points cluster within five mice and a mouse-
clustered model gives p = .108); its arms are ΔDPA (h, n.s.) and ΔGNG (i, r = −0.65, p = .002; ρ =
−0.62, p = .003; the one arm surviving the clustered model, β = −0.013, p = .018).

j, Laser-ON behaviour keeps its DPA–GNG balance (r = +0.44, p = .20).

k, l, Discriminability is spared: d′ ON vs OFF sits on the unity line for the memory code (k;
sample-axis d′ at late delay; LMM laser p = .34, n = 10 observations) and the GNG code (l; choice-
axis d′ at mid-delay; p = .74). The projection sets the code’s position (f–i) without degrading its
quality — the position-not-fidelity principle of Fig. 4.

## References

> Working author–year list keyed to the inline [Author Year] tags (both this file and
> `discussion_draft.md`); converted to numbered Nature format by the reference manager at
> submission. Driscoll 2024 and Pospisil 2025 verified by search 2026-09-02; the rest are
> standard anchors. Note the two distinct Stringer 2019 papers (a = movements; b = cvPCA).

- **[Bernardi 2020]** Bernardi, S., Benna, M. K., Rigotti, M., Munuera, J., Fusi, S. & Salzman,
  C. D. The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell* **183**,
  954–967 (2020).
- **[Driscoll 2024]** Driscoll, L. N., Shenoy, K. & Sussillo, D. Flexible multitask computation
  in recurrent networks utilizes shared dynamical motifs. *Nat. Neurosci.* **27**, 1349–1363
  (2024).
- **[Golub 2018]** Golub, M. D., Sadtler, P. T., Oby, E. R., Quick, K. M., Ryu, S. I.,
  Tyler-Kabara, E. C., Batista, A. P., Chase, S. M. & Yu, B. M. Learning by neural
  reassociation. *Nat. Neurosci.* **21**, 607–616 (2018).
- **[Kaufman 2014]** Kaufman, M. T., Churchland, M. M., Ryu, S. I. & Shenoy, K. V. Cortical
  activity in the null space: permitting preparation without movement. *Nat. Neurosci.* **17**,
  440–448 (2014).
- **[Kobak 2016]** Kobak, D., Brendel, W., Constantinidis, C., Feierstein, C. E., Kepecs, A.,
  Mainen, Z. F., Qi, X.-L., Romo, R., Uchida, N. & Machens, C. K. Demixed principal component
  analysis of neural population data. *eLife* **5**, e10989 (2016).
- **[Libby 2021]** Libby, A. & Buschman, T. J. Rotational dynamics reduce interference between
  sensory and memory representations. *Nat. Neurosci.* **24**, 715–726 (2021).
- **[Liu 2014]** Liu, D., Gu, X., Zhu, J., Zhang, X., Han, Z., Yan, W., Cheng, Q., Hao, J.,
  Fan, H., Hou, R., Chen, Z., Chen, Y. & Li, C. T. Medial prefrontal activity during delay
  period contributes to learning of a working memory task. *Science* **346**, 458–463 (2014).
- **[Mante 2013]** Mante, V., Sussillo, D., Shenoy, K. V. & Newsome, W. T. Context-dependent
  computation by recurrent dynamics in prefrontal cortex. *Nature* **503**, 78–84 (2013).
- **[Musall 2019]** Musall, S., Kaufman, M. T., Juavinett, A. L., Gluf, S. & Churchland, A. K.
  Single-trial neural dynamics are dominated by richly varied movements. *Nat. Neurosci.* **22**,
  1677–1686 (2019).
- **[Oby 2019]** Oby, E. R., Golub, M. D., Hennig, J. A., Degenhart, A. D., Tyler-Kabara,
  E. C., Yu, B. M., Chase, S. M. & Batista, A. P. New neural activity patterns emerge with
  long-term learning. *Proc. Natl Acad. Sci. USA* **116**, 15210–15215 (2019).
- **[Panichello 2021]** Panichello, M. F. & Buschman, T. J. Shared mechanisms underlie the
  control of working memory and attention. *Nature* **592**, 601–605 (2021).
- **[Parthasarathy 2017]** Parthasarathy, A., Herikstad, R., Bong, J. H., Medina, F. S.,
  Libedinsky, C. & Yen, S.-C. Mixed selectivity morphs population codes in prefrontal cortex.
  *Nat. Neurosci.* **20**, 1770–1779 (2017).
- **[Pospisil 2025]** Pospisil, D. A. & Pillow, J. W. Revisiting the high-dimensional geometry
  of population responses in the visual cortex. *Proc. Natl Acad. Sci. USA* **122**,
  e2506535122 (2025).
- **[Rigotti 2013]** Rigotti, M., Barak, O., Warden, M. R., Wang, X.-J., Daw, N. D., Miller,
  E. K. & Fusi, S. The importance of mixed selectivity in complex cognitive tasks. *Nature*
  **497**, 585–590 (2013).
- **[Sadtler 2014]** Sadtler, P. T., Quick, K. M., Golub, M. D., Chase, S. M., Ryu, S. I.,
  Tyler-Kabara, E. C., Yu, B. M. & Batista, A. P. Neural constraints on learning. *Nature*
  **512**, 423–426 (2014).
- **[Stringer 2019a]** Stringer, C., Pachitariu, M., Steinmetz, N., Reddy, C. B., Carandini, M.
  & Harris, K. D. Spontaneous behaviors drive multidimensional, brainwide activity. *Science*
  **364**, eaav7893 (2019).
- **[Stringer 2019b]** Stringer, C., Pachitariu, M., Steinmetz, N., Carandini, M. & Harris,
  K. D. High-dimensional geometry of population responses in visual cortex. *Nature* **571**,
  361–365 (2019).
- **[Vyas 2020]** Vyas, S., Golub, M. D., Sussillo, D. & Shenoy, K. V. Computation through
  neural population dynamics. *Annu. Rev. Neurosci.* **43**, 249–275 (2020).
- **[Yang 2019]** Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T. & Wang, X.-J. Task
  representations in neural networks trained to perform many cognitive tasks. *Nat. Neurosci.*
  **22**, 297–306 (2019).

---

## Extended Data Figures

> Consolidated from the granular supplement set into **9 multi-panel Extended Data (ED) figures** (Nature
> Neuroscience caps ED at 10); the trial-count reporting figure → Supplementary Information. Each ED figure
> backs specific main-figure claims/_(caveats)_; all panels are in the shared house style (gallery Supp
> tab). Stats are current run values (verified 2026-08-03). **Trims applied 2026-08-03:** former S5
> (demixed axes) and S17 (d′ standalone) cut as redundant; former S13 folded into ED 6; flows (former S7)
> removed → "extra".
>
> **COMPOSED 2026-09-02** — the 9 ED pages + the SI figure now exist as real composed figures with
> justified in-figure captions: `make_ed_figures.py` → `figures/ed/png/ed_fig{1..9}.png` +
> `si_trialcounts.png` (native-resolution mosaics of the component renders; bold lowercase page letters
> in the left margin; PNG-only — a raster mosaic gains nothing from SVG; share PDFs made from the PNGs).
> The panel letters BELOW are the canon and match the composed pages (ED 3 was relettered a–g; its
> old (a)/(b)/(f) dPCA descriptives have no standalone renders and live in ED 9 — the entries below
> and all in-text refs were updated 2026-09-02). Edit captions in `make_ed_figures.py` and this section
> TOGETHER.

**ED Fig. 1 | Behaviour: learning curves by cohort (Fig. 1).** Five rows (each = the A–E
curve-plus-LMM-forest strip): pooled 9 mice, Jaws (n=5), ChR (n=2), ACC (n=2), and the interleaved
laser-ON trials of the 7 laser mice. Condition effects reproduce (pooled: GNG−DPA β=+0.037 p=0.045;
NoGo−Go +0.072; unpaired−paired −0.185; Go−DPA −0.073). Learning is comparable across cohorts.

**ED Fig. 2 | Behaviour: the DPA↔GNG balance is not a trade-off (Fig. 1e/g/h).** (a) per-animal
DPA-vs-GNG scatter (Naïve co-vary r≈0.67 → Expert decouple r=+0.10); (b) Pareto front (no animal on the
both-optimal corner); (c) a small fixed dual cost (Δ≈−0.03, per-mouse view) with a *positive*
within-trial DPA×GNG coupling (Δ=+0.097, p=0.025); (d) the trial-level GEE companion — dual-vs-pure
cost n.s. within stage, DPA|GNG-correct coupling OR=2.03, p=0.001 (Expert); (e) trial-history
(sub-panels A–H): a preceding dual trial lowers current-Go DPA accuracy (OR=0.81, p=0.047; GNG
history-independent); (f) the blocked-design switch-cost mirrors it (into-dual OR=0.90, p<0.001).

**ED Fig. 3 | Dimensionality: provenance & robustness (Fig. 2b–d).** (a) the previous build of
Fig. 2 (`fig_dimensionality_main_pr.png`, sub-panels A–D): cvPCA schematic, full 12-condition
"all-tasks" spectra + the PR ladder memory 1.0 [1.0, 1.1] → delay 2.0 [1.6, 2.5] → decision 3.3
[2.8, 3.8] (jackknife CIs); the full-state delay's two large dimensions are its context contrasts
(distractor presence and identity); (b) reduced-rank test — held-out fit rises smoothly with no
elbow at 2 (rank-2 = 62–67% of full), backing the "rank-2 geometry, not rank-2 dynamics" caveat;
(c) **the per-mouse cvPCA companion** (`fig_permouse_cvpca.png`, built 2026-09-02): the Fig. 2b
spectra reproduced within each mouse's own simultaneously recorded population (DPA set, same
estimator) — top-1 reliable fraction, memory vs decision window, per stage; noise-limited cells
(reliable-total < 5) drawn open and excluded from the test; medians 0.90/0.93 at mid-delay, expert
memory-vs-decision Wilcoxon p = .047 (6/7), naïve directional p = .22; (d) the full per-fit grid
for the all-tasks set (`dim_all.png`): cvPCA scree, cross-validated PR and shattering per window,
and the per-PC η² coding matrices — condition-mean PCs beyond the reliable ones carry apparent η²
for variables undetermined at that point (sampling noise stripped by cvPCA, not anticipatory
coding: the gotcha flagged in Fig. 2d's footnote); (e) window robustness (`dim_DPA_altwin.png`) —
on full-delay / test windows the DPA-delay PR stays 1.0–1.1; (f) the Go/NoGo cross-decode column
from the DPA subspace, per window (`dim_DPA_gng.png`; main Fig. 2c/d shows the clean mid-delay
value 0.61; the late-delay ~0.7 figure is consummatory-inflated — the DPA geometry is close to,
but not fully, orthogonal to the distractor); (g) **learning removes the premature choice signal
from the dual delay** (`fig_bias_cleanup_ed.png`): in naive mice the upcoming match/nonmatch
choice is decodable from the dual delay state from ED through LD (0.64–0.66 vs shuffle-null ≈0.59,
demixed-axis held-out decoding), and the held-out future-choice separation on a late-delay-defined
— hence reward-free — axis climbs to ~+2 z by LD; in Expert the same signal sits at chance
throughout the delay (0.47–0.49) while post-test decoding is intact (0.96). DPA shows no such
signal at either stage (control). Decodability already at ED (post-sample, pre-distractor) marks
it as a trial-history/bias state rather than premature deliberation. Caveats: on correct trials
choice ≡ trial completion, so state-dependent selection contributes to the naïve separation; and
at the mouse level the learning difference is not individually resolved (Δ accuracy +0.19,
leave-one-mouse-out jackknife CI [−0.09, +0.46], n = 9) — established at the pooled-population
level, where it replicates across three independent pipelines. The shattering dimension (all 462
balanced dichotomies, 0.69 → 0.70 vs shuffle 0.50) is cited from main Fig. 2c and reappears in the
per-fit grids (d–f); the descriptive dPCA scree, per-marginal variance, and shared-memory d′
scatter are in **ED 9** (no standalone renders — the old (a)/(b)/(f) sub-entries of this figure).

**ED Fig. 4 | dPCA no-lick push robustness (corroborates Fig. 4b).** (a) The Naïve→Expert deepening reproduces in raw ΔF/F
(r≈0.997, not a z-score artifact); (b) survives condition-independent time-ramp removal (q0/1/2 =
−0.59/−0.60/−0.61); (c) holds on a Naïve-defined pooled basis (8/9; bootstrap CI [−0.56,−0.08]); and (d) is
population- not individual-level in this pipeline (depth↔accuracy null, r=+0.46, p=0.21 — the calibrated
overlaps pipeline of Fig. 4c is the individual-level assay).

**ED Fig. 5 | Overlaps: coupling/push robustness + movement control (Fig. 4b,c).** (a–b) the Δdepth↔ΔDPA
coupling is ★ under all six normalisations (ρ=−0.83 to −0.90) and survives a fixed common axis (ρ=−0.72)
where the push attenuates to a trend; (c) a resampling battery (Mundlak β=−0.041 p=0.006; jackknife 9/9;
bootstrap CI [−1.00,−0.26]; permutation p=0.008), ΔGNG null throughout; (d) movement control — late-delay
licking is rare, the choice-code depth does not track it (ρ=+0.07), and the push/coupling are unchanged
with a lick covariate.

**ED Fig. 6 | Overlaps: the factorised geometry is robust (Fig. 3e; Fig. 2g).** (a) cross-temporal cosine matrices
— cross-code |cos| ≈ the 0.05 chance floor at all time-pairs, within-code diagonals 0.4–0.9, choice×GNG the
one least-orthogonal pair (~0.29); (b) modular, not mixed, selectivity — per-neuron permutation tuning
(sample 10 / GNG 39 / test 3 / choice 10 %, cross-variable co-tuning at chance); (c) decoder-variant
robustness — the main figure under L1 and LDA decoders (geometry/orthogonality decoder-invariant;
push/coupling clearest under logistic); (d) codes robust to the Go/NoGo distractor — panel-A codes split by
Go vs NoGo (sample/test unperturbed; the action code carries the distractor lick).

**ED Fig. 7 | Opto: chronic silencing + transient behaviour (Fig. 6b–e).** (a–c) control-vs-opto learning
curves for the ACC→Prl, ACC-somata and Prl→ACC batches — ACC→Prl impairs DPA (β=−0.06 p=0.009) and its
unpaired trials (β=−0.12 p=0.014); ACC-somata null; Prl→ACC impairs GNG; (d) transient within-mouse laser
OFF-vs-ON curves (Jaws n=5, sub-panels A–D): DPA p=0.40, GNG p=0.24 — geometric, not a behavioural
knock-down.

**ED Fig. 8 | Opto: laser ON−OFF coupling, 7 mice (Fig. 6g–i).** The acute causal analog of the learning
coupling over all 7 laser mice (5 Jaws + 2 ChR): (a) one point per mouse — GNG ρ=−0.90 (p=0.006, n=7),
DPA rank-n.s. (ρ=+0.55, p=0.21); (b) sample A & B as independent points (n=14) — GNG ρ=−0.60 (p=0.024),
DPA rank-n.s. Backs the Jaws-only axis choice and the alternative-n disclosure.

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
"extra"); the standalone d′ figure (former S17 → already main Fig. 6k,l); demixed-axes loadings/mixing
(former S5 → covered by Fig. 2g + ED 6). **Author-supplied gaps still needed:** histology / viral
expression, imaging FOV + per-mouse cell counts, laser-power / opsin titration.

---

### To reconcile before submission (figure ↔ text integrity)
- **✓ REGISTER PASS (2026-09-02, Albert Compte's review → Leon: "jargon pass now, keep structure"):**
  20 replacements in the body (Abstract→Synthesis) + 1 in the Discussion — ML-flavoured vocabulary
  translated (decode ablation → decoding after removing the plane; bounded equivalence → "agree to
  within X accuracy"; estimators → measures; decoder pipelines → decoder variants; held-out →
  withheld-from-fit at first use; disentangled → factorised; parks → pushes/moves; knob-dependent →
  named explicitly; expressivity → separating capacity; resampling battery → every resampling check;
  well depth → state depth). Claim-first structure and ALL statistics untouched; Methods keeps its
  technical register deliberately. In-figure captions not swept (they drop at submission).
- **✓ CODEX DRAFT REVIEW APPLIED (2026-09-01, adjudicated):** applied = claims calibration
  (abstract/§2/§4/§5/Synthesis/Discussion verbs + "no detectable"), §5 clustered-ΔGNG-first
  reorder, ED count 8→9, estimator-paragraph rewording, expert-day parenthetical, CCGP
  spelled out, n = 9/n = 5 in abstract, 6k,l LMM n stated as 20 obs; upgraded (beyond Codex's
  ask) = Fig 1g pairing-interaction test (p = .61, in script/caption/Methods) and plane-vs-full
  bounded equivalence (|Δ| ≤ 0.012 every cell — "sufficient" retained with formal backing).
  REJECTED with reasons = "days 4–6" (day counts vary 4–6 per mouse; "day 4 to last" is
  correct), push reframing beyond verb level (star = settled decision; full evidential balance
  already led), multiplicity adjustment (stated policy: uncorrected + disclosure + two-pipeline
  replication), Fig 5 placeholder removal (model in preparation — a SUBMISSION blocker only).
  STILL OPEN from the review: §2 density reduction (deferred — restructuring risks stat drift;
  revisit at submission), the pooling-defense sentence (candidate: the cross-animal alignment
  analysis would upgrade it to a panel).
- **✓ Fig. 1g REBUILT (2026-09-01, resolving the predictor audit; user decision — "the delay
  lick leads to more false alarms, not memory corruption; the push prevents these false
  alarms"):** the old build's `licked` predictor pooled cue and test licks (= the test lick on
  96.5% of NoGo trials, near-circular with performance; its OR = 0.56 was carried entirely by
  the test lick — cue-only OR = 0.93 n.s.). The panel now shows the clean propagation:
  predictor = the cue lick (`odr_choice`), outcome = P(lick at test); naïve OR = 3.10
  p = .006 ∗∗, expert OR = 1.50 p = .42 n.s.; FA arm (unpaired) OR = 2.73 p = .090, paired arm
  OR = 9.9 p = .001 toward a hit (the anti-corruption control); cue-lick rate 0.24 → 0.08.
  §1 header, §2/§4/abstract clauses, caption, Methods and behavior.md all recast to
  "propagates to false alarms". Downstream Discussion clause updated.
- **Fig. 2a mid-delay bracket label (audit):** the panel/caption say "5.5–6.3 s" but the sampled
  bins 36–38 map to ≈6.0–6.5 s under the code's own bin→time convention (the 5.5 is the epoch
  onset before the +0.5-s index offset; no trailing-integration code exists). Either relabel the
  bracket ≈6.0–6.5 s or document the frame-timestamp convention that justifies 5.5. The
  substantive property (post-distractor, pre-cue, pre-lick) holds under both mappings.
- **Fig. 6k,l annotation (audit):** the panel prints "n=10" (plotted OFF/ON pairs); the LMM is
  fit on 20 rows (5 mice × 2 stages × OFF/ON). Relabel or reword.
- **Estimator drift check (audit):** `overlaps/fig_ccgp_matrices_pseudo.py`'s module pipeline
  lacks `class_weight='balanced'` while `pca/decoders.py`'s docstring claims the two match —
  verify which estimator actually produced the current `matrices_cache_acc_nopca.pkl` and align.
- **Hard-coded figure literals (audit):** Fig. 2g's "6.2% / 6.4%", all caption β/ρ/p strings,
  and the Fig-5 header stats are literals (currently verified correct) — they will drift
  silently if caches are rebuilt; re-verify at submission render.
- **Minor code-comment rot (audit; comments only, no behaviour):** `exp_frame_states.py`
  docstring still says decision read = 57–62 (code: 60–66); `main_panels.py` comments say KDE
  bins 48–53 / BINS_DELAY 21–53 (actual 45–53 / 18–53); `fig_behavior_opto_main.py` docstring
  misstates why its depth window differs from the main figure. PS null uses |cos| while the
  observed PS is signed (conservative); the plane-basis half-split is not nested for the
  test/dist arms (indirect, disclosed); CCGD tensor CV is unseeded (noted in Methods).
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
- **Figure numbering (UPDATED 2026-09-01, supersedes the 2026-08-04 resolution):** the modelling figure
  IS coming and takes **Fig. 5**; the opto figure is renumbered **Fig. 6** (all refs in this draft, the
  in-figure caption, and the gallery cards updated). Fig. 4 stays the overlaps push/repositioning figure.
  Order: Behaviour (1) → Geometry (2) → One manifold (3) → Learning (4) → Modelling (5, TBD) → Opto (6).

### Framing guardrails (keep the thesis defensible)
- "low-dimensional **geometry** / rank-2 portrait", never "the dynamics are rank-2". Flow-field /
  attractor-dynamics claims are OUT of the paper (kept as "extra") — don't reintroduce "landscape",
  "gated deformation", "bistable" into the main text.
- near-orthogonality = the representational basis of compositionality (factorised code → combine without
  cross-talk) — the load-bearing, solid claim.
- the pre-existing naïve no-lick well is the strongest "structure already there, fine-tuned" evidence —
  lean on it for the reuse claim.
