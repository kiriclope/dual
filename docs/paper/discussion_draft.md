# Discussion — draft v2 (2026-09-01, high-impact rewrite)

> Companion to `results_draft.md` (v6). Anchors and vocabulary from memory
> `reference_literature_positioning` (Liu 2014; Bernardi 2020; Libby & Buschman 2021;
> Parthasarathy 2017; Sadtler 2014 / Golub 2018 / Oby 2019; Kaufman 2014; Rigotti 2013; plus
> Mante 2013, Vyas 2020, Yang 2019, Driscoll 2024, Panichello 2021, Musall/Stringer 2019 added in
> v2). Leads with the three novel claims; does NOT lead with orthogonality itself (well-trodden).
> Citations [Author Year] for the reference manager. All statistics identical to v1.

We set out to ask how prefrontal cortex composes a new task from computations the animal already
performs. Three results answer it. First, decode ablation shows that a single two-dimensional
sample × choice subspace is *necessary and sufficient* for the memory and choice codes of every
task variant — sample and choice decode as well from the plane's two coordinates as from the full
population (bounded equivalence, |Δ| ≤ 0.012 in every mouse and stage) and collapse without
them, while the test code lives outside it (Fig. 3c,d). Second, learning acts *within* this
fixed subspace: it rotates the distractor code onto the choice axis and repositions the
working-memory state along it, and the size of each animal's repositioning predicts its memory
improvement (Fig. 4). Third, a defined top-down input — the ACC→mPFC projection — causally
shifts the state's position on the subspace while sparing the code's discriminability, and is
required for the memory task to be acquired (Fig. 6). Together they support a simple account: the code's geometry is a fixed
constraint, and both learning and top-down control operate on the state's position within it.

**Relation to abstract representational geometry.** The format we find — one dedicated,
near-orthogonal axis per task variable, generalising across task contexts and animals — is the
abstract, disentangled geometry described by cross-condition generalisation in primate PFC and
hippocampus [Bernardi 2020], here observed in mouse mPFC and, critically, *under active
interference*: the same axes must serve a memory while an embedded action task runs through it.
Our data also refine the mixed-selectivity view [Rigotti 2013]: the decision state occupies ≈3
reliable dimensions and the test code is conjunctive (it anti-generalises across sample), so the
high-dimensional component is present — but the *maintained memory* is deliberately minimal, a
single reliable dimension (Fig. 2b). The two regimes coexist with a division of labour:
low-dimensional and abstract where information must be protected and transferred,
higher-dimensional where stimulus–response conjunctions must be resolved. That the abstract
format is built from largely separate, independently tuned neurons (Fig. 2g) rather than from a
rotated mixed code makes the factorisation legible at the level of single cells.

**Two mechanisms of memory protection.** In auditory cortex, memories are protected from incoming
sensory drive by *rotational dynamics* that move the memory into a dimension orthogonal to the
sensory response [Libby 2021]. mPFC in our task uses a different, two-layer scheme: the memory
axis is *statically* orthogonal to both action codes throughout the trial (Fig. 3e), and learning
adds a second layer — a positional offset along the action axis itself, holding the delay state at
an increasingly output-suppressing set-point on the no-lick side (Fig. 4b). The set-point is
conceptually adjacent to, but distinct from, the output-null principle in motor cortex
[Kaufman 2014]: preparation there is held where it *cannot* drive output; the memory here is held
where it actively *opposes* output — a displacement along the potent axis, not a retreat into
null space. Consistent with a protective function, the behavioural cost of the distractor is
precisely an evoked lick that propagates to the test response — the false-alarm route
(Fig. 1g) — the animals that push deeper improve more (Fig. 4c), and the
interference disappears over the same learning period in which the push deepens. The dual-trial
decay of the sample readout (Fig. 3a) is the code-morphing phenomenon described after distractors
in primate PFC [Parthasarathy 2017]; our cross-task transfer (Fig. 2e) adds that the morphed code
remains partly readable by the same axes, and the plane analysis (Fig. 3c) that the morph does not
leave the subspace.

**Learning as reassociation in a natural task.** Brain–computer-interface experiments have shown
that short-term learning is constrained to the network's existing manifold [Sadtler 2014] and
proceeds by *reassociation* — reusing a fixed repertoire of activity patterns while changing what
they are used for [Golub 2018] — with genuinely new dimensions accessible only slowly and
effortfully [Oby 2019]. Our results are, to our knowledge, the first demonstration of the same
principle across natural task acquisition: the subspace and its axes are unchanged by learning
under equivalence bounds (Fig. 2f) and by cross-stage decoding (Fig. 3f, transfer at ~90% of the
within-stage ceiling), while the state's position and the distractor code's alignment change
(Fig. 4). Learning here does not build coding dimensions; it re-parameterises a fixed geometry.
This is also what network models of multitask computation converge on: recurrent networks trained
on task families solve new tasks by composing shared, reusable dynamical motifs rather than by
growing new ones [Yang 2019; Driscoll 2024] — our data place that solution in cortex during real
learning, and identify what the composition physically is: a rotation and a repositioning inside
a conserved subspace.

**Top-down input as a positional signal.** The mPFC delay activity our task depends on is known to
matter most during learning [Liu 2014]; our chronic-versus-acute dissociation refines this:
removing the ACC input throughout training prevents the memory task from being acquired
(Fig. 6b,c), whereas removing it acutely in a trained animal leaves behaviour and code fidelity
intact but shifts the code's position (Fig. 6d–f, k, l). The projection behaves less like a
content channel than like a *bias* input that places the population state on the learned geometry
— top-down control acting on the same variable that learning acts on, the state's position, in
the spirit of executive input that configures rather than carries the computation [Mante 2013;
Panichello 2021]. The contrast between the two couplings is informative: learning's repositioning
buys memory accuracy at no distractor-task cost (Fig. 4c), while the acute displacement trades the
two tasks (Fig. 6g–i) — the slow process finds the factorised solution the fast perturbation
cannot.

**Limitations.** (i) Within each mouse, neurons were recorded simultaneously and tracked across
days; the pooled pseudo-population, however, combines neurons from mice that were never recorded
together, so its single-trial states are assembled across animals and carry no cross-mouse noise
correlations — pooled claims therefore concern the task-conditioned state geometry, and every one
carries an n = 9 within-animal companion computed on genuinely simultaneous data. Larger
simultaneous populations within single animals would further strengthen the trial-by-trial
claims. (ii) The acute optogenetic coupling rests on five animals,
and its joint trade-off does not survive a mouse-clustered model (only the ΔGNG arm does) — we
therefore frame the acute result as position-moving with a robust distractor-side coupling, not
as a full causal reproduction of the learning effect. (iii) The choice axis is defined by licking;
delay-period displacement is not accompanied by delay licking, code fidelity is
movement-insensitive under laser, and the coupling survives a lick covariate (ED 5d), but
uninstructed orofacial movements shape cortical activity broadly [Musall 2019; Stringer 2019a],
and motion-energy regression would further separate motor execution from choice coding. (iv) The
push itself is directional and modest per animal (mixed-model p = .046, per-animal trend); its
force comes from convergence — the same reorganisation seen in alignment, position, behavioural
coupling and causal perturbation. (v) We describe the geometry; a circuit model showing that push
and alignment emerge from training this composition is in preparation (Fig. 5), and our
preliminary latent-dynamics fits point that way.

**Conclusion.** Composition, in this system, is not construction. mPFC brings a factorised,
shared subspace to the problem; learning edits the positions and alignments of states within it;
and a top-down projection supplies the positional signal. If cortical computation is carried by
population geometry, then learning to do two things at once is, quite literally, a matter of
knowing where to put things.
