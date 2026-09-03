# Discussion — draft v4 (2026-09-03, human-voice pass: sentence rhythm and explicit reasoning
# in the Golub/Libby register; limitations as prose; statistics untouched since v1)

> Companion to `results_draft.md` (v6). Anchors and vocabulary from memory
> `reference_literature_positioning` (Liu 2014; Bernardi 2020; Libby & Buschman 2021;
> Parthasarathy 2017; Sadtler 2014 / Golub 2018 / Oby 2019; Kaufman 2014; Rigotti 2013; plus
> Mante 2013, Vyas 2020, Yang 2019, Driscoll 2024, Panichello 2021, Musall/Stringer 2019 added in
> v2). Leads with the three novel claims; does NOT lead with orthogonality itself (well-trodden).
> Citations [Author Year] for the reference manager. All statistics identical to v1.

We set out to ask how prefrontal cortex composes a new task out of computations the animal
already performs. Three results answer the question. First, a single two-dimensional
sample × choice subspace is both necessary and sufficient for the memory and choice codes of
every task variant: sample and choice decode as well from the plane's two coordinates as from
the whole population, to within 0.012 accuracy in every mouse and stage, and collapse when the
plane is removed, while the test code lives outside it (Fig. 3c,d). Second, learning acts within
this fixed subspace. It rotates the distractor code onto the choice axis and moves the
working-memory state along it, and the size of each animal's shift predicts how much its memory
improves (Fig. 4). Third, a defined top-down input, the ACC→mPFC projection, shifts the position
of the state on the subspace while sparing the discriminability of the code, and is required for
the memory task to be acquired (Fig. 6). Taken together, these findings support a simple account
of compositional learning as geometric editing. The geometry of the code is a fixed constraint;
learning and top-down control both operate on the position of the state within it.

The format we find, one dedicated and nearly orthogonal axis per task variable that generalises
across task contexts and across animals, is the abstract geometry that cross-condition
generalisation has revealed in primate prefrontal cortex and hippocampus [Bernardi 2020]. Here
it appears in mouse mPFC, and under active interference: the same axes have to serve a memory
while an embedded action task runs through it. Our data also refine the mixed-selectivity
picture [Rigotti 2013]. The decision state occupies about three reliable dimensions and the test
code is conjunctive, in that it anti-generalises across sample, so the high-dimensional
component is there. The maintained memory, by contrast, is deliberately minimal, a single
reliable dimension (Fig. 2b). The two regimes coexist with a division of labour: low-dimensional
and abstract where information has to be protected and transferred, higher-dimensional where
stimulus–response conjunctions have to be resolved. That the abstract format is built from
largely separate, independently tuned neurons (Fig. 2g), rather than from a rotated mixed code,
makes the factorisation legible at the level of single cells.

In auditory cortex, memories are protected from incoming sensory drive by rotational dynamics
that move the memory into a dimension orthogonal to the sensory response [Libby 2021]. mPFC in
our task uses a different, two-layered scheme. The memory axis is statically orthogonal to both
action codes throughout the trial (Fig. 3e), and learning adds a second layer: a positional
offset along the action axis itself, which holds the delay state at an increasingly
output-suppressing set-point on the no-lick side (Fig. 4b). The set-point is close to, but
distinct from, the output-null principle of motor cortex [Kaufman 2014]. Preparatory activity
there is held where it cannot drive output; the memory here is held where it actively opposes
output, a displacement along the potent axis rather than a retreat into the null space. Several
observations fit a protective function. The behavioural cost of the distractor is an evoked lick
that propagates to the test response, the false-alarm route (Fig. 1g); the animals that push
deeper improve more (Fig. 4c); and the interference disappears over the same period of learning
in which the push deepens. The decay of the sample readout on dual trials (Fig. 3a) is the code
morphing described after distractors in primate PFC [Parthasarathy 2017]. Our cross-task
transfer (Fig. 2e) adds that the morphed code is still partly readable by the same axes, and the
plane analysis (Fig. 3c) that the morph does not leave the subspace.

Brain–computer-interface experiments have shown that short-term learning is confined to the
network's existing manifold [Sadtler 2014] and proceeds by reassociation, reusing a fixed
repertoire of activity patterns while changing what they are used for [Golub 2018], with
genuinely new dimensions accessible only slowly and with effort [Oby 2019]. To our knowledge,
our results are the first demonstration of the same principle during the natural acquisition of
a task. The subspace and its axes are unchanged by learning, under equivalence bounds (Fig. 2f)
and by cross-stage decoding (Fig. 3f; transfer at about 90% of the within-stage ceiling), while
the position of the state and the alignment of the distractor code change (Fig. 4). Learning
here does not build coding dimensions; it re-parameterises a fixed geometry. This is also the
solution that network models of multitask computation converge on: recurrent networks trained on
families of tasks solve new tasks by composing shared, reusable dynamical motifs rather than by
growing new ones [Yang 2019; Driscoll 2024]. Our data place that solution in cortex during real
learning and identify what the composition physically is, a rotation and a repositioning inside
a conserved subspace.

The mPFC delay activity our task depends on is known to matter most during learning [Liu 2014].
Our chronic-versus-acute dissociation sharpens this. Removing the ACC input throughout training
prevents the memory task from being acquired (Fig. 6b,c), whereas removing it acutely in a
trained animal leaves behaviour and the fidelity of the code intact but shifts the code's
position (Fig. 6d–f,k,l). The projection behaves less like a channel for content than like a
bias input that places the population state on the learned geometry. This is top-down control
acting on the same variable that learning acts on, the position of the state, in the spirit of
executive inputs that configure a computation rather than carry it [Mante 2013;
Panichello 2021]. The contrast between the two couplings is telling. Learning's repositioning
buys memory accuracy at no cost to the distractor task (Fig. 4c), whereas the acute displacement
trades the two tasks against each other (Fig. 6g–i). The slow process finds the factorised
solution that the fast perturbation cannot.

Several limitations bound these conclusions. First, within each mouse the neurons were recorded
simultaneously and tracked across days, but the pooled pseudo-population combines neurons from
mice that were never recorded together. Its single-trial states are therefore assembled across
animals and carry no cross-mouse noise correlations. Pooled claims accordingly concern the
task-conditioned geometry of the states, and every one of them carries an n = 9 within-animal
companion computed on genuinely simultaneous data, including the dimensionality spectra
themselves (per-mouse cvPCA, Extended Data Fig. 3c); larger simultaneous populations within
single animals would strengthen the trial-by-trial claims further. Second, the acute optogenetic
coupling rests on five animals, and its joint trade-off does not survive a mouse-clustered model
(only the ΔGNG arm does). We therefore frame the acute result as position-moving with a robust
distractor-side coupling, not as a full causal reproduction of the learning effect. Third, the
choice axis is defined by licking. The delay-period displacement is not accompanied by delay
licking, the fidelity of the code is insensitive to laser-related movement, and the coupling
survives a lick covariate (ED 5d), but uninstructed orofacial movements shape cortical activity
broadly [Musall 2019; Stringer 2019a], and motion-energy regression would separate motor
execution from choice coding more completely. Fourth, the push itself is directional and modest
per animal (mixed-model p = .046, per-animal trend); its force comes from convergence, the same
reorganisation appearing in alignment, position, behavioural coupling and causal perturbation.
Fifth, we describe the geometry; a circuit model showing that push and alignment emerge from
training on this composition is in preparation (Fig. 5), and our preliminary latent-dynamics
fits point in that direction.

Composition, in this system, is not construction. mPFC brings a factorised, shared subspace to
the problem; learning edits the positions and alignments of states within it; and a top-down
projection supplies the positional signal. If cortical computation is carried by population
geometry, then learning to do two things at once is, quite literally, a matter of knowing where
to put things.
