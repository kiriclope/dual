# Discussion — draft v6.2 (2026-09-07, review fixes: "required" → impairs, "first demonstration" dropped,
# stable-code refs, ED 6f pointer; v6.1 = claim 1 without "necessary and sufficient"; stats untouched)

> Companion to `results_draft.md` (v6). Anchors and vocabulary from memory
> `reference_literature_positioning` (Liu 2014; Bernardi 2020; Libby & Buschman 2021;
> Parthasarathy 2017; Sadtler 2014 / Golub 2018 / Oby 2019; Kaufman 2014; Rigotti 2013; plus
> Mante 2013, Vyas 2020, Yang 2019, Driscoll 2024, Panichello 2021, Musall/Stringer 2019 added in
> v2). Leads with the three novel claims; does NOT lead with orthogonality itself (well-trodden).
> Citations [Author Year] for the reference manager. All statistics identical to v1.

We set out to ask how prefrontal cortex composes a new task out of computations the animal
already performs. Three results answer the question. First, the memory and the choice are
carried by one two-dimensional sample × choice plane, the same plane on every trial type and
before and after learning; the test code lies outside it, and the distractor is drawn into it
as learning proceeds (Figs 2e, 3c–f). Second, learning acts within this fixed subspace. It rotates the distractor code onto the choice axis and moves the working-memory state along it,
and the size of each animal's shift predicts how much its memory improves (Fig. 4). Third, a
defined top-down input, the ACC→mPFC projection, shifts the position of the state on the
subspace while sparing the discriminability of the code, and its silencing during training
impairs the acquisition of the memory task (Fig. 6). Together, these findings support a simple
account of compositional learning as geometric editing. The geometry of the code is a fixed
constraint, and learning and top-down control both operate on the position of the state within
it.

The format we find, one dedicated and nearly orthogonal axis per task variable that generalizes
across trial types and is present in every animal, is the abstract geometry that
cross-condition generalization has revealed in primate prefrontal cortex and hippocampus
[Bernardi 2020]. Here it appears in mouse mPFC, and under active interference, because the same
axes have to serve a memory while an embedded action task runs through it. Our data also refine
the mixed-selectivity picture [Rigotti 2013]. The decision state occupies about three reliable
dimensions and the test code is conjunctive, in that it anti-generalizes across sample
(Extended Data Fig. 6f), so the high-dimensional component is there. The maintained memory, by
contrast, is deliberately minimal, a single reliable dimension (Fig. 2b). The two regimes
coexist with a division of labour: low-dimensional and abstract where information has to be
protected and transferred, higher-dimensional where stimulus–response conjunctions have to be
resolved. That the abstract format is built from largely separate, independently tuned neurons
(Fig. 2g), rather than from a rotated mixed code, makes the factorization legible at the level
of single cells.

In auditory cortex, memories are protected from incoming sensory drive by rotational dynamics
that move the memory into a dimension orthogonal to the sensory response [Libby 2021]. In our
task, mPFC uses a different, two-layered scheme. The memory axis is statically orthogonal to
both the choice and the distractor code throughout the trial (Fig. 3d), a stable rather than a
dynamic memory code [Murray 2017; Spaak 2017], and learning adds a second layer, a positional
offset along the choice axis itself, which holds the delay state at an increasingly
output-suppressing set-point on the no-lick side (Fig. 4b). The set-point is close to, but
distinct from, the output-null principle of motor cortex [Kaufman 2014]. Preparatory activity
there is held where it cannot drive output; the memory here is held where the readout of the
same axis is no-lick, a displacement along the potent axis rather than a retreat into the null
space. Several observations fit a protective function. The behavioral cost of the distractor is
an evoked lick that propagates to the test response, the false-alarm route (Fig. 1g); the
animals that push deeper improve more (Fig. 4c); and the false-alarm arm of that propagation is
selectively weakened, relative to its hit arm, over the same period of learning in which the
push deepens. The decay of the sample readout on dual trials (Fig. 3a) is the code morphing
described after distractors in primate PFC [Parthasarathy 2017]. Our cross-task transfer (Fig.
2e) adds that the morphed code is still partly readable by the same axes.

Brain–computer-interface experiments have shown that short-term learning is confined to the
network's existing manifold [Sadtler 2014] and proceeds by reassociation, reusing a fixed
repertoire of activity patterns while changing what they are used for [Golub 2018], with
genuinely new dimensions accessible only slowly and with effort [Oby 2019]. Our results extend
the same principle to the natural acquisition of a composite task. The subspace and its axes
are unchanged by dual task learning, under equivalence bounds (Fig. 2f) and by cross-stage
decoding (Fig. 3e; transfer at 90% of the within-stage ceiling for the sample axis and 72% for the choice axis), while the position of
the state and the alignment of the distractor code change (Fig. 4). Learning here does not
build coding dimensions; it re-parameterizes a fixed geometry. This is also the solution that
network models of multitask computation converge on, since recurrent networks trained on
families of tasks solve new tasks by composing shared, reusable dynamical motifs rather than by
growing new ones [Yang 2019; Driscoll 2024]. Our data place that solution in cortex during real
learning and identify what the composition physically is, a rotation and a repositioning inside a conserved subspace.

The mPFC delay activity our task depends on is known to matter most during learning [Liu 2014].
Our chronic-versus-acute dissociation sharpens this. Removing the ACC input throughout training
impairs learning of the dual task, with the deficit on the memory task (Fig. 6b,c), whereas
removing it acutely in a trained animal leaves behavior and the fidelity of the code intact but
shifts the code's position (Fig. 6d–f,k,l). The projection behaves less like a channel for
content than like an input that biases where the population state sits on the learned geometry.
This is top-down control acting on the same variable that learning acts on, the position of the
state, in the spirit of executive inputs that configure a computation rather than carry it
[Mante 2013; Panichello 2021]. The contrast between the two couplings is telling. Learning's
repositioning buys memory accuracy at no cost to the distractor task (Fig. 4c), whereas the
acute displacement is coupled to the distractor task (Fig. 6g–i). The slow process arrives at a
solution that the brief perturbation does not reproduce.

Three features of the design set the scope of these conclusions. The pooled pseudo-population
combines neurons recorded in different animals, so its single-trial states carry no cross-mouse
noise correlations, and for this reason every pooled result is paired with a within-animal
companion computed on simultaneously recorded neurons, including the dimensionality spectra
themselves (n = 9; Extended Data Fig. 3c). Because the mice had learned DPA and GNG separately
before dual task recordings began, our claims concern the composition of the two: the axes
present in the first dual task sessions were the ones that learning went on to use, and
composing the tasks added none. And although the choice axis is defined by licking, the
delay-period displacement occurs without delay licks, the fidelity of the code is insensitive
to laser-related movement, and the behavioral coupling survives a lick covariate (Extended Data
Fig. 5d).

Composition, in this system, looked like editing rather than construction. The prefrontal
population brings a factorized, shared subspace to the problem, learning edits the positions
and alignments of states within it, and a top-down projection influences where the state sits.
If cortical computation is carried by population geometry, then learning to do two things at
once is largely a matter of where the states are put.

