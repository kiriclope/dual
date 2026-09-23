# A circuit model of the gated no-lick repositioning — modelling section (draft v1.7)

> **v1.7 (2026-09-23): the objective is now the probit cross-entropy of every scored event (Methods, eq. 6); new Results section on the depth law and the other computables; Supplementary Note §13. Numbers to be re-anchored on the eight networks of `sweep_lif_log` when they land.**

> **v1.6 (2026-09-23): the no-lick ablation (ED 19) — without the dual-stage no-lick term the wells rise into the lick half in 12/12, tied or not: the cost is the push, the symmetry its geometry. v1.5: every accuracy in Fig. 5, ED 17 and ED 18 rescored with the any-time lick criterion (a lick = κ₁ crossing the line at any step of the window, as for the animals); NoGo is then ≈ 0 at the trained wells, and the perturbation (drive to the cue) shows the NoGo response appearing only for wells 2–3 η deep.**

> **v1.4 (2026-09-22, night): the post-stimulus scoring sweep landed — the NoGo paragraph is rewritten from it (the post-cue lick was priced all along; the networks pay the price). Fig. 5 re-laid out on Leon's plan — a model, b curriculum, c the symmetries with a tied network's flow under each element, d the group across the curriculum, e the push, f–h performance against the well position; the lick probabilities, the checkpoint flows and the populations move to Extended Data Fig. 18.**

> **v1.3 (2026-09-22, evening): Fig. 5d (the group across the curriculum) and 5h–j added — DPA and Go/NoGo performance against the well position, moved by a delay-only drive along m₁ (the model twin of the optogenetic experiment of Fig. 6).**

> **v1.2 (2026-09-22, evening): Fig. 5 reworked — matrix-image schematic, lick probabilities in Fig. 1's colours, error bars and a Wilcoxon test on the push, the ties as paired heights, six populations named.**
> **v1.1 (2026-09-22, later): σ₁/V scaffolds on the fixed code added — every scaffold fails NoGo under the frozen bias; two exact-tie networks hold a four-well memory.**
> **v1.0 (2026-09-22): first assembly of the modelling part for the main draft's §5 slot (Fig. 5 in preparation).**
> Written to the standard of results_draft.md v12.5: claim-sentence headings, every panel cited, seeds as the unit
> of replication, exact statements separated from soft ones. Source note and full derivations:
> https://claude.ai/artifact/ANVVa4bWp1bzB4fByKJFwW and https://claude.ai/artifact/TVnuduX42gSBzCWvxDeSyd. This page: https://claude.ai/artifact/YXnnS3NxWR3TcDPqa9RoXL.
> Figures: `figures/paper_share/modelling/` (scripts in `~/rnn/paper/`). **Numbers in brackets [AUTHOR: …] are pending
> the two sweeps still running (σ₁/V scaffolds on the fixed code; every choice read after its stimulus).**

<!-- RESULTS -->

## A rank-2 network trained on the same curriculum reproduces the no-lick repositioning

To ask what kind of circuit produces the repositioning of Fig. 4, we trained recurrent networks with the
minimal structure the data allow: a rank-2 recurrent matrix, so that the network's state lives on a plane
spanned by a sample coordinate κ₀ and a choice coordinate κ₁, the sample and choice axes of Figs 3 and 4, with the
lick read out as the sign of κ₁ (Fig. 5a; Methods). Inputs for the sample, the Go/NoGo odor, the response
cue and the test odor project onto the units, and the networks were trained on the animals' curriculum,
DPA first, then Go/NoGo, then the dual task, with the same trial structure and timing as the behavior
(Fig. 1a). Eight networks differing only in their random initialization learned all three stages (Fig. 5b; Extended Data Fig. 18a; DPA
choice 0.99–1.00 on DPA-only trials after every stage; after the dual stage Go 1.00 and NoGo 0.96–1.00). The
curriculum also reproduced the cost of the dual task: after the Go/NoGo stage, before any dual training, the memory
survived a Go trial in most networks (DPA choice on Go trials, median 0.93) while the NoGo response on dual trials
was at chance in several (median 0.61), the model's version of the delay-lick interference of Fig. 1, and the dual
stage removed it.

The trained networks reproduced the geometry of Figs 3–4. After the DPA stage the two samples were held as
two attractors of the autonomous dynamics on the sample axis, one for each odor, sitting on the lick line
κ₁ = 0 (Extended Data Fig. 18c, left; memory-well height −0.3 to +0.2 η, where η is the s.d. of the state under the
trained noise). The Go/NoGo stage left them there (Extended Data Fig. 18c, middle). The dual stage moved both wells into
the no-lick half of the plane, together, in every network (Extended Data Fig. 18c, right, and Fig. 5e; 8/8 networks with both
wells below the line, heights −0.6 to −1.9 η). This is the no-lick push of Fig. 4b in a circuit: what the
dual task adds is a standing displacement of the memory state along the choice axis, away from the lick
line, and it is produced only when the memory has to be carried through a lick decision.

## The repositioning is forbidden by a symmetry of the memory task and produced by an asymmetry of the dual task

Why the memory sits on the lick line after DPA, and why the dual stage can move it, both follow from the
symmetry of the tasks (Extended Data Figs 11 and 12). The DPA objective is unchanged by three
relabelings of its stimuli: exchanging A with B and C with D (σ₁), exchanging A with B alone with the
response flipped (σ₂), and exchanging C with D alone with the response flipped (σ₃); with the identity they
form the Klein four-group V (Fig. 5c). Each acts on the plane by a sign flip, σ₁ on κ₀, σ₂ on both coordinates, σ₃
on κ₁, and a network whose units respect the relabeling has a flow that respects it exactly (Methods,
eq. 3). The consequences are exact and were confirmed in every network we tied to an element of the group
(Fig. 5c, lower row; Extended Data Fig. 13): under σ₁ the two memory wells share a height; under σ₃, or the whole group, the
mean memory is pinned on the lick line, because that line is invariant under the flow; and under any
element that flips κ₁ the choice readout is deaf to the Go/NoGo inputs, the overlap of the choice readout with
the Go and NoGo input columns being identically zero. The free networks obey the group approximately,
which is why their DPA memory sits on the line: nothing in the memory task breaks the symmetry that pins it
there.

The dual task does. Its objective is one-sided: a lick is penalized wherever it is wrong, including the
delay and the NoGo response, and nothing penalizes not licking. This cost is not invariant under any
element that flips κ₁, so of the four-group only σ₁ survives the dual stage, and the cost's gradient has a
definite sign, downward, wherever the state is above the line in a no-lick window (Methods, eq. 6). The
displacement of Fig. 5e is this explicit symmetry-breaking field acting on a memory that the memory task
had left symmetric. The one parameter the model has that can carry the break, the mean of the choice
readout across units, moved in the direction the field pushes (⟨n₁⟩ = −0.04 to −0.42 after the dual stage,
zero to within 0.02 before it).

The networks show this accounting stage by stage (Fig. 5d). The Go/NoGo stage is where the group first loses
its two response-flipping elements, and it loses them on the input side: under σ₂ or σ₃ the overlap of the
choice readout with the Go and NoGo input columns is forced to zero, so a network that respects either
element cannot hear the odor; learning the task installs these overlaps in every network (n₁ᵀw_Go from
−0.6…0.1 after DPA to 1.1…2.1 after Go/NoGo, n₁ᵀw_NoGo to −2.3…−1.2), while the sample readout, protected
by σ₁, keeps its overlaps small. The autonomous field, which does not see the input columns, breaks σ₂ and
σ₃ only in the dual stage, when the no-lick cost acts on it (equivariance residual of σ₂ and σ₃ over the
disk |κ| ≤ 1.5, median 0.30 and 0.57 after DPA, 0.80 and 0.83 after Dual), and it keeps σ₁ throughout
(median 0.25 at every checkpoint). This is why the push of Fig. 5e is a rigid displacement: the element
that survives exchanges the two samples and fixes the choice axis, so the wells move together, to one height.

The cost, not the symmetry, is what moves the wells. Removing the no-lick term from the dual stage alone,
everything else unchanged, sends both wells up into the lick half in every network (12/12; 0.2 to 1.5 η above
the line after the dual stage, against 0.6 to 1.9 η below with the term), whether the network is free, tied
to σ₁ during DPA, or tied to σ₁ through the whole curriculum (Extended Data Fig. 19). With the tie held
throughout the two wells stay at exactly one height while they rise. The remaining terms of the dual
objective, the Go response and the pairing, are not neutral on the choice axis: they leave a net upward pull
on the delay state, and the no-lick term is what overrides it. The symmetry fixes the geometry of the push,
one height for the two wells; the one-signed cost fixes its direction.

## Holding the symmetry during DPA training makes the repositioning reliable

If the symmetry is what keeps the memory on the line and the dual cost is what moves it, then holding the
symmetry exactly during the DPA stage and releasing it for the rest of the curriculum should give the
dual stage a memory it can move as one. We trained networks tied to σ₁ alone, to the whole group, and to
each of σ₂ and σ₃ alone, during DPA only. At the DPA checkpoint the memory obeyed each tie as predicted
(Extended Data Fig. 13): a level pair just below the line under σ₁ (−0.3 η in 4/4), a pair pinned on the line under the
whole group (|κ₁| ≤ 0.02) and under σ₃ (0.00), and an antipodal pair that chose the line under σ₂
(|κ₁| ≤ 0.07). After release, every scaffold ended the curriculum with both memory wells below the line in most networks
(Extended Data Fig. 15; 4/4 under σ₁, σ₂ and σ₃, 3/4 under the whole group, heights −0.3 to −1.9 η), where the networks
trained without a tie reached 6/8 in our first run and 8/8 once the input bias was frozen in the dual stage as
the inputs are (Methods). The two failures of the first run were the two networks that had left the DPA
stage with the inversion σ₂ most intact relative to σ₁, and both kept one memory well above the line, the
configuration a residual σ₂ permits and σ₁ forbids. The whole-group network that missed lost its memory
altogether in the dual stage (a single well, DPA at chance), and with the tie held exactly two of eight
scaffolded networks already held their memory as the four-well configuration the group allows (Extended Data Fig. 13,
σ₁ and V, one seed each), the memory carrying a choice sign before any lick was required.

The scaffold has a cost, and it falls on the NoGo response. With the bias frozen in the dual stage, every
scaffolded network licked at the NoGo cue on a large fraction of dual trials (NoGo accuracy read during the
cue: σ₁ 0.12–0.97, σ₂ 0.55–0.76, σ₃ 0.18–0.91, whole group 0.01–1.00), while the free networks held it
(0.96–1.00); the same scaffolds trained with the bias free in the dual stage held it too (0.95–1.00) at the
price of a bias that grew by half. The failure is in the response rather than in the memory: before the cue
the NoGo state sits at 1 to 2.6 η below the line in every network, the cue, which arrives on the Go channel,
lifts it to −0.3 to −1.0 η during the cue, and it settles 0.1 to 0.5 η above the line once the cue is off. Read
after the cue rather than during it, every network, free or tied, therefore licks on a majority of NoGo trials.
The objective prices this lick, with a hinge on κ₁ > 0 from the cue onset to the test, but the price is small
next to the Go response target on the same channel, and training settles on the trade. Moving the response
windows after the stimuli, for training and for scoring alike, does not change it: networks trained with
every choice read in the 0.5 s after its stimulus reach the same DPA performance (0.88–1.00) and the same Go
performance (1.00) but NoGo accuracies of 0.13–0.58 free and 0.03–0.94 across the scaffolds, with the dual
stage loss settling higher (1.7–2.1 against 1.5–1.8). In the model the NoGo response is a suppression that
holds during the cue and not after it; a NoGo that outlasts the cue would need the NoGo memory to hold the
state below the line against the cue's push, which the one-sided cost, at its present weight, does not buy.

## The sample code is untouched because the retained symmetry keeps the Go/NoGo inputs off the sample readout

The paper's second finding, that the code itself does not change while the state moves (Figs 3g, 4e), has
the same origin. The element that survives every stage, σ₁, exchanges the two samples and leaves the choice
axis alone; a network that respects it has a sample readout whose overlap with the Go, NoGo and cue input
columns is exactly zero, so the Go/NoGo stage cannot write on the memory (Extended Data Fig. 14). In the
tied networks this overlap stayed at 0.000 through the curriculum; in the free networks it stayed small
in most seeds and leaked in two (|n₀·w_nogo|/N up to 1.1), which is the model's version of the sample
coding being preserved on average but not in every animal (Fig. 4b).

At the level of single units the group also says what a trained population must look like, and the
networks comply (Extended Data Figs 13 and 18b). The units of every trained network fall into six
populations: four that carry a sample sign and a choice sign in all four combinations, mixed units in the
sense of Fig. 3g, and a pair of pure-choice units with no sample loading that carries the choice attractor.
These are exactly the two kinds of orbit the group allows for a population, and they are made by training
from an isotropic initialization: a unit's trained position is uncorrelated with its initial one. The
structure lives in the units' input weights, which are quantized, and not in their readout weights, which
remain continuous within each population. Read as a prediction for recordings, single-neuron tuning should
be categorical, six classes, while the weight a neuron carries in the population readout should vary
continuously within a class.

## Moving a well along the choice axis trades DPA against Go/NoGo performance

If the position of the memory wells on the choice axis is what the dual stage optimizes, then moving a well
should move performance, and the two tasks should be pulled in opposite directions: a well deeper in the
no-lick half is safer against a NoGo lick but further from the lick threshold the test must reach on a
paired trial. We tested this in the trained networks by a delay-only perturbation that mimics the optogenetic
manipulation of Fig. 6, a constant drive to every unit in proportion to its choice write weight m₁, applied
from the end of the sample until the stimulus that triggers the response arrives (Methods). The drive shifts the state
along the choice axis while it lasts and leaves the trained field, including the test-driven one, untouched;
so the well is moved without retraining and without acting on the response. Because both wells move
together, each drive strength gives two well locations, one per sample. Across eight networks, 81 drive
strengths and both wells, DPA accuracy stayed at ceiling for wells from 1.8 η below the line to 2 η above it
and fell on both sides (Fig. 5f): for deeper wells from misses only (mean 0.73 below −2.4 η; Spearman
ρ = 0.85 over the wells below the trained position), and for wells more than 2 η above the line from false
alarms only (the unpaired trial no longer returns below the line before the response is read; mean 0.79).
Go/NoGo accuracy, scored as the animals are, a lick being any crossing of the line in the response window,
depends on the well the cue finds: at the trained position, 0.9 η below the line, the cue's transient
crosses the line on nearly every NoGo trial (NoGo 0.02), and the NoGo response only appears as the well is
pushed deeper, 0.31 at 2 to 2.5 η below, 0.61 at 2.5 to 3 η, 0.78 to 0.89 beyond 3 η, where Go begins to
fail (0.77 to 0.88) because the cue no longer lifts the state across the line (Fig. 5g; ρ = −0.85). The two
tasks are traded against each other along the choice axis. Combining them as the probability that both
responses of a dual trial are correct, the product of the two accuracies, gives the dual performance as a
function of the well position (Fig. 5h): it is 0.51 at the trained position and peaks at 0.64 for wells 2 to
3 η below the line, where the NoGo gain outweighs the DPA misses. The trained position is therefore not the
optimum of the animals' criterion: it is the optimum of the objective the network was trained on, whose
no-lick term prices the mean excursion above the line rather than any crossing (Methods), and under which
the transient costs little. A network trained to withhold any lick would have to place its wells 2 to 3 η
deep and pay for it in DPA misses.

The group reads these curves directly. Under σ₁, which every stage keeps, the drive maps the A well onto the
B well, so the two wells sit at one height and give the same performance at every drive (the A and B points
coincide in Fig. 5f–h). Under σ₃, which the dual cost breaks, performance would be a mirror-symmetric
function of the well position about the lick line; the measured curves are not (Fig. 5h, the σ₃ image of
the dual curve), by the same asymmetry that moved the wells there.

## The likelihood of the lick sets the depth of the wells

Because the objective is the likelihood of the behavior, the geometry it produces is computable
(Supplementary Note §13). The no-lick term pushes a well down with a force that decays like the Gaussian tail
of the noise, and the paired response pulls it up with the mirror tail centered on the state the test
displaces it to; with each term a mean over its own steps the window lengths cancel, and the well settles
where the two tails balance,

$$d^\ast = \frac{k}{2} + \frac{\eta^2}{k}\ln\frac{w_{\mathrm{nl}}}{w_{\mathrm{p}}},$$

half the test-evoked displacement k below the line at equal weights, plus a logarithmic correction. The
networks obey it: with k measured on paired trials and d from the delay, d/(k/2) = 1.01 ± 0.03 for the
probability cost and 1.08 ± 0.06 for its cross-entropy over four seeds each, against 0.85 ± 0.13 for
networks trained on a softplus at the wrong scale, which optimize a different model [to be re-stated on the
eight networks of the final recipe]. The same likelihood converts any geometry to behavior, per-step false
alarms Φ(−d) at the well and hits Φ(k − d) at the test, which is the model's psychometric function and the
formula behind Fig. 5f–h; it gives the optimum under the animals' any-crossing criterion, the same balance
with the window lengths in place of the weights, which is why that optimum lies deeper than the trained well
by ln(T_NoGo/T_test)/k; it drives the carrier of the symmetry break, the mean of the choice readout, downward
at a rate set by the hazard at the well, fast near the line and vanishing once the well is deep; it makes the
objective convex in the height of the well, so the four-well configuration of the whole-group tie is a
DPA-stage object and not a dual-stage optimum; and its curvature at the well, the Fisher information of the
lick model, is the stiffness with which training holds the well against a displacement such as the
optogenetic one of Fig. 6. A control that moves the wells by a parameter change instead, a mean added to n₁,
shifts the readout itself together with the field, so every response moves with the well and both tasks
collapse on either side of the trained position (Extended Data Fig. 17); only a drive confined to the
delay isolates the position of the well.

## Methods

### Network

A rank-2 recurrent network of N = 1024 units, $W_{\mathrm{rec}} = m\,n^{\mathrm{T}}/N$ with $m, n \in
\mathbb{R}^{N\times 2}$, input weights $W_{\mathrm{in}}$ with a per-unit bias $b$, gain $g = 1$ and a
Gaussian-CDF transfer function $\varphi(u) = \tfrac12[1 + \mathrm{erf}(u/\sqrt2)]$ (a leaky
integrate-and-fire rate function). The state collapses onto the two overlaps $\kappa_j = n_j^{\mathrm{T}}
r/N$ (eq. 1), κ₀ the sample and κ₁ the choice coordinate; the lick is the sign of κ₁ in the response window.
A fixed point under a constant input $x$ is a zero of the planar field

$$F(\kappa; x) = \tfrac{1}{N}\, n^{\mathrm{T}}\varphi\big(g(m\kappa + W_{\mathrm{in}}x + b)\big) - \kappa, \tag{1}$$

which is the field drawn in Fig. 5c (lower row) and Extended Data Fig. 18c, and its stable zeros the wells. The initialization draws $m$ and $n$
as one isotropic Gaussian population with $\langle n\rangle = 0$, $b = 0$, and overlap λ = 7 on both
modes (both modes bistable on their own). Input noise is isotropic across channels; η denotes the s.d. of
the state it induces (0.37).

### Tasks and curriculum

Trials follow Fig. 1a with dt = 15 ms: DPA (sample 2–3 s, test 8–9 s, 11 s trials), Go/NoGo (odor 2–3 s,
cue 4–4.5 s), dual (sample, Go/NoGo odor, cue, test at 2, 4, 6, 8 s). Each network was trained on DPA
(250 epochs), then Go/NoGo (100), then the dual task (150), with the sample mode and the sample inputs
frozen during Go/NoGo and all inputs and the bias frozen during the dual stage. Every event the animal is
scored on is priced by the likelihood of a probit lick model with the network's own noise as its scale: a
lick at step t is the event that the noisy readout crosses the line, P(lick) = Φ(κ₁(t)/η), and the cost of a
required lick is −log Φ(κ₁/η), of a required no-lick −log Φ(−κ₁/η) (eq. 6). A lick is required in the last
0.5 s of the test on paired trials and in the last 0.5 s of the cue on Go trials; a no-lick is required in
the last 0.5 s of the test on unpaired trials, on NoGo trials from cue onset to the test, and throughout the
delay of the dual trials that carry no Go/NoGo odor, which sit on the memory wells. The internal requirements
keep quadratic forms: the sample is held on κ₀ for 0.5 s after its offset, the Go/NoGo rule on κ₁ for 0.2 s
before the cue, and the state is pinned at zero before the sample. Nothing else constrains κ₁, in particular
nothing during the DPA delay. Adam, learning rate 0.01, batch 516 trials, eight seeds. [Networks of the
earlier hinge-and-softplus objective are kept for comparison: Extended Data Fig. 20.]

### Symmetries and ties

A relabeling σ of a task is a permutation $S_\sigma$ of the input channels with a plane action
$D_\sigma$; for DPA, $D_1 = \mathrm{diag}(-1,+1)$ with $S_1 = (A\,B)(C\,D)$, $D_2 = -I$ with $S_2 =
(A\,B)$, $D_3 = \mathrm{diag}(+1,-1)$ with $S_3 = (C\,D)$. A network is tied to σ when a permutation ρ
of its units satisfies

$$\rho\, m = m D_\sigma, \qquad \rho\, n = n D_\sigma, \qquad \rho\, W_{\mathrm{in}} = W_{\mathrm{in}}
S_\sigma, \qquad \rho\, b = b, \tag{2}$$

and then, for every input and every state,

$$F(D_\sigma\kappa;\, S_\sigma x) = D_\sigma\, F(\kappa;\, x). \tag{3}$$

Equation (3) makes the attractor set closed under $D_\sigma$, the noise-averaged trajectory of a
relabeled trial the image of the original, and a line fixed by $D_\sigma$ invariant under every input the
relabeling fixes; the forced-zero overlaps follow from relabeling the sum over units, $n_j^{\mathrm{T}}
w_c = \chi_j(\sigma)\, n_j^{\mathrm{T}} w_{S_\sigma c}$, where $\chi_j(\sigma) = \pm1$ is the sign
$D_\sigma$ puts on mode $j$ (eq. 4). Tying is implemented by splitting the units into blocks, one per
group element, building the initialization by copying a prototype block with the signs and channel swaps
of eq. (2), and projecting the parameters back onto the subspace of eq. (2) after every optimizer step by
a signed average over the blocks, an orthogonal projection; the tie is exact to floating-point precision
throughout training (verified for every element on random networks with random inputs, residual ≤ 5 ×
10⁻⁸). The scaffold networks were tied during the DPA stage only. The equivariance residual of a trained
network, $v_\sigma = \|F(D_\sigma\kappa) - D_\sigma F(\kappa)\| / \|F\|$ over the disk |κ| ≤ 1.5, measures
how far its autonomous field is from respecting σ.

### The dual cost as a breaking field

With $W$ the steps where a lick is wrong and $H(x) = \varphi(x)/\Phi(-x)$ the Gaussian hazard,

$$L_{\mathrm{nl}} = \Big\langle -\log\Phi\big(-\kappa_1(t)/\eta\big)\Big\rangle_{t\in W}, \qquad
\frac{\partial L_{\mathrm{nl}}}{\partial\kappa_1} = \frac{1}{\eta}\,H\big(\kappa_1/\eta\big) > 0, \tag{6}$$

is invariant under κ₁ → −κ₁ for no window, so every element with a minus sign on κ₁ is broken by the dual
objective and its gradient is one-signed; the gradient decays like the Gaussian tail below the line, which is
what sets the depth of the wells (Supplementary Note §13). For a transfer function that is a constant plus an odd function,
the field's even part is $\Psi(\kappa) + \Psi(-\kappa) = 2c\langle n\rangle + \tfrac1N\sum_i n_i
[\psi(u_i + \beta_i) - \psi(u_i - \beta_i)]$ (eq. 5), so with the bias frozen the dual cost can break the
inversion only through $\langle n_1\rangle$.

### Readouts

A lick is scored when κ₁ crosses zero at any step of the response window (the last 0.5 s of the cue for the
Go/NoGo response and of the test for the pairing response, or the 0.5 s after each stimulus in the post-stimulus
configuration), the criterion used for the animals, where any lick in the window counts. The training sweeps'
own accuracies (Extended Data Figs 13–17) use the window mean of κ₁ instead, the criterion of the training code;
the two agree for DPA and Go, whose states sit 2 to 3 η from the line throughout the window, and differ for NoGo,
where the cue drives a transient of 0.4 to 0.9 η above the line that the window mean averages away (Results).
Every NoGo number in the main text and in Fig. 5 uses the any-time criterion unless stated otherwise.

Wells are the stable zeros of the noise-averaged field found from 41 seeds on the disk |κ| ≤ 2.5 and
classified by their Jacobian; the memory wells are the attractors with |κ₀| > 0.5 nearest the line on each
side, reported in units of η. Overlaps $n_j^{\mathrm{T}} w_c/N$ are computed from the trained parameters.
Populations are read in the (m₀, m₁) plane of the write vectors; the four mixed populations are the sign
quadrants of units with |m₀| > 1.5, the pure-choice pair the units with |m₀| < 0.8 and |m₁| > 2.5.

### Perturbation of the well position

To move the memory wells along the choice axis without retraining, a constant per-unit drive δ·m̂₁ (m̂₁ the
choice write vector scaled to unit r.m.s.) was added to the input current of every unit on every trial, from
sample offset until the onset of the stimulus that triggers the response: the test on DPA-only trials, the
cue on dual trials. The drive is off while the response is formed, so the trained field, including its
test-driven part, acts unchanged from a displaced starting point. δ took 81 values from −0.6 to 1.0. For
each network and δ, 2048 DPA-only trials and 2048 dual trials were simulated under the trained noise. The
well location of each sample is the mean κ₁ (in units of η) of that sample's trials at the last step of the
drive; DPA accuracy is scored per sample on DPA-only trials and Go/NoGo accuracy per sample on dual trials,
in the response windows of the training configuration and with the any-time lick criterion (Readouts); the
dual performance is the product of the two
accuracies of the same network, sample and δ, the probability that both responses of a dual trial are correct
if they fail independently. The control adds δ/2 to every entry of n₁ instead
(nine values from −1.2 to 1.2), which moves the wells but also the field and the readout κ₁ = n₁ᵀr/N
itself. Pooled points (network × sample × δ) are binned in 0.5 η steps of the well location, with a bootstrap
95% CI per bin and a Spearman correlation over all points.

### Statistics

The unit of replication is the network (seed). Exact predictions (residuals, forced-zero overlaps, orbit
structure) are reported as the count of networks in which they hold to three decimals; soft ones (heights,
counts of wells below the line) as ranges and counts over seeds. No hypothesis test is used on eight
seeds.

## Extended: figure legends, definitions and derivations

### Figure legends

Figure 5 | A rank-2 network trained on the animals' curriculum reproduces the no-lick repositioning, and
the symmetry of the tasks says why it happens only in the dual stage and where it stops.

a, The model. Three input channels (sample A/B, Go/NoGo odor and cue, test C/D) drive a recurrent population of
N = 1024 units whose connectivity is rank two, W_rec = (m₀n₀ᵀ + m₁n₁ᵀ)/N, shown as the trained matrix of one network
(40 units, ordered by population) and as its two outer products; the state is the pair of overlaps κ_j = n_jᵀr/N, a
sample coordinate and a choice coordinate, and the lick is the sign of κ₁ at the readout.

b, The curriculum. The three stages as trial timelines (sample A/B, indigo and teal; Go/NoGo odor, blue and green;
cue, gray; test C/D, gray; hatched red, the response window in which the lick is read as the sign of κ₁), with what
each stage trains and freezes; eight networks from eight random initializations.

c, The DPA task and its symmetries. Left, the task table (lick iff the pair matches) and the three relabelings that
leave the objective unchanged; with the identity they form the Klein four-group V. Upper row, each element and the
whole group on the plane: its matrix D, a state (black) and its image(s) (open; dotted, the map), and the pair of
memory wells (A, indigo; B, teal) it allows a symmetric network to have: a mirror pair at one height under σ₁, an
antipodal pair under σ₂, a pair on the sample axis at unrelated distances under σ₃, a pair pinned on the lick line
(or the four-well configuration, faint) under the whole group. Lower row, the autonomous flow at the DPA checkpoint
of a network trained with that element (or the whole group) held exactly (arrows, the field of eq. 1; background,
speed, dark is slow; dashed, the lick line), with the memory wells of every network trained under the same tie.

d, The group across the curriculum. Left, which elements each stage keeps: all four hold in DPA (σ₃ and the whole
group pin the wells on the lick line); the one-sided Go/NoGo objective breaks σ₂ and σ₃, which flip the response;
the no-lick cost of the dual stage is a downward field, and σ₁ makes both wells move as one. Middle, the overlaps of
the choice readout n₁ (filled) and the sample readout n₀ (open) with the Go (blue) and NoGo (green) input columns at
each checkpoint, one line per network, thick the median: σ₂ and σ₃ force the choice overlaps to zero, σ₁ the sample
ones. Right, the equivariance residual of the autonomous field for each element (‖F(Dκ) − DF(κ)‖/‖F‖ over the disk
|κ| ≤ 1.5), one line per network, thick the median: σ₂ and σ₃ are broken in the dual stage, σ₁ is kept. Lower row, under the three columns of the account, the autonomous flow of one free
network at the corresponding checkpoint (arrows, the field of eq. 1; background, speed, dark is slow; dashed, the
lick line), with the memory wells of all eight networks overlaid (A, indigo; B, teal): pinned on the line after
DPA, unmoved after GNG, both below after Dual.

e, The push. Memory-well height across the curriculum in units of the noise s.d. η, one line per network (A
filled, B open; colour, network), with the mean ± 95% bootstrap CI (black); Wilcoxon signed-rank test on the
DPA → Dual change per sample (networks with the well at both checkpoints). Both wells end below the line in 8/8.

f–h, Performance against the position of the well, moved by a drive along m₁ applied from the end of the sample
until the stimulus that triggers the response arrives (the test on DPA-only trials, the cue on dual trials),
which displaces the state and leaves the trained field unchanged (Methods). A lick is any crossing of the line
in the response window, as for the animals. Gray, every point (8 networks × 2 wells × 81 drive strengths; A
filled, B open); coloured, mean ± 95% bootstrap CI in bins of 0.5 η of the well location. f, DPA accuracy on
DPA-only trials against the well location of the sample when the test arrives: deeper wells cost misses, wells
far above the line cost false alarms. g, Go/NoGo accuracy on dual trials against the well location the cue
finds: the NoGo response appears only for wells 2 η or more below the line, where Go begins to fail. h, Dual
performance, the probability that both responses of a dual trial are correct (the product of f and g for the
same network, sample and drive), against the well location at the cue; gray dotted, the trained position, the
optimum of the training objective but not of the any-time criterion. Two signatures of the group: the A and B
points coincide in f–h because the surviving σ₁ maps one well onto the other; and the curves are not
mirror-symmetric about the lick line (red dotted, the σ₃ image of the dual curve), because the no-lick cost
broke σ₃.

Extended Data Fig. 11 | The four-group of DPA and what it predicts for the wells and the populations. The task and its relabelings; the actions on the plane and on the units; per tie, the wells the autonomous flow may have and the population orbits.

Extended Data Fig. 12 | The predictions for the overlaps, the inputs, and the rest. Per tie, the overlaps of the read vectors with every input column and with the other mode as predicted scatters, the input combinations against the modes (an antipode keeps a correlation, a mirror image forces it to zero), and the table of the remaining predictions (trial trajectories, invariant lines, image wells, behavior, single units).

Extended Data Fig. 13 | The tied networks at the DPA checkpoint, and the population orbits. Top, the autonomous field and the (m₀, m₁) populations of a network tied to each element or to the whole group, with the attractors of the other seeds, the residuals and the counts. Bottom, the orbit types of V in the unit plane, the population counts and the angular harmonics they allow, against a whole-group-tied, a free and an initial ensemble.

Extended Data Fig. 14 | The Go/NoGo stage. Top, the relabeling τ and the one-sided objective that breaks it; which elements of V survive the rule (the deafness lemma); per element, the fields under Go and NoGo input, what the stage can change and the lattice images. Bottom, the autonomous field and the populations after the stage for a released σ₁, a tied τ on Go/NoGo alone, a released σ₃, and the free network.

Extended Data Fig. 15 | The dual stage. Top, two responses on one lick axis leave Z₂ = {e, σ₁}; the no-lick cost as a downward field; per element, the flow after Dual, the populations and what the stage can change. Bottom, the released networks of every scaffold and of the free recipe after the whole curriculum.

Extended Data Fig. 16 | The same machinery on two simpler tasks. Top, Go/NoGo alone with a two-sided and a one-sided objective, free and tied to its Z₂. Bottom, a delayed two-alternative choice: ties to each plane representation, a tie to a non-symmetry trains to chance, a whole-group tie holds the memory on an invariant axis, −I permits a rotation.

Extended Data Fig. 17 | The perturbation in full. Top, the delay-only drive along m₁: DPA accuracy per sample, hit and false-alarm rates, and Go and NoGo accuracy against the well location, per network. Bottom, the control that adds a mean to n₁, which moves the field and the readout with the well: both tasks collapse on either side of the trained position.

Extended Data Fig. 18 | The free networks across the curriculum. a, Lick probability by trial type at the three
checkpoints, from simulated trials under the trained noise (mean ± SEM over 8 networks): at the test on paired
(filled) and unpaired (open) trials in the DPA-only (red), Go (blue) and NoGo (green) contexts, and at the cue on Go
and NoGo trials (dashed); after the Go/NoGo stage, before any dual training, the networks lick at the NoGo cue on a
large fraction of trials, the interference the dual stage removes. b, Units of a whole-group-tied network at the DPA
checkpoint in the plane of their write vectors: six populations, four carrying a sample sign and a choice sign
(colour, sample sign; opaque, choice +) and a pure-choice pair on the axis (gray), the two orbit types the group
allows. c, The autonomous flow of one network at the three checkpoints (arrows, the field of eq. 1; background,
speed; dashed, the lick line), with the memory wells of all eight networks overlaid (A, indigo; B, teal).

Extended Data Fig. 19 | The no-lick cost is the push. The autonomous flow after the dual stage, one panel per network,
for the free recipe with the no-lick term (reference, top row) and for three sets trained without any no-lick
term in the dual stage, everything else unchanged: free, tied to σ₁ during DPA and released, and tied to σ₁
through all three stages. First column, the wells the theory allows in each case; orange, the wells found; dashed,
the lick line. Without the cost every network ends with both wells above the line; with the tie held they rise at
exactly one height.

### References

- **[Beiran 2021]** Beiran, M., Dubreuil, A., Valente, A., Mastrogiuseppe, F. & Ostojic, S. Shaping dynamics with multiple populations in low-rank recurrent networks. *Neural Comput.* **33**, 1572–1615 (2021).
- **[Bressloff 2001]** Bressloff, P. C., Cowan, J. D., Golubitsky, M., Thomas, P. J. & Wiener, M. C. Geometric visual hallucinations, Euclidean symmetry and the functional architecture of striate cortex. *Phil. Trans. R. Soc. B* **356**, 299–330 (2001).
- **[Dubreuil 2022]** Dubreuil, A., Valente, A., Beiran, M., Mastrogiuseppe, F. & Ostojic, S. The role of population structure in computations through neural dynamics. *Nat. Neurosci.* **25**, 783–794 (2022).
- **[Golubitsky 2002]** Golubitsky, M. & Stewart, I. *The Symmetry Perspective* (Birkhäuser, 2002).
- **[Hirokawa 2019]** Hirokawa, J., Vaughan, A., Masset, P., Ott, T. & Kepecs, A. Frontal cortex neuron types categorically encode single decision variables. *Nature* **576**, 446–451 (2019).
- **[Kunin 2021]** Kunin, D., Sagastuy-Brena, J., Ganguli, S., Yamins, D. L. K. & Tanaka, H. Neural mechanics: symmetry and broken conservation laws in deep learning dynamics. *ICLR* (2021).
- **[Machens 2005]** Machens, C. K., Romo, R. & Brody, C. D. Flexible control of mutual inhibition: a neural model of two-interval discrimination. *Science* **307**, 1121–1124 (2005).
- **[Mastrogiuseppe 2018]** Mastrogiuseppe, F. & Ostojic, S. Linking connectivity, dynamics, and computations in low-rank recurrent neural networks. *Neuron* **99**, 609–623 (2018).
- **[Raposo 2014]** Raposo, D., Kaufman, M. T. & Churchland, A. K. A category-free neural population supports evolving demands during decision-making. *Nat. Neurosci.* **17**, 1784–1792 (2014).
- **[Rigotti 2013]** Rigotti, M., Barak, O., Warden, M. R., Wang, X.-J., Daw, N. D., Miller, E. K. & Fusi, S. The importance of mixed selectivity in complex cognitive tasks. *Nature* **497**, 585–590 (2013).
- **[Salinas 2000]** Salinas, E. & Thier, P. Gain modulation: a major computational principle of the central nervous system. *Neuron* **27**, 15–21 (2000).
- **[Schuessler 2020]** Schuessler, F., Dubreuil, A., Mastrogiuseppe, F., Ostojic, S. & Barak, O. The interplay between randomness and structure during learning in RNNs. *NeurIPS* **33** (2020).
- **[Sussillo 2013]** Sussillo, D. & Barak, O. Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. *Neural Comput.* **25**, 626–649 (2013).
- **[Wang 2002]** Wang, X.-J. Probabilistic decision making by slow reverberation in cortical circuits. *Neuron* **36**, 955–968 (2002).

### Supplementary Note: definitions and derivations

> The full derivations (group-theoretic definitions, the equivariance theorem, orbits, the even part, the deafness lemma, the wells of every tie) are inserted below from the companion document at build time.

<!-- DERIVATIONS -->

## Working appendix

> Working appendix. The Extended Data figures above are assembled from the source note's figures (theory rows and
> tied/free simulations); their house-style renders are in `figures/paper_share/modelling/`. Pending: the σ₁/V
> fixed-code rerun (the ties, Extended Data Fig. 15), the post-stimulus scoring sweep (NoGo numbers), ED 11–16 renders.
