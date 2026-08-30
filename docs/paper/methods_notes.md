# Methods notes — staged paragraphs for the manuscript

> Paper-ready Methods text, accumulated per figure as analyses are finalised. Companion to
> `results_draft.md`. Started 2026-08-10 with the Fig 2 (dimensionality) block; add sections for other
> figures as their methods get locked.

## Dimensionality of the pseudo-population (Fig. 2)

**Cross-validated PCA (cvPCA).** We estimated the reliable dimensionality of the pseudo-population
(3,319 neurons; 12 conditions = 3 tasks × 2 samples × 2 test odours; correct, laser-off trials) with
cross-validated PCA (Stringer et al., 2019). For each of 30 random halvings, the trials of every
(mouse, condition) pool were split into two disjoint halves, yielding two independent condition-mean
pseudo-populations (neurons partition disjointly across the 9 mice, so the split is independent per
mouse). A PCA basis was fit on one half and the variance of the other half was evaluated by
cross-projection; the two directions were averaged. In expectation, trial-to-trial noise averages to
zero in this cross-term and only variance that replicates across independent halves — signal — is
retained. This reliable variance is distinct from ordinary explained variance, which is evaluated in
the same data that defined the components and therefore includes sampling noise absorbed by the
basis; on our data the two disagree qualitatively — the naive condition-mean scree is nearly
identical for the one-dimensional mid-delay state and the three-dimensional decision state, whereas
the reliable spectra separate them, and only ~9–24% of the condition-mean variance replicates.
Because the PCA basis is itself fit on one noisy half, per-component cvPCA values estimate the signal
variance along the empirical axes rather than the true signal eigenvalues: the total reliable
variance is unbiased, but the top of the spectrum is deflated by basis misalignment and the tail
correspondingly inflated (Pospisil & Pillow, PNAS, "Revisiting the high-dimensional geometry of
population responses in visual cortex"). This bias flattens the spectrum and thus inflates the
participation ratio — conservative for the low-dimensionality claims made here — and the coarse
contrasts we report (PR ≈ 1 vs 2 vs 3, with order-of-magnitude spectral gaps) do not require the
debiased spectral-shape estimators developed for power-law-exponent estimation. Pure-noise components
can come out slightly negative (unlike explained variance, which is non-negative by construction);
fractions and the PR use the positive-clipped spectrum. (The factor-contrast decomposition used for
per-variable reliable variances applies the same cross-product to fixed design contrasts; with no
fitted basis it is free of this bias, which is why its noise estimates dip symmetrically around
zero.) All of these estimators operate on condition means: they characterise the task-conditioned
state geometry, and are blind to trial-to-trial (within-condition) dimensions — the dimensionality
claims are about the condition/state geometry, not the single-trial state space. Neurons were
z-scored by a stage-level, condition-agnostic standard deviation before
averaging. We used repeated split-half CV (30 random halvings, both directions averaged) rather than
k > 2 folds because the estimator is a cross-product of two independent condition-mean estimates,
whose variance is minimised by equal halves (n₁ = n₂ = n/2); k-fold variants use noisier per-fold
means and additionally deflate the spectrum through basis misalignment. Effective dimensionality was
summarised as the participation ratio, PR = (Σλ)²/Σλ², over the positive reliable spectrum
(Extended Data; the main figure reports the spectra themselves). The null
shuffled condition labels within mouse (preserving trial counts and noise structure) and is compared
on reliable variance, not on PR (the PR of a near-zero noise spectrum is undefined-ly large); the null
spectrum shown in Fig. 2b is normalised by the real spectrum's positive total (its own total is near
zero, so a self-normalised null would be meaningless). Windows:
mid-delay (bins 36–38, data 5.5–6.3 s: it opens at distractor offset (4.5–5.5 s) and closes before
the Go/NoGo response cue (6.5–7 s) and its reward (7–7.5 s), so no Go/NoGo lick has yet occurred —
the clean maintenance window; used throughout Fig. 2b–d; the distractor odour's sensory tail may
contribute at its start) and decision (bins 57–65, fully post-test-onset); legacy analyses (the Extended-Data PR build) additionally use late delay
(bins 48–53; a 0.5-s trailing-window convention, data 7.5–8.8 s, closing 0.17 s before test onset). The time-resolved (trajectory) dimensionality
is not reported because its shuffle null retains ~half the variance through the condition-independent
time ramp.

**Error bars on the spectra (Fig. 2b) and on the PR (Extended Data).** 95% confidence intervals are
from a leave-one-mouse-out jackknife: the averaged reliable spectrum (per-component fractions) or its
PR was recomputed nine times, excluding each mouse's neurons and trials in turn;
CI = value ± 1.96 × SE_jackknife (fractions clipped to [0, 1]; the PR lower bound clipped at its floor
of 1). Mice, not trial
splits, are the exchangeable unit (neurons partition by mouse); split-level percentiles quantify only
trial-split stability and are anti-conservative as population error bars. The "unchanged with
learning" statement is quantified with the same jackknife applied to Δ(Naïve − Expert): the 95% CI
spans zero for every reliable-spectrum component (|Δ fraction| ≤ 0.22) and every decodable variable
(|Δ accuracy| ≤ 0.19) at both windows. We report this as absence of detectable change at the stated
precision (CI half-widths 0.03–0.30), not as strict equivalence.

**Per-variable decoding power (Fig. 2c).** For each condition set (DPA: 4 conditions; dual: 8) and
window, each task variable's demixed axis was computed by applying its orthogonal design contrast to
condition means estimated from a training half of the trials (with binary factors and window-averaged
states, each dPCA marginalisation is rank-1, so this is the demixed component; Kobak et al., 2016).
Single held-out pseudo-trials (one random test-half trial per mouse per pseudo-trial, 10 per
condition) were projected onto the axis and classified by the training-set class midpoint; performance
is balanced accuracy, averaged over 15 train/test splits. A variable counts as "in play" when its
accuracy exceeds the 95th percentile of a within-mouse label-shuffle null (100 shuffles, full pipeline
rerun per shuffle; each window is tested against its own null, uncorrected — the pre-test windows are
strongly correlated). The null draws are single-split accuracies while the tested statistic is a
15-split average, so the threshold is conservative (the single-split null has the larger variance). On correct trials the decision-window "choice" is confounded with action and
outcome (match ≡ lick ≡ reward); the lick-specific dissociation of the action code is provided by the
per-mouse decoder analyses of Fig. 3. This decoding count is amplitude-free — a compact axis and a dominant axis face
the same criterion — and complements the variance-weighted spectra: it involves no basis fit on noisy
means, so it is immune to the cvPCA spectral bias discussed above. Because DPA trials contain no
distractor, the "gng ×" entry is a cross-decode: Go-vs-NoGo was decoded from dual-task pseudo-trials
projected into the DPA-state subspace (the top-3 PCs of the DPA condition means; LDA on the 3-D
projection, disjoint train/test halves, same shuffle null, 100 shuffles) — measuring how much of the distractor
code the DPA geometry itself carries. The per-PC version of the same cross-decode (above-chance
fraction, 2 × (accuracy − 0.5)) forms the boxed gng × column of Fig. 2d.

**What each dimension codes (Fig. 2d).** For each task set (dual: 8 conditions; DPA: 4) and window, we
computed condition-mean PCs (neurons z-scored across condition means) and decomposed each PC's
across-condition variance onto mutually orthogonal factor contrasts — sample, distractor identity
(Go vs NoGo), test, and choice (the sample × test interaction) — as η². The design is balanced, so the
contrasts are orthogonal and the η² per PC are exhaustive — each PC's η² sum to 1, so η² is a *share*
whose chance level is 1/3 per contrast, not 1/2. For a direction carrying no signal the three shares
are the squared coordinates of a uniform point on the sphere, i.e. Beta(½,1) each: median 0.25, and
P(η² ≥ 0.64) = 0.20. Large η² entries on unreliable components are therefore expected and are not
evidence of coding; the corresponding variables' decodability is tested directly in Fig. 2c.
Condition-mean PCs beyond the reliable dimensionality are faded in Fig. 2d; the reliable rank is the
number of leading components whose leave-one-mouse-out jackknife CI stays above 1% of the reliable
variance in Fig. 2b (DPA mid-delay 1, dual mid-delay 2, both decision windows 3). The gng ×
cross-decoding column is exempt from the fading because it is a decoding accuracy, not an η². During the DPA delay their apparent test/choice η²
cannot be anticipatory coding: the test odour is drawn independently of the sample, so no delay-period
signal can carry test identity (or the choice it determines) — and empirically these components fail
cross-validation and their variables decode at chance (Fig. 2c). They are sampling noise of the
condition means (η² across a finite condition set must land on some contrast for any direction,
including noise directions). This does not deny preparatory activity, which this decomposition would
place elsewhere: condition-linked anticipation appears as decodable structure (the naive-stage
dual-delay choice/bias state, Fig. 2c † and Extended Data, removed by learning), postural/motor
preparation appears as the condition-independent position of the delay state on the action axis
(Fig. 4), and purely temporal (timing/ramp) components are shared across conditions and are removed
with the grand mean by construction. PC4 of the 4-condition DPA set is the degenerate null direction of
the centred condition means (~0% variance) and is displayed for symmetry with the dual set.

**One decoder throughout (Figs 2–3).** Every linear readout in these figures — the axes that form the
plot coordinates of Fig. 3a, the axis-geometry cosines of Fig. 3b, the action-code cross-decoding of
Fig. 3c, the cross-task generalisation of Fig. 3d and their per-animal companions — uses the same
pipeline: standardisation, PCA to min(20, n_features, n_samples−1) components, then L2-regularised
logistic regression (C = 1, balanced classes), defined once in `pca/decoders.py` and imported by
every script. Where a decision direction is needed as a geometric axis, it is the pipeline's own
decision vector mapped back into neuron space (V^T·coef, undoing the standardisation) and normalised,
so the decoder and the axis are literally the same vector and cannot disagree. Earlier builds mixed
difference-of-class-means axes for the geometry panels with logistic regression for the decoding
panels; both are linear readouts and the conclusions are unchanged (the axis-overlap matrix moved
from 0.39→0.53 to 0.32→0.46 for action × distractor, with sample orthogonal to both in either case),
but the mixed version invited comparisons between quantities built by different estimators.

**Shattering dimension (Extended Data; cited in Results §3).** All 462 balanced 6-vs-6 dichotomies of
the 12 conditions were decoded at the decision window with a leakage-free pseudo-population decoder:
disjoint train/test trial halves per (mouse, condition), 24 pseudo-trials per condition,
StandardScaler + PCA(30) fit on the training half only, then LDA per dichotomy; performance is balanced
accuracy on the held-out half, averaged over 8 pseudo-population resamples. The shuffle null permutes
condition labels of the pseudo-trials (0.50). The shattering dimension is the mean over dichotomies.

Scripts: `pca/exp_dimensionality.py` (cvPCA, coding, η²), `pca/exp_dimensionality_fits.py`
(per-task-set fits), `pca/exp_dimensionality_ci.py` (full-462 shattering; split-stability check),
`pca/exp_dimensionality_jk.py` (PR jackknife CIs), `pca/exp_dpca_count.py` (per-variable demixed-axis
decoding, Fig. 2c), `pca/exp_cdec_support.py` (spectrum jackknife CIs, Fig. 2b; DPA-subspace gng
cross-decode), `pca/exp_dpa_gng_column.py` (per-PC gng column, Fig. 2d); figure
`pca/fig_dimensionality_main.py` (`--pr` renders the previous PR/all-tasks build for ED). Reference
doc: `docs/pca/dimensionality.md`.
