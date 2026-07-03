# dPCA no-lick push — factorization diagnostics (2026-07-03 session)

Working notes from a session probing the story-figure **section 4** ("learning pushes the memory into
no-lick", tasks axis) after the user observed it looked like a push **in Naive too**, and asked which
dPCA construction reproduces the **linear decoders'** result of *no push in Naive*.

> ⚠ **READ `[[project_pca_flows]]` "No-lick push arc — SETTLED CONCLUSIONS" FIRST.** Some findings below
> re-derive things that prior work already diagnosed and (argues) resolved. This note is an **open thread /
> tension log**, NOT a retraction of the settled push. Do not act on "section 4 is an artifact" without the
> reconciliation in the last section.

## RESOLVED 2026-07-03 — the push survives CI removal; my "time artifact" concern was WRONG.

Reproduced section 4's per-mouse push (`load_mouse`/`depth_of`, tasks axis, per-mouse Expert-anchored sign)
across CI-removal q=0/1/2 on the existing `f-sample-test-tasks_dpca_<mouse>[_ci{q}]` DUMs:

| q | Naive | Expert | push E−N | p(Wilcoxon) | deepen |
|---|---|---|---|---|---|
| 0 | −1.30 | −1.88 | −0.59 | 0.012 | 8/9 |
| 1 | −1.02 | −1.60 | −0.58 | 0.008 | 8/9 |
| 2 | −1.00 | −1.60 | −0.61 | 0.012 | 8/9 |

**The deepening is essentially unchanged by removing the time ramp** (−0.59 → −0.61, still p≤.012, 8/9).
**Why my session's concern was wrong:** the `time` ramp is **condition-independent** → identical Naive/Expert
→ it **cancels in the deepening (E−N)**. The 0.55 tasks↔time cosine inflates the *absolute* tasks position
(−1.3 vs a smaller true value) but NOT the *learned change*, which is what section 4 claims. My "crude
trajectory time-regression halves the push" (item 1) over-subtracted because tasks and time are *shape*-
correlated — not because the deepening is temporal. **Section 4's push is real; keep it.** "Naive at 0" is a
display anchor, not a claim — Naive is genuinely at −1.0 even after CI removal. The items below stand as a
record of the exploration but their pessimistic reading is superseded by this table.

## What this session found

1. **The `tasks` axis is ~half a `time` axis.** |cos(tasks-decoder, time-decoder)| = **0.55** in *both*
   stages (0.553 Naive / 0.549 Expert); sample/test/choice are all ≤0.14 with time. A shared regression:
   the DPA `tasks`-axis trajectory is **57% explained by the time-axis trajectory** (`task ≈ 0.39·time`).
   Crude time-removal (regress out the time-component trajectory) roughly **halves the LATE push**
   (E−N −0.53 → −0.24) and flips the *memory-delay* residual positive (no no-lick there).

2. **Section 4 is not per-stage.** `load_st()` loads the **Expert** DUM and projects *both* stages onto the
   **Expert** tasks axis, then subtracts `dN` (Naive-anchoring). So Naive is read through Expert's axis and
   re-anchored — a common-frame projection. (Section 2's trajectory grid *is* per-stage.)

3. **Timing of the DPA `tasks` dip is post-cue, not memory-delay.** Per-stage z-scored, DPA sits ≈0 through
   sample+delay (−0.06 at 3–4.5 s) and only ramps to −0.5 after ~6 s (cue/reward), monotonically → matches
   a time ramp. Naive ≈ Expert at every window in the per-stage frame.

4. **New factorization `f-sample-distractor-test` (built this session).** Cleanly quarantines the ramp into
   the **`time`** marginal (DPA time-axis −1.36 LATE / −3.06 test, **identical** Naive/Expert). **No push on
   any stimulus axis**; `sample:test` (match) is balanced. Confirms the ramp is what leaked into `tasks`.

5. **New factorization `f-sample-test-choice` fit on ALL trials (`--fit-all-trials`, built this session).**
   The `choice` (response) marginal — the direct decoder analog — shows **no no-lick push / if anything
   reversed** (DPA choice-axis: Naive −0.28 → Expert +0.07 at memory). **Error-confounded**: the axis is fit
   including errors, and Naive makes far more errors than Expert, so the two stages' choice axes are built
   from different error structure. Not a clean test.

6. Per-stage, per-mouse `sample:test` (choice) push at LATE: mean E−N −0.05, **Wilcoxon p=0.73** (null).

## Tension with the SETTLED conclusions (must reconcile before acting)

The prior settled work (`[[project_pca_flows]]`, `docs/pca/flows_handoff.md`) already addressed much of this,
and its conclusions partly **conflict** with a naive reading of the above:

- **"No push in Naive" is a DOCUMENTED RETRACTED ARTIFACT.** Settled note (3): the "Naive≈0 / choice-axis
  polarization" was a circular-orientation + future-test-leakage artifact, and the pooled "Naive→0 after CI
  removal" was **cross-mouse sign cancellation** (JawsM18 +4.4 vs ChRM04 −4.0). "Naive wells at zero is FALSE
  for the tasks push — already −1.3 in Naive; learning DEEPENS an already-present push." → My items 5–6
  ("no push in Naive" on choice/per-stage) are **consistent with that artifact**, not a new result.
- **Per-stage own-basis failure = axis-sign noise, not absence.** Settled ladder (1a): own per-mouse refit
  per stage gives Naive −0.35→Expert −0.63, **p=0.73** — the *same* null I got — and prior work attributes
  it to sign noise from weak-lick mice (ChRM23/ACCM03/ACCM04 have ~0 Go−NoGo separation → random axis sign),
  NOT to absence of the effect. This is *why* the canonical figure uses a shared/pooled axis.
- **The tasks push reportedly SURVIVES CI (time) removal.** Settled note: robust to CI removal q0/q1/q2
  (Δ≈−0.6, p≤.012) and confirmed in raw ΔF/F (r≈0.997). **I did NOT run the proper `--remove-ci q1/q2` test
  this session** — my item 1 is a *crude trajectory-level* time regression, weaker evidence. The
  `_ci{1,2}_dpca` DUMs already exist (per settled notes); the honest next step is to re-measure the push on
  those, not to conclude "it's the ramp" from the 0.55 cosine alone.

## Open question to resolve next

Which axis do the **linear decoders** (overlaps CCGD) use for the "no push in Naive" the user cites, and how
does it square with the settled **tasks**-axis push (which has Naive already at −1.3, deepening)? The `choice`
CCGD decodes the animal's main-test lick response (hit/FA=lick, CR/miss=no-lick — `src/common/get_data.py`),
≈ `sample:test` on correct trials. Settled note (2) says the *choice*-axis delay push is ≈0 in Naive AND
Expert (action-axis-specific), so "no push in Naive on the choice axis" may be *expected and not the headline*.
Reconcile the decoder's axis + window against settled (1)/(2) before reworking section 4.

## Assets built this session (reusable)

- Code: `cv_dpca(..., perf_filter=True)` in `src/pca/dpca.py` (set `False` to fit on all trials);
  `run_pseudo.py --fit-all-trials` (tags DUM `_fitall`).
- DUMs in `../data/pca/`: `pseudo_*_ALL_{Naive,Expert}_..._f-sample-distractor-test_dpca` and
  `..._f-sample-test-choice_dpca_fitall` (both stages, `pseudo_{traj,labels,marglabels,weights,evr}_`).
