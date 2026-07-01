# Project: mPFC Population Geometry (Dual Task)

## Paper hypothesis

mPFC encodes sample identity and lick action on near-orthogonal axes. With learning (Naive→Expert), DPA delay-period activity moves along the **lick (choice) axis** — away from the lick sector — without disrupting sample discriminability along the **sample axis**. Distance to the lick boundary predicts within-animal DPA accuracy.

> **dPCA-flows evidence (2026-06-23) — SUPPORTS the hypothesis, via the tasks (action) axis.** The robust, learning-related delay movement is along the **`tasks` axis, which is itself a lick/no-lick (action) axis**: DualGo (lick) sits at +0.96, DualNoGo/DPA (no-lick) at the negative end; the axis tracks the actual lick (DPA lick > no-lick at response) and is **not orthogonal to the choice axis** (decoder cos +0.22, 77°). The DPA WM delay state sits at the no-lick end and is **pushed further into no-lick with learning** — robust (|depth| 1.29→1.88, p=0.012, mouse-bootstrap CI [+0.29,+0.87]; sample-memory axis flat as a control). So this **is** the no-lick push the hypothesis names, measured via the Go↔DPA/NoGo action contrast. Caveats: (i) the **CI/`time`** ramp also deepens but is condition-independent (global, not lick-specific) → keep unbundled; (ii) the trial-by-trial **choice/sample:test** axis shows only a small Expert-specific *lick-readiness* during the delay (predicts whether the animal licks, not the correct choice; significant within-stimulus, marginal under mouse-resampling). See PCA findings below + `docs/pca/flows_handoff.md`.

> **Overlaps weight-cosine evidence (2026-06-24) — SUPPORTS the hypothesis, and supplies the geometric mechanism.** From the decoder **weight vectors in raw neuron space** (`run_overlaps.py --save-weights`): the `sample`, `choice`, `test` axes are **mutually orthogonal at the ±1/√N chance floor at every epoch, both stages**, and each is temporally stable (within-code cosine 0.4–0.9). Crucially, **`sample` and `choice` (memory×action) are mildly anti-aligned in Naive (cos −0.07) and orthogonalize with learning** (|cos| 0.083→0.029; paired Wilcoxon p=0.020, 7/9 mice; reliability-controlled — within-code self-stability unchanged across stages, so not a noise artifact), and *only* that pair (sample–test/choice–test n.s.). This is the mechanism that lets the no-lick push move the delay state along the choice axis **without disrupting the sample axis**: in Naive the axes overlap so a push would drag sample with it; orthogonalization makes the push non-destructive. Convergent with the dPCA push from an independent angle. Honest limit: "orthogonal" = at the chance floor (statistically independent), a small shift, n=9. See `docs/overlaps/overview.md` (Interpretation & synthesis) + `fig_overlaps_cosine.py` / `fig_overlaps_orthogonalization.py`.

---

## Three subprojects

### 1. Overlaps (`/home/leon/dual/overlaps/`)
Cross-generalising decision codes (CCGD): decoders trained on sample identity and lick choice cross-generalise across time. Measures how the sample and choice codes co-evolve during the delay in the sample×choice plane.
- **Results so far (2026-07-01):** (i) sample/choice/test axes **mutually orthogonal + temporally stable** (weight cosines); (ii) **no-lick push** deepens with learning, depth↔DPA accuracy (ρ=−0.67, p=0.05), and its **mechanism = sample×choice orthogonalise with learning** (p=0.020, 7/9 mice); (iii) dual coding **live in the delay** (choice code non-flat, ED/MD); (iv) faithful **bistable autonomous** rank-2 flow (dPCA port). See the consolidated summary at the top of `docs/overlaps/overview.md`.
- See `docs/overlaps/overview.md`, `docs/overlaps/routines.md`, `docs/overlaps/feedback.md`

### 2. PCA / dPCA flows (`/home/leon/dual/pca/`)
Pseudo-population PCA + dPCA and latent flow fields (Figure 2E and follow-ups). Active; docs in `docs/pca/`
(see `docs/pca/flows_handoff.md`).
- Scripts: `pca/run_pseudo.py` (build dPCA bases), `pca/plot_pseudo_flow.py` (autonomous + input-driven flows),
  `pca/fig_dpca_flow_lowrank_shared.py` (rank-2 gain-modulated flows, `--push` adds the no-lick offset).

**Corrected findings (2026-06-23 — supersede earlier mid-thread claims):**
- **ROBUST — the no-lick push:** the `tasks` marginal is a **lick/no-lick (action) axis** (DualGo=lick at
  +0.96; DualNoGo/DPA=no-lick at the negative end; tracks the actual lick; not ⊥ to choice, cos +0.22). The
  DPA WM delay state sits at the no-lick end and is **pushed further into no-lick with learning** — per-mouse
  |depth| Naive 1.29 → Expert 1.88 (Wilcoxon p=0.012, 8/9 mice, sign-free; mouse-bootstrap 95% CI
  [+0.29,+0.87]; **sample-memory axis flat as a specificity control** Δ−0.04). Confirmed in raw ΔF/F (r=0.997,
  not a normalisation artifact). Present in Naive AND grows with training. Large (~1.3–1.9 σ). The flows
  capture it via `--push`. **NB:** the condition-independent `time`/CI ramp also deepens but is global (not
  lick-specific) → report unbundled from the tasks (lick/no-lick) push.
- **WEAK (small, Expert-only):** the **lick (choice / sample:test) axis during the DELAY** shows a small
  **lick-readiness** signal — the delay state predicts WHETHER the animal will lick, controlling for the
  stimulus (within-stimulus permutation p=0.036, per-mouse Wilcoxon p=0.031 / 6-of-7 mice; marginal under
  mouse-resampling bootstrap; ~0.18 σ). It is **motor-preparation bias, NOT anticipation** of the correct
  choice (the choice depends on the future test). Naive: null.
- **RETRACTED artifacts (do not cite):** an apparent large "choice-axis polarization / Naive≈0 / deep
  choice push with learning" was a sign-orientation + future-test-leakage artifact (orienting per-mouse by
  the statistic being tested; not controlling the stimulus). See `flows_handoff.md` for the methodology.

### 3. Decoders (`/home/leon/dual/decode/`)
Single-neuron and population decoders tracking sample and lick axes across learning days. Figure 3 of the paper.
- Script: `decode/fig3BF.py`
- Key outputs: sample axis, lick axis, Δlick per mouse, distance to lick boundary
- Docs to be created when work begins: `docs/decoders/`

---

## How the subprojects relate

All three use the same 9 mice, same recording sessions, same trial structure — see `docs/shared_data.md`.
- **Overlaps** works on `X_single` (CCGD decision values, cross-time generalisation matrix)
- **PCA** works on `traj_all` (per-trial PCA projections)
- **Decoders** projects onto sample/lick axes defined from single-neuron weights

The sample and choice axes in overlaps correspond to the sample and lick axes in decoders — they capture the same geometry from different analysis angles.
