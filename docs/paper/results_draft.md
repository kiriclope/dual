# Compositional learning by geometric editing — main paper (draft v12)

> **v12.23 (2026-09-09): Fig. 4 TRIAL SETS LABELLED IN THE PANELS** (Leon): b "Naive/Expert · DPA trials", c "Δ DPA accuracy, DPA trials" / "Δ GNG accuracy, dual trials", d "Naive unpaired DPA trials", e "choice-code d′, DPA trials". The DPA arm of c was tried on the dual trials and on all trials (Leon's question): identical ranks, ρ = −0.63, p = .067 in both, against −0.72 / .030 on the distractor-free trials; Leon kept the DPA trials and the companion numbers are now stated in §4 and the legend (`--dualperf` draws the dual-trial arm; `exp_push_nogo_coupling.py` prints every slice).

> **v12.22 (2026-09-08): Fig. 4a BACK ON THE DISTRACTOR-FREE CHOICE AXIS (Leon, option 2).** The dual-trial version of Fig. 4a (v12.21) turned out heterogeneous across animals for real (Go/NoGo-balanced lick classes changed nothing: |cos| p = .30, cross-decode p = .25; all trials: .098/.055), so the choice axis for the alignment test is fitted on the DPA trials, where its alignment with the Go/NoGo axis cannot be inherited from distractor-evoked activity in the same trials. Numbers: pooled transfer 0.42 [0.11, 0.64] → 0.50 [0.29, 0.70]; per-mouse |cos| 0.063 → 0.104 (9/9, p = .004); cross-decode 0.53 → 0.60 (7/9, p = .020). The dual-trial result is disclosed in Methods; flags `--dualact` / `--allact` / `--strat` keep the alternatives reproducible. Fig. 3c–e keep their all-trial choice classes.

> **v12.21 (2026-09-08): Fig. 3c–e CHOICE DECODERS ON ALL TRIAL TYPES** (Leon: "we need to compute that on all trials not just dpa"): the per-mouse choice classes (lick vs no-lick at the test) now pool DPA, Go and NoGo trials in the plane ablation (c), the cosines (d, pooled and per mouse) and the per-mouse cross-stage test (e), like the CCGD choice axis; the DPA-only classes are behind `--dpachoice`. Fig. 3c choice brackets are now starred (0.63 → 0.56, p = .012); the test's plane-vs-full comparison is a trend (p = .055); per-mouse sample × choice |cos| 0.06/0.08; per-mouse choice cross-stage T/W 0.59. Fig. 4a's choice side now reads lick vs no-lick on the DUAL trials (Go + NoGo), the same trials as its Go/NoGo side, in both the pooled matrix and the per-mouse scatter (`--dpaact` restores DPA-only). RESULT: the per-animal convergence is now a TREND — |cos| 0.072 → 0.112 and cross-decode 0.56 → 0.60, each up in 6/9 mice, Wilcoxon p = .20 (paired t .12/.10); pooled transfer 0.31 [0.16, 0.51] → 0.47 [0.25, 0.67]. "Rotates onto the choice axis" is softened to "aligns further with" in the Abstract, Introduction, §3, §4, the Fig. 4 legend and the Discussion; the §4 first movement is stated as a trend.

> **v12.20 (2026-09-08): CANONICAL AXIS WINDOWS UNIFIED** (Leon: "same windows for all panels", then "I like from test onset to 0.5 s after test offset"): sample and distractor axes on bins 36–38 (6.0–6.5 s) and choice and test axes on bins 54–62 (9.0–10.5 s) in BOTH pipelines (was overlaps 16–47 / 57–62 / 58–83 / 33–38 and pseudo-population 36–38 / 57–65). Every axis-dependent number in §§2–4, §6, Methods and the legends is re-derived from the rebuilt caches (Figs 2c–g, 3c–e, 4a–e, 6g–i; ED 5; ED 6e–g). Consequences, stats verbatim: push stronger (β = −1.15, p = .007; per-animal p = .055), coupling weaker but starred (ρ = −0.72, p = .030; all six normalizations p ≤ .05; leave-one-out 4/9; permutation .039); Fig. 3c choice out-vs-full is a trend (p = .098); Fig. 3e choice transfer 0.72; NoGo-only arm a positive trend (ρ = +0.65, p = .060); false-alarm arm n.s. (ρ = +0.59, p = .097); out-of-context plane: sample 1.02/0.82, choice 0.77/0.54; Fig. 6 ΔGNG arm stronger (clustered p = .009) and the raw trade-off null. The five-window comparison (A–E) is in memory `project_axis_windows.md`; variant pages A–E remain published.

> **v12.19 (2026-09-08): Fig. 3d keeps ONE per-mouse scatter (sample × choice)**; the sample × dist scatter is gone
> too (Leon). §3 now cites Fig. 3d for the sample × choice cosine and gives the sample × distractor value (0.09)
> as text from the same analysis.

> **v12.18 (2026-09-08): Fig. 3d cosine matrices are 2×2 (sample × choice)** (Leon: "remove dist from the panel d
> matrices"); the per-mouse scatters still show the memory axis against the choice and the distractor axes; the
> 0.32 → 0.47 choice × distractor overlap is quoted in §3 as a pooled number (per-animal test in Fig. 4a). Also
> fixed line-wrapped "Fig. 3e/3f" references that the v12.17 relettering had missed.

> **v12.17 (2026-09-08): FIG. 3 RESTRUCTURED** (Leon): the dual late-delay storyboard column is gone (b is 2×4:
> mid-delay | decision for DPA and dual); the per-mouse plane grid (old d) moves to Extended Data Fig. 6g (own
> render `fig_manifold_permouse_plane.png`); the cosine panel keeps the distractor axis in its matrices but
> drops the choice × dist per-mouse scatter (that learning test stays in Fig. 4a); old e → d, old f → e, stacked
> beside c where the grid was; figure height 14.6 → 12.4 in. References relettered in text, legends, Methods,
> Discussion, ED 6 title. No statistics changed.

> **v12.16 (2026-09-08): DPA cross-decode column OUT of Fig. 2d** (Leon: "remove dist from panel d dpa, and keep the
> cross dec for figure 3"). Fig 2d DPA matrices are sample/test/choice only (`DCROSS = False`); the pooled
> DPA-subspace distractor cross-decode (0.61 at mid-delay, permutation p = .031) now sits in the §3 plane paragraph
> beside the per-mouse Fig 3c result; Fig 3 unchanged; Methods sentence re-pointed.

> **v12.15 (2026-09-08): Fig. 2 e/f MOVED TO THE MID-DELAY / DECISION WINDOWS + panel c in panel-b format + panel g
> coloured by selectivity class + panel e ceilings/PS null → legend** (Leon: "use mid-delay to be consistent with the
> rest of the figure"; "make panel c identical in format to panel b"; "we cannot see the two clouds"; "remove the
> numbers in the test labels, and the PS null"). New caches: `matrices_cache_mddec_acc_nopca.pkl`
> (`fig_ccgp_matrices_pseudo.py --mddec --acc --nopca`) and `PM_GEN_mddec_nopca` (`exp_permouse_frame.py --nopca
> --mddec`); PS was already md/decision. NUMBERS CHANGED: choice transfer 0.73–1.01 (was 0.65–0.99), sample
> 0.27–0.90 asymmetric (was 0.06–0.56), test ≥0.72; panel f: sample Δ 0.00 [−.05,+.05] p=1.00, test +0.03 p=.04,
> choice +0.04 p=.20 — the "all n.s. and bounded ±0.05" equivalence is RETIRED; the claim is now a stable transferred
> FRACTION (medians 0.87–0.99 both stages, all p≥.31). ED 6f re-rendered on the new per-mouse cache.

> **v12.14 (2026-09-08): MATCHED SHUFFLE NULL for Fig. 2c + Fig 2 b/d/e/f edits** (Leon: "why is shuffle accuracy so
> high?" → the drawn null was the 95th pct of SINGLE-SPLIT accuracies on 40/80 held-out pseudo-trials, SD 0.06–0.10,
> while the bar is a 15-split mean; `exp_dpca_count.py` now scores each shuffle as the same 15-split mean → null95
> 0.52–0.55; only verdict change in the drawn cells: naïve dual sample at decision 0.63 now above the null; Methods
> sentence rewritten; "≈0.59" → "≈0.53" in §2 + ED 3g). Also: panel d PC1–3 only; panel b window tags on both
> columns; panel c tag top-right; e/f legends now state their windows (sample LATE delay 8.0–9.0 s, test/choice
> during the test odor 9.5–10.0 s — NOT the mid-delay/decision states; the v12.10 "at mid-delay" wording was wrong).

> **v12.13 (2026-09-08): Fig. 2c DECLUTTERED** (Leon: "panel c has too much information"). Removed from the
> panel: the hatched DPA-subspace distractor cross-decode bar + its orange callout (the 0.61 / p = .031 result
> now lives in panel d's orange column, its legend and the §2 sentence), the naïve null lines (one null mark per
> bar, the expert 95th percentile) and two legend entries. Kept: expert bars, naïve circles, dagger, chance
> line. No statistic changed; legends c/d mirrored here.

> **v12.12 (2026-09-08): Fig. 1g CROSS-STAGE INTERACTION** (Leon: "we need the interaction between naive and
> expert"). One GEE over both stages: lick × stage OR ratio 0.68 [0.45, 1.04], p = .074 (trend; 6/9 mice; p = .16
> with a trial-type covariate); lick × stage × pairing p = 10⁻⁴ (the hit arm rises 3.08 → 4.52, the false-alarm
> arm falls 2.34 → 1.48). §1, the legend, Methods and the in-panel text now say the propagation PERSISTED but
> became pairing-selective; "weaker" is no longer asserted as a tested difference.

> **v12.11 (2026-09-08): Fig. 1g POOLED over Go + NoGo trials** (Leon, on noticing 5/9 expert
lines: "keep the > figure the same but use all trials"). NoGo-only had dropped four expert mice
below the 3-trial minimum. New > stats in §1, the Fig. 1 legend and Methods (naïve OR 2.32 /
expert 1.48, both significant; expert pairing > interaction p = .009 → the false-alarm arm is
attenuated, the hit arm persists; v12.12 adds the CROSS-STAGE tests: lick × stage OR ratio 0.68
[0.45, 1.04] p = .074 trend, 6/9 mice; lick × stage × pairing p = 10⁻⁴ — the selectivity change
is the significant effect, so "weaker" is now "persisted but became selective"). The claim
"propagation no > longer detectable in experts" is RETIRED; downstream wording (intro, §4, Fig.
4 legend, Discussion) now says > "attenuated/weakening", not "disappeared/vanishing".
Trial-type-adjusted ORs added as a robustness line.

> **v12.10 (2026-09-07): REVIEW FIXES, round 2 (outside referee + figure reviewer; verified)** — intro dichotomy made
> operational (construction = new dimensions; editing = positions AND alignments within the set); Fig 2e sample
> transfer stated as partial (0.06–0.56) with the choice at 0.65–0.99, the corrected-PS claim dropped (script +
> legend), r = −0.02; §3 "memory not lost" now rests on ED 6e; distractor plane SHARE 0.19→0.35 (p=.039, 7/9)
> stated; orthogonality quoted against the random-direction floor; per-animal cross-stage ratios 0.86/0.73;
> NEW `overlaps/exp_push_sdt_coupling.py`: the coupling is carried by the fall in false alarms (ρ=+0.71 p=.032),
> not hits (+0.23), d′/c undecided (−0.50/−0.50), survives controlling for naïve accuracy (partial r=−0.83) and
> is change–change (expert levels ρ=−0.32 n.s.); decoder/axis dependence and the push's normalization
> dependence disclosed in Results; §5: ED 7 specificity controls, blocked curriculum + no-opsin illumination,
> "exploratory" at n = 5, OFF-trained axis; Methods: pooled-vs-per-mouse accuracy, no video [AUTHOR],
> ChR2 OFF trials, decision read window = response window, curriculum "changing sample odors" [AUTHOR];
> figures: Fig 1g "delay lick" ticks, Fig 2 mid-delay 6.0–6.5 s + PS line, Fig 3 windows + c/f legend,
> Fig 4 "unpaired" + label collisions, Fig 6 per-day ON−OFF stars removed, g verdict from the clustered
> model, baked letters blanked, n = 20 obs., no "trainLD_TEST"; ED 6f = fig_manifold_supp composed.

> **v12.9 (2026-09-07): REVIEW FIXES, round 1 (internal rigor reviewer; verified against caches)** — §3 no longer says
> the plane's removal "collapsed" decoding (residual stays above chance: sample 0.70→0.56 p=.004, choice 0.67→0.61
> p=.020; Fig 3c legend matched); the "frame neither gained dimensions nor turned" claim is restricted to the
> memory and choice axes (the distractor axis DOES turn, §4); "required" → "impairs/needed for normal learning"
> (intro, §5, Fig 6 legend, Discussion); push sentence names the per-animal trend; absence claims (4d, 4e, 6k,l)
> → "no detectable/no evidence"; anticipatory-choice null (≈0.59) quoted; naïve r = 0.66; "exactly one" → "one";
> 6i leads with the clustered model, 6j reworded; ED 8 ChR2 pooling disclosed; licking control sentence in §4;
> Methods: control illumination, ACC-mice placeholder, single-expert-session note; dangling "(Extended Data)"
> refs → ED 6f (fig_manifold_supp, to compose); Murray 2017 + Spaak 2017 added. Stats verbatim otherwise.

> **v12.8 (2026-09-07): OUT-OF-CONTEXT PLANE TEST written in** (Leon: "no new panel, write the sentence and the
> ED entry"). One §3 paragraph after the plane paragraph (pooled medians 0.98–1.02 refit / 0.86–0.89 no
> refit; per mouse 0.72 / 0.78; the in-plane displacement clause), a Methods paragraph in the plane
> subsection, and ED 6e (`fig_ooc_plane_ed.py`, composed into the ED 6 page). Numbers from
> `docs/pca/dimensionality.md` 2026-09-07 (caches OOC_PLANE_PSEUDO_nopca / OOC_PLANE_nopca, 20 resamples).

> **v12.7 (2026-09-07): intro gap sentence made a question** (artifact comment "this should be a question"):
> "Which of the two describes the natural acquisition of a composite task?" closes intro ¶2 (Kaufman and Yang
> pose their gap as a question in the Introduction; questions stay out of the Results).

> **v12.6 (2026-09-07): PLANE STORY RECAST — route A** (artifact comment: "worried that we are overselling this
> point"; VERIFIED in `pca/exp_permouse_plane.py`: for sample and choice the full decoder is fit on the same
> trials with the same estimator as the axis that spans the plane, so plane = full is by construction and
> collapse without the plane is expected for a binary code). §3 question 2 → "what else lives in that plane";
> the plane paragraph now states the sample/choice arms as a consistency check and makes the test (outside) and
> distractor (partly inside, drawn in with learning) results the finding; "double dissociation" gone; Discussion
> claim 1 no longer says "necessary and sufficient" (its 0.012 bound stays in §3); the "morph does not leave the
> subspace" clause dropped; Fig. 3 legend title and panel c/d entries recast (CAP_PARAS + re-render). Stats
> verbatim. Fig. 1h keeps the distractor-free DPA trials (Leon's decision; alternatives logged in memory).

> **v12.5 (2026-09-07): §4 set-point sentence rephrased** (artifact comment "rephrase this sentence"): the
> "delay-lick to false-alarm chain … is precisely the chain such a position suppresses" sentence is now two
> plain ones (further from the lick boundary → a delay lick is less likely; that is the behavioral change of
> Fig. 1g).

> **v12.4 (2026-09-07): §1 hedge sentence CUT** (artifact comment on "Because Fig. 1g is a within-trial
> association, it identifies the route of the interference rather than its cause": "this sentence is
> useless"). Removed; the paragraph now ends on the expert result. The opening sentence of the Results is
> untouched (the auto-acknowledgement had guessed the wrong sentence).

> **v12.3 (2026-09-07): "LICKING", NOT "THE LICK"** (artifact comment: "it is not 'the lick' that interferes
> but 'licking'"). The interfering factor is the act of licking, so the two §1 sentences that named "the
> lick" as what interferes / what the memory must be protected from now say "licking"; "delay lick" stays
> as the defined term for the event.

> **v12.2 (2026-09-07): "TASKS", NOT "COMPONENTS"** (artifact comment: "I don't like to call each sub task a
> component. let's call them what they are 'tasks'"). Canon: the dual task combines two tasks, the DPA task
> and the GNG task; "component" survives only in its PCA sense (cvPCA/principal components). Nine
> replacements in the Results, one in the Discussion; legends and scripts never used the word in the task
> sense, so no re-render.

> **v12.1 (2026-09-07): RECORDINGS INTRODUCED** (artifact comment on the Results opening: "we need to
> introduce more in detail the recordings"). New second untitled paragraph: GCaMP6s/hSyn in prelimbic
> mPFC, two-photon through a chronic implant [AUTHOR: lens/window], same field of view every session,
> 4–6 daily sessions and 113–693 registered neurons per mouse (3,319 in all; counts read from the CCGD
> weights `valid` masks), opsin/laser scoping (7 laser mice, laser-off trials only before the ACC
> section, 5,568 trials), and the two analysis levels (within-mouse / pseudo-population with per-animal
> companions). Methods imaging paragraph carries the same counts. Still missing for a NatNeuro Fig. 1: a
> field-of-view image with example traces (author-supplied).

> **v12 (2026-09-07): STYLE, VOICE AND STORYLINE PASS** (Leon: "work on the phrasing, the storyline,
> the text itself: style, humanize … align perfectly with a Nature Neuroscience publication"). Abstract →
> ACC section and the Discussion re-read sentence by sentence against the corpus of the eight NatNeuro
> papers we cite (`docs/paper/natneuro_style_guide.md`; Golub 2018 and Libby 2021 full texts as the
> cadence reference). Changes: two-word verdict sentences ("It did." / "It did not.") and the "told a
> different story" tic removed; §4 opening rebuilt (editing → what moves → two movements on one axis)
> and its dangling "set-point anticipated above" now answers §2's deferred question (a standing no-lick
> set-point, not a premature decision, with the ED 3g pointer); the "no-lick push" defined once, in
> quotes, before "the push" is used; the delay-lick propagation logic spelled out in §1 (a lick begets a
> lick whichever answer is correct); mPFC and ACC defined on first use in the Introduction; sentences
> re-cut to the corpus cadence (Results mean 23.8 words per sentence, 26% over 30, 17% under 12; "we"
> 9.9 per 1,000; prose semicolons 2.4 per 1,000; 0 em-dashes; 0 questions); abstract 150 words, no
> numerals; Discussion cadence matched and two sentences no longer open with "mPFC". ALL STATISTICS
> VERBATIM (digit-token multiset diff old vs new = punctuation only); no claim strengthened or weakened.

> **v11.3 (2026-09-07): DISCUSSION SCOPE PASS** (Leon on the limitations paragraph: "very harsh
> limitations… maybe we should let the reviewers complain and not actually give them the gun"). The six
> enumerated limitations are now three sentences of scope, each stated with its control: the
> pseudo-population (paired with the per-animal companions), the single-task pretraining (phrased as
> what the claims are about, not as a doubt about innateness), and the lick-defined choice axis (the
> delay displacement occurs without delay licks; the coupling survives a lick covariate). Cut: the
> n = 5 clustering caveat (§5 keeps it in the approved wording), the "modest push" sentence (§4 carries
> the statistics), the model-in-preparation line (own section). The rest of the Discussion was made
> consistent with the Results canon: "prevents the memory task from being acquired" → "impairs
> learning of the dual task, with the deficit on its memory component"; "trades the two tasks against
> each other" → "is coupled to the distractor task"; "generalizes … across animals" → "present in every
> animal" (the cross-animal plane analysis was retracted as circular on 2026-09-01). No statistics
> touched.

> **v11.2 (2026-09-04): STORYLINE PASS** (Leon: style, flow and story matter now, not word counts).
> The Introduction set up two named alternatives, construction and editing, and the Results never
> named them again, so the paper's central question went missing for four sections. They are now
> threaded through: the Introduction states the opposite predictions the two make; the behaviour
> section hands over by asking which route learning took; section 2 closes the first test
> (construction predicts new coding dimensions, and there are none); section 3 closes the second
> (construction could still have rotated the existing axes, and it did not); section 4 opens on
> what is left (editing predicts states move within a fixed frame, so what moves?). This is the
> Golub adjudication pattern, in which each named hypothesis is disposed of where its evidence
> lands. No statistics touched, no claim strengthened.

> **v11.1 (2026-09-04): CLARITY PASS on 14 threads** — (1) BUILD FIX: the draft artifact split
> Introduction from Results at the first `## ` heading, so the untitled Results opening paragraphs
> rendered inside the Introduction; an explicit HTML comment marker now sets that boundary; (2) metrics
> named where numbers were bare (balanced accuracy against chance 0.5; eta-squared defined in
> place; cvPCA reliable-variance fractions labelled); (3) cvPCA cited [Stringer 2019b];
> (4) "leave-one-mouse-out confidence intervals" -> "jackknife confidence intervals across mice";
> (5) the section-2 forward reference no longer spoils the section-4 set-point; (6) the section-2
> qualifications paragraph and the section-5 limits paragraph rewritten in plain prose; (7) the
> plane-decoding method explained; (8) naive NoGo choice trace added beside the expert one (+0.1,
> verified from the trace cache); (9) Fig. 1h trial sets named and its closing sentence made
> concrete; (10) "how the delay lick did its damage" -> "by what route the delay lick interfered
> with the memory"; (11) intro "a dual task composed of". ALL STATISTICS VERBATIM.

> **v11 (2026-09-04): NATURE NEUROSCIENCE STYLE PASS** — Abstract, Introduction and Results
> rewritten against a corpus-derived house-style guide (`docs/paper/natneuro_style_guide.md`,
> built from the eight Nature Neuroscience papers we cite; full texts and the two long-form
> analyses in the job tmp). Changes: (1) ABSTRACT cut 246 → 150 words (the journal limit) with
> ALL numerals removed, rebuilt on the corpus architecture context → gap → "Here we show" pivot →
> findings in past tense → one present-tense implication; (2) INTRODUCTION 4 → 3 paragraphs
> (framing+gap / construct-vs-edit / approach + full findings roadmap), one gap sentence per
> paragraph boundary with the stock verb, citations front-loaded, closing paragraph nearly
> citation-free; (3) RESULTS now open with TWO UNTITLED paragraphs (task, curriculum, stage
> definitions, the "delay lick" term) before the first heading, as in Golub/Libby/Kaufman;
> (4) all four rhetorical questions removed from Results (corpus 0–2 per paper); (5) headings
> normalized to declarative claim sentences ≤10 words — §2 "One coding dimension per task
> variable, shared across trial types" → "Each task variable occupies its own coding dimension",
> §3 "A single sample × choice plane is necessary, sufficient and stable" → "A single plane
> carries the memory and choice codes"; (6) purpose-clause openers ("To determine…, we…",
> "To locate…, we…", "To test whether…, we…"); (7) "Critically" removed (0 occurrences in the
> corpus); colon-explanations and prose semicolons cut (prose semicolons now 1.5 per 1,000 words,
> corpus 0.6–9.1); long paragraphs split to the corpus 120–165-word range. ALL STATISTICS
> VERBATIM; no claim strengthened or weakened.

> **v10.4 (2026-09-04, later): §4 + §5 STORY REWRITE, model section, vocabulary** — 7 activated
> threads. §4 and §5 rewritten in the NatNeuro register (stats verbatim); NEW placeholder subsection
> "A circuit model of the gated no-lick repositioning" (Fig. 5 in preparation); "impaired the learning
> of DPA" → "impaired learning of the dual task, and the deficit fell on its memory component"; the
> dPCA-estimator caveat moved from §4 to Methods (disclosure kept); VOCABULARY: the lick axis is the
> "choice axis" everywhere — "action axis/dimension/codes" retired in the intro, §4, §5, Discussion and
> the Fig 3e / 4a / 6g–i legends (+ CAP_PARAS; Figs 3, 4, 6 re-rendered); dPCA's own "action axis" in
> the ED 9 sentence is now labelled as that decomposition's axis; "nonpaired" → "unpaired". NoGo-only
> coupling: see the §4 sentence + `overlaps/exp_push_nogo_coupling.py`.

> **v10.3 (2026-09-04, later): §3 STORY REWRITE** — 4 activated threads. §3 rewritten paragraph by
> paragraph in the NatNeuro register (questions → traces → sufficiency/necessity → stability), stats
> verbatim; "cross-type transfer" explained in plain words (a sample decoder trained on one trial
> type still reads the sample on the other two after the distractor, so the memory survives and only
> its projection on the fixed axis fades); the "every … every … every" sentence rewritten in text and
> Fig. 3b legend (CAP_PARAS, Fig 3 re-rendered); the NoGo choice-trace sentence VERIFIED from the
> trace cache (expert NoGo choice-axis late-delay mean −1.9 (no-lick trials) / −1.0 (lick trials),
> 7/9 mice below baseline on both; Go +4.6/+5.1) — the cue drives Go positive and NoGo negative.

> **v10.2 (2026-09-04, later): §2 STORY REWRITE + DPA NAMING** — 5 activated threads. (1) §2
> rewritten paragraph by paragraph for the NatNeuro register (question → measurement → claim;
> "line in population space", "frame", "axes"; ML phrasing out; "dimensionality and composition of
> the pseudo-population geometry" gone); every statistic verbatim; (2) the "free of the decision"
> sentence no longer contradicts §4: training removes TRIAL-BY-TRIAL choice information from the
> delay, while the expert delay state rests at a fixed no-lick position on the choice axis that
> does not differ detectably with the upcoming choice ("a standing no-lick set-point, not a
> premature decision"); (3) the working-memory task is named precisely as the olfactory delayed
> paired association (DPA) task in the Abstract and intro ¶3; §1 now says "the DPA task" (defined
> earlier).

> **v10.1 (2026-09-04, later): INTRO + §1 COMMENT PASS** — 7 activated threads. (1) Intro ¶3 is
> now GENERAL (dual task = working-memory task + GNG discrimination in its delay; composition;
> imaging; per-animal companions) and the full task description lives at the top of §1, which now
> opens "We trained head-fixed mice (n = 9) on the dual task" (not "the DPA task") and decomposes
> it into DPA + GNG with reward/no-punishment/trial mix/curriculum; (2) NAMING CANON "delay lick"
> = any lick emitted during the delay (defined in §1 ¶3; the Go response or an unwarranted NoGo
> lick) — replaces "intruding/intrusive lick", "cue lick" in the §1 heading ("…arises from delay
> licks"), §1, intro closing, §4, Methods, Fig. 1 legend + CAP_PARAS (Fig 1 re-rendered);
> (3) "distractor performance" → "GNG performance" (§1 ¶4); (4) ED lick figure request logged
> again (assets: overlaps_lick_control.png, licks.py). ALL statistics verbatim.

> **v10 (2026-09-04): INTRODUCTION COMMENT PASS** — 8 activated artifact comments (Leon, 2026-09-04)
> on the Abstract + Introduction. Applied: (1) the opening is now about complex behavior running
> several computations at once, not "two things" (Abstract and intro ¶1); (2) the interference
> premise is stated the way the literature states it — same prefrontal neurons carry memory and
> respond to stimuli/actions [Jacob 2014; Parthasarathy 2017; Musall 2019]; a second task in the
> delay weakens the memory representation [Watanabe 2014] as in human dual task interference
> [Pashler 1994]; distractors transiently disrupt/reformat the code; subspace separation limits it
> [Libby 2021] — our own "the action captures the response" mechanism is no longer a premise
> (it is Fig. 1's result); intro ¶1 rewritten around this; (3) the task is introduced AS the dual
> task ("designed to force the composition: the dual task"; "learned a dual task" in the
> Abstract) and only then decomposed into DPA + GNG; (4) "dual task" unhyphenated everywhere
> (text, legends, Discussion, notes; hyphen kept only inside verbatim reference titles), §1
> heading → "The cost of the dual task arises from the intruding lick"; (5) "lick decision" →
> "choice" (Abstract, §4, Fig. 4a legend + its CAP_PARAS); (6) two references added. ALL
> statistics verbatim.

> **v9 (2026-09-04): REVIEW-COMMENT PASS** — 20 artifact comments (Leon, 2026-09-04) + the
> carried NoGo-accuracy comment. Applied: (1) American English throughout text and legends
> (behavior, odor, color, factorized, generalize…); (2) terminology: GNG defined once in the
> intro, trials called Go / NoGo trials (not DualGo/DualNoGo) after that; (3) intro: the
> "same network" sentence now rests on citations [Jacob 2014; Parthasarathy 2017; Libby 2021]
> rather than assertion, no-punishment stated, curriculum → Methods, imaging described as every
> session (naïve = early, expert = late dual task sessions), the technical closing sentence
> replaced by a plain one; (4) §1 opens with a full task + setup description and a narrative
> learning paragraph, "second lick" → "that test lick", "lick interference" → "intrusive delay
> licks", decoupling sentence rewritten; (5) §2: no forward reference to Fig. 4, the premature-
> choice/dPCA/qualifications paragraph rewritten in plain language (PR and rank-2 numbers moved
> out to ED 3, where they already live); (6) SCOPING FIX raised twice: the geometry was measured
> after single-task pretraining, so "before learning" → "when dual task training began" in the
> Abstract, intro, §2, §3, legends and Discussion, plus a new sixth limitation; (7) Methods:
> curriculum paragraph + no-punishment sentence (author placeholders for criteria); (8) ED Fig. 10
> (licks) drafted as a proposal; (9) NoGo-accuracy coupling logged as an open analysis. ALL
> statistics verbatim.

> **v8 (2026-09-03): HUMAN-VOICE PASS** (user: "make them sound as human as possible… as an
> expert in systems neuroscience… Nature Neuroscience standards… get inspiration from the
> literature"). Calibrated against the full text of Golub et al. 2018 (Nat Neurosci; fetched
> from Europe PMC): unhurried sentences of varied length, questions posed directly, reasoning
> made explicit ("We next asked how the lick did its damage"), fewer dash-chains and colon
> stacks, explanatory "because/so that" clauses in place of compressed appositions, calm
> what-changed / what-did-not phrasing. Structure and headings unchanged from v7; **ALL
> statistics and calibrated claims verbatim** (push ∗ + per-animal trend; A-vs-B n.s.; Fig 6g
> raw trend vs 6i robust arm; "no detectable" dimensions; equivalence bounds). Discussion v4
> and the figure legends received the same pass.

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
> "The dual task cost arises from the intruding action"; §2 low-D → "One coding dimension per
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
> and enables dual task learning"*. (The v6 colon title is retired — the cited NatNeuro corpus
> [Golub 2018; Libby 2021; Parthasarathy 2017; Driscoll 2024] titles are single declarative claims.)
>
> Figure order: **1 Behavior · 2 Geometry (low-D, factorized, shared) · 3 One subspace (the
> sample × choice plane: sufficiency + stability) · 4 Learning (alignment + push + coupling) ·
> 5 Modeling (TBD) · 6 Opto (causal).** Publication-ready standard: **every panel is referenced and
> described.** Panel letters/stats verified against the RENDERED figures (Fig 1 `behavior_main.png`;
> Fig 2 `fig_dimensionality_main.png`; Fig 3 `fig_manifold_main.png` — canonical no-PCA build;
> Fig 4 `fig_overlaps_main_ab_dpaact.png`; Fig 6 `behavior_opto_main.png`). Caveats to carry are
> flagged inline as _(caveat: …)_; guardrails and to-reconcile items at the bottom.

---

## Abstract

Holding a memory while other events demand a response is difficult, because the prefrontal
neurons maintaining it also respond to them. How a circuit manages both is unknown. Here we
show that mouse medial prefrontal cortex accommodates a second task by moving activity within
an existing representation rather than building a new one. We imaged prelimbic neurons as mice
learned a delayed paired association with an embedded Go/NoGo discrimination. Memory,
distractor and choice occupied nearly orthogonal axes of one low-dimensional subspace, shared across trial types and already present when dual task training began. Learning added no
detectable coding dimension. Instead, the distractor code rotated onto the choice axis and the memory state moved along it to a no-lick set-point whose depth predicted each animal's
improvement. Silencing anterior cingulate input displaced the state without degrading its code.
Learning and top-down control act on where the state sits within a fixed geometry.

---

Behavior in a natural setting is rarely one thing at a time. An animal keeps a goal in mind
while it monitors new stimuli, decides whether they call for an action and produces it, and
these computations run concurrently in overlapping cortical circuits. Prefrontal cortex makes
the problem concrete. The neurons that hold a memory across a delay also respond to intervening
stimuli and to the animal's own actions [Jacob 2014; Parthasarathy 2017; Musall 2019]. When a
second task is performed during the delay, the memory representation weakens and behavior
suffers, in monkey prefrontal cortex as in human dual task interference [Watanabe 2014; Pashler
1994], and even a distractor that requires no response is encoded by the same neurons and
transiently disrupts or reformats the memory code [Jacob 2014; Parthasarathy 2017]. Cortex can
limit such interference by keeping sensory input and stored information in separate subspaces
of population activity [Libby 2021]. How a circuit arrives at such an arrangement when a new
demand is imposed on it remains unknown.

Population recordings suggest where such an arrangement might live. Cortical computation is
increasingly described in terms of the low-dimensional geometry of population activity, the
subspaces and coding axes along which neural states move [Mante 2013; Kaufman 2014; Vyas 2020].
The format of that geometry matters for what a downstream reader can do with it, because
representations in which each variable has its own, roughly orthogonal axis generalize to new
conditions and let variables be read and combined without interference [Rigotti 2013; Bernardi
2020]. Seen this way, a cortical population could acquire a composite task in two quite
different ways. It could construct, adding new coding dimensions to make room for the second
task; in brain-computer-interface experiments this kind of outside-manifold learning is slow
and effortful [Oby 2019]. Or it could edit, keeping the existing set of coding dimensions and
changing only what happens within it: where the states sit, and how the codes for the new
demands align with the existing axes. Editing is what within-manifold reassociation looks like
in the same experiments, where it is the fast and natural mode of learning [Sadtler 2014; Golub
2018], and it is what one would expect if the geometry is shared infrastructure for a family of
tasks [Yang 2019; Driscoll 2024]. Construction and editing make opposite predictions.
Construction requires the set of coding dimensions to change as the second task is acquired,
whereas editing requires that set to stay fixed while the states within it move. Which of the
two describes the natural acquisition of a composite task?

To force such an acquisition, we trained mice on a dual task composed of an olfactory delayed
paired association (DPA) and a Go/NoGo (GNG) discrimination embedded in its delay, so that the
same delay period sometimes demanded an intervening action and sometimes did not. Delay
activity in medial prefrontal cortex (mPFC) is required for learning this class of task [Liu
2014], and we imaged its prelimbic population throughout dual task learning. Here we show that
mPFC does not build a new representation for the composition. The memory and the choice are
carried on nearly orthogonal axes of a low-dimensional subspace that is already present, with
the same axes, when dual task training begins, and learning adds no dimension that we could
detect. What learning changes lies inside that subspace: the distractor code aligns with the pre-existing choice axis, and the working-memory state moves along that axis into the half that
suppresses licking. This shift accompanies the uncoupling of the delay lick from the false
alarms it produced, predicts memory performance animal by animal, and is itself moved by
top-down input from the anterior cingulate cortex (ACC), a projection whose silencing during
training impairs learning of the memory task. The geometry of the code is a fixed constraint,
and both learning and top-down input act on where the state sits within it.

<!-- RESULTS -->

We trained head-fixed mice (n = 9) on the dual task while imaging prelimbic mPFC (Fig. 1a). The
dual task combines two tasks. In the DPA task, one of two sample odors (A or B) was followed by
a 6-s delay and then by one of two test odors (C or D). Licking at the test was rewarded with
water when the pair matched (A→C, B→D) and went unrewarded otherwise, so the sample had to be
held in working memory across the delay. In the GNG task, placed inside that delay, a
distractor odor was followed by a response cue at which the mouse licked for water (Go trials)
or withheld licking (NoGo trials). Two thirds of the trials in every session carried the
discrimination and one third did not (DPA trials), the three trial types were interleaved, and
errors were never punished.

To follow the population through learning, we expressed GCaMP6s in prelimbic mPFC under the
human synapsin promoter and imaged the same field of view with two-photon microscopy, through a
chronically implanted [AUTHOR: GRIN lens / cranial window], on every dual task session
(Methods). Each mouse contributed 4–6 daily sessions and 113–693 simultaneously recorded
neurons (3,319 in all), and because the field of view was held fixed and the neurons were
registered across days, the same cells were followed from the first session to the last, so
that every comparison between stages is made on the same neurons. Seven of the nine mice also
expressed an opsin in ACC for the projection manipulation reported in the final section, and in
those mice light was delivered on a pseudo-random half of the trials; every result before that
section uses laser-off trials only (5,568 trials over the nine mice). We analyze the data at
two levels. Within each mouse we work on its own simultaneously recorded neurons, which
supplies the animal-level statistics throughout, and for the geometry analyses we also pool the
neurons of all nine mice into a pseudo-population, pairing every pooled result with a
within-animal companion (Methods).

Mice reached the dual task through a fixed curriculum, learning DPA and then GNG on their own
before the two were combined (Methods). Throughout, we compare the first three dual task
sessions, which we call naïve, with the later sessions, which we call expert. We refer to any
lick emitted during the delay as a "delay lick", whether it is the Go response the task
requires or an unwarranted lick on a NoGo trial.

## The cost of the dual task arises from delay licks

Performance on both tasks improved over the six dual task sessions, and the two accuracies
converged (Fig. 1b; mixed-effects model, GNG−DPA β = +0.037, p = 0.045; condition × day β =
−0.039, p = 8 × 10⁻⁴). They improved in different ways. On the GNG side the gain came from NoGo
trials, that is, from learning to withhold rather than to lick (Fig. 1c; NoGo−Go β = +0.072, p
= 0.034). On the DPA side it came from the unpaired trials, which started far below the paired
trials and improved fastest (Fig. 1d; unpaired−paired β = −0.185, p < 10⁻⁴; condition × day β =
+0.088, p < 10⁻⁴; effect sizes summarized in Fig. 1f). In both tasks, then, what the animals
mainly learned was when not to lick.

To locate the cost of combining the two tasks, we compared unpaired DPA accuracy across the
three trial types. Accuracy was depressed when the distractor called for a Go response (Fig.
1e; Go−DPA β = −0.073, p = 0.038) but not when it called for a NoGo response (NoGo−DPA p =
0.78). What interfered with the memory was licking in the middle of the delay, not the
distractor odor.

We next asked by what route the delay lick interfered with the memory. Pooling the Go and NoGo
trials, on which the response cue makes a delay lick possible, a naïve animal that emitted a
delay lick at the distractor cue was more than twice as likely to lick at the test as well
(Fig. 1g; trial-level regression, OR = 2.32, p = 4 × 10⁻⁴; OR = 2.81 after adjusting for trial
type), and this did not depend on whether the trial was paired (lick × pairing interaction p =
0.60). On unpaired trials that test lick is by definition a false alarm (OR = 2.34, p = 0.004),
whereas on paired trials it is a hit (OR = 3.08, p = 0.02). A delay lick therefore made the
animal more likely to lick again whichever answer was correct, which is the signature of a
response propagating through the trial rather than of a memory being degraded. In expert
animals the propagation persisted (OR = 1.48, p = 0.004; adjusted OR = 1.88) but had become
selective (lick × pairing interaction p = 0.009): it was strong where the second lick was a hit
(OR = 4.52, p < 10⁻⁴) and weak where it would have been a false alarm (OR = 1.48, p = 0.02).
Tested across stages in one model, the overall propagation fell only as a trend (lick × stage
interaction, OR ratio 0.68, 95% CI [0.45, 1.04], p = 0.074; lower in 6 of 9 mice), whereas its
selectivity for pairing changed clearly (lick × stage × pairing interaction p = 10⁻⁴).
Unwarranted delay licks on NoGo trials had also largely disappeared (rate 0.24 → 0.08).
Learning did not abolish the chain from delay lick to test lick; it disconnected the arm of
that chain that produces false alarms.

Even in expert mice the two tasks were not both performed at their best. For each animal we
compared memory accuracy on the distractor-free DPA trials with discrimination accuracy on the
Go and NoGo trials. The two were not detectably related across the nine animals (Fig. 1h;
Pearson r = +0.10, p = 0.80; Spearman ρ = +0.35, p = 0.36), and no animal reached the corner at
which both are performed at ceiling (mean shortfall 0.18; mean DPA 0.88, GNG 0.87), whereas in
naïve mice the two accuracies had still co-varied (r = 0.66). Expert animals therefore did not
trade one task against the other, and each settled at its own balance between them. This leaves
the circuit with a precise problem, which is to protect the sample memory from the licking that
the discrimination demands in the middle of the delay. We turned to the population geometry to
ask how it does so, and which of the two routes, construction or editing, learning had taken.

## Each task variable occupies its own coding dimension

To determine how much of the population's activity the memory occupies, and how it sits
relative to the distractor and the choice, we analyzed a pseudo-population of 3,319 neurons at
two moments of the trial: mid-delay, after the distractor but before any cue or lick, and the
decision period that follows the test odor (Fig. 2a). We counted dimensions with
cross-validated PCA [Stringer 2019b], in which the axes are found on one half of the trials and
the variance along them is measured on the other half, so that only structure that replicates
across independent trials is counted.

By this measure the memory occupied a single dimension. During the delay of DPA trials, one
component accounted for all of the reliable variance in the population state (Fig. 2b; fraction
1.00, 95% CI [0.98, 1.00], jackknife across mice). Go and NoGo trials added one further
dimension at the same moment (0.92 + 0.07 [0.01, 0.13]), which we identify below as the
distractor axis, and the decision period spread the state over about three (reliable-variance
fractions 0.66/0.17/0.17 on DPA trials and 0.61/0.30/0.05 on Go and NoGo trials). The same
picture held animal by animal, on each mouse's own simultaneously recorded neurons. Wherever
the reliable variance could be resolved, one component dominated the delay spectrum (Extended
Data Fig. 3c; median top-1 fraction 0.90 naïve, 0.93 expert, n = 7 resolvable mice per stage),
and the decision period was higher-dimensional than the delay within the same mice (expert 0.93
versus 0.61, Wilcoxon p = 0.047, 6/7 mice; naïve in the same direction, p = 0.22). The
maintained memory is a line in population space.

Each of these dimensions corresponded to one task variable, and it was occupied only when the
task called for that variable. We trained a decoder along each variable's own axis and tested
it on withheld pseudo-trials, scoring balanced accuracy against a shuffle null and a chance
level of 0.5 (Fig. 2c). In the delay only the sample could be read out (0.89 on DPA trials,
0.81 on Go and NoGo trials), together with the distractor on Go and NoGo trials (1.00), and the
test odor and the choice stayed at chance until the test arrived, when all of them became
decodable (choice 0.81–0.85, test 0.60–0.62). The principal components were themselves the task
variables, each loading on a single factor of the design (Fig. 2d; η², the share of a
component's condition-mean variance explained by one factor, where 1 means the component codes
that factor alone). The memory line was the sample axis (η² = 0.93). In the delay of Go and
NoGo trials the state held one large distractor axis (η² = 0.98, 37% of condition-mean
variance) beside a smaller sample axis (0.91, 14%), and the decision period added choice and test axes (0.84, 0.76), with some mixing of sample, test and choice on its smaller components. The
memory axis is small but reliable, and it lies close to orthogonal to the larger distractor and
choice axes that it has to withstand.

This frame was shared across the three trial types, and it was already in place in the first
dual task sessions. A choice decoder trained on one trial type read the other two nearly as
well as its own (Fig. 2e; cells give the transferred fraction of decodable signal, (cross −
0.5)/(within − 0.5); 0.41–0.97 for the choice at decision), and so did a test decoder (0.53 and above, with four of its six cells above the within-task level). The sample transferred partially and asymmetrically at mid-delay (0.27–0.90): decoders
trained on Go or NoGo trials read the DPA trials well (0.76–0.80), whereas the DPA-trained
decoder read the dual trials less well (0.27–0.44), consistent with the shift of the sample
readout within the plane after the distractor (Fig. 3a); the sample information itself remains
fully readable from the same plane once its readout is refit in place (Extended Data Fig. 6e).
Within each mouse, the cross-type accuracy of the sample was the same in naïve and expert
sessions (Fig. 2f; Δ = 0.00, 95% CI [−0.05, +0.05], Wilcoxon p = 1.00, n = 9), and so were those of the test and the choice (+0.01, p = 0.43; +0.01, p = 0.82), leaving the fraction transferred unchanged (per-mouse medians 0.41–0.88, all p ≥ 0.65). At the level of single neurons the arrangement was carried by
largely separate populations rather than by conjunctive tuning: per-neuron sample and choice d′
were uncorrelated (r = −0.02), and the fraction of neurons selective for both matched the
independence prediction (Fig. 2g; 5.2% versus 5.2%). One axis per variable, with the axes
nearly orthogonal, is what a memory needs in order to survive an action performed in the middle
of it, because variables on separate axes cannot overwrite one another.

This frame is what construction predicts learning should change, and as far as we could detect
it did not change. Naïve and expert spectra, decodability and coding patterns were
near-identical (Fig. 2b–d), and jackknife confidence intervals across mice, leaving out one
animal at a time, included zero on the naïve−expert difference for every spectrum component and
every decodable variable (Methods). Composing the two tasks added no representational dimension
that we could detect, and removed none. The first prediction of construction therefore fails,
and whatever learning changes has to lie inside the frame rather than in the frame itself.

One delay signal did disappear with learning, and it was not part of the frame. In naïve mice
the animal's upcoming choice could be read from delay activity well before the test odor
(0.64–0.66 accuracy from early through late delay against a shuffle null of about 0.53, at the
level of the pooled population), as though the decision were being taken ahead of the evidence;
this is the single exception marked in Fig. 2c. In trained mice the same readout stayed at
chance until the test (Extended Data Fig. 3g). Training therefore removed trial-by-trial choice
information from the delay. Removing that information is not the same as vacating the choice
axis, and what the trained delay state does occupy on that axis is the subject of a later
section.

Two limits of this measurement should be stated. First, the memory state is one-dimensional
partly by construction, because a DPA trial asks the animal to hold a single binary variable;
variance-weighted estimates of dimensionality and the full twelve-condition spectra are given
in Extended Data Fig. 3. Second, these numbers describe the geometry of the states that the
population visits, not the dynamics that carry it between them, which are of higher rank. An
independent decomposition of the same data by demixed PCA gives the same picture (Extended Data
Fig. 9): time courses along single axes sharpened with learning without reorganizing, the
choice and action axes of that decomposition became more aligned (|cos| 0.147 → 0.222, p <
0.001), and the sample and test axes separated (0.098 → 0.033, p = 0.008).

## A single plane carries the memory and choice codes

A shared, low-dimensional code is not yet a single structure that the animal reuses. We
therefore asked three questions of increasing strength. The first is whether the memory and the
choice occupy one two-dimensional subspace, the sample × choice plane. The second is what else
lives in that plane, and what does not. The third is whether it is the same plane before and
after dual task learning.

We began by reading the two axes of the frame on each trial type separately (Fig. 3a;
projections of withheld trials onto each mouse's cross-validated decoder axes, baseline-zeroed,
in one common unit per mouse). On DPA trials the sample code held steady across the whole
delay. On Go and NoGo trials the same readout faded after the distractor (lower in 9/9 naïve
and 8/9 expert mice), the signature of code morphing that follows an interfering stimulus in
primate prefrontal cortex [Parthasarathy 2017], here seen along a fixed axis. The memory itself
was not lost: read from the same plane with the readout refit in place, the sample remained
fully decodable on Go and NoGo trials after the distractor (Extended Data Fig. 6e), so what
fades on the fixed axis is the projection of the memory rather than the information.

The choice axis behaved differently. The Go trace rose sharply at the cue on lick and no-lick
trials alike, because every correct Go trial licks at the cue, so this is the motor and reward
transient of the required lick rather than a choice signal, and the split between upcoming lick
and no-lick opened only at the test. On NoGo trials the same axis moved the other way. In
expert mice the NoGo trace ran below baseline from the distractor through the late delay (7/9
mice), on the no-lick side of the axis, consistent with active withholding, whereas in naïve
mice it stayed close to baseline over that window (+0.1 on trials with no upcoming lick), so
this displacement appeared with training. We then viewed the same data as geometry, taking
snapshots of the plane at two moments in the trial, mid-delay and decision (Fig. 3b; each
window re-centered per mouse on its mean state, so that the panels show the arrangement of the
conditions rather than their absolute position). Whatever the trial type and the moment, the
conditions separated along the same two axes.

We then asked what else lives in this plane, and what does not. Each mouse's plane is spanned
by its own sample and choice decoder axes, so projecting the population onto those two
directions reduces every trial to a pair of numbers, and projecting it onto everything
orthogonal to them leaves the rest of the population with the plane removed. We decoded each
variable from the plane, from the residual, and from the full population (withheld trials;
paired Wilcoxon tests, n = 9; Fig. 3c and Extended Data Fig. 6g). For the sample and the choice this is a consistency
check rather than a test, because the plane is built from their own decoder directions. As expected, the two coordinates decoded them as well as the whole population (to within 0.001 accuracy in every mouse and stage). Removing the plane reduced the decoding, from 0.70 to 0.56 for the sample (p = .004) and from 0.63 to 0.56 for the choice (p = .012), but did not abolish it, as expected for a redundant population code in which many directions carry the same information. The informative results concern the other two variables.
The test code lay outside the plane: its two coordinates decoded it at 0.52, against 0.57 from the full population (p = .055), and removing the plane left it untouched (p = .82). The distractor fell between the two, with a real but
partial share of the plane (p = .004). The pooled population gave the same verdict for the
memory subspace as a whole: read from the top three principal components of the DPA state, the
distractor decoded at only 0.61 at mid-delay (permutation p = 0.031; Methods).

The plane also held beyond the trials on which it was fitted. A plane fitted on the DPA trials
of one stage read the sample and the choice on every other stage, trial type and moment of the
trial as well as a plane fitted in that context for the sample (Extended Data Fig. 6e; median 1.02 of the in-context signal with the two-dimensional readout refit in place, 0.82 with the reference decoder applied unchanged), and at 0.77 of it for the choice (0.54 unchanged), whose signal at the decision window is the weaker of the two; per mouse, medians 0.81 and 0.73. Where the unchanged decoder failed and the refit did not, on dual-task trials after the distractor, the sample code had moved within the plane rather than left it.

The same pattern held in every animal (Extended Data Fig. 6g), and it carried the one change
with learning in this section. The distractor's plane-only accuracy rose (0.58 → 0.65, p = .027, 8/9 mice) while its full-population decodability rose less reliably (0.79 → 0.86, p = .13), so that the share of the distractor signal carried by the plane grew from 0.28 to 0.37 (chance-referenced, 7/9 mice, p = .16): animal by animal the distractor code moved toward the plane, a change we quantify in the next section. The sample axis
lay at the level expected for independent directions relative to the choice axis (Fig. 3d; |cos| = 0.06 in naïve and 0.08 in expert mice) and to the distractor axis (0.09; same analysis, Methods), against a floor of ≈0.05 for random directions in these populations, whereas the overlap between choice and distractor was partial and growing (0.29 → 0.42 after correction for split-half reliabilities, pooled; the per-animal raw counterpart is tested in Fig. 4a).

Finally, the plane was the same plane before and after learning (Fig. 3e). Decoder axes trained in one stage read the withheld activity of the other stage at 90% of the within-stage ceiling for the sample and 72% for the choice (transfer/within 0.90 and 0.72, robust to scoring both stages under one common scaling), and within each animal the transfer was 0.86 of the within-stage ceiling for the sample and 0.59 for the choice. Construction could
still have survived the dimension count by rotating the memory and choice axes into a new
arrangement, and this test leaves it no room there: axes fitted before learning read the
activity after it, so the memory and choice axes neither gained a dimension nor turned
detectably. What did turn, the next section shows, was the distractor axis, and it turned within this plane. In the vocabulary of brain-computer-interface learning [Sadtler 2014; Golub
2018], what remains is within-manifold learning, in which the subspace is a fixed constraint
and learning moves states inside it.

The abstract character of the format, its generalization across conditions [Bernardi 2020], was
likewise present in the first dual task sessions and preserved. Per-mouse cross-condition
generalization sat on the naïve = expert line for the sample and choice codes; only the test code nudged upward (p = .04, uncorrected across the four variables), which we report without a verdict (Extended Data Fig. 6f). Decoding all 462 balanced dichotomies of the 12
conditions, the shattering dimension, gave 0.67–0.68 against a shuffle floor of 0.50 and an
unstructured ceiling of 1 (Extended Data Fig. 3), unchanged by learning (Δ = +0.01). High
generalization with moderate shattering is the abstract, compressed regime that Bernardi et al.
described in hippocampus and prefrontal cortex [Bernardi 2020].

## Learning repositions the memory state along the choice axis

That leaves editing, which predicts that learning moves states within the fixed frame, so we
asked what moves. Two things did, and both concerned the choice axis. The first was the
distractor code. In naïve mice the distractor code and the choice code were only partly
aligned, and with learning the distractor code rotated onto the choice axis (Fig. 4a). For this test the choice axis was fitted on the distractor-free DPA trials, so that its alignment with the distractor axis cannot be inherited from distractor-evoked activity in the same trials. A decoder trained on one code read the other with a chance-referenced transfer of 0.42 [0.11, 0.64] in naïve animals and 0.50 [0.29, 0.70] in expert animals, and within animals both the raw cosine between the two axes (0.063 → 0.104, p = .004, 9/9 mice) and the cross-decoding (0.53 → 0.60, p = .020) increased. The same convergence measured on the dual-task trials themselves was present but heterogeneous across animals (Methods). The distractor's demand had become readable as what it is for the animal, a choice.

The second was the memory itself. Learning moved the working-memory state along that same axis,
toward the no-lick side. In expert mice the delay state of DPA trials sat further into the
no-lick half of the choice axis than in naïve mice (Fig. 4b; mixed model over 9 mice and 36
observations, β = −1.15, p = 0.007), a directional shift that reached only a trend when each animal was taken as its own unit (per-animal Wilcoxon p = 0.055, 6/9 mice). We call this displacement the
"no-lick push". It answers the question left open above: what the trained delay state occupies
on the choice axis is a standing no-lick set-point, not a premature decision, since the expert
delay carries no trial-by-trial choice information (Extended Data Fig. 3g). The further the
delay state sits from the lick boundary, the less likely a lick is to escape during the delay.
This is the change that learning produced behaviorally, in which unwarranted delay licks became
rare and the propagation from delay lick to false alarm was selectively weakened (Fig. 1g). The
shift was numerically larger for sample A than for sample B (ΔA ≈ −1.87, p = 0.074; ΔB ≈ −0.44, p = 0.30), but the direct paired comparison across the same nine mice was not significant (p = 0.098, and n.s. in every axis × normalization build we tried). The difference is not a decoder
artifact, since the sample and choice axes are orthogonal (per-mouse |cos| = 0.04) and leakage
between them would displace A and B in opposite directions, which only 3/9 mice showed. We
therefore report the repositioning itself and make no claim about sample specificity.

The size of the push predicted behavior across animals. The further a mouse had moved its delay
state toward no-lick, the more its DPA accuracy had improved (Fig. 4c, left; per-mouse Spearman ρ = −0.72, p = 0.030, n = 9, on the distractor-free DPA trials; on the dual trials, where the cost is paid, or on all trials the same ranks give ρ = −0.63, p = 0.067), whereas the same change bore no relationship to GNG accuracy on the dual trials (Fig. 4c, right; ρ = +0.13, p = 0.73). Split by trial type, Go trials alone were null (ρ = −0.02, p = 0.97), and NoGo trials alone, the arm of the distractor task that improves with learning, showed a trend of the opposite sign (ρ = +0.65, p = 0.060): if anything, the animals that pushed deepest gained least on NoGo trials. The coupling is therefore specific to memory performance; whether it carries a cost for the distractor task cannot be settled at this n. It held under every normalization we tried (ρ = −0.67 to −0.80, all p ≤ 0.05), with mixed resampling support (bootstrap CI excluding 0; permutation p = 0.039; significant in 4 of 9 leave-one-mouse-out folds, ρ = −0.64 to −0.81). It tracked the fall in false alarms rather than the hit rate (per-mouse Spearman of the change in depth against the change in false-alarm rate ρ = +0.59, p = 0.097; against the change in hit rate ρ = +0.11, p = 0.78), the route by which the delay lick interfered in the first place (Fig. 1g), although neither arm was significant on its own. A signal-detection decomposition could not tell a gain in sensitivity from a shift in criterion at this n (Δd′ ρ = −0.37, p = 0.33; Δc ρ = −0.30, p = 0.43), but a general shift toward withholding is not what the coupling measures, since the NoGo trials on which withholding also improves show no coupling of the same sign. It did not depend on where the animals started (partial correlation controlling for naïve accuracy, r = −0.70, p = 0.038), and it is a relationship between changes: within the expert stage alone, depth and accuracy were not correlated across animals (ρ = −0.25, p = 0.52). Two dependences are disclosed. The coupling requires the axis to be fitted per stage and
read with the logistic decoder (on a single axis shared by both stages it weakens to a trend, ρ = −0.63, p = 0.067, and under L1-regularized or linear-discriminant decoders it is not significant, ρ = −0.63 and −0.62; Extended Data Figs 5b and 6c), and it does not replicate on the demixed-PCA axis (Methods).

Two controls sharpen the interpretation. First, the push is a property of animals rather than
of trials. Within naïve unpaired trials, the trial-by-trial depth of the state did not
detectably separate correct rejections from false alarms (Fig. 4d; sample A Δ(CR−FA) = −1.05, p = 0.26; sample B +1.17, p = 0.49), so we found no evidence that the repositioning is a
within-stage readout of accuracy rather than a between-animal learning effect. Second, what
moved was the position of the state rather than the code itself, since we found no detectable
change in the discriminability of the choice code across learning (Fig. 4e; d′ 0.55 → 0.63, Δ = +0.09, p = 0.57). Third, licking did not account for the repositioning: late-delay licks were
rare, their rate did not track the depth of the state (ρ = +0.07), and the push and the
coupling were unchanged with a lick covariate (Extended Data Fig. 5d). Two caveats also apply.
The push is directional rather than a precise magnitude, because part of the per-stage change
is a reorganization of the decoder axis itself: on a fixed common axis it is not significant, and it reaches significance only in the evoked-s.d. units used here and in whole-trial-s.d. units (p = 0.045), remaining a trend of the same sign (p = 0.08–0.11) under the other normalizations of the axis (Extended Data Fig. 5a,b). And the
behavioral coupling is an individual-difference correlation over nine animals, whose robustness
and limits are set out in Methods.

## A circuit model of the gated no-lick repositioning

A circuit model of the gated no-lick repositioning is in preparation (Fig. 5). **[AUTHOR:
modelling section to be written once Fig. 5 is built; the heading should then become a claim
sentence like the others.]**

## ACC input shifts the state's position but not the code

If the composition is implemented as an edit to a fixed geometry, some input has to supply the
edit, and the anterior cingulate cortex, which projects to the prelimbic region we recorded
from, is a natural candidate for such a top-down signal. We expressed GCaMP6s in mPFC and the
inhibitory opsin Jaws in ACC, and silenced ACC terminals in mPFC with 635-nm light during the
delay (Fig. 6a). Silencing the projection on every trial throughout training, in a separate
between-group cohort trained on a blocked curriculum (9 opto versus 9 control mice, the
controls receiving the same illumination without opsin), impaired learning of the dual task,
and the deficit fell on the memory task (Fig. 6b). A mixed model placed the impairment on DPA,
most strongly on its unpaired trials, while sparing GNG (Fig. 6c; DPA β = −0.06, p = 0.009;
DPA-unpaired β = −0.12, p = 0.014; GNG n.s.). Silencing ACC cell bodies instead of their
terminals in mPFC did not produce this deficit, and silencing the reverse projection, from
prelimbic cortex to ACC, impaired the GNG task instead (Extended Data Fig. 7a–c), so the effect
is specific to the ACC→mPFC projection. This is the same DPA-selective vulnerability we saw
behaviorally in Fig. 1, and the same dependence on the learning phase that has been reported
for mPFC delay activity itself [Liu 2014].

In the imaged cohort we silenced the projection transiently instead, on a pseudo-random half of
the delay periods, so that every comparison is laser ON against OFF within the same mouse
(Jaws, n = 5). To read where the delay state sat on the learned geometry, we projected ON and
OFF trials alike onto the choice axis trained on laser-OFF trials. Transient silencing produced
no gross change in behavior (Fig. 6d,e; DPA p = 0.40, GNG p = 0.24), yet it displaced the delay
choice code in each mouse (Fig. 6f), in a direction that differed between animals. Across the
five animals, the displacement predicted the change in GNG accuracy (Fig. 6i; Spearman ρ = −0.61, p = 0.004 over the 20 points; mouse-clustered model over 20 mouse × stage × sample observations, β = −0.011, p = 0.009). The joint DPA−GNG trade-off was significant by rank (Fig. 6g; ρ = +0.46, p = 0.041 over n = 20 points that cluster within five mice; clustered model p = 0.24), and the DPA arm was not (Fig. 6h; ρ = +0.30, p = 0.19). Under
laser, DPA and GNG accuracy remained unrelated across the mouse × stage points, as they were
without laser (Fig. 6j; r = +0.44, p = 0.20). With five animals, the acute coupling is
exploratory.

Finally, we asked whether the input changed what the subspace encoded, and found no evidence
that it did. The discriminability of the memory axis (Fig. 6k; sample A versus B, late delay)
and of the GNG choice axis (Fig. 6l; Go versus NoGo, mid-delay) was spared under laser (LMM
over 20 mouse × stage × laser observations from the five mice; sample p = 0.34, GNG p = 0.74).
Acute ACC→mPFC input therefore shifts where the delay state sits along the choice axis, the
variable that learning acts on (read here on an axis trained on laser-OFF trials; Methods),
while leaving the content of the code intact, and the chronic experiment shows that the
projection is needed for the memory task to be learned normally (Fig. 6b,c). The two couplings
mirror each other in an informative way. Learning's repositioning is coupled to memory
performance at no cost to GNG (Fig. 4c), whereas the acute displacement is coupled to GNG. A
momentary push on the same axis does not reproduce the factorized, memory-specific improvement
that learning achieves within the fixed subspace.

Two limits bound this interpretation. The chronic and the acute experiments were run in
different animals, so the requirement for learning and the acute displacement are not two
measurements of the same mouse. And the acute coupling rests on five animals contributing four
points each, so we treat the GNG relationship (Fig. 6i), which survives a model that respects
that grouping, as the robust arm and the joint trade-off (Fig. 6g) as a trend (Methods).
Computing the same coupling over all seven mice that received laser gives the same answer
(Extended Data Fig. 8; GNG ρ = −0.90, p = 0.006, with the DPA arm null); the two additional
mice expressed the excitatory opsin ChR2, so that pooled test asks only whether a laser-induced
displacement, of either sign, tracks the change in GNG accuracy.

---

## Methods

> Assembled 2026-09-01 and **audited against the producing code the same day** (three-agent
> code audit; every parameter below carries file-level evidence — discrepancies found in the
> first assembly are fixed here and listed under "To reconcile"). Author-supplied experimental
> details still needed are marked **[AUTHOR: …]**.

### Animals and behavioral task

Nine adult mice were used for imaging (five expressing Jaws for ACC→mPFC silencing, two ChR2,
two with ACC-targeted controls that contributed no laser-ON trials **[AUTHOR: construct and
light delivery in these two mice]**); a separate behavioral cohort (9 opto vs 9 control) was
used for the chronic-silencing training experiment (Fig. 6b,c). **[AUTHOR: strain, sex, age,
housing, water restriction, licence/ethics statement.]** Mice learned a delayed
paired-association (DPA) task: a sample odor (A or B, 2–3 s) followed after a 6-s delay by a
test odor (C or D, 9–10 s), with a lick response in the 10–11-s window rewarded on matching
sample–test pairs and unrewarded (false alarm) otherwise. On dual task trials a Go/NoGo (GNG)
discrimination was embedded in the delay: a distractor odor at 4.5–5.5 s, a response cue at
6.5–7.0 s, lick-for-reward on Go. DualGo, DualNoGo and distractor-free DPA trials were
interleaved within sessions. Mice progressed through a fixed curriculum (DPA → GNG → Dual, six
dual task sessions **[AUTHOR: the curriculum panel of Fig. 1a shows a "changing sample odors"
step between GNG training and dual task shaping; state which odors changed and when, since the
scoping of the learning claims to the composition depends on it]**); days 1–3 are analyzed as
"naïve" and day 4 to the last day as "expert" (mice contribute 4–6 recorded days; one mouse has
a single expert session, and the leave-one-mouse-out checks reported below include it). Errors
were never punished: an incorrect lick simply went unrewarded, and a missed reward was not
signaled. **Curriculum.** Mice were trained on DPA alone until they performed it reliably, then
on GNG alone, and only then on the dual task, in which DPA, Go and NoGo trials were
interleaved; all six dual task sessions were imaged, so "naïve" and "expert" refer to early
versus late dual task sessions in animals that had already learned each task separately
**[AUTHOR: sessions per stage, criterion at each stage, shaping steps]**. Trials are analyzed
in 84 bins over 14 s (nominal 6 Hz; bin b ≈ [b/6, (b+1)/6) s). The decoder axes use one definition in both pipelines: sample and distractor axes on bins 36–38 (6.0–6.5 s, after the distractor and before the cue, allowing the ≈0.5-s GCaMP rise) and choice and test axes on bins 54–62 (9.0–10.5 s, from test onset to 0.5 s after test offset). Two read-out window conventions coexist in the codebase and are stated per analysis below: the single-trial (overlaps) pipeline indexes epochs directly (baseline bins 0–11; mid-delay 33–38; late delay 45–53), whereas the pseudo-population pipeline offsets each epoch onset by 0.5 s (mid-delay bins 36–38; late delay 48–53).

### Behavioral statistics (Fig. 1)

Learning curves were modeled on per-mouse × day × condition accuracies (proportions, not
trials) with linear mixed-effects models — accuracy ~ condition × centered day with a random
intercept per mouse (REML); per-day markers are uncorrected Wald/Welch tests requiring ≥4 mice
per group. Random-effect variances near the boundary make per-day p-values mildly
anti-conservative, as noted in the text. Trial-level associations (Fig. 1g; history effects, ED
2) used logistic GEEs clustered by mouse (exchangeable working correlation), fit separately per
stage. Fig. 1g models the probability of licking at the DPA test on Go and NoGo trials, the
trials that carry a response cue (DPA trials record no delay lick), as a function of the delay
lick at the distractor cue (the pure delay-period lick variable); a model with a trial-type
covariate gives the adjusted odds ratios quoted in the legend; a lick × pairing interaction
term tests whether the propagation differs between paired and unpaired trials (in naïve mice it
does not, p = 0.60, so the pooled estimate applies to the unpaired arm, where the test lick is
the false alarm; in expert mice it does, p = 0.009, and the two arms are reported separately),
and the paired arm is the propagation control (a delay lick there predicts a hit — incompatible
with memory corruption, diagnostic of response propagation). Changes with learning were tested
in one GEE over both stages with a lick × stage interaction and, for the pairing selectivity, a
lick × stage × pairing interaction, because separate per-stage fits do not test the difference
between stages. The panel was rebuilt 2026-09-01 (the original build's predictor pooled cue and
test licks, which is near-circular with performance) and pooled over Go and NoGo trials
2026-09-08 (it had used NoGo trials only, which left four expert mice below the three-trial
minimum for a per-mouse line). Across-animal relationships are Pearson/Spearman correlations
over n = 9 mice. All tests two-sided; p-values uncorrected and reported exactly.

### Two-photon imaging and pseudo-population

Prelimbic mPFC was imaged with two-photon microscopy through hSyn-GCaMP6s **[AUTHOR: surgery,
window/lens, rig, frame rate, ROI extraction/registration pipeline, per-mouse cell counts and
FOVs]**, yielding 3,319 neurons across the 9 mice (113–693 per mouse; 4–6 daily sessions per
mouse, neurons registered across days). Because the pseudo-population combines an order of
magnitude more neurons than any single animal, pooled accuracies exceed the per-animal ones
(for example choice 0.96 pooled against 0.67 per mouse); every pooled claim is therefore paired
with its per-animal companion. Orofacial video was not acquired **[AUTHOR: confirm]**; the
lick-based movement controls are in Extended Data Fig. 5d, and uninstructed movements are known
to shape cortical activity broadly [Musall 2019; Stringer 2019a]. The laser-OFF trials of the
two ChR2 mice were interleaved with excitation trials; the leave-one-mouse-out checks include
these mice. The pseudo-population analyses (Fig. 2 and the per-mouse geometry analyses of Fig. 3c–e) use correct, laser-OFF trials; the single-trial projection analyses (Figs 3a,b, 4, 6) use
all laser-OFF trials, with correctness filters only where stated (per-mouse × stage × task
counts in Supplementary Information; 5,568 laser-OFF trials in the balanced set). Neurons
partition disjointly across mice, so all pseudo-population resampling and jackknifing respects
mouse identity (mice are the exchangeable unit).

### Optogenetics (Fig. 6)

CaMKII-Jaws-tdTomato was expressed in ACC and its mPFC terminals illuminated at 635 nm
**[AUTHOR: viral titres/coordinates, fibre placement, laser power, histology]**. Two designs
are combined, both targeting the same ACC→mPFC projection and analyzed strictly separately: (i)
chronic training silencing — every-trial illumination throughout learning, compared between
groups (9 opto vs 9 control, the controls receiving the same illumination without opsin; Fig.
6b,c), with direction controls (ACC cell bodies; Prl→ACC) in ED 7; (ii) transient silencing in
the imaged, trained cohort — laser ON on a pseudo-random 50% of delay periods, within-mouse ON
vs OFF (Jaws, n = 5; Fig. 6d–l). In the transient design all decoder axes are trained on
laser-OFF trials only (no correctness filter on the projected ON trials — a correct-only filter
would be survivor-biased), and ON trials are projected through the fixed OFF axis. The opto depth axis is the same choice axis as in Fig. 4 (bins 54–62; a window sweep is reported to avoid cherry-picking), with per-mouse projections divided by the standard deviation of the baseline bins — the locked earlier unit, which differs from Fig. 4's evoked-SD unit and predates the opto analysis. Discriminability under laser (Fig. 6k,l) is per-axis d′ — sample A vs B on the sample
axis (late delay, bins 45–53) and Go vs NoGo on the choice axis (mid-delay, bins 33–38) —
modeled as d′ ~ laser + stage with a random intercept per mouse (20 observations = 5 mice × 2
stages × laser OFF/ON).

### Cross-validated population decoders (the CCGD pipeline; Figs 3a,b, 4, 6)

All single-trial code readouts come from one pipeline. For each mouse, stage and target variable
(sample, choice/lick, test, distractor), an L2-regularised logistic-regression decoder
(class-balanced; regularisation strength selected by an inner 5-fold cross-validation over 10
log-spaced values, 10⁻⁴–10⁴) was trained in that mouse's neuron space on all laser-OFF trials
of that stage, and every trial's decision function was evaluated cross-temporally
(train bin × test bin) with stratified 5-fold cross-validation (folds stratified on
odor-pair × task × day), so that every projected trial is scored out-of-fold — trials from
other conditions or laser-ON trials are never in any training set. Decision functions
(normalized by the weight-vector norm per train bin) were averaged over the training window of
the relevant axis: sample code, bins 36–38; test code, bins 54–62; choice ("action") code, the test window, bins 54–62; distractor code, bins 36–38 (the shared axis definition above). Per mouse, each code's
projections were then baseline-centered (subtracting the pooled mean over baseline bins 0–11)
and expressed in evoked-SD units — divided by the temporal standard deviation of that mouse's
baseline-centered, class-signed mean trajectory (all laser-OFF trials, both stages;
"pooled-evoked"). Alternative normalizations, a fixed common axis, a resampling battery, a lick covariate and L1/LDA decoder variants are reported in ED 5 and ED 6c; the push is normalization-sensitive, and the coupling holds across normalizations but not under the L1/LDA decoders (Results). The delay-state depth used in Fig. 4 is the choice-axis projection averaged over late
delay (bins 45–53, pre-test), on all laser-OFF DPA trials (correct and error).

### One estimator for the geometry analyses (Figs 2e–g, 3c–f)

The pseudo-population and per-mouse geometry analyses — plane ablation (full-population and
residual arms), cross-task generalization, cross-stage transfer, axis cosines — use one shared
estimator, defined once and imported by every script: standardisation, an *optional* PCA
compression to min(20, n_features, n_samples − 1) components, then L2-regularised logistic
regression (C = 1, class-balanced). The canonical build omits the PCA step (no-PCA); the PCA-20
build is the robustness companion, and every starred result is required to hold in both. Where a decision
direction is used as a geometric axis it is the pipeline's own decision vector mapped back to
neuron space (undoing PCA and standardisation) and unit-normalized, so decoder and axis are the
same vector and cannot disagree. (The one deliberate exception: the plane arm of the ablation
decodes from only two coordinates and uses a bare logistic regression on them.)

### Cross-validated dimensionality (Fig. 2b)

We estimated the reliable dimensionality of the pseudo-population (12 conditions = 3 tasks ×
2 samples × 2 test odors) with cross-validated PCA [Stringer 2019b]. For each of 30 random halvings (20 for the
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
(scale only — no mean subtraction; condition means are centered across conditions inside the
estimator). Repeated split-half CV is used rather than k > 2 folds because the estimator is a
cross-product of two independent condition-mean estimates whose variance is minimized by equal
halves. These estimators operate on condition means: they characterize the task-conditioned
state geometry, not the single-trial state space. A per-mouse companion applies the identical
estimator within each mouse's own simultaneously recorded neurons (DPA 4-condition set, 30
halvings, ≥6 trials per condition; cells whose reliable-variance total falls below 5 are
flagged noise-limited, drawn open, and excluded from the paired memory-vs-decision Wilcoxon;
Extended Data Fig. 3c). The null is a label-shuffled realization of
the full pipeline (condition labels permuted within mouse, trial counts preserved), normalized
by the real spectrum's positive total. Windows (pseudo-population convention): mid-delay
(bins 36–38 — the final third of the 5.5–6.5-s post-distractor epoch, closing at the Go/NoGo
cue onset, so no cue or lick has occurred) and decision (bins 54–62, from test onset to 0.5 s after test offset); the legacy PR analyses additionally use late delay (bins 48–53). 95% CIs are leave-one-mouse-out jackknife
with a t(8) = 2.306 multiplier on the jackknife SE (fractions clipped to [0, 1]; the PR floored
at 1); the "unchanged with learning" statement applies the same jackknife to Δ(Naïve − Expert),
whose CI spans zero for every component and variable — reported as absence of detectable change
at the stated precision (CI half-widths 0.03–0.30), not strict equivalence.

### Per-variable decoding power and PC coding (Fig. 2c,d)

For each condition set and window, each variable's demixed axis was computed by applying its
orthogonal design contrast to condition means from a training half (binary factors and
window-averaged states make each dPCA marginalisation rank-1 [Kobak 2016]). Held-out
pseudo-trials (one test-half trial per mouse per pseudo-trial, 10 per condition) were projected
on the axis and classified by the training-set class midpoint; performance is balanced accuracy
over 15 splits, tested against the 95th percentile of a within-mouse label-shuffle null matched
to the statistic (100 shuffles; for each shuffle the full pipeline is re-run and the accuracy
averaged over the same 15 splits, so the null is the sampling distribution of the plotted mean;
its 95th percentile lies at 0.52–0.55). The distractor cross-decode from the DPA-state subspace
quoted in the plane section cross-decodes Go vs NoGo from held-out dual pseudo-trials (24 per
condition, disjoint train/test halves, 8 repeats) projected into the DPA-state subspace — the
top-3 PCs of the DPA condition means, which are estimated from all DPA trials (the held-out
split applies to the decoded dual trials) — with LDA on the 3-D projection and a 1,000-shuffle
within-mouse label-permutation null. For Fig. 2d, each condition-mean PC's across-condition
variance was decomposed as η² onto mutually orthogonal factor contrasts (sample, distractor,
test, choice; plus a task contrast in the 12-condition set); the balanced design makes the
shares exhaustive (chance 1/3 per contrast, Beta(½,1) under no signal — large η² on unreliable
components is expected and is not evidence of coding, which is why decodability is tested
directly in Fig. 2c). PCs beyond the reliable rank are faded; the rank is the number of leading
components that individually exceed twice the shuffle floor, accumulated until 95% of the
reliable variance is reached (DPA mid-delay 1, dual mid-delay 2, both decisions 3). Apparent
delay-period test/choice η² in DPA cannot be anticipatory coding — the test odor is drawn
independently of the sample — and fails cross-validation; it is condition-mean sampling noise.

### Generalization, parallelism, abstraction (Figs 2e–g, ED)

Cross-task generalization (Fig. 2e; sample at mid-delay, test and choice at the decision
window, the same states as Fig. 2b–d) trains a decoder on one task and tests on held-out trials
of another, reported as (cross − 0.5)/(within − 0.5) against the test task's own within-task
ceiling; per-mouse companions and their learning equivalence (Δ 95% CIs within ±0.05) are in
Fig. 2f. The parallelism score (PS) is decoder-free: per task, a class-difference coding vector
is computed from condition means on independent trial halves (correct laser-OFF trials,
per-mouse subspaces written into the shared neuron space, unit-normalized), and PS is the
cross-half, sign-preserving cosine averaged over the three task pairs (10 half-splits); the
null permutes class labels within task (100 draws, 95th percentile), and the
reliability-corrected PS (raw ÷ split-half reliability) is reported as an estimate — it can
exceed 1 at low reliability — alongside the raw value. CCGP follows Bernardi et al. [Bernardi
2020], computed on the pseudo-population with leakage-free matched cross-validation and
label-shuffle nulls (per-mouse companion: Wilcoxon, n = 9). The shattering dimension decodes
all 462 balanced 6-vs-6 dichotomies of the 12 conditions at the decision window (disjoint
train/test halves per mouse × condition, 24 pseudo-trials per condition, scaler + PCA(30) fit
on the training half only, LDA per dichotomy, 8 resamples; null: pseudo-trial condition labels
permuted, 0.50). Per-neuron selectivity (Fig. 2g, ED 6b) uses per-neuron d′ (pooled-variance;
sample at mid-delay across tasks, choice at the decision window on correct DPA trials) with a
selectivity threshold at the 95th percentile of a within-mouse label-permutation |d′| null
(computed on the sample window and applied to both); the fraction of double-selective neurons
is compared descriptively with the product of the marginal fractions, and the cached statistic
is the correlation of |d′| across neurons.

### The sample × choice plane: ablation, cosines, cross-stage transfer (Fig. 3)

Panel-a traces replay the CCGD projections (above) on correct trials per task, mean ± SEM
across mice; panel-b snapshots read the same projections at two moments (overlaps convention:
mid-delay bins 33–38; late delay 45–53; decision read bins 60–66, the 10–11-s response window,
later than the 54–62 axis-training window) and re-center each window per mouse on its
cross-condition mean state, so they display condition geometry, not absolute position (ellipses
= 1 SEM of the across-mouse mean). Plane ablation (Fig. 3c and Extended Data Fig. 6g): per
mouse, the plane is the QR-orthonormalized span of that mouse's sample axis (mid-delay) and
behavioral choice axis (decision window, lick vs no-lick at the test on all three trial types, errors included), fit on one half of the trials; each variable was then decoded from (i) only the two plane coordinates, (ii) the
residual after projecting the plane out, and (iii) the full population — training on the same
half and scoring the held-out half, 10 random half-splits, paired Wilcoxon across mice (n = 9);
the plane-vs-full equivalence is additionally bounded per cell (|Δ| ≤ 0.012 accuracy in every
mouse × stage for sample and choice; the plane arm decodes from two coordinates with a bare
logistic regression). Axis-angle matrices (Fig. 3d) report attenuation-corrected cosines with
the split-half reliabilities disclosed alongside; the per-animal statistics use the raw
(uncorrected) per-mouse |cos| (Fig. 4a). Cross-stage transfer (Fig. 3e) trains the pipeline in
one stage and tests on held-out trials of the other — as a pooled pseudo-trial analysis (24–48
pseudo-trials per class, 8 resamples) and per mouse on real trials (10 half-splits; per-mouse
ratios require a within-stage ceiling > 0.52) — summarized as the chance-referenced ratio
(cross − 0.5)/(within − 0.5); a scaling-sensitivity check re-scoring the test stage in the
training stage's per-neuron scaling changes the pooled ratios by ≤ 0.02.

Out-of-context plane test (Extended Data Fig. 6e). To test the plane without circularity, the
sample axis (mid-delay) and the choice axis (decision window) were fitted on partition A of the
DPA trials of one stage and orthonormalized, and in every other context (stage × trial type ×
window; sample at early delay, mid-delay and decision, choice at the decision) three readouts
were scored on partition B2 of that context: a two-feature logistic regression on the fixed
plane's coordinates, trained on partition A of the test context; the same readout on a plane
fitted in the test context (its own axis at the tested window plus the other axis at its
canonical window); and the reference axis decoder applied without refitting. The captured
fraction is (fixed − 0.5)/(in-context − 0.5), reported where the in-context ceiling exceeds
0.60; cells above 1.2 are denominator artefacts. Pooled pseudo-trials (24 per condition for
sample, 48 per class for choice; 20 resamples) and a per-mouse companion on real trials
(half-splits, 20 resamples; the ratio pooled over a mouse's eligible cells). Decoding from the
residual after projecting the plane out stays near the full-population level because population
codes are redundant, so the test makes no necessity claim.

### Repositioning and couplings (Fig. 4)

The distractor↔choice alignment (Fig. 4a) pairs the distractor axis (Go vs NoGo at mid-delay, dual trials) with a choice axis fitted on the distractor-free DPA trials (lick vs no-lick at the test), so that the alignment cannot be inherited from distractor-evoked activity within the same trials; it is quantified per mouse as the raw axis cosine and as symmetric cross-decoding between the two codes (train on one, test held-out on the other), each compared Naïve vs Expert by Wilcoxon (n = 9). Fitting the choice axis on the dual trials instead gives the same mean change with a larger spread across animals (|cos| +0.040, 6/9 mice, p = 0.20; cross-decoding +0.041, p = 0.20; balancing Go and NoGo within the lick classes does not change this), which we report for transparency. The Fig. 3 choice decoders pool all three trial types. Delay-state depth (Fig. 4b) is the choice-axis projection of DPA delay states (late
delay bins 45–53, per-mouse evoked-SD units, all laser-OFF trials); the stage effect is a mixed
model — depth ~ stage + sample with a random intercept per mouse, over 36 mouse × stage ×
sample observations from 9 mice — with a per-animal Wilcoxon companion on the nine
Expert − Naive differences. The depth↔accuracy coupling (Fig. 4c) is a between-mouse Spearman
correlation over the nine per-mouse means of Δdepth and Δaccuracy (Expert − Naive) — the fitted
unit is the animal — with robustness established across all six axis × normalization builds and
a resampling battery (jackknife, bootstrap, permutation; ED 5); the GNG arm was also split into
Go-only and NoGo-only accuracy changes (`exp_push_nogo_coupling.py`; both null). One limit is disclosed: the
coupling does not replicate when depth is instead measured on the dPCA-derived tasks axis
(r = +0.46, p = 0.21). The FA/CR control (Fig. 4d) is
a paired t-test on per-mouse median depths of correct-rejection vs false-alarm trials (naïve
nonpaired trials, ≥3 trials per cell). The d′ control (Fig. 4e) computes the choice code's
pooled-variance d′ at the axis window (bins 54–62) per mouse × stage and compares stages by
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
**[AUTHOR: repository/DOI]**; imaging and behavioral data at **[AUTHOR: archive/DOI]**.
(Reproducibility note: the CCGD tensor's cross-validation partition is currently unseeded —
re-running the tensor build permutes folds; all downstream statistics average over folds, but
bit-exact tensor reproduction requires seeding, flagged for the deposition.)

---

## Figure legends

> Mirrored from the scripts’ CAP_PARAS (the single source of truth — edit BOTH together;
> American English + Go/NoGo terminology since v9, 2026-09-04; Fig. 5, modelling, in preparation).

Figure 1 | Combining working memory with an embedded action is costly, and the cost comes from the
delay lick rather than from the distractor odor. Recorded cohort, nine mice, laser-off trials.
Curves show the mean ± SEM across mice; ∗ p < .05, ∗∗ p < .01, ∗∗∗ p < .001 (per-day linear mixed
models, uncorrected; day 6 n = 4).

a, Task design. Each trial is a delayed paired-association (DPA) problem. A sample odor (A or B) is
followed by a 6-s delay and then a test odor (C or D); the mouse licks if the pair matches (A→C,
B→D) and withholds otherwise, so the sample has to be held in working memory across the delay. On
two thirds of trials a Go/NoGo (GNG) discrimination is embedded inside that delay: a distractor
odor, then a response cue, with a lick required on Go trials and withheld on NoGo trials. The
remaining trials are pure DPA. All three trial types are interleaved within every session, so the
memory must survive both the distractor odor and the act of responding to it. Right, the training
curriculum.

b–e, Learning curves, per-mouse and per-day accuracy. b, DPA and GNG performance. c, GNG split by
distractor identity. d, DPA paired and unpaired trials. e, DPA unpaired trials by the task context
surrounding them. The difficulty is not licking to the right odor but not licking while a lick-
demanding task runs through the middle of the memory period.

f, Linear mixed model over panels b–e (fixed effects ± 95% CI; filled circles, condition offset;
open squares, condition × day slope; random intercept per mouse). GNG−DPA β = +0.037 (p = 0.045),
with the gap narrowing over days; NoGo−Go +0.072 (p = 0.034); unpaired−paired −0.185 (p < 10⁻⁴),
narrowing over days; Go−DPA −0.073 (p = 0.038).

g, Where the interference acts. Probability of licking at the DPA test on Go and NoGo trials,
split by whether the animal licked at the distractor cue (thin lines, single mice). In naïve
mice a delay lick more than doubled the odds of licking again at the test (trial-level GEE, OR
= 2.32, p = 4 × 10⁻⁴; adjusting for trial type, OR = 2.81), whichever answer was correct (lick
× pairing interaction p = .60): on unpaired trials that test lick is the false alarm (OR =
2.34, p = .004), on paired trials it is a hit (OR = 3.08, p = .02). In expert mice the
propagation persisted (OR = 1.48, p = .004; adjusted OR = 1.88) but had become
pairing-dependent (interaction p = .009): strong where the test lick is a hit (OR = 4.52, p <
10⁻⁴), weak where it would be a false alarm (OR = 1.48, p = .02). Across stages the overall
propagation fell only as a trend (lick × stage interaction, OR ratio 0.68 [0.45, 1.04], p =
.074; lower in 6/9 mice), whereas its pairing selectivity changed (lick × stage × pairing
interaction p = 10⁻⁴). Unwarranted delay licks on NoGo trials fall from 0.24 to 0.08 of trials.
Learning did not abolish the chain of delay lick and test lick; it disconnected its false-alarm
arm, the arm the no-lick repositioning in Fig. 4 acts on.

h, Learned, but not jointly optimal. Expert DPA accuracy against GNG accuracy for each animal
(color, mouse; marker, opsin group; star, the corner where both tasks are optimal). No animal
reaches the corner (mean gap 0.18), and the two accuracies are uncorrelated across mice (r = +0.10,
p = .80; ρ = +0.35, p = .36; n = 9); each animal settles its own balance. The problem the population
geometry has to solve is therefore a precise one: shield the memory from the very action the animal
is required to produce.

Figure 2 | The population geometry is minimal and factorized. The working memory occupies a
single dimension, each task variable has its own nearly orthogonal coding axis, and the memory
and choice axes are shared across trial types. All panels use the pseudo-population (3,319
neurons, nine mice, 12 conditions). The memory state is the mid-delay window (6.0–6.5 s, after
the distractor and before any cue or lick); the decision state runs from test onset to 0.5 s after test offset (9.0–10.5 s).

a, Trial timeline, the two analyzed states, and the logic of cross-validated PCA (cvPCA). Condition
means are estimated on one half of the trials and evaluated on the other half (30 random half-
splits, both directions averaged), so only structure that replicates across independent trial halves
counts toward the geometry.

b, The memory manifold is a line. Fraction of reliable condition-mean variance per cvPCA component
(error bars, leave-one-mouse-out jackknife 95% CI, t(8); dashed gray, within-mouse label-shuffle
null). The DPA mid-delay state occupies a single reliable dimension. The dual tasks add exactly one,
the distractor axis (0.92 against sample 0.07), and the decision state spreads to about three. Naïve
and expert spectra are near-identical; learning does not change the dimensionality.

c, Each axis carries its variable when, and only when, the task engages it. Decoding accuracy
along each demixed coding axis on withheld pseudo-trials (expert, bars; naïve, open circles),
against the expert label-shuffle null (95th percentile of a null matched to the plotted
statistic, short line). The dagger marks the single exception, an anticipatory choice signal in
the naïve mid-delay state (0.66 against its own null) that disappears with learning.

d, The principal components are the task variables. η² of each condition-mean PC against the
design contrasts (rows, PCs labeled with their percentage of condition-mean variance; a cell
near 1 means that the PC codes that variable alone). The geometry is factorized rather than
mixed. Rows beyond the reliable rank of panel b are faded; dual rows show 4 of the 7 centered
contrasts.

e, Cross-task transfer of the decoders (expert; sample at mid-delay, test and choice at
decision, the states of b–d; each decoder is trained and tested in the same window). Cells give
the transferred fraction of decodable signal, (cross − 0.5)/(within − 0.5); the within-task accuracies are 0.94/0.78/0.80 for the sample, 0.68/0.58/0.56 for the test and 0.86/0.75/0.81 for the choice (DPA/Go/NoGo); hatched cells have a ratio above 1 (cross above within) and are not read as fractions. The choice transfers largely (0.41–0.97), and the test completely (0.53 and above; four of six cells exceed the within-task level, whose accuracies are low). The sample transfer is partial and asymmetric (0.27–0.90): decoders trained
on Go or NoGo trials read the DPA trials well (0.76–0.80), whereas the DPA-trained decoder
reads the dual trials less well (0.27–0.44), consistent with the shift of the sample readout
within the plane after the distractor (Fig. 3a; Extended Data Fig. 6e). Below each matrix is
the parallelism score (PS), the geometric twin of the transfer test (sample 0.28, test 0.06, choice 0.16; label-shuffle 95th percentiles 0.04–0.05).

f, The shared frame precedes dual task learning. Per-mouse mean cross-task accuracy (same
windows as e), naïve against expert; points on the unity line indicate no change. The sample is unchanged (Δ = 0.00, 95% CI [−0.05, +0.05], Wilcoxon p = 1.00, n = 9), and so are the test (+0.01, [−0.01, +0.03], p = .43) and the choice (+0.01, [−0.03, +0.06], p = .82); the fraction transferred is unchanged (per-mouse medians 0.41–0.88, all p ≥ .65).

g, The factorization is visible neuron by neuron. Per-neuron discriminability (d′, within
mouse) for sample at mid-delay against choice at decision (n = 3,319; gray square,
label-shuffle floor; color, the axis with the larger |d′|, sample in indigo or choice in green;
orange, neurons above the floor on both axes). |d′| across the two variables is uncorrelated (r
= −0.02), and the fraction of both-selective neurons (5.2%) equals the independence prediction (5.2%). Largely separate populations carry the two axes, which is the single-neuron basis of
the factorized geometry.

Figure 3 | One manifold. The two-dimensional sample × choice plane carries the memory and
choice codes on every trial type, excludes the test code, and is the same plane before and
after dual task learning. A code is the projection of population activity onto a per-mouse
cross-validated decoder axis (the decoder never sees the trials it projects), baseline-zeroed,
in units of that mouse’s evoked s.d.; correct laser-off trials; mean ± SEM across nine mice; p
values uncorrected.

a, The two axes of the frame, read in each task (columns, DPA | Go | NoGo × sample / choice;
rows, naïve | expert). The DPA sample code is maintained across the delay. On dual trials the
same readout decays after the distractor, the code-morphing signature that follows an
interfering stimulus (lower in 9/9 naïve and 8/9 expert mice); whether the memory itself
survives, or only this readout, is answered by the transfer in Fig. 2e. On the choice axis the
Go trace rises at the cue in both trial classes, a motor and reward transient (every correct Go
trial licks at the cue), and the lick/no- lick split opens only at the test; the expert NoGo
trace runs below baseline through the late delay (7/9 mice), consistent with active
withholding. Distractor and test codes are shown in Extended Data Fig. 6f.

b, The same data as geometry. Snapshots of the sample × choice plane at mid-delay (5.5–6.5 s)
and decision (10.0–11.2 s, the response window); the choice axis is trained during the test (9.0–10.5 s). Each
panel is re-centered per mouse on the mean state of that window, so it shows the geometry of
the conditions rather than their absolute position (the shared ramp and the push are carried by
a and by Fig. 4b). Dots, per-mouse condition means (at least three correct trials); ellipses,
SEM across mice; large marker, grand mean; filled = lick, open = no-lick; circle, triangle and
square = DPA, Go and NoGo; color = sample; scale bar, 2 z. Whatever the task and the moment,
the conditions separate along the same two axes.

c, What lives in the plane. Each variable is decoded from the two coordinates of the plane,
from the residual population after the plane is removed, and from the full population (mean ±
SEM, n = 9, stages averaged; withheld trial halves; paired Wilcoxon tests, all comparisons
drawn). Sample and choice decode as well from the plane as from the full population, as they
must, since the plane is built from their own decoder axes, and removing the plane reduces but does not abolish their decoding (sample 0.70 → 0.56, p = .004; choice 0.63 → 0.56, p = .012); the test code is at chance from the plane (0.52 against 0.57 from the full population, p = .055) and untouched without it (p = .82), so it lives
outside the manifold; the distractor’s share is real but partial (p = .004).

d, The memory and choice axes are orthogonal. |cos| between the sample and choice decoder axes,
corrected for attenuation by the split-half reliabilities of the axes (Methods; 0 = orthogonal): 0.06 in naïve and 0.08 in expert mice, the static layer of protection. Right, the
raw within-mouse sample × choice |cos|, naïve against expert (below 0.10 in every mouse at both stages). The choice × distractor overlap, which grows with learning, is quantified in Fig. 4a.
No tests are drawn here.

e, The frame is fixed across dual task learning. Axes trained in one stage read the withheld activity of the other stage (registered neurons) at 90% of the within-stage ceiling for the sample and 72% for the choice (transfer/within 0.90 and 0.72; cross-stage accuracy 0.88 and 0.74 against within-stage 0.92 and 0.83; robust to resampling and to a common-scaling check, with ratios shifting by at most 0.02). Right, the same test within each animal (transfer/within 0.86 for sample, 0.59 for choice). This is within-manifold learning: the state moves inside the frame (Fig. 4b), and
the frame does not rotate.

Figure 4 | Learning edits the geometry, not the code. The distractor code rotates onto the choice axis, and the memory state is pushed along that axis to an output-suppressing no-lick set-point
whose depth predicts each animal’s memory gain. Code depth is the projection onto the choice (lick)
decoder axis, per mouse, baseline-zeroed, in units of evoked s.d.; negative values lie toward no-
lick.

a, The distractor code rotates onto the choice axis. Cross-decoding between the two codes (distractor, Go vs NoGo at mid-delay on dual trials; choice, lick vs no-lick at the test on distractor-free DPA trials; balanced accuracy; diagonal, within-code; off-diagonal, transfer). The chance-referenced transfer grows from 0.42 [0.11, 0.64] in naïve to 0.50 [0.29, 0.70] in expert mice. Right, the same convergence within each animal, naïve against expert: per-mouse |cos| 0.063 → 0.104 (∗ p = .004, 9/9 mice) and cross-decode 0.53 → 0.60 (∗ p = .020), drawn from fixed canonical caches in every build. The distractor’s demand becomes readable as what it is for the animal, a choice.

b, The no-lick push: the memory state is repositioned along the choice axis. DPA delay
trajectories in the sample × choice plane (naïve | expert; strips, distributions of late-delay
depth) and per- mouse late-delay depth. With learning the delay state sinks into the half of
the axis whose readout is “do not lick”, away from the lick boundary, the geometric counterpart
of the weakening lick chain in Fig. 1g. Mixed model β = −1.15, p = .007 ∗ (9 mice, 36 observations); per-animal trend, Wilcoxon p = .055, larger for sample A (Δ = −1.87, p = .074) than for sample B (−0.44, p = .30); the A-versus-B difference itself is n.s., p = .098.

c, The push predicts behavior across animals. Each mouse’s change in depth against its change in
accuracy (circles, the two sample classes per mouse, joined; the regression band, ρ and p are
computed on the nine per-mouse means). The deeper a mouse pushes its memory state, the more its DPA accuracy improves (distractor-free DPA trials, ρ = −0.72, p = .030 ∗; on the dual trials or on all trials the same ranks give ρ = −0.63, p = .067), whereas the same change predicts nothing for GNG (ρ = +0.13, p = .73; Go trials alone ρ = −0.02, p = .97; NoGo trials alone a trend of the opposite sign, ρ = +0.65, p = .060). The coupling is specific to the memory task.

d, The push is a between-animal learning effect, not a trial-level readout of accuracy. Within a
stage (naïve unpaired trials), single-trial depth does not separate correct rejections from false
alarms (sample A, Δ(CR−FA) = −1.05, p = .26; sample B, +1.17, p = .49).

e, Position, not fidelity. The discriminability of the choice code (d′, lick against no-lick)
is unchanged by learning (0.55 → 0.63, p = .57). Learning moves where the memory state sits on
the axis (b), not how well the axis reads out.

Figure 6 | ACC→mPFC input moves the state on the learned geometry. Silencing the projection
during training impairs learning of the memory task, and acutely it shifts the position of the
state without degrading the code. Code depth as in Fig. 4.

a, Design. hSyn-GCaMP6s imaging in mPFC with CaMKII-Jaws-tdTomato in ACC; 635-nm light on a pseudo-
random 50% of trials, restricted to the delay period, so every comparison in d–l is within-mouse,
laser ON against OFF.

b, c, Silencing the projection during training impairs learning of the memory task. Chronic
every-trial silencing during training (a separate between-group cohort; 9 opto and 9 control
mice) impairs DPA acquisition (b). The mixed-model summary (c; group ● and group × day □ fixed
effects ± 95% CI) places the deficit on DPA (β = −0.06, p = 0.009) and on its unpaired trials
(β = −0.12, p = 0.014) while sparing GNG; the memory task is affected specifically.

d–f, Acute silencing moves the state, not behavior. In the recorded cohort (Jaws, n = 5), DPA
(d) and GNG (e) accuracy are unchanged between ON and OFF, yet the same manipulation displaces
each animal’s position along the choice axis (f; per-mouse depth, samples pooled). The
direction of the shift differs across mice (two toward no-lick, three toward lick), so the mean
shift is small relative to its spread.

g–i, The displacement, read on the learned choice axis, predicts behavior. Δdepth (ON−OFF)
against the accompanying change in accuracy (20 points = 5 mice × naïve/expert × sample A/B;
depth on the choice axis trained on laser-OFF trials (bins 54–62, 9.0–10.5 s), read at late delay). The joint trade-off (g) is significant by rank (Spearman ρ = +0.46, p = .041, n = 20; because the points cluster within five mice, a mouse-clustered model gives p = .24). Its arms are ΔDPA (h, ρ = +0.30, p = .19, n.s.) and ΔGNG (i, ρ = −0.61, p = .004), the latter also surviving the mouse-clustered model (β = −0.011, p = .009).

j, Under laser ON, DPA and GNG accuracy remain unrelated across mouse × stage points, as
without laser (r = +0.44, p = .20).

k, l, Position, not fidelity, again. d′ under laser ON against OFF sits on the unity line for
the memory code (k; sample-axis d′ at late delay; LMM laser p = .34, n = 20 observations, 10
OFF/ON pairs) and for the GNG code (l; choice-axis d′ at mid-delay; p = .74). The input sets
the position of the code on the subspace (f–i) without degrading its content, the same
position-not-fidelity principle that governs learning itself (Fig. 4).

## References

> Working author–year list keyed to the inline [Author Year] tags (both this file and
> `discussion_draft.md`); converted to numbered Nature format by the reference manager at
> submission. Driscoll 2024 and Pospisil 2025 verified by search 2026-09-02; Watanabe 2014 and
> Pashler 1994 added 2026-09-04 (intro comment pass); the rest are standard anchors. Note the two distinct Stringer 2019 papers (a = movements; b = cvPCA).

- **[Bernardi 2020]** Bernardi, S., Benna, M. K., Rigotti, M., Munuera, J., Fusi, S. & Salzman,
C. D. The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell* **183**,
954–967 (2020). - **[Driscoll 2024]** Driscoll, L. N., Shenoy, K. & Sussillo, D. Flexible
multitask computation in recurrent networks utilizes shared dynamical motifs. *Nat. Neurosci.*
**27**, 1349–1363 (2024). - **[Golub 2018]** Golub, M. D., Sadtler, P. T., Oby, E. R., Quick,
K. M., Ryu, S. I., Tyler-Kabara, E. C., Batista, A. P., Chase, S. M. & Yu, B. M. Learning by
neural reassociation. *Nat. Neurosci.* **21**, 607–616 (2018). - **[Jacob 2014]** Jacob, S. N.
& Nieder, A. Complementary roles for primate frontal and parietal cortex in guarding working
memory from distractor stimuli. *Neuron* **83**, 226–237 (2014). - **[Kaufman 2014]** Kaufman,
M. T., Churchland, M. M., Ryu, S. I. & Shenoy, K. V. Cortical activity in the null space:
permitting preparation without movement. *Nat. Neurosci.* **17**, 440–448 (2014). - **[Kobak
2016]** Kobak, D., Brendel, W., Constantinidis, C., Feierstein, C. E., Kepecs, A., Mainen, Z.
F., Qi, X.-L., Romo, R., Uchida, N. & Machens, C. K. Demixed principal component analysis of
neural population data. *eLife* **5**, e10989 (2016). - **[Libby 2021]** Libby, A. & Buschman,
T. J. Rotational dynamics reduce interference between sensory and memory representations. *Nat.
Neurosci.* **24**, 715–726 (2021). - **[Liu 2014]** Liu, D., Gu, X., Zhu, J., Zhang, X., Han,
Z., Yan, W., Cheng, Q., Hao, J., Fan, H., Hou, R., Chen, Z., Chen, Y. & Li, C. T. Medial
prefrontal activity during delay period contributes to learning of a working memory task.
*Science* **346**, 458–463 (2014). - **[Mante 2013]** Mante, V., Sussillo, D., Shenoy, K. V. &
Newsome, W. T. Context-dependent computation by recurrent dynamics in prefrontal cortex.
*Nature* **503**, 78–84 (2013). - **[Murray 2017]** Murray, J. D., Bernacchia, A., Roy, N. A.,
Constantinidis, C., Romo, R. & Wang, X.-J. Stable population coding for working memory coexists
with heterogeneous neural dynamics in prefrontal cortex. *Proc. Natl Acad. Sci. USA* **114**,
394–399 (2017). - **[Musall 2019]** Musall, S., Kaufman, M. T., Juavinett, A. L., Gluf, S. &
Churchland, A. K. Single-trial neural dynamics are dominated by richly varied movements. *Nat.
Neurosci.* **22**, 1677–1686 (2019). - **[Oby 2019]** Oby, E. R., Golub, M. D., Hennig, J. A.,
Degenhart, A. D., Tyler-Kabara, E. C., Yu, B. M., Chase, S. M. & Batista, A. P. New neural
activity patterns emerge with long-term learning. *Proc. Natl Acad. Sci. USA* **116**,
15210–15215 (2019). - **[Panichello 2021]** Panichello, M. F. & Buschman, T. J. Shared
mechanisms underlie the control of working memory and attention. *Nature* **592**, 601–605
(2021). - **[Parthasarathy 2017]** Parthasarathy, A., Herikstad, R., Bong, J. H., Medina, F.
S., Libedinsky, C. & Yen, S.-C. Mixed selectivity morphs population codes in prefrontal cortex.
*Nat. Neurosci.* **20**, 1770–1779 (2017). - **[Pashler 1994]** Pashler, H. Dual-task
interference in simple tasks: data and theory. *Psychol. Bull.* **116**, 220–244 (1994). -
**[Pospisil 2025]** Pospisil, D. A. & Pillow, J. W. Revisiting the high-dimensional geometry of
population responses in the visual cortex. *Proc. Natl Acad. Sci. USA* **122**, e2506535122
(2025). - **[Rigotti 2013]** Rigotti, M., Barak, O., Warden, M. R., Wang, X.-J., Daw, N. D.,
Miller, E. K. & Fusi, S. The importance of mixed selectivity in complex cognitive tasks.
*Nature* **497**, 585–590 (2013). - **[Sadtler 2014]** Sadtler, P. T., Quick, K. M., Golub, M.
D., Chase, S. M., Ryu, S. I., Tyler-Kabara, E. C., Yu, B. M. & Batista, A. P. Neural
constraints on learning. *Nature* **512**, 423–426 (2014). - **[Spaak 2017]** Spaak, E.,
Watanabe, K., Funahashi, S. & Stokes, M. G. Stable and dynamic coding for working memory in
primate prefrontal cortex. *J. Neurosci.* **37**, 6503–6516 (2017). - **[Stringer 2019a]**
Stringer, C., Pachitariu, M., Steinmetz, N., Reddy, C. B., Carandini, M. & Harris, K. D.
Spontaneous behaviors drive multidimensional, brainwide activity. *Science* **364**, eaav7893
(2019). - **[Stringer 2019b]** Stringer, C., Pachitariu, M., Steinmetz, N., Carandini, M. &
Harris, K. D. High-dimensional geometry of population responses in visual cortex. *Nature*
**571**, 361–365 (2019). - **[Vyas 2020]** Vyas, S., Golub, M. D., Sussillo, D. & Shenoy, K. V.
Computation through neural population dynamics. *Annu. Rev. Neurosci.* **43**, 249–275 (2020).
- **[Watanabe 2014]** Watanabe, K. & Funahashi, S. Neural mechanisms of dual-task interference
and cognitive capacity limitation in the prefrontal cortex. *Nat. Neurosci.* **17**, 601–611
(2014). - **[Yang 2019]** Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T. & Wang,
X.-J. Task representations in neural networks trained to perform many cognitive tasks. *Nat.
Neurosci.* **22**, 297–306 (2019).

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

**ED Fig. 1 | Behavior: learning curves by cohort (Fig. 1).** Five rows (each = the A–E
curve-plus-LMM-forest strip): pooled 9 mice, Jaws (n=5), ChR (n=2), ACC (n=2), and the interleaved
laser-ON trials of the 7 laser mice. Condition effects reproduce (pooled: GNG−DPA β=+0.037 p=0.045;
NoGo−Go +0.072; unpaired−paired −0.185; Go−DPA −0.073). Learning is comparable across cohorts.

**ED Fig. 2 | Behavior: the DPA↔GNG balance is not a trade-off (Fig. 1e/g/h).** (a) per-animal
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
coupling is ★ under all six normalizations (ρ=−0.83 to −0.90) and survives a fixed common axis (ρ=−0.72)
where the push attenuates to a trend; (c) a resampling battery (Mundlak β=−0.041 p=0.006; jackknife 9/9;
bootstrap CI [−1.00,−0.26]; permutation p=0.008), ΔGNG null throughout; (d) movement control — late-delay
licking is rare, the choice-code depth does not track it (ρ=+0.07), and the push/coupling are unchanged
with a lick covariate.

**ED Fig. 6 | Overlaps: the factorized geometry is robust (Fig. 3d; Fig. 2g).** (g) the
per-mouse plane-ablation grid, formerly Fig. 3d (`fig_manifold_main.py` →
`fig_manifold_permouse_plane.png`; naïve x vs expert y for plane/out/full ×
sample/dist/test/choice; dist plane-only ∗ p = .020/.027); (a) cross-temporal cosine matrices —
cross-code |cos| ≈ the 0.05 chance floor at all time-pairs, within-code diagonals 0.4–0.9,
choice×GNG the one least-orthogonal pair (~0.29); (b) modular, not mixed, selectivity —
per-neuron permutation tuning (sample 10 / GNG 39 / test 3 / choice 10 %, cross-variable
co-tuning at chance); (c) decoder-variant robustness — the main figure under L1 and LDA
decoders (geometry/orthogonality decoder-invariant; push/coupling clearest under logistic); (d)
codes robust to the Go/NoGo distractor — panel-A codes split by Go vs NoGo (sample/test
unperturbed; the action code carries the distractor lick); (f) per-mouse companions of Figs 2e
and 3 (`pca/fig_manifold_supp.py`; internal panels A–C) — the four codes over time on their own
axes (the definitional reference for the distractor and test codes), per-mouse CCGP naïve
against expert (present from the first sessions; the test-code change depends on the decoder
variant, p = .04/.73, no verdict), and per-mouse cross-task generalization (within- against
cross-task accuracy, expert); (e) the out-of-context plane test (`pca/exp_ooc_plane_pseudo.py`,
`exp_ooc_plane.py`, `fig_ooc_plane_ed.py`) — a plane fitted on the naïve DPA trials reads the
sample and the choice in every other stage, trial type and moment at 0.98–1.02 of a plane
fitted in that context (2-D readout refit) and 0.86–0.89 with no refit; the no-refit drop on
expert NoGo trials after the distractor, with the refit intact, shows the sample code moving
within the plane; per-mouse medians 0.72 (sample, n = 9) and 0.78 (choice, n = 8).

**ED Fig. 7 | Opto: chronic silencing + transient behavior (Fig. 6b–e).** (a–c) control-vs-opto learning
curves for the ACC→Prl, ACC-somata and Prl→ACC batches — ACC→Prl impairs DPA (β=−0.06 p=0.009) and its
unpaired trials (β=−0.12 p=0.014); ACC-somata null; Prl→ACC impairs GNG; (d) transient within-mouse laser
OFF-vs-ON curves (Jaws n=5, sub-panels A–D): DPA p=0.40, GNG p=0.24 — geometric, not a behavioral
knock-down.

**ED Fig. 8 | Opto: laser ON−OFF coupling, 7 mice (Fig. 6g–i).** The acute causal analog of the
learning coupling over all 7 laser mice (5 Jaws + 2 ChR): (a) one point per mouse — GNG ρ=−0.90
(p=0.006, n=7; the two ChR2 mice received the opposite, excitatory manipulation, so the pooled
test is sign-agnostic), DPA rank-n.s. (ρ=+0.55, p=0.21); (b) sample A & B as independent points
(n=14) — GNG ρ=−0.60 (p=0.024), DPA rank-n.s. Backs the Jaws-only axis choice and the
alternative-n disclosure.

**ED Fig. 9 | dPCA demixed axes: trajectories, mixing, and the shared plane (Fig. 2/3).** The dPCA story
build (`fig_dpca_story_main.py`): demixing schematic + descriptive scree + marginal contrasts; the 2×4
Naïve/Expert trajectory grid (single-axis time courses sharpen without reorganizing); the full pairwise
axis-mixing slopegraph (choice–task binds, 0.147→0.222 p<0.001; sample–test demixes, 0.098→0.033 p=0.008 —
neuron-bootstrap, not across-animal); the per-mouse shared-memory d′ scatter (Δ=−0.07, p=0.91, flat); and
the sample × action linking plane (one shared sample axis across DPA/Go/NoGo, ⊥ the pre-existing action
axis — the bridge cited in §3).

**ED Fig. 10 | Licking behavior (PROPOSED 2026-09-04, to build).** Lick rasters and lick-rate
time courses per trial type (DPA / Go / NoGo) and stage (naïve / expert), aligned to sample,
distractor cue and test, with the delay-period (intrusive) licks that feed Fig. 1g's predictor
marked, and per-mouse delay-lick rates. Requested in the review of §1 ("we need an ED figure with
the actual licks"); Nature caps ED at 10, so this fills the last slot.

**Supplementary Information**
- **Trial counts per mouse** — per-mouse × stage × task counts entering the pseudo-population (balanced by
  design; 5,568 laser-OFF trials total). _(Analysis-balanced counts, not raw behavioral trial numbers —
  see Methods.)_

**Omitted:** retracted dPCA choice-polarization figures (`dpca_flow_learning_ingain*`,
`dpca_flow_autonomous_choice`, `dpca_choice_ci_qsweep`); the flow-field / bistability analysis (former S7 →
"extra"); the standalone d′ figure (former S17 → already main Fig. 6k,l); demixed-axes loadings/mixing
(former S5 → covered by Fig. 2g + ED 6). **Author-supplied gaps still needed:** histology / viral
expression, imaging FOV + per-mouse cell counts, laser-power / opsin titration.

---

### To reconcile before submission (figure ↔ text integrity) - **push ↔ NoGo-trial accuracy
(comment 2026-09-02) — CLOSED, see below.** Fig. 4c right pools Go and NoGo trials (GNG
accuracy). Compute per-mouse Δdepth ↔ ΔNoGo accuracy and ΔGo accuracy separately (the push is a
withholding set-point, so a NoGo-specific coupling is the natural prediction); report either
way. Analysis pending. **CLOSED 2026-09-04:** `overlaps/exp_push_nogo_coupling.py` — same n = 9
per-mouse Spearman as Fig. 4c; NoGo-only ρ = +0.34, p = 0.38 (Pearson r = +0.32, p = 0.41);
Go-only ρ = +0.20, p = 0.61; DPA reproduces ρ = −0.83, p = 0.005. Sentence added to §4, the
Fig. 4c legend and Methods. - **OPEN (comment 2026-09-04): ED Fig. 10, licking behavior** — to
build (entry drafted above). - **Panel-label spelling**: the American-English sweep covered the
text and legends; axis labels and in-panel strings inside the figure scripts still need the
same pass. - **✓ REGISTER PASS (2026-09-02, Albert Compte's review → Leon: "jargon pass now,
keep structure"):** 20 replacements in the body (Abstract→Synthesis) + 1 in the Discussion —
ML-flavoured vocabulary translated (decode ablation → decoding after removing the plane;
bounded equivalence → "agree to within X accuracy"; estimators → measures; decoder pipelines →
decoder variants; held-out → withheld-from-fit at first use; disentangled → factorized; parks →
pushes/moves; knob-dependent → named explicitly; expressivity → separating capacity; resampling
battery → every resampling check; well depth → state depth). Claim-first structure and ALL
statistics untouched; Methods keeps its technical register deliberately. In-figure captions not
swept (they drop at submission). - **✓ CODEX DRAFT REVIEW APPLIED (2026-09-01, adjudicated):**
applied = claims calibration (abstract/§2/§4/§5/Synthesis/Discussion verbs + "no detectable"),
§5 clustered-ΔGNG-first reorder, ED count 8→9, estimator-paragraph rewording, expert-day
parenthetical, CCGP spelled out, n = 9/n = 5 in abstract, 6k,l LMM n stated as 20 obs; upgraded
(beyond Codex's ask) = Fig 1g pairing-interaction test (p = .61, in script/caption/Methods) and
plane-vs-full bounded equivalence (|Δ| ≤ 0.012 every cell — "sufficient" retained with formal
backing). REJECTED with reasons = "days 4–6" (day counts vary 4–6 per mouse; "day 4 to last" is
correct), push reframing beyond verb level (star = settled decision; full evidential balance
already led), multiplicity adjustment (stated policy: uncorrected + disclosure + two-pipeline
replication), Fig 5 placeholder removal (model in preparation — a SUBMISSION blocker only).
STILL OPEN from the review: §2 density reduction (deferred — restructuring risks stat drift;
revisit at submission), the pooling-defense sentence (candidate: the cross-animal alignment
analysis would upgrade it to a panel). - **✓ Fig. 1g POOLED over Go + NoGo trials (2026-09-08,
Leon: "keep the figure the same but use all trials"; DPA trials carry no cue so no delay lick
is recorded on them). New numbers: naïve OR = 2.32 p = 4×10⁻⁴ (trial-type-adjusted 2.81), FA
arm 2.34 p = .004, paired arm 3.08 p = .02, interaction p = .60; expert OR = 1.48 p = .004
(adjusted 1.88), FA arm 1.48 p = .02, paired arm 4.52 p < 10⁻⁴, interaction p = .009; NoGo
unwarranted-lick rate 0.24 → 0.08 unchanged; 9/9 per-mouse lines in both stages. CONSEQUENCE:
the expert propagation is no longer "not detectable" — it is weaker and pairing-selective, so
"vanishing lick chain" → "weakening" (Fig. 4 legend), intro/§4/Discussion wording softened. The
old NoGo-only entry follows.** - **✓ Fig. 1g REBUILT (2026-09-01, resolving the predictor
audit; user decision — "the delay lick leads to more false alarms, not memory corruption; the
push prevents these false alarms"):** the old build's `licked` predictor pooled cue and test
licks (= the test lick on 96.5% of NoGo trials, near-circular with performance; its OR = 0.56
was carried entirely by the test lick — cue-only OR = 0.93 n.s.). The panel now shows the clean
propagation: predictor = the cue lick (`odr_choice`), outcome = P(lick at test); naïve OR =
3.10 p = .006 ∗∗, expert OR = 1.50 p = .42 n.s.; FA arm (unpaired) OR = 2.73 p = .090, paired
arm OR = 9.9 p = .001 toward a hit (the anti-corruption control); delay-lick rate 0.24 → 0.08.
§1 header, §2/§4/abstract clauses, caption, Methods and behavior.md all recast to "propagates
to false alarms". Downstream Discussion clause updated. - **Fig. 2a mid-delay bracket label
(audit):** the panel/caption say "5.5–6.3 s" but the sampled bins 36–38 map to ≈6.0–6.5 s under
the code's own bin→time convention (the 5.5 is the epoch onset before the +0.5-s index offset;
no trailing-integration code exists). Either relabel the bracket ≈6.0–6.5 s or document the
frame-timestamp convention that justifies 5.5. The substantive property (post-distractor,
pre-cue, pre-lick) holds under both mappings. - **Fig. 6k,l annotation (audit):** the panel
prints "n=10" (plotted OFF/ON pairs); the LMM is fit on 20 rows (5 mice × 2 stages × OFF/ON).
Relabel or reword. - **Estimator drift check (audit):**
`overlaps/fig_ccgp_matrices_pseudo.py`'s module pipeline lacks `class_weight='balanced'` while
`pca/decoders.py`'s docstring claims the two match — verify which estimator actually produced
the current `matrices_cache_acc_nopca.pkl` and align. - **Hard-coded figure literals (audit):**
Fig. 2g's "6.2% / 6.4%", all caption β/ρ/p strings, and the Fig-5 header stats are literals
(currently verified correct) — they will drift silently if caches are rebuilt; re-verify at
submission render. - **Minor code-comment rot (audit; comments only, no behavior):**
`exp_frame_states.py` docstring still says decision read = 57–62 (code: 60–66);
`main_panels.py` comments say KDE bins 48–53 / BINS_DELAY 21–53 (actual 45–53 / 18–53);
`fig_behavior_opto_main.py` docstring misstates why its depth window differs from the main
figure. PS null uses |cos| while the observed PS is signed (conservative); the plane-basis
half-split is not nested for the test/dist arms (indirect, disclosed); CCGD tensor CV is
unseeded (noted in Methods). - **Fig. 2 DECODE BUILD ADOPTED (2026-08-10, supersedes the PR
build below):** panels B/C/D now share one grid, DPA vs dual × mid-delay vs decision. **b**
cvPCA reliable spectra per set (leave-one-mouse- out jackknife 95% CIs; common 6-component
axis), **c** per-variable decoding power (held-out pseudo-trials along each variable's demixed
axis vs shuffle nulls, Kobak-style; incl. the hatched "gng ×" = Go/NoGo cross-decoded from the
DPA-state subspace, 0.61 at mid-delay), **d** η² matrices DPA-first, PC1–3 in both sets (PC4 row removed 2026-09-08), with
the boxed gng × column on DPA. Mid-delay (bins 36–38, pre-cue/pre-lick) replaces late delay in
b/c; d stays late-delay (footnoted). The PR bars + all-tasks spectra moved to **ED 3(a1)** and
render via `fig_dimensionality_main.py --pr` → `fig_dimensionality_main_pr.png`. §2 rewritten
accordingly (spectra + decoding numbers headline; PR quoted in the caveat with ED pointer).
Metric rationale (PR variance-weighting challenged → decoding adopted; dot-strip variant
rejected) and all numbers are logged in memory (`project_dimensionality`). - **Fig. 2 REPLACED
(2026-08-10, message-first FINAL):** the dPCA story build (`fig_dpca_story_main.py`) is
superseded by `pca/fig_dimensionality_main.py` → `figures/pseudo/dimensionality/
fig_dimensionality_main.png`, built around ONE message ("one dedicated axis per task variable —
the working memory is a line"): **a** trial-timeline + split-half cvPCA schematic, **b** cvPCA
reliable spectrum (delay, Naïve/Expert + shuffled null), **c** PR bars with split-level CIs,
Naïve+Expert overlaid (memory ≈1 · delay ≈2 · decision ≈3.3), **d** the η² PC-coding matrices,
Expert row of four (dual-delay · dual-decision · DPA-delay · DPA-decision). Everything
off-message moved out: shattering → ED 3(a0) + cited from §3; dPCA trajectory grid,
axis-mixing, linking plane, shared-memory scatter → **ED 9** (= `fig_dpca_story_main.png`, kept
rendering); old scree + marginal variance + Naïve η² matrices + per-task-set fits → ED 3.
Design iterations that got here (matrices are the real-data panel; a derived chips graphic was
rejected) are logged in memory. Data: `figures/pseudo/dimensionality/results.pkl` (merged
caches; CIs from `exp_dimensionality_ci.py`). Earlier "Fig. 2 flow-free — DONE" and "Fig 3
panels updated" notes below describe superseded builds. - **Overlaps split (2026-08-04; Fig 3
panels updated 2026-08-05):** the single overlaps figure was split into **Fig. 3** (manifold /
abstraction, `fig_overlaps_manifold.py` → `fig_overlaps_manifold.png`, panels a–e = code traces
/ within-vs-cross-task matrix / **shared action axis (Go/NoGo↔DPA-lick cross- decode)** /
**cross-context generalization summary** / CCGP-across-learning; the dPCA linking plane moved
to Fig 2g and the weak weight-cosine panel was dropped) and **Fig. 4** (push / repositioning,
`fig_overlaps_main_native.py` → `fig_overlaps_main_ab_dpaact.png`, push panels a–d = push
planes+depth / Δdepth↔Δacc coupling / Naïve FA-CR / action-code d′). A stale
`fig_overlaps_main_ab.png` (4-panel, 18-obs LMM coupling) still exists in the repo — do NOT
ship it; use only the current builds (the gallery Main tab was repointed to the `_dpaact`
file). - **Fig. 2 flow-free — DONE (2026-08-03):** `fig_dpca_story_main.py` regenerated without
the section-3 flow grid or the section-4 flow panels → 7-panel **a** schematic, **b** scree,
**c** per-task variance, **d** trajectory grid, **e** axis-mixing, **f** no-lick push
(p=0.012), **g** sample-memory preserved (p=0.10); title "The dPCA **geometry**…". Push stats
reproduce (−0.59, 8/9). The Main-tab `fig_dpca_story_main.png` now shows the flow-free build. -
**Titles left-aligned — DONE (2026-08-03):** the subplot titles in both new supplements were
already `loc='left'`; the coupling-battery's one centered element (a full-width suptitle
banner) resisted left-alignment under the tight-bbox save and was removed (its message lives in
the two left panel titles + the ED 5 legend). d′ figure unchanged (no centered element).

### Open drafting decisions
- **Fig. 1h framing:** "decoupled / suboptimal balance" (defensible, n.s. correlation) vs a stronger
  "trade-off" (not supported at n = 9). Draft uses the conservative version.
- **Fig. 2 sample-coding wording:** "preserved" (correct-trials, p = 0.10) vs "preserved and sharpens"
  (all-trials, p = 0.02). Draft uses "preserved".
- **"reuse the same manifold"** is carried by the near-orthogonal, temporally stable axes present in naïve
  + the pre-existing no-lick well; phrased as "factorized scaffold retained; state re-sculpted", NOT
  "identical manifold, pure translation" (decoder-axis reorganization is part of the push).
- **Figure numbering (UPDATED 2026-09-01, supersedes the 2026-08-04 resolution):** the modeling figure
  IS coming and takes **Fig. 5**; the opto figure is renumbered **Fig. 6** (all refs in this draft, the
  in-figure caption, and the gallery cards updated). Fig. 4 stays the overlaps push/repositioning figure.
  Order: Behavior (1) → Geometry (2) → One manifold (3) → Learning (4) → Modeling (5, TBD) → Opto (6).

### Framing guardrails (keep the thesis defensible)
- "low-dimensional **geometry** / rank-2 portrait", never "the dynamics are rank-2". Flow-field /
  attractor-dynamics claims are OUT of the paper (kept as "extra") — don't reintroduce "landscape",
  "gated deformation", "bistable" into the main text.
- near-orthogonality = the representational basis of compositionality (factorized code → combine without
  cross-talk) — the load-bearing, solid claim.
- the pre-existing naïve no-lick well is the strongest "structure already there, fine-tuned" evidence —
  lean on it for the reuse claim.
