# Behaviour — data structures, laser encoding, and learning-curve figures

Reference for the behavioural analyses (DPA + DualGo/DualNoGo). Covers **two distinct
datasets** that must not be confused, the on-disk `.mat` structure of the training
batches, how the `Trials` array decodes, how the laser is (and is not) encoded, and the
figures built from all of it.

Source pptx: `/home/leon/dual/Behavioral Data Description.pptx` (Tony lab, Xian, 2023-01-31).

---

## 1. Two behavioural datasets — DO NOT confuse them

| | **Recorded cohort** | **Behavioural training batches** |
|---|---|---|
| Path | `dual_task/dual_data/data/2Samples-DualTask-BehavioralData/` | `/storage/leon/dual_task/data/behavior/` |
| Animals | 9 recorded mice (`ChRM04/23`, `ACCM03/04`, `JawsM01/06/12/15/18`) | many behaviour-only mice, no recordings |
| Format | per-trial `AllTrials` struct (9 cols incl. a `laser` column) | `session_N.mat` (SerialData + block/trial arrays) |
| **Laser** | **trial-interleaved ON/OFF** (per-trial `laser` flag exists) | **every-trial silencing** — NO per-trial ON/OFF |
| Laser contrast | within-mouse ON−OFF | **between-group opto vs control** |
| Figures | `overlaps/fig_behavior_learning*.py`, `fig_behavior_laser_compare.py` | `overlaps/fig_behavior_learning_batch.py` |

The recorded cohort is loaded from the pickle
`overlaps/data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0[_laser_targets_choice].pkl`
(built upstream from `AllTrials`); it genuinely has interleaved laser, so the within-mouse
ON−OFF delta (`fig_behavior_learning_delta.py`) is valid there.

The training batches are a **separate every-trial-silencing design** (see §4). "On/off
trials" do **not** exist in them; the only laser distinction is the group (folder name).

### The four training batches
- `DualTask_DPA_vs_Single_DPA/` — 8 `DPA_mouse_*` (single DPA) vs 8 `Dual_mouse_*`; 16 days; **no optogenetics**.
- `DualTask-Silencing-ACC/` — 11 control + 10 opto; 16 days.
- `DualTask-Silencing-ACC-Prl/` — 9 control + 9 opto; 12 days.
- `DualTask-Silencing-Prl-ACC/` — 10 control + 12 opto; 11 days.

Each mouse folder holds `session_N.mat`, `N` = 0-indexed training day (daily, no gaps →
`day = N+1`). Group = folder-name prefix (`opto_`/`control_`, or `DPA_`/`Dual_`); each animal
is wholly opto or wholly control (between-animal grouping).

---

## 2. `session_N.mat` contents (79 variables)

Example: `DualTask-Silencing-ACC/opto_mouse_0/session_8.mat`.

**Identity — `DataID*` strings** (provenance metadata):
`DataID = "20200731-DualTask-Box1-M1-DualTask"` and sub-task variants `DataID1/2/3`
(`…DualTaskDPA`, `…DualTaskDRT`, `…PureDPA`). Format
`<YYYYMMDD=training day> · <protocol> · <Box#=rig> · <M##=unique animal> · <sub-task block>`.
The date advances one calendar day per session; `Box/M` is the original rig identity that
the pipeline renamed to `opto_mouse_*`/`control_mouse_*`. ("DRT" = the go/nogo distractor
task = GNG/ODR.)

**Per-trial outcome arrays `(N,4)`** — see §3: `Trials`(192 all DPA), `TrialsP`(64 pure),
`TrialsD`(128 dual DPA), `Trials1`(128 GNG), `TrialsD1`/`TrialsD2`(64 Go / 64 NoGo),
`CongTrials`/`IncongTrials`(64/64 congruent/incongruent distractor).

**Block summaries `(8 blocks, 6)` = `[block#, hit, miss, fa, cr, perf%]`** — read by
`get_perf_mice`: `Data`(all-DPA), `DataP`(pure), `DataD`/`DataD1`/`DataD2`(dual/Go/NoGo),
`Data1`(GNG), plus `Data_Cong*`/`Data_Incong*` and `Results_Cong*` `(8,7)`.

**Timestamps `(N,2)` = `[time_ms, odor_id]`**: `Sample`, `SampleP/D/D1/D2`, `Test`,
`FirstOdor`, `SecondOdor` (+`1` dual variants). Pure-vs-dual within `Trials` is recovered as
`isP = np.isin(Sample[:,0], SampleP[:,0])`.

**Reaction times `(N,2)`**: `ReactionTime_DPA`(192), `ReactionTime_DRT`(128), + Cong/Incong/D1/D2.

**Timing scalars (s)**: `Delay=10`, `FDelay=3`, `LDelay=3`, `ITI=8`, `ResponseDelay=1`,
`MeantrialLen=22`, `Odor≈1`, `Water`/`Water1`. `LaserPeriod=11` (which epoch is stimulated —
the delay).

**Counts**: `TrialNum*`(8,) trials/block; `AbolishTrials`(8,) aborted trials/block.

**Raw streams**: `SerialData (≈11334,5)` full event log (§4); `lickTime`, `laserTime` flat
timestamp logs (`laserTime` = the 213 onset + 213 offset events); scalar reward durations.

There is **no `(N,)` per-trial laser array and no `laser` column** in these files.

---

## 3. The `Trials` array `(192, 4)` — decoded & validated

| col (0-idx) | meaning | values |
|---|---|---|
| col0 | **sample** (first) odor identity | 1 / 2 (96/96) |
| col1 | **test** (second) odor identity | 1 / 2 (96/96) |
| **col2** | **outcome** ← use this | **1=hit, 2=miss, 3=FA, 4=CR** |
| col3 | trial-type code (deterministic from sample,test) | 1,2,3,4 (48 each) |

`col3` encodes the pair structure — the *cross* combos are the matching pair:

| sample→test | col3 | pairing | correct response |
|---|---|---|---|
| 1→2 | 1 | **paired** | lick (hit) |
| 2→1 | 2 | **paired** | lick (hit) |
| 1→1 | 3 | unpaired | withhold (CR) |
| 2→2 | 4 | unpaired | withhold (CR) |

Outcome respects pairing exactly: col3∈{1,2}→hit/miss only; col3∈{3,4}→FA/CR only.

**Decoding recipe (everything from col2):**
- `performance` (DPA correct) = `col2 ∈ {1,4}` (hit or CR)
- `pair`: paired = `col2 ∈ {1,2}`, unpaired = `col2 ∈ {3,4}`  (≡ col3∈{1,2} vs {3,4})

**`Trials1` (GNG)**: col3 = 1 (Go) / 2 (NoGo); col2 = GNG outcome (same 1/2/3/4 code);
`odr_perf = col2 ∈ {1,4}`. Row-aligned to the dual trials chronologically.

**Validation**: Trials-derived DPA perf = 60.9% matches the acquisition's own block-summary
`Data[:,5]` mean = 61.0%.

> ⚠️ The `Trials` outcome code (col2: 1hit·2miss·3FA·4CR) differs from the **SerialData**
> outcome code (§4: 7hit·6miss·5CR·4FA). Same events, different encodings.

---

## 4. `SerialData` and the laser — every-trial silencing

`SerialData (N,5)` columns (pptx 1-indexed → 0-indexed): `[0]=time_ms`, `[1]=85` (const board
byte), `[2]=marker1` ("Column 3"), `[3]=marker2` ("Column 4"), `[4]=170`. Events are
`(marker1, marker2)` pairs. Lick = `(0,1)`.

**Event codes differ by batch** (from pptx). ACC & Prl-ACC:

| event | code | event | code |
|---|---|---|---|
| Sample1 on/off | 16/8, 16/0 | Test1 on/off | 11/3, 11/0 |
| Sample2 on/off | 17/9, 17/0 | Test2 on/off | 12/4, 12/0 |
| Go odor on | 13/5 | Response cue | 15/7 |
| NoGo odor on | 14/6 | **Laser onset / offset** | **65/1 / 65/0** |
| DPA Hit/Miss/CR/FA | 7/3, 6/3, 5/3, 4/3 | ODR Hit/Miss/CR/FA | 7/1, 6/1, 5/1, 4/1 |

ACC-Prl uses different task codes (Sample1=90/12, Sample2=89/11, Go=10/2, NoGo=18/10,
Test1=15/7, cue=9/1) but the **same laser code 65/1, 65/0**. DPA-vs-Single has no laser codes.

**The laser is delivered on the delay of ~every trial, in both opto and control:**
- `65/1` (onset) and `65/0` (offset) are the **onset/offset of one ~9 s laser epoch** (the
  delay); their counts are exactly equal every session (they are pairs, not per-trial labels).
- Deserialising the stream into trials (grammar: first-odor onset → delay → distractor/cue →
  test → outcome) and checking each trial for a `65` epoch gives **100% ON** (191–213/213).
- Scan of **all 794 sessions** across the three silencing batches: laser ON% = **99% every
  session**, opto and control alike; **0 sessions** in the 30–70% (interleaved) range.
- No marker splits ~50/50; the laser-command slot (`marker2==11`) is always `65`, never a
  non-65 "off" variant.

**Raw `.ser` cross-check.** Each raw date folder holds a `<date>_DualTask_Box#_M#.ser` —
Java-serialized `ArrayList<int[5]>` with the *identical* `[time,85,marker1,marker2,170]`
structure; it is the source `SerialData` is built from (`.ser` ≈11397 events vs `.mat` 11334;
the `.mat` drops a few pre-session boundary events). Deserialised, the `.ser` shows the same
laser pattern (65/1≈214 onsets, 65/0≈213 offsets; 191/192 trials ON = 99%). No hidden
per-trial flag at the raw level either.

**Conclusion:** the training batches are an every-trial silencing design (opto = silenced
each trial; control = same light command, no opsin). No within-mouse ON−OFF exists →
the laser contrast is **between-group opto − control**.

---

## 5. Existing code that reads the batches (none decodes on/off)

- Notebooks `dual_data/org/behavior.org`, `dual_data/org/dual_data.org` load the batch
  `session_N.mat` via `src/licks/licks.py` and do **licks + performance only**.
- `licks.py::get_licks_and_times` parses `SerialData` for task events (samples, distractors,
  tests, hit/miss/cr/fa, licks) — **never references code 65 (laser)**.
- `licks.py::get_perf_mice` reads the block summaries and sets `df["opto"]` from the folder
  name (`"opto" in mouse`) — **group-level only, never per-trial**.
- `dual_data/org/behavior_dpa_gng.org` DOES build a per-trial `laser` column — but from the
  **recorded cohort** `2Samples-DualTask-BehavioralData` `AllTrials` (col 9), a different
  dataset. Not applicable to the training batches.

---

## 6. Figures

All in `overlaps/`; outputs under `overlaps/figures/overlaps/behavior/`. PNG @dpi=300 + SVG
(`svg.fonttype='none'`). Python: `/home/leon/mambaforge/envs/dual/bin/python` (cd into
`overlaps/` first).

**Recorded cohort** (`figures/overlaps/behavior/`):
- `fig_behavior_learning.py [--jaws|--chr|--acc|--on]` — 5-panel learning curves
  (A DPA-vs-GNG, B Go-vs-NoGo, C paired-vs-unpaired, D unpaired-by-task, E LMM β forest).
  Per-mouse/day accuracy proportions + random-intercept LMM; per-day stars via per-day LMM.
- `fig_behavior_learning_delta.py [--jaws|--chr]` — **within-mouse Δ(ON−OFF)** laser effect.
- `fig_behavior_laser_compare.py [--jaws|--chr]` — OFF-vs-ON summary + within-mouse LMM.
- `fig_behavior_learning_offon.py [--jaws|--chr]` — OFF vs ON **absolute** learning curves,
  the recorded-cohort analog of the batch `--ctrlopto`: 4 panels (A DPA perf, B GNG perf,
  C DPA unpaired; OFF grey / ON indigo) + panel D within-mouse LMM `perf ~ laser×day +
  (1|mouse)` laser-effect forest. Default emits pooled + `_jaws` + `_chr` (opsins split —
  opposite manipulations). Result: **no significant gross behavioural laser effect** (Jaws
  n=5: DPA p=0.40, GNG p=0.24, unpaired p=0.90; ChR n=2 → mean Δ only). Contrast with the
  batches' between-group silencing, which IS behavioural in ACC-Prl/Prl-ACC — the recorded
  laser moves the neural code, not gross behaviour.
- `fig_behavior_dpa_vs_gng.py` — per-animal **DPA perf (x) vs GNG perf (y)** scatter in the
  main-figure convention (tab10 per-mouse colours, ● Jaws / ▲ ChR / ■ ACC markers, white edge,
  y=x diagonal, across-animal regression line + Pearson/Spearman). Two panels Naive | Expert;
  emitted for laser OFF (`_off`, n=9) and ON (`_on`, n=7 — ACC mice have no laser trials).
  Result: DPA & GNG **co-vary across animals when Naive** (OFF r=+0.66 p=0.051, ρ=+0.55; ON
  r=+0.74 p=0.055) but **decouple by Expert** (OFF r=+0.10 ρ=+0.35 ns; ON r=+0.43 ρ=+0.57 ns).
  A shared "good-learner" axis early that dissolves as animals specialise.
  `--unpaired` restricts **both** axes to unpaired (`pair==0`) trials (DPA *and* GNG on the same
  trials — `pair` is a DPA property but the whole-trial subset is kept consistent). On unpaired
  trials the Expert slope flips slightly negative (OFF r=−0.37, ON similar) but stays ns — a
  ceiling artifact (experts pinned at DPA≈1.0 can only vary GNG-downward), not a real trade-off.
- `fig_behavior_dual_cost.py [--unpaired]` — **is DPA↔GNG a capacity trade-off? No.** Two y=x
  scatters (OFF, one dot/mouse/stage, open Naive → filled Expert joined by a line): **A** dual-task
  cost = DPA acc pure (x) vs dual (y); Expert points fall just below the diagonal (Δ=−0.030 paired
  **p=0.048**; Naive ns) → a small fixed cost. **B** trial coupling = DPA acc | GNG-error (x) vs |
  GNG-correct (y); Expert points sit **above** the diagonal (Δ=+0.097 **p=0.025**; Naive ns) → DPA
  is *better* on GNG-correct trials, opposite sign of a trade-off.
- `fig_behavior_pareto.py [--all] [--on]` — Pareto front made **explicit** (ringed = non-dominated,
  faded + shaded = dominated interior, grey staircase = frontier). Default unpaired-OFF; `--all` =
  all trials, `--on` = laser ON (n=7, ACC dropped). A point is non-dominated if no animal beats it
  on *both* DPA & GNG. Result: **dominated interior, not a frontier** — Expert OFF 3/9 on front
  (6/9 dominated), Expert ON 1/7 (JawsM18 dominates the field). Not a Pareto-optimal population.
- `fig_behavior_dual_cost_trials.py [--on]` — the cost analysis at the **trial level** (GEE
  logistic, each trial a data point, clustered by mouse; forest of odds ratios, 4 rows =
  {all,unpaired}×{Naive,Expert}). **Dual-task cost:** OR(dual)≈0.8, marginal (Expert OFF p=0.07).
  **Trial coupling:** OR(GNG-correct) is **>1 and significant in Expert** — OFF all OR=2.03
  **p=0.001**, ON all OR=2.54 **p<0.001**, ON unpaired OR=1.88 **p=0.002** (Naive ns throughout).
  Getting GNG right ~doubles the odds of DPA correct on the same trial = shared engagement, the
  OPPOSITE of a within-trial trade-off. (Note statsmodels `gee` is Gaussian-free logistic here.)

**Conclusion — DPA↔GNG is not a capacity trade-off; learning is NOT Pareto-optimal** (holds OFF
and ON). Consistent across all three levels: (1) between animals — dominated interior, best
learners improve on *both* (Naive positive coupling r≈+0.7, decouples by Expert, ns); (2) explicit
Pareto front — only 1–3 of 7–9 animals non-dominated, no descending envelope; (3) trial-by-trial —
DPA & GNG correctness **positively** coupled (Expert OR≈2, p≤.002), not negatively. There *is* a
small fixed dual-task cost (~2–3 pp / OR≈0.8, marginal) — a mild shared capacity overhead, not a
resource traded between tasks. The "trade-off-shaped" tilt on unpaired-Expert trials is a DPA
ceiling artifact, ns. Read-out: a **shared competence/engagement factor + small fixed dual cost**,
not efficient DPA/GNG resource allocation. Caveats: n=9 mice (between-animal tests underpowered →
lean on trial-level GEE); coupling is associational, not a causal capacity manipulation; ON = n=7.

### The behavioural main figure — `fig_behavior_main.py`

Publication-ready assembly (recorded cohort, laser OFF, 9 mice; loads the non-laser pickle,
`target=='choice'`, `laser==0`). `figures/overlaps/behavior/{png,svg}/behavior_main.*`. Layout =
`GridSpec(3,12)`; message-based panel titles _(titles REMOVED 2026-09-03 — see the Nature-style block at the end)_. Panels:
- **A** — schematics: (i) a setup cartoon, (ii) the DPA+GNG **task scheme** (`dual_task_scheme.svg`),
  (iii) the **Curriculum training** pipeline (`dual_training_scheme_vector.svg`, hand-authored vector,
  conventions colours: DPA red / GNG blue / dual orange; "shaping" = paired-trials-only task version,
  GNG has "training" not shaping). Cartoon is sized **smaller** than the task scheme.
- **B–E** — the five learning curves from `fig_behavior_learning.py` (helpers copied per repo
  convention): B DPA-vs-GNG, C Go-vs-NoGo, D paired-vs-unpaired, E unpaired-by-task; per-day LMM stars.
- **F** — LMM effect-size forest (condition + condition×day βs).
- **G** — **the intrusive lick propagates to the test** (REBUILT 2026-09-01 after the predictor
  audit; user framing: the delay lick leads to FALSE ALARMS — the chain the Fig-4 push suppresses):
  NoGo trials, P(lick at test) split by the CUE lick (`odr_choice`, the pure delay lick), Naive vs
  Expert, per-mouse lines + mean±SEM, GEE(test-lick ~ cue-lick, clustered by mouse):
  Naive OR=3.10 **p=.006 `**`**, Expert OR=1.50 p=.42 `ns`; FA arm (unpaired) OR=2.73 p=.090,
  paired arm OR=9.9 p=.001 toward a HIT (anti-corruption control); **lick × pairing interaction
  n.s. (Naive p=.608, Expert p=.367) → the pooled propagation OR covers the FA arm** (printed by
  the script; in the caption + Methods since the Codex-calibration pass); cue-lick rate 0.24→0.08.
  [Historical: the pre-2026-09-01 build split DPA accuracy by `licks` = (cue OR test lick) —
  near-circular (the test lick = 96.5% of the predictor); its Naive OR=0.56 p=.006 was carried
  by the test lick and is RETIRED. Do not quote it.]
- **H** — **experts reach a suboptimal balance**: per-mouse DPA-vs-GNG scatter (Naive→Expert), y=x,
  Pearson/Spearman + mean gap-to-optimal.

The panel-A cartoon is a **continuous-line B&W vector** traced from the original `~/dual/mouse.svg`
illustration by `overlaps/make_mouse_lineart.py`: rsvg-render → darkness-threshold (L<135) to its own
bold outlines → morphological close → erase baked-in labels + stray apparatus fragments → flip
horizontal so the mouse faces the task → **vectorise with potrace** (`potracer`, pure-python; `pip
install potracer` in env `dual`) → emit `mouse_lineart.svg` (traced path + vector labels
Head-fixed/Odor/Water). Re-run the script to regenerate; the figure just renders the SVG.

**Training batches** — `fig_behavior_learning_batch.py` (`figures/overlaps/behavior/batch/`):
- `--batch <name> --group <control|opto|DPA|Dual>` → 5-panel per-group curves. Panels use
  **data-adaptive per-panel y-limits**; chance line drawn only when in range; N_MIN=4.
- `--batch DualTask_DPA_vs_Single_DPA --compare` → single overlay: single-DPA vs dual-DPA vs
  dual-GNG performance vs day (LMM group + group×day). Result: dual training significantly
  impairs DPA (group p≈0.001, group×day p≈0.008); GNG is learned fastest.
- `--batch <Silencing-…> --delta` → **between-group Δ(opto − control)** silencing effect per
  condition vs day + panel-E LMM `perf ~ group×day + (1|mouse)` group-effect forest.
- `--batch <Silencing-…> --ctrlopto` → control-vs-opto **absolute** learning curves for 3
  metrics (DPA perf, GNG perf, DPA-unpaired), per-day Welch stars + per-panel LMM group test.

  **Silencing effect is batch-specific** (LMM group effect, opto−control at mean day):
  | batch | DPA perf | GNG perf | DPA unpaired |
  |---|---|---|---|
  | ACC | ns (p=0.57) | ns (p=0.63) | ns |
  | ACC-Prl | β=−0.06 **p=0.009**, group×day p=0.0003 | ns | β=−0.12 **p=0.014**, group×day p<0.001 |
  | Prl-ACC | ns (group×day p=0.067) | β=−0.05 **p=0.017** | — |

  I.e. ACC-Prl silencing impairs DPA (esp. unpaired), Prl-ACC silencing impairs GNG, ACC null.
  (Earlier "gross behaviour is null" was an overstatement from looking at ACC alone.)

Colours: DPA/paired `#d62728` red · GNG/Go `#1f77b4` blue · NoGo `#2ca02c` green ·
dual-DPA `#ff7f0e` orange (compare figure only).

### The behavioural OPTO figure — `fig_behavior_opto_main.py`

**Main Fig 6 since 2026-09-01** (was Fig 5 — Fig 5 is reserved for the upcoming modelling figure;
older entries below saying "Fig 5" mean THIS figure). Companion to the main figure; one unified
story about the **ACC→mPFC(Prl)** projection across
manipulation regimes. `figures/overlaps/behavior/{png,svg}/behavior_opto_main.*`. Layout =
a single **4 equal-height row** gridspec: **first row = scheme A (left, cols 0:6) + batch B (DPA
curve, cols 6:9) + E (LMM forest, cols 9:12), B & E `set_box_aspect(1)` → square**; then F–H (row 2),
I–K (row 3), L–N (row 4). **The batch GNG-spared curve (C) and DPA-unpaired curve (D) were removed**
(E's forest still carries all three metrics), so panel letters run A, B, E, F…N (no C/D). No bottom
caption strip (removed); message-titled panels and row banners _(BOTH REMOVED 2026-09-03 — Nature-style block at the end)_ over the recorded / overlaps / last rows
(the batch first row has none — panels are self-titled). Helpers copied inline from
`fig_behavior_learning_offon.py`, `fig_behavior_learning_batch.py --ctrlopto`, and
`plot_scatter_laser.py`, so those stay untouched. Panels:
- **A** — scheme from `~/dual/opto.png` (recorded-cohort design: hSyn-GCaMP6s imaging in mPFC +
  CaMKII-Jaws-tdTomato in ACC, 635 nm laser-on 50 % pseudo-random delay trials). Placed
  full-width (`aspect='auto'` scheme axis; its own baked-in `a`/`b`).
- **B** — training batch **ACC-Prl** DPA performance, chronic every-trial silencing, between-group
  opto vs control (9 v 9) vs day → **DPA impaired**.
- **C** — training batch **LMM group-effect forest** (opto−control per metric DPA / GNG / DPA-unp;
  `perf ~ group×day + (1|mouse)`; DPA β≈−0.06 `**`, unpaired β≈−0.12 `*`, group×day `***`). The
  GNG-spared curve and DPA-unpaired curve were **removed**; C's forest still summarises all three
  metrics. First row = **A, B, C** (panel letters run A,B,C,D…L — no gaps).
- **D, E** — recorded cohort, transient delay-only laser, **within-mouse ON vs OFF, Jaws inhibition
  only (n=5)**: DPA & GNG OFF/ON curves; per-day stars = one-sample ΔON−OFF. (The old within-mouse
  LMM laser forest — all CIs cross 0, no gross behavioural effect — was **dropped**; still in
  `fig_behavior_learning_offon.py`.)
- **F** — per-mouse choice-code depth, laser **OFF vs ON** (Jaws, A&B-pooled paired lines + group
  mean±SEM), in the recorded within-mouse row. The laser moves each animal's code (M06 up, M01/M15
  down, others flat) while the group mean is ~flat — this shift is the **x-axis of G–I**.
- **G** — **TRADE-OFF contrast (headline coupling stat), in the same row as H/I:** Δdepth vs
  `ΔDPA − ΔGNG`. The trade-off hypothesis (depth↑ → DPA↑ *and* GNG↓) makes one joint prediction —
  depth positively predicts (ΔDPA−ΔGNG) — pooling both arms into a single test. **Significant on the
  pre-committed trainLD_TEST axis, no window search: Pearson r=+0.53 p=0.016 (readout 45–53, updated
  2026-07-15; was r=+0.48 p=0.034 on the old 27–53 readout).** H and I are its two arms (I ΔGNG robust
  `*`; H ΔDPA the same-signed positive *trend*, n.s. on its own). Chosen over two individually-starred
  H+I panels because forcing H's `*` needs the argmax 51–56 window + Spearman specifically = window×stat
  selection a reviewer would flag. **This panel REPLACED the old trial-level GEE readout-vs-silencing
  forest** (removed from the script; git history: DPA readout preserved OFF OR=1.41 / ON OR=1.46).
  **Mixed-model caveat (`_gi_lmm`, added 2026-07-15 per user request to mirror the main non-opto fig's
  panel-C model `Δacc ~ Δdepth + C(sample) + (1|mouse)`):** computed and PRINTED to stdout, NOT drawn
  (drawn stat stays the between-animal correlation). Under it G drops to n.s. (β=+0.021 p=0.108) while
  I survives (β=−0.013 p=0.018); **but note this LMM tests the WITHIN-mouse slope** — for a
  between-animal coupling the random intercept absorbs the signal (see the H/I stats note below), so it
  is a conservative lower bound, not the primary test. The proper clustering-robust check is the
  per-mouse-mean correlation (I n=5 r=−0.80). Adding C(stage) doesn't rescue G (p=.147) and softens I
  to p=.052.
- **H, I** — overlaps causal coupling, laser ON−OFF, **Jaws only**, depth = DPA choice-code on the
  **trainLD_TEST** axis (bins **45-59** = LD+TEST, the main-overlaps-figure convention), readout
  window **45-53 (LD epoch, pre-test)** — narrowed 2026-07-15 from the old broad 27-53 and **unified
  with the main overlaps figure** (the coupling is readout-robust: `*` at every window, end-of-delay is
  slightly cleaner). Also sets F's & G's depth (F–J all on one axis). Points = **5 Jaws × {Naive ▲,
  Expert ●} × odor A/B = 20** (A&B joined within each mouse×stage). **Square** panels, **Pearson star,
  Spearman shown** (agrees). H (ΔDPA) n.s. (r=+0.34 p=0.15); **I (ΔGNG) `*` (r=−0.65 p=0.002,
  ρ=−0.62 p=0.003).**

  **Stats note (important — this is a *between-animal* coupling):** mice with a bigger
  laser-induced Δdepth show a bigger Δaccuracy. A `(1|mouse)` random-intercept LMM is the
  **wrong** model here — it absorbs the between-mouse variance (the signal) into the intercept
  and tests only the within-mouse slope; do NOT use it for H/I (contrast K/L, where the effect
  *is* within-mouse OFF-vs-ON so the random intercept is correct). The coupling is robust across
  every slicing (Expert-10 r=−0.78 p=0.008; 20-pt r=−0.61 p=0.004; **per-mouse-mean r=−0.80**,
  n=5 p=0.10) and Spearman agrees.

  **Axis choice (trainLD_TEST) — why, and the honest caveat:** the depth↔ΔGNG coupling
  *strengthens monotonically* as the training window slides from early delay toward the choice
  epoch — per-mouse-mean r: trainDELAY(18-53) −0.14 → trainLD(45-53) −0.67 → **trainLD_TEST(45-59)
  −0.80** → trainLDTEST05(51-56) −0.93. This is interpretable (the code is most behaviour-predictive
  near the decision), but picking the *argmax* window (the narrow 51-56 boundary) would be post-hoc
  cherry-picking. So the headline uses **trainLD_TEST** = the pre-committed main-overlaps-figure
  axis (not the max), and the monotonic window-sweep is the honest way to report the boundary result.
  ΔDPA stays n.s. on every axis (specificity holds). Readout on **45-53** (LD epoch) keeps depth in the
  delay (pre-test, ends bin 53), so it is not circular despite the axis being trained through TEST.
  (The per-mouse-mean window-sweep r's above were computed at the prior 27-53 readout; the monotonic
  train-axis trend is unchanged.)
- **`--eqnorm`** (added 2026-07-15, mirrors the main figure's flag) — equal-weight normalization:
  `_depth_on_axis` divides each mouse's choice-code depth by its WHOLE-TRIAL std (signal scale) instead
  of BASELINE std, so the per-mouse Δdepth (F–J) isn't SNR-weighted toward high-baseline-SNR mice. d′
  panels K/L are scale-invariant → unchanged. Output → separate `figures/overlaps/behavior/eqnorm/`.
  All headline coupling stats survive (G p=0.019, I p=0.001, H n.s.).

Row 4 (last row) — **behavioural balance under silencing + code discriminability** (recorded, Jaws n=5):
- **J** — **DPA vs GNG performance in laser-ON trials** (the balance plane of the non-opto main
  figure, its panel H), **10 pts = 5 Jaws × {Naive ○, Expert ●}**, Naive→Expert joined per mouse;
  optimal corner starred, unity dashed. Descriptive (ON across-point r=+0.44 p=0.20). Uses
  `performance` (DPA) / `odr_perf` (GNG) on target==choice rows, laser==1. (Replaced the trade-off
  panel here, which moved up to **G** alongside H/I.)
- **K** — **d′ laser ON vs OFF scatter** of the **DPA memory code**: sample-axis discriminability
  **d′ (odor A vs B)** at **late delay** (bins_LD 45–53). x = d′ OFF, y = d′ ON, dashed **unity
  line** = spared, square. **10 points = 5 Jaws × {Naive ○, Expert ●}.**
- **L** — **d′ ON vs OFF scatter** of the **GNG code**: **choice-axis** discriminability
  **d′ (Go vs NoGo)** at **mid-delay** (bins_MD 33–38, the Go/NoGo cue). The choice axis separates
  Go/NoGo (d′ peaks ≈0.56 at mid-delay) so no separate distractor decode is needed. Same 10-point
  Naive/Expert design.

**Stat (K,L)** = LMM **`d′ ~ laser + stage + (1|mouse)`** (mouse random effect handles the
repeated Naive/Expert + OFF/ON measures; converges). **Laser n.s.: sample p=0.34, GNG p=0.74 →
discriminability spared.** IMPORTANT — a trial-level `signal×laser` mixed model looks significant
(p<.001) but that is **pseudoreplication**: with a random *slope* for the effect it collapses
(sample p=0.34 non-conv, GNG p=0.10). The per-mouse effect is heterogeneous in sign, so the
mouse-level LMM (n.s.) is the honest read — do NOT headline the trial-level interaction.

Windows: A/B read at late delay (memoranda held to the comparison), Go/NoGo at mid-delay (cue
onset) — both inside the laser/delay window. The **sample axis is decoded separately**
(`run_overlaps.py --scaler none --no-raw --with-laser --targets sample` →
`X_..._laser_targets_sample.pkl`, ~10 min, gitignored); it validly separates A/B (d′≈1.2),
**unlike the `depth`/choice DV** — a neurometric decomposition of `depth` was tried and rejected
(`depth` at 27–53 separates neither stimulus nor choice, d′≈0; it is a code *engagement/quality*
signal that predicts correctness (L), not a discrimination axis). The earlier *behavioural*
d′/criterion SDT (both spared, all n.s.) is superseded by M/N (kept in git history as a control).

Row-4 geometry story: the transient laser moves the code's **position** (I, push) while sparing
its **readout** (L) and its **discriminability** (M,N) — three distinct geometric properties.

Design notes (settled after iteration — see [[project_behavior_opto_figure]]): F–J are
Jaws-only by request; the batch (B–E) is a different chronic cohort, so the figure deliberately
mixes a within-mouse Jaws story (F–H) with a between-group batch story (B–E). The recorded
Jaws/ChR laser and the ACC-Prl batch target the **same ACC→mPFC(Prl) projection**
(`docs/overlaps/laser_onoff.md`). Panels I/J can also reproduce main-figure panel E of
`fig_overlaps_main_ab.svg` (all 7 mice, trainLD_TEST, Spearman star) by widening the mouse set +
axis — the committed version is the Jaws/trainLD/Pearson variant.

### Trial-history supplements — `fig_behavior_history.py` + `fig_behavior_history_batch.py`

Does the CURRENT trial's performance depend on the PREVIOUS trial's task type (DPA / Go /
NoGo)? Previous task is taken **within session** (first trial of each session dropped, no
cross-session carry-over); one row per trial. Two outcomes: DPA memory (`performance`,
hit/CR, defined every trial) and GNG distractor (`odr_perf`, dual trials only). Stat =
trial-level logistic **GEE clustered by mouse** (Exchangeable), controlling for the previous
trial's own DPA outcome (`prev_perf`); `prev_dual` = previous task was Go or NoGo.

**Recorded cohort — `fig_behavior_history.py`** (9 mice, laser OFF, `target=='sample'` slice).
The recorded design is genuinely **interleaved** (transition rate 0.70, P(prev|cur)≈⅓ each),
so previous-task is a clean trial-lag contrast. 8 panels, two outcome rows (A–D DPA memory,
E–H GNG): grid (current×previous), marginal, Go-current zoom, GEE forest. Results:
- **DPA memory** — marginal history effect is **null** (by previous task 75.4/75.6/75.4%,
  prev-dual OR=1.01 p=0.90). The one signal is **specific to Go-current trials**: DPA memory
  drops after a dual trial (77.3→~72%), **prev-dual OR=0.81 p=0.047** (prev-outcome-controlled,
  so not win-stay); between-mouse it is a **trend** (per-mouse paired Wilcoxon p=0.10, 6/9). No
  effect on DPA-current or NoGo-current trials.
- **GNG accuracy** — **history-independent** (Go- and NoGo-current both n.s.; ~80–81% regardless
  of previous task).
Output `figures/overlaps/behavior/{png,svg}/behavior_history.*`.

**Training batches — `fig_behavior_history_batch.py`** (control mice pooled across the 3
silencing batches, 30 mice, light-only). ⚠️ **The batches CANNOT be analysed the same way**:
they use a **blocked design (~4 pure-DPA then ~8 dual, repeating)**, so previous-task is
**confounded with pure↔dual block position** — a Go-current trial preceded by DPA is *always*
the first dual trial after a pure block (P(prev=DPA | Go-current)=0.19 vs P(prev=DPA | DPA-
current)=0.64). The naive `prev_dual` contrast is collinear with a task-switch and gives a
spurious huge "Go OR=0.59\*\*\*" — **do not use it** (an earlier `_batch_pooled` version of the
naive figure was deleted). Instead the batch is a **switch-cost figure** (4 panels):
- **A** block structure (example session); **B** switch cost on DPA memory: switching **into
  dual** costs memory (1st-dual-after-pure vs mid-block **OR=0.90 p<0.001**) while switching
  **into pure** does not (OR=1.03 p=0.44) — asymmetric; **C** GNG accuracy **warms up**
  monotonically across the dual block (93.7→95.8%, 1st-dual switch OR=0.88 p=0.013); **D** the
  only unconfounded lag contrast, **prev-Go vs prev-NoGo on dual trials** (matched block
  position; controls pos + current task + prev outcome) — **null on DPA memory** (OR=0.98
  p=0.44), **weak on GNG** (OR=0.84 p=0.019, would not survive correction).
Flags `--batch <name> --group <control|opto>` analyse a single batch/group. Output
`figures/overlaps/behavior/{png,svg}/behavior_history_batch[_<tag>].*`.

**Convergence:** the recorded Go-cost (OR=0.81) and the batch into-dual switch-cost (OR=0.90)
point the **same way** — a preceding distractor-containing trial mildly taxes current DPA
memory — but both are small and the batch's blocked design precludes a clean trial-lag test,
so state the design limitations explicitly if this goes in the paper.

**Does ACC→Prl silencing amplify the into-dual switch-cost? No** (tested 2026-07-18, ACC-Prl
batch, 9 control vs 9 opto, dual-current trials, GEE `perf ~ switch*opto` clustered by mouse).
The **group×switch interaction is null and the wrong sign** for amplification: DPA memory
β(switch:opto)=+0.064, p=0.24 (control switch cost −2.2 pp / OR=0.90 p=0.02; opto −0.9 pp /
OR=0.96 p=0.24 — if anything *smaller* under silencing); per-mouse switch cost control +2.1 pp
vs opto +0.9 pp, Mann-Whitney p=0.33; GNG β(switch:opto)=+0.10, p=0.27. What silencing does is
depress dual DPA memory **uniformly** (opto main effect OR=0.78, ~64% vs ~71% — the documented
ACC-Prl DPA impairment), landing equally on 1st-dual and mid-block trials rather than
concentrating at the pure→dual transition. Caveat: 9 v 9 rules out only a *large* (~≥2–3 pp)
interaction. Not figured (clean negative); numbers logged here.

---

## 7. Key statistical stance (see also `docs/shared_feedback.md`)

Per-mouse/day accuracy proportions + Gaussian LMM (statsmodels `MixedLM`), random intercept
per mouse: `perf ~ C(condition)*day_centred + (1|mouse)`. Per-day stars = per-day MixedLM
`perf ~ C(cond) + (1|mouse)`, uncorrected/exploratory, N_MIN=4. Rejected along the way:
per-day Wilcoxon (multiplicity), binomial GEE (too conservative with 9 clusters),
`BinomialBayesMixedGLM` (VB posterior overconfident, p≈1e-12). LMM RE variance sits near the
boundary → treat as mildly anti-conservative.

## 2026-08-31 in-figure captions (Figs 1 & 6 — the opto figure, numbered Fig 5 at the time)
Both behaviour figures now draw a justified caption below the panels (same mechanism as Figs 2/3:
`pca/figcaption.draw_justified`; caption text lives in each script's `CAP_PARAS`). Fig 1's old
one-line footnote was folded into the caption title. Fig 5 (now Fig 6) kept its row banners until 2026-09-03 (removed since); its caption
carries the header's exact stats (trade-off r=+0.53 p=.016 with the mixed-model caveat printed;
K/L LMM p=.34/.74) and appends a [BUILD VARIANT] tag on the axis/norm variant builds (as does
Fig 4). The poster build skips the caption.
- 2026-08-31 (later): both captions rewritten to the house caption standard (claim+roadmap
  title; per panel what/how/shows/stats; shared vocabulary; see memory feedback_caption_style)
  — numbers unchanged.

### 2026-09-03 — Nature-style pass (applies to every main figure; supersedes older layout notes above)
- **No claim-sentence panel titles** on the mains any more (user decision "go for it", 5-figure sweep,
  commit 22d0d4a): the legend's first sentence and its `a,` entries carry the claims; panels keep only
  orientation labels (condition / window / axis names). Fig 6's row banners are gone as well; its b/d/e
  panels carry bare "DPA"/"GNG" labels so they self-identify. Older mentions of "message-based titles"
  or "row banners" in this file are historical.
- **Panel letters are lowercase bold** (Nature final-artwork spec); in-text refs use `Fig. 1g` style.
- **Legends are Nature-format** ("Figure N | title." then `a, …` entries), geometry-voiced and in the
  human-voice register; they live in each script's `CAP_PARAS` AND in the draft's "## Figure legends"
  section — edit both together. The captioned PNG is the working/gallery build; `--nocap` is the
  submission build and `make_submission_figs.py` exports 183 mm vector PDFs.
- **Print scale**: each main defines `PS` (Fig 1 1.45 · Fig 2 1.15 · Fig 3 1.30 · Fig 4 1.10 via
  `main_panels.py` · Fig 6 1.35) so printed text lands at 5–7 pt. Convention in CLAUDE.md.
