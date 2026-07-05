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
`GridSpec(3,12)`; message-based panel titles. Panels:
- **A** — schematics: (i) a setup cartoon, (ii) the DPA+GNG **task scheme** (`dual_task_scheme.svg`),
  (iii) the **Curriculum training** pipeline (`dual_training_scheme_vector.svg`, hand-authored vector,
  conventions colours: DPA red / GNG blue / dual orange; "shaping" = paired-trials-only task version,
  GNG has "training" not shaping). Cartoon is sized **smaller** than the task scheme.
- **B–E** — the five learning curves from `fig_behavior_learning.py` (helpers copied per repo
  convention): B DPA-vs-GNG, C Go-vs-NoGo, D paired-vs-unpaired, E unpaired-by-task; per-day LMM stars.
- **F** — LMM effect-size forest (condition + condition×day βs).
- **G** — **intrusive licks impair DPA early**: NoGo trials, no-lick vs intrusive-lick DPA accuracy,
  Naive vs Expert, per-mouse lines + mean±SEM, GEE(DPA-correct ~ lick, clustered by mouse) OR/p with
  significance brackets (Naive OR=0.56 **p=.006 `**`**; Expert OR=0.76 p=.50 `ns`).
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

Companion to the main figure; one unified story about the **ACC→mPFC(Prl)** projection across
manipulation regimes. `figures/overlaps/behavior/{png,svg}/behavior_opto_main.*`. Layout =
nested gridspec (full-width scheme banner over a 3-row body), message-titled panels, row
banners naming the design (within-mouse vs between-group). Helpers copied inline from
`fig_behavior_learning_offon.py`, `fig_behavior_learning_batch.py --ctrlopto`, and
`plot_scatter_laser.py`, so those stay untouched. Panels:
- **A** — scheme from `~/dual/opto.png` (recorded-cohort design: hSyn-GCaMP6s imaging in mPFC +
  CaMKII-Jaws-tdTomato in ACC, 635 nm laser-on 50 % pseudo-random delay trials). Placed
  full-width (`aspect='auto'` scheme axis; its own baked-in `a`/`b`).
- **B–E** — training batch **ACC-Prl**, chronic every-trial silencing, between-group opto vs
  control (9 v 9): DPA impaired / GNG spared / DPA-unpaired deficit / LMM group-effect forest
  (`perf ~ group×day + (1|mouse)`; DPA β≈−0.06 `**`, unpaired β≈−0.12 `*`, group×day `***`).
- **F–H** — recorded cohort, transient delay-only laser, **within-mouse ON vs OFF, Jaws
  inhibition only (n=5)**: DPA & GNG OFF/ON curves + within-mouse LMM forest
  (`perf ~ laser×day + (1|mouse)`) — all CIs cross 0 → **no gross behavioural effect**.
- **I** — per-mouse choice-code depth, laser **OFF vs ON** (Jaws, A&B-pooled paired lines +
  group mean±SEM). Shows the laser moves each animal's code (M06 up, M01/M15 down, others flat)
  while the group mean is ~flat — the shift here is the x-axis of J/K, and explains "code moves
  per-mouse yet no gross behavioural effect" (panel H).
- **J, K** — overlaps causal coupling, laser ON−OFF, **Jaws only, A&B taken as independent
  points** (5 mice → 10 pts, all trials), depth = DPA choice-code late-delay on the **trainLD**
  axis (bins 45-53), late-delay test-time window 27-53. **Square** panels (`set_box_aspect(1)`),
  Pearson-based star. J (ΔDPA) n.s. (r≈+0.39); K (ΔGNG) `*` (r≈−0.66 p≈0.037).

Row 4 — **mechanism / robustness** (recorded, Jaws n=5, Expert):
- **L** — trial-level **GEE logistic** `accuracy ~ depth_z`, cluster-robust by mouse
  (exchangeable), OR per within-mouse SD of depth, **fit SEPARATELY for laser OFF vs ON**
  (side-by-side). The opto-specific test: is the code→behaviour readout changed by silencing?
  **No — DPA readout is preserved: OFF OR=1.41 p=0.007 `**`, ON OR=1.46 p=0.014 `*`** (nearly
  identical). GNG null both (OFF 0.98, ON 1.01 n.s.; GNG arm regresses GNG accuracy on the
  DPA-choice-axis projection measured on GNG trials). Distinct from the OFF-only depth↔DPA scatter
  (`plot_scatter_perf.py`) — here the point is that silencing leaves the mapping intact.
- **M, N** — signal-detection **d′** (sensitivity) and **criterion** (bias) per mouse,
  loglinear-corrected, laser OFF vs ON, paired-t on ΔON−OFF. **All n.s.** (d′: DPA p=0.19, GNG
  p=0.37; c: DPA p=0.66, GNG p=0.99). The transient laser spares BOTH sensitivity and bias — the
  "shifts bias not sensitivity" hypothesis is **not** supported; reinforces panel H.

Design notes (settled after iteration — see [[project_behavior_opto_figure]]): F–J are
Jaws-only by request; the batch (B–E) is a different chronic cohort, so the figure deliberately
mixes a within-mouse Jaws story (F–H) with a between-group batch story (B–E). The recorded
Jaws/ChR laser and the ACC-Prl batch target the **same ACC→mPFC(Prl) projection**
(`docs/overlaps/laser_onoff.md`). Panels I/J can also reproduce main-figure panel E of
`fig_overlaps_main_ab.svg` (all 7 mice, trainLD_TEST, Spearman star) by widening the mouse set +
axis — the committed version is the Jaws/trainLD/Pearson variant.

---

## 7. Key statistical stance (see also `docs/shared_feedback.md`)

Per-mouse/day accuracy proportions + Gaussian LMM (statsmodels `MixedLM`), random intercept
per mouse: `perf ~ C(condition)*day_centred + (1|mouse)`. Per-day stars = per-day MixedLM
`perf ~ C(cond) + (1|mouse)`, uncorrected/exploratory, N_MIN=4. Rejected along the way:
per-day Wilcoxon (multiplicity), binomial GEE (too conservative with 9 clusters),
`BinomialBayesMixedGLM` (VB posterior overconfident, p≈1e-12). LMM RE variance sits near the
boundary → treat as mildly anti-conservative.
