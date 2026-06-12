# Shared Data Structures

## Mice (9 animals, all subprojects)
```python
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
```
- Day counts vary: JawsM01=4, JawsM12/ChRM23/ACC*=5, others=6
- `stage` / `learning`: 'Naive' (days 1–3), 'Expert' (days 4–last), defined by `days=['first','last']` in `set_options`

---

## Time bins (84 bins = 14 s at 6 Hz, T_WINDOW=0.0)
```python
BINS_BL    = options['bins_BL']       # 0–11   (baseline)
BINS_DELAY = options['bins_DELAY']    # 18–53  (full delay, 36 bins)
BINS_LATE  = BINS_DELAY[int(0.6*len(BINS_DELAY)):]  # ≈ 39–53 (last 40% of delay)
BINS_TEST  = options['bins_TEST']     # 54–59  (test epoch)
```

---

## CCGD data — overlaps subproject
**Files:** `data/overlaps/X_log_generalizing_overlaps_none_l1_ratio_0.0.pkl`
          `data/overlaps/labels_log_generalizing_overlaps_none_l1_ratio_0.0.pkl`

```
X_single shape: (16704, 2, 84, 84)
  dim 0: trials
  dim 1: [probas, dfs]  → always use index 1 (dfs = decision function values)
  dim 2: T_train (training time bin)
  dim 3: T_test  (test time bin)
```

```
y_single shape: (16704, 24)
Key columns: mouse, tasks, stage, target, odor_pair, laser,
             performance, odr_perf, response, odr_response,
             sample_odor, test_odor, dist_odor, licks, day
```

- `target`: 'sample' / 'choice' / 'test' — which decoder row this trial belongs to
- `tasks`: 'DPA', 'DualGo', 'DualNoGo' — task of the trial
- `odor_pair`: 0–3; pairs 0,1 = sample A (#332288 indigo); pairs 2,3 = sample B (#44AA99 teal)
- `laser`: 0 = laser off (use these), 1 = laser on (exclude)
- `performance`: 1 = correct on DPA task
- `odr_perf`: 1 = correct on GNG task (only valid for DualGo/DualNoGo)

---

## PCA data — pca subproject
**File:** `data/pca/single_traj_pca_TEST_Expert_standard_loo_correct_odor_pair.pkl`

```
traj_all shape: (9216, 10, 84)
  dim 0: trials
  dim 1: PCs (10 components)
  dim 2: time bins (84)
```

```
labels_pca key columns: sample_odor, dist_odor, test_odor, tasks, response,
  laser, day, choice, odr_choice, pair, odr_perf, odr_choice, odr_response,
  odor_pair, learning, performance, mouse, sample, test, distractor
```

- `choice`: DPA test response (lick/no-lick) — present for ALL trial types
- `odr_choice`: GNG lick during delay cue — ONLY DualGo/DualNoGo; NaN for DPA
- **`choice` ≠ `odr_choice`** — common source of confusion

---

## X_epoch normalization (canonical recipe — used in ALL overlaps scripts)
```python
# Select train epoch and average over training time bins; take dfs channel
X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)

# Per-mouse baseline normalisation
for mouse in ALL_MICE:
    m  = (y_single.mouse == mouse).values
    sd = X_ep[m][:, BINS_BL].std()
    if sd > 0:
        X_ep[m] /= sd
```

---

## Correct trial filter (overlaps scripts)
```python
idx_correct = (
    (y_single.laser == 0) &
    (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)
# All-laser-off (correct + incorrect):
idx_laser = (y_single.laser == 0)
```

---

## Axis conventions (decoders subproject)
- **Sample axis**: LR weight vector decoding `sample` (0=A, 1=B) on expert DPA trials; sample-B → positive
- **Lick axis**: LR weight vector decoding `odr_choice` on Dual trials (class_weight='balanced'), orthogonalised to sample axis; lick=1 → positive
- Both defined per-mouse from last 2 days
- DPA delay states sit on the **negative** (no-lick) side
