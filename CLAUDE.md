# CLAUDE.md — /home/leon/dual

## Session startup
Read these docs at the start of every session:
- `docs/meta_project.md` — paper overview and subprojects
- `docs/shared_data.md` — data structures, bins, normalization
- `docs/shared_feedback.md` — coding principles and past errors

Then load the subproject docs for the area being worked on:
- **Overlaps**: `docs/overlaps/overview.md`, `docs/overlaps/routines.md`, `docs/overlaps/feedback.md` —
  for the main figure start with `docs/overlaps/main_figure.md` (full reproduction guide: hypothesis,
  routines, method, results, caveats); `docs/overlaps/laser_onoff.md` is the laser ON−OFF causal analog
- **PCA**: `docs/pca/README.md` (index) — for the dPCA story main figure start with
  `docs/pca/story_figure_reproduction.md` (full reproduction guide: hypotheses, routines, math, results,
  caveats); also `story_figure_methods.md`, `story_figure_review.md`, `flows_handoff.md`
- **Decoders**: `docs/decoders/` (when populated)
- **Behaviour / opto**: `docs/behavior.md` (learning curves, ACC→Prl silencing, laser ON−OFF figure)

**When resuming in-progress work, read the memory FIRST.** The "where I left off / what's settled /
resume from here" state lives in the file-based memory (`/home/leon/.claude/projects/-home-leon-dual/memory/`
— the `MEMORY.md` index + per-topic files like `project_overlaps_main_native.md`, which carry settled
conclusions, resume points, and dead-ends), plus any handoff doc (e.g. `docs/pca/flows_handoff.md`). Read
the relevant memory file(s) and handoff doc for the current state BEFORE the reference/reproduction docs,
and treat memory + `overview.md` as authoritative when a reproduction guide looks stale.

## Project structure
Paper: **mPFC population geometry (Dual task)** — sample identity and lick action on near-orthogonal axes;
with learning the DPA delay state is pushed along the lick/action axis into the no-lick region without
disrupting sample coding, and depth↔behaviour link. Full hypothesis + current findings: `docs/meta_project.md`.

All areas share the same dataset (9 mice, DPA + DualGo + DualNoGo tasks). Directories (note: `pca/` and
`decode/` are DISTINCT — don't conflate):
- **Overlaps** — `/home/leon/dual/overlaps/` — CCGD (cross-generalising decision codes): sample / choice(lick) /
  test / GNG decoder axes; the no-lick push, the shared action code, depth↔accuracy. Docs `docs/overlaps/`.
- **PCA / dPCA flows** — `/home/leon/dual/pca/` — pseudo-population PCA + dPCA + latent flow fields
  (Fig 2E and follow-ups). Docs `docs/pca/` (resume from `flows_handoff.md`).
- **Decoders** — `/home/leon/dual/decode/` — single-neuron / population decoders on the sample & lick axes
  across learning (Fig 3). Docs `docs/decoders/` (when populated).
- **Behaviour / opto** — `/home/leon/dual/behavior/` — learning curves + the ACC→Prl silencing story and the
  within-mouse laser ON−OFF figure. Docs `docs/behavior.md`.

## Python environment
```bash
/home/leon/mambaforge/envs/dual/bin/python
```
Always `cd` into the script's directory before running (scripts use relative paths like `../data/`).

## Key conventions
- Always import with `sys.path.insert(0, '/home/leon/dual/')` before project imports
- Per-mouse BL normalisation applied after X_epoch averaging — see `shared_data.md`
- Sample A = odor_pairs [0,1] (#332288 indigo), Sample B = [2,3] (#44AA99 teal)
- Condition titles: DPA / Go / NoGo (strip "Dual" prefix in figures)

## Figure style — matplotlib conventions (SUPPLEMENTS MUST MATCH THE MAIN FIGURES)
Every overlaps/opto figure AND every supplement uses ONE shared style. Do NOT hand-pick larger fonts,
bold panel titles, or dpi=300 for supplements — copy this block verbatim (it is the header of
`overlaps/fig_overlaps_main_native.py`). Nature-Neuroscience print typography: 6–8 pt at final size, thin rules.
```python
import seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook')          # MUST come AFTER importing src.common.plot_utils (which sets
sns.set_style('ticks')               #   "poster" at module level → huge ticks; set_style/rcParams alone don't undo it)
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8                          # panel titles: ax.set_title(..., loc='left', fontsize=TITLE_FS)  — NOT bold
```
Rules that go with it:
- **Panel titles** left-aligned, `fontsize=TITLE_FS` (8), NOT bold. **Panel letters** (A/B/C…) are the only
  bold text: `fig.text(..., fontsize=11, fontweight='bold')`.
- **Significance markers**: `fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55'`
  (`*` when p<.05 else `n.s.`). **Stats text** (β/ρ/p lines): `fontsize=6.5, color='0.3'`.
- **Per-mouse colour** = `sns.color_palette('tab10')` keyed by `ALL_MICE` (same mouse = same colour across
  panels); fill = category (sample A solid / B open, or cr/fa); scatter `s≈28–42`, thin `linewidths≈0.6–1.0`.
- **Save BOTH** PNG and SVG for every figure (`savefig.dpi=400`; SVG carries `svg.fonttype='none'`, and `*.svg`
  is gitignored so only the PNG is committed). Axis/other font sizes: labels 8, ticks 7, small annotations 6–6.5.

## Behaviour
- Verify data structure and code behaviour empirically before asserting — see `shared_feedback.md`
- Before editing a script, read it in full
- After changing a script, run it and confirm no errors before reporting done

## Logging & keeping docs current (do this from time to time, not only at session end)
- **Auto-log to memory as you go.** Write settled conclusions, gotchas, decisions, and dead-ends to the
  file-based memory (`/home/leon/.claude/projects/-home-leon-dual/memory/`) at natural checkpoints —
  whenever a result is settled, a normalization/axis/window is chosen, or a wrong path is ruled out —
  not just when asked. Update the existing memory file for a topic rather than duplicating it.
- **Update the docs when findings change**, in the same pass. Don't let a figure's script and its docs
  drift apart across a long exploration.
- **When you touch the docs, sweep the WHOLE doc set for that area, not just the one file you last
  edited.** For overlaps that means checking ALL of `docs/overlaps/` (`overview.md`, the reproduction
  guide `main_figure.md`, `routines.md`, `feedback.md`, `laser_onoff.md`) plus `docs/shared_*` — and the
  memory index — for claims the change makes stale. The authoritative source (`overview.md` + memory) and
  the reproduction/routines guides must AGREE; if an older guide describes a superseded build, at minimum
  add a staleness pointer to the authoritative file rather than leaving wrong numbers.
- Before finishing a task that changed a figure/analysis, do a quick pass: memory updated? every relevant
  doc updated or redirected? If unsure which docs are affected, `grep` the changed concepts across
  `docs/` and check each hit.
- NOTE: the "delegate all doc writing via ask-kimi" workflow below is currently BROKEN (no API key) —
  edit the docs directly.

-------------------------------------------------------------------------------------------

## Cheap-Worker Delegation Tools (Token Saving)

Three CLI tools delegate bulk I/O to a cheap worker model. Use them to save tokens.

### ask-kimi — bulk reading
For reading files >400 lines, or when you'd otherwise read 3+ files:

```bash
ask-kimi --paths <file1> <file2>... --question "<specific question>"
```

Returns a structured summary. Use that instead of reading files yourself.
Only read files directly when you need to make edits to specific lines.

### kimi-write — boilerplate generation
For generating tests, config files, docstrings, or repetitive code patterns:

```bash
kimi-write --spec "<what to write>" --context <existing-similar-file> --target <output-path>
```

Then review the output and edit only what needs fixing.

### extract-chat — chat transcript extraction
Extracts human-readable text from Claude Code JSONL transcripts:

```bash
extract-chat <session.jsonl> -o /tmp/chat.txt
```

### Security
- **Never print, display, or repeat the value of any variable or file containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD`** — extract and use silently only

### Documentation workflow (MANDATORY)
**NEVER write documentation directly. Always delegate:**

1. Extract chat: `extract-chat <latest-session.jsonl> -o /tmp/chat.txt`
2. Ask worker to read chat + existing docs and suggest updates:
   `ask-kimi --paths /tmp/chat.txt <doc-files> --question "read chat, give exact changes for docs"`
3. Apply the worker's changes via Edit tool

### When NOT to delegate
- Tasks under ~2000 tokens of work (delegation overhead isn't worth it)
- Architectural decisions, debugging, safety-critical code
- Anything requiring careful reasoning
- When exact line numbers are needed for editing
