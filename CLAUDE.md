# CLAUDE.md — /home/leon/dual

## Session startup
Read these docs at the start of every session:
- `docs/meta_project.md` — paper overview and subprojects
- `docs/shared_data.md` — data structures, bins, normalization
- `docs/shared_feedback.md` — coding principles and past errors

Then load the subproject docs for the area being worked on:
- **Overlaps**: `docs/overlaps/overview.md`, `docs/overlaps/routines.md`, `docs/overlaps/feedback.md`
- **PCA**: `docs/pca/` (when populated)
- **Decoders**: `docs/decoders/` (when populated)

## Project structure
Three subprojects, all using the same dataset (9 mice, DPA + DualGo + DualNoGo tasks):
- **Overlaps** — `/home/leon/dual/overlaps/` — CCGD sample×choice analysis
- **PCA** — `/home/leon/dual/decode/` — pseudo-population PCA (Fig 2E)
- **Decoders** — `/home/leon/dual/decode/` — single-neuron decoders (Fig 3)

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
- Save both PNG (`dpi=300`) and SVG (`svg.fonttype='none'`) for every figure

## Behaviour
- Verify data structure and code behaviour empirically before asserting — see `shared_feedback.md`
- Before editing a script, read it in full
- After changing a script, run it and confirm no errors before reporting done

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
