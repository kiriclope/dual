# Nature Neuroscience house style — corpus-derived guide

> Built 2026-09-04 from the eight Nature Neuroscience papers this manuscript cites, read in full
> where open access allowed: Golub 2018, Libby & Buschman 2021, Kaufman 2014, Musall 2019,
> Yang 2019, Driscoll 2024 (full text); Parthasarathy 2017, Watanabe & Funahashi 2014 (abstract
> only — not in Europe PMC). Extracted texts and the two long-form analyses live in the job tmp
> (`style_analysis_A.md`, `style_analysis_B.md`). Counts below are measured, not recalled.

## Journal limits (nature.com/neuro/content, Articles)

| Item | Limit |
|---|---|
| Abstract | 150 words, unreferenced |
| Main text | 4,500 words (excludes abstract, Methods, references, legends) |
| Display items | 8 |
| References | ~50 |
| Structure | Results and Methods carry topical subheadings; the Discussion does not |

## 1. Abstract

Eight-paper range: **149–180 words, 6–11 sentences, ZERO numerals, ZERO citations.**
Architecture, invariant across all eight:

1. **Context, 1–3 sentences**, setting up one specific tension, present tense.
2. **The gap, one sentence**, with an explicit unknown-verb.
3. **The pivot, exactly one sentence**: "Here we show…" (Libby, Parthasarathy), "Here we find…"
   (Kaufman), "Here, we trained…" (Yang), "We studied…" (Golub), "In the present work, we…"
   (Driscoll), "We wondered whether…" (Musall).
4. **Findings, 1–5 sentences, past tense**, in the order they appear in Results.
5. **Exactly one closing implication, present tense, hedged**: "These results indicate…",
   "This suggests…", "…may thus help…". Never a method, never a caveat.

Not done: no P values, no n, no error bars, no effect sizes, no "for the first time".

## 2. Introduction

- **3–5 paragraphs, 65–200 words each**, one job apiece: broad framing → prior work and its limit
  → the gap → approach plus a full findings roadmap.
- **Citations front-loaded.** ¶1–2 carry 4–15 call-outs; the gap paragraph and the closing
  paragraph carry 0–2.
- **One explicit gap sentence at a paragraph boundary**, with the field's stock verb: "it is
  unknown how…", "it has remained unclear how…", "is not yet understood", "little is known
  about…", "Much less is known about…", "A major open question … is how…".
- **Convert the gap into a falsifiable object** before testing it: two named assumptions
  (Musall), three named hypotheses (Golub), a 2×2 of possibilities (Yang).
- **The last paragraph gives away every result, including the negative ones**, one sentence per
  Results section in Results order, closing on the thesis. It opens with a first-person pivot
  ("We therefore…", "To study…, we…", "Here we test…").

## 3. Results

**Structure**
- Open with **2–8 untitled paragraphs** before the first heading: recording, task in one
  "Briefly," sentence, analysis framework, and any coined term (Golub 2, Libby 3, Kaufman 8).
- **Hold one heading grammar.** Either declarative claim sentences of 6–10 words (Libby) or noun
  phrases of 2–6 words (Golub, Kaufman, Musall, Driscoll). Never questions, never over 10 words.
  Across 22 headings in the modelling papers only 18% are claim sentences; Libby uses claims
  throughout. Repeat a head noun so the heading list reads as one argument.
- **Paragraphs 120–165 words, 2–7 per subsection.**
- **Define a coined term once, in quotes, then use it unchanged** ("Henceforth we refer to these
  factors as population activity patterns").

**Paragraph skeleton** (nine dissected paragraphs, all the same four beats)
1. Motivation or prediction, often a purpose clause: "To quantify…, we…" (~25% of paragraphs).
2. What was done, with the safeguard or null named in the same breath ("To avoid circularity…").
3. The finding, past tense, figure pointer parenthesised at the end of the clause.
4. One interpretive sentence — at the end of most subsections, about a quarter to a third of
   paragraphs; the rest end on evidence.

**Statistics**
- **Claim in the sentence, evidence in the parenthesis**, ordered figure → effect ± error → P →
  n → test name: "(Fig. 4e; single/n = 0.33, n = 522, p<1/1000, one-sided permutation test)".
- **Never make the test the grammatical subject.** No "A t-test revealed that…".
- Inline P values are optional: Golub, Musall, Yang and Driscoll have **zero** in their Results
  and put the tests in legends and a Methods statistics block; Libby and Kaufman report them
  inline. Either is house style, but be consistent.

**Sentences and vocabulary**
- **Mean 19–26 words per sentence** (measured: 20.7–25.6). 15–30% over 30 words, a handful under
  12 to land a point ("Only then is preparatory activity examined.").
- **"we" 8–16 per 1,000 words** — first-person active, not impersonal passive.
- **Em-dashes ≈ 0** (zero in ~11,000 words of Results across three full texts). Semicolons
  0.6–9.1 per 1,000 words. Parentheses do the aside work.
- **Rhetorical questions in Results: 0–2 per paper.** Questions belong in the Introduction.
- **Transitions are plain**: "However", "First,/Second,/Third,", "Thus/Therefore", "Consistent
  with", "Together/Taken together", "In contrast", "Conversely", "We next/then/first", "To
  test/quantify/determine…", "These results".
- **Emphasis adverbs are rationed**: across ~22,000 words, "Importantly" 4×, "Notably" 1×,
  "Interestingly" 1×, **"Critically" 0×, "Surprisingly" 0×**.
- **Hedge asymmetrically**: state the observation flatly in the past tense, hedge only the
  inference ("indicating that", "suggesting that", "consistent with"). Reserve "may/might/could"
  for alternatives and caveats. "Prove", "conclusively", "clearly show" appear nowhere.

**Figures**
- Parenthetical pointer at the **end of the clause it supports** (79–94% of references). Name the
  panel and the visual cue when directing the eye: "(Fig. 6b, blue)", "(Fig. 4B, top right)",
  "(Fig. 1b, black line lies within yellow plane)". Reserve "Figure N shows…" for introducing a
  whole analysis.

**Confounds**
- Give them their own real estate: an enumerated controls paragraph ("First… Second… Third…"),
  a labelled subsection ("Basic controls", "Controls for interpretation", "Potential influences
  on learning strategy"), or a flat admission. Name the competing model, state what it predicts,
  then show the data against it.
- **Close the Results by disposing of the last threat to the interpretation**, not by summarising
  the paper. The Discussion's first sentence restates question and answer in one line.

## 4. Conformance: this manuscript before and after the 2026-09-04 style pass

| Metric | Corpus / limit | Draft v10.4 | Draft v11 |
|---|---|---|---|
| Abstract words | ≤150 | 246 | **149** |
| Numerals in abstract | 0 | 4 | **0** |
| Main text words | ≤4,500 | 6,091 | 5,791 |
| Results mean words/sentence | 19–26 | 21.7 | **21.0** |
| Rhetorical questions in Results | 0–2 | 4 | **0** |
| Prose semicolons per 1,000 words | 0.6–9.1 | 16.1 | **1.5** |
| "we" per 1,000 words | 8–16 | 6.6 | **8.5** |
| Em-dashes | ≈0 | 0 | 0 |
| Untitled opening paragraphs | 2–8 | 0 | **2** |
| Heading grammar | one grammar, ≤10 words | mixed, 8–11 | **claims, 8–10** |
| "Critically" | 0 | 1 | **0** |

The one metric still out of range is main-text length: 5,791 against the 4,500-word limit.
Introduction (535) and Results (3,902) are both within the corpus range on their own; the overage
sits in the Discussion (1,354) and in the Results' qualification and control paragraphs. Cutting
it is a content decision, not a style one.
