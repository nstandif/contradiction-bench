# Find → verify: does a second, targeted pass beat D1 Ultra native?

**No, not at detecting contradictions.** On the 624 test documents the pipeline scores AUROC 0.841
against D1 Ultra native's 0.854. The difference is not significant either way. What verify does
add is naming the conflicting pair, which native mode cannot do.

Run on 2026-10-19 over all 891 ContraDoc documents for $0.18 in Decider calls, with no failures.
The code is on branch `feat/verify` (`cbench.verify`, plus a `native:verify` configuration in the
evaluator and leaderboard), and is not merged.

## Pipeline

1. **Find.** Use the cached native runs of D1 Ultra and Decider v1.2. Each one shortlists the
   sentences it scores at least 0.3, keeping at least 1 and at most 5, and the two shortlists
   are merged. The result averages 3.5 sentences per document and contains the planted sentence
   in 87.9% of contradictory test documents.
2. **Verify.** Perplexity Decider v1.2 reads the whole document. For each shortlisted sentence it
   gets one choice question: "Which sentence does `sentences[i]` contradict, if any?" The options
   are every other sentence, `itself` and `none`. Each question is asked with the options in both
   orders and the two answers are averaged. Two random sentences outside the shortlist are asked
   as well, as controls.
3. **Score.** The document score is the highest 1 − P(none) across the shortlist.

Earlier variants failed, in a 15-document spike outside the repo. In those, each pair was scored
on its own, without the rest of the document, by the Decider blended with tasksource NLI. Every
clean document was flagged: once the surrounding context is gone, plot developments ("reported
dead" … "not dead") and statements attributed to different people look like contradictions.

## Detection (test split, AUROC)

Brackets on AUROC are 95% bootstrap intervals. Δ is a paired bootstrap against D1 Ultra native.

| System | AUROC | 95% CI | Δ vs D1 Ultra [95% CI] |
|---|---|---|---|
| D1 Ultra native | 0.854 | [0.822, 0.882] | — |
| Decider v1.2 native | 0.846 | [0.813, 0.877] | −0.008 [−0.029, +0.012] |
| **Verify, 1 − P(none)** (chosen on dev) | 0.841 | [0.809, 0.871] | −0.013 [−0.036, +0.009] |
| Verify, most likely partner's probability | 0.815 | [0.780, 0.846] | −0.039 [−0.067, −0.012] |
| Verify, partner − controls | 0.792 | [0.751, 0.826] | −0.062 [−0.093, −0.030] |
| Rank average: D1 + Decider native | 0.860 | [0.828, 0.889] | +0.006 [−0.005, +0.017] |
| Rank average: D1 + verify | 0.851 | [0.818, 0.879] | −0.002 [−0.016, +0.011] |
| Rank average: D1 + Decider + verify | 0.857 | [0.824, 0.886] | +0.003 [−0.012, +0.017] |

- **The scoring choice came from dev** (0.835 for 1 − P(none) against 0.816 for the most likely
  partner). Two hand-picked documents had suggested the opposite.
- **The random controls hurt.** Subtracting their score adds noise rather than removing a
  per-document baseline.
- **As a second opinion on D1's misses, verify loses.** Both use thresholds tuned on dev. Of the
  79 contradictory documents D1 misses, verify catches 25. Of the 248 clean documents D1 clears,
  it falsely flags 40.

## By contradiction type and scope (that group against all clean test documents)

| Group | n | D1 Ultra native | Decider v1.2 native | Verify |
|---|---|---|---|---|
| Causal | 29 | 0.810 | 0.784 | 0.764 |
| Content | 205 | 0.849 | 0.842 | 0.844 |
| Emotion/Mood/Feeling | 61 | 0.823 | 0.824 | 0.819 |
| Factual | 42 | 0.869 | 0.859 | 0.812 |
| Negation | 60 | 0.923 | 0.912 | 0.909 |
| Numeric | 49 | 0.934 | 0.881 | 0.878 |
| Perspective/View/Opinion | 66 | 0.795 | 0.801 | 0.809 |
| Relation | 17 | 0.782 | 0.837 | 0.837 |
| Scope: global | 106 | 0.920 | 0.928 | 0.923 |
| Scope: local | 154 | 0.830 | 0.816 | 0.824 |
| Scope: intra-sentence | 55 | 0.794 | 0.775 | 0.731 |

Verify is weakest on contradictions within a single sentence: it rarely answers `itself`
(8.9% of top answers). It is also weak on Factual and Numeric contradictions, which D1 already
handles well.

## Localisation: where verify does help

| Measure (314 contradictory test documents) | Value |
|---|---|
| D1 Ultra: top sentence is the planted one (Hit@1) | 0.697 |
| Verify: top pair contains the planted sentence | 0.748 |
| Verify: top pair is exactly a gold pair (259 documents with a labelled partner) | 0.544 |
| Verify: forward and reversed option orders agree on the chosen answer | 0.885 |

Native mode can say which sentence is suspicious, but not what it conflicts with. Verify names
the exact gold pair in just over half of the documents. That makes it useful for explaining a
detection that D1 or the Decider has already made, not for making the detection.
