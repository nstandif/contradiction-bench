# Find → verify and find → explain

Both pipelines run a second, targeted pass over sentences that the native runs flagged.

- **Verify** asks whether there is a contradiction. **It does not beat D1 Ultra native.** On the
  624 test documents the best verifier scores AUROC 0.841, against D1 Ultra native's 0.854.
- **Explain** asks which sentence conflicts with which. **This is what works.** It names the exact
  gold pair in 61.8% of contradictory test documents. The best NLI model's top candidate pair
  manages 37.1%, and native runs can't name a pair at all.

Both are in `cbench.verify`. They appear in `report.md`, and on the leaderboard as
`native:verify` rows (cost vs AUROC) and in the "Naming the pair" section (`explain`). Runs on
2026-10-19 covered all 891 ContraDoc documents and cost $0.18 with the Decider as verifier and
$0.26 with D1 Ultra, with no failures. Rerun with
`uv run python -m cbench.verify --verifier <model>`.

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

## Round two: explain

Of the cases where verify named the wrong pair, 13.5% of documents had the right answer on the
shortlist but trusted a different shortlisted sentence. Verify's own confidence is a poor guide to
which sentence to trust. In the Venus in Furs example, it was surer of two consecutive quote lines
(s12 → s11, 0.78) than of the planted conflict (s3 → s7, 0.48). The finders' native sentence
scores pick the right sentence more often.

Explain therefore splits the work:
- **The finders choose the sentence:** the mean of D1 Ultra's and the Decider's native scores.
- **The verifiers name its partner:** answers are averaged when there is more than one verifier.

D1 Ultra was run as a second verifier on identical questions.

| Contradictory test documents with a labelled partner (259) | Pair@1 | 95% CI | Pair@2 |
|---|---|---|---|
| Decider verify, verify's confidence picks the sentence (round one) | 0.544 | | 0.579 |
| Decider explain | 0.591 | [0.537, 0.653] | 0.680 |
| D1 Ultra explain | 0.614 | [0.556, 0.676] | 0.695 |
| **D1 Ultra + Decider explain** | **0.618** | [0.560, 0.680] | **0.699** |
| Best pairwise NLI, top cand_k20 pair (deberta-base-long-nli) | 0.371 | [0.317, 0.429] | 0.413 |
| Always choosing the right shortlisted sentence (ceiling) | 0.680 | | |

Gaps against round one are paired bootstraps of Pair@1 over the same documents:

- **Decider explain:** +0.046 [+0.008, +0.085].
- **D1 Ultra explain:** +0.069 [+0.023, +0.116].
- **Averaging both verifiers:** +0.073 [+0.031, +0.112]. That is only +0.004 [−0.019, +0.027]
  over D1 Ultra alone, because the two verifiers name the same partner 81% of the time. The
  cheapest strong setup is D1 Ultra as the only verifier, at $0.26 per pass.

How the variants were chosen: on dev (115 documents), averaging both verifiers with the finders
choosing the sentence scored best (0.678). Using 1 − P(none) to choose the sentence scored 0.652.

When asked about the planted sentence, D1 Ultra's first answer is the gold partner 74.5% of the
time and its top three include it 90.8% of the time. The Decider's figures are 70.0% and 86.1%.
The remaining errors are mostly unrelated sentences (adjacent sentences are rare) and the 7.7%
of documents where the planted sentence never reaches the shortlist.

As a detector, D1 Ultra verify scores 0.824 AUROC, significantly below D1 Ultra native
(gap 0.030 [0.006, 0.052]). Averaging the two verifiers' 1 − P(none) reaches 0.850 on test. So
the practical split is: D1 native decides *whether* there is a contradiction, and explain says
*where* it is and *what it conflicts with*.

## What didn't work, in order tried

1. **Each pair scored on its own** (Decider + tasksource NLI blend, 15-document spike): every
   clean document was flagged, so it worked out as always-yes.
2. **Scoring by the most likely partner's probability:** a worse detector than 1 − P(none) (0.815).
3. **Random control sentences subtracted as a baseline:** 0.792, worse than without them.
4. **Verify as a second opinion on D1's misses:** it caught 25 of 79 misses but falsely flagged
   40 of 248 cleared clean documents.
5. **A second verifier for detection:** no significant gain. The verifiers are too correlated.
