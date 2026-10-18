# contradiction-bench

Benchmark of contradiction detectors on ContraDoc (Li et al., NAACL 2024): hosted
decision models (TypeSafe Jev, Perplexity Decider, SubQ Snap, OpenAI Decisions)
against 3-way NLI models.

## Why the full dataset

ContraDoc is balanced (449 contradictory / 442 clean). Always use all of it: on a
sample with few clean documents, false positives barely register and F1 looks
far better than it is.

## Setup

```sh
uv sync                                # Python 3.12, CPU-only torch
uv run python -m cbench.fetch          # ContraDoc pinned to commit d1c42d1, sha256-checked
uv run python -m cbench.candidates --k 20
uv run pytest
```

There is no GPU on the dev machine. API keys are read from the environment or a
gitignored `.env` at the project root (a worktree has its own root, so copy
`.env` into it): `TYPESAFE_API_KEY`, `PERPLEXITY_API_KEY`, `SUBQ_API_KEY`,
`OPENAI_API_KEY`.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Pairs | `cbench.candidates --k K` | `data/work/pair_test.jsonl`, `data/work/cand_kK.jsonl` |
| NLI | `cbench.nli <pairs> [--model M] [--limit N]` | `results/cache/nli/<model>/<pairs>.jsonl` |
| Decision models | `cbench.decide <jev\|pplx\|subq\|openai> <pairs\|native> [--source cand_k20] [--dry-run]` | `results/cache/decide/<provider>/<model>/...` |
| Report | `cbench.evaluate [--source cand_k20]` | `results/report.md`, `results/metrics.json` |

Every scorer resumes from its cache, so interrupted runs continue where they
stopped. `--dry-run` prints the first request without needing a key.

## Design decisions

- **Same pairs for every pairwise system.** Doc-level pairwise scores use the
  top-K TF-IDF pairs per document. Candidate recall is reported because it caps
  every pairwise system.
- **Two evaluations.** `pair_test` is gold contradictory pairs (321) against the
  most similar pair in each clean document (442 hard negatives), testing models in
  isolation. Document level uses ContraDoc's 449/442 split.
- **Native mode for decision models** sends the whole document with one noul per
  sentence plus one document-level noul. Pairwise methods structurally miss the
  73 intra-sentence contradictions; native mode does not.
- **Split.** Stratified 30% dev / 70% test (seed 0). Thresholds are tuned for F1
  on dev only. AUROC is the headline because max-over-pairs grows with length,
  so AUROC is also reported by length tertile. The always-yes row is the floor.
- **Option-order bias.** Jev's docs report a first-option bias, so pair questions
  are asked twice with reversed options and averaged; `order_gap` in the report
  is the mean absolute difference.
- **NLI** scores both directions and takes the max contradiction probability.
- `contra_type` is multi-label, so per-type recall counts a document once per type.

## Provider facts (checked 2026-10-18)

| Provider | Endpoint | Model | Price / M input | Questions per request |
|---|---|---|---|---|
| TypeSafe | `api.typesafe.ai/v1/systemone` | `jev-1.13.0` | $0.042 | not documented (64 used) |
| Perplexity | `api.perplexity.ai/v1/decisions` | `pplx-decider-v1.1-27b` | $0.02 | 128 |
| SubQ | `api.subq.ai/v1/systemone` | `subq-snap-preview` (public preview) | $0.05 | 16 |
| OpenAI | `api.openai.com/v1/decisions` | `gpt-6-luna` (public beta) | $0.10 | 200 |

- **Perplexity serves v1.2 for the v1.1 name** and echoes back the name sent, so
  results labelled v1.1 are really v1.2. Exact v1.1 needs the open 27B weights
  (`perplexity-ai/pplx-decider-v1.1-27b`) on a CUDA GPU with about 49 GB.
- OpenAI uses a different schema (`input` text, `predicate`, `choices` arrays,
  per-question refusals); `decide.request_body`/`parse_answers` translate it.
  Refused answers fail that item and are logged, not scored as zero.
- Jev's documented weak spots (numbers, dates, negation, long state) overlap
  ContraDoc's Numeric, Negation and Factual types; check per-type recall.

## Status

- Built: data loading, pair building, NLI scorer, decision-model client (all four
  providers, both modes), evaluator, tests.
- Not yet run: no model scores exist. NLI throughput on this CPU is unmeasured;
  time `cbench.nli pair_test --limit 320` before planning a full run.
- Open decisions: K (recall 59.5% at 20, 68.5% at 50); NLI model size
  (DeBERTa-v3-large default vs base for speed); whether to add embedding-based
  candidates to raise recall.
