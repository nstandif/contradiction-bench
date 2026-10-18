# contradiction-bench

Benchmark of contradiction detectors on ContraDoc (Li et al., NAACL 2024):
hosted decision models (Jev, Perplexity Decider v1.2, Microsoft-Decision-1,
Span-02, GPT-6 Luna Decisions through OpenRouter or Vercel AI Gateway; SubQ Snap
through its own API) against 3-way NLI models run locally.

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

There is no GPU on the dev machine. Keys come from the environment or a
gitignored `.env` at the project root: `OPENROUTER_API_KEY` or
`AI_GATEWAY_API_KEY`, plus `SUBQ_API_KEY` for SubQ Snap. A git worktree has its own root, so copy `.env` into it;
it is deleted with the worktree.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Pairs | `cbench.candidates --k K` | `data/work/pair_test.jsonl`, `data/work/cand_kK.jsonl` |
| NLI | `cbench.nli <pairs> [--model M] [--limit N]` | `results/cache/nli/<model>/<pairs>.jsonl` |
| Decision models | `cbench.decide <openrouter\|vercel\|subq> <pairs\|native> <alias> [--source cand_k20] [--model slug] [--dry-run]` | `results/cache/decide/<slug>/...` |
| Report | `cbench.evaluate [--source cand_k20]` | `results/report.md`, `results/metrics.json` |

- Scorers resume from their cache. Each cached row stores a fingerprint of the
  text it scored, so changing sentence splitting re-scores the affected items
  instead of reusing stale scores.
- `--limit N` scores a seeded random sample, which is safe for timing runs. The
  evaluator reports a system only once it has scored every item of an input and
  lists partial runs separately.
- `--dry-run` prints the first request without a key.

## Gateways and models (catalogues checked 2026-10-18)

All three routes take the TypeSafe body (`{model, state, questions}`,
`noul`/`choice`): OpenRouter at `openrouter.ai/api/alpha/decisions`, Vercel at
`ai-gateway.vercel.sh/typesafe/v1/systemone`, SubQ at `api.subq.ai/v1/systemone`.

| Alias | OpenRouter slug | Vercel slug | SubQ | $/M input |
|---|---|---|---|---|
| `jev` | `typesafe/jev-1.13` | `typesafe-ai/jev-1.13.0` | - | 0.042 |
| `pplx` | `perplexity/pplx-decider-v1.2` | same | - | 0.02 |
| `msft` | `microsoft/microsoft-decision-1` | same | - | 0.042 |
| `span` | `respan/span-02` | not listed | - | 0.01 |
| `luna` | `openai/gpt-6-luna-decisions` | same | - | 0.10 |
| `snap` | not listed | not listed | `subq-snap-preview` | 0.05 |

- **Perplexity Decider is v1.2 by choice.** v1.1 is gone from both gateways and
  from Perplexity's API (the v1.1 name now serves v1.2).
- **SubQ Snap** is in public preview with limited capacity (new accounts may be
  waitlisted) and allows at most 16 questions per request, which is why every
  model gets 16-question chunks in native mode.
- Microsoft-Decision-1 and Jev-on-Vercel have 32K context; native mode stays
  well under that (ContraDoc documents are at most about 2,200 tokens).
- Native mode sends 16 questions per request, so a document's state is billed
  once per chunk. At these prices the full native run costs cents per model.
- Span-02's `reasoning` budget isn't exposed through the gateways, so it runs in
  its default "think when unsure" mode.
- Refer to `GET https://openrouter.ai/api/v1/models?output_modalities=decisions`
  and `GET https://ai-gateway.vercel.sh/v1/models` (type `evaluation`) for the
  current catalogues; neither needs a key.

## Design decisions

- **Same pairs for every pairwise system.** Doc-level pairwise scores use the
  top-K TF-IDF pairs per document. Candidate recall caps every pairwise system
  equally, so model-vs-model comparisons are fair, but pairwise-vs-native
  comparisons are not: native mode sees every sentence.
- **Two evaluations.** `pair_test` is gold contradictory pairs (321) against the
  most similar pair in each clean document (442 hard negatives), testing models in
  isolation. Document level uses ContraDoc's 449/442 split.
- **Native mode** sends the whole document with one noul per sentence plus one
  document-level noul. Pairwise methods structurally miss the 73
  intra-sentence contradictions; native mode does not.
- **Split.** Stratified 30% dev / 70% test (seed 0). Thresholds are tuned for F1
  on dev only. AUROC is the headline because max-over-pairs grows with length,
  so it is also reported by length tertile. Always-yes is the floor (F1 0.669).
- **Option-order bias.** Jev's docs report a first-option bias, so pair questions
  are asked twice with reversed options and averaged; `order_gap` is the mean
  absolute difference.
- **NLI** scores both directions and takes the max contradiction probability.
- `contra_type` is multi-label, so per-type recall counts a document once per type.
- Jev's documented weak spots (numbers, dates, negation, long state) overlap
  ContraDoc's Numeric, Negation and Factual types; check per-type recall.

## Status

- Built and tested: data loading, pair building, NLI scorer, gateway client
  (both gateways, both modes), evaluator (end-to-end test on fabricated caches).
- Not yet run: no real model scores exist. NLI throughput on this CPU is
  unmeasured; time `cbench.nli pair_test --limit 320` before planning a full run.
- Open decisions: K (gold-pair recall 59.5% at K=20, 68.5% at K=50); NLI model
  size (DeBERTa-v3-large default vs base for speed); whether to add
  embedding-based candidates to raise recall.
