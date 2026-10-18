# contradiction-bench

Benchmark of contradiction detectors on ContraDoc (Li et al., NAACL 2024): every
hosted decision model on OpenRouter plus SubQ Snap, against 19 public 3-way NLI
models run locally. The question it answers: how well do shipped, off-the-shelf
models detect self-contradictions in real documents?

## Start here (handoff)

State as of 2026-10-18: **everything is built and tested; no real scores exist
yet.** In order, the next agent should:

1. **Set up the checkout** (data and venv are gitignored):
   `uv sync && uv run python -m cbench.fetch && uv run python -m cbench.candidates --k 20 && uv run pytest`.
2. **NLI timing.** `uv run python -m cbench.nli time` measures CPU throughput per
   model and writes `results/nli_timing.json` with projected hours for the full
   sweep. A first run was started on 2026-10-18; if that file exists, reuse it.
   Timing pairs are cached, so they count toward the full run.
3. **Decide where each NLI model runs.** Models whose projected time is
   acceptable run locally with `cbench.nli sweep`. For the rest, see
   "Hosted NLI" below; the user asked for a hosted option for any model too slow
   on CPU, and hasn't picked one yet.
4. **Decision models need keys** in `.env` at the project root:
   `OPENROUTER_API_KEY` and `SUBQ_API_KEY`. The user has not supplied them yet.
   Then: `cbench.sweep probe`, show the user which models pass, then
   `cbench.sweep run` (about $6 for the default tests; see "Cost").
5. **Report.** `cbench.evaluate` writes `results/report.md` and `metrics.json`.
   Commit the report, `results/catalog.json`, `results/probe.json` and
   `results/nli_timing.json`; caches stay out of git.

Ask the user before anything that costs money or takes hours of compute. They
have so far approved: the NLI sweep (all 19 models), the decision-model sweep
(all OpenRouter models plus SubQ Snap), and Perplexity Decider v1.2 in place
of v1.1.

## Why the full dataset

ContraDoc is balanced (449 contradictory / 442 clean). Always use all of it: on a
sample with few clean documents, false positives barely register and F1 looks
far better than it is.

## Layout

| Module | Purpose |
|---|---|
| `cbench.fetch` | ContraDoc pinned to commit `d1c42d1`, sha256-checked, into `data/raw/` |
| `cbench.data` | sentence splitting (pysbd), gold evidence location, dev/test split, fingerprints |
| `cbench.candidates` | `pair_test` and top-K TF-IDF candidate pairs into `data/work/` |
| `cbench.nli` | 3-way NLI scoring: `run`, `time`, `sweep` |
| `cbench.decide` | decision-model client for OpenRouter, Vercel AI Gateway, SubQ |
| `cbench.sweep` | decision-model sweep: `catalog`, `estimate`, `probe`, `run` |
| `cbench.evaluate` | report with bootstrap CIs, per-type/scope/length breakdowns |

All scores cache to `results/cache/` as JSONL and every scorer resumes. Each row
stores a fingerprint of the exact text scored, so changing sentence splitting
re-scores affected items instead of reusing stale scores. `--limit N` scores a
seeded random sample. The evaluator only reports a system that has scored every
item of an input and lists partial runs separately.

## Evaluation design

- **Two tests.** `pair_test`: gold contradictory pairs (321) against the most
  similar pair in each clean document (442 hard negatives), testing models on
  identical inputs. Document level: ContraDoc's 449/442 documents.
- **Document-level modes.** Pairwise systems score the top-20 TF-IDF pairs per
  document and take the max. Candidate recall caps them (gold pair present in
  59.5% of documents at K=20, 68.5% at K=50); this cap is equal across pairwise
  models, so model-vs-model comparisons are fair, but pairwise-vs-native are not.
  Decision models also run in **native** mode: the whole document with one noul
  per sentence plus one document-level noul, which can catch the 73
  intra-sentence contradictions that pairwise methods structurally miss.
- **Split.** Stratified 30% dev / 70% test (seed 0): 267 dev, 624 test
  documents. Thresholds are tuned for F1 on dev only. Always-yes is the floor
  (F1 0.669).
- **Uncertainty.** AUROC is the headline, with 95% percentile bootstrap
  intervals (1,000 resamples of test documents). To compare systems, use the
  "Gap to best" column: a paired bootstrap of the AUROC difference on shared
  resamples, which is much tighter than comparing per-system intervals because
  every system scores the same documents. A gap whose interval excludes 0 is a
  real difference; with roughly 40 systems, expect a few such results by chance.
  AUROC is also reported by length tertile because max-over-pairs grows with
  document length.
- **Option-order bias.** Pair questions are asked twice with reversed options and
  averaged; `order_gap` reports the mean absolute difference (Jev's docs report a
  first-option bias).
- **NLI** scores both directions and takes the max contradiction probability.
- `contra_type` is multi-label, so per-type recall counts a document once per type.

## NLI models

19 models in `cbench.nli.NLI_MODELS`, all with a contradiction class (checked
via each `config.json` on 2026-10-18): MoritzLaurer DeBERTa-v3 large/base and
mDeBERTa, cross-encoder DeBERTa-v3 large/base/small/xsmall and RoBERTa/DistilRoBERTa,
roberta-large-mnli, bart-large-mnli, Microsoft DeBERTa large/xlarge/v2-xlarge/v2-xxlarge
MNLI, ynie RoBERTa-large SNLI+MNLI+FEVER+ANLI, VitaminC ALBERT-xlarge, tasksource
DeBERTa-v3-large and deberta-base-long. Excluded:
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c` (two-way, no
contradiction class). VitaminC's `REFUTES`/`SUPPORTS` map to contradiction/entailment.
Weights total about 20 GB; disk is not a constraint. Several Microsoft and ynie
models ship only `pytorch_model.bin`, loaded with torch's weights-only loader.

### Hosted NLI (for models too slow on CPU)

Researched 2026-10-18; no option chosen yet.
- **Replicate:** only `bart-large-mnli`, wrapped as zero-shot classification.
  Wrong interface: it returns label scores, not premise/hypothesis 3-way probabilities.
- **HF serverless (`hf-inference`):** a few of these models, also as
  zero-shot classification. Same problem.
- **HF Inference Endpoints (recommended if needed):** dedicated GPU running any
  Hub model with raw sequence-classification output; T4 $0.50/h, L4 $0.80/h,
  billed per minute, scale-to-zero. Would need a small HTTP scorer alongside
  `NLIScorer`. Modal or similar GPU-per-second services are the alternative.

## Decision models

`cbench.sweep catalog` fetches the live OpenRouter catalogue
(`/api/v1/models?output_modalities=decisions`, no key) into
`results/catalog.json`. On 2026-10-18 it held 27 models; excluded are the three
`~…-latest` aliases, `:free` duplicates, and the Span-01 family (a conversation
behaviour scorer with a different state contract). That leaves 20 OpenRouter
models plus SubQ Snap.

- **Routes.** OpenRouter (`/api/alpha/decisions`), Vercel AI Gateway
  (`/typesafe/v1/systemone`) and SubQ (`api.subq.ai/v1/systemone`) all take the
  TypeSafe body `{model, state, questions}` with `noul`/`choice`, so every model
  gets identical requests. The sweep uses OpenRouter plus SubQ; `cbench.decide`
  also supports Vercel, whose catalogue lacks Span-02 and SubQ.
- **Probe before running.** Some models reportedly have unstated limits (a
  2,000-byte state cap on Clef/Clef Flash, a choice-only Tev1). `sweep probe`
  sends one real request per model and test, using that test's longest input so
  a size cap fails the probe rather than the full run, into `results/probe.json`;
  `sweep run` only runs what passed.
- **Perplexity Decider is v1.2** by the user's choice; v1.1 is no longer served
  anywhere (Perplexity's API maps the v1.1 name to v1.2).
- **SubQ Snap** is direct-API only, in public preview with limited capacity, and
  caps requests at 16 questions. Native mode uses 16-question chunks for every
  model so requests stay identical.
- Span-02's `reasoning` budget isn't exposed through gateways; it runs in its
  default "think when unsure" mode.

### Cost

`cbench.sweep estimate` (JSON chars / 4, so approximate): about $6 for all 21
models on `pair_test` + `native`, about $12 including `cand_k20`. Per model,
$0.12 (Span-02, Decider Flash) to $1.86 (Clef Omni).

## Environment

- No GPU on the dev machine; CPU-only torch via the PyTorch CPU index. The CPU
  is an AMD Ryzen 7 9700X: 8 physical cores, 16 threads. Don't run other heavy
  work (including `pytest`) during NLI timing; it skews the numbers.
- First timing results (2026-10-18, contended by a concurrent `pytest`, so
  pessimistic): DeBERTa-v3-large 0.8 pairs/s (~6.4 h for pair_test + cand_k20),
  DeBERTa-v3-base 2.5 pairs/s (~2.1 h). Even uncontended, large models are
  expected around 3 pairs/s, so the 900M–1.5B Microsoft DeBERTa models likely
  need the hosted option. Re-time contended models with
  `cbench.nli time --models <id>` after deleting their entry from
  `results/nli_timing.json`; already-cached pairs are skipped, so it samples new ones.
- Python 3.12 via uv (system Python is 3.14, too new for some wheels).
- A stale `VIRTUAL_ENV` from another project may be set in the shell; `unset
  VIRTUAL_ENV` before `uv run` to silence the warning.
- Edits on `main` are blocked by a hook outside worktrees; do work on a branch
  in a worktree and fast-forward `main`.
