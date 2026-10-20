# contradiction-bench

Benchmark of contradiction detectors on ContraDoc (Li et al., NAACL 2024): every
hosted decision model on OpenRouter plus SubQ Snap, against 19 public 3-way NLI
models. The question it answers: how well do shipped, off-the-shelf
models detect self-contradictions in real documents?

## Status

As of 2026-10-19: **first full sweep done**; `results/report.md` and
`metrics.json` are committed and the Streamlit app reads them.

- **Decision models:** 21 of 21 complete on `pair_test` and `native`.
  ScaleDown Classify-1 rejects object states unless they hold only `text`/`document`,
  so `decide.STRING_STATE_MODELS` sends it the same JSON as a string. It bills the
  state once per question, so native costs about 11× the `sweep estimate` figure
  ($2.52). OpenAI refuses GPT-6 Luna's question
  `s51` on document `3499318685_4` on every retry. `decide.score_doc` drops refused
  questions, re-asks the rest and records them in `refused`; the evaluator counts
  them as "no" and marks the system with an asterisk.
- **NLI:** 19 of 19 complete on `pair_test` and `cand_k20`. Twelve ran on CPU. The
  seven too slow for it (`results/nli_timing.json`: the three MoritzLaurer models,
  VitaminC ALBERT-xlarge, Microsoft deberta-xlarge, v2-xlarge and v2-xxlarge; 3–33 h
  each) ran on a Modal T4 via `cbench.nli_modal` in about 1.5 GPU-hours.
- **Not run:** `cand_k20` for decision models (about $6 more).
- **Find → verify / explain** (`cbench.verify`, `results/verify.md`): native sentence scores
  shortlist suspicious sentences, then a verifier reads the whole document and answers "which
  sentence does this contradict?". Verify (detection) doesn't beat D1 Ultra native: 0.841 vs
  0.854. Explain (finders pick the sentence, verifiers name the partner) gets the exact gold pair
  in 61.8% of documents. Verifiers run so far: Decider v1.2 and D1 Ultra, cached as
  `verify_all.jsonl` under each model's cache directory.

To rebuild the report from published scores: `uv run python -m cbench.fetch --cache`,
then `cbench.candidates --k 20` and `cbench.evaluate`. To re-run anything: set up the checkout (`uv sync --all-groups &&
uv run python -m cbench.fetch && uv run python -m cbench.candidates --k 20`),
put `OPENROUTER_API_KEY` and `SUBQ_API_KEY` in `.env`, then rerun the scorers;
they resume from `results/cache/`. Rebuild with `cbench.evaluate` and commit
`report.md`, `metrics.json`, `probe.json` and `nli_timing.json`; caches stay out
of git.

Ask before anything that costs money or takes hours of compute.

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
| `cbench.nli_modal` | the same NLI scoring on a Modal GPU |
| `cbench.decide` | decision-model client for OpenRouter, Vercel AI Gateway, SubQ |
| `cbench.sweep` | decision-model sweep: `catalog`, `estimate`, `probe`, `run` |
| `cbench.verify` | find → verify (detection) and find → explain (naming the pair) over native runs |
| `cbench.evaluate` | report with bootstrap CIs, per-type/scope/length breakdowns, pair naming |
| `cbench.dashboard` + `app/` | Streamlit explorer over `results/metrics.json`: cost-vs-AUROC leaderboard, compare page |

Run the explorer with `uv sync --group app && uv run streamlit run app/streamlit_app.py`;
`CBENCH_METRICS=<path>` points it at another metrics file. Chart builders live in `app/charts.py`; `app/export_charts.py` renders them to
`docs/img/` for the README, so rerun it after rebuilding the report. Costs in the app are
estimates (catalogue price × `sweep estimate` tokens) and need `data/work`. Compare
only shows gaps to the best system, because `metrics.json` stores no other paired
gaps. `tests/test_app.py` skips unless the `app` group is installed.

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
  **Why TF-IDF:** it is model-free, so every pairwise system gets the same
  candidates, and no tested retriever beats it at K=20. On 2026-10-19, gold-pair
  recall at K=20 (321 documents) was:
  - TF-IDF 0.595, BM25 0.570, MiniLM-L6 0.558, bge-small 0.526.
  - BM25 + vector hybrids (reciprocal-rank fusion) reached at most 0.595. At
    K=50 they reach 0.726, against 0.685 for TF-IDF.

  Similarity is the wrong signal: 20.9% of gold pairs rank beyond 50 for every
  retriever, and 12.1% share no content words. Ranking pairs by D1 Ultra's
  native sentence scores reaches 0.788 at K=20. That ranking depends on one
  model, so it isn't used as the shared baseline.
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
- **Naming the pair** (`explain` in `metrics.json`): Pair@1/Pair@2 over contradictory test
  documents that have a labelled partner (`ref_idx`, 259 documents). Pairwise systems compete
  with their top candidate pairs, so candidate recall caps them.

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

### GPU scoring on Modal

`cbench.nli_modal` runs `NLIScorer` on a Modal GPU (T4 by default, `--gpu` to
change) and writes the same cache files, so runs resume and mix freely with CPU
runs. GPU scores match CPU scores to within 1.2e-6. Needs `uv sync --group modal`
and `modal token new`; `HF_TOKEN` in `.env` is passed to the containers, and
weights persist in the `cbench-hf-cache` volume.
- All models run as threads inside one Modal app. Separate processes per model
  hit Modal's app-creation rate limit.
- The container rebuilds the module's secrets list, so it must be defined the same
  way locally and remotely, or the container fails with "Function has N
  dependencies but container got M object ids".
- The Modal Starter plan includes $30/month of compute; the full seven-model run
  used about 1.5 T4-hours (under $1).

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
- **Perplexity Decider is v1.2**; v1.1 is no longer served
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
  expected around 3 pairs/s, so the 900M–1.5B Microsoft DeBERTa models need
  `cbench.nli_modal`. Re-time contended models with
  `cbench.nli time --models <id>` after deleting their entry from
  `results/nli_timing.json`; already-cached pairs are skipped, so it samples new ones.
- Python 3.12 via uv (system Python is 3.14, too new for some wheels).
