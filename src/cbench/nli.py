"""Score sentence pairs with 3-way NLI models, in both directions.

  run <pairs> [--model M]   score one model on one pair file
  time [--sample N]         measure CPU throughput per model -> results/nli_timing.json
  sweep [--tests ...]       run every model in NLI_MODELS on each pair file
"""

import argparse
import json
import random
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cbench.candidates import Pair, read_pairs
from cbench.paths import CACHE, RESULTS

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_TOKENS = 256
SAMPLE_SEED = 0
TIMING = RESULTS / "nli_timing.json"

# Every public 3-way NLI checkpoint with a contradiction class that we know of
# among widely used ones (checked 2026-10-18). Two-way entailment models are
# excluded because they cannot separate contradiction from neutral.
NLI_MODELS = [
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "cross-encoder/nli-deberta-v3-large",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-roberta-base",
    "cross-encoder/nli-distilroberta-base",
    "FacebookAI/roberta-large-mnli",
    "facebook/bart-large-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-xlarge-mnli",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "tals/albert-xlarge-vitaminc-mnli",
    "sileod/deberta-v3-large-tasksource-nli",
    "tasksource/deberta-base-long-nli",
]

# VitaminC uses fact-verification labels for the same classes.
LABEL_ALIASES = {"contradiction": {"contradiction", "refutes"}, "entailment": {"entailment", "supports"}}


def label_index(id2label: dict, name: str) -> int:
    for index, label in id2label.items():
        if label.lower() in LABEL_ALIASES[name]:
            return int(index)
    raise ValueError(f"no {name} label in {id2label}")


class NLIScorer:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.contradiction = label_index(self.model.config.id2label, "contradiction")
        self.entailment = label_index(self.model.config.id2label, "entailment")
        self.batch_size = batch_size

    @torch.inference_mode()
    def probabilities(self, premises: list[str], hypotheses: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            premises, hypotheses, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt"
        )
        return self.model(**batch).logits.softmax(-1)

    def score(self, pairs: list[Pair]) -> list[dict]:
        rows = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            a, b = [p.a for p in chunk], [p.b for p in chunk]
            fwd = self.probabilities(a, b)
            rev = self.probabilities(b, a)
            for p, f, r in zip(chunk, fwd, rev):
                rows.append(
                    {
                        "key": p.key,
                        "fingerprint": p.fingerprint,
                        "p_contra_ab": f[self.contradiction].item(),
                        "p_contra_ba": r[self.contradiction].item(),
                        "p_entail_ab": f[self.entailment].item(),
                        "p_entail_ba": r[self.entailment].item(),
                    }
                )
        return rows


def cache_path(model_name: str, pairs_name: str):
    return CACHE / "nli" / model_name.replace("/", "__") / f"{pairs_name}.jsonl"


def pending_pairs(model_name: str, pairs_name: str, limit: int | None) -> list[Pair]:
    out = cache_path(model_name, pairs_name)
    done = {(r := json.loads(line))["key"]: r.get("fingerprint") for line in out.open()} if out.exists() else {}
    pending = [p for p in read_pairs(pairs_name) if done.get(p.key) != p.fingerprint]
    if limit is not None:
        pending = random.Random(SAMPLE_SEED).sample(pending, min(limit, len(pending)))
    return pending


def score_model(model_name: str, pairs_name: str, *, limit: int | None = None, batch_size: int = 32) -> tuple[int, float]:
    """Score pending pairs; returns (pairs scored, seconds spent scoring, excluding model load)."""
    pending = pending_pairs(model_name, pairs_name, limit)
    print(f"{model_name} {pairs_name}: {len(pending)} pending", flush=True)
    if not pending:
        return 0, 0.0
    pending.sort(key=lambda p: len(p.a) + len(p.b))
    scorer = NLIScorer(model_name, batch_size)
    out = cache_path(model_name, pairs_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    flush_every = batch_size * 10
    with out.open("a") as f:
        for start in range(0, len(pending), flush_every):
            for row in scorer.score(pending[start : start + flush_every]):
                f.write(json.dumps(row) + "\n")
            f.flush()
            scored = min(start + flush_every, len(pending))
            rate = scored / (time.perf_counter() - started)
            print(f"  {scored}/{len(pending)} pairs, {rate:.1f} pairs/s", flush=True)
    return len(pending), time.perf_counter() - started


def time_models(models: list[str], sample: int, batch_size: int) -> dict:
    """Throughput on a seeded cand_k20 sample; projects hours for the full sweep."""
    totals = {name: len(read_pairs(name)) for name in ("pair_test", "cand_k20")}
    timing = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    for model_name in models:
        scored, seconds = score_model(model_name, "cand_k20", limit=sample, batch_size=batch_size)
        if not scored:
            continue
        rate = scored / seconds
        timing[model_name] = {
            "pairs_per_second": round(rate, 2),
            "sample": scored,
            "threads": torch.get_num_threads(),
            "projected_hours": {name: round(n / rate / 3600, 2) for name, n in totals.items()},
        }
        print(f"{model_name}: {rate:.1f} pairs/s, full sweep ~{sum(timing[model_name]['projected_hours'].values()):.1f} h")
        RESULTS.mkdir(parents=True, exist_ok=True)
        TIMING.write_text(json.dumps(timing, indent=2))
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("pairs", help="pair file in data/work, e.g. pair_test or cand_k20")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--limit", type=int, help="score a seeded random sample of N pending pairs")
    timing = sub.add_parser("time")
    timing.add_argument("--sample", type=int, default=128)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--tests", nargs="+", default=["pair_test", "cand_k20"])
    for p in (timing, sweep):
        p.add_argument("--models", nargs="+", default=NLI_MODELS)
    for p in (run, timing, sweep):
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.command == "run":
        score_model(args.model, args.pairs, limit=args.limit, batch_size=args.batch_size)
    elif args.command == "time":
        time_models(args.models, args.sample, args.batch_size)
    else:
        for test in args.tests:
            for model_name in args.models:
                score_model(model_name, test, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
