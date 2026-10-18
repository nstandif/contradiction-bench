"""Score sentence pairs with a 3-way NLI model, in both directions."""

import argparse
import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cbench.candidates import Pair, read_pairs
from cbench.paths import CACHE

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_TOKENS = 256


class NLIScorer:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.contradiction = labels["contradiction"]
        self.entailment = labels["entailment"]
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
                        "p_contra_ab": f[self.contradiction].item(),
                        "p_contra_ba": r[self.contradiction].item(),
                        "p_entail_ab": f[self.entailment].item(),
                        "p_entail_ba": r[self.entailment].item(),
                    }
                )
        return rows


def cache_path(model_name: str, pairs_name: str):
    return CACHE / "nli" / model_name.replace("/", "__") / f"{pairs_name}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", help="pair file in data/work, e.g. pair_test or cand_k20")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="score only the first N pending pairs (timing runs)")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out = cache_path(args.model, args.pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["key"] for line in out.open()} if out.exists() else set()
    pending = [p for p in read_pairs(args.pairs) if p.key not in done][: args.limit]
    print(f"{len(done)} cached, {len(pending)} to score with {args.model}")

    scorer = NLIScorer(args.model, args.batch_size)
    started = time.perf_counter()
    flush_every = args.batch_size * 10
    with out.open("a") as f:
        for start in range(0, len(pending), flush_every):
            for row in scorer.score(pending[start : start + flush_every]):
                f.write(json.dumps(row) + "\n")
            f.flush()
            elapsed = time.perf_counter() - started
            scored = min(start + flush_every, len(pending))
            print(f"{scored}/{len(pending)} pairs, {scored / elapsed:.1f} pairs/s", flush=True)


if __name__ == "__main__":
    main()
