"""Score ContraDoc with decision models through OpenRouter, Vercel AI Gateway, or SubQ.

Both gateways accept the TypeSafe request shape, {model, state, questions}
with noul/choice questions, for every decision model they carry. SubQ Snap is
on neither gateway, so it is called on SubQ's own API, which takes the same
shape. One request format covers every model and wording is identical.

Modes:
- pairs: one request per sentence pair (same pairs as the NLI run). A 3-way
  choice is asked twice with the options in opposite orders, because Jev's
  docs report a bias toward the first option.
- native: one request per document holding all its sentences, with a noul per
  sentence and one for the whole document. Catches intra-sentence
  contradictions that pairwise methods cannot.
"""

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass

import httpx

from cbench.candidates import Pair, read_pairs
from cbench.data import Doc, load_docs
from cbench.paths import CACHE, ROOT

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
SAMPLE_SEED = 0
QUESTIONS_PER_REQUEST = 16  # SubQ's cap; applied to every model so requests stay identical
# ScaleDown rejects object states unless they hold only text/document fields; it takes the same JSON as a string.
STRING_STATE_MODELS = {"scaledown/classify-1"}


REFUSAL = re.compile(r'refused to answer question \\?"(?P<question>[^"\\]+)')


class Refused(RuntimeError):
    def __init__(self, question: str, message: str) -> None:
        super().__init__(message)
        self.question = question


@dataclass(frozen=True)
class Gateway:
    url: str
    key_env: str


GATEWAYS = {
    "openrouter": Gateway("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
    "vercel": Gateway("https://ai-gateway.vercel.sh/typesafe/v1/systemone", "AI_GATEWAY_API_KEY"),
    "subq": Gateway("https://api.subq.ai/v1/systemone", "SUBQ_API_KEY"),
}

# Catalogue slugs checked 2026-10-18. Pass --model for anything else.
MODELS = {
    "jev": {"openrouter": "typesafe/jev-1.13", "vercel": "typesafe-ai/jev-1.13.0"},
    "pplx": {"openrouter": "perplexity/pplx-decider-v1.2", "vercel": "perplexity/pplx-decider-v1.2"},
    "msft": {"openrouter": "microsoft/microsoft-decision-1", "vercel": "microsoft/microsoft-decision-1"},
    "span": {"openrouter": "respan/span-02"},
    "luna": {"openrouter": "openai/gpt-6-luna-decisions", "vercel": "openai/gpt-6-luna-decisions"},
    "snap": {"subq": "subq-snap-preview"},
}

PAIR_INSTRUCTIONS = "Both sentences come from the same document. How does `sentence_b` relate to `sentence_a`?"
PAIR_CRITERIA = {
    "contradicts": "They cannot both be true: they give incompatible facts, numbers, dates, relationships, "
    "causes, feelings, or opinions about the same entity or event",
    "compatible": "They can both be true together, including when one adds detail the other lacks",
    "unrelated": "They are about different things, so neither bears on the other",
}
CONTRADICTION_DEFINITION = (
    "A contradiction means statements that cannot both be true about the same entity or event: "
    "incompatible facts, numbers, dates, relationships, causes, feelings, or opinions."
)


def pair_questions() -> dict:
    forward = dict(PAIR_CRITERIA)
    backward = dict(reversed(PAIR_CRITERIA.items()))
    return {
        "forward": {"type": "choice", "instructions": PAIR_INSTRUCTIONS, "criteria": forward},
        "backward": {"type": "choice", "instructions": PAIR_INSTRUCTIONS, "criteria": backward},
    }


def native_state(doc: Doc) -> dict:
    return {"sentences": [{"id": i, "text": s} for i, s in enumerate(doc.sentences)]}


def native_questions(doc: Doc) -> dict:
    questions = {
        "document": {
            "type": "noul",
            "instructions": "Does the document in `sentences` contradict itself anywhere? "
            + CONTRADICTION_DEFINITION,
        }
    }
    for i in range(len(doc.sentences)):
        questions[f"s{i}"] = {
            "type": "noul",
            "instructions": f"Does `sentences[{i}].text` contradict another sentence in `sentences`, "
            f"or contradict itself? " + CONTRADICTION_DEFINITION,
        }
    return questions


def chunked(questions: dict, size: int) -> list[dict]:
    items = list(questions.items())
    return [dict(items[i : i + size]) for i in range(0, len(items), size)]


def wire_state(model: str, state: dict) -> dict | str:
    return json.dumps(state, ensure_ascii=False) if model in STRING_STATE_MODELS else state


def resolve_model(gateway: str, alias: str | None, slug: str | None) -> str:
    if slug:
        return slug
    if alias is None:
        raise SystemExit("pass a model alias or --model <slug>")
    if gateway not in MODELS[alias]:
        raise SystemExit(f"{alias} is not on {gateway}; available there: "
                         f"{sorted(a for a, g in MODELS.items() if gateway in g)}")
    return MODELS[alias][gateway]


def load_env(name: str) -> str | None:
    if value := os.environ.get(name, "").strip():
        return value
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip().removeprefix("export ").strip() == name:
                return value.strip().strip("'\"")
    return None


class Client:
    def __init__(self, gateway: Gateway, model: str, key: str, concurrency: int) -> None:
        self.gateway = gateway
        self.model = model
        self.http = httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(90.0))
        self.semaphore = asyncio.Semaphore(concurrency)

    async def decide(self, state: dict, questions: dict) -> dict:
        body = {"model": self.model, "state": wire_state(self.model, state), "questions": questions}
        async with self.semaphore:
            response = await self._post(body)
        if response.is_error:
            if match := REFUSAL.search(response.text):
                raise Refused(match["question"], response.text[:500])
            raise RuntimeError(f"{response.status_code} from {self.gateway.url}: {response.text[:500]}")
        return response.json()

    async def _post(self, body: dict) -> httpx.Response:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self.http.post(self.gateway.url, json=body)
            except httpx.TransportError as error:
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError(f"network error after {MAX_ATTEMPTS} attempts: {error!r}") from error
                await asyncio.sleep(_backoff(attempt))
                continue
            refused = REFUSAL.search(response.text) is not None
            if response.status_code not in RETRY_STATUSES or refused or attempt == MAX_ATTEMPTS - 1:
                return response
            await asyncio.sleep(_retry_delay(response, attempt))
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self.http.aclose()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return _backoff(attempt)


def _backoff(attempt: int) -> float:
    return min(60.0, 2**attempt) + random.random()


def served_model(result: dict) -> str | None:
    """OpenRouter reports the resolved version in `model`; Vercel in its routing metadata."""
    routing = result.get("provider_metadata", {}).get("gateway", {}).get("routing", {})
    return routing.get("canonicalSlug") or result.get("model")


async def score_pair(client: Client, pair: Pair) -> dict:
    result = await client.decide({"sentence_a": pair.a, "sentence_b": pair.b}, pair_questions())
    answers = result["answers"]
    return {
        "key": pair.key,
        "fingerprint": pair.fingerprint,
        "p_contra_forward": answers["forward"]["probabilities"]["contradicts"],
        "p_contra_backward": answers["backward"]["probabilities"]["contradicts"],
        "choice_forward": answers["forward"]["choice"],
        "choice_backward": answers["backward"]["choice"],
        "served_model": served_model(result),
        "usage": result.get("usage"),
    }


async def decide_skipping_refusals(client: Client, state: dict, questions: dict) -> tuple[dict | None, list[str]]:
    """A provider refusal names one question; drop it and ask the rest again."""
    refused = []
    while questions:
        try:
            return await client.decide(state, questions), refused
        except Refused as refusal:
            if refusal.question not in questions:
                raise
            refused.append(refusal.question)
            questions = {k: v for k, v in questions.items() if k != refusal.question}
    return None, refused


async def score_doc(client: Client, doc: Doc) -> dict:
    state = native_state(doc)
    outcomes = await asyncio.gather(
        *(decide_skipping_refusals(client, state, q) for q in chunked(native_questions(doc), QUESTIONS_PER_REQUEST))
    )
    results = [r for r, _ in outcomes if r is not None]
    refused = [q for _, qs in outcomes for q in qs]
    answers = {name: a["noul"] for r in results for name, a in r["answers"].items()}
    row = {
        "key": doc.id,
        "fingerprint": doc.fingerprint,
        "p_document": answers.get("document"),
        "p_sentences": [answers.get(f"s{i}") for i in range(len(doc.sentences))],
        "served_model": served_model(results[0]) if results else None,
        "usage": [r.get("usage") for r in results],
    }
    return {**row, "refused": refused} if refused else row


def cache_path(model: str, mode: str, source: str):
    return CACHE / "decide" / model.replace("/", "__") / f"{mode}_{source}.jsonl"


def cached_fingerprints(path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {(row := json.loads(line))["key"]: row.get("fingerprint") for line in path.open()}


def work_items(mode: str, source: str) -> list[Pair] | list[Doc]:
    return read_pairs(source) if mode == "pairs" else list(load_docs())


def first_request(model: str, mode: str, items: list[Pair] | list[Doc]) -> dict:
    if mode == "pairs":
        state, questions = {"sentence_a": items[0].a, "sentence_b": items[0].b}, pair_questions()
    else:
        state, questions = native_state(items[0]), native_questions(items[0])
    return {"model": model, "state": wire_state(model, state), "questions": chunked(questions, QUESTIONS_PER_REQUEST)[0]}


def require_key(gateway: Gateway) -> str:
    if key := load_env(gateway.key_env):
        return key
    raise SystemExit(f"set {gateway.key_env} in the environment or in {ROOT / '.env'}")


@dataclass
class RunSummary:
    scored: int = 0
    failed: int = 0
    first_error: str | None = None


async def score_model(
    gateway_name: str,
    model: str,
    mode: str,
    source: str,
    *,
    limit: int | None = None,
    only_keys: set[str] | None = None,
    concurrency: int = 8,
    log=print,
) -> RunSummary:
    gateway = GATEWAYS[gateway_name]
    items = [it for it in work_items(mode, source) if only_keys is None or _key(it) in only_keys]
    out = cache_path(model, mode, source if mode == "pairs" else "all")
    done = cached_fingerprints(out)
    pending = [it for it in items if done.get(_key(it)) != it.fingerprint]
    if limit is not None:
        pending = random.Random(SAMPLE_SEED).sample(pending, min(limit, len(pending)))
    log(f"{gateway_name}/{model} {mode}: {len(items) - len(pending)} cached, {len(pending)} pending")

    summary = RunSummary()
    if not pending:
        return summary
    client = Client(gateway, model, require_key(gateway), concurrency)
    scorer = score_pair if mode == "pairs" else score_doc
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("a") as f:
            tasks = [asyncio.create_task(scorer(client, it)) for it in pending]
            for n, task in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    row = await task
                except (RuntimeError, KeyError, TypeError, ValueError) as error:
                    summary.failed += 1
                    summary.first_error = summary.first_error or f"{error!r}"[:600]
                    log(f"failed: {error!r}"[:600])
                    continue
                summary.scored += 1
                f.write(json.dumps(row) + "\n")
                f.flush()
                if n % 50 == 0 or n == len(tasks):
                    log(f"{gateway_name}/{model} {mode}: {n}/{len(tasks)} done, {summary.failed} failed")
    finally:
        await client.aclose()
    return summary


async def run(args: argparse.Namespace) -> None:
    model = resolve_model(args.gateway, args.alias, args.model)
    if args.dry_run:
        print(f"POST {GATEWAYS[args.gateway].url}")
        print(json.dumps(first_request(model, args.mode, work_items(args.mode, args.source)), indent=2)[:3000])
        return
    await score_model(args.gateway, model, args.mode, args.source, limit=args.limit, concurrency=args.concurrency)


def _key(item: Pair | Doc) -> str:
    return item.key if isinstance(item, Pair) else item.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gateway", choices=sorted(GATEWAYS))
    parser.add_argument("mode", choices=["pairs", "native"])
    parser.add_argument("alias", nargs="?", choices=sorted(MODELS), help="model alias; or pass --model")
    parser.add_argument("--model", help="exact gateway model slug, overriding the alias")
    parser.add_argument("--source", default="pair_test", help="pair file for pairs mode, e.g. cand_k20")
    parser.add_argument("--limit", type=int, help="score a seeded random sample of N pending items")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="print the first request; no key needed")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
