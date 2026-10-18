"""Score ContraDoc with hosted decision models (TypeSafe Jev, Perplexity Decider, SubQ Snap, OpenAI).

Jev, Perplexity and SubQ share one body, {model, state, questions} with
noul/choice questions. OpenAI's Decisions API takes the same questions in a
different shape (predicate/choice arrays, text input), so requests are
translated per dialect. Wording and state are identical across providers so
differences come from the model.

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
from dataclasses import dataclass

import httpx

from cbench.candidates import Pair, read_pairs
from cbench.data import Doc, load_docs
from cbench.paths import CACHE, ROOT

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6


@dataclass(frozen=True)
class Provider:
    url: str
    model: str
    key_env: str
    max_questions: int
    dialect: str = "systemone"


PROVIDERS = {
    "jev": Provider("https://api.typesafe.ai/v1/systemone", "jev-1.13.0", "TYPESAFE_API_KEY", 64),
    # Perplexity's API serves pplx-decider-v1.2 for this name; see AGENTS.md.
    "pplx": Provider("https://api.perplexity.ai/v1/decisions", "pplx-decider-v1.1-27b", "PERPLEXITY_API_KEY", 128),
    "subq": Provider("https://api.subq.ai/v1/systemone", "subq-snap-preview", "SUBQ_API_KEY", 16),
    "openai": Provider("https://api.openai.com/v1/decisions", "gpt-6-luna", "OPENAI_API_KEY", 200, "openai"),
}


def request_body(dialect: str, model: str, state: dict, questions: dict) -> dict:
    if dialect == "systemone":
        return {"model": model, "state": state, "questions": questions}
    return {
        "model": model,
        "input": json.dumps(state, indent=1, ensure_ascii=False),
        "questions": [_openai_question(name, q) for name, q in questions.items()],
    }


def _openai_question(name: str, question: dict) -> dict:
    if question["type"] == "noul":
        return {"type": "predicate", "name": name, "instructions": question["instructions"]}
    return {
        "type": "choice",
        "name": name,
        "instructions": question["instructions"],
        "choices": [{"value": v, "description": d} for v, d in question["criteria"].items()],
    }


def parse_answers(dialect: str, body: dict) -> dict[str, dict]:
    """Normalise to {name: {"noul": p}} or {name: {"choice": c, "probabilities": {option: p}}}.

    OpenAI refusals are omitted, so callers must tolerate missing names.
    """
    if dialect == "systemone":
        return body["answers"]
    answers = {}
    for a in body["answers"]:
        if a["type"] == "predicate":
            answers[a["name"]] = {"noul": a["probability"]}
        elif a["type"] == "choice":
            answers[a["name"]] = {
                "choice": a["choice"],
                "probabilities": {p["value"]: p["probability"] for p in a["probabilities"]},
            }
    return answers

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


def load_key(env: str) -> str | None:
    if value := os.environ.get(env, "").strip():
        return value
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ").strip() == env:
                return value.strip().strip("'\"")
    return None


class Client:
    def __init__(self, provider: Provider, model: str, key: str, concurrency: int) -> None:
        self.provider = provider
        self.model = model
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(90.0)
        )
        self.semaphore = asyncio.Semaphore(concurrency)

    async def decide(self, state: dict, questions: dict) -> dict:
        body = request_body(self.provider.dialect, self.model, state, questions)
        async with self.semaphore:
            for attempt in range(MAX_ATTEMPTS):
                response = await self.http.post(self.provider.url, json=body)
                if response.status_code not in RETRY_STATUSES:
                    break
                await asyncio.sleep(_retry_delay(response, attempt))
        if response.is_error:
            raise RuntimeError(f"{response.status_code} from {self.provider.url}: {response.text[:500]}")
        body = response.json()
        return {
            "answers": parse_answers(self.provider.dialect, body),
            "model": body.get("model"),
            "usage": body.get("usage"),
        }

    async def aclose(self) -> None:
        await self.http.aclose()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return min(60.0, 2**attempt) + random.random()


async def score_pair(client: Client, pair: Pair) -> dict:
    result = await client.decide({"sentence_a": pair.a, "sentence_b": pair.b}, pair_questions())
    answers = result["answers"]
    if not {"forward", "backward"} <= answers.keys():
        raise RuntimeError(f"refused or missing answers for {pair.key}: {sorted(answers)}")
    return {
        "key": pair.key,
        "p_contra_forward": answers["forward"]["probabilities"]["contradicts"],
        "p_contra_backward": answers["backward"]["probabilities"]["contradicts"],
        "choice_forward": answers["forward"]["choice"],
        "choice_backward": answers["backward"]["choice"],
        "served_model": result.get("model"),
        "usage": result.get("usage"),
    }


async def score_doc(client: Client, doc: Doc) -> dict:
    state = native_state(doc)
    results = await asyncio.gather(
        *(client.decide(state, q) for q in chunked(native_questions(doc), client.provider.max_questions))
    )
    answers = {name: a["noul"] for r in results for name, a in r["answers"].items()}
    expected = {"document", *(f"s{i}" for i in range(len(doc.sentences)))}
    if missing := expected - answers.keys():
        raise RuntimeError(f"refused or missing answers for {doc.id}: {sorted(missing)[:5]}")
    return {
        "key": doc.id,
        "p_document": answers["document"],
        "p_sentences": [answers[f"s{i}"] for i in range(len(doc.sentences))],
        "served_model": results[0].get("model"),
        "usage": [r.get("usage") for r in results],
    }


def cache_path(provider: str, model: str, mode: str, source: str):
    return CACHE / "decide" / provider / model / f"{mode}_{source}.jsonl"


async def run(args: argparse.Namespace) -> None:
    provider = PROVIDERS[args.provider]
    model = args.model or provider.model
    if args.mode == "pairs":
        items = read_pairs(args.source)
        state, questions = {"sentence_a": items[0].a, "sentence_b": items[0].b}, pair_questions()
    else:
        items = list(load_docs())
        state, questions = native_state(items[0]), native_questions(items[0])

    out = cache_path(args.provider, model, args.mode, args.source if args.mode == "pairs" else "all")
    done = {json.loads(line)["key"] for line in out.open()} if out.exists() else set()
    pending = [it for it in items if _key(it) not in done][: args.limit]
    print(f"{args.provider}/{model} {args.mode}: {len(done)} cached, {len(pending)} pending")

    if args.dry_run:
        first = chunked(questions, provider.max_questions)[0]
        print(json.dumps(request_body(provider.dialect, model, state, first), indent=2)[:3000])
        return

    key = load_key(provider.key_env)
    if not key:
        raise SystemExit(f"set {provider.key_env} in the environment or in {ROOT / '.env'}")

    client = Client(provider, model, key, args.concurrency)
    scorer = score_pair if args.mode == "pairs" else score_doc
    out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    try:
        with out.open("a") as f:
            tasks = [asyncio.create_task(scorer(client, it)) for it in pending]
            for n, task in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    row = await task
                except RuntimeError as error:
                    failures += 1
                    print(f"failed: {error}", flush=True)
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                if n % 50 == 0 or n == len(tasks):
                    print(f"{n}/{len(tasks)} done, {failures} failed", flush=True)
    finally:
        await client.aclose()


def _key(item: Pair | Doc) -> str:
    return item.key if isinstance(item, Pair) else item.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("provider", choices=sorted(PROVIDERS))
    parser.add_argument("mode", choices=["pairs", "native"])
    parser.add_argument("--source", default="pair_test", help="pair file for pairs mode, e.g. cand_k20")
    parser.add_argument("--model", help="override the provider's default model id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="print the first request; no key needed")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
