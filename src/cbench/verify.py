"""Find → verify and find → explain: finders shortlist suspicious sentences, a verifier names what each contradicts.

The finders are cached native runs. Each keeps the sentences it scores at least
SHORTLIST_MIN_SCORE (at least 1, at most SHORTLIST_MAX) and the shortlist is their
union. The verifier reads the whole document and answers one choice question per
shortlisted sentence: which other sentence it contradicts, `itself`, or `none`.
Each question is asked in both option orders and averaged. Two random sentences
outside the shortlist are asked too, as per-document controls.

Verify detects: the document score is the highest 1 − P(none) over the shortlist,
chosen on dev over the single most likely partner's probability.

Explain names the pair: the finders' mean native score picks the sentence, and the
verifiers' averaged answer for it gives ranked partners. Letting verify's own
confidence pick the sentence names the gold pair less often.
"""

import argparse
import asyncio
import json
import zlib
from pathlib import Path

import numpy as np

from cbench.data import Doc, load_docs
from cbench.decide import (
    CONTRADICTION_DEFINITION,
    GATEWAYS,
    QUESTIONS_PER_REQUEST,
    Client,
    cache_path,
    chunked,
    native_state,
    require_key,
    served_model,
)

FINDERS = ("liquid/d1-ultra", "perplexity/pplx-decider-v1.2")
VERIFIER = "perplexity/pplx-decider-v1.2"
SHORTLIST_MIN_SCORE = 0.3
SHORTLIST_MAX = 5
CONTROLS = 2
NONE, ITSELF = "none", "itself"


def finder_shortlist(sentence_scores: list[float]) -> set[int]:
    scores = np.array(sentence_scores)
    keep = max(1, min(SHORTLIST_MAX, int(np.sum(scores >= SHORTLIST_MIN_SCORE))))
    return set(np.argsort(-scores)[:keep].tolist())


def plan(doc: Doc, finder_scores: list[list[float]]) -> tuple[list[int], list[int]]:
    shortlist = sorted(set().union(*(finder_shortlist(s) for s in finder_scores)))
    rest = [i for i in range(len(doc.sentences)) if i not in shortlist]
    rng = np.random.default_rng(zlib.crc32(doc.id.encode()))
    controls = sorted(rng.choice(rest, min(CONTROLS, len(rest)), replace=False).tolist()) if rest else []
    return shortlist, controls


def partner_question(i: int, n: int, reverse: bool) -> dict:
    options = {f"s{j}": f"It contradicts `sentences[{j}].text`" for j in range(n) if j != i}
    options[ITSELF] = f"`sentences[{i}].text` contradicts itself"
    options[NONE] = "It contradicts no sentence in the document"
    items = list(options.items())
    return {
        "type": "choice",
        "criteria": dict(reversed(items) if reverse else items),
        "instructions": f"Read the whole document in `sentences`. Which sentence does `sentences[{i}].text` "
        "contradict, if any? " + CONTRADICTION_DEFINITION,
    }


def verify_questions(n: int, sentences: list[int]) -> dict:
    return {f"q{i}_{order}": partner_question(i, n, order == "r") for i in sentences for order in "fr"}


async def score_doc(client: Client, doc: Doc, finders: tuple[str, ...], finder_scores: list[list[float]]) -> dict:
    shortlist, controls = plan(doc, finder_scores)
    questions = verify_questions(len(doc.sentences), shortlist + controls)
    results = await asyncio.gather(
        *(client.decide(native_state(doc), q) for q in chunked(questions, QUESTIONS_PER_REQUEST))
    )
    answers = {name: a for r in results for name, a in r["answers"].items()}
    return {
        "key": doc.id,
        "fingerprint": doc.fingerprint,
        "finders": list(finders),
        "shortlist": shortlist,
        "controls": controls,
        "probabilities": {name: a["probabilities"] for name, a in answers.items()},
        "choices": {name: a["choice"] for name, a in answers.items()},
        "served_model": served_model(results[0]),
        "usage": [r.get("usage") for r in results],
    }


def answer(row: dict, i: int) -> dict[str, float]:
    forward, backward = row["probabilities"][f"q{i}_f"], row["probabilities"][f"q{i}_r"]
    return {option: (forward[option] + backward[option]) / 2 for option in forward}


def best_partner(probabilities: dict[str, float]) -> tuple[str, float]:
    return max(((o, p) for o, p in probabilities.items() if o != NONE), key=lambda x: x[1])


def sentence_scores(row: dict, n: int) -> list[float]:
    """Each shortlisted sentence and its best partner score that sentence's best-partner probability."""
    scores = [0.0] * n
    for i in row["shortlist"]:
        partner, p = best_partner(answer(row, i))
        for j in (i, int(partner[1:])) if partner.startswith("s") else (i,):
            scores[j] = max(scores[j], p)
    return scores


def doc_score(row: dict) -> float:
    return max(1 - answer(row, i)[NONE] for i in row["shortlist"])


def predicted_pair(row: dict) -> tuple[int, str]:
    i = max(row["shortlist"], key=lambda i: best_partner(answer(row, i))[1])
    return i, best_partner(answer(row, i))[0]


def finder_choice(row: dict, finder_scores: list[list[float]]) -> int:
    return max(row["shortlist"], key=lambda i: np.mean([s[i] for s in finder_scores]))


def explain(rows: list[dict], finder_scores: list[list[float]]) -> tuple[int, list[str]]:
    """Finder-chosen sentence and its partners ranked by the verifiers' averaged answers."""
    i = finder_choice(rows[0], finder_scores)
    answers = [answer(r, i) for r in rows]
    mean = {o: float(np.mean([a[o] for a in answers])) for o in answers[0] if o != NONE}
    return i, sorted(mean, key=lambda o: -mean[o])


def is_gold_pair(doc: Doc, i: int, option: str) -> bool:
    if option == ITSELF:
        return any(e == r == i for e in doc.evidence_idx for r in doc.ref_idx)
    return tuple(sorted((i, int(option[1:])))) in doc.gold_pairs


def measured_cost(rows: list[dict]) -> float:
    return sum(u.get("cost") or 0.0 for r in rows for u in r["usage"] if u)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def native_sentence_scores(model: str) -> dict[str, tuple[str, list[float]]]:
    return {r["key"]: (r["fingerprint"], r["p_sentences"]) for r in read_rows(cache_path(model, "native", "all"))}


async def run(args: argparse.Namespace) -> None:
    finders = tuple(args.finders)
    docs = list(load_docs())
    finder_rows = {m: native_sentence_scores(m) for m in finders}
    missing = {m: sum(finder_rows[m].get(d.id, ("",))[0] != d.fingerprint for d in docs) for m in finders}
    if any(missing.values()):
        raise SystemExit(f"finders need complete native runs first; missing documents: {missing}")
    out = cache_path(args.verifier, "verify", "all")
    done = {r["key"]: r for r in read_rows(out)}
    pending = [
        d for d in docs
        if (r := done.get(d.id)) is None or r["fingerprint"] != d.fingerprint or tuple(r["finders"]) != finders
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{args.verifier} verify: {len(docs) - len(pending)} cached, {len(pending)} pending")
    if not pending:
        return
    gateway = GATEWAYS[args.gateway]
    client = Client(gateway, args.verifier, require_key(gateway), args.concurrency)
    failed = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("a") as f:
            tasks = [
                asyncio.create_task(score_doc(client, d, finders, [finder_rows[m][d.id][1] for m in finders]))
                for d in pending
            ]
            for n, task in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    row = await task
                except (RuntimeError, KeyError, TypeError, ValueError) as error:
                    failed += 1
                    print(f"failed: {error!r}"[:600])
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                if n % 50 == 0 or n == len(tasks):
                    print(f"{n}/{len(tasks)} done, {failed} failed")
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", choices=sorted(GATEWAYS), default="openrouter")
    parser.add_argument("--verifier", default=VERIFIER)
    parser.add_argument("--finders", nargs="+", default=list(FINDERS))
    parser.add_argument("--limit", type=int, help="score only the first N pending documents")
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
