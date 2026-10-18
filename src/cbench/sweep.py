"""Run every decision model on OpenRouter, plus SubQ Snap, through the benchmark.

  catalog   fetch the live OpenRouter catalogue, apply exclusions -> results/catalog.json
  estimate  approximate input-token cost per model and test (no key needed)
  probe     one real request per model and test -> results/probe.json
  run       full runs for every model and test that passed its probe

Some models have constraints the catalogue doesn't state (small state caps,
choice-only heads), so the probe decides what runs instead of guesswork.
"""

import argparse
import asyncio
import json
import urllib.request
from datetime import date

from cbench.decide import QUESTIONS_PER_REQUEST, chunked, native_questions, native_state, pair_questions, score_model, work_items
from cbench.paths import RESULTS

CATALOG_URL = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
CATALOG = RESULTS / "catalog.json"
PROBE = RESULTS / "probe.json"
TESTS = ("pair_test", "native", "cand_k20")
DEFAULT_TESTS = ("pair_test", "native")
DIRECT_ROUTES = [{"route": "subq", "id": "subq-snap-preview", "price_per_token": 0.05e-6}]


def exclusion(model_id: str) -> str | None:
    if model_id.startswith("~"):
        return "alias that redirects to a pinned model already in the list"
    if model_id.endswith(":free"):
        return "free duplicate of a paid model"
    if model_id.startswith("respan/span-01"):
        return "conversation behaviour scorer with a different state contract"
    return None


def fetch_catalog() -> dict:
    with urllib.request.urlopen(CATALOG_URL, timeout=60) as response:
        data = json.load(response)["data"]
    included, excluded = [], []
    for m in sorted(data, key=lambda m: m["id"]):
        entry = {"route": "openrouter", "id": m["id"], "price_per_token": float(m["pricing"]["prompt"]),
                 "context_length": m.get("context_length")}
        if reason := exclusion(m["id"]):
            excluded.append({**entry, "reason": reason})
        else:
            included.append(entry)
    return {"fetched": date.today().isoformat(), "source": CATALOG_URL,
            "models": included + DIRECT_ROUTES, "excluded": excluded}


def load_catalog() -> dict:
    if not CATALOG.exists():
        raise SystemExit("run `python -m cbench.sweep catalog` first")
    return json.loads(CATALOG.read_text())


def mode_source(test: str) -> tuple[str, str]:
    return ("native", "all") if test == "native" else ("pairs", test)


def estimated_tokens(test: str) -> int:
    """Rough input tokens (JSON chars / 4) for one full pass of a test."""
    mode, source = mode_source(test)
    total = 0
    for item in work_items(mode, source):
        if mode == "pairs":
            bodies = [{"state": {"sentence_a": item.a, "sentence_b": item.b}, "questions": pair_questions()}]
        else:
            state = native_state(item)
            bodies = [{"state": state, "questions": q} for q in chunked(native_questions(item), QUESTIONS_PER_REQUEST)]
        total += sum(len(json.dumps(b)) for b in bodies) // 4
    return total


def selected(catalog: dict, only: list[str] | None) -> list[dict]:
    models = catalog["models"]
    if only:
        models = [m for m in models if m["id"] in only]
        if missing := set(only) - {m["id"] for m in models}:
            raise SystemExit(f"not in catalog: {sorted(missing)}")
    return models


async def probe(models: list[dict], tests: list[str]) -> dict:
    results = json.loads(PROBE.read_text()) if PROBE.exists() else {}

    async def one(m: dict, test: str) -> None:
        mode, source = mode_source(test)
        summary = await score_model(m["route"], m["id"], mode, source, limit=1, concurrency=1, log=lambda _: None)
        ok = summary.failed == 0
        results.setdefault(m["id"], {})[test] = {"ok": ok, "error": summary.first_error}
        print(f"{'ok  ' if ok else 'FAIL'} {m['id']} {test} {summary.first_error or ''}"[:300], flush=True)

    await asyncio.gather(*(one(m, t) for m in models for t in tests))
    PROBE.write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


async def run_all(models: list[dict], tests: list[str], parallel_models: int, concurrency: int) -> None:
    probes = json.loads(PROBE.read_text()) if PROBE.exists() else {}
    gate = asyncio.Semaphore(parallel_models)
    jobs = []
    for m in models:
        for test in tests:
            if not probes.get(m["id"], {}).get(test, {}).get("ok"):
                print(f"skip {m['id']} {test}: no passing probe", flush=True)
                continue
            jobs.append((m, test))

    async def one(m: dict, test: str) -> None:
        mode, source = mode_source(test)
        async with gate:
            summary = await score_model(m["route"], m["id"], mode, source, concurrency=concurrency,
                                        log=lambda line: print(line, flush=True))
        print(f"finished {m['id']} {test}: {summary.scored} scored, {summary.failed} failed", flush=True)

    await asyncio.gather(*(one(m, t) for m, t in jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["catalog", "estimate", "probe", "run"])
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=list(DEFAULT_TESTS))
    parser.add_argument("--models", nargs="+", help="restrict to these model ids")
    parser.add_argument("--parallel-models", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8, help="requests in flight per model")
    args = parser.parse_args()

    if args.command == "catalog":
        catalog = fetch_catalog()
        RESULTS.mkdir(parents=True, exist_ok=True)
        CATALOG.write_text(json.dumps(catalog, indent=2))
        print(f"{len(catalog['models'])} models, {len(catalog['excluded'])} excluded -> {CATALOG}")
        for m in catalog["excluded"]:
            print(f"  excluded {m['id']}: {m['reason']}")
        return

    models = selected(load_catalog(), args.models)
    if args.command == "estimate":
        tokens = {t: estimated_tokens(t) for t in args.tests}
        print(" ".join(f"{t}: ~{n / 1e6:.2f}M tokens" for t, n in tokens.items()))
        total = 0.0
        for m in models:
            cost = sum(tokens.values()) * m["price_per_token"]
            total += cost
            print(f"  {m['id']:<42} ${cost:.3f}")
        print(f"total ${total:.2f} for {len(models)} models")
    elif args.command == "probe":
        asyncio.run(probe(models, args.tests))
    else:
        asyncio.run(run_all(models, args.tests, args.parallel_models, args.concurrency))


if __name__ == "__main__":
    main()
