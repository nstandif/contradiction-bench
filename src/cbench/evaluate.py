"""Compare every cached system and write results/report.md and results/metrics.json.

Thresholds are tuned for F1 on the dev split and applied to test. AUROC is the
headline number because max-over-pairs scores drift with document length.
Cached rows whose fingerprint no longer matches the current text are ignored,
and a system is reported only once it has scored every item of an input.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from cbench.candidates import DEFAULT_K, candidate_recall, read_pairs
from cbench.data import Doc, docs_by_id, load_docs, split_ids
from cbench.paths import CACHE, RESULTS, WORK
from cbench import verify


@dataclass
class System:
    name: str
    pair_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    doc_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    sentence_scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    order_gap: dict[str, float] = field(default_factory=dict)
    pipelines: dict[str, dict] = field(default_factory=dict)
    verify_rows: list[dict] = field(default_factory=list)
    incomplete: dict[str, str] = field(default_factory=dict)
    refused_questions: int = 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def current_rows(path: Path, expected: dict[str, str]) -> list[dict]:
    """Rows for items that still exist with identical text; the last write wins."""
    rows = {r["key"]: r for r in read_jsonl(path) if expected.get(r["key"]) == r.get("fingerprint")}
    return list(rows.values())


def pair_fingerprints(source: str) -> dict[str, str] | None:
    if not (WORK / f"{source}.jsonl").exists():
        return None
    return {p.key: p.fingerprint for p in read_pairs(source)}


def _add_pair_scores(system: System, source: str, rows: list[dict], expected: int, score) -> None:
    if len(rows) < expected:
        system.incomplete[source] = f"{len(rows)}/{expected}"
        return
    system.pair_scores[source] = {r["key"]: score(r) for r in rows}


def discover_systems() -> list[System]:
    systems = []
    for model_dir in sorted((CACHE / "nli").glob("*")):
        system = System(f"nli:{model_dir.name.split('__')[-1]}")
        for path in sorted(model_dir.glob("*.jsonl")):
            if (expected := pair_fingerprints(path.stem)) is None:
                continue
            rows = current_rows(path, expected)
            _add_pair_scores(system, path.stem, rows, len(expected), lambda r: max(r["p_contra_ab"], r["p_contra_ba"]))
        systems.append(system)

    doc_fingerprints = {d.id: d.fingerprint for d in load_docs()}
    for model_dir in sorted((CACHE / "decide").glob("*")):
        system = System(model_dir.name.replace("__", "/"))
        for path in sorted(model_dir.glob("pairs_*.jsonl")):
            source = path.stem.removeprefix("pairs_")
            if (expected := pair_fingerprints(source)) is None:
                continue
            rows = current_rows(path, expected)
            _add_pair_scores(
                system, source, rows, len(expected), lambda r: (r["p_contra_forward"] + r["p_contra_backward"]) / 2
            )
            if source in system.pair_scores:
                system.order_gap[source] = float(
                    np.mean([abs(r["p_contra_forward"] - r["p_contra_backward"]) for r in rows])
                )
        native = current_rows(model_dir / "native_all.jsonl", doc_fingerprints)
        if native and len(native) < len(doc_fingerprints):
            system.incomplete["native"] = f"{len(native)}/{len(doc_fingerprints)}"
        elif native:
            system.refused_questions = sum(len(r.get("refused", [])) for r in native)
            sentences = {r["key"]: [p or 0.0 for p in r["p_sentences"]] for r in native}
            system.doc_scores["native:document"] = {r["key"]: r["p_document"] or 0.0 for r in native}
            system.doc_scores["native:max-sentence"] = {k: max(v) for k, v in sentences.items()}
            system.sentence_scores["native"] = sentences
        _add_verify(system, model_dir, doc_fingerprints)
        systems.append(system)
    return systems


def _add_verify(system: System, model_dir: Path, doc_fingerprints: dict[str, str]) -> None:
    rows = current_rows(model_dir / "verify_all.jsonl", doc_fingerprints)
    if not rows:
        return
    if len(rows) < len(doc_fingerprints):
        system.incomplete["verify"] = f"{len(rows)}/{len(doc_fingerprints)}"
        return
    if len({tuple(r["finders"]) for r in rows}) > 1:
        system.incomplete["verify"] = "mixed finders"
        return
    lengths = {d.id: len(d.sentences) for d in load_docs()}
    system.doc_scores["native:verify"] = {r["key"]: verify.doc_score(r) for r in rows}
    system.sentence_scores["native:verify"] = {r["key"]: verify.sentence_scores(r, lengths[r["key"]]) for r in rows}
    system.pipelines["native:verify"] = {"finders": rows[0]["finders"], "measured_cost": verify.measured_cost(rows)}
    system.verify_rows = rows


def tune_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_f1, best_t = -1.0, 0.5
    for t in np.unique(scores):
        f1 = _f1(labels, scores >= t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    tp = np.sum(labels & predicted)
    denominator = 2 * tp + np.sum(~labels & predicted) + np.sum(labels & ~predicted)
    return float(2 * tp / denominator) if denominator else 0.0


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    return {
        "f1": _f1(labels, predicted),
        "accuracy": float(np.mean(labels == predicted)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else None


BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 0


def auroc_interval(labels: np.ndarray, scores: np.ndarray, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap over test items; each item is one document, so this resamples documents."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, len(labels), len(labels))
        if (value := auroc(labels[idx], scores[idx])) is not None:
            values.append(value)
    if not values:
        return None
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def add_paired_deltas(results: dict[str, dict], level: float = 0.95) -> None:
    """AUROC gap to the best system, bootstrapped on shared resamples of the same test items.

    Every system scores the same test documents, so paired resampling is far
    tighter than comparing per-system intervals. Consumes each result's `_test`.
    """
    scored = {name: r for name, r in results.items() if r["auroc_test"] is not None}
    if not scored:
        return
    best = max(scored, key=lambda name: scored[name]["auroc_test"])
    best_keys, labels, best_scores = scored[best]["_test"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = [rng.integers(0, len(labels), len(labels)) for _ in range(BOOTSTRAP_ROUNDS)]
    tail = (1 - level) / 2 * 100
    for name, result in scored.items():
        keys, _, scores = result["_test"]
        if name == best or keys != best_keys:
            continue
        deltas = [
            a - b
            for idx in resamples
            if (a := auroc(labels[idx], best_scores[idx])) is not None
            and (b := auroc(labels[idx], scores[idx])) is not None
        ]
        result["vs_best"] = {
            "system": best,
            "delta": scored[best]["auroc_test"] - result["auroc_test"],
            "ci95": (float(np.percentile(deltas, tail)), float(np.percentile(deltas, 100 - tail))),
        }
    for result in results.values():
        result.pop("_test", None)


def evaluate_scores(
    scores: dict[str, float],
    label_of: Callable[[str], bool],
    split_of: Callable[[str], str],
) -> dict | None:
    keys = sorted(scores)
    dev = [k for k in keys if split_of(k) == "dev"]
    test = [k for k in keys if split_of(k) == "test"]
    if not dev or not test:
        return None
    y_dev = np.array([label_of(k) for k in dev])
    y_test = np.array([label_of(k) for k in test])
    s_dev = np.array([scores[k] for k in dev])
    s_test = np.array([scores[k] for k in test])
    threshold = tune_threshold(y_dev, s_dev)
    return {
        "_test": (tuple(test), y_test, s_test),
        "n_test": len(test),
        "auroc_test": auroc(y_test, s_test),
        "auroc_ci95_test": auroc_interval(y_test, s_test),
        "threshold": threshold,
        **{f"{k}_test": v for k, v in binary_metrics(y_test, s_test >= threshold).items()},
        "predicted_test": {k: bool(scores[k] >= threshold) for k in test},
    }


def pair_level(systems: list[System]) -> dict:
    pairs = {p.key: p for p in read_pairs("pair_test")}
    docs, splits = docs_by_id(), split_ids()
    results = {}
    for system in systems:
        scores = system.pair_scores.get("pair_test")
        if not scores:
            continue
        result = evaluate_scores(scores, lambda k: pairs[k].label, lambda k: splits[pairs[k].doc_id])
        if result is None:
            continue
        negatives = [k for k in scores if not pairs[k].label]
        by_type = {}
        for contra_type in sorted({t for p in pairs.values() if p.label for t in docs[p.doc_id].contra_types}):
            positives = [k for k in scores if pairs[k].label and contra_type in docs[pairs[k].doc_id].contra_types]
            keys = positives + negatives
            by_type[contra_type] = {
                "n_pos": len(positives),
                "auroc": auroc(np.array([pairs[k].label for k in keys]), np.array([scores[k] for k in keys])),
            }
        result.pop("predicted_test")
        results[system.name] = {
            **result,
            "n_scored": len(scores),
            "auroc_by_type_all_splits": by_type,
            "order_gap": system.order_gap.get("pair_test"),
        }
    add_paired_deltas(results)
    return results


def doc_scores_from_pairs(pair_scores: dict[str, float]) -> tuple[dict, dict]:
    """Doc score = max pair score; sentence score = max over pairs containing it.

    `pair_scores` is complete for its source (enforced in discover_systems).
    """
    doc_max: dict[str, float] = {}
    sentence_max: dict[str, dict[int, float]] = defaultdict(dict)
    for key, score in pair_scores.items():
        doc_id, i, j = key.rsplit(":", 2)
        doc_max[doc_id] = max(doc_max.get(doc_id, 0.0), score)
        for s in (int(i), int(j)):
            sentence_max[doc_id][s] = max(sentence_max[doc_id].get(s, 0.0), score)
    return doc_max, dict(sentence_max)


def localization(sentence_scores: dict, docs: dict[str, Doc], splits: dict[str, str]) -> dict:
    hits = {1: 0, 3: 0}
    n = 0
    for doc_id, scores in sentence_scores.items():
        doc = docs[doc_id]
        if not doc.label or splits[doc_id] != "test" or not doc.evidence_idx:
            continue
        ranked = (
            sorted(range(len(scores)), key=lambda i: -scores[i])
            if isinstance(scores, list)
            else sorted(scores, key=lambda i: -scores[i])
        )
        n += 1
        for k in hits:
            hits[k] += bool(set(ranked[:k]) & set(doc.evidence_idx))
    return {"n": n, **{f"hit@{k}": v / n if n else None for k, v in hits.items()}}


def doc_level(systems: list[System], source: str) -> dict:
    docs, splits = docs_by_id(), split_ids()
    results = {}
    configurations = []
    for system in systems:
        if source in system.pair_scores:
            doc_scores, sentence_scores = doc_scores_from_pairs(system.pair_scores[source])
            configurations.append((f"{system.name} [{source}]", doc_scores, sentence_scores, None, {}))
        for name, scores in system.doc_scores.items():
            sentences = system.sentence_scores.get(name, system.sentence_scores.get("native", {}))
            extra = {"refused_questions": system.refused_questions} if system.refused_questions else {}
            configurations.append((f"{system.name} [{name}]", scores, sentences, system.pipelines.get(name), extra))
    for name, doc_scores, sentence_scores, pipeline, extra in configurations:
        result = evaluate_scores(doc_scores, lambda k: docs[k].label, lambda k: splits[k])
        if result is None:
            continue
        predicted = result.pop("predicted_test")
        results[name] = {
            **result,
            "n_docs_scored": len(doc_scores),
            "auroc_by_length_tertile": length_tertiles(doc_scores, docs, splits),
            "recall_by": recall_breakdown(predicted, docs),
            "localization": localization(sentence_scores, docs, splits),
            **({"pipeline": pipeline} if pipeline else {}),
            **extra,
        }
    add_paired_deltas(results)
    return results


def explainable_docs(docs: dict[str, Doc], splits: dict[str, str]) -> list[str]:
    return sorted(k for k, d in docs.items() if splits[k] == "test" and d.label and d.ref_idx)


def pair_hits(hits: list[tuple[bool, bool]], level: float = 0.95) -> dict:
    """pair@1 / pair@2 with a percentile bootstrap interval for pair@1 over the same documents."""
    top1, top2 = np.array([h[0] for h in hits]), np.array([h[1] for h in hits])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = [top1[rng.integers(0, len(top1), len(top1))].mean() for _ in range(BOOTSTRAP_ROUNDS)]
    tail = (1 - level) / 2 * 100
    return {
        "n": len(hits),
        "pair@1": float(top1.mean()),
        "pair@1_ci95": (float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))),
        "pair@2": float(top2.mean()),
    }


def explain_level(systems: list[System], source: str) -> dict:
    """How often each system's top pair (and top two) is a gold contradicting pair."""
    docs, splits = docs_by_id(), split_ids()
    keys = explainable_docs(docs, splits)
    results = {}
    verifiers = [s for s in systems if s.verify_rows]
    groups = [[v] for v in verifiers]
    if len({tuple(v.verify_rows[0]["finders"]) for v in verifiers}) == 1 and len(verifiers) > 1:
        groups.append(verifiers)
    native = {s.name: s.sentence_scores["native"] for s in systems if "native" in s.sentence_scores}
    for group in groups:
        finders = group[0].verify_rows[0]["finders"]
        if any(m not in native for m in finders):
            continue
        by_doc = [{r["key"]: r for r in v.verify_rows} for v in group]
        hits = []
        for k in keys:
            i, partners = verify.explain([rows[k] for rows in by_doc], [native[m][k] for m in finders])
            hits.append((verify.is_gold_pair(docs[k], i, partners[0]),
                         any(verify.is_gold_pair(docs[k], i, o) for o in partners[:2])))
        name = " + ".join(v.name for v in group) + " [explain]"
        results[name] = {**pair_hits(hits), "pipeline": {
            "finders": finders, "measured_cost": sum(verify.measured_cost(v.verify_rows) for v in group)}}
    for system in systems:
        if source not in system.pair_scores:
            continue
        ranked = defaultdict(list)
        for key, score in system.pair_scores[source].items():
            doc_id, i, j = key.rsplit(":", 2)
            ranked[doc_id].append((score, (int(i), int(j))))
        hits = []
        for k in keys:
            top = [pair for _, pair in sorted(ranked.get(k, []), reverse=True)[:2]]
            gold = [tuple(sorted(p)) in docs[k].gold_pairs for p in top]
            hits.append((bool(gold[:1] and gold[0]), any(gold)))
        results[f"{system.name} [{source}]"] = pair_hits(hits)
    return results


def length_tertiles(scores: dict[str, float], docs: dict[str, Doc], splits: dict[str, str]) -> dict:
    test = [k for k in scores if splits[k] == "test"]
    lengths = np.array([len(docs[k].sentences) for k in test])
    cuts = np.quantile(lengths, [1 / 3, 2 / 3])
    bins = np.digitize(lengths, cuts)
    out = {}
    for b, label in enumerate(["short", "medium", "long"]):
        keys = [k for k, kb in zip(test, bins) if kb == b]
        out[label] = auroc(np.array([docs[k].label for k in keys]), np.array([scores[k] for k in keys]))
    return out


def recall_breakdown(predicted: dict[str, bool], docs: dict[str, Doc]) -> dict:
    """Recall over test positives; contra_type is multi-label, so documents count once per type."""
    groups: dict[str, dict[str, list[bool]]] = {"contra_type": defaultdict(list), "scope": defaultdict(list), "doc_type": defaultdict(list)}
    for doc_id, hit in predicted.items():
        doc = docs[doc_id]
        if not doc.label:
            continue
        for t in doc.contra_types:
            groups["contra_type"][t].append(hit)
        groups["scope"]["+".join(doc.scopes)].append(hit)
        groups["doc_type"][doc.doc_type].append(hit)
    return {g: {k: {"n": len(v), "recall": sum(v) / len(v)} for k, v in sorted(vals.items())} for g, vals in groups.items()}


def always_yes(docs: tuple[Doc, ...], splits: dict[str, str]) -> dict:
    labels = np.array([d.label for d in docs if splits[d.id] == "test"])
    return binary_metrics(labels, np.ones_like(labels, dtype=bool))


def render(report: dict) -> str:
    lines = ["# Contradiction detection on ContraDoc", ""]
    lines += [
        f"Test split: {report['n_test_docs']} documents. Thresholds tuned for F1 on the dev split.",
        f"AUROC brackets are 95% bootstrap intervals ({BOOTSTRAP_ROUNDS} resamples of test documents). "
        "To compare two systems use the gap column instead: it resamples the same documents for both, "
        "so it is much tighter. A gap whose interval excludes 0 is a real difference; with many systems "
        "in one table, expect a few such results by chance.",
        "",
        "## Candidate pairs",
        "",
        "Pairwise systems only see the top-K TF-IDF pairs per document, so this caps their recall.",
        "",
        "| K | Gold pair in candidates | Evidence sentence in candidates |",
        "|---|---|---|",
    ]
    for k, r in report["candidate_recall"].items():
        lines.append(f"| {k} | {r['gold_pair_recall']:.1%} of {r['docs_with_gold_pair']} | {r['evidence_sentence_recall']:.1%} of {r['docs_with_evidence']} |")
    lines += ["", "## Pair level (gold pairs vs hardest pair in clean documents)", ""]
    lines += ["| System | AUROC | Gap to best | F1 | Acc | Order gap | n test |", "|---|---|---|---|---|---|---|"]
    for name, r in sorted(report["pair_level"].items(), key=lambda x: -(x[1]["auroc_test"] or 0)):
        gap = f"{r['order_gap']:.3f}" if r["order_gap"] is not None else "-"
        lines.append(
            f"| {name} | {_auroc(r)} | {_gap(r)} | {_f(r['f1_test'])} | {_f(r['accuracy_test'])} | {gap} | {r['n_test']} |"
        )
    lines += ["", "## Document level", ""]
    yes = report["always_yes"]
    lines += [
        "| System | AUROC | Gap to best | F1 | Acc | P | R | Hit@1 | Hit@3 | Docs |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| always yes | - | - | {_f(yes['f1'])} | {_f(yes['accuracy'])} | {_f(yes['precision'])} | {_f(yes['recall'])} | - | - | - |",
    ]
    for name, r in sorted(report["doc_level"].items(), key=lambda x: -(x[1]["auroc_test"] or 0)):
        loc = r["localization"]
        mark = "*" if r.get("refused_questions") else ""
        lines.append(
            f"| {name}{mark} | {_auroc(r)} | {_gap(r)} | {_f(r['f1_test'])} | {_f(r['accuracy_test'])} | {_f(r['precision_test'])} "
            f"| {_f(r['recall_test'])} | {_f(loc['hit@1'])} | {_f(loc['hit@3'])} | {r['n_docs_scored']} |"
        )
    if refused := {n.split(" [")[0]: r["refused_questions"] for n, r in report["doc_level"].items() if r.get("refused_questions")}:
        lines += ["", "\\* The provider refused some questions, which count as \"no\": "
                  + ", ".join(f"{name} ({n})" for name, n in sorted(refused.items())) + "."]
    if report.get("explain"):
        n = next(iter(report["explain"].values()))["n"]
        lines += [
            "",
            "## Explain (naming the contradicting pair)",
            "",
            f"Contradictory test documents with a labelled partner sentence ({n}). Pair@1: the top predicted pair "
            "is a gold pair; Pair@2: one of the top two is. Explain systems take the sentence the finders rate "
            "highest and the partner the verifiers name for it; pairwise systems use their highest-scoring "
            "candidate pairs, so candidate recall caps them.",
            "",
            "| System | Pair@1 | 95% CI | Pair@2 |",
            "|---|---|---|---|",
        ]
        for name, r in sorted(report["explain"].items(), key=lambda x: -x[1]["pair@1"]):
            lo, hi = r["pair@1_ci95"]
            lines.append(f"| {name} | {r['pair@1']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['pair@2']:.3f} |")
    lines += ["", "### Recall by contradiction type (test positives; multi-label)", ""]
    configs = sorted(report["doc_level"])
    types = sorted({t for r in report["doc_level"].values() for t in r["recall_by"]["contra_type"]})
    if configs:
        lines += ["| Type | " + " | ".join(configs) + " |", "|---" * (len(configs) + 1) + "|"]
        for t in types:
            cells = []
            for c in configs:
                v = report["doc_level"][c]["recall_by"]["contra_type"].get(t)
                cells.append(f"{v['recall']:.2f} (n={v['n']})" if v else "-")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines += ["", "### Recall by scope", "", "| Scope | " + " | ".join(configs) + " |", "|---" * (len(configs) + 1) + "|"]
        scopes = sorted({s for r in report["doc_level"].values() for s in r["recall_by"]["scope"]})
        for s in scopes:
            cells = []
            for c in configs:
                v = report["doc_level"][c]["recall_by"]["scope"].get(s)
                cells.append(f"{v['recall']:.2f} (n={v['n']})" if v else "-")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    if report["incomplete"]:
        lines += ["", "## Not reported (partial runs)", ""]
        lines += [f"- {system} [{source}]: {progress}" for system, source, progress in report["incomplete"]]
    return "\n".join(lines) + "\n"


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _gap(result: dict) -> str:
    if not (vs := result.get("vs_best")):
        return "best" if result.get("auroc_test") is not None else "-"
    lo, hi = vs["ci95"]
    return f"{vs['delta']:.3f} [{lo:.3f}, {hi:.3f}]"


def _auroc(result: dict) -> str:
    ci = result.get("auroc_ci95_test")
    return _f(result["auroc_test"]) + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def build_report(source: str) -> dict:
    docs, splits = load_docs(), split_ids()
    systems = discover_systems()
    return {
        "n_test_docs": sum(1 for d in docs if splits[d.id] == "test"),
        "candidate_recall": {
            name.removeprefix("cand_k"): candidate_recall(docs, read_pairs(name))
            for name in sorted(p.stem for p in WORK.glob("cand_k*.jsonl"))
        },
        "pair_level": pair_level(systems),
        "doc_level": doc_level(systems, source),
        "explain": explain_level(systems, source),
        "always_yes": always_yes(docs, splits),
        "incomplete": [(s.name, src, progress) for s in systems for src, progress in s.incomplete.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=f"cand_k{DEFAULT_K}")
    args = parser.parse_args()

    report = build_report(args.source)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    (RESULTS / "report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
