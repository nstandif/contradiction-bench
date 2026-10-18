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


@dataclass
class System:
    name: str
    pair_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    doc_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    sentence_scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    order_gap: dict[str, float] = field(default_factory=dict)
    incomplete: dict[str, str] = field(default_factory=dict)


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
            system.doc_scores["native:document"] = {r["key"]: r["p_document"] for r in native}
            system.doc_scores["native:max-sentence"] = {r["key"]: max(r["p_sentences"]) for r in native}
            system.sentence_scores["native"] = {r["key"]: r["p_sentences"] for r in native}
        systems.append(system)
    return systems


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
        "n_test": len(test),
        "auroc_test": auroc(y_test, s_test),
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
            configurations.append((f"{system.name} [{source}]", doc_scores, sentence_scores))
        for name, scores in system.doc_scores.items():
            configurations.append((f"{system.name} [{name}]", scores, system.sentence_scores.get("native", {})))
    for name, doc_scores, sentence_scores in configurations:
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
        }
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
    lines += ["| System | AUROC | F1 | Acc | Order gap | n test |", "|---|---|---|---|---|---|"]
    for name, r in sorted(report["pair_level"].items(), key=lambda x: -(x[1]["auroc_test"] or 0)):
        gap = f"{r['order_gap']:.3f}" if r["order_gap"] is not None else "-"
        lines.append(
            f"| {name} | {_f(r['auroc_test'])} | {_f(r['f1_test'])} | {_f(r['accuracy_test'])} | {gap} | {r['n_test']} |"
        )
    lines += ["", "## Document level", ""]
    yes = report["always_yes"]
    lines += [
        "| System | AUROC | F1 | Acc | P | R | Hit@1 | Hit@3 | Docs |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| always yes | - | {_f(yes['f1'])} | {_f(yes['accuracy'])} | {_f(yes['precision'])} | {_f(yes['recall'])} | - | - | - |",
    ]
    for name, r in sorted(report["doc_level"].items(), key=lambda x: -(x[1]["auroc_test"] or 0)):
        loc = r["localization"]
        lines.append(
            f"| {name} | {_f(r['auroc_test'])} | {_f(r['f1_test'])} | {_f(r['accuracy_test'])} | {_f(r['precision_test'])} "
            f"| {_f(r['recall_test'])} | {_f(loc['hit@1'])} | {_f(loc['hit@3'])} | {r['n_docs_scored']} |"
        )
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
