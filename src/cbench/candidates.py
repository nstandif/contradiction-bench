"""Build the sentence pairs every pairwise system scores.

Two inputs, written to data/work/:
- pair_test: gold contradictory pairs (positives) against the most similar pair
  in each clean document (hard negatives). Tests the models in isolation.
- cand_k{K}: the top-K most lexically similar pairs per document. Shared by
  every pairwise system so doc-level results compare models, not pair choices.
"""

import argparse
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from cbench.data import Doc, fingerprint, load_docs
from cbench.paths import WORK

DEFAULT_K = 20


@dataclass(frozen=True)
class Pair:
    key: str
    doc_id: str
    i: int
    j: int
    a: str
    b: str
    label: bool | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.a, self.b)


def similarity_matrices(docs: tuple[Doc, ...]) -> dict[str, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectorizer.fit(s for d in docs for s in d.sentences)
    matrices = {}
    for d in docs:
        m = vectorizer.transform(d.sentences)
        matrices[d.id] = (m @ m.T).toarray()
    return matrices


def ranked_pairs(sim: np.ndarray) -> list[tuple[int, int]]:
    n = sim.shape[0]
    pairs = list(itertools.combinations(range(n), 2))
    return sorted(pairs, key=lambda p: (-sim[p], p))


def build_pair_test(docs: tuple[Doc, ...], sims: dict[str, np.ndarray]) -> list[Pair]:
    pairs = []
    for d in docs:
        if d.label:
            if not d.gold_pairs:
                continue
            i, j = min(d.gold_pairs)
            label = True
        else:
            ranked = ranked_pairs(sims[d.id])
            if not ranked:
                continue
            i, j = ranked[0]
            label = False
        pairs.append(Pair(f"{d.id}:{i}:{j}", d.id, i, j, d.sentences[i], d.sentences[j], label))
    return pairs


def build_candidates(docs: tuple[Doc, ...], sims: dict[str, np.ndarray], k: int) -> list[Pair]:
    return [
        Pair(f"{d.id}:{i}:{j}", d.id, i, j, d.sentences[i], d.sentences[j])
        for d in docs
        for i, j in ranked_pairs(sims[d.id])[:k]
    ]


def candidate_recall(docs: tuple[Doc, ...], pairs: list[Pair]) -> dict[str, float | int]:
    chosen: dict[str, set[tuple[int, int]]] = {}
    for p in pairs:
        chosen.setdefault(p.doc_id, set()).add((p.i, p.j))
    gold_docs = [d for d in docs if d.gold_pairs]
    hit_docs = sum(1 for d in gold_docs if d.gold_pairs & chosen.get(d.id, set()))
    evidence_docs = [d for d in docs if d.evidence_idx]
    covered = sum(
        1
        for d in evidence_docs
        if any(e in pair for pair in chosen.get(d.id, set()) for e in d.evidence_idx)
    )
    return {
        "docs_with_gold_pair": len(gold_docs),
        "gold_pair_recall": hit_docs / len(gold_docs),
        "docs_with_evidence": len(evidence_docs),
        "evidence_sentence_recall": covered / len(evidence_docs),
    }


def write_pairs(name: str, pairs: list[Pair]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / f"{name}.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__) + "\n")


def read_pairs(name: str) -> list[Pair]:
    return [Pair(**json.loads(line)) for line in _lines(name)]


def _lines(name: str) -> Iterator[str]:
    with (WORK / f"{name}.jsonl").open() as f:
        yield from f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    docs = load_docs()
    sims = similarity_matrices(docs)

    pair_test = build_pair_test(docs, sims)
    write_pairs("pair_test", pair_test)
    print(
        f"pair_test: {sum(p.label for p in pair_test)} gold positives, "
        f"{sum(not p.label for p in pair_test)} hard negatives"
    )

    candidates = build_candidates(docs, sims, args.k)
    write_pairs(f"cand_k{args.k}", candidates)
    print(f"cand_k{args.k}: {len(candidates)} pairs")
    print(json.dumps(candidate_recall(docs, candidates), indent=2))


if __name__ == "__main__":
    main()
