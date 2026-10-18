"""Load ContraDoc, split documents into sentences, and locate the gold evidence."""

import json
import re
from dataclasses import dataclass
from functools import cache

import pysbd
from sklearn.model_selection import train_test_split

from cbench.paths import CONTRADOC_JSON

DEV_FRACTION = 0.3
SPLIT_SEED = 0
MIN_MATCH_CHARS = 20
MIN_TOKEN_OVERLAP = 0.6


@dataclass(frozen=True)
class Doc:
    id: str
    label: bool
    doc_type: str
    sentences: tuple[str, ...]
    contra_types: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    evidence_idx: tuple[int, ...] = ()
    ref_idx: tuple[int, ...] = ()

    @property
    def gold_pairs(self) -> set[tuple[int, int]]:
        return {
            (min(e, r), max(e, r))
            for e in self.evidence_idx
            for r in self.ref_idx
            if e != r
        }


@cache
def load_docs() -> tuple[Doc, ...]:
    raw = json.loads(CONTRADOC_JSON.read_text())
    segmenter = pysbd.Segmenter(language="en", clean=False)
    docs = []
    for doc_id, item in raw["pos"].items():
        sentences = split_sentences(segmenter, item["text"])
        docs.append(
            Doc(
                id=doc_id,
                label=True,
                doc_type=item["doc_type"],
                sentences=sentences,
                contra_types=tuple(item["contra_type"]),
                scopes=parse_scope(item["scope"]),
                evidence_idx=locate(item["evidence"], sentences),
                ref_idx=tuple(
                    sorted({i for ref in item.get("ref sentences") or [] for i in locate(ref, sentences)})
                ),
            )
        )
    for doc_id, item in raw["neg"].items():
        docs.append(
            Doc(
                id=doc_id,
                label=False,
                doc_type=item["doc_type"],
                sentences=split_sentences(segmenter, item["text"]),
            )
        )
    return tuple(docs)


@cache
def split_ids() -> dict[str, str]:
    """Fixed stratified dev/test split; thresholds are tuned on dev only."""
    docs = sorted(load_docs(), key=lambda d: d.id)
    dev, _ = train_test_split(
        [d.id for d in docs],
        test_size=1 - DEV_FRACTION,
        stratify=[d.label for d in docs],
        random_state=SPLIT_SEED,
    )
    dev = set(dev)
    return {d.id: "dev" if d.id in dev else "test" for d in docs}


def docs_by_id() -> dict[str, Doc]:
    return {d.id: d for d in load_docs()}


def split_sentences(segmenter: pysbd.Segmenter, text: str) -> tuple[str, ...]:
    """Some source texts drop the space after a full stop ("fuel.On"); restore it first."""
    text = re.sub(r"(?<=[a-z][.!?])(?=[A-Z])", " ", text)
    return tuple(s.strip() for s in segmenter.segment(text) if s.strip())


def parse_scope(scope: str | list[str]) -> tuple[str, ...]:
    """Scopes appear as 'local', "['local']", ['local'], or 'global, intra'."""
    items = scope if isinstance(scope, list) else [scope]
    names = []
    for item in items:
        names += [p.strip(" '\"") for p in item.strip("[]").split(",")]
    return tuple(sorted({n for n in names if n}))


def locate(target: str, sentences: tuple[str, ...]) -> tuple[int, ...]:
    """Indices of sentences holding `target`: substring match, else best token overlap."""
    t = _norm(target)
    if len(t) >= MIN_MATCH_CHARS:
        hits = tuple(
            i
            for i, s in enumerate(sentences)
            if len(n := _norm(s)) >= MIN_MATCH_CHARS and (n in t or t in n)
        )
        if hits:
            return hits
    target_tokens = set(_tokens(t))
    if not target_tokens:
        return ()
    overlaps = [len(target_tokens & set(_tokens(_norm(s)))) / len(target_tokens) for s in sentences]
    best = max(range(len(sentences)), key=overlaps.__getitem__, default=None)
    if best is None or overlaps[best] < MIN_TOKEN_OVERLAP:
        return ()
    return (best,)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text)
