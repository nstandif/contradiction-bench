import json
import random

import numpy as np
import pysbd
import pytest

from cbench import evaluate
from cbench.candidates import read_pairs
from cbench.data import load_docs, locate, parse_scope, split_sentences
from cbench.decide import MODELS, chunked, resolve_model
from cbench.evaluate import tune_threshold
from cbench.paths import CONTRADOC_JSON, WORK

needs_data = pytest.mark.skipif(
    not CONTRADOC_JSON.exists() or not (WORK / "cand_k20.jsonl").exists(),
    reason="run cbench.fetch and cbench.candidates --k 20 first",
)


def test_parse_scope_handles_every_stored_shape():
    assert parse_scope("local") == ("local",)
    assert parse_scope("['global']") == ("global",)
    assert parse_scope(["intra"]) == ("intra",)
    assert parse_scope("global, intra") == ("global", "intra")


def test_split_sentences_restores_missing_space_after_full_stop():
    segmenter = pysbd.Segmenter(language="en", clean=False)
    assert split_sentences(segmenter, "They left for fuel.On the planet, they found copper.") == (
        "They left for fuel.",
        "On the planet, they found copper.",
    )


def test_locate_prefers_substring_then_falls_back_to_token_overlap():
    sentences = ("Mrs. Frisby is a widowed mouse.", "She feels calm and collected after the turmoil today.")
    assert locate("She feels calm and collected after the turmoil today.", sentences) == (1,)
    assert locate("she felt calm and collected after the turmoil", sentences) == (1,)
    assert locate("completely unrelated words here", sentences) == ()


def test_resolve_model_maps_aliases_per_gateway():
    assert resolve_model("openrouter", "jev", None) == "typesafe/jev-1.13"
    assert resolve_model("vercel", "jev", None) == "typesafe-ai/jev-1.13.0"
    assert resolve_model("vercel", None, "any/slug") == "any/slug"
    assert resolve_model("subq", "snap", None) == "subq-snap-preview"
    assert "vercel" not in MODELS["span"]
    with pytest.raises(SystemExit):
        resolve_model("vercel", "span", None)
    with pytest.raises(SystemExit):
        resolve_model("openrouter", "snap", None)


def test_chunked_respects_question_limit():
    chunks = chunked({f"q{i}": {} for i in range(37)}, 16)
    assert [len(c) for c in chunks] == [16, 16, 5]
    assert list(chunks[0])[0] == "q0"


def test_tune_threshold_maximises_f1():
    labels = np.array([True, True, False, False])
    scores = np.array([0.9, 0.6, 0.5, 0.1])
    assert tune_threshold(labels, scores) == 0.6


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@needs_data
def test_evaluator_end_to_end_on_fabricated_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CACHE", tmp_path)
    rng = random.Random(0)
    docs = {d.id: d for d in load_docs()}
    pair_test, candidates = read_pairs("pair_test"), read_pairs("cand_k20")

    def nli_row(pair, score):
        return {"key": pair.key, "fingerprint": pair.fingerprint, "p_contra_ab": score, "p_contra_ba": score}

    perfect_pairs = [nli_row(p, float(p.label)) for p in pair_test]
    perfect_docs = [nli_row(p, float(docs[p.doc_id].label)) for p in candidates]
    _write(tmp_path / "nli" / "perfect" / "pair_test.jsonl", perfect_pairs)
    _write(tmp_path / "nli" / "perfect" / "cand_k20.jsonl", perfect_docs)
    _write(tmp_path / "nli" / "random" / "pair_test.jsonl", [nli_row(p, rng.random()) for p in pair_test])
    _write(tmp_path / "nli" / "partial" / "pair_test.jsonl", perfect_pairs[:10])

    native = [
        {
            "key": d.id,
            "fingerprint": d.fingerprint,
            "p_document": float(d.label),
            "p_sentences": [1.0 if i in d.evidence_idx else 0.0 for i in range(len(d.sentences))],
        }
        for d in docs.values()
    ]
    stale = dict(native[0], fingerprint="stale")
    _write(tmp_path / "decide" / "vendor__oracle" / "native_all.jsonl", native[1:] + [stale])
    _write(tmp_path / "decide" / "vendor__full" / "native_all.jsonl", native)

    report = evaluate.build_report("cand_k20")

    assert report["pair_level"]["nli:perfect"]["auroc_test"] == 1.0
    assert abs(report["pair_level"]["nli:random"]["auroc_test"] - 0.5) < 0.1
    assert "nli:partial" not in report["pair_level"]
    assert ("nli:partial", "pair_test", f"10/{len(pair_test)}") in report["incomplete"]

    assert report["doc_level"]["nli:perfect [cand_k20]"]["auroc_test"] == 1.0
    full = report["doc_level"]["vendor/full [native:max-sentence]"]
    assert full["auroc_test"] == 1.0 and full["localization"]["hit@1"] == 1.0
    assert ("vendor/oracle", "native", f"{len(docs) - 1}/{len(docs)}") in report["incomplete"]

    yes = report["always_yes"]
    assert yes["precision"] == pytest.approx(314 / 624) and yes["recall"] == 1.0
    assert "| always yes |" in evaluate.render(report)
