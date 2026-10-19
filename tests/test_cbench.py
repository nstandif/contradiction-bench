import json
import random

import numpy as np
import pysbd
import pytest

from cbench import evaluate, verify
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


def test_sweep_excludes_aliases_free_duplicates_and_span_01():
    from cbench.sweep import exclusion

    assert exclusion("~typesafe/jev-latest")
    assert exclusion("inception/mercury-decide:free")
    assert exclusion("respan/span-01-lite")
    assert exclusion("respan/span-02") is None
    assert exclusion("typesafe/jev-1.14") is None


def test_nli_label_index_handles_case_and_vitaminc_names():
    from cbench.nli import label_index

    assert label_index({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}, "contradiction") == 0
    assert label_index({"0": "SUPPORTS", "1": "REFUTES", "2": "NOT ENOUGH INFO"}, "contradiction") == 1
    assert label_index({"0": "SUPPORTS", "1": "REFUTES", "2": "NOT ENOUGH INFO"}, "entailment") == 0
    with pytest.raises(ValueError):
        label_index({0: "entailment", 1: "not_entailment"}, "contradiction")


@needs_data
def test_probe_uses_the_longest_input():
    from cbench.sweep import largest_item_key

    docs = load_docs()
    longest = max(docs, key=lambda d: sum(len(s) for s in d.sentences))
    assert largest_item_key("native") == longest.id
    pairs = read_pairs("pair_test")
    assert largest_item_key("pair_test") == max(pairs, key=lambda p: len(p.a) + len(p.b)).key


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

    def verify_row(d):
        shortlist = sorted(d.evidence_idx) or [0]
        options = [f"s{j}" for j in range(len(d.sentences))] + ["itself", "none"]
        probabilities = {o: 0.0 for o in options}
        probabilities["itself" if d.label else "none"] = 1.0
        answers = {f"q{i}_{o}": probabilities for i in shortlist for o in "fr"}
        return {"key": d.id, "fingerprint": d.fingerprint, "finders": ["vendor/full"], "shortlist": shortlist,
                "controls": [], "probabilities": answers, "usage": [{"cost": 0.001}]}

    _write(tmp_path / "decide" / "vendor__full" / "verify_all.jsonl", [verify_row(d) for d in docs.values()])

    report = evaluate.build_report("cand_k20")

    assert report["pair_level"]["nli:perfect"]["auroc_test"] == 1.0
    assert report["pair_level"]["nli:perfect"]["auroc_ci95_test"] == (1.0, 1.0)
    random_result = report["pair_level"]["nli:random"]
    assert abs(random_result["auroc_test"] - 0.5) < 0.1
    low, high = random_result["auroc_ci95_test"]
    assert low < random_result["auroc_test"] < high and 0.05 < high - low < 0.2
    gap = random_result["vs_best"]
    assert gap["system"] == "nli:perfect" and gap["ci95"][0] > 0.3
    assert "vs_best" not in report["pair_level"]["nli:perfect"]
    assert "_test" not in random_result
    assert "nli:partial" not in report["pair_level"]
    assert ("nli:partial", "pair_test", f"10/{len(pair_test)}") in report["incomplete"]

    assert report["doc_level"]["nli:perfect [cand_k20]"]["auroc_test"] == 1.0
    full = report["doc_level"]["vendor/full [native:max-sentence]"]
    assert full["auroc_test"] == 1.0 and full["localization"]["hit@1"] == 1.0
    assert ("vendor/oracle", "native", f"{len(docs) - 1}/{len(docs)}") in report["incomplete"]
    verified = report["doc_level"]["vendor/full [native:verify]"]
    assert verified["auroc_test"] == 1.0 and verified["localization"]["hit@1"] == 1.0
    assert verified["pipeline"] == {"finders": ["vendor/full"], "measured_cost": pytest.approx(0.001 * len(docs))}
    explained = report["explain"]["vendor/full [explain]"]
    intra = [d for d in docs.values() if evaluate.split_ids()[d.id] == "test" and d.label and d.ref_idx]
    assert explained["n"] == len(intra) and explained["pipeline"]["finders"] == ["vendor/full"]
    assert explained["pair@1"] == pytest.approx(np.mean([verify.is_gold_pair(d, min(d.evidence_idx), "itself") for d in intra]))
    assert "nli:perfect [cand_k20]" in report["explain"]
    assert "## Explain (naming the contradicting pair)" in evaluate.render(report)

    yes = report["always_yes"]
    assert yes["precision"] == pytest.approx(314 / 624) and yes["recall"] == 1.0
    assert "| always yes |" in evaluate.render(report)


def test_verify_shortlist_keeps_confident_sentences_within_bounds():
    assert verify.finder_shortlist([0.1, 0.2, 0.05]) == {1}
    assert verify.finder_shortlist([0.9, 0.4, 0.1, 0.35]) == {0, 1, 3}
    assert len(verify.finder_shortlist([0.9] * 12)) == verify.SHORTLIST_MAX


@needs_data
def test_verify_plan_unions_finders_and_draws_controls_outside_the_shortlist():
    doc = next(d for d in load_docs() if len(d.sentences) >= 10)
    a = [0.0] * len(doc.sentences); a[2] = 0.9
    b = [0.0] * len(doc.sentences); b[5] = 0.8
    shortlist, controls = verify.plan(doc, [a, b])
    assert shortlist == [2, 5]
    assert len(controls) == verify.CONTROLS and not set(controls) & set(shortlist)
    assert verify.plan(doc, [a, b]) == (shortlist, controls)


def test_verify_questions_offer_every_other_sentence_in_both_orders():
    forward, backward = verify.partner_question(1, 4, False), verify.partner_question(1, 4, True)
    assert list(forward["criteria"]) == ["s0", "s2", "s3", "itself", "none"]
    assert list(backward["criteria"]) == list(reversed(forward["criteria"]))
    assert list(verify.verify_questions(4, [1, 3])) == ["q1_f", "q1_r", "q3_f", "q3_r"]


def test_verify_scores_documents_by_p_not_none_and_sentences_by_best_partner():
    spread = {f"s{j}": 0.03 for j in range(20)} | {"itself": 0.0, "none": 0.4}
    focused = {"s0": 0.0, "s2": 0.5, "itself": 0.0, "none": 0.5}
    row = {"shortlist": [1, 4], "probabilities": {
        "q1_f": spread, "q1_r": spread, "q4_f": focused, "q4_r": focused}}
    assert verify.doc_score(row) == pytest.approx(0.6)
    assert verify.predicted_pair(row) == (4, "s2")
    assert verify.sentence_scores(row, 6) == pytest.approx([0.03, 0.03, 0.5, 0, 0.5, 0])


@needs_data
def test_explain_takes_the_finders_sentence_and_averages_verifier_partners():
    doc = next(d for d in load_docs() if d.label and d.ref_idx and d.evidence_idx != d.ref_idx)
    e, r = doc.evidence_idx[0], next(x for x in doc.ref_idx if x != doc.evidence_idx[0])
    other = next(i for i in range(len(doc.sentences)) if i not in (e, r))
    options = [f"s{j}" for j in range(len(doc.sentences))] + ["itself", "none"]
    def row(partner_of_e, partner_p):
        answers = {}
        for i, (o, p) in {e: (partner_of_e, partner_p), other: (f"s{r}", 0.99)}.items():
            probabilities = {x: 0.0 for x in options} | {o: p, "none": 1 - p}
            answers |= {f"q{i}_f": probabilities, f"q{i}_r": probabilities}
        return {"shortlist": sorted({e, other}), "probabilities": answers}
    finder = [0.0] * len(doc.sentences); finder[e] = 0.9; finder[other] = 0.4
    i, partners = verify.explain([row(f"s{r}", 0.6), row(f"s{other}", 0.5)], [finder])
    assert i == e and partners[0] == f"s{r}"
    assert verify.is_gold_pair(doc, i, partners[0]) and not verify.is_gold_pair(doc, i, partners[1])
