import numpy as np
import pysbd

from cbench.data import locate, parse_scope, split_sentences
from cbench.decide import chunked, pair_questions, parse_answers, request_body
from cbench.evaluate import tune_threshold


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


def test_openai_translation_round_trips_choice_and_predicate():
    questions = {**pair_questions(), "doc": {"type": "noul", "instructions": "Any contradiction?"}}
    body = request_body("openai", "gpt-6-luna", {"sentence_a": "x", "sentence_b": "y"}, questions)
    assert isinstance(body["input"], str) and '"sentence_a": "x"' in body["input"]
    forward = body["questions"][0]
    assert forward["name"] == "forward" and [c["value"] for c in forward["choices"]][0] == "contradicts"
    assert [c["value"] for c in body["questions"][1]["choices"]][0] == "unrelated"
    assert body["questions"][2] == {"type": "predicate", "name": "doc", "instructions": "Any contradiction?"}

    response = {
        "answers": [
            {"type": "choice", "name": "forward", "choice": "contradicts", "confidence": 0.9,
             "probabilities": [{"value": "contradicts", "probability": 0.9}, {"value": "compatible", "probability": 0.1}]},
            {"type": "refusal", "name": "backward"},
            {"type": "predicate", "name": "doc", "probability": 0.7},
        ]
    }
    answers = parse_answers("openai", response)
    assert answers["forward"]["probabilities"]["contradicts"] == 0.9
    assert answers["doc"] == {"noul": 0.7}
    assert "backward" not in answers


def test_chunked_respects_provider_question_limit():
    questions = {f"q{i}": {} for i in range(37)}
    chunks = chunked(questions, 16)
    assert [len(c) for c in chunks] == [16, 16, 5]
    assert list(chunks[0])[0] == "q0"


def test_tune_threshold_maximises_f1():
    labels = np.array([True, True, False, False])
    scores = np.array([0.9, 0.6, 0.5, 0.1])
    assert tune_threshold(labels, scores) == 0.6
