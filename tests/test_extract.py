"""Extraction is tested through its pure seam, against a recorded reply.

No test in this file touches the network. `parse_extraction_response` is where
every decision that can corrupt the graph is made, so that is what is asserted.
"""

import json
from pathlib import Path

import pytest

from src.llm.openrouter import _completion_from, _is_retryable_body_error, extract_json
from src.models import ORIGIN_LLM, Provenance, Triple
from src.pipeline.extract import _write_checkpoint, load_checkpoint, parse_extraction_response

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openrouter" / "extract_response.json"

BLOCK_TEXT = "Dana runs the data team at Rivet Labs and has been there since 2019."


@pytest.fixture
def reply():
    body = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return _completion_from(body, "anthropic/claude-haiku-4.5").text


def parse(reply, text=BLOCK_TEXT):
    return parse_extraction_response(
        reply,
        block_uid="blk-003",
        page_title="Project Halyard",
        block_text=text,
        model="anthropic/claude-haiku-4.5",
    )


def test_extract_json_survives_code_fences(reply):
    assert isinstance(extract_json(reply), list)


def test_extract_json_survives_leading_prose():
    assert extract_json('Sure, here you go:\n{"a": 1}') == {"a": 1}


def test_extract_json_heals_truncated_array():
    truncated = (
        '[\n  {"subject": "a", "predicate": "p", "object": "b"},\n'
        '  {"subject": "c", "predicate": "p", "object": "d"},\n'
        '  {"subject": "the author", "subject_type": "person", "predi'
    )
    assert extract_json(truncated) == [
        {"subject": "a", "predicate": "p", "object": "b"},
        {"subject": "c", "predicate": "p", "object": "d"},
    ]


def test_extract_json_truncated_with_no_complete_objects_raises():
    with pytest.raises(ValueError):
        extract_json('[\n  {"subject": "a", "predi')


def test_grounded_triples_are_kept(reply):
    triples = parse(reply)
    pairs = {(t.subject, t.predicate, t.object) for t in triples}
    assert ("Dana", "works_at", "Rivet Labs") in pairs


def test_ungrounded_triple_is_dropped(reply):
    """The fixture's third triple quotes text the block does not contain.

    That is the model answering from its own knowledge of the world. Admitting
    it would put a fact in the graph that no note supports.
    """
    triples = parse(reply)
    assert all(t.object != "NASA" for t in triples), "an unquotable fact was admitted"


def test_every_triple_carries_llm_provenance(reply):
    for triple in parse(reply):
        assert triple.provenance.origin == ORIGIN_LLM
        assert triple.provenance.block_uid == "blk-003"
        assert triple.provenance.model
        assert triple.provenance.prompt_version


def test_malformed_elements_do_not_lose_the_batch():
    raw = json.dumps(
        [
            "not a dict",
            {"subject": "A"},  # no predicate or object
            {"subject": "X", "predicate": "knows", "object": "X"},  # self-edge
            {"subject": "A", "predicate": "knows", "object": "B", "confidence": "high"},
        ]
    )
    triples = parse(raw, text="A knows B")
    assert len(triples) == 1
    assert triples[0].confidence == 0.5, "unparseable confidence should fall back"


def test_unknown_entity_type_falls_back_to_concept():
    raw = json.dumps(
        [{"subject": "A", "subject_type": "spaceship", "predicate": "p", "object": "B"}]
    )
    assert parse(raw, text="A p B")[0].subject_type == "concept"


def test_empty_array_is_a_valid_answer():
    assert parse("[]") == []


def test_non_array_reply_raises():
    with pytest.raises(ValueError):
        parse('{"subject": "A"}')


def test_retryable_body_error_recognizes_the_504_abort_shape():
    """Seen for real during an OpenRouter rate-limit window: a 200 response
    whose body carries the failure instead of an HTTP error status. Before
    this classifier existed, that shape skipped retry entirely."""
    assert _is_retryable_body_error({"message": "The operation was aborted", "code": 504})
    assert _is_retryable_body_error({"message": "rate limited, please retry", "code": 429})


def test_non_retryable_body_error_is_not_retried():
    assert not _is_retryable_body_error({"message": "invalid model id", "code": 400})
    assert not _is_retryable_body_error("just a string, not a dict")


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "extract_checkpoint.json"
    triple = Triple(
        subject="A",
        subject_type="concept",
        predicate="p",
        object="B",
        object_type="concept",
        subject_description="",
        object_description="",
        confidence=1.0,
        provenance=Provenance(
            block_uid="blk-1",
            page_title="Page",
            origin=ORIGIN_LLM,
            extracted_at="2026-01-01T00:00:00+00:00",
            model="m",
            prompt_version="v1",
            quote=None,
        ),
    )
    _write_checkpoint(path, [triple], [{"uid": "blk-2", "error": "boom"}], ["blk-1", "blk-2"])

    triples, failures, done_uids = load_checkpoint(path)
    assert triples == [triple]
    assert failures == [{"uid": "blk-2", "error": "boom"}]
    assert done_uids == {"blk-1", "blk-2"}
