"""Blocking is deterministic; arbitration is tested through its pure seam."""

import json
from pathlib import Path

from src.llm.openrouter import _completion_from
from src.models import Provenance, Triple
from src.pipeline.resolve import (
    candidate_blocks,
    normalize,
    parse_resolution_response,
    run,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openrouter" / "resolve_response.json"


def block_containing(name, blocks):
    return next(b for b in blocks if name in b)


def test_blocking_groups_prefix_and_abbreviation_variants():
    blocks = candidate_blocks(["Dana", "Dana Whitfield", "Rivet Labs"])
    dana = block_containing("Dana Whitfield", blocks)
    assert "Dana" in dana
    assert "Rivet Labs" not in dana


def test_blocking_groups_initialisms():
    blocks = candidate_blocks(["JFK", "John F Kennedy"])
    assert len(block_containing("JFK", blocks)) == 2


def test_blocking_keeps_unrelated_names_apart():
    blocks = candidate_blocks(["Kestrel", "Halyard", "Hollis"])
    assert all(len(b) == 1 for b in blocks)


def test_blocking_is_case_and_punctuation_insensitive():
    assert normalize("D. Whitfield") == "d whitfield"
    blocks = candidate_blocks(["dana whitfield", "Dana Whitfield"])
    assert len(blocks) == 1 and len(blocks[0]) == 2


def test_every_input_name_lands_in_exactly_one_group():
    cluster = ["Dana", "D. Whitfield", "Dana Whitfield", "Dana Reyes"]
    body = json.loads(FIXTURE.read_text(encoding="utf-8"))
    groups = parse_resolution_response(_completion_from(body, "m").text, cluster)

    placed = [m for g in groups for m in g["members"]]
    assert sorted(placed) == sorted(cluster), "a name was lost or duplicated"


def test_names_the_model_dropped_come_back_as_singletons():
    """Silently losing an entity here deletes every fact attached to it."""
    cluster = ["Dana", "Dana Whitfield", "Ghost Entity"]
    raw = json.dumps(
        {"groups": [{"canonical": "Dana Whitfield", "members": ["Dana", "Dana Whitfield"]}]}
    )
    groups = parse_resolution_response(raw, cluster)
    assert any(g["members"] == ["Ghost Entity"] for g in groups)


def test_invented_names_are_discarded():
    cluster = ["Dana"]
    raw = json.dumps(
        {"groups": [{"canonical": "Dana", "members": ["Dana", "Someone Not In The Cluster"]}]}
    )
    groups = parse_resolution_response(raw, cluster)
    placed = [m for g in groups for m in g["members"]]
    assert placed == ["Dana"]


def test_a_name_is_never_claimed_by_two_groups():
    cluster = ["Dana", "Dana Whitfield"]
    raw = json.dumps(
        {
            "groups": [
                {"canonical": "Dana Whitfield", "members": ["Dana", "Dana Whitfield"]},
                {"canonical": "Dana", "members": ["Dana"]},
            ]
        }
    )
    groups = parse_resolution_response(raw, cluster)
    placed = [m for g in groups for m in g["members"]]
    assert len(placed) == len(set(placed))


def test_paraphrased_canonical_falls_back_to_a_real_member():
    cluster = ["Dana", "Dana Whitfield"]
    raw = json.dumps(
        {"groups": [{"canonical": "Ms. Dana Whitfield (data lead)", "members": cluster}]}
    )
    groups = parse_resolution_response(raw, cluster)
    assert groups[0]["canonical"] in cluster


def _triple(s, s_type, p, o, o_type):
    prov = Provenance(block_uid="b1", page_title="P", origin="llm", extracted_at="2026-01-01T00:00:00Z")
    return Triple(subject=s, subject_type=s_type, predicate=p, object=o, object_type=o_type, provenance=prov)


def test_run_picks_one_type_per_canonical_name_by_majority_vote():
    """The exact-match path needs no LLM call, so this exercises type voting
    without touching the network: three mentions of the SAME name, typed
    differently across separate (isolated) extraction calls — place, source,
    concept, concept. Without a canonical type, those would silently split
    into separate nodes at assembly despite sharing one name — see the
    live-data regression test in test_assemble_store_query.py."""
    triples = [
        _triple("Polaris", "place", "etymology_from", "Latin", "concept"),
        _triple("Polaris", "source", "means", "pole star", "concept"),
        _triple("Polaris", "concept", "used_for", "navigation", "concept"),
        _triple("Polaris", "concept", "indicates", "latitude", "concept"),
    ]
    result = run(triples, use_llm=False)
    assert result.type_for("Polaris", fallback="MISSING") == "concept"


def test_type_for_falls_back_when_name_never_seen():
    from src.pipeline.resolve import ResolutionMap

    result = ResolutionMap()
    assert result.type_for("Nobody", fallback="concept") == "concept"
