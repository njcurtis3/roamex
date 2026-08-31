"""The back half of the pipeline: assemble -> store -> subgraph -> grounded answer.

Everything here is offline. The one model-shaped step (`query`) is exercised
through `parse_query_response` against a recorded reply.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.llm.openrouter import _completion_from
from src.models import (
    ORIGIN_LLM,
    Graph,
    Provenance,
    Triple,
    canonical_id,
)
from src.pipeline import query as query_stage
from src.pipeline.assemble import assemble, stats
from src.pipeline.query import parse_query_response, serialize
from src.pipeline.resolve import ResolutionMap
from src.roam import parse as rp
from src.store.graph import GraphStore
from src.eval import score

ROAM_FIXTURE = Path(__file__).parent.parent / "fixtures" / "roam" / "sample_export.json"
QUERY_FIXTURE = Path(__file__).parent.parent / "fixtures" / "openrouter" / "query_response.json"

NOW = datetime.now(timezone.utc).isoformat()


def prov(uid="blk-003"):
    return Provenance(
        block_uid=uid,
        page_title="Project Halyard",
        origin=ORIGIN_LLM,
        extracted_at=NOW,
        model="test-model",
        prompt_version="extract/v1",
        quote="Dana runs the data team at Rivet Labs",
    )


@pytest.fixture
def base():
    return rp.parse(json.loads(ROAM_FIXTURE.read_text(encoding="utf-8")))


@pytest.fixture
def triples():
    return [
        Triple(
            subject="Dana",
            subject_type="person",
            predicate="works_at",
            object="Rivet Labs",
            object_type="org",
            provenance=prov(),
            confidence=0.95,
        ),
        Triple(
            subject="Dana",
            subject_type="person",
            predicate="wrote",
            object="Kestrel parser",
            object_type="tool",
            provenance=prov("blk-008"),
            confidence=0.9,
        ),
    ]


def test_resolved_name_attaches_to_the_existing_roam_page(base, triples):
    """"Dana" resolved to "Dana Whitfield" must land on the page, not beside it."""
    resolution = ResolutionMap(mapping={"dana": "Dana Whitfield"})
    graph = assemble(base, triples, resolution)

    page_id = canonical_id("page", "Dana Whitfield")
    assert page_id in graph.nodes
    assert any(e.source_id == page_id and e.predicate == "works_at" for e in graph.edges)
    assert "Dana" in graph.nodes[page_id].aliases

    person_ids = [n.id for n in graph.nodes.values() if n.name == "Dana Whitfield"]
    assert len(person_ids) == 1, "a shadow node was created beside the page"


def test_llm_entity_with_no_matching_page_becomes_a_new_node(base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    kestrel = [n for n in graph.nodes.values() if n.name == "Kestrel parser"]
    assert len(kestrel) == 1 and kestrel[0].type == "tool"


def test_resolved_name_with_inconsistent_types_still_collapses_to_one_node(base):
    """Reproduces a real bug found running roamex on live notes 2026-08-28.

    Each extraction call sees one block in isolation, so the same entity can
    come back typed `place` in one triple and `concept` in another. Because
    `canonical_id` keys on (type, name), that alone silently split "Polaris" /
    "Stella Polaris" into THREE separate nodes even though resolve correctly
    merged the name. Resolution must decide the type too, or its merges do
    not actually collapse anything in the graph.
    """
    triples = [
        Triple(subject="Polaris", subject_type="place",
               predicate="etymology_from", object="Stella Polaris",
               object_type="concept", provenance=prov()),
        Triple(subject="Stella Polaris", subject_type="source",
               predicate="means", object="pole star", object_type="concept",
               provenance=prov("blk-008")),
        Triple(subject="Polaris", subject_type="concept",
               predicate="used_for", object="navigation", object_type="concept",
               provenance=prov("blk-009")),
    ]
    resolution = ResolutionMap(
        mapping={"polaris": "Stella Polaris", "stella polaris": "Stella Polaris"},
        types={"stella polaris": "concept"},  # majority vote: 2 concept, 1 place, 1 source
    )
    graph = assemble(base, triples, resolution)

    polaris_nodes = [n for n in graph.nodes.values() if "Polaris" in n.name]
    assert len(polaris_nodes) == 1, (
        f"expected one merged node, got {len(polaris_nodes)}: "
        f"{[(n.id, n.type) for n in polaris_nodes]}"
    )
    assert polaris_nodes[0].type == "concept"


def test_self_edge_from_over_merge_is_dropped(base):
    """If resolution collapses both ends, the "fact" is about nothing."""
    triples = [
        Triple(
            subject="Dana",
            subject_type="person",
            predicate="is",
            object="D. Whitfield",
            object_type="person",
            provenance=prov(),
        )
    ]
    resolution = ResolutionMap(
        mapping={"dana": "Dana Whitfield", "d whitfield": "Dana Whitfield"}
    )
    graph = assemble(base, triples, resolution)
    assert not any(e.predicate == "is" for e in graph.edges)


def test_dangling_edge_is_rejected_loudly():
    graph = Graph()
    from src.models import Edge, Node

    graph.add_node(Node.make("page", "A"))
    with pytest.raises(ValueError, match="unknown node"):
        graph.add_edge(
            Edge(
                source_id=canonical_id("page", "A"),
                predicate="p",
                target_id="page:doesnotexist",
                provenance=prov(),
            )
        )


def test_roundtrip_through_the_store_preserves_provenance(tmp_path, base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    db = tmp_path / "g.db"
    with GraphStore(db) as store:
        store.write(graph)
    with GraphStore(db) as store:
        loaded = store.load()

    assert len(loaded.nodes) == len(graph.nodes)
    assert len(loaded.edges) == len(graph.edges)
    for edge in loaded.edges:
        assert edge.provenance.block_uid, "an edge came back without its source block"


def test_subgraph_is_bounded_by_hops(tmp_path, base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    db = tmp_path / "g.db"
    with GraphStore(db) as store:
        store.write(graph)
        seed = canonical_id("page", "Dana Whitfield")
        near = store.subgraph([seed], hops=1)
        far = store.subgraph([seed], hops=3)
    assert len(near.nodes) <= len(far.nodes)
    assert seed in near.nodes


def test_find_nodes_matches_aliases(tmp_path, base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    db = tmp_path / "g.db"
    with GraphStore(db) as store:
        store.write(graph)
        assert [n.name for n in store.find_nodes("Dana")] != []


def test_serialize_labels_every_triple_with_its_source(base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    text, index = serialize(graph)
    assert index, "nothing was serialized"
    for line in text.splitlines():
        assert "block " in line, f"a triple was shown without provenance: {line}"


def test_bare_citation_identifiers_are_repaired_not_rejected():
    """Reproduces a real reply from gemini-3.6-flash 2026-08-28: valid JSON
    except `"citations": [e1, e3, e5]` used bare identifiers instead of
    quoted strings, which is invalid JSON and raised out of every real query
    before this repair existed. Traced to query/v1's own schema example
    showing an unquoted `edge_id` placeholder — fixed in query/v2, but the
    repair stays as defense-in-depth for whatever model repeats it."""
    raw = (
        '{\n  "answer": "The graph does not connect these.",\n'
        '  "citations": [e1, e3, e5],\n  "sufficient": false\n}'
    )
    index = {f"e{i}": {"edge_id": f"e{i}", "source": "A", "predicate": "p",
                        "target": "B", "page_title": "P", "block_uid": "b",
                        "origin": ORIGIN_LLM, "quote": None} for i in (1, 3, 5)}
    answer = parse_query_response(raw, index, "q", "m")
    assert len(answer.citations) == 3
    assert answer.invalid_citations == []


def test_repair_does_not_touch_a_lookalike_word_in_the_answer_prose():
    """The repair is scoped to the citations array so it can't corrupt prose
    that happens to contain a token shaped like an edge id."""
    raw = (
        '{\n  "answer": "See variable e1 in the source code for context.",\n'
        '  "citations": ["e1"],\n  "sufficient": true\n}'
    )
    index = {"e1": {"edge_id": "e1", "source": "A", "predicate": "p", "target": "B",
                     "page_title": "P", "block_uid": "b", "origin": ORIGIN_LLM, "quote": None}}
    answer = parse_query_response(raw, index, "q", "m")
    assert "e1" in answer.answer  # untouched
    assert len(answer.citations) == 1


def test_hallucinated_citation_is_flagged_not_silently_accepted():
    """The fixture cites `e99`, which was never shown to the model."""
    body = json.loads(QUERY_FIXTURE.read_text(encoding="utf-8"))
    reply = _completion_from(body, "m").text
    index = {
        "e1": {"edge_id": "e1", "source": "Dana Whitfield", "predicate": "works_at",
               "target": "Rivet Labs", "page_title": "P", "block_uid": "b1",
               "origin": ORIGIN_LLM, "quote": None},
        "e2": {"edge_id": "e2", "source": "Rivet Labs", "predicate": "builds_for",
               "target": "Project Halyard", "page_title": "P", "block_uid": "b2",
               "origin": ORIGIN_LLM, "quote": None},
    }
    answer = parse_query_response(reply, index, "who works at Rivet Labs?", "m")
    assert len(answer.citations) == 2
    assert answer.invalid_citations == ["e99"]


def test_truncated_reply_salvages_the_partial_answer_instead_of_raising():
    """Reproduces a real reply from 2026-08-30: max_tokens cut the reply off
    mid-string, inside the `"answer"` value, before any `[` ever appeared —
    so extract_json's array healer has nothing to salvage and every real
    query before this fix raised straight through as "Query failed: no JSON
    found...", discarding an answer the model had mostly finished writing."""
    raw = (
        '{\n  "answer": "Based on the provided graph, **AI 2027** is a subject '
        'read [e92, e93] by the author, and serves as the author\'s best gu'
    )
    answer = parse_query_response(raw, {}, "what can you tell me about AI 2027?", "m")
    assert answer.answer.startswith("Based on the provided graph, **AI 2027**")
    assert "cut off" in answer.answer
    assert answer.sufficient is False
    assert answer.citations == []


def test_truncated_reply_with_no_answer_field_still_raises():
    """A reply cut off before `"answer"` even started has nothing to salvage —
    this should still fail loudly (caught by ask()'s own guard) rather than
    silently returning an empty answer."""
    raw = '{\n  "citat'
    with pytest.raises(ValueError):
        parse_query_response(raw, {}, "q", "m")


def test_provenance_coverage_is_total(base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    report = score.score_provenance(graph)
    assert report["node_coverage"] == 1.0
    assert report["edge_coverage"] == 1.0


def test_ask_fails_gracefully_instead_of_crashing_the_cli(tmp_path, base, triples, monkeypatch):
    """Reproduces a real failure found running roamex on live notes 2026-08-28:
    a truncated/malformed model reply raised out of `ask()` uncaught, taking
    down the whole `query` CLI command with a stack trace. extract.run() and
    resolve.run() both fail closed instead; ask() must too."""
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    db = tmp_path / "g.db"
    with GraphStore(db) as store:
        store.write(graph)

        def boom(*a, **kw):
            raise ValueError("no JSON found in model reply: '{\"answer\": \"trunc")

        monkeypatch.setattr(query_stage, "complete", boom)
        answer = query_stage.ask(store, "who works at Rivet Labs?")

    assert answer.sufficient is False
    assert "Query failed" in answer.answer
    assert answer.seeds, "seed names should still be populated even on failure"


def test_stats_separates_roam_edges_from_llm_edges(base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    s = stats(graph)
    assert s["edges_by_origin"]["roam-link"] > 0
    assert s["edges_by_origin"]["llm"] == 2
    assert s["nodes_without_provenance"] == 0
