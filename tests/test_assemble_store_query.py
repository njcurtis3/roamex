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


def test_provenance_coverage_is_total(base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    report = score.score_provenance(graph)
    assert report["node_coverage"] == 1.0
    assert report["edge_coverage"] == 1.0


def test_stats_separates_roam_edges_from_llm_edges(base, triples):
    graph = assemble(base, triples, ResolutionMap(mapping={"dana": "Dana Whitfield"}))
    s = stats(graph)
    assert s["edges_by_origin"]["roam-link"] > 0
    assert s["edges_by_origin"]["llm"] == 2
    assert s["nodes_without_provenance"] == 0
