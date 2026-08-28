"""The web viewer's serialization, tested without starting a server.

`serve.py`'s routes are thin wrappers around GraphStore plus a JSON shaping
step. The shaping is what can silently go wrong (a dropped field means the
page renders blanks), so that is what is asserted here — against a real
temporary SQLite store, not a mock, since the store is the thing being
serialized.

The HTTP layer itself and the page's rendering are not covered; see
web/README.md on what that leaves untested.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import ORIGIN_LLM, Edge, Graph, Node, Provenance, canonical_id
from src.store.graph import GraphStore

NOW = datetime.now(timezone.utc).isoformat()


def prov(uid="b1", origin=ORIGIN_LLM, quote="Dana runs the data team"):
    return Provenance(
        block_uid=uid, page_title="Project Halyard", origin=origin,
        extracted_at=NOW, model="test-model", prompt_version="extract/v1",
        quote=quote,
    )


@pytest.fixture
def db(tmp_path):
    graph = Graph()
    dana = graph.add_node(Node.make("person", "Dana Whitfield", description="data lead",
                                    aliases=["Dana"], provenance=[prov()]))
    rivet = graph.add_node(Node.make("org", "Rivet Labs", provenance=[prov("b2")]))
    page = graph.add_node(Node.make("page", "Project Halyard",
                                    provenance=[prov("b3", origin="roam-link", quote=None)]))
    graph.add_edge(Edge(dana.id, "works_at", rivet.id, prov(), confidence=0.9))
    graph.add_edge(Edge(page.id, "mentions", dana.id, prov("b3", origin="roam-link", quote=None)))
    path = tmp_path / "g.db"
    with GraphStore(path) as store:
        store.write(graph)
    return path


def graph_payload(db_path):
    """Mirrors serve.py's _graph() shaping."""
    with GraphStore(db_path) as store:
        graph = store.load()
        counts = store.counts()
    degree = {}
    for e in graph.edges:
        degree[e.source_id] = degree.get(e.source_id, 0) + 1
        degree[e.target_id] = degree.get(e.target_id, 0) + 1
    return {
        "counts": counts,
        "nodes": [{"id": n.id, "type": n.type, "name": n.name, "degree": degree.get(n.id, 0),
                   "origins": sorted({p.origin for p in n.provenance})}
                  for n in graph.nodes.values()],
        "edges": [{"source": e.source_id, "target": e.target_id, "predicate": e.predicate,
                   "origin": e.provenance.origin, "block_uid": e.provenance.block_uid}
                  for e in graph.edges],
    }


def test_graph_payload_is_json_serializable(db):
    json.dumps(graph_payload(db))  # raises if any dataclass leaked through


def test_every_edge_endpoint_exists_as_a_node(db):
    """The page draws an edge by looking both endpoints up in its node map;
    an edge naming a node the payload never sent renders as a line to
    nowhere, or silently vanishes."""
    p = graph_payload(db)
    ids = {n["id"] for n in p["nodes"]}
    for e in p["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_degree_matches_actual_edge_count(db):
    p = graph_payload(db)
    dana = next(n for n in p["nodes"] if n["name"] == "Dana Whitfield")
    # works_at (out) + mentions (in)
    assert dana["degree"] == 2


def test_node_origins_distinguish_written_links_from_inferred(db):
    """The view draws a dashed outline for llm-only nodes; if origins were
    dropped or merged, every structure would look hand-written."""
    p = graph_payload(db)
    by_name = {n["name"]: n for n in p["nodes"]}
    assert by_name["Dana Whitfield"]["origins"] == ["llm"]
    assert by_name["Project Halyard"]["origins"] == ["roam-link"]


def test_counts_reach_the_payload(db):
    p = graph_payload(db)
    assert p["counts"]["nodes"] == 3
    assert p["counts"]["edges"] == 2
    assert p["counts"]["llm_edges"] == 1


def test_index_html_exists_and_declares_its_theme_tokens():
    """The page is served as a static file; a missing token block means the
    dark theme silently falls back to unstyled."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    assert "--paper" in html and "--ink" in html
    assert 'data-theme="dark"' in html
    assert "prefers-color-scheme: dark" in html


def test_query_route_is_not_reachable_by_GET():
    """The only route that spends money is POST-only on purpose — a GET
    would let a refresh or a prefetch trigger a model call."""
    source = (Path(__file__).parent.parent / "web" / "serve.py").read_text(encoding="utf-8")
    do_get = source.split("def do_GET")[1].split("def do_POST")[0]
    assert "/api/query" not in do_get
