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
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import ORIGIN_LLM, Edge, Graph, Node, Provenance, canonical_id
from src.store.graph import GraphStore

NOW = datetime.now(timezone.utc).isoformat()


def _strip_line_comments(js: str) -> str:
    """Drop `//` comments so a source assertion checks code, not prose about
    the code — the comment explaining this very bug mentions the API name."""
    return "\n".join(line.split("//")[0] for line in js.splitlines())


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


def test_pointer_is_not_captured_on_pointerdown():
    """Regression, reported 2026-08-28: clicking a structure did nothing.

    setPointerCapture() retargets the compatibility `click` event to the
    capturing element, so capturing on pointerdown sent every click to the
    <svg> (which deselects) instead of the structure's own handler. Capture
    must only be taken once a drag actually starts, inside pointermove.
    """
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    down = _strip_line_comments(
        html.split('addEventListener("pointerdown"')[1].split('addEventListener("pointermove"')[0]
    )
    assert "setPointerCapture" not in down, (
        "pointerdown captures the pointer again — this makes every structure unclickable"
    )
    move = _strip_line_comments(
        html.split('addEventListener("pointermove"')[1].split('addEventListener("pointerup"')[0]
    )
    assert "setPointerCapture" in move, "a drag outside the svg needs capture once it starts"


def test_both_click_handlers_consume_the_drag_suppression_flag():
    """The structure handler calls stopPropagation, so the svg handler can't
    be relied on to clear the flag — a drag ending on a structure would
    otherwise leave it set and swallow the *next* genuine click."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    assert html.count("consumeSuppressedClick()") >= 2


DAILY_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October"
    r"|November|December) \d{1,2}(st|nd|rd|th), \d{4}$"
)


@pytest.mark.parametrize("title", [
    "June 25th, 2021", "March 7th, 2023", "August 1st, 2026", "April 2nd, 2022",
    "October 3rd, 2021",
])
def test_daily_note_titles_are_recognised(title):
    assert DAILY_RE.match(title)


@pytest.mark.parametrize("title", [
    # All real titles from a live graph. A looser "contains a month name"
    # test would fold these into the Daily Notes group and effectively hide
    # them, which is worse than not grouping at all.
    "Dad's letter on January 7th 2022",   # no comma before the year
    "Dad's letter on October 22 2021",    # no ordinal suffix
    "Day 1: June 19",                     # month mid-title
    "Notes from June 25th, 2021",         # daily-note date, but prefixed
    "June 25th, 2021 retrospective",      # daily-note date, but suffixed
])
def test_pages_that_merely_mention_a_date_are_not_daily_notes(title):
    assert not DAILY_RE.match(title)


def test_index_regex_in_the_page_matches_this_one():
    """The page builds its regex from a MONTHS array; if that drifts from the
    pattern asserted above, these tests stop describing the real behaviour."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    assert r"\\d{1,2}(st|nd|rd|th), \\d{4}$" in html
    for month in ["January", "December"]:
        assert f'"{month}"' in html


def test_daily_notes_sort_by_date_not_alphabetically():
    """Alphabetical order on daily notes puts April before January, which is
    nonsense for a date-titled folder."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    js = html.split("function dailyKey")[1].split("const INDEX_CAP")[0]
    # year dominates, then month, then day
    assert "10000" in js and "100" in js


def _viewer_codes() -> dict:
    """Parse the CODE map out of index.html."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    block = html.split("const CODE = {")[1].split("};")[0]
    return dict(re.findall(r"(\w+):\s*\"(\w+)\"", block))


def test_every_extractor_type_has_a_code_in_the_viewer():
    """`ENTITY_TYPES` in prompts.py is the vocabulary the model chooses from.
    A type there with no code renders as `??` in the index with no legend
    entry — silently, since nothing else notices."""
    from src.llm.prompts import ENTITY_TYPES

    codes = _viewer_codes()
    missing = [t for t in ENTITY_TYPES if t not in codes]
    assert not missing, f"no index code for extractor types: {missing}"


def test_codes_are_unique():
    """Two types sharing a code makes the index ambiguous."""
    codes = _viewer_codes()
    assert len(set(codes.values())) == len(codes)


def test_every_code_has_a_legend_note():
    """The legend and the hover tooltip both read TYPE_NOTE; a code without
    one shows an empty tooltip."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    notes = html.split("const TYPE_NOTE = {")[1].split("};")[0]
    for t in _viewer_codes():
        assert f"{t}:" in notes, f"{t} has a code but no legend note"


def test_concept_note_admits_it_is_the_fallback_bucket():
    """extract.py's _valid_type() falls back to `concept` for anything
    outside the vocabulary, so that count mixes real concepts with
    extraction misses. The legend should say so rather than let the number
    read as pure signal."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    notes = html.split("const TYPE_NOTE = {")[1].split("};")[0]
    note = notes.split("concept:")[1].split("\n")[0]
    assert "fallback" in note.lower()


def test_query_route_is_not_reachable_by_GET():
    """The only route that spends money is POST-only on purpose — a GET
    would let a refresh or a prefetch trigger a model call."""
    source = (Path(__file__).parent.parent / "web" / "serve.py").read_text(encoding="utf-8")
    do_get = source.split("def do_GET")[1].split("def do_POST")[0]
    assert "/api/query" not in do_get


def _path_finder_js() -> str:
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    return html.split("/* ---------- path finder")[1].split("/* ---------- tabs")[0]


def test_path_finder_makes_no_network_call():
    """The pitch is 'free, no model call' — findPath and its wiring must stay
    pure client-side traversal over data already fetched by /api/graph. A
    fetch() sneaking in here would silently turn a free feature into a paid
    one, or one that needs a route that doesn't exist."""
    js = _strip_line_comments(_path_finder_js())
    assert "fetch(" not in js


def test_path_finder_route_and_tab_exist():
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="path"' in html
    assert 'id="panePath"' in html and 'id="pathForm"' in html


def test_pane_wraps_long_unbroken_text():
    """Regression, found 2026-08-30: a quote containing a markdown link's
    unbroken URL overflowed the fixed-width right panel and was silently
    clipped by #body's overflow:hidden instead of wrapping. overflow-wrap is
    an inherited CSS property, so setting it once on .pane (the shared
    ancestor of every quote/answer/source line) fixes all of them."""
    html = (Path(__file__).parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    pane_rule = html.split(".pane {", 1)[1].split("}", 1)[0]
    assert "overflow-wrap" in pane_rule


def test_path_edge_key_includes_block_uid():
    """Two distinct edges can share source, target, and predicate (the same
    relation stated in two different blocks). Without block_uid in the key,
    highlighting one would highlight both, and clearing one wouldn't clear
    the other's highlight."""
    js = _path_finder_js()
    fn = js.split("function edgeKey")[1].split("function buildAdjacency")[0]
    assert "block_uid" in fn
