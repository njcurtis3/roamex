"""The base graph is deterministic, so it can be asserted exactly."""

import json
from pathlib import Path

import pytest

from src.models import ORIGIN_ROAM_LINK, canonical_id
from src.roam import parse as rp

FIXTURE = Path(__file__).parent.parent / "fixtures" / "roam" / "sample_export.json"


@pytest.fixture
def export():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_strip_markup_removes_roam_syntax():
    assert rp.strip_markup("Kicked off with [[Dana Whitfield]] and #[[Rivet Labs]]") == (
        "Kicked off with Dana Whitfield and Rivet Labs"
    )
    assert rp.strip_markup("see ((blk-001)) here") == "see here"
    assert rp.strip_markup("a #tag and [[a page]]") == "a tag and a page"


def test_iter_blocks_reaches_nested_children(export):
    halyard = export[0]
    uids = {b["uid"] for b, _ in rp.iter_blocks(halyard)}
    assert "blk-003" in uids, "nested block was not walked"
    assert "blk-004" in uids


def test_parse_creates_page_nodes_and_link_edges(export):
    graph = rp.parse(export)
    dana = canonical_id("page", "Dana Whitfield")
    halyard = canonical_id("page", "Project Halyard")
    assert dana in graph.nodes
    assert halyard in graph.nodes
    assert any(
        e.source_id == halyard and e.target_id == dana for e in graph.edges
    ), "no edge from Project Halyard to Dana Whitfield"


def test_all_base_edges_carry_roam_link_provenance(export):
    graph = rp.parse(export)
    assert graph.edges
    for edge in graph.edges:
        assert edge.provenance.origin == ORIGIN_ROAM_LINK
        assert edge.provenance.block_uid


def test_attribute_syntax_becomes_the_predicate(export):
    """`status:: [[Active]]` is a relation the author typed. Read it, don't paraphrase."""
    graph = rp.parse(export)
    active = canonical_id("page", "Active")
    predicates = {e.predicate for e in graph.edges if e.target_id == active}
    assert "status" in predicates, f"expected a `status` predicate, got {predicates}"


def test_subtree_limits_the_walk_but_keeps_link_targets(export):
    graph = rp.parse(export, subtree="Project Halyard")
    # Only Halyard's own blocks are walked...
    pages = {n.name for n in graph.nodes.values()}
    assert "Hollis Conference" not in pages, "walked a page outside the subtree"
    # ...but pages it links to must still exist, or its edges would dangle.
    assert "Dana Whitfield" in pages
    assert "Rivet Labs" in pages


def test_subtree_miss_raises_rather_than_returning_empty(export):
    """An empty graph and a typo'd page name must not look the same."""
    with pytest.raises(ValueError, match="no page titled"):
        rp.parse(export, subtree="Page That Does Not Exist")


def test_blocks_for_extraction_skips_short_and_link_only_blocks(export):
    blocks = rp.blocks_for_extraction(export, subtree="Project Halyard")
    uids = {b["uid"] for b in blocks}
    assert "blk-003" in uids, "a prose block was filtered out"
    assert "blk-006" not in uids, "'ok' is too short to extract from"
    assert "blk-007" not in uids, "link-only block is already covered by parse()"


def test_extraction_text_is_stripped_of_markup(export):
    blocks = rp.blocks_for_extraction(export, subtree="Project Halyard")
    by_uid = {b["uid"]: b for b in blocks}
    assert "[[" not in by_uid["blk-002"]["text"]
