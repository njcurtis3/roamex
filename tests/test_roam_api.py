"""Tests the one part of roam/api.py that can be tested without a live Roam
graph: converting an already-fetched pull result into the page/children
shape roam.parse expects. Everything network-shaped in that module
(RoamClient, fetch_graph) is untested here on purpose — see api.py's
module docstring on why the wire format itself is unverified.
"""

import json
from pathlib import Path

import pytest

from src.roam.api import RoamAPIError, _convert_blocks, convert_pulled_pages
from src.roam.parse import iter_blocks, parse

FIXTURE = Path(__file__).parent.parent / "fixtures" / "roam" / "api_pull_response.json"


@pytest.fixture
def raw_result():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["result"]


def test_converts_to_the_shape_load_export_returns(raw_result):
    pages = convert_pulled_pages(raw_result)
    assert pages[0]["title"] == "Project Halyard"
    assert pages[0]["uid"] == "page-halyard"
    assert isinstance(pages[0]["children"], list)
    assert pages[1]["title"] == "Dana Whitfield"
    assert pages[1]["children"] == []


def test_blocks_carry_uid_string_and_nested_children(raw_result):
    pages = convert_pulled_pages(raw_result)
    top = pages[0]["children"]
    assert {b["uid"] for b in top} == {"blk-001", "blk-002"}
    kicked_off = next(b for b in top if b["uid"] == "blk-002")
    assert kicked_off["string"] == "Kicked off with [[Dana Whitfield]] in March."
    assert kicked_off["children"][0]["uid"] == "blk-003"


def test_children_are_sorted_by_block_order_not_response_order(raw_result):
    """The fixture deliberately lists blk-002 (order 1) before blk-001
    (order 0) — Datalog gives no ordering guarantee, so this must not
    silently trust response order."""
    pages = convert_pulled_pages(raw_result)
    top = pages[0]["children"]
    assert [b["uid"] for b in top] == ["blk-001", "blk-002"]


def test_output_is_directly_consumable_by_roam_parse(raw_result):
    """The whole point of matching load_export()'s shape: parse() must run
    against it with zero changes."""
    pages = convert_pulled_pages(raw_result)
    graph = parse(pages)
    assert any(n.name == "Project Halyard" for n in graph.nodes.values())
    assert any(n.name == "Dana Whitfield" for n in graph.nodes.values())


def test_entity_with_no_title_is_not_a_page():
    raw = [[{"block/uid": "x", "block/string": "not a page"}]]
    assert convert_pulled_pages(raw) == []


def test_alternate_key_spellings_are_tried_before_giving_up():
    """The one thing this module could not verify offline: whether Datomic
    pull-JSON keeps the leading colon. Both spellings must work."""
    raw = [[{":node/title": "Colon Page", ":block/uid": "u1", ":block/children": []}]]
    pages = convert_pulled_pages(raw)
    assert pages[0]["title"] == "Colon Page"


def test_missing_uid_on_a_block_drops_it_rather_than_crashing():
    blocks = [{"block/string": "no uid here", "block/order": 0}]
    assert _convert_blocks(blocks) == []


def test_block_with_no_string_still_gets_uid_and_empty_string():
    """Distinguishes 'field absent' from 'field is empty' -- silently
    dropping a real block because block/string happened to be "" would be
    the kind of quiet data loss this whole app exists to avoid."""
    blocks = [{"block/uid": "u1", "block/order": 0, "block/children": []}]
    out = _convert_blocks(blocks)
    assert out == [{"uid": "u1", "string": "", "children": []}]
