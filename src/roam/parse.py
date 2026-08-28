"""Roam JSON export -> the base graph. No LLM anywhere in this file.

A Roam export is already a graph. Pages have titles, blocks have `uid`s and
`string`s and nest under `children`, and the prose is full of `[[page links]]`,
`#tags`, `attribute::` pairs and `((block refs))`. All of that is explicit,
human-curated structure — running a language model over it to rediscover what
the format already states would be paying tokens to make ground truth fuzzier.

So this stage is deterministic and total, and it is also what gives the eval
harness its free baseline: the links a human actually wrote are the closest
thing to a gold standard this corpus has.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..models import (
    ORIGIN_ROAM_LINK,
    Edge,
    Graph,
    Node,
    Provenance,
    canonical_id,
)

# [[Some Page]] — non-greedy, and it must not swallow a nested close bracket.
LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
# #tag or #[[Tag With Spaces]]
TAG_RE = re.compile(r"#(?:\[\[([^\[\]]+)\]\]|([A-Za-z0-9_/-]+))")
# ((block-uid-ref))
BLOCKREF_RE = re.compile(r"\(\(([A-Za-z0-9_-]+)\)\)")
# attribute:: value  — Roam's own typed-relation syntax, and the single highest
# signal predicate source in a real graph.
ATTR_RE = re.compile(r"^\s*([^:\n]{1,60})::\s*(.*)$")

PAGE = "page"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_export(path: str | Path) -> list[dict[str, Any]]:
    """Read a Roam JSON export: a top-level list of page objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("expected a Roam export: a JSON list of pages")
    return data


def iter_blocks(
    page: dict[str, Any], _parent_uid: str | None = None
) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Walk a page's block tree depth-first, yielding (block, parent_uid)."""
    for child in page.get("children", []) or []:
        yield child, _parent_uid
        yield from iter_blocks(child, child.get("uid"))


def strip_markup(text: str) -> str:
    """Block text with Roam's link syntax reduced to plain words.

    What the LLM stage sees. The brackets are noise to a model being asked to
    read prose, and leaving them in invites it to echo the syntax back instead
    of reasoning about the sentence.
    """
    text = LINK_RE.sub(r"\1", text)
    text = TAG_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = BLOCKREF_RE.sub("", text)
    return " ".join(text.split())


def parse(export: list[dict[str, Any]], subtree: str | None = None) -> Graph:
    """Build the base graph from an export.

    `subtree` limits the walk to one page title — the MVP path. Ingesting a
    whole personal graph on the first run is how you spend a lot of money
    discovering the schema was wrong.
    """
    pages = export
    if subtree is not None:
        wanted = subtree.casefold().strip()
        pages = [p for p in export if str(p.get("title", "")).casefold().strip() == wanted]
        if not pages:
            titles = ", ".join(sorted(str(p.get("title", "")) for p in export)[:10])
            raise ValueError(f"no page titled {subtree!r}. First titles: {titles}")

    graph = Graph()
    now = _now()

    # Pages first: every link target must exist as a node before any edge to it
    # is added, including links pointing at pages outside the subtree.
    for page in pages:
        title = str(page.get("title", "")).strip()
        if not title:
            continue
        graph.add_node(
            Node.make(
                PAGE,
                title,
                description=f"Roam page: {title}",
                provenance=[
                    Provenance(
                        block_uid=str(page.get("uid", title)),
                        page_title=title,
                        origin=ORIGIN_ROAM_LINK,
                        extracted_at=now,
                    )
                ],
            )
        )

    for page in pages:
        title = str(page.get("title", "")).strip()
        if not title:
            continue
        page_id = canonical_id(PAGE, title)

        for block, _parent in iter_blocks(page):
            text = str(block.get("string", ""))
            uid = str(block.get("uid", ""))
            if not text.strip() or not uid:
                continue

            prov = lambda quote: Provenance(  # noqa: E731
                block_uid=uid,
                page_title=title,
                origin=ORIGIN_ROAM_LINK,
                extracted_at=now,
                quote=quote,
            )

            # An `attribute:: value` line is a relation the user typed by hand.
            # Its predicate is better than anything a model would infer, so it
            # is read literally rather than paraphrased.
            attr = ATTR_RE.match(text)
            predicate_default = "mentions"
            targets_from_attr: list[str] = []
            if attr:
                predicate_default = _slug(attr.group(1))
                targets_from_attr = [
                    *LINK_RE.findall(attr.group(2)),
                    *(m[0] or m[1] for m in TAG_RE.findall(attr.group(2))),
                ]

            linked = [*LINK_RE.findall(text), *(m[0] or m[1] for m in TAG_RE.findall(text))]
            for target_title in linked:
                target_title = target_title.strip()
                if not target_title or target_title == title:
                    continue
                target = graph.add_node(
                    Node.make(
                        PAGE,
                        target_title,
                        description=f"Roam page: {target_title}",
                        provenance=[prov(text)],
                    )
                )
                predicate = (
                    predicate_default if target_title in targets_from_attr else "mentions"
                )
                graph.add_edge(
                    Edge(
                        source_id=page_id,
                        predicate=predicate,
                        target_id=target.id,
                        provenance=prov(text),
                        confidence=1.0,
                    )
                )

    return graph


def _slug(raw: str) -> str:
    return "_".join(strip_markup(raw).lower().split()) or "attribute"


def blocks_for_extraction(
    export: list[dict[str, Any]], subtree: str | None = None, min_chars: int = 24
) -> list[dict[str, str]]:
    """The block texts worth paying a model to read.

    Short blocks and blocks that are nothing but links are filtered out: the
    first carry no extractable proposition, and the second are already fully
    represented by `parse()`. This filter is the main cost lever on the extract
    stage, and it is deliberately conservative — it drops what is provably
    redundant, not what merely looks unpromising.
    """
    pages = export
    if subtree is not None:
        wanted = subtree.casefold().strip()
        pages = [p for p in export if str(p.get("title", "")).casefold().strip() == wanted]

    out: list[dict[str, str]] = []
    for page in pages:
        title = str(page.get("title", "")).strip()
        for block, _parent in iter_blocks(page):
            uid = str(block.get("uid", ""))
            raw = str(block.get("string", ""))
            plain = strip_markup(raw)
            if not uid or len(plain) < min_chars:
                continue
            # Nothing but link markup: parse() already has everything here.
            if not LINK_RE.sub("", TAG_RE.sub("", raw)).strip(" -*#[]()"):
                continue
            out.append({"uid": uid, "page_title": title, "text": plain, "raw": raw})
    return out
