"""The shapes every stage of the pipeline passes along.

These dataclasses are the contract between stages. `parse` emits them, `extract`
adds to them, `resolve` rewrites their ids, `assemble` persists them, `query`
reads them back. Nothing else is shared between stages.

Provenance is not optional anywhere. Every node and every edge carries the Roam
block uid it came from, so any answer can be traced to the block that produced
it. A fact without provenance is a fact this app cannot defend, and is dropped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA_VERSION = 1

# Where a fact came from. `roam-link` facts are read straight off the export's
# own [[links]]/#tags/((refs)) and are treated as ground truth; `llm` facts are
# inferred from block prose and are not.
ORIGIN_ROAM_LINK = "roam-link"
ORIGIN_LLM = "llm"
ORIGINS = (ORIGIN_ROAM_LINK, ORIGIN_LLM)


def canonical_id(kind: str, name: str) -> str:
    """A deterministic id for an entity.

    Deterministic so that re-ingesting the same export produces the same graph
    rather than a second copy of it. Case- and whitespace-insensitive, because
    Roam page titles drift in both.
    """
    normalized = " ".join(name.split()).casefold()
    digest = hashlib.sha1(f"{kind}\x00{normalized}".encode("utf-8")).hexdigest()
    return f"{kind}:{digest[:16]}"


@dataclass
class Provenance:
    """Where one fact came from, and how it got here."""

    block_uid: str
    page_title: str
    origin: str  # ORIGIN_ROAM_LINK | ORIGIN_LLM
    extracted_at: str  # ISO 8601 UTC
    model: str | None = None  # the OpenRouter model id, when origin == llm
    prompt_version: str | None = None  # which prompt produced it
    quote: str | None = None  # the span of block text supporting the fact

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin {self.origin!r}; expected one of {ORIGINS}")
        if not self.block_uid:
            raise ValueError("provenance requires a block_uid")


@dataclass
class Node:
    """An entity. Either a Roam page, or something an LLM found inside a block."""

    id: str
    type: str  # page | block | person | project | org | concept | ...
    name: str
    description: str = ""  # disambiguation context; what resolve() reasons over
    aliases: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)

    @classmethod
    def make(cls, type: str, name: str, **kw: Any) -> "Node":
        return cls(id=canonical_id(type, name), type=type, name=name, **kw)


@dataclass
class Edge:
    """A typed, directed, provenanced relation between two nodes.

    The graph is a multigraph: the same (source, predicate, target) may be
    asserted by several blocks. Those are not duplicates to be collapsed — each
    is independent evidence, and `assemble` keeps all of their provenance.
    """

    source_id: str
    predicate: str
    target_id: str
    provenance: Provenance
    confidence: float = 1.0  # roam-link edges are 1.0; llm edges are the model's

    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.predicate, self.target_id)


@dataclass
class Graph:
    """A whole graph in memory. Small enough to hold; see CLAUDE.md on scale."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> Node:
        """Merge by id. Same entity seen twice accumulates provenance/aliases."""
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        existing.provenance.extend(node.provenance)
        for alias in node.aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
        if not existing.description and node.description:
            existing.description = node.description
        return existing

    def add_edge(self, edge: Edge) -> None:
        """Reject dangling edges loudly.

        An edge to a node that was never added is a bug in the stage that
        emitted it, and silently dropping it would make the graph quietly
        lossy — the failure this pipeline is least able to detect later.
        """
        for end in (edge.source_id, edge.target_id):
            if end not in self.nodes:
                raise ValueError(f"edge references unknown node {end!r}")
        self.edges.append(edge)

    def neighbors(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if node_id in (e.source_id, e.target_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }


@dataclass
class Triple:
    """A candidate relation, before it is admitted to the graph.

    This is what `extract` returns and what `assemble` consumes. It names
    entities by *string*, not by id, because at extraction time nothing has been
    resolved yet — "Buzz" and "Buzz Aldrin" are still two different strings.
    """

    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    provenance: Provenance
    confidence: float = 1.0
    subject_description: str = ""
    object_description: str = ""
