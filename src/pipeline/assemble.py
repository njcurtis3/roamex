"""Stage 4 — base graph + resolved triples -> one graph.

Deterministic. No model here. Everything this stage does is bookkeeping, and it
is the stage where provenance either survives or is quietly lost, which is why
it does no reasoning of its own.

Where an LLM triple names an entity that already exists as a Roam page, the two
converge on the same node id — that is the point. The implicit relation the
model found gets attached to the page the author already maintains, rather than
starting a parallel shadow graph beside it.
"""

from __future__ import annotations

from ..models import Edge, Graph, Node, Triple, canonical_id
from .resolve import ResolutionMap, normalize

PAGE = "page"


def assemble(
    base: Graph, triples: list[Triple], resolution: ResolutionMap | None = None
) -> Graph:
    """Fold LLM triples into the base graph, applying the resolution map."""
    graph = Graph(nodes=dict(base.nodes), edges=list(base.edges))

    # A page's title is the strongest alias evidence in the corpus: the author
    # wrote it deliberately. An extracted name that normalizes to an existing
    # page title is that page, not a new entity.
    page_by_norm = {
        normalize(n.name): n for n in graph.nodes.values() if n.type == PAGE
    }

    for triple in triples:
        source = _node_for(
            graph,
            page_by_norm,
            resolution,
            triple.subject,
            triple.subject_type,
            triple.subject_description,
            triple,
        )
        target = _node_for(
            graph,
            page_by_norm,
            resolution,
            triple.object,
            triple.object_type,
            triple.object_description,
            triple,
        )
        if source.id == target.id:
            # Resolution collapsed both ends into one entity; a self-edge is not
            # a fact about anything.
            continue
        graph.add_edge(
            Edge(
                source_id=source.id,
                predicate=triple.predicate,
                target_id=target.id,
                provenance=triple.provenance,
                confidence=triple.confidence,
            )
        )

    return graph


def _node_for(
    graph: Graph,
    page_by_norm: dict[str, Node],
    resolution: ResolutionMap | None,
    raw_name: str,
    entity_type: str,
    description: str,
    triple: Triple,
) -> Node:
    name = resolution.canonical(raw_name) if resolution else raw_name

    existing_page = page_by_norm.get(normalize(name))
    if existing_page is not None:
        existing_page.provenance.append(triple.provenance)
        if raw_name != existing_page.name and raw_name not in existing_page.aliases:
            existing_page.aliases.append(raw_name)
        return existing_page

    node = graph.add_node(
        Node(
            id=canonical_id(entity_type, name),
            type=entity_type,
            name=name,
            description=description,
            aliases=[raw_name] if raw_name != name else [],
            provenance=[triple.provenance],
        )
    )
    return node


def stats(graph: Graph) -> dict:
    """What assemble produced. The numbers the eval harness starts from."""
    from ..models import ORIGIN_LLM, ORIGIN_ROAM_LINK

    by_origin = {ORIGIN_ROAM_LINK: 0, ORIGIN_LLM: 0}
    predicates: dict[str, int] = {}
    for edge in graph.edges:
        by_origin[edge.provenance.origin] = by_origin.get(edge.provenance.origin, 0) + 1
        predicates[edge.predicate] = predicates.get(edge.predicate, 0) + 1

    types: dict[str, int] = {}
    unprovenanced = 0
    for node in graph.nodes.values():
        types[node.type] = types.get(node.type, 0) + 1
        if not node.provenance:
            unprovenanced += 1

    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "edges_by_origin": by_origin,
        "node_types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "top_predicates": dict(
            sorted(predicates.items(), key=lambda kv: -kv[1])[:15]
        ),
        "nodes_without_provenance": unprovenanced,
    }
