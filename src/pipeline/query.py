"""Stage 5 — a question -> a grounded, cited answer.

Retrieval is deterministic (seed match, then a bounded k-hop walk); only the
final reasoning step is a model, and it is given the subgraph and nothing else.

The whole point of this stage is the citation. An answer that cannot name the
edges supporting it is the model reciting what it already knew about the world,
which is precisely the failure a personal knowledge graph exists to avoid —
your notes are the authority here, not the model's training data. So citations
are verified against the serialized subgraph after the fact, and an answer that
cites an edge it wasn't shown is flagged rather than returned clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm import prompts
from ..llm.openrouter import complete, extract_json, model_for
from ..models import Graph
from ..store.graph import GraphStore


@dataclass
class Answer:
    question: str
    answer: str
    sufficient: bool
    citations: list[dict] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    triples_shown: int = 0
    model: str = ""


def serialize(graph: Graph) -> tuple[str, dict[str, dict]]:
    """Subgraph -> the text the model reads, plus an id->edge lookup.

    Every line carries its source block, so the model is looking at provenance
    while it reasons rather than being asked to recall it afterwards.
    """
    lines: list[str] = []
    index: dict[str, dict] = {}
    for i, edge in enumerate(graph.edges, 1):
        edge_id = f"e{i}"
        source = graph.nodes.get(edge.source_id)
        target = graph.nodes.get(edge.target_id)
        if source is None or target is None:
            continue
        prov = edge.provenance
        lines.append(
            f'[{edge_id}] {source.name} --{edge.predicate}--> {target.name}   '
            f'(source: page "{prov.page_title}", block {prov.block_uid}, {prov.origin})'
        )
        index[edge_id] = {
            "edge_id": edge_id,
            "source": source.name,
            "predicate": edge.predicate,
            "target": target.name,
            "page_title": prov.page_title,
            "block_uid": prov.block_uid,
            "origin": prov.origin,
            "quote": prov.quote,
        }
    return "\n".join(lines), index


def parse_query_response(
    raw_text: str, index: dict[str, dict], question: str, model: str
) -> Answer:
    """Model reply -> Answer, with citations checked. Pure; no network."""
    data = extract_json(raw_text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")

    citations, invalid = [], []
    for cite in data.get("citations") or []:
        key = str(cite).strip()
        if key in index:
            citations.append(index[key])
        else:
            invalid.append(key)

    return Answer(
        question=question,
        answer=str(data.get("answer", "")).strip(),
        sufficient=bool(data.get("sufficient", False)),
        citations=citations,
        invalid_citations=invalid,
        triples_shown=len(index),
        model=model,
    )


def find_seeds(store: GraphStore, question: str, limit: int = 6) -> list[str]:
    """Seed entities for the walk: graph node names that occur in the question.

    Longest-first so "Buzz Aldrin" wins over "Buzz". Deliberately literal — a
    smarter seeder is the right next improvement, and doing it with embeddings
    rather than another model call is the cheaper version of that.
    """
    graph = store.load()
    q = question.casefold()
    hits: list[tuple[int, str]] = []
    for node in graph.nodes.values():
        for name in [node.name, *node.aliases]:
            folded = name.casefold().strip()
            if len(folded) >= 3 and folded in q:
                hits.append((len(folded), node.id))
                break
    hits.sort(reverse=True)
    seen, out = set(), []
    for _, node_id in hits:
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out[:limit]


def ask(
    store: GraphStore,
    question: str,
    *,
    hops: int = 2,
    model: str | None = None,
    max_triples: int = 300,
) -> Answer:
    model = model or model_for("query")
    seeds = find_seeds(store, question)
    if not seeds:
        return Answer(
            question=question,
            answer=(
                "No entity in the question matches a node in the graph, so there is "
                "nothing to reason over. Try naming a page or person that exists in "
                "the ingested subtree."
            ),
            sufficient=False,
            model=model,
        )

    sub = store.subgraph(seeds, hops=hops)
    if len(sub.edges) > max_triples:
        # Truncating beats blowing the context window, but it is a real loss of
        # recall — say so rather than silently answering from a partial view.
        sub.edges = sorted(sub.edges, key=lambda e: -e.confidence)[:max_triples]

    triples, index = serialize(sub)
    # See openrouter.py's DEFAULT_MODELS comment: a reasoning-capable model with
    # too tight a budget spends it all on reasoning and returns nothing — hit
    # for real on `extract`. query is left able to reason (that is what keeps
    # multi-hop answers honest instead of pattern-matched), so the fix here is
    # a wider budget, not reasoning={"enabled": False}.
    completion = complete(
        prompts.QUERY_SYSTEM,
        prompts.QUERY_USER.format(question=question, triples=triples),
        model,
        max_tokens=3072,
    )
    answer = parse_query_response(completion.text, index, question, completion.model)
    answer.seeds = [sub.nodes[s].name for s in seeds if s in sub.nodes]
    return answer
