"""Measure the two stages that can silently be wrong.

Four numbers, kept separate on purpose. A single blended "quality" score hides
which stage regressed, and these stages fail in unrelated ways:

- **extraction** — precision and recall against a gold triple set.
- **resolution** — merge recall, and the over-merge rate, which is the one that
  quietly corrupts the graph.
- **provenance** — the share of nodes and edges that can name their source. This
  should be 1.0. Anything less is a bug, not a tuning knob.
- **grounding** — do the cited edges exist, and did the model admit when the
  graph was insufficient.

Roam gives extraction a free baseline the paper's toy corpus did not have: the
author's own [[links]] are human-curated ground truth. `link_baseline` scores
LLM edges against them — not a full gold set, but it catches a broken prompt
before anyone hand-labels anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ..models import ORIGIN_LLM, ORIGIN_ROAM_LINK, Graph, Triple
from ..pipeline.resolve import normalize


def _key(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    return (normalize(subject), predicate.strip().lower(), normalize(obj))


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @classmethod
    def compute(cls, predicted: set, gold: set) -> "PRF":
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return cls(round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn)


def score_extraction(triples: list[Triple], gold: list[dict], ignore_predicate: bool = False):
    """Extraction P/R/F1 against a gold set.

    `ignore_predicate` scores whether the right entities were connected at all,
    separately from whether the relation was named the same way. Predicate
    wording is where free-form extraction disagrees with a human most often and
    matters least, so seeing both numbers tells you which problem you have.
    """
    def strip(k):
        return (k[0], k[2]) if ignore_predicate else k

    predicted = {strip(_key(t.subject, t.predicate, t.object)) for t in triples}
    gold_set = {
        strip(_key(g["subject"], g.get("predicate", ""), g["object"])) for g in gold
    }
    return PRF.compute(predicted, gold_set)


def score_resolution(groups: list[dict], gold_groups: list[list[str]]) -> dict:
    """Did resolution merge what should merge, without merging what shouldn't?

    Scored over *pairs*, not groups: it is the pairwise decisions that are
    right or wrong, and a group-level score would mark a cluster that got one
    member wrong as entirely wrong.
    """
    def pairs(members: list[str]) -> set[tuple[str, str]]:
        norm = sorted({normalize(m) for m in members})
        return {(a, b) for i, a in enumerate(norm) for b in norm[i + 1 :]}

    predicted: set[tuple[str, str]] = set()
    for g in groups:
        predicted |= pairs(g["members"])
    gold: set[tuple[str, str]] = set()
    for g in gold_groups:
        gold |= pairs(g)

    prf = PRF.compute(predicted, gold)
    singletons = sum(1 for g in groups if len(g["members"]) == 1)
    return {
        **asdict(prf),
        # The number to watch. Every false positive here is two entities' facts
        # welded together, and there is no signal afterwards saying they were
        # ever separate.
        "over_merge_rate": round(prf.false_positives / len(predicted), 4) if predicted else 0.0,
        "missed_merge_rate": round(prf.false_negatives / len(gold), 4) if gold else 0.0,
        "singleton_groups": singletons,
        "total_groups": len(groups),
    }


def score_provenance(graph: Graph) -> dict:
    """Every fact should name its source. This should read 1.0."""
    nodes_ok = sum(1 for n in graph.nodes.values() if n.provenance)
    edges_ok = sum(1 for e in graph.edges if e.provenance and e.provenance.block_uid)
    quoted = sum(
        1 for e in graph.edges if e.provenance.origin == ORIGIN_LLM and e.provenance.quote
    )
    llm_edges = sum(1 for e in graph.edges if e.provenance.origin == ORIGIN_LLM)
    return {
        "node_coverage": round(nodes_ok / len(graph.nodes), 4) if graph.nodes else 1.0,
        "edge_coverage": round(edges_ok / len(graph.edges), 4) if graph.edges else 1.0,
        "llm_edge_quote_coverage": round(quoted / llm_edges, 4) if llm_edges else 1.0,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }


def link_baseline(graph: Graph) -> dict:
    """LLM edges vs. the author's own [[links]] — the free sanity check.

    `agreement` is how often the model connected a pair the author had already
    linked. It is a floor, not a target: the whole reason this pipeline exists
    is to find relations the author never linked, so those show up here as
    "novel" rather than as errors. Agreement near zero means the prompt is
    broken; agreement near one means the stage is redundant.
    """
    roam_pairs = {
        (e.source_id, e.target_id)
        for e in graph.edges
        if e.provenance.origin == ORIGIN_ROAM_LINK
    }
    roam_pairs |= {(b, a) for a, b in roam_pairs}
    llm_pairs = {
        (e.source_id, e.target_id)
        for e in graph.edges
        if e.provenance.origin == ORIGIN_LLM
    }
    if not llm_pairs:
        return {"llm_pairs": 0, "agreement": 0.0, "novel": 0, "note": "no LLM edges"}
    overlap = len(llm_pairs & roam_pairs)
    return {
        "llm_pairs": len(llm_pairs),
        "roam_pairs": len(roam_pairs) // 2,
        "agreement": round(overlap / len(llm_pairs), 4),
        "novel": len(llm_pairs) - overlap,
    }


def score_grounding(answers: list) -> dict:
    """Did answers cite real edges, and admit when the graph fell short?"""
    if not answers:
        return {"answers": 0}
    cited = sum(1 for a in answers if a.citations)
    hallucinated = sum(1 for a in answers if a.invalid_citations)
    return {
        "answers": len(answers),
        "with_citations": cited,
        "citation_rate": round(cited / len(answers), 4),
        # A fabricated edge id is the loudest possible signal that the model
        # answered from its own knowledge instead of the subgraph.
        "hallucinated_citation_rate": round(hallucinated / len(answers), 4),
        "declared_insufficient": sum(1 for a in answers if not a.sufficient),
    }


def load_gold(path: str | Path) -> dict:
    """Gold file: {"extraction": [...], "resolution": [[...]], "questions": [...]}"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
