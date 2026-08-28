"""SQLite persistence for the assembled graph.

`GraphStore` is the only thing in this app that knows SQL. Stages hand it
`models.Graph` and get `models.Graph` back; nothing upstream of here sees a
cursor. That seam is what lets the store be swapped later without touching a
pipeline stage.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..models import SCHEMA_VERSION, Edge, Graph, Node, Provenance

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class GraphStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- write ------------------------------------------------------------

    def write(self, graph: Graph) -> None:
        """Persist a whole graph. Upserts nodes, appends edges and provenance.

        Not idempotent for edges by design: re-running extraction over the same
        blocks is a second, independent assertion of the same relation, and the
        eval harness needs to see that it happened twice. Wipe the db to start
        clean.
        """
        cur = self.conn.cursor()
        for node in graph.nodes.values():
            cur.execute(
                """INSERT INTO nodes (id, type, name, description, aliases)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     description = CASE WHEN nodes.description = ''
                                        THEN excluded.description
                                        ELSE nodes.description END,
                     aliases = excluded.aliases""",
                (node.id, node.type, node.name, node.description, json.dumps(node.aliases)),
            )
            for prov in node.provenance:
                self._write_prov(cur, "node", node.id, prov)

        for edge in graph.edges:
            cur.execute(
                "INSERT INTO edges (source_id, predicate, target_id, confidence) VALUES (?, ?, ?, ?)",
                (edge.source_id, edge.predicate, edge.target_id, edge.confidence),
            )
            self._write_prov(cur, "edge", str(cur.lastrowid), edge.provenance)

        self.conn.commit()

    @staticmethod
    def _write_prov(cur: sqlite3.Cursor, kind: str, ref: str, prov: Provenance) -> None:
        cur.execute(
            """INSERT INTO provenance
               (subject_kind, subject_ref, block_uid, page_title, origin,
                extracted_at, model, prompt_version, quote)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kind,
                ref,
                prov.block_uid,
                prov.page_title,
                prov.origin,
                prov.extracted_at,
                prov.model,
                prov.prompt_version,
                prov.quote,
            ),
        )

    # -- read -------------------------------------------------------------

    def load(self) -> Graph:
        """Read the whole graph back into memory."""
        graph = Graph()
        prov_by_node: dict[str, list[Provenance]] = {}
        for row in self.conn.execute(
            "SELECT * FROM provenance WHERE subject_kind = 'node'"
        ):
            prov_by_node.setdefault(row["subject_ref"], []).append(self._row_to_prov(row))

        for row in self.conn.execute("SELECT * FROM nodes"):
            graph.nodes[row["id"]] = Node(
                id=row["id"],
                type=row["type"],
                name=row["name"],
                description=row["description"],
                aliases=json.loads(row["aliases"]),
                provenance=prov_by_node.get(row["id"], []),
            )

        prov_by_edge: dict[str, Provenance] = {}
        for row in self.conn.execute(
            "SELECT * FROM provenance WHERE subject_kind = 'edge'"
        ):
            prov_by_edge[row["subject_ref"]] = self._row_to_prov(row)

        for row in self.conn.execute("SELECT * FROM edges"):
            prov = prov_by_edge.get(str(row["id"]))
            if prov is None:
                # An edge with no provenance is exactly the failure mode this
                # app exists to prevent. Refuse to hand it back as a fact.
                continue
            graph.edges.append(
                Edge(
                    source_id=row["source_id"],
                    predicate=row["predicate"],
                    target_id=row["target_id"],
                    provenance=prov,
                    confidence=row["confidence"],
                )
            )
        return graph

    @staticmethod
    def _row_to_prov(row: sqlite3.Row) -> Provenance:
        return Provenance(
            block_uid=row["block_uid"],
            page_title=row["page_title"],
            origin=row["origin"],
            extracted_at=row["extracted_at"],
            model=row["model"],
            prompt_version=row["prompt_version"],
            quote=row["quote"],
        )

    def find_nodes(self, term: str, limit: int = 10) -> list[Node]:
        """Seed lookup for `query`: match a name or a recorded alias."""
        graph = self.load()
        needle = term.casefold().strip()
        exact, partial = [], []
        for node in graph.nodes.values():
            names = [node.name] + node.aliases
            folded = [n.casefold() for n in names]
            if needle in folded:
                exact.append(node)
            elif any(needle in n for n in folded):
                partial.append(node)
        return (exact + partial)[:limit]

    def subgraph(self, seed_ids: list[str], hops: int = 2) -> Graph:
        """The k-hop neighborhood around the seeds — what the model reasons over.

        Bounded on purpose. Serializing an unbounded neighborhood is how a
        grounded prompt quietly becomes a whole-graph dump the model skims.
        """
        graph = self.load()
        frontier = {i for i in seed_ids if i in graph.nodes}
        seen = set(frontier)
        kept: list[Edge] = []
        for _ in range(max(0, hops)):
            next_frontier: set[str] = set()
            for edge in graph.edges:
                if edge.source_id in frontier or edge.target_id in frontier:
                    kept.append(edge)
                    for end in (edge.source_id, edge.target_id):
                        if end not in seen:
                            seen.add(end)
                            next_frontier.add(end)
            if not next_frontier:
                break
            frontier = next_frontier

        out = Graph()
        for node_id in seen:
            out.nodes[node_id] = graph.nodes[node_id]
        deduped = {id(e): e for e in kept}
        out.edges = list(deduped.values())
        return out

    def counts(self) -> dict[str, int]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "nodes": q("SELECT COUNT(*) FROM nodes"),
            "edges": q("SELECT COUNT(*) FROM edges"),
            "provenance": q("SELECT COUNT(*) FROM provenance"),
            "llm_edges": q(
                "SELECT COUNT(*) FROM provenance WHERE subject_kind='edge' AND origin='llm'"
            ),
        }
