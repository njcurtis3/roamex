"""Local viewer for an assembled roamex graph.

    python web/serve.py                 # opens a browser at 127.0.0.1:8790
    python web/serve.py --db work/graph.db --port 8791 --no-open

Stdlib `http.server`, no dependencies, four routes:

    GET  /                  index.html
    GET  /api/graph         nodes + edges + counts, read from the db per request
    GET  /api/node/<id>     one node with its edges and full provenance
    POST /api/query         run pipeline.query.ask() and return the cited answer

Everything except `/api/query` is read-only and offline. `/api/query` is the
one route that costs money and touches the network, because `ask()` calls a
model — it is a POST for exactly that reason, so nothing triggers spend by
being loaded, prefetched, or refreshed.

WHY THIS IMPORTS FROM src/ AND web/ STAYS OUT OF src/
The dependency runs one way on purpose (CLAUDE.md § The frontend: "nothing in
src/ may depend on web/"). This is the viewer for the pipeline, so it reads
the pipeline's own store and query code rather than reimplementing either —
a second SQL layer here would be a second thing to keep correct. Delete web/
entirely and the pipeline still runs; delete src/ and this is meaningless.

BINDS 127.0.0.1 ON PURPOSE. Nothing here is authenticated and the payload is
the contents of a personal knowledge base. Do not bind it wider.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# web/ is not a package inside src/; make the project root importable so this
# runs as a plain script from anywhere in the repo.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.openrouter import load_dotenv  # noqa: E402
from src.pipeline import query as query_stage  # noqa: E402
from src.store.graph import GraphStore  # noqa: E402

INDEX = Path(__file__).parent / "index.html"

# Guardrail on the one expensive route. The viewer is meant to be poked at, and
# an accidental loop in the page (or a held-down key) would otherwise spend
# real money per keystroke.
MAX_QUESTION_CHARS = 500


class Handler(BaseHTTPRequestHandler):
    db_path = "work/graph.db"

    def log_message(self, fmt, *args):
        # http.server's default logs every asset fetch to stderr; keep the
        # console readable so a real error is visible when it happens.
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write(f"  {fmt % args}\n")

    # -- helpers ----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This serves personal notes; make sure a shared/proxy cache never
        # holds them, and that a stale graph is never shown after a re-run.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _store(self) -> GraphStore | None:
        if not Path(self.db_path).exists():
            self._error(404, f"{self.db_path} not found — run `python -m src.cli assemble` first.")
            return None
        return GraphStore(self.db_path)

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            if not INDEX.exists():
                return self._error(500, "index.html is missing")
            return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/graph":
            return self._graph()
        if path.startswith("/api/node/"):
            return self._node(path[len("/api/node/"):])
        self._error(404, "no such route")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/query":
            return self._error(404, "no such route")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._error(400, "body must be JSON")

        question = str(payload.get("question", "")).strip()
        if not question:
            return self._error(400, "a question is required")
        if len(question) > MAX_QUESTION_CHARS:
            return self._error(400, f"question is longer than {MAX_QUESTION_CHARS} characters")
        try:
            hops = max(1, min(4, int(payload.get("hops", 2))))
        except (TypeError, ValueError):
            hops = 2

        store = self._store()
        if store is None:
            return
        try:
            answer = query_stage.ask(store, question, hops=hops)
        except Exception as exc:
            # ask() already fails closed for parse errors; this catches the
            # rest (no API key, network down) so the page shows a message
            # rather than the browser showing a dead request.
            return self._error(502, f"query failed: {exc}")
        finally:
            store.close()

        self._json({
            "question": answer.question,
            "answer": answer.answer,
            "sufficient": answer.sufficient,
            "seeds": answer.seeds,
            "triples_shown": answer.triples_shown,
            "model": answer.model,
            "citations": answer.citations,
            "invalid_citations": answer.invalid_citations,
        })

    def _graph(self) -> None:
        store = self._store()
        if store is None:
            return
        try:
            graph = store.load()
            counts = store.counts()
        finally:
            store.close()

        # Degree drives node size in the view, so compute it once here rather
        # than making the page walk every edge per node.
        degree: dict[str, int] = {}
        for e in graph.edges:
            degree[e.source_id] = degree.get(e.source_id, 0) + 1
            degree[e.target_id] = degree.get(e.target_id, 0) + 1

        self._json({
            "counts": counts,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "name": n.name,
                    "description": n.description,
                    "aliases": n.aliases,
                    "degree": degree.get(n.id, 0),
                    "origins": sorted({p.origin for p in n.provenance}),
                }
                for n in graph.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "predicate": e.predicate,
                    "origin": e.provenance.origin,
                    "confidence": e.confidence,
                    "page_title": e.provenance.page_title,
                    "block_uid": e.provenance.block_uid,
                }
                for e in graph.edges
            ],
        })

    def _node(self, node_id: str) -> None:
        from urllib.parse import unquote

        node_id = unquote(node_id)
        store = self._store()
        if store is None:
            return
        try:
            graph = store.load()
        finally:
            store.close()

        node = graph.nodes.get(node_id)
        if node is None:
            return self._error(404, "no such node")

        edges = []
        for e in graph.edges:
            if e.source_id != node_id and e.target_id != node_id:
                continue
            other_id = e.target_id if e.source_id == node_id else e.source_id
            other = graph.nodes.get(other_id)
            edges.append({
                "direction": "out" if e.source_id == node_id else "in",
                "predicate": e.predicate,
                "other_id": other_id,
                "other_name": other.name if other else other_id,
                "other_type": other.type if other else "unknown",
                "origin": e.provenance.origin,
                "confidence": e.confidence,
                "page_title": e.provenance.page_title,
                "block_uid": e.provenance.block_uid,
                "quote": e.provenance.quote,
            })

        self._json({
            "id": node.id,
            "type": node.type,
            "name": node.name,
            "description": node.description,
            "aliases": node.aliases,
            "edges": edges,
            "provenance": [
                {
                    "block_uid": p.block_uid,
                    "page_title": p.page_title,
                    "origin": p.origin,
                    "model": p.model,
                    "prompt_version": p.prompt_version,
                    "quote": p.quote,
                }
                for p in node.provenance
            ],
        })


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="roamex-web", description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="work/graph.db")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1", help="do not bind this wider; see module docstring")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    if not Path(args.db).exists():
        sys.exit(
            f"{args.db} not found. Build a graph first:\n"
            f"  python -m src.cli pull\n"
            f"  python -m src.cli parse --export exports/roam.json --subtree \"Some Page\"\n"
            f"  python -m src.cli extract --export exports/roam.json --subtree \"Some Page\"\n"
            f"  python -m src.cli resolve && python -m src.cli assemble"
        )

    Handler.db_path = args.db
    server = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"roamex viewer: {url}   (db: {args.db})")
    print("Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
