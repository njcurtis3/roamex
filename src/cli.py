"""roamex CLI — the entry point for every stage.

Each stage writes its output to disk under `work/` and the next stage reads it
back. That is deliberate: extraction costs real money, and a pipeline that
re-runs it because the query prompt changed is a pipeline you stop using.
Stages are resumable and independently inspectable.

    python -m src.cli pull       [--out exports/roam.json] [--depth N]
    python -m src.cli pages      --export roam.json
    python -m src.cli parse      --export roam.json --subtree "Some Page"
    python -m src.cli extract    --export roam.json --subtree "Some Page" [--limit N] [--dry-run]
    python -m src.cli resolve
    python -m src.cli assemble
    python -m src.cli query      "who works at Acme?"
    python -m src.cli eval       [--gold eval/gold/<name>.json]
    python -m src.cli stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .eval import score
from .llm import catalog, prompts
from .llm.openrouter import load_dotenv, model_for
from .models import Provenance, Triple
from .pipeline import assemble as assemble_stage
from .pipeline import extract as extract_stage
from .pipeline import query as query_stage
from .pipeline import resolve as resolve_stage
from .roam import api as roam_api
from .roam import parse as roam_parse
from .store.graph import GraphStore

WORK = Path("work")
DB = WORK / "graph.db"
TRIPLES = WORK / "triples.json"
RESOLUTION = WORK / "resolution.json"
BASE = WORK / "base_graph.json"


def _work() -> None:
    WORK.mkdir(exist_ok=True)


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


def _load_triples() -> list[Triple]:
    if not TRIPLES.exists():
        sys.exit(f"{TRIPLES} not found — run `extract` first.")
    return [
        Triple(
            **{**t, "provenance": Provenance(**t["provenance"])}
        )
        for t in json.loads(TRIPLES.read_text(encoding="utf-8"))
    ]


# -- commands -------------------------------------------------------------


def cmd_pull(args) -> None:
    """Fetch the graph from Roam's live API instead of a manual export.

    See roam/api.py's module docstring before trusting this: the endpoint
    and headers are sourced from a community client, and the exact JSON key
    format of a pulled entity was never verified against a live graph. This
    command IS that verification — if it fails, the error should say
    exactly which assumption broke rather than fail silently.
    """
    token = os.environ.get("ROAM_API_TOKEN")
    graph = os.environ.get("ROAM_GRAPH_NAME")
    if not token or not graph:
        sys.exit(
            "ROAM_API_TOKEN and ROAM_GRAPH_NAME must be set in .env. "
            "See .env.example."
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"pulling graph {graph!r} (depth={args.depth})...")
    try:
        count = roam_api.pull_and_save(graph, token, str(out), depth=args.depth)
    except roam_api.RoamAPIError as exc:
        sys.exit(f"pull failed: {exc}")
    print(f"  wrote {count} pages to {out}")


def cmd_pages(args) -> None:
    """List page titles and block counts. How you pick an MVP subtree."""
    export = roam_parse.load_export(args.export)
    rows = []
    for page in export:
        title = str(page.get("title", "")).strip()
        if not title:
            continue
        blocks = list(roam_parse.iter_blocks(page))
        rows.append((len(blocks), title))
    rows.sort(reverse=True)
    print(f"{len(rows)} pages\n")
    for count, title in rows[: args.limit]:
        print(f"  {count:5d} blocks  {title}")


def cmd_parse(args) -> None:
    _work()
    export = roam_parse.load_export(args.export)
    graph = roam_parse.parse(export, subtree=args.subtree)
    _dump(BASE, graph.to_dict())
    print(f"  {len(graph.nodes)} nodes, {len(graph.edges)} edges (from Roam links only)")


def cmd_extract(args) -> None:
    _work()
    export = roam_parse.load_export(args.export)
    blocks = roam_parse.blocks_for_extraction(export, subtree=args.subtree)
    if args.limit:
        blocks = blocks[: args.limit]

    print(f"{len(blocks)} blocks qualify for extraction")
    if args.dry_run:
        # The cost preview. Look at this before pointing the stage at a real
        # graph — it is the difference between a few cents and a surprise.
        model = args.model or model_for("extract")
        chars = sum(len(b["text"]) for b in blocks)
        # Each call also carries the system prompt, which for a short block is
        # most of the input. Counting only block text understates the bill.
        system_tokens = len(prompts.EXTRACT_SYSTEM) // 4
        input_tokens = chars // 4 + system_tokens * len(blocks)
        print(f"  ~{chars:,} chars of block text; ~{input_tokens:,} input tokens with prompts")
        print(f"  model: {model}   calls: {len(blocks)}")
        try:
            models = catalog.fetch()
            info = catalog.find(models, model)
            if info is None:
                print(f"  !! {model} is not in the OpenRouter catalog — check the id")
            else:
                est = catalog.estimate(info, calls=len(blocks), input_tokens=input_tokens)
                print(
                    f"  estimated cost: ${est['total_usd']:.4f}"
                    f"  (in ${est['cost_input_usd']:.4f} + out ${est['cost_output_usd']:.4f})"
                )
        except Exception as exc:
            print(f"  (could not price it: {exc})")
        for b in blocks[:5]:
            print(f"    {b['uid']}: {b['text'][:90]}")
        return

    triples, failures = extract_stage.run(blocks, model=args.model)
    _dump(TRIPLES, [{**asdict(t)} for t in triples])
    print(f"  {len(triples)} triples from {len(blocks)} blocks, {len(failures)} failures")
    # Always write, even empty: a stale failures file from a prior broken run
    # must not sit there looking current next to a clean triples.json.
    failures_path = WORK / "extract_failures.json"
    if failures:
        _dump(failures_path, failures)
    elif failures_path.exists():
        failures_path.unlink()
        print(f"  removed stale {failures_path}")


def cmd_resolve(args) -> None:
    _work()
    triples = _load_triples()
    resolution = resolve_stage.run(triples, model=args.model, use_llm=not args.no_llm)
    _dump(RESOLUTION, {
        "mapping": resolution.mapping,
        "types": resolution.types,
        "groups": resolution.groups,
    })
    merged = [g for g in resolution.groups if len(g["members"]) > 1]
    print(f"  {len(resolution.groups)} groups, {len(merged)} of them merges")


def cmd_assemble(args) -> None:
    _work()
    if not BASE.exists():
        sys.exit(f"{BASE} not found — run `parse` first.")
    base = _graph_from_dict(json.loads(BASE.read_text(encoding="utf-8")))

    triples = _load_triples() if TRIPLES.exists() else []
    resolution = None
    if RESOLUTION.exists():
        raw = json.loads(RESOLUTION.read_text(encoding="utf-8"))
        resolution = resolve_stage.ResolutionMap(
            mapping=raw["mapping"],
            # older resolution.json files predate the `types` field
            types=raw.get("types", {}),
            groups=raw["groups"],
        )

    graph = assemble_stage.assemble(base, triples, resolution)
    if DB.exists() and not args.append:
        DB.unlink()
    with GraphStore(DB) as store:
        store.write(graph)
    print(json.dumps(assemble_stage.stats(graph), indent=2))
    print(f"  wrote {DB}")


def cmd_query(args) -> None:
    if not DB.exists():
        sys.exit(f"{DB} not found — run `assemble` first.")
    with GraphStore(DB) as store:
        answer = query_stage.ask(store, args.question, hops=args.hops, model=args.model)
    print(f"\n{answer.answer}\n")
    if answer.seeds:
        print(f"seeds: {', '.join(answer.seeds)}")
    print(f"sufficient: {answer.sufficient}   triples shown: {answer.triples_shown}")
    if answer.citations:
        print("\ncitations:")
        for c in answer.citations:
            print(
                f"  {c['source']} --{c['predicate']}--> {c['target']}"
                f"   [page \"{c['page_title']}\", block {c['block_uid']}]"
            )
    if answer.invalid_citations:
        print(f"\n!! model cited edges it was not shown: {answer.invalid_citations}")


def cmd_eval(args) -> None:
    if not DB.exists():
        sys.exit(f"{DB} not found — run `assemble` first.")
    with GraphStore(DB) as store:
        graph = store.load()

    report: dict = {
        "provenance": score.score_provenance(graph),
        "link_baseline": score.link_baseline(graph),
    }

    if args.gold:
        gold = score.load_gold(args.gold)
        if gold.get("extraction") and TRIPLES.exists():
            triples = _load_triples()
            report["extraction"] = asdict(
                score.score_extraction(triples, gold["extraction"])
            )
            report["extraction_ignoring_predicate"] = asdict(
                score.score_extraction(triples, gold["extraction"], ignore_predicate=True)
            )
        if gold.get("resolution") and RESOLUTION.exists():
            raw = json.loads(RESOLUTION.read_text(encoding="utf-8"))
            report["resolution"] = score.score_resolution(raw["groups"], gold["resolution"])
        if gold.get("questions"):
            answers = []
            with GraphStore(DB) as store:
                for q in gold["questions"]:
                    answers.append(query_stage.ask(store, q["question"], model=args.model))
            report["grounding"] = score.score_grounding(answers)

    print(json.dumps(report, indent=2))
    _work()
    _dump(WORK / "eval_report.json", report)


def cmd_models(args) -> None:
    """The live OpenRouter catalog, cheapest first. No API key needed.

    Prices and ids move; this reads them rather than trusting anything written
    down in this repo. `blended` assumes ~4:1 input:output, the shape of an
    extraction call.
    """
    models = catalog.fetch()
    if args.match:
        needle = args.match.lower()
        rows = [m for m in models if needle in m.id.lower() or needle in m.name.lower()]
        rows.sort(key=lambda m: m.prompt_per_mtok * 4 + m.completion_per_mtok)
    else:
        rows = catalog.cheapest(models, limit=args.limit, include_free=args.free)

    print(f"{len(models)} models available; showing {len(rows)}\n")
    print(f"{'model id':<48} {'in $/Mtok':>10} {'out $/Mtok':>11} {'context':>9}")
    print("-" * 82)
    for m in rows:
        print(
            f"{m.id:<48} {m.prompt_per_mtok:>10.4f} {m.completion_per_mtok:>11.4f} {m.context:>9,}"
        )
    print(
        "\nSet one per stage in .env: ROAMEX_MODEL_EXTRACT / _RESOLVE / _QUERY."
        "\nCheap is fine for `extract` — every triple is quote-checked against its"
        "\nsource block, so junk is dropped rather than believed. Verify with"
        "\n`src.cli eval` before trusting a model on `resolve`, where a wrong merge"
        "\nis silent and unrecoverable."
    )


def cmd_stats(args) -> None:
    if not DB.exists():
        sys.exit(f"{DB} not found — run `assemble` first.")
    with GraphStore(DB) as store:
        print(json.dumps(store.counts(), indent=2))


def _graph_from_dict(data: dict):
    from .models import Edge, Graph, Node

    graph = Graph()
    for n in data["nodes"]:
        graph.nodes[n["id"]] = Node(
            id=n["id"],
            type=n["type"],
            name=n["name"],
            description=n.get("description", ""),
            aliases=n.get("aliases", []),
            provenance=[Provenance(**p) for p in n.get("provenance", [])],
        )
    for e in data["edges"]:
        graph.edges.append(
            Edge(
                source_id=e["source_id"],
                predicate=e["predicate"],
                target_id=e["target_id"],
                provenance=Provenance(**e["provenance"]),
                confidence=e.get("confidence", 1.0),
            )
        )
    return graph


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="roamex")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="fetch the graph from Roam's live API")
    p.add_argument("--out", default="exports/roam.json")
    p.add_argument("--depth", type=int, default=roam_api.DEFAULT_DEPTH)
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("pages", help="list page titles; pick a subtree")
    p.add_argument("--export", required=True)
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_pages)

    p = sub.add_parser("parse", help="Roam export -> base graph (no LLM)")
    p.add_argument("--export", required=True)
    p.add_argument("--subtree", default=None)
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("extract", help="block prose -> candidate triples (LLM)")
    p.add_argument("--export", required=True)
    p.add_argument("--subtree", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--dry-run", action="store_true", help="count and cost, no calls")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("resolve", help="merge duplicate entity mentions")
    p.add_argument("--model", default=None)
    p.add_argument("--no-llm", action="store_true", help="blocking only, no arbitration")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("assemble", help="base + triples -> the graph store")
    p.add_argument("--append", action="store_true", help="keep the existing db")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("query", help="ask a grounded question")
    p.add_argument("question")
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("eval", help="score extraction, resolution, provenance, grounding")
    p.add_argument("--gold", default=None)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("models", help="live OpenRouter catalog, cheapest first")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--match", default=None, help="filter by id or name substring")
    p.add_argument("--free", action="store_true", help="include $0 models")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("stats", help="what is in the store")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
