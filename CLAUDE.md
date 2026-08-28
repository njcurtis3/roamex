# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Standalone app under the repos/ umbrella. Never import from a sibling app; see ../graph_agents/CLAUDE.md.

## What this repository is

`roamex` turns a Roam Research export into a queryable, provenance-tracked knowledge graph.
A Roam export is *already* a graph — pages, blocks, `[[links]]`, `#tags`, `attribute:: value`
— so the first stage reads that structure directly and deterministically. The value this app
adds is the **second** layer: an LLM reads block *prose* and extracts the relations the author
stated in a sentence but never bothered to link. "Started at Acme in 2019," with no `[[Acme]]`
anywhere, is invisible to Roam's own graph and is exactly what belongs here. Every fact,
from either layer, carries the block uid it came from, so any answer can be traced back to
the note that produced it.

## Run it

```bash
python -m pip install -r requirements.txt
cp .env.example .env          # then add your OpenRouter key

python -m src.cli pages   --export exports/roam.json            # pick a subtree
python -m src.cli parse   --export exports/roam.json --subtree "Project X"
python -m src.cli extract --export exports/roam.json --subtree "Project X" --dry-run
python -m src.cli extract --export exports/roam.json --subtree "Project X" --limit 20
python -m src.cli resolve
python -m src.cli assemble
python -m src.cli query "who works on Project X?"
python -m src.cli eval --gold eval/gold/sample.json
python -m src.cli stats

python -m pytest              # full suite; every test is offline
```

`parse`, `assemble`, `stats` and the offline half of `eval` need no key and no network.
`extract`, `resolve` and `query` call OpenRouter.

**Always `--dry-run` before a real `extract`.** It prints the call count and token estimate
without spending anything. `extract` is one call per qualifying block and is where essentially
all of this app's cost lives.

Deploy: there isn't one. This is a local CLI over local files. If that ever changes it is
`ops` work behind a human gate, because it would mean putting a personal knowledge base
somewhere it can be reached.

## Where the models come from

Model calls go to **OpenRouter** (`src/llm/openrouter.py`), not to a vendor SDK. Day-to-day
iteration on this repo happens in **OpenCode**, not Claude Code — model choice, tiering, and
pipeline runs are driven from there. Claude Code scaffolded the repo; it is not the runtime.

Tiering is per stage and is set in `DEFAULT_MODELS`, overridable per stage via
`ROAMEX_MODEL_EXTRACT` / `_RESOLVE` / `_QUERY` in `.env`. Spend on judgment, not on volume:
`extract` reads one block and fills a fixed schema (cheap, high-volume); `resolve` decides
whether two names are one entity, which is the call that silently corrupts the graph when it
is wrong (not cheap).

## Architecture

```
src/cli.py              entry point; every stage is a subcommand
src/models.py           Node / Edge / Triple / Provenance — the contract between stages
src/roam/parse.py       STAGE 1  export -> base graph. Deterministic, no LLM.
src/pipeline/extract.py STAGE 2  block prose -> candidate triples. LLM.
src/pipeline/resolve.py STAGE 3  merge duplicate mentions. Blocking + LLM arbitration.
src/pipeline/assemble.py STAGE 4 base + triples -> one graph. Deterministic, no LLM.
src/pipeline/query.py   STAGE 5  question -> k-hop subgraph -> cited answer. LLM.
src/store/graph.py      SQLite persistence; the only file that knows SQL
src/eval/score.py       precision/recall, over-merge rate, provenance coverage, grounding
fixtures/               a tiny Roam export + recorded model replies. Every test reads these.
eval/gold/              hand-labelled answers to score against
work/                   all stage output, including graph.db. Gitignored, reproducible.
```

Data flow: each stage writes a file under `work/` and the next reads it back. That is
deliberate — extraction costs real money, and a pipeline that re-runs it because the query
prompt changed is one you stop using. Stages are resumable and separately inspectable.

## Constraints — things an agent will otherwise get wrong

1. **Never commit the Roam export or anything derived from it.** `exports/`, `work/`, and
   unlabelled gold sets are gitignored. This is a whole personal knowledge base — contacts,
   health, money, unguarded opinions about people who never agreed to be in a git repo. A
   note committed once is in every clone forever. When adding a fixture, write a *synthetic*
   one; never trim a real page down and call it redacted.
2. **Provenance is not optional.** Every node and every edge carries a `block_uid`. `Provenance`
   raises without one, `GraphStore.load()` refuses to return an unprovenanced edge, and
   `eval` reports coverage that should read `1.0`. A fact this app cannot trace is a fact it
   cannot defend, which is the entire reason it exists rather than a RAG index.
3. **Extracted triples must quote the block.** `parse_extraction_response` drops any triple
   whose `quote` is not a substring of the source block. That check is what separates "the
   note says this" from "the model knows this about the world" — and the second one is
   unfalsifiable once it is in the graph. Do not relax it to raise recall.
4. **Over-merging is the expensive error; missed merges are cheap.** A missed merge leaves a
   duplicate node you can see and fix. An over-merge welds two entities' facts together and
   leaves no signal that they were ever separate. Every default in `resolve.py` — failing
   closed on an error, refusing to merge on conflicting descriptions, restoring dropped names
   as singletons — is biased that way on purpose. `over_merge_rate` is the number to watch.
5. **`parse` is deterministic and stays that way.** No model touches stage 1. The author's own
   `[[links]]` are the closest thing to ground truth this corpus has, and `link_baseline` in
   the eval harness depends on them being untouched by inference. Running an LLM over
   structure the format already states is paying tokens to make ground truth fuzzier.
6. **The pure/network seam is the testability contract.** Every LLM stage splits into a
   `run()` that calls out and a `parse_*_response()` that does not. Tests only ever exercise
   the second, against fixtures in `fixtures/openrouter/`. No test may require a key or a
   network call — one that does cannot run in CI or in an isolated worktree.
7. **Node ids are derived, never generated.** `canonical_id(type, name)` is a hash of the
   normalized name, so re-ingesting the same export produces the same graph rather than a
   second copy of it.
8. **Prompts are versioned.** `EXTRACT_VERSION` and friends are written into the provenance
   of every fact. Change a prompt, bump its version — that string is the only way to ask
   later whether a prompt change broke extraction.
9. **Subtree first.** `--subtree` exists because ingesting a whole personal graph on the first
   run is how you spend a lot of money discovering the schema was wrong. Prove a page, then
   widen.

## Scale — what has and has not been established

The pipeline is proven end-to-end on a synthetic three-page fixture. **That is a correctness
proof, not a capacity plan.** Nothing here has met a real graph yet, and the two things most
likely to break first are known:

- `GraphStore.load()` reads the entire graph into memory on every call, and `subgraph()` calls
  it. Fine at thousands of edges; not at a million. The fix when it matters is a recursive CTE
  in SQL — the schema already supports it.
- `resolve`'s blocking is O(n²) over unique names. Fine at thousands; not beyond.

Neither is worth fixing before a measurement says so.

## The frontend

`web/` is a deliberate placeholder. The plan is a graph/query UI, and it is not built,
because a UI over a pipeline whose output has not been validated is a UI you rewrite. Build
it once `query` returns answers worth looking at. Nothing in `src/` may depend on `web/`.
