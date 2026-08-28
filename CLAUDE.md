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

python -m src.cli pull    [--out exports/roam.json]              # optional: live pull instead of a manual export
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

`parse`, `assemble`, `stats`, `models` and the offline half of `eval` need no key.
`extract`, `resolve` and `query` call OpenRouter.

**Always `--dry-run` before a real `extract`.** It prints the call count, token estimate, and
a live dollar figure for the configured model. `extract` is one call per qualifying block and
is where essentially all of this app's cost lives.

Deploy: there isn't one. This is a local CLI over local files. If that ever changes it is
`ops` work behind a human gate, because it would mean putting a personal knowledge base
somewhere it can be reached.

## Where the models come from

Model calls go to **OpenRouter** (`src/llm/openrouter.py`) over plain HTTP, not to a vendor
SDK. Day-to-day iteration on this repo happens in **OpenCode**, not Claude Code. Two separate
model settings that are easy to confuse:

- **The pipeline's** models — what `extract`/`resolve`/`query` call. Set in `.env` via
  `ROAMEX_MODEL_EXTRACT` / `_RESOLVE` / `_QUERY`, or per-run with `--model`. Nothing about
  this depends on which editor or agent you are using.
- **OpenCode's own** model — the assistant you are chatting with while editing. Configured
  inside OpenCode, and unrelated to the above.

`python -m src.cli models` prints the live OpenRouter catalog sorted cheapest-first, so the
repo never hardcodes a price list that goes stale. `--match` filters it.

Tiering is by **verifiability**, not prestige:

| Stage | Volume | Checked afterwards? | Spend |
|---|---|---|---|
| `extract` | high | yes — every triple must quote its block | cheap |
| `resolve` | low | **no** — a bad merge is undetectable later | don't cheapen blindly |
| `query` | low | partly — citations are validated | middle |

A cheap extractor is affordable precisely because `parse_extraction_response` throws away
what it can't ground. That argument does **not** transfer to `resolve`, where nothing
downstream can tell a correct merge from a wrong one. If you cheapen that stage, run
`eval --gold` and read `over_merge_rate` before believing the graph.

## The Roam API integration — status

`python -m src.cli pull` fetches the graph live from `https://api.roamresearch.com` instead
of a manual browser export, via `src/roam/api.py`. It returns exactly the page/children shape
`roam.parse.load_export()` reads from a file, so nothing downstream changes — `pull` is a
second way to produce the input, not a parallel pipeline.

**Verified 2026-08-28** against a real graph (1,490 pages via `pull` vs. 1,487 in a manual
export taken two days earlier). Every uid and block string matched exactly across all 1,487
pages present in both — 0 mismatches. The 3 extra pages in the live pull were legitimate:
two daily-note pages Roam auto-created in the two days since the manual export, and one
`API Token: <name>` page Roam itself creates when a token is generated. Total block count
matched exactly (7,446 = 7,446), and the pulled export ran through `parse` and produced
identical output to the manual export on the same subtree. The endpoint paths, auth headers,
and peer-redirect handling (sourced from a community client, not Roam's own docs — see
`src/roam/api.py`'s module docstring for why) and the Datalog attribute names all held up on
this first real run, including the one piece that genuinely could not be checked beforehand:
the JSON key spelling a pulled entity serializes to.

**What this does NOT establish:** only one graph and one token were tested, at the default
`depth=20`, on one run. Not yet exercised: a graph nested deeper than 20 levels (would
silently truncate — `_selector()` has no depth-exceeded warning), the 401/403/404/429/503
error paths (never triggered, so their messages are unverified in practice even though the
code paths exist), or repeated pulls to confirm nothing changes between them. Re-verify after
any change to `api.py`, and treat those specific gaps as real, not hedging.

Get `ROAM_API_TOKEN` from your graph's own settings in Roam (an "API tokens" section) — this
tool cannot generate one for you. `ROAM_GRAPH_NAME` is the graph's name as it appears in its
URL. Both go in `.env`; see `.env.example`.

## Architecture

```
src/cli.py              entry point; every stage is a subcommand
src/models.py           Node / Edge / Triple / Provenance — the contract between stages
src/roam/api.py         OPTIONAL  live pull from Roam's Backend API -> same shape load_export() returns
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
9. **Subtree first — and the reason is no longer money.** At current cheap-model prices the
   entire 7.4k-block graph extracts for about $0.16, so cost is not the argument. The two
   arguments that remain are better ones: a bad prompt or a wrong entity schema is much
   cheaper to find on 59 blocks than on 4,757, and a full run ships ~916k characters of a
   personal knowledge base — health, family, finances — to a third-party inference provider
   in one go. Prove the schema on a page you would not mind a stranger reading, then widen
   deliberately.

## Scale — what has and has not been established

**Updated 2026-08-28** after running a 1,000-block sample pulled across a real ~1,500-page,
~7,450-block graph (21% of the full extractable set) — no longer just the synthetic fixture.
Result: 594 triples extracted with 0 failures, assembled into 2,082 nodes / 3,211 edges with
100% provenance, `eval` and a `hops=3` query both ran in under a second of retrieval time. The
full graph has been priced (~$0.16 at current cheap-model rates) but never run end-to-end —
that is the next real scale test, not this one.

**One real bug found and fixed at this scale**, not present at small-subtree scale: a single
common word (`God`, `church`, `Christian`) acts as a hub in `resolve`'s token-subset blocking
rule, bridging otherwise-unrelated names into one component via transitive closure — one real
cluster reached 73 members and its arbitration call predictably overflowed `max_tokens`. It
failed safely (no over-merge — every member left unmerged, the correct direction), but wasted
a real API call on something structurally doomed. `MAX_ARBITRATION_CLUSTER` (25) now skips
straight to the same safe outcome for free. **This caps the symptom, not the cause**: blocking
is still too permissive around short common-word hubs, and some "entity" names extraction
returns are really whole clauses, not entities. Both are genuine precision issues worth their
own pass, not yet done.

**One measured, still-fine limitation**: `ask()` calls `GraphStore.load()` twice per query
(once via `find_seeds`, once via `subgraph()`) — a real redundancy. At 2,082/3,211 scale that
cost ~0.8s of an 11.5s query, ~90% of which was the model round-trip. Not yet the bottleneck;
worth fixing when `load()` itself measurably is, alongside the recursive-CTE change already
noted below.

What's still genuinely untested past this 1,000-block sample:
- The full graph at ~4,750 extractable blocks — 4.75× this sample, never run.
- `GraphStore.load()` reading the *whole* graph on every call, at real scale (a million edges,
  not a few thousand). The fix when it matters is a recursive CTE in SQL — the schema already
  supports it. Not worth building before a measurement past this session's says so.
- `resolve`'s blocking is O(n²) over unique names — fine through ~630 (measured this session);
  unknown beyond that.

## The frontend

`web/` is built (2026-08-28): `python web/serve.py` serves an isometric read-only viewer over
`work/graph.db`. See `web/README.md` for the design language it follows, its routes, and what
is and isn't tested. Three constraints that matter here:

- **The dependency runs one way.** `web/serve.py` imports `src.store.graph` and
  `src.pipeline.query`; **nothing in `src/` may import from `web/`.** The viewer reads the
  pipeline's own store and query code rather than reimplementing either — a second SQL layer
  would be a second thing to keep correct. Delete `web/` and the pipeline still runs.
- **`/api/query` is POST, deliberately.** It is the only route that costs money, and a GET
  would let a page load, refresh, or browser prefetch trigger a model call.
  `tests/test_web.py` asserts it never appears in `do_GET`.
- **Binds `127.0.0.1`, read-only.** Nothing is authenticated and the payload is a personal
  knowledge base. No route writes to the db or to Roam.

**The rendering is unverified.** The JSON shaping is tested and the isometric layout was
smoke-tested headless for cell collisions at 2,082 nodes, but the page has never been opened
in a browser — the Chrome extension was not connected when it was written. Treat visual
correctness as unproven until someone looks at it.
