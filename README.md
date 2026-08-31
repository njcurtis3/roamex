# roamex

Turns a Roam Research graph into a queryable knowledge graph where every answer cites the
block it came from.

A Roam export is already a graph: pages, blocks, `[[links]]`, `#tags`, and `attribute:: value`
pairs. Stage one reads that structure directly, with no model involved. Stage two sends block
prose to a language model, which extracts the relations the author stated in a sentence but
never linked. A block reading "Started at Acme in 2019, reporting to Dana" contains no
`[[Acme]]` and no `[[Dana]]`, so Roam's own graph shows nothing. roamex turns it into
`author --works_at--> Acme` and `author --reports_to--> Dana Whitfield`. Every node and every
edge carries the uid of the block that produced it, so any answer traces back to a note.

```
$ python -m src.cli query "who wrote the Kestrel parser?"

Dana Whitfield wrote the Kestrel parser. She is the data lead at Rivet Labs,
which builds tooling for Project Halyard.

seeds: Kestrel parser
sufficient: true   triples shown: 14

citations:
  Dana Whitfield --wrote--> Kestrel parser      [page "Dana Whitfield", block blk-008]
  Dana Whitfield --works_at--> Rivet Labs       [page "Project Halyard", block blk-003]
```

## Prerequisites

- Python 3.13, per the header of `requirements.txt`.
- `pytest` is the only installed dependency. The pipeline itself runs on the standard library:
  exports are JSON, the store is `sqlite3`, and the model call is one POST through `urllib`.
- An OpenRouter API key. The `extract`, `resolve`, and `query` stages call
  [OpenRouter](https://openrouter.ai). Every other command runs offline.
- A Roam export as JSON at `exports/roam.json`, or a `ROAM_API_TOKEN` and `ROAM_GRAPH_NAME`
  for the `pull` command. Take the token from the API tokens section of the graph's own
  settings in Roam. roamex cannot generate one.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/cli.py` | Entry point, one subcommand per stage |
| `src/models.py` | `Node`, `Edge`, `Triple`, and `Provenance`, the contract between stages |
| `src/roam/` | Export loading, the deterministic parse, and the live API pull |
| `src/pipeline/` | One module per stage: extract, resolve, assemble, query |
| `src/llm/` | OpenRouter client, versioned prompts, and the model catalog |
| `src/store/` | SQLite persistence and schema, the only code here that knows SQL |
| `src/eval/` | Precision, recall, over-merge rate, provenance coverage, grounding |
| `web/` | Local read-only viewer, documented in `web/README.md` |
| `fixtures/` | Synthetic Roam export and recorded model replies that every test reads |
| `eval/gold/` | Hand-labelled answers to score a run against |
| `tests/` | pytest suite, offline |
| `docs/` | Screenshots this README links |

## Local Development

Install the dependency and create the config file:

```bash
python -m pip install -r requirements.txt
cp .env.example .env      # add OPENROUTER_API_KEY, and the Roam pair if you use pull
```

Run the pipeline. Each stage writes a file under `work/` and the next stage reads it back, so
a stage is resumable and inspectable on its own:

```bash
python -m src.cli pull                                        # writes exports/roam.json
python -m src.cli pages   --export exports/roam.json          # find a page to start on
python -m src.cli parse   --export exports/roam.json --subtree "Project X"
python -m src.cli extract --export exports/roam.json --subtree "Project X" --dry-run
python -m src.cli extract --export exports/roam.json --subtree "Project X"
python -m src.cli resolve
python -m src.cli assemble
python -m src.cli query "what depends on the old parser?"
```

Without a Roam API token, skip `pull` and drop a manual JSON export at `exports/roam.json`.
Every other command works unchanged.

`--dry-run` reports the call count and a priced estimate without making any calls. Prices come
from the OpenRouter catalog at runtime, so this repo hardcodes no price list:

```
375 blocks qualify for extraction
  ~68,134 chars of block text; ~164,033 input tokens with prompts
  model: qwen/qwen3.7-flash   calls: 375
  estimated cost: $0.0122  (in $0.0049 + out $0.0073)
```

Run the test suite:

```bash
python -m pytest
```

Every test runs offline. Each model-backed stage splits into a `run()` that calls the network
and a `parse_*_response()` that does not. The tests exercise only the second half, against the
recorded replies in `fixtures/openrouter/`.

## Pipeline stages

| Stage | What it does | Model call |
| --- | --- | --- |
| `pull` | Fetches the graph from Roam's HTTP API into an export file | no |
| `pages` | Lists page titles and block counts, so you can pick a subtree | no |
| `parse` | Reads links, tags, and attributes into a base graph | no |
| `extract` | Reads block prose and returns candidate triples, each quoting its block | yes |
| `resolve` | Merges duplicate mentions such as `Dana` and `Dana Whitfield` | on ambiguity |
| `assemble` | Folds triples into the base graph and writes `work/graph.db` | no |
| `query` | Expands a k-hop subgraph around the question and answers with citations | yes |
| `eval` | Scores extraction, resolution, provenance coverage, and grounding | only with `--gold` questions |
| `models` | Prints the live OpenRouter catalog, cheapest first | no |
| `stats` | Prints what the store holds | no |

Each stage picks its own model, tiered by how checkable its output is. Extraction is
high-volume and independently verified, because every triple must quote its source block or
get dropped, so a cheap model is affordable. Resolution is low-volume and unverifiable after
the fact, so it takes a stronger model. `src/llm/openrouter.py` holds the defaults:
`qwen/qwen3.7-flash` for extract, and `google/gemini-3.6-flash` for resolve and query. Override
any of them with `ROAMEX_MODEL_EXTRACT`, `ROAMEX_MODEL_RESOLVE`, and `ROAMEX_MODEL_QUERY` in
`.env`, or per run with `--model`.

## Viewer

```bash
python web/serve.py          # opens 127.0.0.1:8790
python web/serve.py --db work/graph.db --port 8791 --no-open
```

![The roamex viewer: an index on the left, an isometric map of the graph in the middle, and a detail panel on the right showing every source block behind the selected entity](docs/viewer.png)

A read-only viewer served from `work/graph.db`. Run `assemble` first. Entities appear as
isometric structures whose height encodes degree, so the busiest entities stand tallest. A
solid outline marks a link written by hand in Roam. A dashed outline marks a link a model
inferred from prose. The right panel carries four tabs: Detail, Path, Ask, and About. Detail
shows what an entity says, what refers to it, and every source block behind each claim. Read
`web/README.md` for the routes, the design language, and the test coverage.

### Path — the shortest connection between two entities

![The Path tab: from "AI 2027" to "working-without-externally-provided-feedback", found directly connected by a "mentions" edge, with the connecting path highlighted in accent orange across the isometric map](docs/viewer-path.png)

Path runs a breadth-first search in the browser, over the graph the page already holds. The
search makes no model call and costs nothing. Path searches the whole graph whatever the
origin filter is set to. It switches the filter back to "all" when the filter would otherwise
hide the path it found.

### Ask — grounded question answering

![The Ask tab, asked "What did I write about feedback loops in AI 2027?": the answer says the graph shows AI 2027 mentions feedback-loops-working-without-externally-provided-feedback but does not contain the actual text, flagged "graph did not have enough" and "93 triples shown", with one citation and its quoted source block including the block's full URL, wrapped rather than clipped](docs/viewer-ask.png)

Ask is the one feature in the viewer that spends a model call. Ask builds an answer only from
triples in the graph, and every claim cites the page and the block it came from. When the
graph holds too little to answer, the reply says so and sets `sufficient: false` rather than
padding the answer with plausible filler.

## Deployment

roamex has no deploy pipeline, no CI configuration, and no hosted component. The CLI and the
viewer both run on a local machine against files under `work/`. `web/serve.py` binds
`127.0.0.1`, and hosting it anywhere reachable would publish a personal knowledge base with no
authentication in front of it.

## Notes and gotchas

- Never commit the export or anything derived from it. `exports/`, `work/`, and unlabelled
  gold sets are gitignored. A note committed once stays in every clone forever.
- `extract` sends block text to a third-party inference provider. A full-graph run ships the
  whole knowledge base in one go, so prove the prompt and the schema on one `--subtree` first.
- `parse_extraction_response` drops any triple whose quote is not a substring of its source
  block. Relaxing that check raises recall and admits claims the notes never made.
- An over-merge in `resolve` welds two entities together and leaves no signal that they were
  ever separate. A missed merge leaves a visible duplicate node instead. `eval` reports
  `over_merge_rate`, which is the number to watch after changing the resolve model.
- A reasoning model can spend its whole `max_tokens` budget on reasoning and return empty
  content, which looks exactly like a broken extractor. `extract.run()` passes
  `reasoning={"enabled": False}` for that reason. A different reasoning model needs the same
  override or a larger `max_tokens`.
- `resolve` blocks candidate names by shared tokens, and one common word acts as a hub that
  bridges unrelated names into a single cluster. `MAX_ARBITRATION_CLUSTER` in
  `src/pipeline/resolve.py` caps the symptom at 25 members by leaving such a cluster unmerged.
  The blocking rule itself is unchanged.
- `pull` requests a fixed nesting depth, 20 by default. `_selector()` in `src/roam/api.py`
  raises no warning when a graph nests deeper, so blocks below that depth go missing without
  a message.
- Prompts are versioned in `src/llm/prompts.py`, and the version string is written into the
  provenance of every fact. Change a prompt without bumping its version, and no later run can
  tell which prompt produced a given fact.
- Every node and edge carries a block uid. `Provenance` raises without one, and
  `GraphStore.load()` refuses to return an edge that has none.
- Nothing in `src/` imports from `web/`. Delete `web/` and the pipeline still runs.

## Further reading

- [CLAUDE.md](CLAUDE.md) — architecture, the scale measurements taken so far, and the
  constraints behind each design decision
- [web/README.md](web/README.md) — viewer routes, design language, and what the tests cover
- The module docstring in [src/roam/api.py](src/roam/api.py) — where the Roam API endpoint
  paths and headers came from, and which of them a live run has confirmed
- [OpenRouter](https://openrouter.ai) — the model gateway every model-backed stage calls
