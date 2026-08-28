# roamex

Turn a Roam Research export into a queryable knowledge graph where every answer cites the
block it came from.

A Roam export is already a graph: pages, blocks, `[[links]]`, `#tags`, `attribute:: value`.
roamex reads all of that directly and deterministically — no model involved, because running
one over structure the format already states just makes ground truth fuzzier.

The part worth paying for is the second pass. A language model reads the *prose* inside your
blocks and extracts the relations you stated in a sentence but never linked:

> "Started at Acme in 2019, reporting to Dana."

No `[[Acme]]`, no `[[Dana]]`, so Roam's own graph shows nothing. roamex turns it into
`author --works_at--> Acme` and `author --reports_to--> Dana Whitfield`, attached to the pages
you already keep, each edge stamped with the block uid that produced it.

Then you can ask it things, and check its work:

```
$ python -m src.cli query "who wrote the Kestrel parser?"

Dana Whitfield wrote the Kestrel parser. She is the data lead at Rivet Labs,
which builds tooling for Project Halyard.

seeds: Kestrel parser
sufficient: true   triples shown: 14

citations:
  Dana Whitfield --wrote--> Kestrel parser      [page "Dana Whitfield", block blk-008]
  Dana Whitfield --works_at--> Rivet Labs       [page "Project Halyard", block blk-003]
  Rivet Labs --builds_tooling_for--> Project Halyard  [page "Rivet Labs", block blk-009]
```

## The pipeline

| Stage | What it does | Model? |
|---|---|---|
| `parse` | export → base graph from links, tags, attributes | no |
| `extract` | block prose → candidate triples, each quoting its source | yes |
| `resolve` | "Dana" / "D. Whitfield" / "Dana Whitfield" → one entity | yes, only on ambiguity |
| `assemble` | fold triples into the base graph, keep all provenance | no |
| `query` | question → k-hop subgraph → answer with citations | yes |
| `eval` | precision/recall, over-merge rate, provenance coverage | no |

Models are called through [OpenRouter](https://openrouter.ai), tiered per stage — extraction
is high-volume and mechanical, resolution is low-volume and consequential.

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env                     # add your OpenRouter key

python -m src.cli pages   --export exports/roam.json          # find a page to start on
python -m src.cli parse   --export exports/roam.json --subtree "Project X"
python -m src.cli extract --export exports/roam.json --subtree "Project X" --dry-run
python -m src.cli extract --export exports/roam.json --subtree "Project X"
python -m src.cli resolve
python -m src.cli assemble
python -m src.cli query "what depends on the old parser?"
```

Start with `--subtree`. Ingesting a whole personal graph on the first run is how you spend a
lot of money discovering the schema was wrong.

Everything except `extract`, `resolve` and `query` runs offline with no key.

## Three things it refuses to do

**Assert what your notes don't say.** An extracted relation must quote the block it came
from, verbatim. If the model can't quote it, the triple is dropped — that's the line between
"the note says this" and "the model happens to know this," and the second one is
unfalsifiable once it's in the graph.

**Merge two people who share a first name.** A missed merge leaves a duplicate node you can
see and fix. An over-merge fuses two entities' facts with no signal they were ever separate.
Resolution fails closed at every branch, and `eval` reports the over-merge rate.

**Answer without showing its work.** Query reasons over a serialized subgraph and nothing
else. Citations are checked against what the model was actually shown; a cited edge that
wasn't in the prompt is flagged, not returned clean.

## Your notes stay yours

`exports/`, `work/`, and unlabelled gold sets are gitignored — the export, the derived graph,
the database, all of it. The test fixtures are synthetic, not redacted real notes.

Block text is sent to whichever model you point OpenRouter at. There's no exclusion filter
yet; if you need one, it belongs in `blocks_for_extraction`.

## Testing

```bash
python -m pytest
```

Every test runs offline. Each LLM stage splits into a `run()` that calls the network and a
`parse_*_response()` that doesn't; tests only exercise the second, against recorded replies
in `fixtures/openrouter/`.

## Status

Proven end-to-end on a synthetic fixture — a correctness proof, not a capacity plan. The
frontend in `web/` is a placeholder; it gets built once `query` returns answers worth
looking at. See `CLAUDE.md` for the known scale limits and why they're not fixed yet.
