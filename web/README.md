# web/ — the local viewer

A read-only local viewer for an assembled graph: browse entities as an isometric map,
read where every claim came from, and ask the graph questions.

```bash
python web/serve.py                                  # opens 127.0.0.1:8790
python web/serve.py --db work/graph.db --port 8791 --no-open
```

Needs a graph first (`python -m src.cli assemble`). The server tells you so and names the
commands if `work/graph.db` isn't there.

## What it shows

Entities are isometric structures on a khaki grid. **Height encodes degree** — the busiest
entities stand tallest and sit nearest the centre. **A solid outline means you wrote that
link by hand** in Roam; **dashed means a model inferred it from prose** and could quote a
block for it. Edges follow the same rule: solid grid-coloured for Roam's own links, dashed
accent for inferred ones. Selecting a structure dims everything it doesn't touch.

The right panel has three tabs: **Detail** (what an entity says, what refers to it, and
every source block behind it), **Ask** (a grounded question against the whole graph — the
one thing here that costs a model call), and **About**.

## Design

Follows the [system-atlas](https://github.com/inkboard/system-atlas) design language —
khaki paper `#E6DFBE`, ink `#17170F`, IBM Plex Mono, 1.5px borders with 2px hard shadows,
`<mark>` for key phrases, a left index / canvas / right panel layout, and its isometric
projection (`P(gx,gy,z) = [(gx−gy)·36, (gx+gy)·18 − z]` on a 72×36 tile, structures sorted
by `gx+gy` for painter's order). Theme tokens sit on `:root`, redefined under
`prefers-color-scheme: dark` (guarded so an explicit light choice wins) and
`[data-theme="dark"]`.

Atlas is a tool for *explaining a system architecture* in progressive-disclosure chapters.
This is a tool for *exploring a knowledge graph you didn't design and can't fully hold in
your head*. So the visual grammar is carried over; the chapter machinery is not — there's
no authored narrative here, and pretending there was one would mean inventing structure
your notes don't have.

## Routes

| Route | Method | Cost |
|---|---|---|
| `/` | GET | free |
| `/api/graph` | GET | free — nodes, edges, counts |
| `/api/node/<id>` | GET | free — one node, its edges, full provenance |
| `/api/query` | **POST** | **a model call** |

`/api/query` is POST specifically so that loading, refreshing, or prefetching the page
can't spend money. A test asserts it never appears in `do_GET`.

## Constraints

- **Read-only.** No route writes to the database, and nothing here touches Roam.
- **Binds `127.0.0.1`.** Nothing is authenticated and the payload is the contents of a
  personal knowledge base. Do not bind it wider.
- **The dependency runs one way.** `web/` imports from `src/`; nothing in `src/` may import
  from `web/`. Delete this directory and the pipeline still runs.
- **The whole graph is sent to the page** in one `/api/graph` response. Fine at the scale
  tested (2,082 nodes / 3,211 edges); this is the first thing that will need paging if the
  graph gets much larger.

## What's tested, and what isn't

`tests/test_web.py` covers the JSON shaping against a real temporary store — the part that
fails silently, since a dropped field renders as a blank rather than an error — plus source
guards for the bugs that have actually bitten (pointer capture, index grouping, type codes).
It does not start a server.

**Nothing here tests the rendering.** The layout is smoke-tested headless for cell
collisions at 2,082 nodes, and the page has been confirmed rendering correctly by hand (see
the screenshot in the top-level README), but nothing catches a visual regression
automatically. Known rough edge: at full-graph zoom, structure labels in the dense centre
overlap into noise — zoom in, or use the index.
