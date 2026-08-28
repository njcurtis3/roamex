# web/ — placeholder

Not built yet, on purpose.

The plan is a local viewer over `work/graph.db`: browse entities, see an entity's edges with
their source blocks, run a query and inspect which triples the model was shown versus which
it cited.

It is not built because a UI over a pipeline whose output has not been validated against real
notes is a UI you rewrite. The order is: ingest a real subtree → read the eval numbers → fix
the prompts → *then* build something to look at it with.

When it is built:

- It reads `work/graph.db`. Nothing in `src/` may import from here — the pipeline stays
  runnable with this directory deleted.
- It is a local tool. Serving a personal knowledge base anywhere reachable is `ops` work
  behind a human gate, not a dev-server default.
