-- roamex graph store.
--
-- Deliberately boring: three tables in SQLite, not a graph database. The graph
-- this app targets is one person's notes, and every query it runs is a k-hop
-- neighborhood around a handful of seeds. Recursive CTEs do that fine. Reach
-- for a real graph engine when a measured query is too slow, not before.
--
-- `provenance` is its own table rather than a column because one node or edge
-- can be asserted by many blocks, and every one of those assertions is
-- independent evidence worth keeping.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    aliases     TEXT NOT NULL DEFAULT '[]'  -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);

CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    predicate  TEXT NOT NULL,
    target_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    confidence REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

-- One row per (fact, supporting block). subject_kind tells you whether the
-- fact is a node or an edge; subject_ref is the node id or the edge rowid.
CREATE TABLE IF NOT EXISTS provenance (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_kind   TEXT NOT NULL CHECK (subject_kind IN ('node', 'edge')),
    subject_ref    TEXT NOT NULL,
    block_uid      TEXT NOT NULL,
    page_title     TEXT NOT NULL,
    origin         TEXT NOT NULL CHECK (origin IN ('roam-link', 'llm')),
    extracted_at   TEXT NOT NULL,
    model          TEXT,
    prompt_version TEXT,
    quote          TEXT
);

CREATE INDEX IF NOT EXISTS idx_prov_subject ON provenance(subject_kind, subject_ref);
CREATE INDEX IF NOT EXISTS idx_prov_block ON provenance(block_uid);
