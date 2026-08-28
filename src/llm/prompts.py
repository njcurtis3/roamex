"""Prompts, versioned.

Every prompt carries a version string that is written into the provenance of
every fact it produces. That is not bookkeeping — it is the only way to answer
"did that prompt change break extraction?" after the fact. Change a prompt,
bump its version, and the eval harness can compare the two populations.
"""

EXTRACT_VERSION = "extract/v1"
RESOLVE_VERSION = "resolve/v1"
QUERY_VERSION = "query/v1"

ENTITY_TYPES = ["person", "org", "project", "place", "concept", "event", "tool", "source"]

EXTRACT_SYSTEM = f"""You extract structured relations from a single note written by one person in their personal knowledge base.

Return ONLY a JSON array. Each element:
{{
  "subject": "entity name as a person would refer to it",
  "subject_type": one of {ENTITY_TYPES},
  "predicate": "snake_case verb phrase, e.g. works_at, wrote, lives_in, depends_on",
  "object": "entity name",
  "object_type": one of {ENTITY_TYPES},
  "subject_description": "<=12 words disambiguating this entity, or \\"\\"",
  "object_description": "<=12 words disambiguating this entity, or \\"\\"",
  "confidence": 0.0-1.0,
  "quote": "the exact substring of the note that states this"
}}

Rules:
- Extract only what the note ITSELF states. Never use your own knowledge of the world. If the note says "met Dana", you do not know Dana's employer.
- `quote` must be a verbatim substring of the note. If you cannot quote it, do not extract it.
- Prefer specific predicates over `mentions`. A relation that is only "these two words appeared together" is noise; skip it.
- Resolve pronouns to the entity they refer to only when the note makes it unambiguous. Otherwise skip.
- First person ("I", "my") refers to the note's author: use the subject "the author" with type "person".
- Speculation, questions, and things the author is considering are NOT facts. Skip "maybe X should Y".
- An empty array is a correct and expected answer for a note that states no relations. Do not invent one to be useful.
"""

EXTRACT_USER = """Note from page "{page_title}":

{text}

JSON array:"""


RESOLVE_SYSTEM = """You decide whether entity mentions from one person's notes refer to the SAME real entity.

You are given a cluster of candidate names that a cheap string filter thought might match, each with any description context available.

Return ONLY a JSON object:
{
  "groups": [
    {"canonical": "the best full name for this entity",
     "members": ["every name in this cluster that IS this entity"],
     "reason": "<=20 words"}
  ]
}

Rules:
- Every input name must appear in exactly one group. A name that matches nothing else forms its own group of one.
- Merge only on evidence: an abbreviation, a nickname, a first-name-only reference, a spelling variant, a typo. Two different people who share a first name are TWO groups.
- When the descriptions conflict, do not merge. An over-merge silently corrupts every fact attached to both entities and is far more expensive than a missed merge, which just leaves a duplicate node.
- Prefer the most complete form as `canonical` — "Buzz Aldrin" over "Buzz".
"""

RESOLVE_USER = """Candidate cluster:

{candidates}

JSON:"""


QUERY_SYSTEM = """You answer questions using ONLY the knowledge-graph triples provided.

Each triple is given as:
  [edge_id] Subject --predicate--> Object   (source: page "P", block UID)

Return ONLY a JSON object:
{
  "answer": "your answer in prose, or an explicit statement that the graph does not contain it",
  "citations": [edge_id, ...],
  "sufficient": true/false
}

Rules:
- Use ONLY the triples. You have no other knowledge of this person, their projects, or their contacts. If the triples do not answer the question, set "sufficient": false and say what is missing.
- Every claim in `answer` must be supported by an edge listed in `citations`. An uncited claim is a bug.
- Multi-hop is expected: chain triples together and cite every edge in the chain.
- Do not smooth over gaps. "The graph shows A relates to B, but nothing connects B to C" is a better answer than a plausible guess.
"""

QUERY_USER = """Question: {question}

Triples:
{triples}

JSON:"""
