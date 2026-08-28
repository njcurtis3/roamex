"""Stage 3 — decide which mentions are the same entity.

Two passes, cheap then expensive:

1. **Blocking** (deterministic, free). Group names that could plausibly match at
   all — normalized-equal, initials, one is a prefix/token-subset of the other.
   This exists to keep the model from being asked O(n^2) questions.
2. **Arbitration** (a model, only on ambiguous blocks). Reads descriptions and
   decides. Blocks of size 1, and blocks where every member normalizes to the
   same string, never reach it — there is nothing to judge.

The asymmetry that governs this whole stage: a **missed merge** leaves a
duplicate node, visible and fixable. An **over-merge** welds two entities'
facts together, and afterwards there is no signal left saying they were ever
separate. Every default here is biased toward the recoverable error.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from ..llm import prompts
from ..llm.openrouter import complete, extract_json, model_for
from ..models import Triple


def normalize(name: str) -> str:
    return " ".join(name.lower().replace(".", " ").replace("-", " ").split())


@dataclass
class ResolutionMap:
    """name -> canonical name, plus the groups it came from.

    `types` exists because `models.canonical_id` is deliberately keyed on
    (type, name) — that is what keeps "Washington" the person distinct from
    "Washington" the place. But each extraction call sees one block in
    isolation, so the *same* entity can come back typed `place` in one call and
    `concept` in another. If resolution only unified the name, those mentions
    would resolve to the same name but different node ids, and the merge this
    stage exists to do would silently fail to collapse anything. So this stage
    now owns type identity too: one canonical type per canonical name, decided
    by majority vote across every occurrence.
    """

    mapping: dict[str, str] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    groups: list[dict] = field(default_factory=list)

    def canonical(self, name: str) -> str:
        return self.mapping.get(normalize(name), name)

    def type_for(self, canonical_name: str, fallback: str) -> str:
        return self.types.get(normalize(canonical_name), fallback)


def candidate_blocks(names: list[str]) -> list[list[str]]:
    """Deterministic blocking. Returns clusters of possibly-same names."""
    unique = sorted({n.strip() for n in names if n.strip()})
    parent = {n: n for n in unique}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, a in enumerate(unique):
        na, ta = normalize(a), set(normalize(a).split())
        for b in unique[i + 1 :]:
            nb, tb = normalize(b), set(normalize(b).split())
            if na == nb:
                union(a, b)
            elif ta and tb and (ta < tb or tb < ta):
                # "Buzz" vs "Buzz Aldrin": one name's tokens contain the other's.
                union(a, b)
            elif _initials_match(na, nb):
                union(a, b)

    clusters: dict[str, list[str]] = {}
    for n in unique:
        clusters.setdefault(find(n), []).append(n)
    return sorted(clusters.values(), key=lambda c: (-len(c), c[0]))


def _initials_match(a: str, b: str) -> bool:
    """"jfk" vs "john f kennedy". Only fires for a 2+ letter acronym."""
    for short, long in ((a, b), (b, a)):
        if " " in short or len(short) < 2 or " " not in long:
            continue
        if "".join(w[0] for w in long.split()) == short:
            return True
    return False


def parse_resolution_response(raw_text: str, cluster: list[str]) -> list[dict]:
    """Model reply -> groups. Pure; no network.

    Enforces the invariant the prompt asks for: every input name lands in
    exactly one group. Names the model dropped are restored as singletons, and
    names it invented are discarded — silently losing an entity here would
    delete every fact attached to it downstream.
    """
    data = extract_json(raw_text)
    if isinstance(data, list):
        data = {"groups": data}
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object with a `groups` key")

    allowed = {n: n for n in cluster}
    seen: set[str] = set()
    groups: list[dict] = []

    for raw_group in data.get("groups") or []:
        if not isinstance(raw_group, dict):
            continue
        members = [
            allowed[m]
            for m in (raw_group.get("members") or [])
            if isinstance(m, str) and m in allowed and m not in seen
        ]
        if not members:
            continue
        seen.update(members)
        canonical = str(raw_group.get("canonical", "")).strip()
        if canonical not in members:
            # Trust the members, not a canonical the model may have paraphrased.
            canonical = max(members, key=len)
        groups.append(
            {
                "canonical": canonical,
                "members": members,
                "reason": str(raw_group.get("reason", "")).strip(),
            }
        )

    for name in cluster:
        if name not in seen:
            groups.append({"canonical": name, "members": [name], "reason": "unmerged"})
    return groups


def run(
    triples: list[Triple],
    *,
    model: str | None = None,
    verbose: bool = True,
    use_llm: bool = True,
) -> ResolutionMap:
    """Resolve every entity name appearing in `triples`."""
    model = model or model_for("resolve")

    descriptions: dict[str, set[str]] = {}
    type_votes: dict[str, Counter[str]] = {}
    for t in triples:
        for name, desc, entity_type in (
            (t.subject, t.subject_description, t.subject_type),
            (t.object, t.object_description, t.object_type),
        ):
            name = name.strip()
            bucket = descriptions.setdefault(name, set())
            if desc:
                bucket.add(desc)
            type_votes.setdefault(name, Counter())[entity_type] += 1

    result = ResolutionMap()
    for cluster in candidate_blocks(list(descriptions)):
        # Nothing to arbitrate: a lone name, or names that are literally the
        # same string once normalized.
        if len(cluster) == 1 or len({normalize(c) for c in cluster}) == 1:
            canonical = max(cluster, key=len)
            groups = [{"canonical": canonical, "members": cluster, "reason": "exact"}]
        elif not use_llm:
            groups = [{"canonical": c, "members": [c], "reason": "llm-disabled"} for c in cluster]
        else:
            payload = json.dumps(
                [
                    {"name": c, "descriptions": sorted(descriptions.get(c, ()))}
                    for c in cluster
                ],
                indent=2,
            )
            try:
                # See openrouter.py's DEFAULT_MODELS comment: a reasoning-capable
                # model with too tight a budget spends it all on reasoning and
                # returns empty content — hit for real on `extract`. Unlike
                # extract, this stage is deliberately NOT told to disable
                # reasoning: judgment is the whole reason resolve is allowed to
                # be the expensive stage. The budget is widened instead, so
                # reasoning and the actual JSON both fit.
                completion = complete(
                    prompts.RESOLVE_SYSTEM,
                    prompts.RESOLVE_USER.format(candidates=payload),
                    model,
                    max_tokens=3072,
                )
                groups = parse_resolution_response(completion.text, cluster)
            except Exception as exc:
                # Failing closed means every name stays its own entity: a set of
                # missed merges, which is the recoverable direction.
                if verbose:
                    print(f"  resolve failed for {cluster}: {exc} — leaving unmerged")
                groups = [{"canonical": c, "members": [c], "reason": "error"} for c in cluster]

        for group in groups:
            result.groups.append(group)
            # One canonical type per canonical name, by majority vote across
            # every occurrence in the group. Without this, mentions that
            # resolve merges by NAME can still land on different node ids,
            # because canonical_id also keys on type — the merge would be
            # real in resolution.json and silently absent from the graph.
            votes: Counter[str] = Counter()
            for member in group["members"]:
                votes.update(type_votes.get(member, Counter()))
            if votes:
                result.types[normalize(group["canonical"])] = votes.most_common(1)[0][0]
            for member in group["members"]:
                result.mapping[normalize(member)] = group["canonical"]
            if verbose and len(group["members"]) > 1:
                print(f"  merged {group['members']} -> {group['canonical']}")
                if len(votes) > 1:
                    print(f"    type votes {dict(votes)} -> {result.types[normalize(group['canonical'])]}")

    return result
