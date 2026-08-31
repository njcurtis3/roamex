"""Stage 2 — block prose -> candidate triples, via a model.

This is the stage that earns the pipeline. `parse` already captured every
relation the author bothered to write as a [[link]]; this reads the sentences
where they didn't. "Started at Acme in 2019" with no link to a page named Acme
is invisible to the base graph and is exactly what belongs here.

The seam: `run()` does network, `parse_extraction_response()` does not. Every
test in this app exercises the second one.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..llm import prompts
from ..llm.openrouter import complete, extract_json, model_for
from ..models import ORIGIN_LLM, Provenance, Triple


def parse_extraction_response(
    raw_text: str,
    *,
    block_uid: str,
    page_title: str,
    block_text: str,
    model: str,
    now: str | None = None,
) -> list[Triple]:
    """Model reply -> Triples. Pure; no network.

    Drops anything malformed rather than raising: one bad element in a batch of
    fifty should not lose the other forty-nine. What it will not do is drop a
    *quote* check — a triple whose quote is not in the block is the model
    asserting something the note does not say, and admitting it would put an
    unfalsifiable fact in the graph.
    """
    now = now or datetime.now(timezone.utc).isoformat()
    data = extract_json(raw_text)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}")

    haystack = " ".join(block_text.split()).casefold()
    triples: list[Triple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        obj = str(item.get("object", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        if not (subject and obj and predicate) or subject.casefold() == obj.casefold():
            continue

        quote = str(item.get("quote", "")).strip()
        if quote and " ".join(quote.split()).casefold() not in haystack:
            # Ungrounded: the model produced a quote the block does not contain.
            continue

        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        triples.append(
            Triple(
                subject=subject,
                subject_type=_valid_type(item.get("subject_type")),
                predicate="_".join(predicate.lower().split()),
                object=obj,
                object_type=_valid_type(item.get("object_type")),
                subject_description=str(item.get("subject_description", "")).strip(),
                object_description=str(item.get("object_description", "")).strip(),
                confidence=max(0.0, min(1.0, confidence)),
                provenance=Provenance(
                    block_uid=block_uid,
                    page_title=page_title,
                    origin=ORIGIN_LLM,
                    extracted_at=now,
                    model=model,
                    prompt_version=prompts.EXTRACT_VERSION,
                    quote=quote or None,
                ),
            )
        )
    return triples


def _valid_type(raw: object) -> str:
    t = str(raw or "").strip().lower()
    return t if t in prompts.ENTITY_TYPES else "concept"


def run(
    blocks: list[dict[str, str]],
    *,
    model: str | None = None,
    verbose: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
):
    """Extract over a list of blocks from `roam.parse.blocks_for_extraction`.

    One call per block. Batching several blocks into one call is cheaper and is
    the obvious next optimization — it is not done yet because per-block calls
    keep provenance trivially correct, and getting provenance wrong is the one
    error this app cannot recover from later.

    If `checkpoint_path` is given, progress is flushed there every
    `checkpoint_every` blocks. Extraction costs real, non-refundable money per
    call — a run killed from outside (a closed session, a sleeping machine,
    not a code error) used to lose every call made so far, because nothing
    was written to disk until the very end. `cli.cmd_extract` reads this file
    back to resume rather than re-pay for already-completed blocks.
    """
    model = model or model_for("extract")
    triples: list[Triple] = []
    failures: list[dict[str, str]] = []
    done_uids: list[str] = []

    for i, block in enumerate(blocks, 1):
        user = prompts.EXTRACT_USER.format(
            page_title=block["page_title"], text=block["text"]
        )
        try:
            completion = complete(
                prompts.EXTRACT_SYSTEM,
                user,
                model,
                max_tokens=4096,
                reasoning={"enabled": False},
            )
            got = parse_extraction_response(
                completion.text,
                block_uid=block["uid"],
                page_title=block["page_title"],
                block_text=block["text"],
                model=completion.model,
            )
            triples.extend(got)
            if verbose:
                print(f"  [{i}/{len(blocks)}] {block['uid']}: {len(got)} triples")
        except Exception as exc:  # one bad block must not end the run
            failures.append({"uid": block["uid"], "error": str(exc)})
            if verbose:
                print(f"  [{i}/{len(blocks)}] {block['uid']}: FAILED — {exc}")
        done_uids.append(block["uid"])

        if checkpoint_path and (i % checkpoint_every == 0 or i == len(blocks)):
            _write_checkpoint(checkpoint_path, triples, failures, done_uids)

    return triples, failures


def _write_checkpoint(
    path: Path,
    triples: list[Triple],
    failures: list[dict[str, str]],
    done_uids: list[str],
) -> None:
    path.write_text(
        json.dumps(
            {
                "done_uids": done_uids,
                "triples": [asdict(t) for t in triples],
                "failures": failures,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_checkpoint(path: Path) -> tuple[list[Triple], list[dict[str, str]], set[str]]:
    """Read back a checkpoint written by `run()`. Pure; no network."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    triples = [
        Triple(**{**t, "provenance": Provenance(**t["provenance"])})
        for t in raw["triples"]
    ]
    return triples, raw["failures"], set(raw["done_uids"])
