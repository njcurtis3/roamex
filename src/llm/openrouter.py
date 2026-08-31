"""The only place this app talks to a network.

Stdlib `urllib` on purpose: one POST to one JSON endpoint does not justify a
dependency. Every stage that uses a model calls through here, and every stage
that uses a model also exposes a pure `parse_*_response()` function beside it,
so the whole pipeline is provable offline against recorded fixtures.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tier by job, not by habit. Extraction is high-volume and mechanical — it reads
# one block and fills a fixed schema, and the pipeline independently verifies
# every triple it returns by checking the quote against the source block. A
# cheap model that occasionally writes junk is therefore *affordable* here: the
# junk gets dropped rather than believed. Resolution is the opposite — low
# volume, and a wrong call silently welds two entities together with nothing
# downstream able to detect it. Query has to reason over a subgraph and cite it
# honestly. Spend accordingly.
#
# These are cheap defaults chosen for a first real run. They are a starting
# point, not a recommendation: `src.cli models` prints the live catalog, and
# `src.cli eval` is how you find out whether a given model is good enough for
# YOUR notes. Do not trust a cheap model here on the strength of this comment —
# measure it. Override per stage in .env.
#
# GOTCHA, found running the real graph: qwen/qwen3.7-flash is a reasoning
# model. With no `reasoning` override it spent the entire max_tokens budget on
# reasoning tokens and returned empty content — 25 of 59 blocks failed, and the
# rest were truncated JSON. That looked exactly like a broken extractor and was
# actually a config gap. `extract.run()` now passes `reasoning={"enabled":
# False}` (1.7s/211 tokens per block vs. 29s/5k). If you swap in another
# reasoning-capable model for any stage, either pass the same override or raise
# `max_tokens` enough to leave room for the actual output after reasoning.
DEFAULT_MODELS = {
    "extract": "qwen/qwen3.7-flash",
    "resolve": "google/gemini-3.6-flash",
    "query": "google/gemini-3.6-flash",
}


def model_for(stage: str) -> str:
    return os.environ.get(f"ROAMEX_MODEL_{stage.upper()}", DEFAULT_MODELS[stage])


class OpenRouterError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def complete(
    system: str,
    user: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
    retries: int = 3,
    reasoning: dict[str, object] | None = None,
) -> Completion:
    """One chat completion.

    Raises on a missing key rather than returning empty: an unconfigured run
    that produces zero extracted relations is indistinguishable from a corpus
    that genuinely had none, and that is the failure this app can least afford
    to swallow.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/njcurtis3/roamex",
        "X-Title": "roamex",
    }

    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            err = body.get("error")
            # A transient upstream failure can come back as a 200 with the
            # error embedded in the body instead of an HTTP error status —
            # seen for real as {'message': 'The operation was aborted',
            # 'code': 504} during an OpenRouter rate-limit window. That never
            # raises HTTPError, so it used to skip retry entirely and burn
            # the block permanently on the very first hiccup. Route it
            # through the same backoff as a 429 instead.
            if err and _is_retryable_body_error(err):
                last = OpenRouterError(f"transient upstream error: {err}")
            else:
                return _completion_from(body, model)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            # 4xx other than rate-limit is a bad request; retrying re-sends the
            # same bad request and just costs wall-clock.
            if exc.code != 429 and 400 <= exc.code < 500:
                raise OpenRouterError(f"HTTP {exc.code}: {detail}") from exc
            last = OpenRouterError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last = OpenRouterError(f"network error: {exc}")
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise last or OpenRouterError("request failed")


def _is_retryable_body_error(err: object) -> bool:
    """Is an error embedded in a 200 response body transient?

    Only 429/5xx-shaped and abort/timeout-worded errors get the retry
    budget — a genuinely bad request (bad model id, malformed payload) would
    just fail the same way three times and cost wall-clock for nothing.
    """
    if not isinstance(err, dict):
        return False
    code = err.get("code")
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True
    message = str(err.get("message", "")).lower()
    return any(w in message for w in ("abort", "timeout", "timed out", "rate"))


def _completion_from(body: dict, model: str) -> Completion:
    """Pure: an OpenRouter response body -> Completion. Testable from fixtures."""
    if "error" in body and body["error"]:
        raise OpenRouterError(str(body["error"]))
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"unexpected response shape: {json.dumps(body)[:400]}") from exc
    usage = body.get("usage") or {}
    return Completion(
        text=text or "",
        model=body.get("model", model),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )


def extract_json(text: str) -> object:
    """Pull the JSON payload out of a model reply.

    Models fence JSON, prefix it with prose, or both, even under instruction not
    to. Tolerating that here is cheaper than a retry loop and keeps a formatting
    quirk from being recorded as an extraction failure by the eval harness.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try the OUTERMOST structure first. Scanning for `[` before `{` would
    # happily return the inner `"citations": [...]` array out of a reply whose
    # real payload is the object wrapping it — a wrong parse that succeeds.
    candidates = []
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append((start, text[start : end + 1]))
    for _, blob in sorted(candidates):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    # A response cut off mid-object by max_tokens has no closing `]` at all,
    # so it never made it into `candidates` above and would otherwise be
    # thrown away along with every complete object that came before the cut.
    # Salvage those: an extractor call fails or succeeds per block already,
    # so returning the objects that did complete is strictly better than
    # discarding a whole block's output over its last, truncated one.
    start = text.find("[")
    if start != -1:
        healed = _heal_truncated_array(text[start:])
        if healed:
            return healed
    raise ValueError(f"no JSON found in model reply: {text[:200]!r}")


def _heal_truncated_array(text: str) -> list | None:
    """Recover the complete leading objects from a truncated `[...]` array.

    Scans by brace depth (with string/escape awareness) rather than trying to
    fix the trailing fragment itself — a half-written string or object is not
    reliably repairable, but everything closed *before* it is exact JSON.
    """
    depth = 0
    in_string = False
    escape = False
    obj_start = None
    objects = []
    for i, ch in enumerate(text):
        if i == 0 and ch == "[":
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    objects.append(json.loads(text[obj_start : i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return objects or None


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader. Real environment always wins over the file."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except FileNotFoundError:
        pass
