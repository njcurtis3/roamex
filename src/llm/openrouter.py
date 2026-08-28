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
# one block and fills a fixed schema. Resolution is low-volume and genuinely
# judgment-shaped: deciding two names are one entity is the call that, made
# wrong, silently corrupts the graph. Query has to reason over a subgraph and
# cite it. Spend accordingly; override any of these in .env.
DEFAULT_MODELS = {
    "extract": "anthropic/claude-haiku-4.5",
    "resolve": "anthropic/claude-sonnet-4.5",
    "query": "anthropic/claude-sonnet-4.5",
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
    raise ValueError(f"no JSON found in model reply: {text[:200]!r}")


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
