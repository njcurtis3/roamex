"""Live model catalog from OpenRouter.

Model ids and prices change often enough that hardcoding a "cheap model" list
into this repo would make it wrong within months. So the list is fetched, and
`src.cli models` prints it sorted by price. Needs no API key — the catalog
endpoint is public.

`estimate()` turns a dry-run's token counts into an actual dollar figure for a
specific model, which is the number worth looking at before running extraction
over a real graph.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

CATALOG_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class ModelInfo:
    id: str
    name: str
    prompt_per_mtok: float
    completion_per_mtok: float
    context: int

    @property
    def is_free(self) -> bool:
        return self.prompt_per_mtok == 0 and self.completion_per_mtok == 0

    @property
    def is_priced(self) -> bool:
        """OpenRouter's routing pseudo-models (`openrouter/auto`, `fusion`, …)
        report a negative price meaning "depends which model it picks". Sorting
        by price puts them first, which is the opposite of true."""
        return self.prompt_per_mtok >= 0 and self.completion_per_mtok >= 0


def fetch(timeout: int = 30) -> list[ModelInfo]:
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "roamex"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    out: list[ModelInfo] = []
    for m in body.get("data", []):
        pricing = m.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt", 0)) * 1_000_000
            completion = float(pricing.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            continue
        out.append(
            ModelInfo(
                id=m.get("id", ""),
                name=m.get("name", ""),
                prompt_per_mtok=prompt,
                completion_per_mtok=completion,
                context=int(m.get("context_length") or 0),
            )
        )
    return out


def cheapest(models: list[ModelInfo], limit: int = 20, include_free: bool = False):
    """Sorted by blended cost, assuming roughly 4:1 input:output — the shape of
    an extraction call, which sends a long system prompt and gets a short JSON
    array back."""
    pool = [m for m in models if m.id and m.is_priced]
    pool = [m for m in pool if include_free or not m.is_free]
    return sorted(pool, key=lambda m: m.prompt_per_mtok * 4 + m.completion_per_mtok)[:limit]


def find(models: list[ModelInfo], model_id: str) -> ModelInfo | None:
    return next((m for m in models if m.id == model_id), None)


def estimate(
    info: ModelInfo, *, calls: int, input_tokens: int, output_tokens_per_call: int = 150
) -> dict:
    """Dollar estimate for a stage. Rough, and deliberately rounded up-ish:
    a cost preview that under-reports is worse than no preview."""
    out_tokens = calls * output_tokens_per_call
    cost_in = input_tokens / 1_000_000 * info.prompt_per_mtok
    cost_out = out_tokens / 1_000_000 * info.completion_per_mtok
    return {
        "model": info.id,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens_est": out_tokens,
        "cost_input_usd": round(cost_in, 4),
        "cost_output_usd": round(cost_out, 4),
        "total_usd": round(cost_in + cost_out, 4),
    }
