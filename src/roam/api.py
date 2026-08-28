"""Pull a graph from Roam's official Backend API instead of a manual export.

Roam exposes a documented backend API (Datalog/Datomic-style pull queries)
at https://api.roamresearch.com. This module speaks it with stdlib
`urllib` — same "no SDK for one endpoint" convention as llm/openrouter.py —
and returns EXACTLY the shape `roam.parse.load_export()` returns from a
downloaded JSON file: a list of page dicts, each with `title`/`uid` and a
recursively-nested `children` list of block dicts with `uid`/`string`.
Nothing downstream of that shape changes; this is a second way to produce
the same input, not a parallel pipeline.

READ THIS BEFORE TRUSTING IT AGAINST A REAL GRAPH.
The endpoint paths, auth headers, and peer-redirect handling below are
sourced from a community client's published source (2b3pro/Roam-Graph-API,
scripts/roam_backend.py) — Roam's own docs sit behind an in-app page this
tool cannot reach, so this was never cross-checked against the primary
source. The Datalog attribute names (:node/title, :block/uid,
:block/string, :block/children, :block/order) are corroborated across
several independent community sources, which is reasonable but not proof.
What is GENUINELY UNTESTED is the exact JSON key spelling a pulled entity
map serializes to ("block/uid" vs ":block/uid" vs something else) — nobody
publishes that detail, and there is no way to know without a live call.
`_get()` below tries several plausible spellings and RAISES with the
actual keys found if none match, so the first real run fails loud with a
diagnosable error instead of silently returning an empty or wrong graph.
That first real run is the actual test of this module — do not treat it
as working before it has been run once against real credentials.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.roamresearch.com"

# How many levels of block nesting to request in one query. Unrolled
# explicitly rather than using Datomic's unbounded recursive-pull symbol
# (`...`) — Roam's backend may be DataScript rather than full Datomic, where
# recursive-pull support is less certain, and an explicit bound also caps
# how much one pathologically nested page can blow up a single response.
# 20 covers ordinary Roam usage; pass depth= to fetch_graph() to override.
DEFAULT_DEPTH = 20

# create/edit time and user are in the manual JSON export but genuinely not
# needed: roam.parse only ever reads title/uid/string/children (verified by
# grep). Every field requested here is a field something downstream reads —
# no "grab everything in case it's useful" bloat on an API this unverified.
BLOCK_FIELDS = "[:block/uid :block/string :block/order"


class RoamAPIError(RuntimeError):
    pass


def _selector(depth: int) -> str:
    """Build a bounded-depth Datalog pull selector, innermost level first."""
    node = BLOCK_FIELDS + "]"
    for _ in range(depth):
        node = f"{BLOCK_FIELDS} {{:block/children {node}}}]"
    return node


class RoamClient:
    """Talks to one Roam graph's backend API.

    Handles the one wire-level detail that will silently break a naive
    client: Roam's backend redirects the first request for a graph to a
    dedicated `peer-N.api.roamresearch.com:PORT` host, and a client that
    follows redirects automatically (as urllib does by default) will
    typically re-send the request WITHOUT the Authorization header, since
    that is standard cross-origin-redirect security behavior — turning a
    real 401 into a confusing one. This follows the redirect manually,
    with the header intact, and caches the peer host for later calls.
    """

    def __init__(self, graph: str, token: str, timeout: int = 60, retries: int = 4) -> None:
        if not graph or not token:
            raise RoamAPIError("both graph name and API token are required")
        self.graph = graph
        self._token = token
        self.timeout = timeout
        self.retries = retries
        self._base_url = BASE_URL

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self._token}",
            "x-authorization": f"Bearer {self._token}",
        }

    def query(self, datalog: str) -> list:
        """Run a Datalog query, return the `result` array."""
        path = f"/api/graph/{self.graph}/q"
        body = json.dumps({"query": datalog}).encode("utf-8")
        data = self._post(path, body)
        try:
            return data["result"]
        except KeyError as exc:
            raise RoamAPIError(
                f"query response had no 'result' key. Keys present: {list(data)}"
            ) from exc

    def _post(self, path: str, body: bytes) -> dict:
        url = self._base_url + path
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    location = exc.headers.get("Location") if exc.headers else None
                    peer = self._extract_peer(location)
                    if peer:
                        url = peer + path
                        self._base_url = peer  # cache for subsequent calls
                        continue
                    raise RoamAPIError(f"redirected with no usable Location header: {location!r}")
                if exc.code == 401:
                    raise RoamAPIError("401 Unauthorized — check ROAM_API_TOKEN") from exc
                if exc.code == 403:
                    raise RoamAPIError("403 Forbidden — token lacks access to this graph") from exc
                if exc.code == 404:
                    raise RoamAPIError(f"404 — graph {self.graph!r} not found") from exc
                if exc.code in (429, 503):
                    # 429 rate-limited, 503 "graph not ready" (documented as
                    # a cold-start case in the reference client) — both
                    # worth a backoff retry, not an immediate failure.
                    last = RoamAPIError(f"HTTP {exc.code}, retrying")
                    time.sleep(2**attempt)
                    continue
                detail = exc.read().decode("utf-8", "replace")[:400]
                raise RoamAPIError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = RoamAPIError(f"network error: {exc}")
                time.sleep(2**attempt)
        raise last or RoamAPIError("request failed after retries")

    @staticmethod
    def _extract_peer(location: str | None) -> str | None:
        if not location:
            return None
        m = re.search(r"https://(peer-\d+[^/:]*)(?::(\d+))?", location)
        if not m:
            return None
        host, port = m.groups()
        return f"https://{host}:{port}" if port else f"https://{host}"


def _get(d: dict, *keys: str, required: bool = True):
    """Look up a Datomic-pulled field under whichever key spelling the
    server actually used. See the module docstring: this is the one part
    of the wire format that could not be verified without a live call, so
    it fails loud with the real keys rather than guessing wrong silently."""
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise RoamAPIError(
            f"expected one of {keys!r} in a pulled entity, found keys {list(d)}. "
            "The API's JSON key format did not match what this module assumed — "
            "see api.py's module docstring."
        )
    return None


def convert_pulled_pages(raw_result: list) -> list[dict]:
    """Datomic pull results -> the same page/children shape load_export()
    returns from a manual JSON export. Pure; no network — this is what a
    fixture-based test exercises, since the live query cannot be tested
    offline.

    `raw_result` is the API's `result` array: one row per page, each row a
    1-element list `[pulled_entity_map]` (the :find clause has one
    variable). Order of `:block/children` is not guaranteed by Datalog, so
    every level is explicitly sorted by `:block/order`.
    """
    pages: list[dict] = []
    for row in raw_result:
        if not row:
            continue
        entity = row[0]
        title = _get(entity, "node/title", ":node/title", "title", required=False)
        if title is None:
            continue  # an entity with no title is not a page
        uid = _get(entity, "block/uid", ":block/uid", "uid", required=False) or title
        children = _get(entity, "block/children", ":block/children", "children", required=False) or []
        pages.append({
            "title": title,
            "uid": uid,
            "children": _convert_blocks(children),
        })
    return pages


def _convert_blocks(raw_blocks: list) -> list[dict]:
    def order_of(b: dict):
        return _get(b, "block/order", ":block/order", "order", required=False) or 0

    out = []
    for b in sorted(raw_blocks, key=order_of):
        uid = _get(b, "block/uid", ":block/uid", "uid", required=False)
        string = _get(b, "block/string", ":block/string", "string", required=False) or ""
        if not uid:
            continue
        children = _get(b, "block/children", ":block/children", "children", required=False) or []
        out.append({
            "uid": uid,
            "string": string,
            "children": _convert_blocks(children),
        })
    return out


def fetch_graph(client: RoamClient, depth: int = DEFAULT_DEPTH) -> list[dict]:
    """One graph, one query. Returns the load_export()-compatible page list.

    Every page with a `:node/title`, each pulled together with its block
    tree up to `depth` levels of nesting, in a single round trip.
    """
    selector = f"[:node/title :block/uid {{:block/children {_selector(depth)}}}]"
    datalog = f"[:find (pull ?p {selector}) :where [?p :node/title]]"
    raw = client.query(datalog)
    pages = convert_pulled_pages(raw)
    if not pages:
        # A real graph has at least one page. Coming back empty means
        # something in the query or the key-format assumptions is wrong,
        # not that the graph is genuinely empty — say so.
        raise RoamAPIError(
            "query returned 0 pages. Either the graph is genuinely empty, or "
            "the response's key format didn't match this module's assumptions "
            f"(raw result had {len(raw)} rows) — inspect it before trusting "
            "an empty pull."
        )
    return pages


def pull_and_save(graph: str, token: str, out_path: str, depth: int = DEFAULT_DEPTH) -> int:
    """The CLI-facing entry point: fetch, then write load_export()-shaped
    JSON to out_path. Returns the page count."""
    client = RoamClient(graph, token)
    pages = fetch_graph(client, depth=depth)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=2)
    return len(pages)
