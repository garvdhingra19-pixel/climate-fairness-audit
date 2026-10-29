"""OpenAQ v3 station-metadata client.

The API key is read from the OPENAQ_API_KEY environment variable (never hard-code or log it).
Parsing is separate from HTTP so it can be tested offline. NOTE: field names follow the public
v3 docs but were not verified against the live API in the build sandbox; confirm with one real
call and adjust `parse_locations` if the payload differs.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

import pandas as pd

BASE_URL = "https://api.openaq.org/v3"


def _default_fetch(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Only https URLs are allowed.")
    req = urllib.request.Request(url, headers=headers)  # noqa: S310 - scheme checked above
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


class OpenAQClient:
    def __init__(
        self,
        api_key: str | None = None,
        fetch: Callable[[str, dict[str, str], float], dict[str, Any]] | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 1.5,
    ) -> None:
        self._key = api_key or os.environ.get("OPENAQ_API_KEY")
        if not self._key:
            raise ValueError("Set OPENAQ_API_KEY (see .env.example) or pass api_key.")
        self._fetch = fetch or _default_fetch
        self.timeout, self.retries, self.backoff = timeout, retries, backoff

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE_URL}/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"
        headers = {"X-API-Key": self._key, "Accept": "application/json"}
        for attempt in range(self.retries):
            try:
                return self._fetch(url, headers, self.timeout)
            except Exception:
                if attempt == self.retries - 1:
                    raise  # never include headers in the error
                time.sleep(self.backoff**attempt)
        raise RuntimeError("unreachable")

    def get_locations(
        self, countries_id: int, limit: int = 1000, max_pages: int = 50, **params: Any
    ) -> pd.DataFrame:
        """Page through /locations for a country. countries_id: look it up once via /countries."""
        results: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            payload = self._get(
                "locations", {"countries_id": countries_id, "limit": limit, "page": page, **params}
            )
            batch = payload.get("results", [])
            results.extend(batch)
            if len(batch) < limit:
                break
        return parse_locations({"results": results})


def parse_locations(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten a v3 /locations payload into station_id, name, lat, lon, parameters, first/last."""
    rows = []
    for loc in payload.get("results", []):
        coords = loc.get("coordinates") or {}
        params = sorted(
            {(s.get("parameter") or {}).get("name") for s in loc.get("sensors", [])} - {None}
        )
        rows.append(
            {
                "station_id": f"openaq_{loc.get('id')}",
                "name": loc.get("name"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "parameters": ",".join(params),
                "has_pm25": "pm25" in params,
                "first_utc": (loc.get("datetimeFirst") or {}).get("utc"),
                "last_utc": (loc.get("datetimeLast") or {}).get("utc"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "name",
            "lat",
            "lon",
            "parameters",
            "has_pm25",
            "first_utc",
            "last_utc",
        ],
    )
