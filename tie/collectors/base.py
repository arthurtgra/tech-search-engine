"""Base de collectors: cache em disco + cliente HTTP com rate-limit.

O cache evita rebater APIs (e respeita ToS) durante o desenvolvimento iterativo.
Cada resposta crua é gravada por chave; TTL controla o frescor.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import httpx

from tie.config import CACHE_DIR, Settings
from tie.models import RawDocument


class HttpCache:
    def __init__(self, ttl_hours: float = 24.0) -> None:
        self.ttl_s = ttl_hours * 3600

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return CACHE_DIR / f"{h}.json"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        if not p.exists() or (time.time() - p.stat().st_mtime) > self.ttl_s:
            return None
        return p.read_text(encoding="utf-8")

    def set(self, key: str, value: str) -> None:
        self._path(key).write_text(value, encoding="utf-8")


class Collector(ABC):
    source_name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = HttpCache()
        self._last_request = 0.0

    @property
    def min_interval(self) -> float:
        return 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def fetch(self, url: str, params: dict | None = None, headers: dict | None = None) -> str:
        cache_key = url + "?" + json.dumps(params or {}, sort_keys=True)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        self._throttle()
        h = {"User-Agent": self.settings.user_agent}
        if headers:
            h.update(headers)
        resp = httpx.get(
            url, params=params, headers=h, timeout=self.settings.http_timeout_s,
            follow_redirects=True,
        )
        resp.raise_for_status()
        self.cache.set(cache_key, resp.text)
        return resp.text

    @abstractmethod
    def collect(self, query: str, *, max_results: int, since_days: int) -> Iterable[RawDocument]:
        ...
