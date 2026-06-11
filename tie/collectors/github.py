"""Collector GitHub via Search API. Métrica = stars (proxy de momentum)."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from dateutil import parser as dtparser

from tie.collectors.base import Collector
from tie.models import RawDocument, Source

GITHUB_SEARCH = "https://api.github.com/search/repositories"

# Em algumas redes o DNS resolve api.github.com para um IP Azure blackholed
# (ex.: 4.228.31.149). Os IPs clássicos do GitHub respondem normalmente, então
# fixamos a resolução via override de socket.getaddrinfo — SNI/cert continuam
# usando o hostname original. Ver memória github-api-blocked.
_GITHUB_FALLBACK_IPS = ["140.82.112.6", "140.82.113.6", "140.82.121.6", "140.82.114.6"]
_real_getaddrinfo = socket.getaddrinfo
_pinned: dict[str, str] = {}


def _patched_getaddrinfo(host, *args, **kwargs):  # noqa: ANN001, ANN002
    if host == "api.github.com" and host in _pinned:
        host = _pinned[host]
    return _real_getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = _patched_getaddrinfo


def _pin_github_dns() -> str | None:
    """Acha o primeiro IP alcançável do GitHub e fixa api.github.com nele."""
    if "api.github.com" in _pinned:
        return _pinned["api.github.com"]
    for ip in _GITHUB_FALLBACK_IPS:
        try:
            sock = socket.create_connection((ip, 443), timeout=3)
            sock.close()
            _pinned["api.github.com"] = ip
            return ip
        except OSError:
            continue
    return None


class GithubCollector(Collector):
    source_name = "github"

    def collect(self, query: str, *, max_results: int = 100, since_days: int = 365) -> Iterable[RawDocument]:
        if _pin_github_dns() is None:
            raise ConnectionError("nenhum IP do GitHub alcançável nesta rede")
        since = (datetime.now(timezone.utc) - timedelta(days=since_days)).date().isoformat()
        params = {
            "q": f"{query} pushed:>{since}",
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results, 100),
        }
        headers = {"Accept": "application/vnd.github+json"}
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        raw = self.fetch(GITHUB_SEARCH, params=params, headers=headers)
        data = json.loads(raw)
        for repo in data.get("items", []):
            published = None
            if repo.get("created_at"):
                published = dtparser.parse(repo["created_at"])
            desc = repo.get("description") or ""
            topics = repo.get("topics", [])
            yield RawDocument(
                source=Source.GITHUB,
                source_id=repo["full_name"],
                title=repo["full_name"],
                text=f"{desc}\nTopics: {', '.join(topics)}",
                url=repo.get("html_url", ""),
                published_at=published,
                metric=float(repo.get("stargazers_count", 0)),
                metric_name="stars",
                extra={"language": repo.get("language"), "topics": topics},
            )
