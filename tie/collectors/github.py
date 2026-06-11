"""Collector GitHub via Search API. Métrica = stars (proxy de momentum)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from dateutil import parser as dtparser

from tie.collectors.base import Collector
from tie.models import RawDocument, Source

GITHUB_SEARCH = "https://api.github.com/search/repositories"


class GithubCollector(Collector):
    source_name = "github"

    def collect(self, query: str, *, max_results: int = 100, since_days: int = 365) -> Iterable[RawDocument]:
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
