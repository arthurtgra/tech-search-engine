"""Collector arXiv via Atom API (export.arxiv.org). Rate-limit ~3s."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

import feedparser

from tie.collectors.base import Collector
from tie.models import RawDocument, Source

ARXIV_API = "http://export.arxiv.org/api/query"


class ArxivCollector(Collector):
    source_name = "arxiv"

    @property
    def min_interval(self) -> float:
        return self.settings.arxiv_min_interval_s

    def collect(self, query: str, *, max_results: int = 100, since_days: int = 365) -> Iterable[RawDocument]:
        cats = " OR ".join(f"cat:{c}" for c in self.settings.arxiv_categories)
        params = {
            "search_query": f"(all:{query}) AND ({cats})",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        raw = self.fetch(ARXIV_API, params=params)
        feed = feedparser.parse(raw)
        for e in feed.entries:
            published = None
            if getattr(e, "published_parsed", None):
                published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            yield RawDocument(
                source=Source.ARXIV,
                source_id=e.get("id", "").split("/abs/")[-1],
                title=e.get("title", "").replace("\n", " ").strip(),
                text=e.get("summary", "").replace("\n", " ").strip(),
                url=e.get("link", ""),
                published_at=published,
                metric=0.0,
                metric_name="citations",
                extra={"authors": [a.get("name", "") for a in e.get("authors", [])]},
            )
