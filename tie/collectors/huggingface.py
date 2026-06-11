"""Collector Hugging Face via API pública de modelos. Métrica = downloads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime

from dateutil import parser as dtparser

from tie.collectors.base import Collector
from tie.models import RawDocument, Source

HF_MODELS = "https://huggingface.co/api/models"

# A busca da HF é substring no id do modelo — frases multi-palavra quase nunca
# casam. Ignoramos termos genéricos e consultamos por token, mesclando resultados.
_STOPWORDS = {"how", "are", "the", "with", "for", "and", "como", "estão", "criando",
              "techniques", "models", "model", "using", "based", "ai", "a", "o"}


def _keywords(query: str) -> list[str]:
    toks = [t.strip(".,?!\"'").lower() for t in query.split()]
    kept = [t for t in toks if len(t) > 2 and t not in _STOPWORDS]
    return kept or toks[:1]


class HuggingFaceCollector(Collector):
    source_name = "huggingface"

    def collect(self, query: str, *, max_results: int = 100, since_days: int = 365) -> Iterable[RawDocument]:
        per_term = max(10, max_results // max(1, len(_keywords(query))))
        seen_ids: set[str] = set()
        data: list[dict] = []
        for term in _keywords(query):
            params = {
                "search": term, "sort": "downloads", "direction": -1,
                "limit": per_term, "full": "true",
            }
            try:
                for m in json.loads(self.fetch(HF_MODELS, params=params)):
                    if m["id"] not in seen_ids:
                        seen_ids.add(m["id"])
                        data.append(m)
            except Exception:
                continue
        for m in data[:max_results]:
            published: datetime | None = None
            if m.get("createdAt"):
                published = dtparser.parse(m["createdAt"])
            tags = m.get("tags", [])
            card = ""
            if isinstance(m.get("cardData"), dict):
                card = str(m["cardData"].get("model_summary", ""))
            yield RawDocument(
                source=Source.HUGGINGFACE,
                source_id=m["id"],
                title=m["id"],
                text=f"{card}\nTags: {', '.join(str(t) for t in tags)}",
                url=f"https://huggingface.co/{m['id']}",
                published_at=published,
                metric=float(m.get("downloads", 0)),
                metric_name="downloads",
                extra={"pipeline_tag": m.get("pipeline_tag"), "likes": m.get("likes", 0)},
            )
