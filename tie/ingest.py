"""Ingestão: RawDocument -> Document (com dedupe por content_hash)."""

from __future__ import annotations

from tie.collectors import build_collectors
from tie.config import Settings
from tie.db import get_session
from tie.models import Document, RawDocument


def _to_document(raw: RawDocument) -> Document:
    return Document(
        source=raw.source.value,
        source_id=raw.source_id,
        title=raw.title,
        text=raw.text,
        url=raw.url,
        published_at=raw.published_at.replace(tzinfo=None) if raw.published_at else None,
        metric=raw.metric,
        metric_name=raw.metric_name,
        content_hash=raw.content_hash(),
    )


def collect_and_store(
    settings: Settings, query: str, *, max_results: int, since_days: int
) -> dict[str, int]:
    """Roda todos os collectors e persiste novos documentos. Retorna contagem por fonte."""
    counts: dict[str, int] = {}
    with get_session() as session:
        known = {h for (h,) in session.query(Document.content_hash).all()}
        for collector in build_collectors(settings):
            new = 0
            try:
                docs = collector.collect(query, max_results=max_results, since_days=since_days)
                for raw in docs:
                    h = raw.content_hash()
                    if h in known:
                        continue
                    known.add(h)
                    session.add(_to_document(raw))
                    new += 1
            except Exception as exc:  # uma fonte falhar não derruba as outras
                counts[f"{collector.source_name}:error"] = 0
                print(f"[warn] collector {collector.source_name} falhou: {exc}")
            counts[collector.source_name] = new
            session.flush()
    return counts
