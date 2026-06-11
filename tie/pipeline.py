"""Orquestração do pipeline de investigação, reutilizável por CLI e UI.

Centraliza as 5 etapas (coleta → extração → resolução → índice → análise) numa
função única com callback opcional de progresso.
"""

from __future__ import annotations

from collections.abc import Callable

from tie.analytics import build_relations, rank_entities, top_cooccurrences
from tie.config import Settings
from tie.extract import extract_corpus
from tie.index import build_index, semantic_search
from tie.ingest import collect_and_store
from tie.models import EntityType
from tie.report import TIEReport, build_report
from tie.resolve import resolve_entities

SECTION_TYPES: list[tuple[str, EntityType]] = [
    ("Modelos", EntityType.MODEL),
    ("Frameworks", EntityType.FRAMEWORK),
    ("Técnicas", EntityType.TECHNIQUE),
    ("Empresas", EntityType.COMPANY),
    ("Datasets", EntityType.DATASET),
    ("Benchmarks", EntityType.BENCHMARK),
]

ProgressFn = Callable[[str], None]


def run_investigation(
    settings: Settings,
    query: str,
    *,
    max_results: int = 60,
    since_days: int = 365,
    top_k: int = 200,
    min_similarity: float | None = None,
    progress: ProgressFn | None = None,
) -> TIEReport:
    def step(msg: str) -> None:
        if progress:
            progress(msg)

    step("Coletando (arXiv, GitHub, Hugging Face)…")
    counts = collect_and_store(settings, query, max_results=max_results, since_days=since_days)

    step("Extraindo entidades (LLM local)…")
    extract_corpus(settings)

    step("Resolvendo entidades…")
    resolve_entities()

    step("Indexando (embeddings BGE)…")
    build_index(settings)
    doc_ids = semantic_search(settings, query, top_k=top_k, min_similarity=min_similarity)

    step("Analisando padrões e gerando relatório…")
    build_relations(doc_ids)
    ranked_by_type = {
        label: rank_entities(settings, doc_ids, etype=etype.value, top_n=10)
        for label, etype in SECTION_TYPES
    }
    cooc = top_cooccurrences(doc_ids, top_n=12)
    return build_report(
        settings, query, n_docs=len(doc_ids), sources=counts,
        ranked_by_type=ranked_by_type, cooccurrences=cooc,
    )
