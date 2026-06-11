"""CLI do Technology Intelligence Engine.

    python main.py investigate "agentes autônomos"
    python main.py investigate "RAG techniques" --max 80 --since 180

Subcomandos de depuração: collect, extract, index, status.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console

# Console Windows costuma ser cp1252; força UTF-8 para acentos e glifos.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from tie.analytics import build_relations, rank_entities, top_cooccurrences
from tie.config import load_settings
from tie.extract import extract_corpus
from tie.index import build_index, semantic_search
from tie.ingest import collect_and_store
from tie.models import EntityType
from tie.report import build_report, render
from tie.resolve import resolve_entities

app = typer.Typer(add_completion=False, help="Technology Intelligence Engine")
console = Console()

_SECTION_TYPES = [
    ("Modelos", EntityType.MODEL),
    ("Frameworks", EntityType.FRAMEWORK),
    ("Técnicas", EntityType.TECHNIQUE),
    ("Empresas", EntityType.COMPANY),
    ("Datasets", EntityType.DATASET),
    ("Benchmarks", EntityType.BENCHMARK),
]


@app.command()
def investigate(
    query: str = typer.Argument(..., help="Tema a investigar"),
    max_results: int = typer.Option(60, "--max", help="Máx. de itens por fonte"),
    since: int = typer.Option(365, "--since", help="Janela de coleta (dias)"),
    top_k: int = typer.Option(200, "--topk", help="Docs relevantes p/ análise"),
) -> None:
    settings = load_settings()

    with console.status("[1/5] Coletando (arXiv, GitHub, Hugging Face)..."):
        counts = collect_and_store(settings, query, max_results=max_results, since_days=since)
    console.print(f"[1/5] Coleta: {counts}")

    with console.status("[2/5] Extraindo entidades (LLM local)..."):
        n_ext = extract_corpus(settings)
    console.print(f"[2/5] Extração: {n_ext} documentos processados")

    with console.status("[3/5] Resolvendo entidades..."):
        res = resolve_entities()
    console.print(f"[3/5] Resolução: {res}")

    with console.status("[4/5] Indexando (embeddings BGE)..."):
        n_idx = build_index(settings)
    doc_ids = semantic_search(settings, query, top_k=top_k)
    console.print(f"[4/5] Index: +{n_idx} embedados | {len(doc_ids)} docs relevantes")

    with console.status("[5/5] Analisando padrões e gerando relatório..."):
        build_relations(doc_ids)
        ranked_by_type = {
            label: rank_entities(settings, doc_ids, etype=etype.value, top_n=10)
            for label, etype in _SECTION_TYPES
        }
        cooc = top_cooccurrences(doc_ids, top_n=12)
        report = build_report(
            settings, query, n_docs=len(doc_ids), sources=counts,
            ranked_by_type=ranked_by_type, cooccurrences=cooc,
        )
    render(report, console)


@app.command()
def status() -> None:
    """Mostra o tamanho atual do corpus."""
    from sqlalchemy import func

    from tie.db import get_session
    from tie.models import Document, Entity, Mention, Relation

    with get_session() as s:
        console.print({
            "documentos": s.query(func.count(Document.id)).scalar(),
            "entidades": s.query(func.count(Entity.id)).scalar(),
            "menções": s.query(func.count(Mention.id)).scalar(),
            "relações": s.query(func.count(Relation.id)).scalar(),
        })


@app.command()
def collect(
    query: str,
    max_results: int = typer.Option(60, "--max"),
    since: int = typer.Option(365, "--since"),
) -> None:
    counts = collect_and_store(load_settings(), query, max_results=max_results, since_days=since)
    console.print(counts)


if __name__ == "__main__":
    app()
