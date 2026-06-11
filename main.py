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

from tie.config import load_settings
from tie.pipeline import run_investigation
from tie.report import render, save_report

app = typer.Typer(add_completion=False, help="Technology Intelligence Engine")
console = Console()


@app.command()
def investigate(
    query: str = typer.Argument(..., help="Tema a investigar"),
    max_results: int = typer.Option(60, "--max", help="Máx. de itens por fonte"),
    since: int = typer.Option(365, "--since", help="Janela de coleta (dias)"),
    top_k: int = typer.Option(200, "--topk", help="Docs relevantes p/ análise"),
    min_sim: float = typer.Option(0.5, "--min-sim", help="Similaridade mínima (cosseno)"),
) -> None:
    settings = load_settings()
    with console.status("Investigando…") as st:
        report = run_investigation(
            settings, query, max_results=max_results, since_days=since,
            top_k=top_k, min_similarity=min_sim,
            progress=lambda m: st.update(m),
        )
    render(report, console)
    path = save_report(report)
    console.print(f"\n[dim]Relatório salvo em:[/dim] {path}")


@app.command()
def reports(
    show: int = typer.Option(0, "--show", help="Imprime o conteúdo do N-ésimo relatório (1=mais recente)"),
) -> None:
    """Lista os relatórios salvos (ou imprime um com --show N)."""
    from tie.config import REPORTS_DIR

    files = sorted(REPORTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        console.print("[yellow]Nenhum relatório salvo ainda. Rode 'investigate' primeiro.[/yellow]")
        return
    if show:
        from rich.markdown import Markdown

        console.print(Markdown(files[show - 1].read_text(encoding="utf-8")))
        return
    for i, f in enumerate(files, 1):
        ts = f.stat().st_mtime
        from datetime import datetime as _dt

        console.print(f"{i:>2}. {f.name}  [dim]({_dt.fromtimestamp(ts):%Y-%m-%d %H:%M})[/dim]")
    console.print("\n[dim]Veja um com:[/dim] python main.py reports --show 1")


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
