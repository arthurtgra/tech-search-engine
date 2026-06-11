"""Geração e render do relatório de inteligência tecnológica.

O resumo executivo é sintetizado por LLM local, mas SEMPRE ancorado nos números
já calculados (rankings/momentum/co-ocorrência) passados como contexto — para
evitar alucinação. As demais seções são determinísticas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tie.analytics import CoOccurrence, RankedEntity
from tie.config import Settings
from tie.models import EntityType

# Tipos exibidos como seções dedicadas no relatório.
_SECTION_TYPES = [
    ("Modelos", EntityType.MODEL),
    ("Frameworks", EntityType.FRAMEWORK),
    ("Técnicas", EntityType.TECHNIQUE),
    ("Empresas", EntityType.COMPANY),
    ("Datasets", EntityType.DATASET),
    ("Benchmarks", EntityType.BENCHMARK),
]


@dataclass
class TIEReport:
    query: str
    n_docs: int
    sources: dict[str, int]
    sections: dict[str, list[RankedEntity]] = field(default_factory=dict)
    cooccurrences: list[CoOccurrence] = field(default_factory=list)
    summary: str = ""


def _synthesize_summary(settings: Settings, report: TIEReport) -> str:
    if not settings.use_llm_extraction:
        return "(síntese por LLM desativada)"
    facts = [f"Pergunta: {report.query}", f"Documentos analisados: {report.n_docs}"]
    for label, items in report.sections.items():
        if items:
            top = ", ".join(f"{e.name} (x{e.doc_count}, mom={e.momentum})" for e in items[:5])
            facts.append(f"{label}: {top}")
    if report.cooccurrences:
        co = "; ".join(f"{c.a}+{c.b} (x{c.weight})" for c in report.cooccurrences[:6])
        facts.append(f"Co-ocorrências fortes: {co}")
    context = "\n".join(facts)
    prompt = (
        "Você é um analista de inteligência tecnológica. Com base APENAS nos fatos "
        "abaixo (já calculados sobre um corpus real), escreva um resumo executivo de "
        "5-7 linhas em português sobre o que está acontecendo no tema. Cite tendências "
        "(momentum > 1 = crescendo). NÃO invente nada além dos fatos.\n\n" + context
    )
    try:
        client = ollama.Client(host=settings.ollama_host)
        resp = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return resp["message"]["content"].strip()
    except Exception as exc:
        return f"(síntese indisponível: {exc})"


def build_report(
    settings: Settings,
    query: str,
    *,
    n_docs: int,
    sources: dict[str, int],
    ranked_by_type: dict[str, list[RankedEntity]],
    cooccurrences: list[CoOccurrence],
) -> TIEReport:
    report = TIEReport(
        query=query, n_docs=n_docs, sources=sources,
        sections=ranked_by_type, cooccurrences=cooccurrences,
    )
    report.summary = _synthesize_summary(settings, report)
    return report


def render(report: TIEReport, console: Console | None = None) -> None:
    c = console or Console()
    c.print()
    c.print(Panel.fit(f"[bold]Technology Intelligence Engine[/bold]\n{report.query}",
                      border_style="cyan"))

    src = ", ".join(f"{k}={v}" for k, v in report.sources.items() if not k.endswith("error"))
    c.print(f"[dim]Documentos analisados: {report.n_docs}  |  Coleta: {src}[/dim]\n")

    c.print(Panel(report.summary, title="Resumo executivo", border_style="green"))

    for label, items in report.sections.items():
        if not items:
            continue
        t = Table(title=label, title_justify="left", header_style="bold magenta")
        t.add_column("#", width=3)
        t.add_column("Entidade")
        t.add_column("Docs", justify="right")
        t.add_column("Métrica", justify="right")
        t.add_column("Momentum", justify="right")
        for i, e in enumerate(items, 1):
            mom = f"[green]+{e.momentum}[/green]" if e.momentum > 1 else f"{e.momentum}"
            t.add_row(str(i), e.name, str(e.doc_count), f"{e.metric_sum:,.0f}", mom)
        c.print(t)

    if report.cooccurrences:
        t = Table(title="Co-ocorrências (aparecem juntas)", title_justify="left",
                  header_style="bold yellow")
        t.add_column("Par")
        t.add_column("Peso", justify="right")
        for co in report.cooccurrences:
            t.add_row(f"{co.a} [dim]+[/dim] {co.b}", str(co.weight))
        c.print(t)

    c.print(Panel(
        "Confiança expressa como [b]contagem de evidências[/b] (coluna Docs), não como "
        "porcentagem inventada.\n"
        "[b]Limitações:[/b] co-ocorrência ≠ sequência de pipeline; papers têm latência "
        "de indexação; modelos fechados (anunciados em blogs) ficam sub-representados; "
        "extração depende da qualidade do LLM local.",
        title="Confiança & Limitações", border_style="red",
    ))
