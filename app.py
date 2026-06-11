"""Dashboard Streamlit do Technology Intelligence Engine.

    streamlit run app.py

Reutiliza o pipeline (tie.pipeline.run_investigation) e renderiza o relatório
de forma visual: cards de métrica, abas por tipo de entidade, gráficos de
momentum e grafo de co-ocorrência.
"""

from __future__ import annotations

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from tie.config import REPORTS_DIR, load_settings
from tie.pipeline import SECTION_TYPES, run_investigation
from tie.report import TIEReport, save_report, to_markdown

st.set_page_config(
    page_title="Technology Intelligence Engine",
    page_icon="🛰️",
    layout="wide",
)

# Paleta por tipo de entidade (usada no grafo e nos gráficos).
TYPE_COLORS = {
    "model": "#6366f1", "framework": "#22c55e", "technique": "#f59e0b",
    "company": "#ef4444", "dataset": "#06b6d4", "benchmark": "#a855f7",
    "tool": "#84cc16", "institution": "#ec4899", "method": "#f97316",
    "metric": "#14b8a6",
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      .exec-summary {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #e2e8f0; padding: 1.2rem 1.4rem; border-radius: 14px;
        border-left: 4px solid #6366f1; font-size: 1.02rem; line-height: 1.55;
      }
      div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        padding: 0.8rem 1rem; border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _ranked_df(items) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Entidade": [e.name for e in items],
            "Docs": [e.doc_count for e in items],
            "Métrica": [e.metric_sum for e in items],
            "Momentum": [e.momentum for e in items],
        }
    )


def _momentum_chart(df: pd.DataFrame, color: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color=color)
        .encode(
            x=alt.X("Docs:Q", title="Documentos"),
            y=alt.Y("Entidade:N", sort="-x", title=None),
            tooltip=["Entidade", "Docs", "Métrica", "Momentum"],
        )
        .properties(height=max(140, 28 * len(df)))
    )


def _cooc_dot(report: TIEReport) -> str:
    lines = [
        "graph G {",
        '  graph [bgcolor="transparent", overlap=false, splines=true];',
        '  node [style=filled, fontname="Helvetica", fontsize=11, '
        'fontcolor=white, color="#00000000", shape=box, '
        'style="filled,rounded"];',
        '  edge [color="#94a3b8"];',
    ]
    seen: dict[str, str] = {}
    for co in report.cooccurrences:
        seen[co.a] = co.type_a
        seen[co.b] = co.type_b
    for name, etype in seen.items():
        color = TYPE_COLORS.get(etype, "#64748b")
        safe = name.replace('"', "'")
        lines.append(f'  "{safe}" [fillcolor="{color}"];')
    for co in report.cooccurrences:
        w = 1 + co.weight / 2
        a, b = co.a.replace('"', "'"), co.b.replace('"', "'")
        lines.append(f'  "{a}" -- "{b}" [penwidth={w:.1f}];')
    lines.append("}")
    return "\n".join(lines)


def render_report(report: TIEReport) -> None:
    st.title(f"🛰️ {report.query}")
    src = report.sources
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documentos analisados", report.n_docs)
    c2.metric("arXiv", src.get("arxiv", 0))
    c3.metric("GitHub", src.get("github", 0))
    c4.metric("Hugging Face", src.get("huggingface", 0))

    st.markdown(
        f'<div class="exec-summary">{report.summary}</div>', unsafe_allow_html=True
    )
    st.write("")

    labels = [lbl for lbl, _ in SECTION_TYPES]
    type_by_label = {lbl: et.value for lbl, et in SECTION_TYPES}
    tabs = st.tabs(labels)
    for tab, label in zip(tabs, labels):
        with tab:
            items = report.sections.get(label, [])
            if not items:
                st.caption("Nada encontrado nesta categoria.")
                continue
            df = _ranked_df(items)
            left, right = st.columns([1, 1])
            with left:
                st.dataframe(
                    df, hide_index=True, width="stretch",
                    column_config={
                        "Métrica": st.column_config.NumberColumn(format="%d"),
                        "Momentum": st.column_config.NumberColumn(
                            format="%.2f", help=">1 = crescendo na metade recente"
                        ),
                    },
                )
            with right:
                st.altair_chart(
                    _momentum_chart(df, TYPE_COLORS.get(type_by_label[label], "#6366f1")),
                    width="stretch",
                )

    if report.cooccurrences:
        st.subheader("Co-ocorrências — o que aparece junto")
        gcol, tcol = st.columns([1.4, 1])
        with gcol:
            st.graphviz_chart(_cooc_dot(report), width="stretch")
        with tcol:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Par": [f"{c.a} + {c.b}" for c in report.cooccurrences],
                        "Peso": [c.weight for c in report.cooccurrences],
                    }
                ),
                hide_index=True, width="stretch",
            )

    with st.expander("Confiança & Limitações"):
        st.markdown(
            "- **Confiança** = contagem de evidências (coluna *Docs*), não "
            "porcentagem inventada.\n"
            "- Co-ocorrência ≠ sequência de pipeline.\n"
            "- Papers têm latência de indexação; modelos fechados ficam "
            "sub-representados.\n"
            "- Extração depende da qualidade do LLM local."
        )

    st.download_button(
        "⬇️ Baixar relatório (Markdown)",
        data=to_markdown(report),
        file_name=f"{report.query}.md".replace(" ", "-"),
        mime="text/markdown",
    )


# --------------------------------------------------------------------------- #
# Sidebar / navegação
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🛰️ Tech Intel Engine")
    st.caption("Analista tecnológico local — detecta padrões emergentes.")
    mode = st.radio("Modo", ["🔎 Investigar", "📁 Relatórios salvos"], label_visibility="collapsed")
    st.divider()

if mode == "🔎 Investigar":
    with st.sidebar:
        query = st.text_input("Tema", value="autonomous agents",
                              placeholder="ex.: RAG techniques")
        max_results = st.slider("Itens por fonte", 10, 100, 30, step=5)
        since = st.slider("Janela (dias)", 30, 730, 365, step=30)
        min_sim = st.slider("Similaridade mínima", 0.0, 0.9, 0.5, step=0.05)
        run = st.button("Investigar", type="primary", width="stretch")

    if run and query.strip():
        settings = load_settings()
        status = st.status("Investigando…", expanded=True)
        report = run_investigation(
            settings, query.strip(), max_results=max_results, since_days=since,
            min_similarity=min_sim, progress=lambda m: status.write(m),
        )
        status.update(label="Concluído ✅", state="complete", expanded=False)
        save_report(report)
        st.session_state["report"] = report

    if "report" in st.session_state:
        render_report(st.session_state["report"])
    else:
        st.info("Defina um tema na barra lateral e clique **Investigar**.")

else:  # Relatórios salvos
    files = sorted(REPORTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.info("Nenhum relatório salvo ainda. Rode uma investigação primeiro.")
    else:
        with st.sidebar:
            choice = st.selectbox(
                "Relatório",
                files,
                format_func=lambda p: f"{p.stem}  ({datetime.fromtimestamp(p.stat().st_mtime):%d/%m %H:%M})",
            )
        st.markdown(choice.read_text(encoding="utf-8"))
        st.download_button(
            "⬇️ Baixar", data=choice.read_text(encoding="utf-8"),
            file_name=choice.name, mime="text/markdown",
        )
