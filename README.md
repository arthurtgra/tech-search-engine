# Technology Intelligence Engine

Analista tecnológico automatizado, **100% local e gratuito**. Não resume artigos —
detecta *padrões emergentes*: quais modelos/técnicas/empresas aparecem juntos,
o que está crescendo e quais relações estão se consolidando.

```bash
python main.py investigate "autonomous agents"
python main.py investigate "RAG techniques" --max 80 --since 180
python main.py status
```

## O que diferencia isto de um LLM comum

1. **Frescor** — dados posteriores ao cutoff do modelo.
2. **Momentum medido** — crescimento calculado sobre séries temporais reais
   (`TrendSnapshot`), não chutado.
3. **Proveniência** — toda afirmação aponta para documentos citáveis.
4. **Co-ocorrência estatística** — "estes N itens aparecem juntos" é cálculo sobre
   o corpus, que um LLM não faz de cabeça.

Se qualquer um quebrar, vira um chatbot pior. O produto vive na qualidade de
**frescor + entity resolution**.

## Arquitetura (pipeline)

```
collectors → ingest(dedupe) → extract(LLM local) → resolve → index(BGE+LanceDB)
           → semantic_search(escopo) → analytics(co-ocorrência, momentum) → report
```

| Componente | Escolha | Porquê |
|------------|---------|--------|
| Embeddings | BAAI/bge-small-en-v1.5 (384d) | Melhor custo/qualidade local p/ texto técnico curto |
| Vetorial | LanceDB | Embutido, sem servidor, filtro temporal+vetorial |
| Extração | Ollama / qwen2.5:7b (JSON) | Captura entidades novas que NER local perde |
| Persistência | SQLite + WAL | Single-file, suporta collectors concorrentes |
| Grafo | tabela `relations` no SQLite | Co-ocorrência = `GROUP BY`; sem Neo4j no MVP |

## Decisões de produto (honestidade > marketing)

- **Sem "confiança 84%" inventada.** Confiança = contagem de evidências (coluna Docs).
- **Co-ocorrência ≠ pipeline.** O v0 entrega grafo de co-ocorrência ponderado, não
  inferência de sequência (`Script→Storyboard→...`). Inferir ordem a partir de texto
  livre exige ontologia curada — rebaixado para v2.
- **Papers têm latência.** O sinal mais fresco vem de GitHub (stars) e HF (downloads).
  Modelos fechados (Veo/Kling/Gen-4) ficam sub-representados — limitação assumida.

## Limitações conhecidas

- GitHub depende de `api.github.com` (bloqueado em algumas redes); o pipeline degrada
  sem derrubar. Defina `GITHUB_TOKEN` para 5000 req/h quando acessível.
- Busca da HF é substring no id do modelo → consultamos por token e mesclamos.
- Qualidade da extração = qualidade do LLM local.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b   # extração + síntese
```
