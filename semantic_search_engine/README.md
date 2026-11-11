# Semantic Search Engine

Motor de IT Shell que recupera candidatos para una solicitud en lenguaje
natural y los presenta como propuestas de comandos o secuencias de CLI.

> **Seguridad:** el motor no ejecuta comandos. Las propuestas se validan y se
> devuelven para revisión humana; esta propiedad no garantiza por sí sola que
> todos los argumentos o efectos sean correctos.

## Flujo de una consulta

1. `QueryAnalyzer` normaliza la solicitud y obtiene señales de idioma y
   complejidad.
2. `RulePlanner` divide conectores explícitos en pasos y dependencias.
3. `FaissRetriever` codifica cada paso y recupera aliases del índice vectorial.
4. `CommandResolver` carga del corpus YAML la intención, el comando y los
   ejemplos asociados a cada alias.
5. El reranker configurado conserva o reordena los candidatos; el
   `PipelineBuilder` aplica umbrales de selección, abstención y alternativas.
6. `ProposalValidator` identifica parámetros requeridos y patrones que
   ameritan advertencias.
7. El formateador genera una respuesta para CLI, terminal, Markdown o JSON.

## Componentes

| Servicio | Responsabilidad |
|---|---|
| `retrieval_service` | Análisis de consulta, planificación y recuperación FAISS. |
| `reranker_service` | Reordenamiento opcional; `IdentityReranker` conserva el orden por defecto. |
| `pipeline_builder_service` | Resolución de aliases, selección de propuestas y validación. |
| `prompt_builder_service` | Contexto de aclaraciones ante datos faltantes o advertencias. |
| `response_formatter_service` | Formateo de salida para los consumidores del motor. |
| `llm` | Backend opcional de Ollama para análisis/selección y explicación estructurados. |
| `mcp` | Servidor Model Context Protocol para búsqueda, planificación y validación. |

## Instalación y configuración

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

La configuración predeterminada está en
[`config/default.yaml`](config/default.yaml). En el checkout apunta al corpus
local y al índice FAISS MPNet; una instalación del paquete usa las rutas de
datos distribuidas o un archivo alternativo pasado con `--config`.

Parámetros principales:

```yaml
embedding_model: sentence-transformers/all-mpnet-base-v2
reranker_model: null
retrieval_top_k: 15
assistant: rules
ollama_model: qwen2.5:1.5b
ollama_base_url: http://localhost:11434
ollama_timeout: 45
selection:
  min_score: 0.45
  alternative_margin: 0.04
  max_alternatives: 3
```

El backend `rules` es el predeterminado. `ollama` puede activarse por
configuración o con `--assistant ollama` para una única ejecución. Requiere
Ollama local disponible y el modelo configurado descargado. El backend puede
seleccionar entre candidatos recuperados y producir una explicación; la
plantilla final del comando permanece bajo control determinista y atraviesa
`ProposalValidator`.

## Uso desde la CLI

```bash
it-shell "list files in a directory"
it-shell "find files and then compress them" --format markdown
it-shell "crear los directorios padre que falten en /tmp/reportes" \
  --assistant ollama
```

Formatos admitidos:

- `cli`: texto plano; formato predeterminado.
- `terminal`: salida para terminal con colores ANSI.
- `markdown`: respuesta formateada para documentación.
- `json`: respuesta estructurada.

Usa `--top-k` para cambiar el número de vecinos y `--config` para indicar una
configuración propia. Con modelos descargados previamente, la inferencia de
embeddings puede ejecutarse sin consultas a Hugging Face:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  it-shell "list files in a directory"
```

## Servidor MCP

El paquete instala el comando:

```bash
it-shell-mcp --config path/to/config.yaml
```

Desde el checkout también se puede iniciar directamente:

```bash
PYTHONPATH=. .venv/bin/python -m semantic_search_engine.mcp.server
```

El servidor expone operaciones de búsqueda, planificación y validación. No
ejecuta comandos. Consulta la [guía de integración MCP](../docs/mcp_integration_guide.md).

## Evaluación y límites conocidos

- En el benchmark oficial de retrieval de 160 consultas, el baseline FAISS
  reportó MRR 0,5377, Hit@5 0,7125 y Hit@10 0,8000. BM25 reportó MRR 0,2408,
  Hit@5 0,2687 y Hit@10 0,3125; sus cifras no justifican reemplazar FAISS.
  Los protocolos y artefactos están descritos en
  [`evaluation/README.md`](../evaluation/README.md).
- En un conjunto independiente de 122 consultas bilingües, se comparó la
  selección por reglas y Ollama sobre un retrieval BGE-M3 compartido. La
  selección correcta fue 39,3% con reglas y 54,9% con Ollama; en el test de 36
  consultas, 27,8% y 44,4%, respectivamente. Ollama tuvo mayor latencia media
  y un fallback registrado. El resultado mide selección del comando, no
  argumentos ni calidad end-to-end. Ver el
  [informe detallado](../evaluation/capacity/bge_ollama_comparison.md).
- Los vectores BGE no registran hashes de los textos fuente; no se puede
  verificar independientemente qué revisión exacta del corpus se utilizó.
- Los resultados del benchmark histórico SLM en modo mock no son resultados de
  Ollama ni validan su calidad.
- No se ha demostrado mediante ejecución en sandbox que todas las propuestas
  produzcan el efecto solicitado.

## Pruebas

Ejecuta la suite desde la raíz del repositorio:

```bash
python -m pytest -q
```

Las pruebas verifican componentes y contratos; no ejecutan los comandos
propuestos en el sistema anfitrión.
