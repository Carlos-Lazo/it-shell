# PLAN DE CONSTRUCCIÓN – dataset_builder & semantic_search_engine

---
## 1. PROPÓSITO
Este plan traduce el **`plan_implementacion_detallado.txt`** (ver líneas 24‑236) en una hoja de ruta *concreta* para construir, validar y desplegar los dos componentes críticos del proyecto:
- **`dataset_builder`** (parte offline que genera el corpus, embeddings y el índice FAISS).
- **`semantic_search_engine`** (SSE, motor de recuperación en línea y capas de reranking, validación y exposición vía MCP).

---
## 2. ESTRUCTURA DE FASES (alineada con el documento fuente)

### ----------------------------------------------------------------------
#### FASE 1 – REVISIÓN INICIAL Y ESTABLECIMIENTO DE BASE
### ----------------------------------------------------------------------
**Objetivo:** Consolidar la base de código y garantizar que las pruebas unitarias (ver línea 10‑13 del plan) pasen en ambas ramas.
- **Tareas**
  1.1 Auditar `dataset_builder/` y `semantic_search_engine/` (PEP‑8, type‑hints, doc‑strings).  
  1.2 Ejecutar la suite de pruebas (`PYTHONPATH=. dataset_builder/indexer/.venv/bin/python -m unittest discover tests`).
  1.3 Generar informe de auditoría y crear PR de refactorizado.
- **Entregables**: Informe de auditoría, código refactorizado en `main` y `dataset_builder`.
- **Criterio de aceptación**: **100 %** de pruebas pasan y al menos **2 aprobaciones** en el PR.

### ----------------------------------------------------------------------
#### FASE 2 – CURACIÓN Y EXPANSIÓN DEL CORPUS (offline)
### ----------------------------------------------------------------------
**Objetivo (líneas 39‑42):** Subsanar los comandos ausentes y añadir soporte multilingüe.
- **Tareas**
  2.1 Inventariar comandos críticos faltantes (`git`, `docker`, `kubectl`, `systemctl`, etc.).
  2.2 Crear/editar archivos YAML bajo `dataset_builder/corpus/` siguiendo el esquema actual.
  2.3 Añadir descripciones y ejemplos en **español e inglés**.
  2.4 Validar contra `schema_corpus.yaml` y generar `corpus_validation_report.txt`.
- **Entregables**: Corpus actualizado (~30 k entradas) y reporte de validación.
- **Criterio de aceptación**: Cobertura **≥ 95 %** de la lista objetivo y esquema sin errores.

### ----------------------------------------------------------------------
#### FASE 3 – GENERACIÓN DE EMBEDDINGS (offline)
### ----------------------------------------------------------------------
**Objetivo (líneas 33‑38):** Transformar el corpus en vectores semánticos.
- **Tareas**
  3.1 Ejecutar `dataset_builder/indexer/generate_embeddings.py` con `sentence‑transformers/all‑mpnet‑base‑v2` (modo offline).
  3.2 Persistir `vectors.npz` y `aliases.npy` en `dataset_builder/indexer/artifacts/`.
  3.3 Crear wrapper `run_embeddings.sh` que registre log estructurado (`logs/embeddings_*.log`).
- **Entregables**: `vectors.npz`, `aliases.npy`, script wrapper.
- **Criterio de aceptación**: Número de vectores == número de alias; generación completada en < 5 min en CI.

### ----------------------------------------------------------------------
#### FASE 4 – CONSTRUCCIÓN DEL ÍNDICE FAISS (offline)
### ----------------------------------------------------------------------
**Objetivo (líneas 57‑61):** Construir el índice vectorial consumible por SSE.
- **Tareas**
  4.1 Ejecutar `dataset_builder/indexer/build_faiss_index.py` usando artefactos de la fase 3.
  4.2 Ajustar parámetros `nlist=1024`, `nprobe=16` (balance precisión/latencia).
  4.3 Validar búsqueda con `faiss_search_test.py` (top‑k = 15, latencia < 2 s).
- **Entregables**: `index.faiss` versionado en `evaluation/`, documento `faiss_params.md`.
- **Criterio de aceptación**: Búsqueda devuelve los mismos vecinos que en pruebas locales; latencia media < 2 s.

### ----------------------------------------------------------------------
#### FASE 5 – INTEGRACIÓN EN SSE (main) – MOTOR DE RECUPERACIÓN
### ----------------------------------------------------------------------
**Objetivo (líneas 11‑13, 24‑30):** Consumir los artefactos y exponer la API.
- **Tareas**
  5.1 Merge de `dataset_builder` → `main` (solo artefactos y scripts de carga).  
  5.2 Modificar `semantic_search_engine/config/default.yaml` → `retrieval_top_k: 15`, `use_cross_encoder: false/true`.
  5.3 Actualizar `pipeline_builder.py` para usar `retrieval_score` (ya corregido en línea 46‑48) y habilitar reranker opcional.
  5.4 Implementar endpoint REST `/search` en `semantic_search_engine/api/` que devuelve comando, puntuación, explicación y ejemplos.
- **Entregables**: Rama `main` con SSE v2, archivo `api_spec.md` (OpenAPI).
- **Criterio de aceptación**: `curl -X POST /search` responde < 1 s; pruebas de integración (`tests/integration_test.py`) pasan.

### ----------------------------------------------------------------------
#### FASE 6 – RE‑RANKING Y CALIBRACIÓN DE UMBRAL (líneas 44‑48)
### ----------------------------------------------------------------------
- **Tareas**
  6.1 Ejecutar benchmark Cross‑Encoder (`evaluation/evaluate_reranker.py`) con top‑k = 15.
  6.2 Calibrar `min_score`, `alternative_margin` y `max_alternatives` en `config/default.yaml`.
  6.3 Verificar ausencia de abstenciones falsas.
- **Entregables**: Script benchmark actualizado, `default.yaml` calibrado, informe `experimento_reranker_v2.json`.
- **Criterio de aceptación**: MRR ≥ 0.25, Hit@5 ≥ 50 %, latencia < 2 s.

### ----------------------------------------------------------------------
#### FASE 7 – CAPA DE EXPLICACIÓN & PROMPT BUILDER (líneas 118‑144)
### ----------------------------------------------------------------------
- **Tareas**
  7.1 Implementar `prompt_builder_service/prompt_builder.py` y `llm_client.py` (modo offline fallback).
  7.2 Generar salida en terminal, Markdown y JSON.
  7.3 Añadir pruebas unitarias para construcción de prompts.
- **Entregables**: Código del *Prompt Builder*, pruebas, documentación.
- **Criterio de aceptación**: Respuesta determinista cuando el LLM no está disponible.

### ----------------------------------------------------------------------
#### FASE 8 – SERVIDOR MCP (líneas 149‑176)
### ----------------------------------------------------------------------
- **Tareas**
  8.1 Crear `semantic_search_engine/mcp/server.py` usando `mcp-sdk` (STDIO JSON‑RPC).
  8.2 Registrar herramientas `search_commands`, `plan_operation`, `validate_command`.
  8.3 Desarrollar script cliente de pruebas y generar `docs/mcp_integration_guide.md`.
- **Entregables**: Sub‑paquete `mcp/`, guía de integración.
- **Criterio de aceptación**: Cliente MCP recibe plan validado en < 400 ms.

### ----------------------------------------------------------------------
#### FASE 9 – EMPAQUETADO, OPTIMIZACIÓN Y DOCUMENTACIÓN FINAL (líneas 177‑200)
### ----------------------------------------------------------------------
- **Tareas**
  9.1 Crear `pyproject.toml` y CLI `it-shell`.
  9.2 Perfilar RAM/CPU; considerar exportar Cross‑Encoder a ONNX si latencia > 2 s.
  9.3 Actualizar `README.md`, `docs/it_shell_mejoras.txt` y generar informe comparativo final.
- **Entregables**: paquete instalable, binario CLI, informe técnico final.
- **Criterio de aceptación**: Instalación `pip install -e .` funciona; informe muestra mejora ≥ 2× respecto al baseline.

---
## 3. CRONOGRAMA RECOMENDADO (sprints – alineado con sección 3 del plan)
| Sprint | Fase(s) cubierta | Duración estimada |
|--------|-------------------|-------------------|
| 1 | Fase 1 + Fase 2 | 2 semanas |
| 2 | Fase 3 + Fase 4 | 2 semanas |
| 3 | Fase 5 + Fase 6 | 2 semanas |
| 4 | Fase 7 + Fase 8 | 2 semanas |
| 5 | Fase 9 (empaquetado & docs) | 1 semana |

---
## 4. COMANDOS CLAVE DE VERIFICACIÓN (copiados del plan, 214‑227)
```bash
# 1. Ejecutar suite de pruebas completa
PYTHONPATH=. dataset_builder/indexer/.venv/bin/python -m unittest discover tests

# 2. CLI de consulta (modo offline)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python semantic_search_engine/cli.py "find large files" --format markdown

# 3. Evaluación del baseline (retrieval)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/evaluate_retrieval.py
```
---
## 5. RESPONSABLES (asignar en el proyecto)
| Rol | Responsable |
|------|-------------|
| Project Lead | Carlos |
| Dev Lead (SSE) | *por asignar* |
| ML Engineer (embeddings) | *por asignar* |
| Content Engineer (corpus) | *por asignar* |
| QA Engineer | *por asignar* |
| Docs Engineer | *por asignar* |
| DevOps | *por asignar* |

---
### NOTA
Este plan se mantiene fiel a la estructura y numeración del **`plan_implementacion_detallado.txt`** (p.ej., fases 2‑7, cronograma, comandos). Cada fase corresponde a una sección del documento original y a los entregables ya existentes en el repositorio.

---
*Archivo creado:* `docs/plan_construccion_dataset_sse.md`

