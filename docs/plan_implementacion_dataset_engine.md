# Plan de Implementación – Módulos `dataset_builder` y `semantic_search_engine`

## 1. Visión General
Este plan detalla las actividades necesarias para completar, validar y mantener los componentes **`dataset_builder`** y **`semantic_search_engine`** del proyecto *Asistente inteligente de línea de comandos*. Se alinean con el **Plan de Trabajo del Proyecto** y enfocan los esfuerzos en:
- Construcción y actualización del repositorio de conocimiento (corpus).
- Generación y gestión de embeddings vectoriales.
- Indexación FAISS y alias management.
- Desarrollo del motor de recuperación semántica y pipeline de selección/abstención.
- Pruebas unitarias, benchmark y documentación.

## 2. Estructura de la Arquitectura
```
+ dataset_builder/
|   ├─ corpus/               # YAML con descripciones de comandos, opciones y ejemplos
|   ├─ indexer/               # Scripts de generación de embeddings y FAISS index
|   └─ utils/                 # Helpers de pre‑procesado y normalización

+ semantic_search_engine/
|   ├─ pipeline_builder_service/
|   │   └─ pipeline_builder.py   # Lógica de selección, abstención y reranking
|   ├─ config/                 # default.yaml, parámetros del motor
|   └─ api/                    # Endpoints de búsqueda (REST/CLI)
```

## 3. Fases y Tareas
| Fase | Duración | Tareas Principales | Entregables | Responsable |
|------|----------|--------------------|-------------|------------|
| **1. Revisión y Refactorización** | 1 sem | - Auditar código existente (docstrings, tipado).<br>- Normalizar nombres de funciones y estructuras de datos.<br>- Añadir logs estructurados. | Informe de auditoría, código refactorizado. | 🔧 Dev Lead |
| **2. Expansión del Corpus** | 2 sem | - Identificar comandos faltantes (git, docker, systemctl, kubectl, ...).<br>- Crear/editar archivos YAML en `dataset_builder/corpus/` siguiendo el esquema actual.<br>- Añadir ejemplos de uso y descripciones en español e inglés.<br>- Validar consistencia con `schema_corpus.yaml`. | Corpus actualizado (≈ 30 k entradas). | 📚 Content Engineer |
| **3. Generación de Embeddings** | 1 sem | - Ejecutar `indexer/generate_embeddings.py` con modelo `sentence-transformers/all-mpnet-base-v2` (offline).<br>- Guardar `vectors.npz` y `aliases.npy`.
- Automatizar script con parámetro `--output_dir=dataset_builder/indexer/artifacts`.
- Verificar dimensionalidad (768) y número de vectores. | Vectores y alias files versionados. | 🤖 ML Engineer |
| **4. Indexado FAISS** | 1 sem | - Construir índice FAISS (`index.faiss`) con `indexer/build_faiss_index.py`.
- Realizar pruebas de búsqueda rápida (`faiss_search_test.py`).
- Documentar parámetros (nlist, nprobe). | Índice FAISS listo, pruebas de performance. | 🤖 ML Engineer |
| **5. Integración en `semantic_search_engine`** | 2 sem | - Ajustar `pipeline_builder.py` para usar `retrieval_top_k` configurable (default 15).
- Implementar lógica de abstención basada en `retrieval_score` (ya corregida).
- Añadir soporte de reranker (Cross‑Encoder) como paso opcional.
- Exponer configuración vía CLI y `config/default.yaml`. | Pipeline funcional, documentación de parámetros. | 🛠️ Backend Engineer |
| **6. Tests y Benchmark** | 1 sem | - Añadir pruebas unitarias para `dataset_builder` (carga de corpus, generación de embeddings).
- Extender suite `tests/` con casos de búsqueda (FAISS + reranker).
- Ejecutar `evaluate_retrieval.py` y `evaluate_reranker.py` con top‑k = 15.
- Registrar métricas en `evaluation/` (MRR, Hit@k, latencia). | Reportes de métricas, suite de tests 100 % coverage. | ✅ QA Engineer |
| **7. Documentación y CI** | 1 sem | - Actualizar `docs/it_shell_mejoras.txt` con sección *Dataset Builder* y *Semantic Search Engine*.
- Añadir diagramas de arquitectura (Mermaid flowchart).
- Configurar GitHub Actions para: generación automática de embeddings al hacer push en `corpus/` y ejecución de tests. | Docs actualizadas, CI pipeline. | 📖 Docs Engineer |
| **8. Despliegue y Monitoreo** | 1 sem | - Empaquetar motor como paquete pip (`semantic_search_engine/`).
- Crear script de arranque (`run_search_engine.sh`).
- Configurar monitoreo de latencia y uso de memoria (Prometheus + Grafana). | Sistema desplegable, métricas operativas. | 🚀 DevOps |

## 4. Cronograma Detallado (Calendario)
```
Semana 1: Fase 1 – Auditoría y refactorización.
Semana 2‑3: Fase 2 – Expansión del corpus.
Semana 4: Fase 3 – Generación de embeddings.
Semana 5: Fase 4 – Construcción del índice FAISS.
Semana 6‑7: Fase 5 – Integración y ajustes del pipeline.
Semana 8: Fase 6 – Tests, benchmark y reporte de métricas.
Semana 9: Fase 7 – Documentación, diagramas y CI.
Semana 10: Fase 8 – Empaquetado, despliegue y monitoreo.
```

## 5. Riesgos y Mitigaciones
| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Crecimiento rápido del corpus → tiempo de indexado elevado. | Alto | Utilizar *incremental indexing* (FAISS `add_with_ids`) y procesado batch en CI.
| Memoria insuficiente para embeddings de > 50 k comandos. | Medio | Limitar dimensión (768) y aplicar *PCA* opcional (`sklearn.decomposition`).
| Incompatibilidad de versiones de `sentence-transformers` offline. | Bajo | Fijar versión exacta en `requirements.txt` y pre‑cache en CI.
| Latencia del reranker en CPU. | Medio | Habilitar flag `use_cross_encoder=False` por defecto; usar GPU solo en entornos de prueba.

## 6. Métricas de Éxito
- **Cobertura del corpus**: ≥ 95 % de comandos objetivo (git, docker, kubectl, systemctl, etc.).
- **Recall @5**: ≥ 30 % en pruebas de retrieval.
- **MRR**: ≥ 0.20 con top‑k = 15.
- **Latencia media**: ≤ 2 s por consulta (CPU‑only).
- **Cobertura de pruebas**: 100 % de módulos críticos con cobertura ≥ 90 %.

## 7. Responsables y Roles
- **Project Lead**: Carlos (coordinación general).
- **Dev Lead**: (asignar).
- **ML Engineer**: (asignar).
- **Backend Engineer**: (asignar).
- **Content Engineer**: (asignar).
- **QA Engineer**: (asignar).
- **Docs Engineer**: (asignar).
- **DevOps**: (asignar).

---
*Este plan está alineado con el plan de trabajo del proyecto y enfocado en los componentes ya construidos. Servirá como hoja de ruta para los próximos sprints y garantizará la integración, calidad y despliegue robusto del motor de búsqueda semántica.*

