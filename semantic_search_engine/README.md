# Semantic Search Engine

Motor inteligente de recuperación semántica que propone comandos y secuencias CLI a partir de consultas formuladas en lenguaje natural.

> **Principio de diseño**: El motor asiste, propone alternativas y resalta riesgos potenciales; **por diseño nunca ejecuta comandos en el shell** (`executable: false`).

---

## 1. Servicios y Arquitectura

El motor se organiza en componentes modulares con bajo acoplamiento:

* **`retrieval_service`**:
  * `QueryAnalyzer`: normaliza el texto, detecta idioma (`en` / `es`) y evalúa complejidad de la consulta.
  * `RulePlanner`: divide consultas compuestas en pasos explícitos con dependencias secuenciales (`step_1`, `step_2`, etc.) mediante conectores en inglés y español (`and then`, `after that`, `y luego`, `&&`, `|`, etc.).
  * `FaissRetriever`: codifica los pasos mediante `SentenceTransformer` y consulta el índice vectorial FAISS (`IndexFlatIP`) para recuperar candidatos mediante sus aliases (`categoria:id`).
* **`reranker_service`**:
  * Adaptador modular de reordenamiento. Por defecto utiliza `IdentityReranker` (preserva el orden de similitud de FAISS). Puede configurarse opcionalmente con un modelo de `CrossEncoder`.
* **`pipeline_builder_service`**:
  * `CommandResolver`: mapea el alias al documento correspondiente del corpus YAML (intención, sintaxis y ejemplos).
  * `PipelineBuilder`: aplica el umbral de confianza (`min_score`) para abstención y selecciona alternativas dentro del margen configurado (`alternative_margin`).
  * `ProposalValidator`: analiza plantillas de comandos, extrae parámetros requeridos (`<param>`) y detecta patrones de riesgo destructivos (`rm -rf`, `dd`, `mkfs`, etc.) o privilegiados (`sudo`, `shutdown`, `chmod 777`, etc.).
* **`prompt_builder_service`**:
  * Construye el contexto de aclaración cuando se requieren parámetros o se identifican advertencias de seguridad.
* **`response_formatter_service`**:
  * Formatea la propuesta para terminal (`cli`), documentación estructurada (`markdown`) o integración de API (`json`).

---

## 2. Instalación y Configuración

### Requisitos previos
* Python 3.10 o superior.
* Instalación editable desde la raíz del repositorio:
  ```bash
  pip install -e .
  ```
* La configuración local apunta a los artefactos del checkout. Una instalación
  normal usa `/usr/share/it-shell/` o un archivo pasado con `--config`.

### Configuración (`config/default.yaml`)
El comportamiento del motor se parametriza mediante `semantic_search_engine/config/default.yaml`:

```yaml
index_path: ../dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss
aliases_path: ../dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy
corpus_path: ../dataset_builder/corpus
embedding_model: sentence-transformers/all-mpnet-base-v2
reranker_model:                   # Vacío para IdentityReranker; o especificar modelo CrossEncoder
retrieval_top_k: 50              # Cantidad de vecinos recuperados en FAISS
selection:
  min_score: 0.45                # Umbral de abstención ante baja confianza
  alternative_margin: 0.04       # Margen de similitud para proponer alternativas
  max_alternatives: 3            # Límite de comandos alternativos
assistant: rules                 # rules (predeterminado) u ollama
ollama_model: qwen2.5:1.5b
ollama_base_url: http://localhost:11434
ollama_timeout: 45
```

El modo `ollama` traduce/reescribe las consultas de retrieval, selecciona entre
los candidatos recuperados y genera la explicación. El plan, los argumentos y
la plantilla final del comando siguen siendo deterministas: el texto de comando
generado por el modelo se ignora y la propuesta pasa siempre por
`ProposalValidator`. Si Ollama no está disponible, se conserva el retrieval
determinista y la explicación vuelve al generador local.

---

## 3. Ejecución desde la CLI

El comando instalado es `it-shell`; también se puede ejecutar como módulo desde
un checkout.

### Ejecución en línea
```bash
it-shell "list files in a directory"
```

Para activar Ollama solo en una ejecución, sin modificar la configuración:

```bash
it-shell "crear los directorios padre que falten en /tmp/reportes" --assistant ollama
```

Ollama debe estar ejecutándose localmente y tener descargado el modelo
configurado. La CLI no requiere conectividad al servicio cuando se usa el modo
predeterminado `rules`.

### Ejecución fuera de línea (Offline)
Para asegurar reproducibilidad y evitar llamadas a la red durante la inferencia:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
it-shell "list files in a directory"
```

### Formatos de salida soportados
* **CLI (texto para terminal)**:
  ```bash
  it-shell "list files" --format cli
  ```
* **Markdown**:
  ```bash
  it-shell "list files" --format markdown
  ```
* **JSON (estructurado con metadatos y análisis)**:
  ```bash
  it-shell "list files" --format json
  ```

### Consultas multietapa explícitas
```bash
it-shell "find files and then compress them"
```

### Servidor MCP

Instala el paquete con la dependencia MCP incluida y ejecuta:

```bash
it-shell-mcp --config ruta/al/config.yaml
```

---

## 4. Ejecución de Pruebas Unitarias

La suite de pruebas unitarias evalúa la lógica del motor de manera aislada (sin descargas de red):

```bash
PYTHONPATH=. python -m unittest discover tests
```

Pruebas cubiertas:
* `tests/test_abstention.py`: filtrado por `min_score`, abstención explícita y formateo de advertencias.
* `tests/test_planner.py`: conectores secuenciales en inglés y español, detección de complejidad y dependencias encadenadas.
* `tests/test_safety.py`: matriz de comandos seguros vs. riesgosos (destructivos, privilegiados) y garantía de no ejecución.
* `tests/test_reranker.py`: comportamiento de `IdentityReranker`, integración de `CrossEncoderReranker` y preservación de scores.

---

## 5. Límites Conocidos y Métricas

* **Línea base de recuperación**:
  * Evaluado con `sentence-transformers/all-mpnet-base-v2` sobre consultas de prueba:
    * Consultas en inglés: `CommandHit@5` ~41,7 %, `MRR` ~0,207.
    * Consultas en español: `CommandHit@5` ~8,3 %, `MRR` ~0,042 (debido a que los `intents` del corpus están en inglés y el modelo es monolingüe).
* **Abstención por baja confianza**:
  * Consultas fuera de dominio o sin similitud semántica suficiente (< `min_score: 0.45`) devuelven explícitamente `No se encontró una recomendación confiable` y emiten un aviso `[WARNING] no_candidate`.
* **Aislamiento de materiales**:
  * Los artefactos pesados (`.faiss`, `.npz`, `.npy`) se rastrean mediante Git LFS.
  * Los datasets de evaluación y benchmarks se mantienen aislados en la rama dedicada `evaluations`.
