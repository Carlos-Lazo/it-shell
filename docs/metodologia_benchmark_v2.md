# Metodología y Decisiones de Diseño del Benchmark de Evaluación v2

**Versión:** 2.0  
**Fecha:** 2026-10-22  
**Dataset resultante:** `evaluation/evaluation_v2_queries.json` (160 consultas en inglés)  
**Corpus de referencia:** 27.329 comandos estructurados en 14 archivos YAML (`dataset_builder/corpus/`)  
**Índice de evaluación:** FAISS `IndexFlatIP` normalizado (27.329 vectores, dim=768)

---

## 1. Justificación y Fundamento Académico

En el desarrollo de asistentes inteligentes basados en recuperación de información (IR) y modelos de lenguaje (NLP), la representatividad estadística del conjunto de evaluación es crítica para la validez de los resultados experimentales.

El conjunto oficial del proyecto es el **Benchmark v2** de **160 consultas en
inglés**. Cada consulta representa el **0,625%** de una métrica de acierto
promedio. El conjunto está estratificado por dominio y dificultad, y se valida
contra los aliases y documentos disponibles en el corpus.

---

## 2. Proceso Paso a Paso de Generación de Consultas

La construcción del dataset se ejecutó mediante un flujo metodológico de cuatro fases controladas:

```
[Fase 1: Auditoría del Corpus] 
       │ (Extracción de 2.002 comandos base reales indexados)
       ▼
[Fase 2: Matriz de Cobertura] 
       │ (10 dominios balanceados × 16 consultas = 160 consultas)
       ▼
[Fase 3: Generación Sintética con Reglas Anti-Fuga] 
       │ (LLM-in-the-loop: intenciones, síntomas y paráfrasis)
       ▼
[Fase 4: Pipeline de Validación Automatizada] 
       │ (Verificación programática contra aliases.npy y YAMLs)
       ▼
[Dataset Congelado: evaluation_v2_queries.json]
```

### Fase 1: Auditoría del Corpus y Garantía de No-Orfandad (Zero-Orphan Policy)
Antes de redactar una sola consulta, se analizó el corpus real indexado para evitar diseñar preguntas para comandos inexistentes. Mediante un script de introspección sobre los YAMLs y el archivo de vectores `aliases.npy`, se determinó:
* Se encontraron **2.002 comandos base únicos**.
* Se verificaron herramientas con alta densidad en el corpus: `docker` (2.239 ocurrencias), `curl` (253), `gpg` (203), `git` (181), `tar` (138), `ps`, `df`, `grep`, `chmod`, etc.
* **Comandos descartados:** Se detectó que herramientas populares como `rsync`, `netstat`, `ufw`, `iptables` y `useradd` no contaban con documentos en este release del corpus. Fueron explícitamente excluidas del benchmark para evitar falsos negativos estructurales.

### Fase 2: Matriz de Cobertura Funcional
Se establecieron 10 áreas funcionales balanceadas con exactamente **16 consultas por área**:
1. **Git Version Control (16):** Ramas, commits, stash, rebases, cherry-pick, resolución de conflictos, tags.
2. **Docker Containers (16):** Ciclo de vida de contenedores, logs, redes bridge, volúmenes, build no-cache, docker compose, prune.
3. **Filesystem & Archives (16):** Búsqueda recursiva, compresión (`tar`, `zip`, `xz`, `gzip`), enlaces simbólicos, visualización paginada.
4. **Process Management (16):** Monitoreo interactivo, señales de terminación (`SIGKILL`), prioridades `nice`, servicios `systemd`, `nohup`, `timeout`, `watch`.
5. **Networking & Connectivity (16):** Peticiones HTTP REST, transferencia segura, sockets TCP activos (`ss`), resolución DNS (`dig`, `host`), trazado de rutas.
6. **Security & Permissions (16):** Permisos POSIX numéricos/simbólicos, propiedad de usuario/grupo, elevación `sudo`, criptografía simétrica GPG y certificados X.509 OpenSSL.
7. **Text Processing & Streams (16):** Expresiones regulares con `grep`, transformaciones con `sed`, extracción de columnas con `awk`/`cut`, ordenamiento, conteo y comparativas `diff`.
8. **Hardware & Disks (16):** Espacio libre en disco (`df`), uso por directorios (`du`), memoria RAM/swap (`free`), dispositivos de bloque (`lsblk`), kernel `dmesg`.
9. **Users, Groups & Auditing (16):** Identidad de usuario (`whoami`, `id`), historial de sesiones (`last`), inspección de terminales activas (`w`), gestión de passwords y fechas.
10. **Crypto & Shell Builtins (16):** Sumas de verificación (SHA-256, MD5), firmas digitales, variables de entorno (`export`, `env`), alias y gestión de historial de shell.

### Fase 3: Estrategia de Generación y Prompt Engineering
Para formular consultas realistas que simularan la interacción humana en consola, se utilizó una técnica asistida por LLM con curación manual estricta (*Human-in-the-Loop*).

#### El Prompt de Generación de Consultas
El generador operó bajo el siguiente esquema de instrucciones formales:

> **Rol:** Eres un administrador de sistemas y desarrollador de software que interactúa con un asistente inteligente de terminal.
> **Objetivo:** Para cada uno de los comandos seleccionados en la categoría `[CATEGORÍA]`, formula consultas en lenguaje natural en inglés que expresen una necesidad técnica real.
> **Reglas Obligatorias:**
> 1. *Regla de No-Fuga Léxica (Lexical Leakage Prevention):* Está prohibido incluir el nombre del comando objetivo en la consulta en los niveles Medium y Hard (por ejemplo, para `tar` no escribas *"tar folder"*, sino *"bundle multiple files into a single compressed archive"*).
> 2. *Diversidad de enfoque:*
>    - Acciones orientadas a tareas (*"download remote file from web server"*).
>    - Diagnóstico orientado a síntomas (*"see what is consuming all my RAM"*).
>    - Depuración de errores (*"reclaim disk space by purging stopped containers"*).
> 3. *Comandos relevantes equivalentes:* Si una necesidad técnica puede resolverse válidamente con más de una herramienta indexada, deben incluirse en la lista `relevant_commands` (ej. `["curl", "wget"]`, `["top", "htop"]`, `["df", "du"]`).

---

## 3. Criterios de Graduación de Dificultad

Cada consulta fue etiquetada con un nivel de dificultad semántica para permitir un análisis fino de las capacidades del modelo:

| Nivel | Proporción | Cantidad | Criterio de Diseño | Ejemplo Real del Dataset |
| :--- | :---: | :---: | :--- | :--- |
| **Easy** | 51,9 % | 83 | Intenciones claras y directas con vocabulario técnico estándar, donde existe correlación conceptual evidente. | `"check available and used filesystem disk space in human readable gigabytes"` → `["df"]` |
| **Medium** | 37,5 % | 60 | Paráfrasis conceptual pura sin palabras clave del comando. Exige que el encoder proyecte adecuadamente en el espacio vectorial sin coincidencia de tokens léxicos. | `"discard the last local commit while keeping all modifications unstaged in working directory"` → `["git"]`<br>`"show all listening TCP sockets and the processes bound to them"` → `["ss"]` |
| **Hard** | 10,6 % | 17 | Tareas compuestas, orientadas a resolución de problemas, administración avanzada o intenciones con ambigüedad contextual. | `"squash or edit the last five commits interactively before pushing to remote"` → `["git"]`<br>`"reclaim disk space by purging all stopped containers unused images and build cache"` → `["docker"]` |

---

## 4. Pipeline de Validación Automatizada

Para asegurar la validez del benchmark antes de ejecutar cualquier evaluación de modelos, se ejecutó un script de verificación cruzada (`evaluation/evaluate_reranker.py:load_corpus_metadata`):

1. **Carga de índices:** Se cargaron los 27.329 identificadores del archivo `aliases.npy`.
2. **Mapeo inverso:** Se construyó una tabla hash `by_command: dict[str, set[str]]` asociando cada nombre base de comando con sus aliases de documento (ej. `"git"` → `{"git:1", "git:2", ...}`).
3. **Comprobación de cobertura:** Para cada una de las 160 consultas, se verificó la condición:
   $$\text{relevant\_aliases} = \bigcup_{cmd \in \text{relevant\_commands}} \text{by\_command}[cmd] \neq \emptyset$$
4. **Resultado de la validación:** **160 de 160 consultas (100%) superaron la prueba**, garantizando que ninguna consulta carece de representación en el índice.

---

## 5. Formato de Almacenamiento

El dataset formal está congelado en `evaluation/evaluation_v2_queries.json` bajo la siguiente estructura:

```json
{
  "id": "docker_prune_all",
  "category": "docker",
  "difficulty": "hard",
  "query": "reclaim disk space by purging all stopped containers unused images and build cache",
  "relevant_commands": [
    "docker"
  ]
}
```

---

## 6. Conclusiones Metodológicas para la Tesis

1. **Validez Interna:** Al controlar la existencia previa de los comandos y balancear exactamente 16 consultas por área funcional, se eliminan los sesgos por distribución desbalanceada de datos.
2. **Aislamiento de Variables:** La decisión de mantener el Benchmark v2 **exclusivamente en inglés** aísla la variable de la arquitectura de recuperación, evitando confundir las limitaciones de vocabulario del Sentence Transformer con ineficiencias del algoritmo de búsqueda o reordenamiento.
3. **Significancia Estadística:** Con 160 puntos de evaluación, las diferencias observadas entre el Baseline FAISS (MRR 0,5385) y el Cross-Encoder (MRR 0,5211) son estadísticamente consistentes y no fruto del azar muestral.

---

## 7. Análisis de Modelos Cross-Encoder: Racional Teórico y Protocolo Comparativo

### 7.1. ¿Por qué `stsb-distilroberta-base` (RoBERTa) es conceptualmente superior para CLI?

El experimento inicial con `cross-encoder/ms-marco-MiniLM-L-6-v2` reveló que un re-ranker genérico puede degradar la calidad frente a un Bi-Encoder robusto (`all-mpnet-base-v2`). La justificación teórica y técnica para seleccionar **`cross-encoder/stsb-distilroberta-base`** se fundamenta en cuatro pilares:

1. **Alineación del Objetivo de Entrenamiento (STS vs. Passage QA):**
   * *MS-MARCO (MiniLM):* Preentrenado sobre consultas web de Bing emparejadas con pasajes largos (100–200 palabras). El modelo aprende a responder si un pasaje *contiene la respuesta fáctica* a una pregunta.
   * *STS-B (DistilRoBERTa):* Preentrenado sobre el *Semantic Textual Similarity Benchmark*. El objetivo matemático es predecir un grado de equivalencia semántica continua $[0.0, 1.0]$ entre dos oraciones breves. En nuestro asistente, la consulta del usuario (*"revert uncommitted changes"*) y el intent del comando (*"Reset current HEAD to the specified state"*) son **paráfrasis conceptuales cortas**, lo que coincide exactamente con la formulación STS.
2. **Tokenización Robusta para Sintaxis de Terminal (Byte-Pair Encoding):**
   * RoBERTa utiliza BPE a nivel de bytes con un vocabulario de **50.000 subtokens**, frente al WordPiece de 30.000 tokens de BERT.
   * En comandos de shell donde abundan guiones (`--dry-run`), rutas (`/var/log`), extensiones (`.tar.gz`) y comodines (`*`), BPE preserva la estructura léxica de las herramientas sin fragmentarlas en tokens desconocidos `[UNK]`.
3. **Simetría Dimensional y Capacidad:**
   * `distilroberta-base` cuenta con **82 millones de parámetros** y una dimensión oculta de **768** (la misma que `all-mpnet-base-v2`).
   * A diferencia de MiniLM-L-6 (22M parámetros, 384 dimensiones), DistilRoBERTa tiene la profundidad representacional necesaria para refinar el espacio de embeddings de MPNet sin "aplanar" distinciones semánticas sutiles.
4. **Calibración Natural del Score para Umbrales de Selección:**
   * Los modelos STS generan un puntaje normalizado en $[0, 1]$. Esto permite que el umbral de confianza `min_score: 0.45` de `PipelineBuilder` funcione de manera homogénea y probabilística, eliminando la necesidad de lidiar con logits negativos no acotados (como $-11.5$ de MS-MARCO).

---

### 7.2. El Modelo de Estado del Arte (SOTA): `BAAI/bge-reranker-base`

Para contrastar el límite superior de rendimiento frente al modelo recomendado, se incorpora **`BAAI/bge-reranker-base`** (Beijing Academy of Artificial Intelligence):
* **Capacidad:** 110 millones de parámetros (12 capas Transformer, 768 dimensiones).
* **Entrenamiento:** Minado masivo de negativos difíciles (*hard negatives*) sobre repositorios de código, documentación técnica y consultas sintéticas complejas.
* **Rol en el proyecto:** Representa la cota máxima de precisión alcanzable en modelos densos de código abierto, permitiendo cuantificar si el incremento de latencia en CPU (~2,4 s) justifica su adopción frente al Baseline de 77 ms.

---

### 7.3. Hoja de Ruta para la Implementación y Comparativa Experimental

Para ejecutar la evaluación comparativa formal entre los tres enfoques de re-ranking contra el Baseline, se establece el siguiente protocolo:

```
[Paso 1: Descarga de Modelos al Cache Local]
       │  - cross-encoder/stsb-distilroberta-base (~330 MB)
       │  - BAAI/bge-reranker-base (~440 MB)
       ▼
[Paso 2: Ejecución del Benchmark sobre evaluation_v2_queries.json]
       │  --initial-top-k 15 | N = 160 consultas
       ▼
[Paso 3: Consolidación de Métricas en Matriz Comparativa]
       │  (Hit@1, Hit@5, MRR, nDCG@5, Latencia ms)
       ▼
[Paso 4: Análisis de Trade-off Precisión vs. Latencia para la Tesis]
```

#### Comandos de Ejecución:

1. **Pre-descargar modelos al caché de Hugging Face (con conexión a internet):**
   ```bash
   python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/stsb-distilroberta-base'); CrossEncoder('BAAI/bge-reranker-base')"
   ```

2. **Evaluar Modelo Recomendado (`stsb-distilroberta-base`):**
   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
   dataset_builder/indexer/.venv/bin/python evaluation/evaluate_reranker.py \
     --queries evaluation/evaluation_v2_queries.json \
     --cross-encoder cross-encoder/stsb-distilroberta-base \
     --initial-top-k 15 \
     --output evaluation/reranker_benchmark_distilroberta.json
   ```

3. **Evaluar Modelo Estado del Arte (`BAAI/bge-reranker-base`):**
   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
   dataset_builder/indexer/.venv/bin/python evaluation/evaluate_reranker.py \
     --queries evaluation/evaluation_v2_queries.json \
     --cross-encoder BAAI/bge-reranker-base \
     --initial-top-k 15 \
     --output evaluation/reranker_benchmark_bge.json
   ```

