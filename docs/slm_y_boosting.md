# Informe de evidencia: asistente SLM y category boosting en IT Shell

**Revisado:** 11 de noviembre de 2026
**Módulo:** `semantic_search_engine/llm/` y `semantic_search_engine/retrieval_service/`
**Dataset de evaluación:** `evaluation/capacity/plain_source.yaml` (122 consultas revisadas)
**Estado de la evidencia:** el artefacto histórico `slm_pipeline_evaluation.json` fue generado con `MockSLMAssistant` y no evalúa Qwen. Existe además un benchmark separado, con Ollama/Qwen real, que evalúa únicamente selección sobre candidatos compartidos de BGE-M3; no es una evaluación end-to-end.

> **Conclusión principal:** las métricas de `slm_pipeline_evaluation.json` siguen siendo del mock, no de Qwen. El benchmark nuevo sí aporta evidencia reproducible de que Qwen 2.5 1.5B mejora la selección de comandos frente a reglas cuando ambos reciben los mismos candidatos de BGE-M3 (54,9% vs. 39,3% global; 44,4% vs. 27,8% en test). Es una mejora de selección, con mayor latencia y un fallback observado; no demuestra la calidad del flujo completo ni la corrección de argumentos.

---

## 1. Resumen Ejecutivo

Este documento conserva el diseño y los experimentos exploratorios de dos mejoras para IT Shell:
1. **Category Boosting (Soft Filtering):** Una técnica de bonificación de similitud semántica en el retriever FAISS que prioriza comandos de la categoría objetivo sin descartar herramientas multidominio, resolviendo el problema de las falsas exclusiones.
2. **Asistente SLM Desacoplado (*Small Language Model*):** Un módulo basado en Ollama que puede analizar la consulta y seleccionar entre candidatos recuperados. El benchmark mock no mide su calidad; el benchmark separado de BGE-M3 sí mide la selección final, con el alcance limitado descrito en la sección 6.

La arquitectura preserva la garantía central de seguridad del proyecto: **todo comando propuesto tiene `executable = false` y pasa obligatoriamente por el validador determinista (`validator.py`)**.

---

## 2. Arquitectura del Flujo Asistido

El diagrama siguiente describe el **harness experimental**, no el flujo actual
de producción ni una ejecución comprobada de Qwen. En ese harness, `MockSLMAssistant`
simula análisis y selección.

En la integración actual de la aplicación, Ollama es optativo: puede reescribir
la consulta de retrieval, seleccionar entre candidatos ya recuperados y producir
una explicación. La predicción de categoría no se usa para filtrar ni hacer
boost en el motor de producción; el planner y los argumentos siguen siendo
deterministas. El comando ensamblado por el modelo se ignora y la propuesta
pasa por `ProposalValidator`.

El pipeline opera en etapas secuenciales desacopladas:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │             Consulta del Usuario (ES / EN)             │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Etapa 1: Análisis y Enrutamiento con SLM (qwen2.5:1.5b)                                   │
│  - Categoría predicha: filesystem | networking | null                                     │
│  - Subconsulta traducida al inglés (optimizada para man pages)                            │
│  - Extracción determinista de argumentos (rutas, archivos, flags)                         │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Etapa 2: Category-Boosted Retrieval (FAISS Dense all-mpnet-base-v2)                       │
│  - Búsqueda densa en espacio de embeddings normalizados                                   │
│  - Aplicación de Soft Boost (+0.10) a candidatos de la categoría predicha                 │
│  - Reordenamiento global: genera Top-5 candidatos de alta calidad                         │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Etapa 3: Selección Grounded y Ensamblado con SLM                                          │
│  - El modelo recibe únicamente los Top-5 candidatos (alias, comando, intención, ejemplo)  │
│  - Selecciona el índice del candidato óptimo (Grounding estricto, sin alucinar)           │
│  - Ensambla el comando aplicando argumentos del usuario                                   │
│  - Redacta explicación de 1-2 oraciones en español                                        │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Etapa 4: Validación Determinista de Seguridad (validator.py)                              │
│  - Inspección de patrones destructivos (rm -r, rm -rf /, dd, etc.)                        │
│  - Marcado estricto executable = false                                                    │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │          Propuesta Estructurada al Usuario              │
                  └─────────────────────────────────────────────────────────┘
```

---

## 3. Category Boosting frente a Hard Filtering

### El problema del Hard Filtering (Filtro Exclusivo)
Al filtrar rígidamente por categoría (`WHERE category = 'filesystem'`), el motor descarta cualquier comando que pertenezca a otra categoría en el corpus.
En el harness mock sobre `plain_source.yaml`:
* Varias capacidades bajo *filesystem* (como búsqueda de patrones con `grep` o compresión con `tar`) tienen registros documentados bajo `textprocessing.yaml` o `shellprogramming.yaml`.
* El clasificador heurístico simulado tuvo una tasa de exclusión errónea del **17.2%**; el filtro rígido produjo Hit@1 de **29.5%** frente al baseline global **36.1%**. Estos valores caracterizan el mock, no Qwen.

### Formulación del Soft Boosting
En [`FaissRetriever`](../semantic_search_engine/retrieval_service/retriever.py), las distancias FAISS corresponden a producto interno sobre vectores unitarios (similitud coseno en $[-1, 1]$).
Para cada candidato recuperado $c$ con similitud base $s(c)$:

$$s_{\text{final}}(c) = \begin{cases} s(c) + \beta & \text{si } \text{cat}(c) \in C_{\text{boost}} \\ s(c) & \text{en caso contrario} \end{cases}$$

Donde $\beta = 0.10$ es el factor de bonificación y $C_{\text{boost}}$ es la categoría predicha por el asistente o asignada por el oráculo, según la variante.
Posteriormente, la lista de candidatos se reordena descendentemente por $s_{\text{final}}$.

### Resultados exploratorios de retrieval (122 consultas; modo mock)

La tabla siguiente se conserva como salida del harness, pero **no representa a Qwen**. `MockSLMAssistant` usa reglas de palabras clave para categoría y solapamiento de tokens para seleccionar. Su `search_query` es la consulta original: no se llamó a un modelo para traducir o reformular. El encoder para todas las estrategias es `all-mpnet-base-v2` (inglés), y el corpus de intenciones también está en inglés. La partición española es un desajuste lingüístico y no debe usarse para evaluar al SLM.

| Estrategia de Retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Global (Sin Filtro)** | 36.1% | 47.5% | 59.8% | 65.6% | 0.4470 |
| **2. Hard Filter (Exclusivo)** | 29.5% | 39.3% | 49.2% | 57.4% | 0.3721 |
| **3. Soft Boost (+0.10)** | **38.5%** | **50.0%** | **58.2%** | **65.6%** | **0.4661** |
| **4. Oracle Hard Filter** | 29.5% | 43.4% | 48.4% | 54.9% | 0.3717 |
| **5. Oracle Soft Boost (+0.10)** | 36.9% | 49.2% | 55.7% | 63.1% | 0.4465 |

**En la partición de Desarrollo (86 consultas):**
* **Hit@1 Global:** 38.4% $\longrightarrow$ **Hit@1 con Soft Boost:** **43.0%** (+4.6 puntos).
* **Hit@5 con Soft Boost:** **61.6%**.

---

## 4. Implementación del Asistente SLM (`ollama_assistant.py`)

### Esquemas JSON Estrictos
Para evitar texto libre fuera de formato (*chattiness*), se implementaron esquemas JSON forzados vía la API de Ollama (`format=SCHEMA`):

1. **Esquema de Análisis (`QUERY_ANALYSIS_SCHEMA`):**
   * `predicted_category`: string enum o `null`.
   * `category_confidence`: `"high" | "medium" | "low" | "none"`.
   * `language`: `"es" | "en" | "other"`.
   * `steps`: lista de pasos con `task`, `search_query` y `arguments`.
2. **Esquema de Selección (`SELECTION_SCHEMA`):**
   * `selected_index`: entero de base 1 o `null`.
   * `confidence`: nivel de certeza.
   * `assembled_command`: comando con flags y rutas aplicadas.
   * `explanation`: 1 o 2 oraciones explicativas.

### Fallback Determinista
Si Ollama no está en ejecución, se agota el tiempo de espera (`timeout = 45s`) o la respuesta no es parseable:
* La función `fallback_analysis()` clasifica con heurísticas léxicas y genera una subconsulta pasante.
* El selector de contingencia toma el candidato Top-1 de FAISS y devuelve la intención documentada como explicación.
* **El sistema nunca falla con excepción no controlada.**

---

## 5. Afirmaciones históricas de pruebas en vivo (`1.5B` vs `3B`)

Una versión anterior del documento registró estas mediciones como pruebas en vivo. En el repositorio no se hallaron logs de ejecución, solicitudes/respuestas crudas, digest de los modelos, versión de Ollama ni mediciones de recursos que permitan reproducirlas o auditarlas. Por tanto, se conservan únicamente como **notas históricas no verificadas**; no deben citarse como resultados confirmados ni como evidencia comparativa:

| Métrica / Parámetro | `qwen2.5:3b` | `qwen2.5:1.5b` | Impacto / Diferencia |
| :--- | :---: | :---: | :--- |
| **Descarga en disco** | 1.9 GB | **986 MB** | **-48% almacenamiento** |
| **Consumo de memoria RAM** | ~2.5 GB | **~1.4 GB** | **-44% memoria** |
| **Tiempo de Análisis (Paso 1)** | 49.04 s | **21.68 s** | **2.26x más rápido** |
| **Tiempo de Selección (Paso 2)** | 44.68 s | **21.89 s** | **2.04x más rápido** |
| **Tiempo Total de Inferencia** | 93.72 s | **43.57 s** | **Reducción de latencia del 53.5%** |
| **Exactitud en Selección anotada** | 100% | **100%** | Afirmación histórica sin casos anotados archivados |
| **Ensamble de Comando** | `mkdir -p /var/log/app` | `mkdir -p /var/log/app` | Idéntica sintaxis correcta |

### Casos descritos en la nota histórica (no verificables)

#### Caso A: Filesystem (Español, Parámetro anidado)
* **Entrada:** `"crear carpetas anidadas en /var/log/app"`
* **Subconsulta traducida:** `"create nested directories"`
* **Argumentos:** `{'path': '/var/log/app'}`
* **Candidatos ofrecidos:** `[1: rmdir, 2: mkdir, 3: touch]`
* **Candidato seleccionado:** `filesystem:1871` (`mkdir`)
* **Comando ensamblado:** `mkdir -p /var/log/app`
* **Explicación:** *"El comando 'mkdir' es el más adecuado para crear carpetas anidadas, ya que permite crear las carpetas y sus padres al mismo tiempo."*

#### Caso B: Networking (Español, Inspección de sockets)
* **Entrada:** `"mostrar todas las conexiones TCP y puertos en escucha"`
* **Categoría predicha:** `networking` (Confianza: `medium`)
* **Subconsulta traducida:** `"active TCP connections and open ports"`
* **Candidatos ofrecidos:** `[1: ping, 2: ss, 3: mkdir]`
* **Candidato seleccionado:** `networking:500` (`ss`)
* **Comando ensamblado:** `ss -tulpn`
* **Explicación:** *"El comando 'ss' es el más adecuado para identificar todas las conexiones TCP activas y los puertos en escucha en un sistema Debian. Este comando proporciona una vista detallada de las conexiones y puertos abiertos, lo que es perfecto para la tarea solicitada."*

---

## 6. Comparación controlada BGE-M3 + reglas frente a BGE-M3 + Ollama

La comparación usa las 122 consultas revisadas (86 development, 36 test), el
mismo índice BGE-M3, el mismo retrieval Top-15, el mismo umbral y el mismo pool
de hasta cinco candidatos elegibles para ambos selectores. Ollama/Qwen 2.5 1.5B
elige mediante una salida estructurada; las reglas eligen el primer candidato
elegible. No se evalúan análisis ni reformulación de la consulta, explicación,
ensamblado de argumentos o ejecución de comandos.

| Métrica | Reglas | Ollama |
|---|---:|---:|
| Selección correcta, total | 48/122 (39,3%) | 67/122 (54,9%) |
| Selección correcta, test | 10/36 (27,8%) | 16/36 (44,4%) |
| Selección correcta, test inglés | 8/24 (33,3%) | 12/24 (50,0%) |
| Selección correcta, test español | 2/12 (16,7%) | 4/12 (33,3%) |
| Hit@15 del retrieval compartido | 84,4% | 84,4% |
| MRR del retrieval compartido | 0,524 | 0,524 |
| Latencia media total reportada | 7,22 s | 20,45 s |

Ollama hizo 122 llamadas y registró un fallback. En `networking.tar::3`
(development, inglés), no se obtuvo una selección válida y se conservó el
candidato superior `mksquashfs`, incorrecto para la etiqueta `tar`. El informe
no conserva la causa del fallo; el caso se mantiene como incorrecto y no se
atribuye a una causa específica.

Los resultados y su procedencia están en
[`evaluation/capacity/bge_ollama_comparison.md`](../evaluation/capacity/bge_ollama_comparison.md)
y [`evaluation/capacity/bge_ollama_comparison.json`](../evaluation/capacity/bge_ollama_comparison.json).
El digest de Ollama utilizado fue
`65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`.
Los vectores BGE no incluyen hashes del texto fuente, por lo que la revisión
exacta del corpus usada para generarlos no se puede certificar
independientemente.

### Interpretación

En esta muestra el selector de Ollama mejora los aciertos aun con el mismo
retrieval, tanto en inglés como en español. La mejora observada en el test es de
6 selecciones correctas sobre 36; por el tamaño acotado del test, debe tratarse
como evidencia prometedora pero limitada, no como garantía de generalización.
La latencia media total aumentó 13,23 s. Como Hit@15 y MRR son idénticos por
diseño, Ollama no corrige los casos en que BGE-M3 no presenta un candidato
relevante dentro del Top-15. Estas cifras miden el comando del alias elegido,
no su sintaxis final, argumentos, seguridad operacional ni efecto de
ejecución.

## 7. Conclusiones que permite la evidencia disponible

1. **No se puede concluir que 1.5B iguale o supere 3B.** La tabla comparativa de latencia, memoria y exactitud no tiene artefactos de respaldo reproducibles.
2. **El reporte mock solo mide el comportamiento del simulador junto con MPNet inglés.** Su exactitud de clasificación/selección y sus latencias no deben atribuirse a Qwen. Las métricas españolas están además confundidas por el desajuste entre idioma de consulta y encoder/corpus.
3. **El benchmark real respalda la mejora del selector, no del pipeline completo.** La ganancia en este conjunto no demuestra mejor análisis de consultas, aplicación de argumentos ni exactitud funcional end-to-end.
4. **El category boosting está implementado, pero la mejora no está establecida de forma concluyente.** Los resultados existentes son exploratorios/mock y no deben confundirse con el benchmark nuevo de selección.
5. **La integración y el benchmark no verifican que las propuestas funcionen al ejecutarse.** La garantía de no ejecución y la validación determinista son propiedades del código, no evidencia de exactitud semántica o seguridad funcional completa.

El próximo hito debe evaluar el flujo end-to-end —incluidos análisis, argumentos,
abstención y resultado validado— con configuración congelada, trazas de fallback
y particiones separadas; después, comprobar efectos en un sandbox descartable.
Mantener por separado los resultados mock, la comparación de selección y la
evaluación funcional.
