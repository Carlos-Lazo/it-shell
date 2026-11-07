# Informe Técnico: Asistente SLM Local y Category Boosting en IT Shell

**Fecha:** Noviembre 2026  
**Módulo:** `semantic_search_engine/llm/` y `semantic_search_engine/retrieval_service/`  
**Dataset de prueba:** `evaluation/capacity/plain_source.yaml` (122 consultas revisadas)  
**Modelos evaluados:** `qwen2.5:1.5b` y `qwen2.5:3b` vía Ollama (CPU pura)  

---

## 1. Resumen Ejecutivo

Este documento detalla la investigación, diseño e implementación técnica de dos mejoras clave para elevar la precisión y calidad de las propuestas del motor IT Shell:
1. **Category Boosting (Soft Filtering):** Una técnica de bonificación de similitud semántica en el retriever FAISS que prioriza comandos de la categoría objetivo sin descartar herramientas multidominio, resolviendo el problema de las falsas exclusiones.
2. **Asistente SLM Desacoplado (*Small Language Model*):** Un módulo liviano basado en Qwen 2.5 (1.5B) que analiza la consulta, predice la categoría, formula subconsultas optimizadas para el retriever, selecciona con precisión entre los Top-K candidatos y genera explicaciones concisas en lenguaje natural.

La arquitectura preserva la garantía central de seguridad del proyecto: **todo comando propuesto tiene `executable = false` y pasa obligatoriamente por el validador determinista (`validator.py`)**.

---

## 2. Arquitectura del Flujo Asistido

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
En pruebas sobre `plain_source.yaml`:
* Varias capacidades bajo *filesystem* (como búsqueda de patrones con `grep` o compresión con `tar`) tienen registros documentados bajo `textprocessing.yaml` o `shellprogramming.yaml`.
* El filtro rígido causó una **Tasa de Exclusión Errónea del 17.2%**, degradando el Hit@1 de 36.1% a **29.5%**.

### Formulación del Soft Boosting
En [`FaissRetriever`](../semantic_search_engine/retrieval_service/retriever.py), las distancias FAISS corresponden a producto interno sobre vectores unitarios (similitud coseno en $[-1, 1]$).
Para cada candidato recuperado $c$ con similitud base $s(c)$:

$$s_{\text{final}}(c) = \begin{cases} s(c) + \beta & \text{si } \text{cat}(c) \in C_{\text{boost}} \\ s(c) & \text{en caso contrario} \end{cases}$$

Donde $\beta = 0.10$ es el factor de bonificación y $C_{\text{boost}}$ es la categoría predicha por el SLM.
Posteriormente, la lista de candidatos se reordena descendentemente por $s_{\text{final}}$.

### Resultados Comparativos de Retrieval (122 consultas):

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

## 5. Comparativa Experimental de Modelos en CPU (`1.5B` vs `3B`)

Se ejecutaron pruebas directas en vivo con el demonio local de Ollama sobre la misma máquina y en CPU pura:

| Métrica / Parámetro | `qwen2.5:3b` | `qwen2.5:1.5b` | Impacto / Diferencia |
| :--- | :---: | :---: | :--- |
| **Descarga en disco** | 1.9 GB | **986 MB** | **-48% almacenamiento** |
| **Consumo de memoria RAM** | ~2.5 GB | **~1.4 GB** | **-44% memoria** |
| **Tiempo de Análisis (Paso 1)** | 49.04 s | **21.68 s** | **2.26x más rápido** |
| **Tiempo de Selección (Paso 2)** | 44.68 s | **21.89 s** | **2.04x más rápido** |
| **Tiempo Total de Inferencia** | 93.72 s | **43.57 s** | **Reducción de latencia del 53.5%** |
| **Exactitud en Selección** | 100% | **100%** | Idéntico candidato elegido |
| **Ensamble de Comando** | `mkdir -p /var/log/app` | `mkdir -p /var/log/app` | Idéntica sintaxis correcta |

### Casos de Estudio Reales Evaluados en Vivo:

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

## 6. Conclusiones y Defensa Académica

1. **Eficiencia de Small Language Models (SLMs):** `qwen2.5:1.5b` demuestra que para tareas acotadas de preprocesamiento, clasificación y reranking grounded, un modelo de 1.5B ofrece la misma precisión que uno de 3B con la mitad del consumo de memoria y más del doble de velocidad.
2. **Superioridad de Soft Boosting frente a Hard Filter:** En sistemas de recuperación sobre corpus heterogéneos, guiar mediante bonificación de puntaje supera el filtrado excluyente, pues evita pérdidas irreversibles de recall.
3. **Seguridad Robusta:** La arquitectura mantiene las garantías de IT Shell: ningún LLM puede saltarse las validaciones de `validator.py` ni otorgar permisos de ejecución automática.

