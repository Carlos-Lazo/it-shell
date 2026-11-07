# Benchmark end-to-end de precisión — borrador v1

**Estado:** borrador con expectativas pendientes de revisión humana.  
**Objetivo:** medir si la propuesta final resuelve la tarea correctamente y de
forma segura. No es otra evaluación de retrieval: las 160 consultas y sus
Hit@k/MRR/nDCG siguen en `evaluation_v2_queries.json`.
**Dominio de esta ejecución:** comandos de sistema de Debian 12. Docker y Git
quedan fuera de esta medición y deben evaluarse como dominios independientes.

## Conjunto de casos

Los casos están en [`precision_benchmark_v1.json`](precision_benchmark_v1.json);
el esquema se valida con [`evaluate_precision_benchmark.py`](evaluate_precision_benchmark.py).
La versión actual contiene 22 casos nuevos, sin duplicados exactos del benchmark
de retrieval:

| Partición | Casos | Uso |
|---|---:|---|
| development | 13 | Analizar fallos y ajustar |
| validation | 3 | Comparar cambios antes de congelarlos |
| holdout | 6 | Evaluación final; no ajustar contra estos ejemplos |

Se cubren selección, argumentos, planificación multietapa, dependencias,
redirecciones, aclaraciones, abstención, seguridad e inglés/español. El holdout
de seis casos es deliberadamente pequeño para esta primera iteración: no ofrece
por sí solo una estimación estadística estable. El conjunto debe crecer y
revisarse antes de usarlo como criterio de lanzamiento.

Cada caso registra consulta, partición, área, idioma, dificultad, política
esperada, pasos/comandos aceptables, argumentos, dependencias, redirecciones,
datos requeridos e issues de seguridad esperados.

## Revisión humana antes de puntuar

Las 22 anotaciones tienen `status: pending`; sus políticas y comandos aceptables
son hipótesis de trabajo, no etiquetas de referencia. El evaluador solo puntúa
casos en estado `adjudicated`, que deben incluir dos revisores distintos. La
persona que revise debe:

1. confirmar o corregir la política: `propose`, `propose_with_warning`,
   `clarify` o `abstain`;
2. confirmar comandos semánticamente válidos, argumentos y pasos aceptados;
3. especificar con precisión aclaraciones y códigos de riesgo requeridos;
4. dejar holdout sin usar para ajustar planner, reglas, umbrales o prompts.

Las expectativas de casos como copiar entre dos rutas, compresión multietapa y
redirecciones en lenguaje natural requieren especial revisión: puede haber más
de una respuesta correcta o la petición puede necesitar aclaración. Las tareas
de búsqueda ahora comprueban también que filtros como `pdf`, `ERROR` y
`*.tar.gz` aparezcan en la plantilla final, no solo en los argumentos extraídos.

## Qué calcula el evaluador

El evaluador compara la salida estructurada capturada con las etiquetas
adjudicadas y reporta:

- éxito de tarea por caso y exactitud de política;
- conteo exacto de pasos tanto en el plan como en la propuesta;
- aceptación de comandos por familias explícitas;
- coincidencia exacta de argumentos identificados por tipo/valor;
- presencia en la plantilla final de filtros y valores necesarios para completar
  la tarea;
- dependencias y conectores como métricas separadas;
- operador y destino de redirección;
- recall/precision de códigos de issue esperados;
- falsos negativos y recall separados para avisos críticos;
- porcentaje de respuestas que mantienen `executable: false`.

Las métricas de retrieval se conservan aparte. No se usa un LLM para calificar las
salidas ni se considera una explicación fluida como prueba de corrección.

## Capturar y validar

Con el índice y el modelo disponibles localmente:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_precision.yaml \
  --output evaluation/precision_benchmark_v1_debian12_baseline.json

dataset_builder/indexer/.venv/bin/python \
  evaluation/evaluate_precision_benchmark.py \
  --output evaluation/precision_benchmark_v1_debian12_baseline.json \
  --split all \
  --report evaluation/precision_benchmark_v1_debian12_status.json
```

La configuración limita los resultados a las 12 categorías de comandos del
corpus Debian 12 y el capturador comprueba que coincidan con el dominio declarado
en el benchmark. No modifica ni elimina documentos del corpus o de los índices.
La captura Debian-only está en
[`precision_benchmark_v1_debian12_baseline.json`](precision_benchmark_v1_debian12_baseline.json).
Se guardan resultados crudos, revisión, configuración y hashes del snapshot. La
ejecución de referencia **no produce métricas agregadas**, porque las etiquetas
siguen pendientes. El evaluador reporta cada partición como `not_scored` hasta
que haya casos adjudicados. Guarda los hashes del benchmark y de la captura en
`precision_benchmark_v1_debian12_status.json`. No se deben cambiar los estados de
anotación para desbloquear métricas sin realizar la revisión.

### Iteraciones sin consultar el holdout

Para desarrollar cambios, captura únicamente development y validation. La opción
`--split` se puede repetir; así no se vuelve a consultar el holdout hasta la
evaluación final:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_precision.yaml \
  --split development \
  --split validation \
  --output evaluation/precision_benchmark_v1_debian12_devval_iteration_1.json
```

La captura de iteración actual está en
[`precision_benchmark_v1_debian12_devval_iteration_1.json`](precision_benchmark_v1_debian12_devval_iteration_1.json).
La captura baseline original se conserva como referencia anterior a las
correcciones posteriores. Ambas siguen sin puntuación hasta adjudicar las
etiquetas; el archivo de iteración excluye por completo los seis casos holdout.

El modo híbrido BM25 + FAISS se puede comparar sin cambiar el modo denso por
defecto. La configuración conserva el mismo filtro Debian y usa `alpha: 0.5`:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_hybrid_precision.yaml \
  --split development \
  --split validation \
  --output evaluation/precision_benchmark_v1_debian12_hybrid_devval_iteration_1.json
```

La configuración dedicada está en
[`debian12_hybrid_precision.yaml`](debian12_hybrid_precision.yaml). El modo
híbrido solicita el ranking completo antes de filtrar categorías, para evitar
que Docker o Git desplacen resultados Debian.

La comparación exploratoria está guardada en
[`precision_benchmark_v1_debian12_hybrid_devval_iteration_1.json`](precision_benchmark_v1_debian12_hybrid_devval_iteration_1.json).
Sin puntuar etiquetas, se observa que el híbrido recupera `find` para los casos
de búsqueda de logs y PDFs, pero también reemplaza propuestas correctas por
comandos no relacionados en otros casos (incluida una consulta fuera del
dominio). Por eso no se cambió la configuración densa por defecto. Los tiempos
crudos incluyen la carga diferida del modelo en la primera consulta; no son una
medición de latencia estable ni un criterio de aceptación.

### Representación enriquecida de BM25

El índice actual (`bm25_indexs_v2`) busca únicamente el nombre del comando. Para
evitar alterar ese artefacto, la variante enriquecida se construye en una ruta
nueva e indexa el nombre, la intención y el ejemplo. Los campos originales
también se conservan en el JSON raw de cada documento:

```bash
PYTHONPATH=. dataset_builder/indexer/.venv/bin/python \
  dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs_v4_enriched \
  --threads 4 \
  --representation enriched
```

Después de construirlo, se compara con el mismo conjunto sin abrir el holdout:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_hybrid_enriched_precision.yaml \
  --split development \
  --split validation \
  --output evaluation/precision_benchmark_v1_debian12_hybrid_enriched_devval_iteration_2.json
```

`--representation command` mantiene la representación histórica. La variante
enriquecida limita la intención a sus primeros 256 caracteres para evitar que
descripciones extensas de páginas de manual dominen el ranking. Sigue siendo
experimental y no se vuelve predeterminada hasta comparar la recuperación en
development/validation y revisar las anotaciones.
Cada captura registra también el SHA-256 de los archivos del índice BM25
utilizado, además del hash de la configuración, para distinguir los artefactos
de forma reproducible.

La ejecución completa sin límite alteró negativamente varias propuestas frente
al índice anterior. Limitar la intención redujo el tamaño del texto, pero no
cambió las propuestas seleccionadas en los casos revisados que comparamos. Por
lo tanto, esta representación **no demuestra una mejora** y queda como
experimento; el modo denso y el índice BM25 histórico no se modifican. La
captura acotada está en
[`precision_benchmark_v1_debian12_hybrid_enriched_devval_iteration_2.json`](precision_benchmark_v1_debian12_hybrid_enriched_devval_iteration_2.json).

### Prueba de paráfrasis curadas

Como alternativa a indexar texto extenso de manuales, preparé un conjunto
pequeño de paráfrasis para los aliases de `find`, `grep`, `df`, `ls`, `ps`, `tar`
y `cp`. No modifica el corpus: la tabla queda en
[`bm25_search_phrases_debian12.json`](bm25_search_phrases_debian12.json), y cada
captura incluye su hash junto con los hashes del índice.

La nueva representación `curated` indexa el comando con esas frases y no añade
la intención extensa ni el ejemplo. Se construye en un índice independiente:

```bash
PYTHONPATH=. dataset_builder/indexer/.venv/bin/python \
  dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs_v5_curated \
  --threads 4 \
  --representation curated \
  --search-phrases evaluation/bm25_search_phrases_debian12.json
```

La configuración y captura se limitan a development/validation:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_hybrid_curated_precision.yaml \
  --split development \
  --split validation \
  --output evaluation/precision_benchmark_v1_debian12_hybrid_curated_devval_iteration_1.json
```

La salida exploratoria está en
[`precision_benchmark_v1_debian12_hybrid_curated_devval_iteration_1.json`](precision_benchmark_v1_debian12_hybrid_curated_devval_iteration_1.json).
En la recuperación BM25 filtrada a Debian, el comando esperado apareció en
Top-15 para 11 de los 14 pasos de propuesta comparados, frente a tres pasos con
el índice anterior: ocho pasaron desde fuera de Top-15 a dentro. No apareció
para el segundo paso de `grep` ni para la petición en español. Estas son
comparaciones exploratorias de candidatos, no métricas adjudicadas ni una
afirmación de precisión end-to-end.

La variante resultó prometedora para recuperación léxica, pero aún puede
seleccionar otras herramientas o no construir los argumentos correctos. Por
eso queda optativa; no cambié el modo predeterminado ni abrí el holdout. La
selección final requiere adjudicar las etiquetas y revisar los resultados en el
contexto del sistema completo.

## Objetivos provisionales aceptados

La usuaria aceptó estos objetivos provisionales el 2026-10-30. Son criterios para
guiar la evaluación, no una aprobación de las etiquetas actuales ni un criterio
de publicación:

- 100% de invariantes de seguridad: cero falsos negativos en riesgos críticos y
  100% `executable: false`;
- éxito end-to-end de al menos 80% en development y validation, y al menos 5/6
  casos en el holdout actual;
- publicar precision/recall de políticas, argumentos y advertencias, sin
  esconder resultados por idioma, dificultad o área;
- aumentar el holdout antes de usar ese porcentaje como garantía general;
- acordar un límite de latencia p95 y memoria por entorno, separado de la
  exactitud.

Los umbrales no se deben usar para aceptar cambios hasta que se revisen y
adjudique el conjunto. El holdout de seis casos tampoco permite una conclusión
estadística general; debe ampliarse antes de considerarlo representativo.

## Diagnóstico preliminar de la línea base

Los resultados crudos muestran casos que merecen adjudicación: selección de
comandos no relacionados en búsqueda de PDF, grep, procesos y filesystem;
varias respuestas de redirección no capturan el destino; una consulta peligrosa
no produce issue crítico; el sistema no solicita aclaración ante algunas
referencias ambiguas. Son señales para orientar la revisión, **no tasas de error**
ni métricas finales, porque las etiquetas aún no están aprobadas.

La captura baseline anterior a las correcciones no refleja el comportamiento
actual para solicitudes de borrado recursivo. Ahora `rm -r` genera un issue
crítico, y una petición explícita de `rm -rf /` se rechaza sin proponer otro
comando, con una advertencia de que es extremadamente peligrosa. La corrección
se valida con pruebas unitarias y una comprobación del motor Debian. Las
iteraciones de desarrollo se limitan a development y validation; no se ha
recapturado ni usado el holdout para ajustar estos cambios.
