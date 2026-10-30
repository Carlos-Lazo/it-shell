# Fase 7: empaquetado y rendimiento

Fecha de medición: 2026-10-30. Este informe separa el benchmark de recuperación
controlado de la medición local de la aplicación. No se deben comparar sus
latencias como si midieran la misma operación.

## Instalación y comandos de consola

Se añadió `pyproject.toml` para instalar el proyecto con
`pip install -e ".[dev]"`. La instalación editable validada generó ambos comandos:

- `it-shell`: CLI de consulta.
- `it-shell-mcp`: servidor MCP por `stdio`.

El paquete incluye la configuración FHS `semantic_search_engine/config/installed.yaml`,
que espera los índices bajo `/usr/share/it-shell/indexes/` y los YAML bajo
`/usr/share/it-shell/corpus/`. Un checkout editable selecciona la configuración
local existente. El corpus y los índices siguen siendo datos externos al paquete
Python; no se modificaron ni se copiaron como parte de esta fase.

El paquete de datos Debian separado se construye con
`./packaging/build-data-deb.sh 0.1.0`. Se generó localmente
`dist/it-shell-data_0.1.0_all.deb` (60,018,656 bytes; aproximadamente 58 MiB),
con 14 YAML, `index.faiss`, `aliases.npy` y la guía de reproducción. El paquete
instala los datos en las rutas previstas por la configuración FHS. El `.deb` es
un artefacto local ignorado por Git; para instalar actualizaciones desde `apt`
se requiere publicarlo en un repositorio APT. Antes de una publicación oficial,
se debe reemplazar el contacto Maintainer de ejemplo mediante
`IT_SHELL_MAINTAINER="Nombre <correo>"`.

## Benchmark de recuperación comparable

La comparación oficial disponible usa 160 consultas y el snapshot de Fase 2.
Los resultados y hashes se mantienen en
[`evaluation/comparative_report.md`](../evaluation/comparative_report.md) y
[`evaluation/comparative_all.json`](../evaluation/comparative_all.json).

| Configuración | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@5 | Latencia de método |
|---|---:|---:|---:|---:|---:|---:|
| FAISS + all-mpnet-base-v2 | 0.4125 | 0.7125 | 0.8000 | 0.5377 | 0.3613 | 62.47 ms |
| BM25 Pyserini/Lucene | 0.2125 | 0.2687 | 0.3125 | 0.2408 | 0.2056 | 3.99 ms |
| Híbrido BM25 + dense (α=0.5) | 0.3125 | 0.6125 | 0.7312 | 0.4518 | 0.3073 | 172.76 ms |

FAISS sigue siendo el backend oficial: tuvo la mejor calidad entre las
configuraciones comparables. BM25 fue más rápido, pero quedó por debajo en
calidad; el híbrido no superó a FAISS.

La columna muestra `method_per_query` del comparador. El campo
`reported_total_per_query` es 69.41 ms para BM25 y 229.73 ms para el híbrido,
porque suma también la fase de recuperación que el benchmark atribuye por
separado; para FAISS, ambos valores son 62.47 ms.

No se presenta una comparación numérica de extremo a extremo «baseline v1.0 vs.
versión final». El comparativo anterior es el resultado controlado disponible, no una equivalencia
con esa comparación de versiones.

## Perfil local de la CLI

Medición de una consulta real con el índice, corpus y modelo en caché, en Linux
6.12.95+deb12-amd64, Python 3.11.2 y 4 CPUs visibles. Se forzó modo offline.

| Medición | Resultado |
|---|---:|
| Construcción del motor e índice | 0.299 s |
| Primera consulta, incluida carga lazy de MPNet | 16.996 s |
| Consultas siguientes, mediana de 5 repeticiones | 0.138 s |
| RSS tras construir el motor | 132.3 MiB |
| RSS tras cargar el modelo y consultar | 1283.4 MiB |
| Pico RSS del proceso | 1,313,696 KiB (aprox. 1.25 GiB) |

La consulta cálida queda por debajo de 400 ms, mientras que la primera consulta
es más lenta por la carga inicial de los pesos. El pico de RAM es relevante para
despliegues con memoria limitada. Es una medición de una sola máquina y una
consulta, no un percentil ni una garantía de latencia.

El Cross-Encoder permanece desactivado por defecto (`reranker_model: null`). Los
experimentos históricos reportaron entre aproximadamente 0.98 y 6.00 segundos
adicionales por consulta en CPU, pero están marcados como no comparables con el
snapshot oficial. Con el reranker desactivado y la búsqueda cálida por debajo de
400 ms, no hay evidencia suficiente para añadir ONNX Runtime al camino
predeterminado; cualquier optimización del Cross-Encoder debe esperar una
reevaluación contra el snapshot actual.

[dev]
## Verificación realizada

- `pip install --no-deps -e .` completó correctamente en el entorno de validación.
- `it-shell --help` y `it-shell-mcp --help` respondieron correctamente.
- El wheel se construyó y se verificó que incluyera ambas configuraciones y los
  dos entry points de consola.
- El `.deb` de datos se construyó y se inspeccionó su metadata y contenido.
- Las pruebas dirigidas de CLI y MCP pasaron: 9 pruebas, 11 subcasos.
- La suite completa pasó: 79 pruebas y 140 subcasos.
- La inferencia offline utilizó el modelo ya cacheado y completó correctamente.

La instalación de producción debe instalar también las dependencias declaradas
por el proyecto y proporcionar corpus e índices en la ruta configurada.
