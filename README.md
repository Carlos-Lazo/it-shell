# IT Shell

**IT Shell** convierte solicitudes en lenguaje natural en propuestas revisables
de comandos y secuencias de CLI. Recupera herramientas de un corpus documental,
expone alternativas y advierte sobre parámetros faltantes y riesgos.

> **Seguridad por diseño:** IT Shell no ejecuta comandos. Cada propuesta es
> informativa, debe revisarse antes de utilizarse y pasa por validación
> determinista.

## Capacidades

- Recuperación semántica con FAISS y un corpus YAML organizado por categorías.
- Planificación de solicitudes compuestas en pasos explícitos.
- Abstención cuando la evidencia recuperada no supera el umbral configurado.
- Validación de parámetros y advertencias para patrones de comandos sensibles.
- Salida para terminal, Markdown y JSON.
- Asistente opcional con Ollama para seleccionar candidatos y generar
  explicaciones. El modo predeterminado usa reglas deterministas.
- Interfaz MCP para clientes compatibles con Model Context Protocol.

## Arquitectura

El repositorio contiene dos componentes principales:

- [`dataset_builder/`](dataset_builder/README.md) prepara el corpus, los
  embeddings y los índices de búsqueda.
- [`semantic_search_engine/`](semantic_search_engine/README.md) analiza
  consultas, recupera candidatos, valida propuestas y formatea respuestas.

El flujo de consulta y sus decisiones se configuran en
[`semantic_search_engine/config/default.yaml`](semantic_search_engine/config/default.yaml).
El detalle de integración MCP está en
[`docs/mcp_integration_guide.md`](docs/mcp_integration_guide.md).

## Evaluación disponible

Los resultados pertenecen a conjuntos y protocolos distintos; no deben
compararse entre sí como si fueran una única medición.

| Evaluación | Resultado relevante | Alcance |
|---|---|---|
| Retrieval oficial, 160 consultas | FAISS: MRR 0,5377; Hit@5 0,7125; Hit@10 0,8000. BM25: MRR 0,2408; Hit@5 0,2687; Hit@10 0,3125. | Benchmark histórico de retrieval. Consultar [informe y snapshot](evaluation/README.md). |
| Capacidades bilingües, 122 consultas | Con BGE-M3 compartido, selección correcta: reglas 39,3%; Ollama 54,9%. En el test de 36 casos: 27,8% y 44,4%, respectivamente. | Comparación de selección, no del flujo end-to-end. Ollama tuvo mayor latencia media y un fallback. Ver el [informe detallado](evaluation/capacity/bge_ollama_comparison.md). |

El segundo resultado usa `qwen2.5:1.5b` local. Mide únicamente si el comando
del candidato seleccionado coincide con la etiqueta: no evalúa la corrección
de argumentos, la ejecución funcional ni la calidad de todo el pipeline. Los
artefactos BGE tampoco permiten verificar de forma independiente la revisión
exacta del texto fuente usada para generar los vectores.

## Inicio rápido

### Requisitos

- Python 3.10 o superior.
- Corpus YAML y artefactos FAISS compatibles con la configuración activa.
- El modelo de embeddings configurado, disponible en caché o descargable en el
  primer uso.

Desde la raíz del repositorio, crea un entorno e instala el proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Consulta una solicitud:

```bash
it-shell "List files in a directory"
it-shell "find large log files" --format markdown
it-shell "find large log files" --format json
```

El formato predeterminado es `cli`; también están disponibles `terminal`
(colores ANSI), `markdown` y `json`. Se pueden ajustar el número de candidatos
con `--top-k` y la configuración con `--config`:

```bash
it-shell "inspect listening network ports" --top-k 10
it-shell "list files" --config path/to/config.yaml
```

### Activar Ollama

El modo determinista es el predeterminado. Para usar Ollama en una consulta,
inicia el servicio local, descarga el modelo configurado y selecciona el
backend:

```bash
ollama pull qwen2.5:1.5b
it-shell "crear los directorios padre que falten en /tmp/reportes" --assistant ollama
```

Ollama no ejecuta la propuesta. La selección se limita a candidatos recuperados
y el comando final pasa por el mismo validador.

### Usar el servidor MCP

El paquete instala el comando `it-shell-mcp`:

```bash
it-shell-mcp --config path/to/config.yaml
```

También se puede iniciar el servidor `stdio` directamente desde el checkout:

```bash
PYTHONPATH=. .venv/bin/python -m semantic_search_engine.mcp.server
```

## Datos e índices

Los datos y los artefactos de búsqueda no necesariamente están incluidos en una
instalación del paquete. Pueden distribuirse por separado como
`it-shell-data.deb`. Para construirlo desde un checkout que ya contiene el
corpus y los archivos FAISS:

```bash
./packaging/build-data-deb.sh 0.1.0
sudo apt install ./dist/it-shell-data_0.1.0_all.deb
```

En un checkout editable, la configuración local usa los datos del repositorio.
En una instalación del paquete se buscan bajo `/usr/share/it-shell/`; para una
ubicación diferente, pasa un archivo con `--config`. La guía de reconstrucción
del corpus está en
[`dataset_builder/corpus/REPRODUCING.md`](dataset_builder/corpus/REPRODUCING.md).

Para generar embeddings MPNet e índice FAISS desde un corpus disponible:

```bash
PYTHONPATH=. .venv/bin/python dataset_builder/embedder/generate_embeddings.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/embedder/artifacts

PYTHONPATH=. .venv/bin/python dataset_builder/indexer/build_index.py \
  --vectors dataset_builder/embedder/artifacts/mpnet_base_v2_vectors.npz \
  --output dataset_builder/indexer/artifacts/all-mpnet-base-v2
```

La configuración predeterminada de ejecución usa
`sentence-transformers/all-mpnet-base-v2`. BGE-M3 y las variantes BM25/híbridas
se documentan como opciones experimentales; sus índices deben configurarse
explícitamente y no sustituyen por sí mismos los artefactos predeterminados.
Consulta [`dataset_builder/README.md`](dataset_builder/README.md) y
[`evaluation/README.md`](evaluation/README.md) para los protocolos y comandos.

## Pruebas

Con las dependencias de desarrollo instaladas:

```bash
python -m pytest -q
```

Las pruebas ejercitan componentes del planner, retrieval, validación,
formateadores e integraciones. Ninguna prueba convierte IT Shell en un
ejecutor de comandos.

## Documentación

- [Estado general del proyecto](docs/estado_proyecto.txt)
- [Plan de refinamiento](docs/refinament-plan)
- [Guía del motor](semantic_search_engine/README.md)
- [Guía de construcción del corpus e índices](dataset_builder/README.md)
- [Evaluaciones y cómo reproducirlas](evaluation/README.md)
- [Comparación BGE-M3 + reglas frente a Ollama](evaluation/capacity/bge_ollama_comparison.md)
