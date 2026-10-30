# Reproducir el corpus

Esta guía resume cómo volver a generar los datos usados por `it-shell`.

## Debian 12: extracción de páginas man

Ejecuta la extracción desde el mismo entorno Debian 12 estable. El generador
consulta los comandos visibles para el usuario y las páginas man instaladas en
ese sistema:

```bash
cd dataset_builder/webcrawling/debian12
python3 corpus_generator.py /tmp/it-shell-corpus.csv
```

El CSV generado contiene el comando, una opción o sinopsis cuando corresponde y
el texto extraído. Para clasificarlo en YAML por categoría, desde
`dataset_builder/classifier`:

```bash
go run ./cmd/generatecorpus \
  -input /tmp/it-shell-corpus.csv \
  -model data/vocabulary.json \
  -output /tmp/it-shell-corpus
```

## Docker y Git: documentación oficial

El crawler Docker está en
[`webcrawling/docker/docker_crawler.py`](../webcrawling/docker/docker_crawler.py);
consulta la referencia oficial de Docker CLI y conserva la URL de cada página en
el CSV. Como el script escribe el CSV en el directorio de trabajo, ejecútalo en
un directorio temporal:

```bash
mkdir -p /tmp/it-shell-docker-crawl
cd /tmp/it-shell-docker-crawl
python3 /ruta/al/repositorio/dataset_builder/webcrawling/docker/docker_crawler.py
```

Después, desde `dataset_builder/classifier`, transforma el CSV generado:

```bash
mkdir -p /tmp/it-shell-corpus
go run ./cmd/generatedocker \
  -input /tmp/it-shell-docker-crawl/docker_cli_comandos_list.csv \
  -output /tmp/it-shell-corpus/docker.yaml
```


## Índices

Una vez disponibles los YAML, los embeddings y el índice FAISS se generan con
los scripts de `dataset_builder/embedder/` y `dataset_builder/indexer/`, según
los comandos de preparación del [README principal](../../README.md). El modelo
`sentence-transformers/all-mpnet-base-v2` puede descargarse de Hugging Face si
no está en la caché local.
