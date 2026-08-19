# Semantic Search Engine

Motor modular que propone pipelines de comandos; no ejecuta comandos de shell.

## Servicios

- `retrieval_service`: análisis, planificación de subconsultas y recuperación FAISS.
- `reranker_service`: adaptador para ranking identidad o cross-encoder.
- `pipeline_builder_service`: resolución de aliases, alternativas, dependencias y validación.
- `prompt_builder_service`: prompts de aclaración sin invocar LLM.
- `response_formatter_service`: salida JSON, Markdown o CLI.

La evaluación y las métricas son una etapa posterior y no forman parte de esta rama.

## Artefactos / Git LFS

Los vectores `.npz`, índices `.faiss` y aliases `.npy` de `dataset_builder/**` están
marcados en `.gitattributes` para Git LFS. 

#### instalado Git LFS 

``sudo apt install git-lfs ``       # Debian/Ubuntu <br>
``brew install git-lfs    ``        # macOS
<br>
####  Ejecutando

`git lfs install`


#### Atributos



```
git lfs track "dataset_builder/**/*.npz" 
git lfs track "dataset_builder/**/*.faiss" 
git lfs track "dataset_builder/**/*.npy" 
```
