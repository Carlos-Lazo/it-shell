# Checklist para compartir el repositorio

## Situación actual del repo

```
main          ─ ea0e680 (solo .gitignore, .gitattributes, README.md)
 └── engine   ─ 458c243 (8 commits: motor + tests + docs)      ← HEAD
evaluations   ─ 748f6c2 (rama huérfana: scripts de evaluación)
base-corpus   ─ fad4600 (webcrawling + classifier Go)
sbert/…       ─ 266f841 (embedder + indexer scripts)
```

---

## 1. Estrategia de merge

### Opción recomendada: merge `engine` → `main`

```bash
git checkout main
git merge engine --no-ff -m "merge(engine): integrate semantic search engine"
git push origin main
```

> [!IMPORTANT]
> La rama `evaluations` es **huérfana** (no comparte ancestro con `main`). Esto es intencional: contiene solo material de medición. **No se debe mergear a `main`**; quien la necesite hace `git checkout evaluations`.

### Decidir qué hacer con `base-corpus` y `sbert/all-mpnet-base-v2`

Esas ramas contienen los **scripts de construcción** del dataset y del índice. No es obligatorio mergearlas a `main`, pero sí conviene documentar su existencia.

---

## 2. Artefactos de datos (NO están en Git)

El `.gitignore` excluye `*.faiss`, `*.yaml` (corpus), y entornos virtuales. Quien clone el repo **no va a tener**:

| Artefacto | Tamaño | Ruta local | Necesario para |
|-----------|--------|------------|----------------|
| `index.faiss` | 73 MB | `dataset_builder/indexer/artifacts/all-mpnet-base-v2/` | Engine + evaluaciones |
| `aliases.npy` | 2.2 MB | mismo directorio | Engine + evaluaciones |
| Corpus YAML (15 archivos) | 7.5 MB | `dataset_builder/corpus/` | Engine (resolver comandos) + evaluaciones |
| Modelo HuggingFace (`all-mpnet-base-v2`) | 419 MB | `~/.cache/huggingface/hub/` | Engine + evaluaciones |

### Opciones para distribuirlos

| Opción | Pros | Contras |
|--------|------|---------|
| **Git LFS** (recomendado para `index.faiss` + `aliases.npy` + corpus) | Queda en el repo, clonás y listo | Requiere LFS habilitado en GitHub; límite free de 1 GB |
| **Release assets** en GitHub | No contamina el historial | Requiere paso manual de descarga |
| **Script de bootstrap** que los descargue | Automatizable | Dependencia de hosting externo (Drive, S3, etc.) |
| **Excepción al `.gitignore`** para el corpus | Simple | El índice FAISS de 73 MB sigue siendo pesado para Git normal |

> [!WARNING]
> Sin estos archivos, ni la CLI ni las evaluaciones van a funcionar. Es el **bloqueante principal** para compartir.

---

## 3. Entorno Python y dependencias

### `requirements.txt` actual ([semantic_search_engine/requirements.txt](file:///home/carlos/Documentos/facultad/semantic_search/code/semantic_search_engine/requirements.txt))

```
faiss-cpu==1.10.0
numpy==2.2.2
PyYAML==6.0.2
sentence-transformers==3.4.1
```

### Consideraciones

- **No hay `requirements.txt` en la raíz del repo.** Quien clone no sabe qué instalar si no lee el README.
- **No hay `pyproject.toml` ni `setup.py`** — el paquete no es instalable con `pip install -e .`.
- **Se usa un venv ajeno** (`dataset_builder/indexer/.venv/bin/python`) para ejecutar la CLI. Eso no es reproducible en otra máquina.
- **Python 3.11.2** — conviene documentar la versión mínima.

### Acciones sugeridas

- [ ] Crear un `requirements.txt` **en la raíz** (o un symlink) que contenga las dependencias del engine.
- [ ] Documentar en el README cómo crear el venv e instalar:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r semantic_search_engine/requirements.txt
  ```
- [ ] Alternativamente, agregar un `pyproject.toml` mínimo para que `pip install -e .` funcione.

---

## 4. Modelo de HuggingFace

El modelo `sentence-transformers/all-mpnet-base-v2` (~419 MB) se descarga automáticamente al cache de HuggingFace la primera vez. Pero:

- Las evaluaciones se corren con `HF_HUB_OFFLINE=1`, lo que **requiere que el modelo ya esté en cache**.
- Si la persona que recibe el repo tiene conexión a internet, bastará con una primera ejecución online.

### Acción sugerida

- [ ] Documentar que la primera ejecución requiere internet (descarga ~420 MB).
- [ ] Opcionalmente, proveer el comando para pre-descargar:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
  ```

---

## 5. Evaluaciones: la rama `evaluations`

### Archivos en la rama

```
evaluation/
├── evaluate_retrieval.py      ← script principal
├── visualize_metrics.py       ← genera HTML de resultados
├── retrieval_queries.json     ← 20 consultas con ground truth
├── consultas_ingles.txt       ← fuente de consultas
├── retrieval_results.json     ← resultados previos
├── metricas_retrieval.html    ← visualización previa
└── __pycache__/               ← ⚠️ debería eliminarse
```

### Problemas detectados

- **`__pycache__` commiteado** — limpiar con:
  ```bash
  git checkout evaluations
  git rm -r evaluation/__pycache__
  git commit -m "chore: remove __pycache__ from evaluations branch"
  git push origin evaluations
  ```
- **El script de evaluación importa directamente `faiss`, `numpy`, `yaml`, `sentence_transformers`** — no usa el motor (`semantic_search_engine`). Evalúa retrieval FAISS crudo. Necesita los mismos artefactos de datos (index, aliases, corpus).
- **No tiene su propio `requirements.txt`** — las dependencias son las mismas que las del engine, pero conviene explicitarlo.

---

## 6. Archivos no trackeados

`git status` muestra estos archivos sin trackear (no están en ninguna rama):

```
?? dataset_builder/        ← scripts + corpus + artifacts + venv
?? docs/                   ← documentación local
?? it-shell.zip            ← archivo comprimido
?? semantic_search_engine/retrieval_service/README.md
```

### Acción sugerida

- [ ] Decidir qué de `dataset_builder/` debería ir a `main` (al menos el corpus YAML, si no usás LFS).
- [ ] Decidir si `docs/` va a `main` o se mantiene fuera.
- [ ] Eliminar `it-shell.zip` o agregarlo a `.gitignore`.
- [ ] El `retrieval_service/README.md` suelto: ¿debería commitearse?

---

## 7. `.gitignore` — excepciones necesarias

El `.gitignore` actual bloquea **todos los `.yaml`** excepto `config/default.yaml`. Esto impide commitear los archivos de corpus:

```gitignore
*.yml
*.yaml
!semantic_search_engine/config/default.yaml
```

Si decidís incluir el corpus en Git, necesitás agregar una excepción:

```gitignore
!dataset_builder/corpus/*.yaml
```

---

## 8. Resumen de acciones prioritarias

| # | Acción | Impacto |
|---|--------|---------|
| 1 | **Merge `engine` → `main`** | Base del código disponible en rama principal |
| 2 | **Distribuir artefactos de datos** (index.faiss, aliases.npy, corpus YAML) | 🔴 Sin esto nada funciona |
| 3 | **Limpiar `__pycache__`** de la rama `evaluations` | Higiene de repo |
| 4 | **`requirements.txt` en la raíz** o instrucciones claras de setup | Reproducibilidad |
| 5 | **Documentar descarga del modelo** HuggingFace | Primera ejecución |
| 6 | **Documentar cómo correr evaluaciones** (checkout de rama, paths, comando) | Usabilidad |
| 7 | **Resolver archivos no trackeados** (dataset_builder, docs, zip) | Completitud del repo |
| 8 | **Verificar que el README de `main`** refleje la estructura final | Onboarding |
