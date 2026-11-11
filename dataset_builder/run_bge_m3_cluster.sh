#!/usr/bin/env bash
# ==============================================================================
# Script para generar embeddings e índice FAISS con BAAI/bge-m3 en cluster (CPU)
# Ubicación: al mismo nivel que los directorios corpus/ y artifacts/.
# ==============================================================================
set -euo pipefail

THREADS="${1:-24}"
BATCH_SIZE="${2:-32}"
MODEL="${3:-BAAI/bge-m3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directorios al mismo nivel que este script
CORPUS_DIR="$SCRIPT_DIR/corpus"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
VECTORS_FILE="$ARTIFACTS_DIR/bge_m3_vectors.npz"
INDEX_DIR="$ARTIFACTS_DIR/bge-m3"

# Scripts al mismo nivel (con fallback por si se ejecuta dentro del repo original)
if [ -f "$SCRIPT_DIR/generate_embeddings.py" ]; then
    EMBEDDER_SCRIPT="$SCRIPT_DIR/generate_embeddings.py"
else
    EMBEDDER_SCRIPT="$SCRIPT_DIR/embedder/bge-m3/generate_embeddings.py"
fi

if [ -f "$SCRIPT_DIR/build_index.py" ]; then
    INDEXER_SCRIPT="$SCRIPT_DIR/build_index.py"
else
    INDEXER_SCRIPT="$SCRIPT_DIR/indexer/bge-m3/build_index.py"
fi

# Configuración de 24 hilos de CPU en variables de entorno
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=true

# Crear directorio artifacts/ y artifacts/bge-m3 si no existen
mkdir -p "$ARTIFACTS_DIR" "$INDEX_DIR"

# Localizar intérprete de Python
if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
    PY="$PYTHON_BIN"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/indexer/.venv/bin/python" ]; then
    PY="$SCRIPT_DIR/indexer/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PY="$SCRIPT_DIR/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "ERROR: No se encontró un intérprete de Python válido." >&2
    exit 1
fi

echo "===================================================================="
echo " Ejecución de BAAI/bge-m3 en Cluster CPU"
echo "===================================================================="
echo " Directorio base:   $SCRIPT_DIR"
echo " Intérprete Python: $PY"
echo " Hilos de CPU:      $THREADS"
echo " Batch size:        $BATCH_SIZE"
echo " Modelo:            $MODEL"
echo " Directorio corpus: $CORPUS_DIR"
echo " Vector salida:     $VECTORS_FILE"
echo " Índice salida:     $INDEX_DIR"
echo "===================================================================="

START_TOTAL=$(date +%s)

# Paso 1: Generación de Embeddings
echo ""
echo "[Paso 1/3] Generando embeddings con $MODEL..."
START_STEP1=$(date +%s)
"$PY" "$EMBEDDER_SCRIPT" \
    --corpus "$CORPUS_DIR" \
    --output "$ARTIFACTS_DIR" \
    --vector-name "bge_m3_vectors.npz" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --threads "$THREADS"
END_STEP1=$(date +%s)
echo "✓ Paso 1 finalizado en $((END_STEP1 - START_STEP1)) segundos."

# Paso 2: Construcción del Índice FAISS
echo ""
echo "[Paso 2/3] Construyendo índice FAISS FlatIP (1024 dimensiones)..."
START_STEP2=$(date +%s)
"$PY" "$INDEXER_SCRIPT" \
    --vectors "$VECTORS_FILE" \
    --output "$INDEX_DIR" \
    --threads "$THREADS"
END_STEP2=$(date +%s)
echo "✓ Paso 2 finalizado en $((END_STEP2 - START_STEP2)) segundos."

# Paso 3: Verificación de Integridad
echo ""
echo "[Paso 3/3] Verificando índice y aliases..."
"$PY" -c "
import faiss
import numpy as np
from pathlib import Path

index_path = Path('$INDEX_DIR/index.faiss')
aliases_path = Path('$INDEX_DIR/aliases.npy')

assert index_path.exists(), f'No existe {index_path}'
assert aliases_path.exists(), f'No existe {aliases_path}'

index = faiss.read_index(str(index_path))
aliases = np.load(str(aliases_path), allow_pickle=False)

assert index.ntotal == len(aliases), f'Desalineado: {index.ntotal} != {len(aliases)}'
assert index.d == 1024, f'Dimensión esperada 1024, obtenida: {index.d}'

print('Verificación exitosa:')
print(f' - Vectores indexados: {index.ntotal}')
print(f' - Dimensión:          {index.d}')
print(f' - Aliases cargados:   {len(aliases)}')
"

END_TOTAL=$(date +%s)
echo ""
echo "===================================================================="
echo " Proceso completado con éxito en $((END_TOTAL - START_TOTAL)) segundos."
echo " Archivos listos en: $INDEX_DIR"
echo "===================================================================="
