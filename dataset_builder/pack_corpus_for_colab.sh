#!/usr/bin/env bash
# Empaqueta dataset_builder/corpus en un archivo comprimido de ~2.4MB listo para Colab/Kaggle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_FILE="$REPO_ROOT/corpus.tar.gz"

echo "Empaquetando dataset_builder/corpus..."
tar -czf "$OUTPUT_FILE" -C "$REPO_ROOT/dataset_builder" corpus

echo "Archivo creado con éxito: $OUTPUT_FILE ($(du -h "$OUTPUT_FILE" | cut -f1))"
echo "Puedes subirlo directamente a Google Colab o Kaggle."

