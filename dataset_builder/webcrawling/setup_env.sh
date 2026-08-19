#!/usr/bin/env bash
# Crea el entorno aislado del extractor e instala sus dependencias.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$SCRIPT_DIR/.venv"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Entorno preparado en: $VENV_DIR"
echo "Activación: source $VENV_DIR/bin/activate"