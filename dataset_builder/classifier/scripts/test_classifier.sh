#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_BIN="${GO_BIN:-go}"
CLASSIFIER_GO_CACHE="${CLASSIFIER_GO_CACHE:-/tmp/classifier-go-cache}"

if ! command -v "$GO_BIN" >/dev/null 2>&1; then
  GO_BIN="/usr/local/go/bin/go"
fi

if [[ ! -x "$GO_BIN" ]] && ! command -v "$GO_BIN" >/dev/null 2>&1; then
  echo "No se encontró Go. Configurá GO_BIN con la ruta al ejecutable go." >&2
  exit 1
fi

EXAMPLES=(
  "list files in the current directory"
  "copy a file to another directory"
  "find text inside a file"
  "show running processes"
  "download a file from the internet"
  "create a bash script"
)

cd "$ROOT_DIR"
mkdir -p "$CLASSIFIER_GO_CACHE"

run_go() {
  env GOCACHE="$CLASSIFIER_GO_CACHE" "$GO_BIN" "$@"
}

# Se entrena una sola vez antes de evaluar todas las frases.
# run_go run ./cmd -train

for text in "${EXAMPLES[@]}"; do
  printf '\nConsulta: %s\n' "$text"
  run_go run ./cmd -text "$text"
done
