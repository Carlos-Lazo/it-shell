#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=${1:-0.1.0}
OUTPUT=${2:-"$ROOT/dist/it-shell-data_${VERSION}_all.deb"}
MAINTAINER=${IT_SHELL_MAINTAINER:-"it-shell maintainers <maintainers@it-shell.invalid>"}
STAGING=$(mktemp -d)
PACKAGE_ROOT="$STAGING/package"

cleanup() {
    rm -rf -- "$STAGING"
}
trap cleanup EXIT HUP INT TERM

CORPUS_SOURCE="$ROOT/dataset_builder/corpus"
INDEX_SOURCE="$ROOT/dataset_builder/indexer/artifacts/all-mpnet-base-v2"
CORPUS_TARGET="$PACKAGE_ROOT/usr/share/it-shell/corpus"
INDEX_TARGET="$PACKAGE_ROOT/usr/share/it-shell/indexes/all-mpnet-base-v2"

test -d "$CORPUS_SOURCE"
test -f "$INDEX_SOURCE/index.faiss"
test -f "$INDEX_SOURCE/aliases.npy"
test -f "$CORPUS_SOURCE/REPRODUCING.md"

YAML_COUNT=$(find "$CORPUS_SOURCE" -maxdepth 1 -type f -name '*.yaml' | wc -l)
if [ "$YAML_COUNT" -ne 14 ]; then
    printf 'Expected 14 corpus YAML files; found %s\n' "$YAML_COUNT" >&2
    exit 1
fi

mkdir -p \
    "$PACKAGE_ROOT/DEBIAN" \
    "$CORPUS_TARGET" \
    "$INDEX_TARGET" \
    "$PACKAGE_ROOT/usr/share/doc/it-shell-data"
cp "$CORPUS_SOURCE"/*.yaml "$CORPUS_TARGET/"
cp "$INDEX_SOURCE/index.faiss" "$INDEX_SOURCE/aliases.npy" "$INDEX_TARGET/"
cp "$CORPUS_SOURCE/REPRODUCING.md" "$PACKAGE_ROOT/usr/share/doc/it-shell-data/"

cat > "$PACKAGE_ROOT/DEBIAN/control" <<EOF
Package: it-shell-data
Version: $VERSION
Section: misc
Priority: optional
Architecture: all
Maintainer: $MAINTAINER
Description: Corpus and FAISS retrieval index for it-shell
 This package installs the command corpus and matching FAISS index required by
 the it-shell semantic search engine.
EOF

mkdir -p "$(dirname -- "$OUTPUT")"
dpkg-deb --build --root-owner-group "$PACKAGE_ROOT" "$OUTPUT"
printf 'Built %s\n' "$OUTPUT"
