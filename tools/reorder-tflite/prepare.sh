#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLATC_URL="https://github.com/google/flatbuffers/releases/download/v25.9.23/Linux.flatc.binary.g++-13.zip"

if [ ! -f "$SCRIPT_DIR/schema.fbs" ]; then
  echo "Missing schema at $SCRIPT_DIR/schema.fbs" >&2
  exit 1
fi

curl -L "$FLATC_URL" -o /tmp/flatc.zip
unzip -o /tmp/flatc.zip flatc -d "$SCRIPT_DIR"
chmod +x "$SCRIPT_DIR/flatc"
rm -f /tmp/flatc.zip
