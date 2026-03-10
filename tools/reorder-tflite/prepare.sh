#!/usr/bin/env bash
set -e

SCHEMA_URL="https://raw.githubusercontent.com/tensorflow/tensorflow/v2.16.2/tensorflow/lite/schema/schema.fbs"
FLATC_URL="https://github.com/google/flatbuffers/releases/download/v25.9.23/Linux.flatc.binary.g++-13.zip"

curl -L "$SCHEMA_URL" -o schema.fbs
curl -L "$FLATC_URL" -o /tmp/flatc.zip

unzip -o /tmp/flatc.zip flatc -d .
rm -f /tmp/flatc.zip
