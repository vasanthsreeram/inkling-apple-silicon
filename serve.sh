#!/bin/sh
# Serve an Inkling 2 Mini GGUF with the optimised paths: ./serve.sh MODEL.gguf [PORT]
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
[ "$#" -ge 1 ] && [ -f "$1" ] || { echo "usage: ./serve.sh MODEL.gguf [PORT]" >&2; exit 2; }
INKLING_MODEL=$1 . "$root/env.sh"
exec "$root/bin/llama-server" -m "$1" -a inkling -ngl all -c 4096 -np 1 -fa on -b 2048 -ub 1024 \
    --host 127.0.0.1 --port "${2:-18766}"
