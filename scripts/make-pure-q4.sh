#!/bin/sh
# All-4-bit Inkling file (14.8 GB instead of 17.5 GB) from the Q8_0 GGUF, then f16 loop controllers.
#   ./scripts/make-pure-q4.sh Inkling-2-Mini-Q8_0.gguf out-dir
# Expert down-projections use IQ4_NL because their 1,152-wide rows cannot use Q4_K.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
src=$1; out=${2:-.}
"$root/bin/llama-quantize" --allow-requantize --tensor-type ffn_down_exps=iq4_nl --tensor-type attn_v=q4_K \
    --output-tensor-type q4_K --token-embedding-type q4_K "$src" "$out/Inkling-2-Mini-pure-Q4.gguf" Q4_K_M
GGUF_PY="$root/build/source/gguf-py" python3 "$root/scripts/ctrl_f16.py" \
    "$out/Inkling-2-Mini-pure-Q4.gguf" "$out/Inkling-2-Mini-pure-Q4-ctrl16.gguf"
