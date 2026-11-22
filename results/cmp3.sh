#!/bin/zsh
# Three-way comparison on one Mac: our llama-server, LM Studio (same Q4_K_M file), nanoattention (original checkpoint).
# usage: cmp3.sh SERVER_BIN Q4KM_GGUF LMS_MODEL_ID [EXTRA_ENV...]
cd ~/inkling-inference
SRV=$1; GG=$2; LID=$3; shift 3
L=runs/cmp3-$(hostname -s)-$(date +%H%M).jsonl; mkdir -p runs
cool() { sleep ${COOL:-60}; }
export INKLING_SITU=1 INKLING_CLIP=1 INKLING_GATE_MERGE=1 INKLING_SHORTCONV=1 INKLING_REL_MASK=1 INKLING_DEPTH_CACHE=1 INKLING_DEPTH_MIX=1 INKLING_REL_BIAS=1 INKLING_CTRL_FUSE=1 INKLING_CONV_STATE=1 INKLING_ORDER=1
for e in "$@"; do export $e; done
ours() {
  $SRV -m $1 -a inkling -ngl all -c 4096 -np 1 -fa on -b 2048 -ub 1024 --host 127.0.0.1 --port 18772 > runs/cmp3-srv.log 2>&1 &
  P=$!; until curl -sf http://127.0.0.1:18772/health >/dev/null; do sleep 2; done
  python3 bench.py "ours-$2" speed prefill2k --url http://127.0.0.1:18772 --ledger $L > /dev/null
  kill $P; wait $P 2>/dev/null
}
lms() {
  ~/.lmstudio/bin/lms server start --port 1234 >/dev/null 2>&1
  ~/.lmstudio/bin/lms load "$LID" --gpu max --context-length 4096 --identifier inkling -y >/dev/null 2>&1
  python3 bench.py "lmstudio" speed prefill2k --lmstudio --url http://127.0.0.1:1234 --model inkling --ledger $L > /dev/null
  ~/.lmstudio/bin/lms unload --all >/dev/null 2>&1; ~/.lmstudio/bin/lms server stop >/dev/null 2>&1
}
nano() {
  local K=$(cat nano-knobs.txt)
  for p in 2048; do
    nanoattention/zig-out/bin/nano bench models-hf/Inkling-2-Mini --context $((p+192)) --prompt-tokens $p --decode-tokens 128 --repeats 2 --knob batch=${NANO_BATCH:-512} ${=K} 2>/dev/null \
      | python3 -c "import json,sys,time; d=json.load(sys.stdin); print(json.dumps(dict(label='nano',mode='nano_bench',prompt=$p,prefill_tps=d['prefill_tokens_per_second'],decode_tps=d['decode_tokens_per_second'],unix=time.time())))" >> $L
  done
}
if [ -n "$REVERSE" ]; then cool; lms; cool; ours $GG q4km; else cool; ours $GG q4km; cool; lms; cool; nano; cool; ours ${PURE:-local-weights/pure-Q4-ctrl16.gguf} pureq4; fi
python3 - $L <<'PY'
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1])]
for lab in dict.fromkeys(r["label"] for r in rows):
    g=[r for r in rows if r["label"]==lab]
    if lab=="nano":
        for r in g: print(f"{lab:14s} prefill {r['prompt']} {r['prefill_tps']:7.1f}  decode {r['decode_tps']:5.2f} (nano bench, teacher-forced)")
        continue
    dec=[r["wall_decode_tps"] for r in g if r["mode"]=="speed" and r.get("wall_decode_tps")]
    pre=[r["prompt_tokens"]/r["ttft"] for r in g if r["mode"]=="prefill" and r.get("prompt_tokens")]
    print(f"{lab:14s} prefill 2K {st.median(pre):7.1f}  decode {st.median(dec):5.2f}  (median of {len(pre)} / {len(dec)})")
PY
