# Inkling 2 Mini on Apple Silicon

Faster local inference for [Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26B-parameter looped MoE, ~5B active per token) on Apple Silicon Macs, as a patch to llama.cpp's Metal backend.

Inkling isn't a typical MoE. Its middle 12 layers run **three times per token** (48 layer passes), and those passes are tied together by small "loop controller" networks. Stock runtimes handle that badly. This patch adds fused Metal kernels for the parts that are unique to Inkling. Each one is behind an environment flag, and every change was checked against the unpatched model for next-token agreement.

## Results

### MacBook Air (M5, 32 GB, fanless)

Cool machine, all-4-bit weights (see below), `llama-bench`, tokens/s:

| | Prefill, 1,024 tokens | Prefill, 2,048 tokens | Decode |
|---|---:|---:|---:|
| LM Studio, Q4_K_M (for reference) | ~105 | — | ~21.6 |
| This patch, previous version | 334 | 296 | 27.4 |
| This patch, prefill kernels | 460 (best ~500) | 433 | 28.6 |
| **+ decode fusions and f16 controllers** | **460** | **433** | **33.7** |

- Prefill is **~4.4× LM Studio**, and decode is **~1.6× LM Studio**.
- Decode figures come from short `tg32` runs on a cool Air. With the fusions, GPU time per token drops from ~31.5 to ~29.5 ms (5,440 graph nodes instead of 6,880).
- The Air throttles hard under sustained load. Back-to-back runs drop to roughly 240–320 prefill and 16–20 decode. The relative gains held at that temperature too.
- The decode fusions gain **+7% on the Air** and **+25% on the M3 Ultra**. The Air's slower memory makes kernel overhead a smaller share of each token.

### Mac Studio (M3 Ultra)

| | Prefill, 1,024 tokens | Decode |
|---|---:|---:|
| Patch, previous version | 1,005 | 38.5 |
| + prefill kernels | 1,304 | 45.7 |
| **+ decode fusions (this release)** | **1,304** | **57.0** |

Through `llama-server` (six 128-token greedy requests), decode went from **45.8 to 56.7 tok/s**.

### Where decode time goes

A decode token has to read about **3.0 GB of weights**: 1.72 GB of experts (8 of 128 per pass) plus 1.28 GB of everything else. `bench/decode_bench.cpp` runs only those weight-reading matmuls, in model order, at the real types and shapes:

| Decode token | Air (M5) | Studio (M3 Ultra) |
|---|---:|---:|
| Full model, GPU time, before the decode fusions | 33.6 ms | 18.4 ms |
| Weight-reading matmuls alone | 27.1 ms | 11.2 ms |
| **Glue between the matmuls** | **~6.5 ms** | **~7 ms** |

The glue costs about the same absolute time on both machines, so it's per-kernel launch and barrier latency, not memory bandwidth. Before the fusions a token ran ~2,950 kernels with ~2,000 barriers. With them, it runs **2,086 kernels with 1,593 barriers**:

- **Fused loop controllers (`INKLING_CTRL_FUSE`)**: each of the 96 controller applications per token was RMS norm ×2, concat, three tiny matmuls and a merge, about five barriers each. It is now two kernels. These matmuls had been reading their weights at only 72–80 GB/s.
- **Short convolution advances its own state (`INKLING_CONV_STATE`)**: in single-token decode, each of the 192 convolutions per token reads its history from the gathered state copy and writes the shifted window straight back to the cache. That replaces a concat plus a barriered copy, and the arithmetic is unchanged.

The Air's practical ceiling is around **37 tok/s**: 3.0 GB per token at the ~110 GB/s its kernels reach. Going past that means reading fewer bytes, not launching fewer kernels.

### Quality

Every change was compared against the unpatched path on the same weights:

| Change | Check | Result |
|---|---|---|
| Prefill kernels | 8,192 next-token predictions | 99.3% (Air) / 99.5% (Studio) agreement; mean NLL slightly *better* |
| Fused controllers | 8,192 positions, one token at a time (the decode path) | 99.87% agreement; mean NLL 1.03518 → 1.03524 (+0.006%) |
| In-place conv state | same | 100% identical to the controller-only run |
| f16 controllers and routers | 8,192 positions | 100% agreement, identical NLL |
| All-4-bit weights vs original Q4_K_M | against a Q6_K reference | 97.12% → 96.90% agreement; mean NLL +0.4% |
| Full stack via server | 12 short answers, 6 × 128-token greedy | same 11/12 correct; 5/6 continuations identical, one near-tie flips at token 91 |

The disagreements come from floating-point summation order flipping near-ties. None of the changes approximate the model, apart from the optional all-4-bit file.

## What didn't work

- **int8 activations × int4 weights (W4A8) on the M5's neural accelerators** (`INKLING_W4A8`, off by default): correct, but not faster. Q4_K stores a scale every 32 values, so the integer units stall on every block. Dense layers came out slower (3.8 vs 5.4 TFLOP/s).
- **Speculative decoding** with the model's own DSpark drafter: slower on the Air. Verifying k drafted tokens reads almost k times the expert weights, because 128 experts barely overlap, and only 45–59% of drafts are accepted.
- **MLX (oMLX runtime)**: 80–105 prefill and 3–8 decode on the same Air. The Air was swapping during that run, so treat it as a floor, not a fair comparison.
- **Core ML / ANE** for the controllers and draft projections: the planner placed single-token work on the CPU.

## Use it

```sh
./build.sh                                   # fetches llama.cpp @ 0a9105de, applies the patch, builds bin/
./serve.sh /path/to/Inkling-2-Mini-Q4_K_M.gguf   # OpenAI-compatible server on 127.0.0.1:18766
```

Benchmark:

```sh
. ./env.sh
bin/llama-bench -m MODEL.gguf -fa 1 -ub 1024 -p 1024,2048 -n 128
```

`env.sh` turns on the recommended flags, and adds the M5-only expert-tile kernel when it detects an M5. With no flags set, the binary behaves like upstream llama.cpp.

### All-4-bit weights (optional, ~10–14% faster decode)

The Q4_K_M recipe keeps the expert down-projections at Q8_0/Q5_0 and the output head at Q6_K. Making them 4-bit cuts the file from 17.5 to 14.8 GB. Storing the f32 loop controllers and routers as f16 is free (100% agreement) and saves another 142 MB read per token:

```sh
./scripts/make-pure-q4.sh Inkling-2-Mini-Q8_0.gguf ./weights
```

## Flags

| Flag | Effect | Phase |
|---|---|---|
| `INKLING_SITU`, `INKLING_CLIP`, `INKLING_GATE_MERGE`, `INKLING_SHORTCONV` | Fused SiTU activation, QK clipping, controller gate merge and 4-tap convolution | both |
| `INKLING_REL_MASK`, `INKLING_DEPTH_CACHE` | Relative-position mask, depth norms computed once | both |
| `INKLING_DEPTH_MIX`, `INKLING_REL_BIAS` | Fused depth attention; relative bias folded into the attention mask | prefill |
| `INKLING_MOE64=32`, `INKLING_MOE64_WS=1` | M5: 64-token expert tiles, 4 SIMD groups dequantise while 4 run the matmul | prefill |
| `INKLING_CTRL_FUSE`, `INKLING_CONV_STATE` | Decode fusions described above | decode |
| `INKLING_METAL_TIMING` | Print nodes, encode time and GPU time per graph | diagnostics |
| Others in the patch (`W4A8`, `NAX_TILE`, `ROUTE_MAP`, `PAIRED_GATES`, …) | Experiments kept for reproducibility; no gain measured | — |

## Repository

- `patches/inkling-apple-silicon.patch`: the complete change against `ggml-org/llama.cpp@0a9105de42d2c3e3cbc72cb1f398c743ada4f342`
- `bench/decode_bench.cpp`: the matmul-only decode token
- `bench/score_tokens.cpp`, `scripts/compare_tokens.py`: next-token agreement and NLL between two builds or files
- `bench/corpus/gsm8k-sample.txt`: scoring corpus, a sample from [GSM8K](https://github.com/openai/grade-school-math) (MIT)
- `scripts/model_bytes.py`: bytes read per decode token from a GGUF header
- `scripts/make-pure-q4.sh`, `scripts/ctrl_f16.py`: the all-4-bit and f16-controller conversions
- `results/`: raw numbers behind the tables

## Status

- The patch targets one pinned llama.cpp revision and has not been sent upstream.

## Method notes

- On the fanless Air, heat moves results by up to 2×, more than most optimisations. Comparisons there alternate before/after runs with cool-downs in between, and always use a single model file, since two 15 GB files on 32 GB evict each other.
- Bandwidth figures come from Metal copy microbenchmarks: 123–131 GB/s on a cool Air, ~115 GB/s sustained.

## License

MIT. The patch modifies llama.cpp (MIT, © the ggml authors). Model weights aren't included; get them from their publishers under their own terms.
