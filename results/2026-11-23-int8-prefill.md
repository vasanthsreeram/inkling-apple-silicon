# int8 prefill and split encoding on the M5 Air, 23 Nov 2026

MacBook Air M5, 32 GB, fanless, on AC. Base flags: everything in `env.sh` before this release
(SITU CLIP GATE_MERGE SHORTCONV REL_MASK DEPTH_CACHE DEPTH_MIX REL_BIAS CTRL_FUSE CONV_STATE ORDER MOE64=32 MOE64_WS).
`orig` is `pure-Q4-ctrl16`; `tie64` is the same file rewritten by `scripts/tie64.py`.

## Prefill, `llama-bench -p 2048 -ub 1024 -b 2048 -fa 1 -r 2`, three rounds, arms alternating

| Arm | Round 1 | Round 2 | Round 3 | Mean | vs orig |
|---|---:|---:|---:|---:|---:|
| orig | 521.2 | 481.9 | 472.0 | 491.7 | — |
| orig + `W4A8` (scale per 32) | 563.8 | 539.8 | 529.9 | 544.5 | +11% |
| tie64 + `W4A8 W4A8_G64` | 610.5 | 584.2 | 578.9 | 591.2 | +20% |
| + `W4A8_T64` | 637.7 | 610.5 | 612.5 | 620.2 | +26% |
| + `W4A8_ADAPT CONV_FUSE` | 680.0 | 663.9 | 658.2 | 667.4 | +36% |

The full stack won every round. All arms drift down a few percent as the Air warms.

## Decode, `llama-bench -n 128 -r 3`, three rounds

| Arm | Round 1 | Round 2 | Round 3 | Mean |
|---|---:|---:|---:|---:|
| orig | 30.59 | 30.50 | 30.85 | 30.65 |
| orig + `METAL_ENCODERS=2` | 34.94 | 32.80 | 32.44 | 33.39 (+9%) |
| tie64 full stack | 30.16 | 30.22 | 29.00 | 29.79 |
| tie64 full stack + `METAL_ENCODERS=2` | 32.89 | 33.01 | 32.12 | 32.67 (+10%) |

With one encoder thread the CPU spends ~7.4 ms per token encoding against ~28.6 ms of GPU work and the two only partly
overlap. Two encoder threads hide the encode completely (`INKLING_METAL_TIMING`: span equals GPU time).

## Quality, `llama-perplexity` KL divergence against the unpatched `pure-Q4-ctrl16` run, 2,048-token chunks

| Arm | Same top token | Mean KLD | PPL ratio |
|---|---:|---:|---:|
| orig + `W4A8` (scale per 32) | 99.0% | 0.0011 | — |
| tie64 + full stack | 97.3% | 0.0076 | 1.0144 |

The tie64 cost comes from the file format (the same file without int8 scores the same); `T64`, `ADAPT` and `CONV_FUSE`
are bit-identical to `G64` alone. The output head stays on the untied path.

## Through `llama-server` against LM Studio

Same Air, back to back with 30 s cool-downs, ours on Q4_K_M before and after LM Studio (median of both runs),
fresh 2,048-token prompts and six 128-token decodes:

| Engine | Weights | Prefill 2K tok/s | Decode tok/s |
|---|---|---:|---:|
| LM Studio | Q4_K_M | 161 | 22.8 |
| This patch | Q4_K_M (same file) | 376 | 29.8 |
| **This patch** | **tie64, full stack** | **462** | **32.0** |

Through the server, prefill is lower than `llama-bench` because time to first token includes tokenising, the chat template
and sampling, and the Air was already warm from the runs above.

## Kept and dropped

- Kept: `W4A8_G64` (64-wide scales), `W4A8_T64` (64-token int8 tiles), `W4A8_ADAPT` (expert tiles with 32 or fewer
  tokens use the 32-token kernel; +3.5% over T64 alone), `CONV_FUSE` (prefill short convolution reads its history and
  input directly and writes the state in-kernel, removing a CONCAT that was 4% of prefill; bit-identical), `METAL_ENCODERS=2`.
- Dropped: `W4A8_PIPE`, double-buffered weight loading, 17% slower because the larger threadgroup memory cuts occupancy.
  `-ub 2048` is no faster than `-ub 1024`.
- Where prefill time goes now (`INKLING_OP_PROFILE=2`): expert matmuls ~46%, dense matmuls ~20%, attention ~7%.
