# Decode fusions, 23 Nov 2026

Machine: Mac Studio, M3 Ultra. Weights: all-4-bit with f16 controllers (`pure-Q4-ctrl16`) unless noted.
Base flags: SITU CLIP GATE_MERGE SHORTCONV REL_MASK DEPTH_CACHE DEPTH_MIX REL_BIAS.

## llama-bench, alternating (pp1024 / tg128, 3 repeats each)

| Run | fusions | pp1024 | tg128 |
|---|---|---:|---:|
| 1 | off | 1304.13 | 45.73 |
| 2 | on  | 1303.87 | 57.13 |
| 3 | on  | 1303.99 | 56.88 |

## Step by step (tg64, GPU time per token from INKLING_METAL_TIMING)

| Configuration | graph nodes | GPU ms/token | tg64 |
|---|---:|---:|---:|
| baseline, pure-Q4 | 6880 | 18.33–18.50 | 45.7–45.8 |
| baseline, pure-Q4-ctrl16 | 6880 | 18.11–18.61 | 46.2–46.3 |
| + CTRL_FUSE, pure-Q4 | 6400 | 16.73–16.86 | 51.9–52.5 |
| + CTRL_FUSE, pure-Q4-ctrl16 | 6400 | 16.53–16.62 | 52.8–53.0 |
| + CONV_STATE, pure-Q4-ctrl16 | 5440 | 15.63–15.89 | 56.6–58.0 |

Dispatches per decode token: 2,950 (1,993 barriers) → 2,086 (1,593 barriers).

## Matmul-only decode token (bench/decode_bench.cpp)

| Classes | weights MB | Air M5 min ms (GB/s) | Studio M3 Ultra min ms (GB/s) |
|---|---:|---:|---:|
| all | 2,989.7 | 27.10 (110) | 11.19 (267) |
| attention | 726.1 | 6.66 (109) | 2.61 (278) |
| experts | 1,719.9 | 16.38 (105) | 4.22 (407) |
| controllers f32 | 226.5 | 2.83 (80) | 3.16 (72) |
| controllers f16 | 113.2 | 2.10 (54) | — |
| routers | 317.2 | 2.97 (107) | 1.20 (264) |
| output head | 260.5 | 2.18 (120) | 0.62 (420) |

## Quality

- Teacher-forced, batch 1 (decode path), 8 documents × 1,024 positions, pure-Q4:
  - fusions off vs CTRL_FUSE: 8,181/8,192 (99.866%), mean NLL 1.035183 vs 1.035242
  - CTRL_FUSE vs CTRL_FUSE+CONV_STATE: 8,192/8,192, identical NLL
- pure-Q4 vs pure-Q4-ctrl16 (prefill scoring): 8,192/8,192, NLL 1.0343 both
- Server, pure-Q4-ctrl16, fusions off vs on: greedy median 45.8 → 56.7 tok/s; 11/12 short answers correct in both;
  greedy continuations identical for 5/6 prompts over 128 tokens (one flips at token 91)
