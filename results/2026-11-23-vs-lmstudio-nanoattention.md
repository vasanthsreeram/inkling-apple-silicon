# Inkling 2 Mini: this patch vs LM Studio vs nanoattention, 23 Nov 2026

Each machine ran all engines back to back with cool-downs (60 s Studio, 90 s Air). Script: `cmp3.sh` (see below).

- **This patch**: `llama-server`, flags from `env.sh` (+ M5 expert tiles on the Air), `-c 4096 -ub 1024`.
- **LM Studio**: llama.cpp runtime 2.80.1 (the version shipping on both Macs), `--gpu max --context-length 4096`, the **same Q4_K_M file**.
- **nanoattention**: `nano` at commit `6f782a8` (Zig 0.17.0, ReleaseFast), its documented Inkling knob set, on the original
  `thinkingmachines/Inkling-2-Mini@ed40e7da` checkpoint (NVFP4/MXFP8, 19.5 GB). It has no server, so it uses its own `nano bench`
  (seeded tokens, teacher-forced decode). Its README notes MLX parity is not yet established.
- Prefill = fresh 2,048-token prompts, prompt tokens ÷ time to first token (median of 3). Decode = six prompts × 128 tokens (median).

## Mac Studio (M3 Ultra)

| Engine | Weights | Prefill 2K tok/s | Decode tok/s |
|---|---|---:|---:|
| LM Studio | Q4_K_M | 549 | 28.6 |
| nanoattention (batch 512, its best) | original NVFP4/MXFP8 | 870 | 31.2 |
| **This patch** | Q4_K_M (same file as LM Studio) | **1,111** | **54.5** |
| **This patch** | all-4-bit + f16 controllers | **1,135** | **58.5** |

nano at 1,024 tokens: 690 (batch 128) / 831 (256) / 909 (512) prefill, 32.0 decode; batch 1,024 failed.

## MacBook Air (M5, 32 GB, fanless)

| Engine | Weights | Round 1 (ours first) prefill / decode | Round 2 (LM Studio first) prefill / decode |
|---|---|---:|---:|
| LM Studio | Q4_K_M | 81 / 16.2 | 59 / 15.2 |
| nanoattention (batch 128) | original NVFP4/MXFP8 | 101 / 8.4 | — |
| **This patch** | Q4_K_M | **256 / 25.2** | **163 / 17.0** |
| **This patch** | all-4-bit + f16 controllers | **338 / 31.9** | — |

- The fanless Air was warm for both rounds (round 2 hotter); compare rows within a round. Cool `llama-bench` peaks for this patch are 460 prefill (1K) and 33.7–37 decode.
- nano does not fit the Air well: batch 512 ran out of GPU memory, and at batch 128 the 19.5 GB checkpoint did not stay resident —
  a 1,024-token run paged in ~26 GB from SSD (137 prefill / 13.2 decode). Its Air numbers measure that, not its kernels.
