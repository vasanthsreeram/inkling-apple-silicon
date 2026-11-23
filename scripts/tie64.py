"""Rewrite a GGUF so every Q4_K / IQ4_NL weight shares one scale per 64 values.

The file keeps its types and byte layout (so every existing kernel still reads it); only the
values change: in Q4_K the two 32-value sub-blocks of each 64-value chunk get the same 6-bit
scale and min, and in IQ4_NL each pair of 32-value blocks gets the same fp16 scale. That lets an
int8 tensor-op kernel apply scales every 64 values instead of every 32.

usage: GGUF_PY=build/source/gguf-py python3 scripts/tie64.py SRC.gguf DST.gguf [--check]
DST is created as an APFS clone of SRC (cp -c) and rewritten in place.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", "build/source/gguf-py"))
from gguf import GGUFReader  # noqa: E402

KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
KV_MID = (KV[1:] + KV[:-1]) / 2
SKIP = ("token_embd", "output.weight")


def q4k_unpack(b):
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    s = b[:, 4:16].astype(np.int32)
    sc = np.empty((len(b), 8), np.int32)
    mn = np.empty((len(b), 8), np.int32)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    qs = b[:, 16:144].reshape(-1, 4, 32)
    q = np.empty((len(b), 4, 2, 32), np.int32)
    q[:, :, 0] = qs & 0xF
    q[:, :, 1] = qs >> 4
    return d, dmin, sc, mn, q.reshape(-1, 8, 32)


def q4k_dequant(b):
    d, dmin, sc, mn, q = q4k_unpack(b)
    return (d[:, None, None] * sc[:, :, None] * q - (dmin[:, None] * mn)[:, :, None]).reshape(-1, 256)


def fit_affine(x, iters=4):
    lo = np.minimum(x.min(1), 0.0)
    hi = x.max(1)
    s = np.maximum((hi - lo) / 15, 1e-30)
    m = -lo
    best_e = np.full(len(x), np.inf, np.float32)
    best_s, best_m = s.copy(), m.copy()
    for _ in range(iters):
        q = np.clip(np.rint((x + m[:, None]) / s[:, None]), 0, 15)
        e = ((s[:, None] * q - m[:, None] - x) ** 2).sum(1)
        better = e < best_e
        best_e[better], best_s[better], best_m[better] = e[better], s[better], m[better]
        n = x.shape[1]
        sq, sqq, sx, sxq = q.sum(1), (q * q).sum(1), x.sum(1), (x * q).sum(1)
        det = n * sqq - sq * sq
        ok = det > 0
        s_new = np.where(ok, (n * sxq - sq * sx) / np.where(ok, det, 1), s)
        m_new = np.where(ok, -(sx - s_new * sq) / n, m)
        neg = m_new < 0
        s_new = np.where(neg, sxq / np.maximum(sqq, 1e-30), s_new)
        m_new = np.where(neg, 0.0, m_new)
        s, m = np.maximum(s_new, 1e-30), m_new
    return best_s, best_m


def q4k_tie64(b):
    x = q4k_dequant(b).reshape(-1, 4, 64)
    n = len(b)
    s, m = fit_affine(x.reshape(-1, 64))
    s, m = s.reshape(n, 4), m.reshape(n, 4)
    d = (s.max(1) / 63).astype(np.float16)
    dmin = (m.max(1) / 63).astype(np.float16)
    df, dmf = d.astype(np.float32), dmin.astype(np.float32)
    sc = np.clip(np.rint(s / np.where(df > 0, df, 1)[:, None]), 0, 63).astype(np.int32)
    mn = np.clip(np.rint(m / np.where(dmf > 0, dmf, 1)[:, None]), 0, 63).astype(np.int32)
    step = df[:, None] * sc
    q = np.where(step[:, :, None] > 0,
                 np.clip(np.rint((x + (dmf[:, None] * mn)[:, :, None]) / np.where(step > 0, step, 1)[:, :, None]), 0, 15), 0
                 ).astype(np.int32)
    sc8 = np.repeat(sc, 2, 1)
    mn8 = np.repeat(mn, 2, 1)
    out = np.empty_like(b)
    out[:, 0:2] = d.reshape(-1, 1).view(np.uint8)
    out[:, 2:4] = dmin.reshape(-1, 1).view(np.uint8)
    sb = np.empty((n, 12), np.int32)
    sb[:, 0:4] = sc8[:, 0:4] | ((sc8[:, 4:8] >> 4) << 6)
    sb[:, 4:8] = mn8[:, 0:4] | ((mn8[:, 4:8] >> 4) << 6)
    sb[:, 8:12] = (sc8[:, 4:8] & 0xF) | ((mn8[:, 4:8] & 0xF) << 4)
    out[:, 4:16] = sb.astype(np.uint8)
    out[:, 16:144] = (q[:, :, :32] | (q[:, :, 32:] << 4)).astype(np.uint8).reshape(n, 128)
    return out


def iq4_dequant(b):
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    qs = b[:, 2:18]
    idx = np.concatenate([qs & 0xF, qs >> 4], 1)
    return d[:, None] * KV[idx]


def iq4_tie64(b):
    x = iq4_dequant(b).reshape(-1, 64)
    amax_i = np.abs(x).argmax(1)
    vmax = x[np.arange(len(x)), amax_i]
    best_e = np.full(len(x), np.inf, np.float32)
    best_d = np.zeros(len(x), np.float32)
    for d0 in (vmax / -127.0, vmax / 113.0):
        d = np.where(d0 != 0, d0, 1e-30).astype(np.float32)
        for _ in range(4):
            q = np.searchsorted(KV_MID, x / d[:, None])
            v = KV[q]
            e = ((d[:, None] * v - x) ** 2).sum(1)
            better = e < best_e
            best_e[better], best_d[better] = e[better], d[better]
            d = np.where((v * v).sum(1) > 0, (x * v).sum(1) / np.maximum((v * v).sum(1), 1e-30), d).astype(np.float32)
            d = np.where(d != 0, d, 1e-30)
    dh = best_d.astype(np.float16)
    df = dh.astype(np.float32)
    q = np.searchsorted(KV_MID, x / np.where(df != 0, df, 1)[:, None]).astype(np.uint8).reshape(-1, 32)
    out = np.empty_like(b)
    out[:, 0:2] = np.repeat(dh, 2).reshape(-1, 1).view(np.uint8)
    out[:, 2:18] = q[:, :16] | (q[:, 16:] << 4)
    return out


TYPES = {"Q4_K": (144, q4k_tie64, q4k_dequant), "IQ4_NL": (18, iq4_tie64, iq4_dequant)}


def work(job):
    path, off, nbytes, tname, lo, hi = job
    bs, tie, deq = TYPES[tname]
    mm = np.memmap(path, np.uint8, "r+", offset=off, shape=(nbytes // bs, bs))
    src = np.array(mm[lo:hi])
    new = tie(src)
    a, bnew = deq(src).ravel(), deq(new).ravel()
    mm[lo:hi] = new
    mm.flush()
    del mm
    return float(((a - bnew) ** 2).sum()), float((a * a).sum())


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    check = "--check" in sys.argv
    if not check:
        subprocess.run(["cp", "-c", src, dst], check=True)
    r = GGUFReader(src)
    jobs, meta = [], []
    for t in r.tensors:
        tn = t.tensor_type.name
        if tn not in TYPES or t.name.startswith(SKIP):
            continue
        bs = TYPES[tn][0]
        nb = t.n_bytes // bs
        step = 1 << 15
        for lo in range(0, nb, step):
            jobs.append((dst if not check else src, t.data_offset, t.n_bytes, tn, lo, min(nb, lo + step)))
            meta.append(t.name)
        if check:
            jobs = jobs[:1]; meta = meta[:1]
            break
    if check:
        path, off, nbytes, tname, lo, hi = jobs[0]
        bs, tie, deq = TYPES[tname]
        mm = np.memmap(path, np.uint8, "r", offset=off, shape=(nbytes // bs, bs))
        src_b = np.array(mm[lo:lo + 4096])
        new = tie(src_b)
        a, b2 = deq(src_b), deq(new)
        print(meta[0], tname, "rel rms change", np.sqrt(((a - b2) ** 2).sum() / (a * a).sum()))
        if tname == "Q4_K":
            _, _, sc, mn, _ = q4k_unpack(new)
            print("pairs tied:", bool((sc[:, 0::2] == sc[:, 1::2]).all() and (mn[:, 0::2] == mn[:, 1::2]).all()))
        return
    print(f"{len(jobs)} chunks over {len(set(meta))} tensors", flush=True)
    t0 = time.time()
    per = {}
    with ProcessPoolExecutor(6) as ex:
        for i, (res, name) in enumerate(zip(ex.map(work, jobs, chunksize=1), meta)):
            e, s = per.get(name, (0.0, 0.0))
            per[name] = (e + res[0], s + res[1])
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s {name}", flush=True)
    worst = sorted(((np.sqrt(e / s), n) for n, (e, s) in per.items()), reverse=True)
    tot_e = sum(e for e, _ in per.values()); tot_s = sum(s for _, s in per.values())
    print(f"overall rel rms change {np.sqrt(tot_e / tot_s):.4f}; worst:", [(round(v, 4), n) for v, n in worst[:5]])
    print("TIE64_DONE", flush=True)


if __name__ == "__main__":
    main()
