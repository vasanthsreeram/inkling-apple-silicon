// Synthetic Inkling decode token: only the weight-reading matmuls, in model order, at the real
// quant types and shapes (48 layer passes, middle 12 layers visited 3x, 8 routed experts).
// Usage: decode-bench [classes=all] [ctrl_type=f32|f16|q8_0] [router_type=f32|f16]
// classes: comma list of attn,experts,router,ctrl,out (or all)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type parse(const char *s) {
    if (!strcmp(s, "f16")) return GGML_TYPE_F16;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q4_K")) return GGML_TYPE_Q4_K;
    return GGML_TYPE_F32;
}

int main(int argc, char **argv) {
    std::string cls = argc > 1 ? argv[1] : "all";
    auto has = [&](const char *c) { return cls == "all" || cls.find(c) != std::string::npos; };
    const ggml_type tctrl = parse(argc > 2 ? argv[2] : "f32");
    const ggml_type trout = parse(argc > 3 ? argv[3] : "f32");
    const int E = 16, NL = 24, H = 2304, FF = 1152, QD = 3072, KD = 1024;
    ggml_backend_t be = ggml_backend_metal_init();
    ggml_init_params ip = {64 * 1024 * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false), nullptr, true};
    ggml_context *ctx = ggml_init(ip);
    struct L { ggml_tensor *q, *g, *k, *v, *rel, *sink, *o, *router, *gu, *down, *cd[4], *cg[4], *cf[4]; };
    std::vector<L> W(NL);
    for (int l = 0; l < NL; ++l) {
        auto &w = W[l];
        w.q = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, QD);
        w.g = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, QD);
        w.k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, KD);
        w.v = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, KD);
        w.rel = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, 384);
        w.sink = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, 24);
        w.o = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, QD, H);
        w.router = ggml_new_tensor_2d(ctx, trout, H, 128);
        w.gu = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, H, 2 * FF, E);
        w.down = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_NL, FF, H, E);
        if (l >= 6 && l < 18) for (int c = 0; c < 4; ++c) {
            w.cd[c] = ggml_new_tensor_2d(ctx, tctrl, 2 * H, 64);
            w.cg[c] = ggml_new_tensor_2d(ctx, tctrl == GGML_TYPE_Q8_0 ? GGML_TYPE_F16 : tctrl, 64, H);
            w.cf[c] = ggml_new_tensor_2d(ctx, tctrl == GGML_TYPE_Q8_0 ? GGML_TYPE_F16 : tctrl, 64, H);
        }
    }
    ggml_tensor *out = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, 201024);
    ggml_tensor *x0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    std::vector<ggml_tensor *> ids;
    double bytes = 0;
    auto mm = [&](ggml_tensor *w, ggml_tensor *x) { bytes += ggml_nbytes(w); return ggml_mul_mat(ctx, w, x); };
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_tensor *x = x0;
    int passes = 0;
    auto layer = [&](int l, bool ctrl) {
        auto &w = W[l];
        ++passes;
        if (ctrl && has("ctrl")) for (int c = 0; c < 4; ++c) {
            ggml_tensor *z = mm(w.cd[c], ggml_concat(ctx, x, x, 0));
            ggml_tensor *gain = mm(w.cg[c], z), *forget = mm(w.cf[c], z);
            x = ggml_add(ctx, x, ggml_mul(ctx, gain, forget));
        }
        if (has("attn")) {
            ggml_tensor *q = mm(w.q, x), *g = mm(w.g, x);
            ggml_build_forward_expand(gf, mm(w.k, x));
            ggml_build_forward_expand(gf, mm(w.v, x));
            ggml_build_forward_expand(gf, mm(w.rel, x));
            ggml_build_forward_expand(gf, mm(w.sink, x));
            x = ggml_add(ctx, x, mm(w.o, ggml_mul(ctx, q, g)));
        }
        if (has("router")) x = ggml_add(ctx, x, ggml_sum(ctx, mm(w.router, x)));
        if (has("experts")) {
            ggml_tensor *id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
            ids.push_back(id);
            ggml_tensor *gu = ggml_mul_mat_id(ctx, w.gu, ggml_reshape_3d(ctx, x, H, 1, 1), id);
            ggml_tensor *a = ggml_mul(ctx, ggml_view_3d(ctx, gu, FF, 8, 1, gu->nb[1], gu->nb[2], 0),
                                      ggml_view_3d(ctx, gu, FF, 8, 1, gu->nb[1], gu->nb[2], FF * 4));
            ggml_tensor *d = ggml_mul_mat_id(ctx, w.down, a, id);
            bytes += 8.0 * (ggml_nbytes(w.gu) + ggml_nbytes(w.down)) / E;
            x = ggml_add(ctx, x, ggml_view_1d(ctx, d, H, 0));
        }
    };
    for (int l = 0; l < 6; ++l) layer(l, false);
    for (int v = 0; v < 3; ++v) for (int l = 6; l < 18; ++l) layer(l, v > 0);
    for (int l = 18; l < 24; ++l) layer(l, false);
    if (has("out")) x = mm(out, x);
    ggml_build_forward_expand(gf, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_clear(buf, 0);
    // small finite input; weights stay zero (timing only, dequant cost is data independent)
    std::vector<float> xin(H, 0.01f);
    ggml_backend_tensor_set(x0, xin.data(), 0, H * 4);
    std::mt19937 rng(7);
    for (auto *id : ids) {
        std::vector<int> e(E); for (int j = 0; j < E; ++j) e[j] = j; std::shuffle(e.begin(), e.end(), rng);
        ggml_backend_tensor_set(id, e.data(), 0, 8 * 4);
    }
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, gf);
    std::vector<double> t;
    for (int r = 0; r < 30; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf);
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    printf("%-28s nodes=%5d passes=%d weights=%7.1f MB  min %6.2f ms  med %6.2f ms  %6.1f GB/s (min)\n", cls.c_str(),
           ggml_graph_n_nodes(gf), passes, bytes / 1e6, t[0], t[t.size() / 2], bytes / t[0] / 1e6);
    return 0;
}
