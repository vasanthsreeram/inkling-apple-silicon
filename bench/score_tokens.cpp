// Score fixed prefixes in the target vocabulary without storing full logit matrices on disk.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: score-tokens MODEL.gguf CORPUS.txt OUT.tsv [BATCH [WINDOW]]\n";
        return 2;
    }
    std::ifstream corpus(argv[2]);
    std::ofstream results(argv[3]);
    if (!corpus || !results) return 2;
    const int window = argc == 6 ? std::atoi(argv[5]) : 512;
    const int group = argc >= 5 ? std::atoi(argv[4]) : 64;
    if (window < 1 || window > 4096) return 2;
    if (group < 1 || group > window || window % group) return 2;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], params);
    if (!model) return 1;
    const auto *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    results << "document\tposition\tinput_id\tnext_id\targmax_id\tlogp_next\tmargin\n";
    std::string doc;
    int doc_index = 0;
    while (std::getline(corpus, doc, '\f')) {
        int count = -llama_tokenize(vocab, doc.data(), doc.size(), nullptr, 0, false, false);
        if (count < window+1) throw std::runtime_error("document too short");
        std::vector<llama_token> ids(count);
        if (llama_tokenize(vocab, doc.data(), doc.size(), ids.data(), count, false, false) != count) return 1;
        auto cp = llama_context_default_params();
        cp.n_ctx = window;
        cp.n_batch = group;
        cp.n_ubatch = group;
        cp.n_seq_max = 1;
        auto *ctx = llama_init_from_model(model, cp);
        if (!ctx) return 1;
        auto *batch = llama_batch_ext_init(ctx);
        for (int start = 0; start < window; start+=group) {
            llama_batch_ext_clear(batch);
            for (int j=0; j<group; j++) {
                llama_pos pos=start+j;
                int index=llama_batch_ext_add_token(batch, 0, ids[pos]);
                if (index < 0 || !llama_batch_ext_set_pos(batch,index,&pos)
                    || !llama_batch_ext_set_output_logits(batch,index,true)) return 1;
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return 1;
            for (int j=0; j<group; j++) {
                const float *logits=llama_get_logits_ith(ctx,j);
                if (!logits) return 1;
                int first=0;
                float second=-INFINITY;
                for (int t=1; t<n_vocab; t++) {
                    if (logits[t]>logits[first]) {
                        second=logits[first];
                        first=t;
                    } else if (logits[t]>second) {
                        second=logits[t];
                    }
                }
                double denom=0;
                for (int t=0; t<n_vocab; t++) denom+=std::exp(static_cast<double>(logits[t]-logits[first]));
                int pos=start+j;
                double logp=logits[ids[pos+1]]-logits[first]-std::log(denom);
                results<<doc_index<<'\t'<<pos<<'\t'<<ids[pos]<<'\t'<<ids[pos+1]<<'\t'
                       <<first<<'\t'<<logp<<'\t'<<logits[first]-second<<'\n';
            }
        }
        results.flush();
        std::cerr<<"scored document "<<++doc_index<<"\n";
        llama_batch_ext_free(batch);
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
