#include <limits>

#include "llama.h"

template <typename WeiT>
Baichuan<WeiT>::Baichuan(const std::string &modelPath)
    : CommonDecoder<Attention<WeiT, QKPO_Dummy, RmsNorm>, LlamaMLP<WeiT>, float>(modelPath, "baichuan") {
    // Context
    DecoderContext *ctx = this->getContext();

    // Embedding (no need position embed)
    embedding = new TokenEmbedding<float16_t>(ctx);
    setEmbeddingWeights(modelPath);

    // Final LN
    setFinalLnWeight(modelPath);
}

template <typename WeiT>
Baichuan<WeiT>::~Baichuan() {
    delete embedding;
}

template <typename WeiT>
void Baichuan<WeiT>::setEmbeddingWeights(const std::string &modelPath) {
    int vocabSize = embedding->getVocabSize();
    int hiddenSize = embedding->getHiddenSize();

    float *tokenEmb = (float *)malloc(vocabSize * hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.wte.bin", tokenEmb, vocabSize * hiddenSize, this->getDataType());

    embedding->setWeights(tokenEmb);

    free(tokenEmb);
}

template <typename WeiT>
void Baichuan<WeiT>::setFinalLnWeight(const std::string &modelPath) {
    int hiddenSize = embedding->getHiddenSize();

    float *gamma = (float *)malloc(hiddenSize * sizeof(float));

    loadWeight(modelPath + "/model.final_layernorm.weight.bin", gamma, hiddenSize, this->getDataType());

    finalLN.setWeight(gamma, nullptr, hiddenSize);

    free(gamma);
}

// Prepare attention_mask which is like:
//def _get_interleave(n):
//    def _get_interleave_power_of_2(n):
//        start = (2 ** (-2 ** -(math.log2(n) - 3)))
//        ratio = start
//        return [start * ratio ** i for i in range(n)]
//
//    if math.log2(n).is_integer():
//        return _get_interleave_power_of_2(n)
//    else:
//        closest_power_of_2 = 2 ** math.floor(math.log2(n))
//        return _get_interleave_power_of_2(closest_power_of_2) + \
//               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
//def _gen_alibi_mask(n_head, max_pos):
//    import pdb
//    pdb.set_trace()
//    """used in inference only"""
//    slopes = torch.Tensor(_get_interleave(n_head))
//    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
//        n_head, -1, -1)
//    alibi = alibi.view(n_head, 1, max_pos)
//    alibi_mask = torch.triu(
//        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
//    )
//    alibi_mask = alibi_mask.unsqueeze(0) + alibi
//    return alibi_mask
//    
//alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
//if attention_mask is not None:
//    if len(attention_mask.shape) == 2:
//        expanded_mask = attention_mask.to(alibi_mask.dtype)
//        expanded_mask = torch.tril(torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
//                        ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
//    else:
//        expanded_mask = attention_mask 
//    bsz = inputs_embeds.size(0)
//    src_len, tgt_len = alibi_mask.size()[-2:]
//    expanded_mask = expanded_mask.unsqueeze(1).expand(bsz, 1, src_len, tgt_len).to(alibi_mask.dtype)
//    inverted_mask = 1.0 - expanded_mask
//    inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min)
//    attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
//else:
//    attention_mask = alibi_mask

template <typename WeiT>
void Baichuan<WeiT>::prepareAttnMask(int *ids, int step) {
    DecoderContext *ctx = this->getContext();
    int seqLen = ctx->inputSeqLen;

    if (step == 0) {
        int sizeRequired = ctx->batchSize * seqLen * seqLen;
        float *mask = this->getAttnMask(sizeRequired);
        for (int b = 0; b < ctx->batchSize; ++b) {
            auto pmask = mask + b * seqLen * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                memset(pmask + i * seqLen, 0, (i + 1) * sizeof(float)); // bottom left are 0
                std::fill_n(pmask + i * seqLen + i + 1, seqLen - i - 1, std::numeric_limits<float>::lowest());
            }
        }
    } else {
        int sizeRequired = ctx->batchSize * this->accSeqLen;
        float *mask = this->getAttnMask(sizeRequired);
        memset(mask, 0, ctx->batchSize * this->accSeqLen * sizeof(float)); // all elements are 0
    }
}

template <typename WeiT>
void Baichuan<WeiT>::embeddingForward(int *ids, float *output, int batchSize, int seqLen) {
    embedding->forward(ids, output, batchSize, seqLen);
}

template <typename WeiT>
void Baichuan<WeiT>::lastLayerNormForward(float *input, float *output, int rows) {
    finalLN.forward(input, output, rows);
}

template class Baichuan<float>;
template class Baichuan<float16_t>;
template class Baichuan<bfloat16_t>;
template class Baichuan<int8_t>;
