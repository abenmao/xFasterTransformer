// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <cmath>
#include <cstring>
#include <iostream>

#include "bfloat16.h"
#include "float16.h"
#include "rotary_embedding_kernels.h"
#include "transformer_ctx.h"

/*  Sample:
        int bs = 2 headnum = 3 seq = 4  dim = 6;
        int max_len = 10;
        int pos_ids[4] = {2,0,1,3}; //  seq = 4 , Each batch have same value
        int pos_shape[2] = {bs, seq};
        float x[144] = {0, 1, 1,...}; // bs * h * seq * dim = 144
        int xshape[4] = {bs,headnum,seq,dim};
        Forward
        LlamaRotaryEmbedding emb(dim, seq);
        float *embd = emb.forward(x, x_shape, pos_ids, pos_shape);
*/

class LlamaRotaryEmbedding {
public:
    LlamaRotaryEmbedding(DecoderContext *ctx);
    LlamaRotaryEmbedding(const int dim, const int max_position_embeddings = 2048, const float base = 10000);

    ~LlamaRotaryEmbedding() {}

    void forward(float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds);
    void forward(
            bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds);
    void forward(
            float16_t *query, float16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds);

    // For continuous batching
    void forward(float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);
    void forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);
    void forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);

private:
    void yarnFindRange(int &low, int &high, int betaFast, int betaSlow, int dim, float base, int orgMaxPosEmbed);
    void yarnLinearRampMask(float *invFreqMask, int low, int high, int dim, float extraFactor);

private:
    int inv_freq_size = -1;
    int dim = -1;
    int max_position_embeddings = -1;
    float base = -1;
    float *inv_freq = nullptr;
    float *emb_cos = nullptr;
    float *emb_sin = nullptr;
    void *device = nullptr;
};
