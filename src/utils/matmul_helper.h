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
#include <immintrin.h>
#include "allocator.h"
#include "bert_util.h"
#include "bfloat16.h"
#include "dtype.h"
#include "environment.h"
#include "float16.h"
#include "fp8_e4m3.h"
#include "intrinsics_util.h"
#include "my_types.h"
#include "normal_float4x2.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_version.h"
#include "split_util.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "uint4x2.h"
#include "verbose.h"
#include "xdnn.h"

#include <cstring>
#include <map>
#include <tuple>

class MMHelper {
public:
    MMHelper(xft::DeviceKind device_kind, int idx) {
        if (device_kind == xft::DeviceKind::iCPU) {
            kind = dnnl::engine::kind::cpu;
            engine = new dnnl::engine(kind, idx);
            stream = new dnnl::stream(*engine);
        } else if (device_kind == xft::DeviceKind::iGPU) {
            kind = dnnl::engine::kind::gpu;
            engine = new dnnl::engine(kind, idx);
            stream = new dnnl::stream(*engine);
        } else {
            std::cerr << "[Error] Wrong device type." << std::endl;
            std::exit(-1);
        }

        AMXThresholdM = Env::getInstance().getAMXThresholdM();
        primitiveCacheM = Env::getInstance().getPrimitiveCacheM();
        cpu_engine = new dnnl::engine(dnnl::engine::kind::cpu, 0);
        cpu_stream = new dnnl::stream(*cpu_engine);
    }

    template <typename Tin, typename Twei, typename Tout, typename Tbias = float>
    void invoke_onednn_gemm_compute(bool transA, bool transB, int M, int N, int K, float alpha, const Tin *A, int lda, const Twei *B,
            float beta, Tout *C, int ldc, const Tbias *bias = nullptr, const Tin *res = nullptr, int ldres = -1) {
        xft::Matrix<Twei> packedB;
        xft::Matrix<Twei> matrixB;

        int rows = transB ? N : K;
        int cols = transB ? K : N;
        matrixB.Resize(rows, cols);

        // TODO: Need further performance optimization for tanspose
        Tin* AT = (Tin*)malloc(M * K * sizeof(Tin));
        if (transA) {
#pragma omp parallel for
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < M; j++) {
                    AT[j * K + i] = A[i * M + j];
                }
            }
        } else {
            memcpy(AT, A, M * K * sizeof(Tin));
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < K; i++) {
            Twei *dst = matrixB.Data() + i * N;
            const Twei *src = B + i * N;
            memcpy(dst, src, N * sizeof(Twei));
        }

        packWeight(transB, matrixB, packedB);

        if constexpr (std::is_same_v<Twei, bfloat16_t>) {
            onednn_amx_gemm_compute(transA, M, N, K, alpha, AT, lda, packedB.Data(), beta, C, ldc);
        } else {
            onednn_gemm_compute(transA, M, N, K, alpha, AT, lda, packedB.Data(), beta, C, ldc);
        }
        free(AT);
    }

    ~MMHelper() {
        if (engine) delete engine;
        if (stream) delete stream;

        for (auto &pair : matmul_hub) {
            dnnl::matmul::primitive_desc *primitive_desc_ptr = std::get<0>(pair.second);
            dnnl::matmul *matmul_ptr = std::get<1>(pair.second);

            delete primitive_desc_ptr;
            delete matmul_ptr;
        }
    }

    // Pack the MatMul weight from 'src(rows, cols)' to 'weight'
    // trans: 'src' is transposed or not
    // verticalSplit: vertical split or horizontal split, vertical vs. horizontal:
    //  _________________________            _________________________
    // |            |            |          |                         |
    // |            |            |          |_________________________|
    // |            |            |          |                         |
    // |____________|____________|          |_________________________|
    //           vertical                            horizontal
    //
    // ****************************************************************************
    //
    // Vertical split like the left one, but if transposed, like the right one
    //      |<-------- cols ----------|           |<-------- rows ----------|
    //  _    _________________________        _    _________________________
    //  ↑   |            |            |       ↑   |                         |
    //  |   |            |            |       |   |_________________________|
    // rows |            |            |      cols |                         |
    //  ↓   |____________|____________|       ↓   |_________________________|
    //             not_transposed                          transposed
    //
    // ****************************************************************************
    //
    // Horizontal split like the right one, but if transposed, like the left one
    //      |<-------- rows ----------|           |<-------- cols ----------|
    //  _    _________________________        _    _________________________
    //  ↑   |            |            |       ↑   |                         |
    //  |   |            |            |       |   |_________________________|
    // cols |            |            |      rows |                         |
    //  ↓   |____________|____________|       ↓   |_________________________|
    //               transposed                          not_transposed
    //

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            int splitOffset, int splitSize, bool verticalSplit, xft::Matrix<WeiT> &convertedWeight,
            xft::Vector<float> &scaleWeight, xft::Vector<float> &zeroWeight, xft::Vector<float> &sumWeight,
            bool unused) {
        // transform trans cases to no trans cases
        if (trans) {
            std::swap(rows, cols);
            verticalSplit = !verticalSplit;
        }

        int rowOffset, rowSize, colOffset, colSize;
        if (verticalSplit) {
            rowOffset = 0;
            rowSize = rows;
            colOffset = splitOffset;
            colSize = splitSize;
        } else {
            rowOffset = splitOffset;
            rowSize = splitSize;
            colOffset = 0;
            colSize = cols;
        }

        convertedWeight.Resize(rowSize, colSize);

        // FP32 -> FP32
        if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        // FP32 -> FP16
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, float16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                float16_t::cvt_float_to_float16(src, dst, colSize);
            }
        }

        // FP32 -> BF16
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, bfloat16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                bfloat16_t::cvt_float_to_bfloat16(src, dst, colSize);
            }
        }

        // FP32 -> INT8/W8A8
        else if constexpr (std::is_same_v<OriWeiT, float>
                && (std::is_same_v<WeiT, int8_t> || std::is_same_v<WeiT, w8a8_t>)) {
            scaleWeight.Resize(trans ? rowSize : colSize);
            zeroWeight.Resize(trans ? rowSize : colSize);
            const float *src = weight + rowOffset * cols + colOffset;
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            xdnn_sgemm_f32s8f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (int8_t *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            xdnn_hgemm_f32s8f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (int8_t *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif

        }

        // FP32 -> UINT4
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, uint4x2_t>) {
            scaleWeight.Resize(trans ? rowSize : colSize);
            zeroWeight.Resize(trans ? rowSize : colSize);
            const float *src = weight + rowOffset * cols + colOffset;
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            xdnn_sgemm_f32u4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_UINT4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            xdnn_hgemm_f32u4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_UINT4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // FP32 -> NF4
        else if constexpr (std::is_same_v<OriWeiT, float> && std::is_same_v<WeiT, nf4x2_t>) {
            scaleWeight.Resize(trans ? rowSize : colSize);
            zeroWeight.Resize(trans ? rowSize : colSize);
            const float *src = weight + rowOffset * cols + colOffset;
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            xdnn_sgemm_f32nf4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_NF4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            xdnn_sgemm_f32nf4f32_quantize(trans, trans ? rowSize : colSize, trans ? colSize : rowSize, src, cols,
                    0.9999f, (XDNN_NF4x2 *)convertedWeight.Data(), convertedWeight.Stride(), scaleWeight.Data(),
                    zeroWeight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16 -> BF16
        else if constexpr (std::is_same_v<OriWeiT, bfloat16_t> && std::is_same_v<WeiT, bfloat16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        // FP16 -> FP16
        else if constexpr (std::is_same_v<OriWeiT, float16_t> && std::is_same_v<WeiT, float16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        // INT8 -> INT8/W8A8
        else if constexpr (std::is_same_v<OriWeiT, int8_t>
                && (std::is_same_v<WeiT, int8_t> || std::is_same_v<WeiT, w8a8_t>)) {
            int size = trans ? rowSize : colSize;
            int offset = trans ? rowOffset : colOffset;
            scaleWeight.Resize(size);
            zeroWeight.Resize(size);
            if (scales) memcpy(scaleWeight.Data(), scales + offset, size * sizeof(float));
            if (zeros) memcpy(zeroWeight.Data(), zeros + offset, size * sizeof(float));
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        // UINT4 -> UINT4
        else if constexpr (std::is_same_v<OriWeiT, uint4x2_t> && std::is_same_v<WeiT, uint4x2_t>) {
            int size = trans ? rowSize : colSize;
            int offset = trans ? rowOffset : colOffset;
            scaleWeight.Resize(size);
            zeroWeight.Resize(size);
            if (scales) memcpy(scaleWeight.Data(), scales + offset, size * sizeof(float));
            if (zeros) memcpy(zeroWeight.Data(), zeros + offset, size * sizeof(float));
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride() / 2;
                const OriWeiT *src = weight + (rowOffset + i) * cols / 2 + colOffset / 2;
                memcpy(dst, src, colSize * sizeof(WeiT) / 2);
            }
        }

        // INT8 -> BF16
        else if constexpr (std::is_same_v<OriWeiT, int8_t> && std::is_same_v<WeiT, bfloat16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                for (uint64_t j = 0; j < colSize; j++) {
                    const int8_t src = weight[(rowOffset + i) * cols + colOffset + j];
                    bfloat16_t *dst = convertedWeight.Data() + i * convertedWeight.Stride() + j;
                    float scale = scales[colOffset + j];
                    float zero = zeros[colOffset + j];
                    *dst = static_cast<bfloat16_t>(scale * src + zero);
                }
            }
        }

        // UINT4 -> BF16
        else if constexpr (std::is_same_v<OriWeiT, uint4x2_t> && std::is_same_v<WeiT, bfloat16_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                for (uint64_t j = 0; j < colSize; j+=2) {
                    const uint4x2_t *src = weight + (rowOffset + i) * cols / 2 + colOffset / 2 + j / 2;
                    bfloat16_t *dst = convertedWeight.Data() + i * convertedWeight.Stride() + j;
                    float scale1 = scales[colOffset + j];
                    float scale2 = scales[colOffset + j + 1];
                    float zero1 = zeros[colOffset + j];
                    float zero2 = zeros[colOffset + j + 1];
                    dst[0] = static_cast<bfloat16_t>(scale1 * src->get_v1() + zero1);
                    dst[1] = static_cast<bfloat16_t>(scale2 * src->get_v2() + zero2);
                }
            }
        }
        // e4m3 -> e4m3
        else if constexpr (std::is_same_v<OriWeiT, e4m3_t> && std::is_same_v<WeiT, e4m3_t>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < rowSize; i++) {
                WeiT *dst = convertedWeight.Data() + i * convertedWeight.Stride();
                const OriWeiT *src = weight + (rowOffset + i) * cols + colOffset;
                memcpy(dst, src, colSize * sizeof(WeiT));
            }
        }

        else {
            printf("%s:%d: Do not support this kind of weights datatype convertion.\n", __FILE__, __LINE__);
            exit(-1);
        }

        // Compute per column Sums for W8A8
        if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            sumWeight.Resize(trans ? rowSize : colSize);
#pragma omp parallel for
            for (uint64_t i = 0; i < colSize; i++) {
                sumWeight.Data()[i] = 0.0f;
                for (uint64_t j = 0; j < rowSize; j++) {
                    sumWeight.Data()[i] += convertedWeight.Data()[j * convertedWeight.Stride() + i];
                }
            }
        }
    }

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            int numSplit, int splitIdx, bool verticalSplit, xft::Matrix<WeiT> &quantizedWeight,
            xft::Vector<float> &scaleWeight, xft::Vector<float> &zeroWeight, xft::Vector<float> &sumWeight) {
        int totalSize = verticalSplit ? cols : rows;
        // for e4m3_t, size should be multiple of 128 (64 * 2)
        int gran = std::is_same_v<WeiT, e4m3_t> ? 2 : 1;
        std::pair<int, int> range = SplitUtil::getTaskRange(totalSize, gran, numSplit, splitIdx);

        int splitSize = range.second - range.first;
        int splitOffset = range.first;

        convertWeight(trans, rows, cols, weight, scales, zeros, splitOffset, splitSize, verticalSplit, quantizedWeight,
                scaleWeight, zeroWeight, sumWeight, true);
    }

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            xft::Matrix<WeiT> &quantizedWeight, xft::Vector<float> &scaleWeight, xft::Vector<float> &zeroWeight,
            xft::Vector<float> &sumWeight) {
        convertWeight(trans, rows, cols, weight, scales, zeros, 1, 0, true, quantizedWeight, scaleWeight, zeroWeight,
                sumWeight);
    }

    template <typename OriWeiT, typename WeiT>
    void convertWeight(DecoderContext *ctx, bool trans, int rows, int cols, const OriWeiT *weight, const float *scales,
            const float *zeros, bool verticalSplit, xft::Matrix<WeiT> &quantizedWeight, xft::Vector<float> &scaleWeight,
            xft::Vector<float> &zeroWeight, xft::Vector<float> &sumWeight) {
        convertWeight(trans, rows, cols, weight, scales, zeros, ctx->numSplit, ctx->splitIdx, verticalSplit,
                quantizedWeight, scaleWeight, zeroWeight, sumWeight);
    }

    template <typename WeiT>
    void packWeight(bool trans, xft::Matrix<WeiT> &src, xft::Matrix<WeiT> &weight) {
        int K = trans ? src.Cols() : src.Rows();
        int N = trans ? src.Rows() : src.Cols();

        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            weight.Resize(K, N);
            xdnn_sgemm_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            weight.Resize(K, N);
            xdnn_sgemm_f32f16f32_packb(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
#ifdef AMX_FP16_WEIGHT_ONLY_FP16
            int amx_rows = (int)((K + 15) / 16) * 16;
            int amx_cols = (int)((N + 63) / 64) * 64;
            weight.Resize(amx_rows, amx_cols);
            memset(weight.Data(), 0, sizeof(float16_t) * amx_rows * amx_cols);
            xdnn_hgemm_f32f16f32_packb_block(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data(), 16, 64);
#else
            weight.Resize(K, N);
            xdnn_hgemm_f32f16f32_packb(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#endif

#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            int amx_rows = (int)((K + 15) / 16) * 16;
            int amx_cols = (int)((N + 63) / 64) * 64;
            weight.Resize(amx_rows, amx_cols);
            memset(weight.Data(), 0, sizeof(bfloat16_t) * amx_rows * amx_cols);
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            xdnn_sgemm_f32bf16f32_packb(
                    trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            xdnn_bgemm_f32bf16f32_packb(
                    trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            xdnn_sgemm_f32s8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            xdnn_hgemm_f32s8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            using dt = dnnl::memory::data_type;

            auto tag = trans ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab;
            dnnl::memory B_mem({{K, N}, dt::s8, tag}, *cpu_engine, src.Data());
            dnnl::memory::desc desc({K, N}, dt::s8, get_onednn_weight_layout(dt::s8));

            // When converting to oneDNN blocked memory format, padded dims can be larger than [K, N]
            // Resize down Matrix does not change underlying buffer.
            // TODO: Add reserve like function in Matrix, as current 2 times Resize has potential risks.
            auto dims = desc.get_padded_dims();
            weight.Resize(dims[0], dims[1]);
            weight.Resize(K, N);

            dnnl::memory packedB_mem(desc, *cpu_engine, weight.Data());
            dnnl::reorder(B_mem, packedB_mem).execute(*cpu_stream, B_mem, packedB_mem);
            cpu_stream->wait();
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            xdnn_sgemm_f32u4f32_packb(
                    trans, N, K, (const XDNN_UINT4x2 *)src.Data(), src.Stride(), (XDNN_UINT4x2 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            xdnn_hgemm_f32u4f32_packb(
                    trans, N, K, (const XDNN_UINT4x2 *)src.Data(), src.Stride(), (XDNN_UINT4x2 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            xdnn_sgemm_f32nf4f32_packb(
                    trans, N, K, (const XDNN_NF4x2 *)src.Data(), src.Stride(), (XDNN_NF4x2 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            xdnn_hgemm_f32nf4f32_packb(
                    trans, N, K, (const XDNN_NF4x2 *)src.Data(), src.Stride(), (XDNN_NF4x2 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // E4M3
        else if constexpr (std::is_same_v<WeiT, e4m3_t>) {
            int packBlkSize = 32;
            size_t pack_size = xdnn_small_amx_sgemm_bf16f8bf16_packb_size(K, N, packBlkSize);
            if (!weight.isShadow()) weight.Resize((pack_size + N - 1) / N, N);
            memset(weight.Data(), 0, sizeof(e4m3_t) * pack_size);
            xdnn_small_amx_sgemm_bf16f8bf16_packb(trans, N, K, (const XDNN_E4M3 *)src.Data(), src.Stride(),
                    (XDNN_E4M3 *)weight.Data(), packBlkSize);
        }
    }

    template <typename WeiT>
    void transposeWeight(bool trans, xft::Matrix<WeiT> &src, xft::Matrix<WeiT> &dst) {
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        dt weight_dt;
        if constexpr (std::is_same_v<WeiT, float>) {
            weight_dt = dt::f32;
        } else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            weight_dt = dt::bf16;
        } else if constexpr (std::is_same_v<WeiT, float16_t>) {
            weight_dt = dt::f16;
        } else {
            printf(">>> onednn_gemm_compute: input date type not supported.");
            exit(-1);
        }

        int K = trans ? src.Cols() : src.Rows();
        int N = trans ? src.Rows() : src.Cols();
        auto weight_md = memory::desc({K, N}, weight_dt, trans ? tag::ba : tag::ab);
        auto weight_mem = memory(weight_md, *cpu_engine, src.Data());
        auto transposed_weight_md = memory::desc({K, N}, weight_dt, get_onednn_weight_layout(weight_dt));
        auto transposed_weight_mem = memory(transposed_weight_md, *cpu_engine, dst.Data());
        dnnl::reorder(weight_mem, transposed_weight_mem).execute(*cpu_stream, weight_mem, transposed_weight_mem);
        cpu_stream->wait();
    }

    // lds: leading dimension of scaleB
    template <typename InT, typename WeiT, typename OutT>
    void compute(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            int lds = -1, int blockSize = 128) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE(
                    "xdnn_sgemm_compute", xdnn_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute",
                        xdnn_sgemm_f32f16f32_compute(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            } else {
                GEMMVERBOSE("onednn_gemm_compute",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
            }
#elif defined(AMX_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("onednn_amx_gemm_compute",
                    onednn_amx_gemm_compute(
                            transA, M, N, K, alpha, A, lda, (const float16_t *)packedB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute",
                        xdnn_hgemm_f32f16f32_compute(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute",
                            xdnn_hgemm_compute(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute",
                            xdnn_hgemm_f16f16f32_compute(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute",
                    xdnn_sgemm_f32bf16f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            // TODO: xdnn impl?
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_gemm_compute",
                        onednn_amx_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
            } else {
                if (M > AMXThresholdM) {
                    GEMMVERBOSE("onednn_amx_gemm_compute",
                            onednn_amx_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute",
                            xdnn_bgemm_f32bf16f32_compute(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute",
                    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute",
                    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, nullptr, 0, 0.0f, matmul_kinds::Basic));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute",
                    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                            zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute",
                    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                            zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute",
                    xdnn_sgemm_f32nf4f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute",
                    xdnn_hgemm_f32nf4f32_compute(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // E4M3
        else if constexpr (std::is_same_v<WeiT, e4m3_t>) {
            if (true) {
                assert(blockSize == 128);
                if (lds == -1) lds = (K + 127) / 128;
                GEMMVERBOSE("xdnn_gemm_bf16f8bf16_compute",
                        xdnn_small_amx_sgemm_bf16f8bf16_compute(M, N, K, (const XDNN_BF16 *)A, lda,
                                (const XDNN_E4M3 *)packedB, (XDNN_BF16 *)C, ldc, (float *)scaleB, lds, blockSize, alpha,
                                beta, nullptr));
            } else {
                // Decode E4M3 to BF16
                int rows = (K + 15) / 16 * 16;
                int cols = (N + 63) / 64 * 64;
                uint64_t packSize = (uint64_t)rows * cols;
                bfloat16_t *decodedB = (bfloat16_t *)aligned_alloc(64, packSize * sizeof(bfloat16_t));

                assert(blockSize == 128);
                if (lds == -1) lds = (rows + 127) / 128;

#pragma omp parallel for collapse(2)
                for (int j = 0; j < cols; j += 64) {
                    for (int i = 0; i < rows / 2; ++i) { // 2 rows merged into 1
                        int offset = j * rows + i * 128;
                        int scaleOff = j / 128 * lds + i / 64;
                        e4m3_t::to_bf16(packedB + offset, (uint16_t *)decodedB + offset, 128, scaleB[scaleOff]);
                    }
                }

                // Call into BF16 oneDNN gemm
                GEMMVERBOSE("onednn_amx_gemm_compute",
                        onednn_amx_gemm_compute(transA, M, N, K, alpha, A, lda, decodedB, beta, C, ldc));

                // Clean up
                free(decodedB);
            }
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_bias(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_biasadd",
                    xdnn_sgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_biasadd",
                        xdnn_sgemm_f32f16f32_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
            } else {
                GEMMVERBOSE("onednn_gemm_compute_bias",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias,
                                (const InT *)nullptr, -1, matmul_kinds::BiasAdd));
            }
#elif defined(AMX_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("onednn_amx_gemm_compute_biasadd",
                    onednn_amx_gemm_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, (const float16_t *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_biasadd",
                        xdnn_hgemm_f32f16f32_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute_biasadd",
                            xdnn_hgemm_compute_biasadd(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc, bias));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_biasadd",
                            xdnn_hgemm_f16f16f32_compute_biasadd(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_biasadd",
                    xdnn_sgemm_f32bf16f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            // TODO: xdnn impl?
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_gemm_compute_biasadd",
                        onednn_amx_gemm_compute_biasadd(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
            } else {
                if (M > AMXThresholdM) {
                    GEMMVERBOSE("onednn_amx_gemm_compute_biasadd",
                            onednn_amx_gemm_compute_biasadd(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_biasadd",
                            xdnn_bgemm_f32bf16f32_compute_biasadd(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_biasadd",
                    xdnn_sgemm_f32s8f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_biasadd",
                    xdnn_hgemm_f32s8f32_compute_biasadd(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_biasadd",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, nullptr, 0, 0.0f, matmul_kinds::BiasAdd));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_biasadd",
                    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_biasadd",
                    xdnn_hgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_biasadd",
                    xdnn_sgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_biasadd",
                    xdnn_hgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_biasadd_relu",
                    xdnn_sgemm_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_biasadd_relu",
                        xdnn_sgemm_f32f16f32_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
            } else {
                GEMMVERBOSE("onednn_gemm_compute_bias_relu",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias,
                                (const InT *)nullptr, -1, matmul_kinds::BiasAdd_Relu));
            }
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_biasadd_relu",
                        xdnn_hgemm_f32f16f32_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute_biasadd_relu",
                            xdnn_hgemm_compute_biasadd_relu(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc, bias));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_biasadd_relu",
                            xdnn_hgemm_f16f16f32_compute_biasadd_relu(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc, bias));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_biasadd_relu",
                    xdnn_sgemm_f32bf16f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
#elif defined(AMX_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("onednn_amx_gemm_compute_biasadd_relu",
                        onednn_amx_gemm_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, (const float16_t *)packedB, beta, C, ldc, bias));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > AMXThresholdM) {
                GEMMVERBOSE("onednn_amx_gemm_compute_biasadd_relu",
                        onednn_amx_gemm_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias));
            } else {
                GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_biasadd_relu",
                        xdnn_bgemm_f32bf16f32_compute_biasadd_relu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias));
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_biasadd_relu",
                    xdnn_sgemm_f32s8f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_biasadd_relu",
                    xdnn_hgemm_f32s8f32_compute_biasadd_relu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_biasadd_relu",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, nullptr, 0, 0.0f, matmul_kinds::BiasAdd_Relu));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_biasadd_relu",
                    xdnn_sgemm_f32u4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_biasadd_relu",
                    xdnn_hgemm_f32u4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_biasadd_relu",
                    xdnn_sgemm_f32nf4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_biasadd_relu",
                    xdnn_hgemm_f32nf4f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_silu",
                    xdnn_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_silu",
                        xdnn_sgemm_f32f16f32_compute_silu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            } else {
                GEMMVERBOSE("onednn_gemm_compute_silu",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc,
                                (const float *)nullptr, (const InT *)nullptr, -1, matmul_kinds::Silu));
            }
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_silu",
                        xdnn_hgemm_f32f16f32_compute_silu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute_silu",
                            xdnn_hgemm_compute_silu(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_silu",
                            xdnn_hgemm_f16f16f32_compute_silu(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_silu",
                    xdnn_sgemm_f32bf16f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_gemm_compute_silu",
                        onednn_amx_gemm_compute(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, matmul_kinds::Silu));
            } else {
                if (M > AMXThresholdM) {
                    GEMMVERBOSE("onednn_amx_gemm_compute_silu",
                            onednn_amx_gemm_compute(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, matmul_kinds::Silu));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_silu",
                            xdnn_bgemm_f32bf16f32_compute_silu(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_silu",
                    xdnn_sgemm_f32s8f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_silu",
                    xdnn_hgemm_f32s8f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_silu",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, nullptr, 0, 0.0f, matmul_kinds::Silu));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_silu",
                    xdnn_sgemm_f32u4f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_silu",
                    xdnn_hgemm_f32u4f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_silu",
                    xdnn_sgemm_f32nf4f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_silu",
                    xdnn_hgemm_f32nf4f32_compute_silu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_gelu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_gelu",
                    xdnn_sgemm_compute_gelu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_gelu",
                        xdnn_sgemm_f32f16f32_compute_gelu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            } else {
                GEMMVERBOSE("onednn_gemm_compute_gelu",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc,
                                (const float *)nullptr, (const InT *)nullptr, -1, matmul_kinds::Gelu));
            }

#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_gelu",
                        xdnn_hgemm_f32f16f32_compute_gelu(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute_gelu",
                            xdnn_hgemm_compute_gelu(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_gelu",
                            xdnn_hgemm_f16f16f32_compute_gelu(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_gelu",
                    xdnn_sgemm_f32bf16f32_compute_gelu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_gemm_compute_gelu",
                        onednn_amx_gemm_compute(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, matmul_kinds::Gelu));
            } else {
                if (M > AMXThresholdM) {
                    GEMMVERBOSE("onednn_amx_gemm_compute_gelu",
                            onednn_amx_gemm_compute(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, matmul_kinds::Gelu));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_gelu",
                            xdnn_bgemm_f32bf16f32_compute_gelu(
                                    transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_gelu",
                    xdnn_sgemm_f32s8f32_compute_gelu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_gelu",
                    xdnn_hgemm_f32s8f32_compute_gelu(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_gelu",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, nullptr, 0, 0.0f, matmul_kinds::Gelu));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_gelu",
                    xdnn_sgemm_f32u4f32_compute_gelu(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_gelu",
                    xdnn_hgemm_f32u4f32_compute_gelu(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_gelu",
                    xdnn_sgemm_f32nf4f32_compute_gelu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_gelu",
                    xdnn_hgemm_f32nf4f32_compute_gelu(
                            transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_resmul(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const InT *res,
            int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_resmul",
                    xdnn_sgemm_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_resmul",
                        xdnn_sgemm_f32f16f32_compute_resmul(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres));
            } else {
                GEMMVERBOSE("onednn_gemm_compute_resmul",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc,
                                (const float *)nullptr, res, ldres, matmul_kinds::Resmul));
            }
#elif defined(AMX_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("onednn_amx_gemm_compute_resmul",
                        onednn_amx_gemm_compute_resmul(
                                transA, M, N, K, alpha, A, lda, (const float16_t *)packedB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_resmul",
                        xdnn_hgemm_f32f16f32_compute_resmul(
                                transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute_resmul",
                            xdnn_hgemm_compute_resmul(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc, (const XDNN_FP16 *)res,
                                    ldres));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_resmul",
                            xdnn_hgemm_f16f16f32_compute_resmul(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc, (const XDNN_FP16 *)res, ldres));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_resmul",
                    xdnn_sgemm_f32bf16f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, res, ldres));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_gemm_compute_resmul",
                        onednn_amx_gemm_compute_resmul(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
            } else {
                if (M > AMXThresholdM) {
                    GEMMVERBOSE("onednn_amx_gemm_compute_resmul",
                            onednn_amx_gemm_compute_resmul(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_resmul",
                            xdnn_bgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda,
                                    (const XDNN_BF16 *)packedB, beta, C, ldc, res, ldres));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_resmul",
                    xdnn_sgemm_f32s8f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_resmul",
                    xdnn_hgemm_f32s8f32_compute_resmul(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_resmul",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, nullptr, res, ldres, 0.0f, matmul_kinds::Resmul));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_resmul",
                    xdnn_sgemm_f32u4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_resmul",
                    xdnn_hgemm_f32u4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_resmul",
                    xdnn_sgemm_f32nf4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_resmul",
                    xdnn_hgemm_f32nf4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_residential(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const float *bias,
            const InT *res, int ldres, int lds = -1, int blockSize = 128) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_residential",
                    xdnn_sgemm_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_residential",
                        xdnn_sgemm_f32f16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                                (const XDNN_FP16 *)packedB, beta, C, ldc, bias, res, ldres));
            } else {
                GEMMVERBOSE("onednn_gemm_compute_residential",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres,
                                matmul_kinds::Residential));
            }
#elif defined(AMX_FP16_WEIGHT_ONLY_FP16)
            GEMMVERBOSE("onednn_amx_gemm_compute_residential",
                    onednn_amx_gemm_compute_residential(
                            transA, M, N, K, alpha, A, lda, (const float16_t *)packedB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_residential",
                        xdnn_hgemm_f32f16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                                (const XDNN_FP16 *)packedB, beta, C, ldc, bias, res, ldres));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("onednn_amx_gemm_compute_residential",
                            onednn_amx_gemm_compute_residential(transA, M, N, K, alpha, A, lda, (const float16_t *)packedB,
		                    beta, C, ldc, bias, res, ldres));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_residential",
                            xdnn_hgemm_f16f16f32_compute_residential(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc, bias, (const XDNN_FP16 *)res, ldres));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_residential",
                    xdnn_sgemm_f32bf16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_BF16 *)packedB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            // TODO: xdnn impl?
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                GEMMVERBOSE("onednn_amx_gemm_compute_residential",
                        onednn_amx_gemm_compute_residential(
                                transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
            } else {
                if (M > AMXThresholdM) {
                    GEMMVERBOSE("onednn_amx_gemm_compute_residential",
                            onednn_amx_gemm_compute_residential(
                                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres));
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_residential",
                            xdnn_bgemm_f32bf16f32_compute_residential(transA, M, N, K, alpha, A, lda,
                                    (const XDNN_BF16 *)packedB, beta, C, ldc, bias, res, ldres));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_residential",
                    xdnn_sgemm_f32s8f32_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_residential",
                    xdnn_hgemm_f32s8f32_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_residential",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, res, ldres, 0.0f, matmul_kinds::Residential));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_residential",
                    xdnn_sgemm_f32u4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_residential",
                    xdnn_hgemm_f32u4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_residential",
                    xdnn_sgemm_f32nf4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_residential",
                    xdnn_hgemm_f32nf4f32_compute_residential(transA, M, N, K, alpha, A, lda,
                            (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // E4M3
        else if constexpr (std::is_same_v<WeiT, e4m3_t>) {
            if (true) {
                assert(blockSize == 128);
                if (lds == -1) lds = (K + 127) / 128;
                GEMMVERBOSE("xdnn_gemm_bf16f8bf16_compute_residential",
                        xdnn_small_amx_sgemm_bf16f8bf16_compute_residential(M, N, K, (const XDNN_BF16 *)A, lda,
                                (const XDNN_E4M3 *)packedB, (XDNN_BF16 *)C, ldc, (float *)scaleB, lds, blockSize, alpha,
                                beta, nullptr, (const XDNN_BF16 *)res, ldres));
            } else {
                // Decode E4M3 to BF16
                int rows = (K + 15) / 16 * 16;
                int cols = (N + 63) / 64 * 64;
                uint64_t packSize = (uint64_t)rows * cols;
                bfloat16_t *decodedB = (bfloat16_t *)aligned_alloc(64, packSize * sizeof(bfloat16_t));

                assert(blockSize == 128);
                if (lds == -1) lds = (rows + 127) / 128;

#pragma omp parallel for collapse(2)
                for (int j = 0; j < cols; j += 64) {
                    for (int i = 0; i < rows / 2; ++i) { // 2 rows merged into 1
                        int offset = j * rows + i * 128;
                        int scaleOff = j / 128 * lds + i / 64;
                        e4m3_t::to_bf16(packedB + offset, (uint16_t *)decodedB + offset, 128, scaleB[scaleOff]);
                    }
                }

                // Call into BF16 oneDNN gemm
                GEMMVERBOSE("onednn_amx_gemm_compute_residential",
                        onednn_amx_gemm_compute_residential(
                                transA, M, N, K, alpha, A, lda, decodedB, beta, C, ldc, bias, res, ldres));

                // Clean up
                free(decodedB);
            }
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_resext(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const float *bias,
            float gamma, InT *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            GEMMVERBOSE("xdnn_sgemm_compute_resext",
                    xdnn_sgemm_compute_resext(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres));
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
                GEMMVERBOSE("xdnn_sgemm_f32f16f32_compute_resext",
                        xdnn_sgemm_f32f16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
                                beta, C, ldc, bias, gamma, res, ldres));
            } else {
#pragma omp parallel for collapse(2)
                for (uint64_t i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        auto remain = N - j;
                        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                        auto v = xft::load_avx512(mask, &res[i * ldres + j]);
                        v = _mm512_mul_ps(_mm512_set1_ps(gamma), v);
                        xft::store_avx512(&res[i * ldres + j], mask, v);
                    }
                }

                GEMMVERBOSE("onednn_gemm_compute_resext",
                        onednn_gemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres,
                                matmul_kinds::Residential));
            }
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            if constexpr (std::is_same_v<InT, float>) {
                GEMMVERBOSE("xdnn_hgemm_f32f16f32_compute_resext",
                        xdnn_hgemm_f32f16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB,
                                beta, C, ldc, bias, gamma, res, ldres));
            } else if constexpr (std::is_same_v<InT, float16_t>) {
                if constexpr (std::is_same_v<OutT, float16_t>) {
                    GEMMVERBOSE("xdnn_hgemm_compute_resext",
                            xdnn_hgemm_compute_resext(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, (XDNN_FP16 *)C, ldc, bias, gamma,
                                    (const XDNN_FP16 *)res, ldres));
                } else if constexpr (std::is_same_v<OutT, float>) {
                    GEMMVERBOSE("xdnn_hgemm_f16f16f32_compute_resext",
                            xdnn_hgemm_f16f16f32_compute_resext(transA, M, N, K, alpha, (const XDNN_FP16 *)A, lda,
                                    (const XDNN_FP16 *)packedB, beta, C, ldc, bias, gamma, (const XDNN_FP16 *)res,
                                    ldres));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            GEMMVERBOSE("xdnn_sgemm_f32bf16f32_compute_resext",
                    xdnn_sgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB,
                            beta, C, ldc, bias, gamma, res, ldres));
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if constexpr (std::is_same_v<InT, bfloat16_t>) {
                TimeLine t("onednn_amx_gemm_compute_residential");
#pragma omp parallel for collapse(2)
                for (uint64_t i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        auto remain = N - j;
                        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                        auto v = xft::load_avx512(mask, &res[i * ldres + j]);
                        v = _mm512_mul_ps(_mm512_set1_ps(gamma), v);
                        xft::store_avx512(&res[i * ldres + j], mask, v);
                    }
                }
                onednn_amx_gemm_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                if (M > AMXThresholdM) {
                    TimeLine t("onednn_amx_gemm_compute_residential");
#pragma omp parallel for collapse(2)
                    for (uint64_t i = 0; i < M; ++i) {
                        for (int j = 0; j < N; ++j) {
                            res[i * ldres + j] = res[i * ldres + j] * gamma;
                        }
                    }
                    onednn_amx_gemm_compute_residential(
                            transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
                } else {
                    GEMMVERBOSE("xdnn_bgemm_f32bf16f32_compute_resext",
                            xdnn_bgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda,
                                    (const XDNN_BF16 *)packedB, beta, C, ldc, bias, gamma, res, ldres));
                }
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            GEMMVERBOSE("xdnn_sgemm_f32s8f32_compute_resext",
                    xdnn_sgemm_f32s8f32_compute_resext(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C,
                            ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            GEMMVERBOSE("xdnn_hgemm_f32s8f32_compute_resext",
                    xdnn_hgemm_f32s8f32_compute_resext(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C,
                            ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // W8A8
        else if constexpr (std::is_same_v<WeiT, w8a8_t>) {
            GEMMVERBOSE("onednn_amx_gemm_f32s8f32_compute_resext",
                    onednn_amx_gemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, (const int8_t *)packedB, scaleB,
                            zeroB, sumB, beta, C, ldc, bias, res, ldres, gamma, matmul_kinds::Resext));
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            GEMMVERBOSE("xdnn_sgemm_f32u4f32_compute_resext",
                    xdnn_sgemm_f32u4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            GEMMVERBOSE("xdnn_hgemm_f32u4f32_compute_resext",
                    xdnn_hgemm_f32u4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            GEMMVERBOSE("xdnn_sgemm_f32nf4f32_compute_resext",
                    xdnn_sgemm_f32nf4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            GEMMVERBOSE("xdnn_hgemm_f32nf4f32_compute_resext",
                    xdnn_hgemm_f32nf4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                            scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres));
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_batch_C(int M, int N, int K, float *alphaList, const InT *A, int lda, const WeiT *packedBBatch[],
            const float *scalesList[], OutT *CList[], int *ldcList, int *ldsList, int blockSize = 128, int batchSize = 1) {
        if constexpr (std::is_same_v<WeiT, e4m3_t> && std::is_same_v<OutT, bfloat16_t> && std::is_same_v<InT, bfloat16_t>) {
            GEMMVERBOSE("xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C",
                xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C(M, N, K, (const XDNN_BF16 *)A, lda, (const XDNN_E4M3 **)packedBBatch,
                    (XDNN_BF16 **)CList, ldcList, scalesList, ldsList, blockSize, alphaList, batchSize));
        } else if constexpr (std::is_same_v<WeiT, bfloat16_t> && std::is_same_v<OutT, bfloat16_t> && std::is_same_v<InT, bfloat16_t>) {
            GEMMVERBOSE("xdnn_small_amx_sgemm_bf16bf16bf16_compute_batch_C",
                xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_C(M, N, K, (const XDNN_BF16 *)A, lda, (const XDNN_BF16 **)packedBBatch,
                    (XDNN_BF16 **)CList, ldcList, alphaList, batchSize));
        } else {
            printf("%s:%d: Unsupported data type for compute_batch_C", __FILE__, __LINE__);
            exit(-1);
        }
    }

        template <typename InT, typename WeiT, typename OutT>
    void compute_batch_CM(int M, int *N, int K, float *alphaList, const InT *A, int lda, const WeiT *packedBBatch[],
            const float *scalesList[], OutT *CList[], int *ldcList, int *ldsList, int blockSize = 128,
            int batchSize = 1) {
        if constexpr (std::is_same_v<WeiT, e4m3_t> && std::is_same_v<OutT, bfloat16_t>
                && std::is_same_v<InT, bfloat16_t>) {
            GEMVNVERBOSE("xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM",
                    xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM(M, N, K, (const XDNN_BF16 *)A, lda,
                            (const XDNN_E4M3 **)packedBBatch, (XDNN_BF16 **)CList, ldcList, scalesList, ldsList,
                            blockSize, alphaList, batchSize));
        } else {
            printf("%s:%d: Unsupported data type for compute_batch_C", __FILE__, __LINE__);
            exit(-1);
        }
    }
	    template <typename InT, typename WeiT, typename OutT>
    void compute_batch_AM(int M, int N, int *K, float *alphaList, const InT *A[], int *ldaList,
            const WeiT *packedBBatch[], const float *scalesList[], OutT *C, int ldc, int *ldsList, int blockSize = 128,
            int batchSize = 1) {
        if constexpr (std::is_same_v<WeiT, e4m3_t> && std::is_same_v<OutT, bfloat16_t>
                && std::is_same_v<InT, bfloat16_t>) {
            GEMVKVERBOSE("xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM",
                    xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM(M, N, K, (const XDNN_BF16 **)A, ldaList,
                            (const XDNN_E4M3 **)packedBBatch, (XDNN_BF16 *)C, ldc, scalesList, ldsList, blockSize,
                            alphaList, batchSize));
        } else {
            printf("%s:%d: Unsupported data type for compute_residential_batch_A", __FILE__, __LINE__);
            exit(-1);
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_batch_A(int M, int N, int K, float *alphaList, const InT *A[], int *ldaList, const WeiT *packedBBatch[],
            const float *scalesList[], OutT *C, int ldc, int *ldsList, int blockSize = 128, int batchSize = 1) {
        if constexpr (std::is_same_v<WeiT, e4m3_t> && std::is_same_v<OutT, bfloat16_t> && std::is_same_v<InT, bfloat16_t>) {
            GEMMVERBOSE("xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A",
                xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A(M, N, K, (const XDNN_BF16 **)A, ldaList, (const XDNN_E4M3 **)packedBBatch,
                    (XDNN_BF16 *)C, ldc, scalesList, ldsList, blockSize, alphaList, batchSize));
        } else if constexpr (std::is_same_v<WeiT, bfloat16_t> && std::is_same_v<OutT, bfloat16_t> && std::is_same_v<InT, bfloat16_t>) {
            GEMMVERBOSE("xdnn_small_amx_sgemm_bf16bf16bf16_compute_batch_A",
                xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_A(M, N, K, (const XDNN_BF16 **)A, ldaList, (const XDNN_BF16 **)packedBBatch,
                    (XDNN_BF16 *)C, ldc, alphaList, batchSize));
        } else{
            printf("%s:%d: Unsupported data type for compute_residential_batch_A", __FILE__, __LINE__);
            exit(-1);
        }
    }

    int getEngineCount() {
        int count = engine->get_count(kind);
        return count;
    }

private:
    dnnl::engine::kind kind;
    dnnl::engine *engine; // For runtime engine
    dnnl::stream *stream; // For runtime stream
    std::unordered_map<std::string, std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *>> matmul_hub;
    dnnl::engine *cpu_engine;
    dnnl::stream *cpu_stream;

    int AMXThresholdM;
    int primitiveCacheM;

    enum matmul_kinds {
        Basic = 0,
        BiasAdd,
        BiasAdd_Relu,
        Silu,
        Gelu,
        Resmul,
        Residential,
        Resext,
    };

    std::string create_key(bool transA, int M, int N, int K, int matmul_kind) {
        auto intToHex = [](int value, char *buf) {
            const char hexDigits[] = "0123456789ABCDEF";
            for (int i = 7; i >= 0; --i) {
                buf[i] = hexDigits[value & 0xF]; // Get the last 4 bits of value
                value >>= 4; // Shift right by 4 bits to process the next hex digit
            }
        };

        char buf[40] = {0};
        buf[0] = transA ? '1' : '0';
        buf[1] = '_';
        intToHex(M, buf + 2);
        buf[10] = '_';
        intToHex(N, buf + 11);
        buf[19] = '_';
        intToHex(K, buf + 20);
        buf[28] = '_';
        intToHex(matmul_kind, buf + 29);

        return std::string(buf);
    }

    // Cache primitive_desc and matmul
    bool cache_matmul_primitive(dnnl::matmul::primitive_desc *matmul_pd, dnnl::matmul *matmul_prim, bool transA, int M,
            int N, int K, int matmul_kind) {
        // If M < primitiveCacheM or a power of 2, then cache.
        if (M <= primitiveCacheM || ((M & (M - 1)) == 0)) {
            std::string key = create_key(transA, M, N, K, matmul_kind);
            std::tuple<dnnl::matmul::primitive_desc *, dnnl::matmul *> value(matmul_pd, matmul_prim);
            matmul_hub[key] = value;
            return true;
        } else {
            return false;
        }
    }

    dnnl::memory::format_tag get_onednn_input_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::ab;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
            // return dnnl::memory::format_tag::AB32a16b;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in input layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_weight_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            if (dt == dnnl::memory::data_type::bf16) {
                return dnnl::memory::format_tag::BA16a64b2a;
            } else if (dt == dnnl::memory::data_type::f16) {
#ifdef AMX_FP16_WEIGHT_ONLY_FP16
                return dnnl::memory::format_tag::BA16a64b2a;
#else
                return dnnl::memory::format_tag::BA16a64b;
#endif
            } else if (dt == dnnl::memory::data_type::s8) {
                return dnnl::memory::format_tag::BA16a64b4a;
            } else {
                printf("[XFT][ERROR] Unsupport your data type in input layout.");
                std::exit(-1);
            }
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ba;
            // return dnnl::memory::format_tag::BA4b8a8b2a;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in weight layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_bias_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::ab;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in bias layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_shift_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::ab;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in shift layout.");
            std::exit(-1);
        }
    }

    dnnl::memory::format_tag get_onednn_output_layout(dnnl::memory::data_type dt) {
        if (this->kind == dnnl::engine::kind::cpu) {
            return dnnl::memory::format_tag::ab;
        } else if (this->kind == dnnl::engine::kind::gpu) {
            return dnnl::memory::format_tag::ab;
            // return dnnl::memory::format_tag::AB32a16b;
        } else {
            printf("[XFT][ERROR] Need a right engine kind in output layout.");
            std::exit(-1);
        }
    }

    // Tin | Twei | Tout | Tbias | matmul
    // --- | ---- | ---- | ----- | ------
    // f32 | f32  | f32  | f32   | sgemm
    // f32 | f32  | f16  | f32   | sgemm_f32f32f16
    // f32 | f32  | bf16 | f32   | sgemm_f32f32bf16
    // f16 | f32  | f32  | f32   | sgemm_f16f32f32
    // bf16| f32  | f32  | f32   | sgemm_bf16f32f32
    // f16 | f32  | f16  | f32   | sgemm_f16f32f16
    // bf16| f32  | bf16 | f32   | sgemm_bf16f32bf16
    // f32 | f16  | f32  | f32   | hgemm_f32f16f32
    // f32 | f16  | f16  | f32   | hgemm_f32f16f16
    // f16 | f16  | f32  | f32   | hgemm_f16f16f32
    // f16 | f16  | f16  | f32   | hgemm
    // f32 | bf16 | f32  | f32   | bgemm_f32bf16f32
    // f32 | bf16 | bf16 | f32   | bgemm_f32bf16bf16
    // bf16| bf16 | f32  | f32   | bgemm_bf16bf16f32
    // bf16| bf16 | bf16 | f32   | bgemm
    template <typename Tin, typename Twei, typename Tout, typename Tbias = float>
    void onednn_gemm_compute(bool transA, int M, int N, int K, float alpha, const Tin *A, int lda, const Twei *packedB,
            float beta, Tout *C, int ldc, const Tbias *bias = nullptr, const Tin *res = nullptr, int ldres = -1,
            const matmul_kinds postAlg = matmul_kinds::Basic) {
        TimeLine t("onednn_gemm_compute");
        TimeLine t1("onednn_gemm_compute.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        dt input_dt;
        dt weight_dt;
        dt shift_dt;
        if constexpr (std::is_same_v<Twei, float>) {
            input_dt = dt::f32;
            weight_dt = dt::f32;
            shift_dt = dt::f32;
        } else if constexpr (std::is_same_v<Twei, bfloat16_t>) {
            input_dt = dt::bf16;
            weight_dt = dt::bf16;
            shift_dt = dt::bf16;
        } else if constexpr (std::is_same_v<Twei, float16_t>) {
            input_dt = dt::f16;
            weight_dt = dt::f16;
            shift_dt = dt::f16;
        } else {
            printf(">>> onednn_gemm_compute: input and weight date type not supported.");
            exit(-1);
        }

        dt output_dt;
        if constexpr (std::is_same_v<Tout, float>) {
            output_dt = dt::f32;
        } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
            output_dt = dt::bf16;
        } else if constexpr (std::is_same_v<Tout, float16_t>) {
            output_dt = dt::f16;
        } else {
            printf(">>> onednn_gemm_compute: output date type not supported.");
            exit(-1);
        }

        dt bias_dt;
        if constexpr (std::is_same_v<Tbias, float>) {
            bias_dt = dt::f32;
        } else if constexpr (std::is_same_v<Tbias, bfloat16_t>) {
            bias_dt = dt::bf16;
        } else if constexpr (std::is_same_v<Tbias, float16_t>) {
            bias_dt = dt::f16;
        } else {
            printf(">>> onednn_gemm_compute: bias date type not supported.");
            exit(-1);
        }

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, postAlg);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};
            memory::dims bias_dims = {1, N};
            memory::dims shift_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, input_dt, get_onednn_input_layout(input_dt));
            auto weight_md = memory::desc(weight_dims, weight_dt, get_onednn_weight_layout(weight_dt));
            auto output_md = memory::desc(output_dims, output_dt, get_onednn_output_layout(output_dt));
            auto bias_md = memory::desc(bias_dims, bias_dt, get_onednn_bias_layout(bias_dt));
            auto shift_md = memory::desc(shift_dims, shift_dt, get_onednn_shift_layout(shift_dt));

            // Create primitive descriptor and primitive.
            primitive_attr matmul_attr;
            switch (postAlg) {
                case matmul_kinds::Basic: {
                    break;
                }
                case matmul_kinds::Silu: {
                    const float post_alpha = 1.0f;
                    const float post_beta = 0.0f;
                    post_ops matmul_ops;
                    matmul_ops.append_eltwise(algorithm::eltwise_swish, post_alpha, post_beta);
                    matmul_attr.set_post_ops(matmul_ops);
                    break;
                }
                case matmul_kinds::Gelu: {
                    const float post_alpha = 1.0f;
                    const float post_beta = 0.0f;
                    post_ops matmul_ops;
                    matmul_ops.append_eltwise(algorithm::eltwise_gelu_tanh, post_alpha, post_beta);
                    matmul_attr.set_post_ops(matmul_ops);
                    break;
                }
                case matmul_kinds::Residential: {
                    if (res == nullptr) {
                        printf(">>> onednn_gemm_compute: Residential need be valuable.");
                        exit(-1);
                    }

                    post_ops matmul_ops;
                    matmul_ops.append_binary(algorithm::binary_add, shift_md);
                    matmul_attr.set_post_ops(matmul_ops);
                    break;
                }
                default: {
                    printf(">>> onednn_gemm_compute: postAlg type %s not supported.", std::to_string(postAlg).c_str());
                    exit(-1);
                }
            }

            if (postAlg == matmul_kinds::Basic) {
                if (bias != nullptr)
                    matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md);
                else
                    matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md);
            } else {
                if (bias != nullptr)
                    matmul_pd
                            = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md, matmul_attr);
                else
                    matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
            }

            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, postAlg);
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<Tin *>(A));
        }

        memory weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<Twei *>(packedB));
        memory output_mem = memory(matmul_pd->dst_desc(), *engine, C);
        memory bias_mem;
        if (bias != nullptr) { bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<Tbias *>(bias)); }

        memory shift_mem;
        if (res != nullptr) {
            memory::desc shift_md = memory::desc({M, N}, shift_dt, get_onednn_shift_layout(shift_dt));
            if constexpr (std::is_same_v<Tin, float>) {
                shift_mem = memory(shift_md, *engine);
            } else {
                shift_mem = memory(shift_md, *engine, const_cast<Tin *>(res));
            }
        }

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        if (bias != nullptr) { matmul_args.insert({DNNL_ARG_BIAS, bias_mem}); }
        if (res != nullptr) { matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, shift_mem}); }
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_gemm_compute.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float> && !std::is_same_v<Twei, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                void *input_ptr = input_mem.get_data_handle();
                if constexpr (std::is_same_v<Twei, bfloat16_t>) {
                    bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_ptr + i * K, K);
                    if (res != nullptr) {
                        void *shift_ptr = shift_mem.get_data_handle();
                        bfloat16_t::cvt_float_to_bfloat16(res + i * lda, (bfloat16_t *)shift_ptr + i * K, K);
                    }
                } else if constexpr (std::is_same_v<Twei, float16_t>) {
                    float16_t::cvt_float_to_float16(A + i * lda, (float16_t *)input_ptr + i * K, K);
                    if (res != nullptr) {
                        void *shift_ptr = shift_mem.get_data_handle();
                        float16_t::cvt_float_to_float16(res + i * lda, (float16_t *)shift_ptr + i * K, K);
                    }
                } else {
                    printf(">>> onednn_gemm_compute: input and res date type convert not supported.");
                    exit(-1);
                }
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    template <typename Tin, typename Twei, typename Tout>
    void onednn_amx_gemm_compute(bool transA, int M, int N, int K, float alpha, const Tin *A, int lda,
            const Twei *packedB, float beta, Tout *C, int ldc, const matmul_kinds postAlg = matmul_kinds::Basic) {
        TimeLine t("onednn_amx_gemm_compute");
        TimeLine t1("onednn_amx_gemm_compute.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, postAlg);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            dt dt_x16 = std::is_same_v<Twei, bfloat16_t> ? dt::bf16 :
                    std::is_same_v<Twei, float16_t> ? dt::f16 : dt::undef;

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt_x16, get_onednn_input_layout(dt_x16));
            auto weight_md = memory::desc(weight_dims, dt_x16, get_onednn_weight_layout(dt_x16));

            memory::desc output_md;
            if constexpr (std::is_same_v<Tout, float>) {
                output_md = memory::desc(output_dims, dt::f32, get_onednn_output_layout(dt::f32));
            } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, get_onednn_output_layout(dt::bf16));
            } else if constexpr (std::is_same_v<Tout, float16_t>) {
                output_md = memory::desc(output_dims, dt::f16, get_onednn_output_layout(dt::f16));
            } else {
                printf(">>> onednn amx output date type not supported.");
                exit(-1);
            }

            // Create primitive descriptor and primitive.
            switch (postAlg) {
                case matmul_kinds::Basic:
                    matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md);
                    break;
                case matmul_kinds::Silu: {
                    const float post_alpha = 1.0f;
                    const float post_beta = 0.0f;
                    post_ops matmul_ops;
                    matmul_ops.append_eltwise(algorithm::eltwise_swish, post_alpha, post_beta);
                    primitive_attr matmul_attr;
                    matmul_attr.set_post_ops(matmul_ops);
                    matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
                    break;
                }
                case matmul_kinds::Gelu: {
                    const float post_alpha = 1.0f;
                    const float post_beta = 0.0f;
                    post_ops matmul_ops;
                    matmul_ops.append_eltwise(algorithm::eltwise_gelu_tanh, post_alpha, post_beta);
                    primitive_attr matmul_attr;
                    matmul_attr.set_post_ops(matmul_ops);
                    matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
                    break;
                }
                default:
                    printf(">>> onednn amx postAlg type %s not supported.", std::to_string(postAlg).c_str());
                    exit(-1);
                    break;
            }
            matmul_prim = new matmul(*matmul_pd);
            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, postAlg);
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else if constexpr (std::is_same_v<Tin, float16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<float16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<Twei *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_gemm_compute.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                if constexpr (std::is_same_v<Twei, float16_t>) {
                    float16_t::cvt_float_to_float16(A + i * lda, (float16_t *)input_mem.get_data_handle() + i * K, K);
                } else if constexpr (std::is_same_v<Twei, bfloat16_t>) {
                    bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
                }
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    template <typename Tin, typename Twei, typename Tout>
    void onednn_amx_gemm_compute_biasadd(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const Twei *packedB, float beta, Tout *C, int ldc, const float *bias) {
        TimeLine t("onednn_amx_gemm_compute_biasadd");
        TimeLine t1("onednn_amx_gemm_compute_biasadd.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims bias_dims = {1, N};
            memory::dims output_dims = {M, N};

            dt dt_x16 = std::is_same_v<Twei, bfloat16_t> ? dt::bf16 :
                    std::is_same_v<Twei, float16_t> ? dt::f16 : dt::undef;

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt_x16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt_x16, get_onednn_weight_layout(dt_x16));
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else if constexpr (std::is_same_v<Tin, float16_t>) {
                output_md = memory::desc(output_dims, dt::f16, tag::ab);
            } else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive descriptor & primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, matmul_kinds::BiasAdd);
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else if constexpr (std::is_same_v<Tin, float16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<float16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<Twei *>(packedB));
        auto bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_gemm_compute_biasadd.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                if constexpr (std::is_same_v<Twei, float16_t>) {
                    float16_t::cvt_float_to_float16(A + i * lda, (float16_t *)input_mem.get_data_handle() + i * K, K);
                } else if constexpr (std::is_same_v<Twei, bfloat16_t>) {
                    bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
                }
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    template <typename Tin, typename Twei, typename Tout>
    void onednn_amx_gemm_compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const Twei *packedB, float beta, Tout *C, int ldc, const float *bias) {
        TimeLine t("onednn_amx_gemm_compute_biasadd_relu");
        TimeLine t1("onednn_amx_gemm_compute_biasadd_relu.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::BiasAdd_Relu);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims bias_dims = {1, N};
            memory::dims output_dims = {M, N};

            dt dt_x16 = std::is_same_v<Twei, bfloat16_t> ? dt::bf16 :
                    std::is_same_v<Twei, float16_t> ? dt::f16 : dt::undef;

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt_x16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt_x16, get_onednn_weight_layout(dt_x16));
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tin, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else if constexpr (std::is_same_v<Tin, float16_t>) {
                output_md = memory::desc(output_dims, dt::f16, tag::ab);
            } else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (ReLU).
            const float post_alpha = 0.0f;
            const float post_beta = 0.0f;
            post_ops matmul_ops;
            matmul_ops.append_eltwise(algorithm::eltwise_relu, post_alpha, post_beta);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(matmul_ops);

            // Create primitive descriptor & primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, matmul_kinds::BiasAdd_Relu);
        }

        // Repack and convert input data.
        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else if constexpr (std::is_same_v<Tin, float16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<float16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<Twei *>(packedB));
        auto bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_gemm_compute_biasadd_relu.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                if constexpr (std::is_same_v<Twei, float16_t>) {
                    float16_t::cvt_float_to_float16(A + i * lda, (float16_t *)input_mem.get_data_handle() + i * K, K);
                } else if constexpr (std::is_same_v<Twei, bfloat16_t>) {
                    bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
                }
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    template <typename Tin, typename Twei, typename Tout>
    void onednn_amx_gemm_compute_resmul(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const Twei *packedB, float beta, Tout *C, int ldc, const Tin *res, int ldres) {
        TimeLine t("onednn_amx_gemm_compute_resmul");
        TimeLine t1("onednn_amx_gemm_compute_resmul.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Resmul);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims scale_dims = {M, N};
            memory::dims output_dims = {M, N};

            dt dt_x16 = std::is_same_v<Twei, bfloat16_t> ? dt::bf16 :
                    std::is_same_v<Twei, float16_t> ? dt::f16 : dt::undef;

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt_x16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt_x16, get_onednn_weight_layout(dt_x16));
            auto scale_md = memory::desc(scale_dims,
                    std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tout, float16_t> ? dt::f16 :
                        (std::is_same_v<Tout, bfloat16_t> ? dt::bf16 : dt::undef)),
                        tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tout, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else if constexpr (std::is_same_v<Tout, float16_t>) {
                output_md = memory::desc(output_dims, dt::f16, tag::ab);
            } else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (resmul).
            post_ops binary_ops;
            // dst_tmp = dst_tmp * scale
            binary_ops.append_binary(algorithm::binary_mul, scale_md);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(binary_ops);

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, matmul_kinds::Resmul);
        }

        // Repack and convert input data.
        memory::dims scale_dims = {M, N};
        auto scale_md = memory::desc(scale_dims,
                std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tout, float16_t> ? dt::f16 :
                        (std::is_same_v<Tout, bfloat16_t> ? dt::bf16 : dt::undef)),
                        tag::ab);
        dnnl::memory scale_mem;
        if (C == res) {
            scale_mem = memory(scale_md, *engine);
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                memcpy((Tin *)scale_mem.get_data_handle() + i * N, res + i * ldres, N * sizeof(Tin));
            }
        } else {
            scale_mem = memory(scale_md, *engine, (void *)res);
        }

        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else if constexpr (std::is_same_v<Tin, float16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<float16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<Twei *>(packedB));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_gemm_compute_resmul.execute_primitive");

        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                if constexpr (std::is_same_v<Twei, float16_t>) {
                    float16_t::cvt_float_to_float16(A + i * lda, (float16_t *)input_mem.get_data_handle() + i * K, K);
                } else if constexpr (std::is_same_v<Twei, bfloat16_t>) {
                    bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
                }
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    template <typename Tin,  typename Twei, typename Tout>
    void onednn_amx_gemm_compute_residential(bool transA, int M, int N, int K, float alpha, const Tin *A,
            int lda, const Twei *packedB, float beta, Tout *C, int ldc, const float *bias, const Tin *res,
            int ldres) {
        TimeLine t("onednn_amx_gemm_compute_residential");
        TimeLine t1("onednn_amx_gemm_compute_residential.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Residential);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B), and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims bias_dims = {1, N};
            memory::dims shift_dims = {M, N};
            memory::dims output_dims = {M, N};

            dt dt_x16 = std::is_same_v<Twei, bfloat16_t> ? dt::bf16 :
                    std::is_same_v<Twei, float16_t> ? dt::f16 : dt::undef;

            // Create primitive descriptor.
            auto input_md = memory::desc(input_dims, dt_x16, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt_x16, get_onednn_weight_layout(dt_x16));
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
            auto shift_md = memory::desc(shift_dims,
                    std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tout, float16_t> ? dt::f16 :
                        (std::is_same_v<Tout, bfloat16_t> ? dt::bf16 : dt::undef)),
                        tag::ab);
            memory::desc output_md;
            if constexpr (std::is_same_v<Tout, float>) {
                output_md = memory::desc(output_dims, dt::f32, tag::ab);
            } else if constexpr (std::is_same_v<Tout, bfloat16_t>) {
                output_md = memory::desc(output_dims, dt::bf16, tag::ab);
            } else if constexpr (std::is_same_v<Tout, float16_t>) {
                output_md = memory::desc(output_dims, dt::f16, tag::ab);
            } else {
                printf(">>> onednn amx output date type not supported.");
            }

            // Create primitive post-ops (residential): dst_tmp = dst_tmp + shift
            post_ops matmul_ops;
            matmul_ops.append_binary(algorithm::binary_add, shift_md);
            primitive_attr matmul_attr;
            matmul_attr.set_post_ops(matmul_ops);

            if (bias != nullptr) {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, bias_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            } else {
                // Create primitive descriptor and primitive.
                matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md, matmul_attr);
                matmul_prim = new matmul(*matmul_pd);
            }

            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, matmul_kinds::Residential);
        }

        // Repack and convert input data.
        memory::dims shift_dims = {M, N};
        auto shift_md = memory::desc(shift_dims,
                std::is_same_v<Tin, float> ? dt::f32 : (std::is_same_v<Tout, float16_t> ? dt::f16 :
                        (std::is_same_v<Tout, bfloat16_t> ? dt::bf16 : dt::undef)),
                        tag::ab);

        memory input_mem;
        if constexpr (std::is_same_v<Tin, float>) {
            input_mem = memory(matmul_pd->src_desc(), *engine);
        } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<bfloat16_t *>(A));
        } else if constexpr (std::is_same_v<Tin, float16_t>) {
            input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<float16_t *>(A));
        } else {
            printf(">>> onednn amx input date type not supported.");
        }

        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<Twei *>(packedB));
        memory bias_mem;
        auto shift_mem = memory(shift_md, *engine, (void *)res);
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);
        if (bias != nullptr) { bias_mem = memory(matmul_pd->bias_desc(), *engine, const_cast<float *>(bias)); }

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, shift_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        if (bias != nullptr) { matmul_args.insert({DNNL_ARG_BIAS, bias_mem}); }
        t1.release();

        // Executions.
        TimeLine t2("onednn_amx_gemm_compute_bias_residential.execute_primitive");
        // Reorder
        if constexpr (std::is_same_v<Tin, float>) {
#pragma omp parallel for
            for (uint64_t i = 0; i < M; ++i) {
                if constexpr (std::is_same_v<Twei, float16_t>) {
                    float16_t::cvt_float_to_float16(A + i * lda, (float16_t *)input_mem.get_data_handle() + i * K, K);
                } else if constexpr (std::is_same_v<Twei, bfloat16_t>) {
                    bfloat16_t::cvt_float_to_bfloat16(A + i * lda, (bfloat16_t *)input_mem.get_data_handle() + i * K, K);
                }
            }
        }

        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    void onednn_amx_gemm_s8s8s32(bool transA, int M, int N, int K, float alpha, const int8_t *A, int lda,
            const int8_t *B, float beta, int32_t *C, int ldc) {
        TimeLine t("onednn_amx_gemm_s8s8s32");
        TimeLine t1("onednn_amx_gemm_s8s8s32.create_primitive");
        using namespace dnnl;
        using tag = memory::format_tag;
        using dt = memory::data_type;

        matmul::primitive_desc *matmul_pd;
        matmul *matmul_prim;
        std::string key = create_key(transA, M, N, K, matmul_kinds::Basic);
        auto it = matmul_hub.find(key);
        bool isCached = true;
        if (it != matmul_hub.end()) {
            matmul_pd = std::get<0>(it->second);
            matmul_prim = std::get<1>(it->second);
        } else {
            // Source (A), weights (B) and destination (C) matrix dimensions.
            memory::dims input_dims = {M, K};
            memory::dims weight_dims = {K, N};
            memory::dims output_dims = {M, N};

            // Create memory descriptors and memory objects for src, weights, bias, and dst.
            auto input_md = memory::desc(input_dims, dt::s8, tag::ab);
            auto weight_md = memory::desc(weight_dims, dt::s8, get_onednn_weight_layout(dt::s8));
            memory::desc output_md;
            output_md = memory::desc(output_dims, dt::s32, tag::ab);

            // Create primitive descriptor and primitive.
            matmul_pd = new matmul::primitive_desc(*engine, input_md, weight_md, output_md);
            matmul_prim = new matmul(*matmul_pd);

            // Cache primitive_desc and matmul
            isCached = cache_matmul_primitive(matmul_pd, matmul_prim, transA, M, N, K, matmul_kinds::Basic);
        }

        auto input_mem = memory(matmul_pd->src_desc(), *engine, const_cast<int8_t *>(A));
        auto weight_mem = memory(matmul_pd->weights_desc(), *engine, const_cast<int8_t *>(B));
        auto output_mem = memory(matmul_pd->dst_desc(), *engine, C);

        // Create the primitive args.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, input_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        matmul_args.insert({DNNL_ARG_DST, output_mem});
        t1.release();

        // Executions.
        TimeLine t2("onednn_gemm_s8s8s32.execute_primitive");
        matmul_prim->execute(*stream, matmul_args);
        stream->wait();

        // Free primitive_desc and matmul if not cached.
        if (!isCached) {
            delete matmul_pd;
            delete matmul_prim;
        }
    }

    void onednn_amx_gemm_f32s8f32_compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const int8_t *B, const float *scaleB, const float *zeroB, const float *sumB, float beta, float *C, int ldc,
            const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {
        if (transA || (N % 16) != 0 || alpha != 1.0f || beta != 0.0f) {
            printf("%s:%d: Not implemented.\n", __FILE__, __LINE__);
            exit(-1);
        }

        // split M dimension if M*N is too big
        const int max_MN = 4 * 1024 * 1024;
        int numSplit = M * N / max_MN + 1;
        for (int i = 0; i < numSplit; i++) {
            std::pair<int, int> range = SplitUtil::getTaskRange(M, numSplit, i);
            int MB = range.second - range.first;
            uint64_t offset = range.first;
            onednn_amx_gemm_f32s8f32_compute_base(transA, MB, N, K, alpha, A + offset * lda, lda, B, scaleB, zeroB,
                    sumB, beta, C + offset * ldc, ldc, bias, res + offset * ldres, ldres, gamma, kind);
        }
    }

    void onednn_amx_gemm_f32s8f32_compute_base(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const int8_t *B, const float *scaleB, const float *zeroB, const float *sumB, float beta, float *C, int ldc,
            const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {

#define ALLOC(DATATYPE, VALUE, SIZE)                  \
    std::unique_ptr<DATATYPE, decltype(&free)> VALUE( \
            static_cast<DATATYPE *>(xft::alloc(SIZE * sizeof(DATATYPE))), &free)
        ALLOC(int8_t, quantizedA, M * K);
        ALLOC(float, scaleA, M);
        ALLOC(float, zeroA, M);
        ALLOC(float, sumA, M);
        ALLOC(int32_t, C_int32, M * N);
#undef ALLOC

        TimeLine t1("onednn_amx_gemm_f32s8f32_compute.quantA");
        quantize_s8(M, K, A, lda, quantizedA.get(), K, scaleA.get(), zeroA.get(), sumA.get());
        t1.release();

        onednn_amx_gemm_s8s8s32(transA, M, N, K, alpha, quantizedA.get(), K, B, beta, C_int32.get(), N);

        TimeLine t2("onednn_amx_gemm_f32s8f32_compute.dequantC");
        dequant(M, N, C_int32.get(), N, C, ldc, scaleA.get(), zeroA.get(), sumA.get(), scaleB, zeroB, sumB, bias, res,
                ldres, gamma, kind);
    }

    // Per row quantization of activations
    // src: weight, dst: int8 qweight
    // weight = qweight * scale + zero
    //
    // Also compute per row sums
    // sum = sum_of_one_row(qweight * scale + zero)
    void quantize_s8(
            int M, int N, const float *src, int lda, int8_t *dst, int ldb, float *scale, float *zero, float *sum) {
#pragma omp parallel for
        for (uint16_t i = 0; i < M; i++) {
            __m512 vmax = _mm512_loadu_ps(src + i * lda);
            __m512 vmin = vmax;
            for (int j = 16; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 x = _mm512_maskz_loadu_ps(mask, src + i * lda + j);
                vmax = _mm512_mask_max_ps(vmax, mask, vmax, x);
                vmin = _mm512_mask_min_ps(vmin, mask, vmin, x);
            }

            float fmax = _mm512_reduce_max_ps(vmax);
            float fmin = _mm512_reduce_min_ps(vmin);

            //float fscale = (fmax - fmin) / 255.0f;
            //float fzero = (127 * fmin + 128 * fmax) / 255.0f;
            float fzero = (fmin + fmax) / 2.0f;
            float fscale = std::max(std::abs(fmax - fzero), std::abs(fzero - fmin)) / 127.0f;
            scale[i] = fscale;
            zero[i] = fzero;

            // weight = qweight * scale + zero
            // qweight = weight * (1/scale) + (-zero/scale)
            __m512 vscale = _mm512_set1_ps(1.0f / fscale);
            __m512 vzero = _mm512_set1_ps(-fzero / fscale);

            __m512i vsum = _mm512_setzero_epi32();
            for (int j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 x = _mm512_maskz_loadu_ps(mask, src + i * lda + j);
                x = _mm512_maskz_fmadd_ps(mask, x, vscale, vzero);
                __m512i ix = _mm512_maskz_cvtps_epi32(mask, x);
                vsum = _mm512_add_epi32(vsum, ix);
                __m128i i8x = _mm512_maskz_cvtepi32_epi8(mask, ix);
                memcpy(dst + i * ldb + j, &i8x, (remain >= 16 ? 16 : remain));
            }
            sum[i] = _mm512_reduce_add_epi32(vsum) * fscale + N * fzero;
        }
    }

    template <typename DequantOp, typename PostOp>
    void dequant_base(int M, int N, const int32_t *C_int32, const int ldc_int32, float *C, const int ldc,
            const DequantOp &dequant_op, const PostOp &post_op) {
#pragma omp parallel for collapse(2)
        for (uint64_t i = 0; i < M; i++) {
            for (int j = 0; j < N; j += 16) {
                __m512i xi = _mm512_load_epi32(C_int32 + i * ldc_int32 + j);
                __m512 x = dequant_op(xi, i, j);
                x = post_op(x, i, j);
                _mm512_storeu_ps(C + i * ldc + j, x);
            }
        }
    }

    // dequant C_int32 to C
    //
    // sumA = sum(QWeightA * ScaleA + ZeroA)
    // sumB = sum(QWeightB)
    // C_int32 = QWeightA * QWeightB
    //
    // C = WeightA * WeightB
    //   = (QWeightA * ScaleA + ZeroA) * (QWeightB * ScaleB + ZeroB)
    //   = ScaleA * ScaleB * C_int32 + SumB * ScaleB * ZeroA + ZeroB * SumA
    void dequant(int M, int N, const int32_t *C_int32, const int ldc_int32, float *C, const int ldc,
            const float *scaleA, const float *zeroA, const float *sumA, const float *scaleB, const float *zeroB,
            const float *sumB, const float *bias, const float *res, int ldres, float gamma, matmul_kinds kind) {
        auto dequant_op = [scaleA, zeroA, sumA, scaleB, zeroB, sumB](__m512i &vi, int row, int col) {
            __m512 v = _mm512_cvtepi32_ps(vi);
            __m512 vscaleB = _mm512_loadu_ps(scaleB + col);
            __m512 vzeroB = _mm512_loadu_ps(zeroB + col);
            __m512 vsumB = _mm512_loadu_ps(sumB + col);
            __m512 vscaleA = _mm512_set1_ps(scaleA[row]);
            __m512 vsumA = _mm512_set1_ps(sumA[row]);
            __m512 vzeroA = _mm512_set1_ps(zeroA[row]);
            return v * vscaleA * vscaleB + vsumB * vscaleB * vzeroA + vsumA * vzeroB;
        };

        auto no_post_op = [](__m512 &v, int row, int col) { return v; };
        auto biasadd = [bias](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            return _mm512_add_ps(v, vbias);
        };
        auto biasadd_relu = [bias](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            return _mm512_max_ps(_mm512_add_ps(v, vbias), _mm512_setzero_ps());
        };
        auto residential = [res, ldres](__m512 &v, int row, int col) {
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            return _mm512_add_ps(v, vres);
        };
        auto biasadd_res = [bias, res, ldres](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            return _mm512_add_ps(_mm512_add_ps(v, vbias), vres);
        };
        auto resext = [res, ldres, gamma](__m512 &v, int row, int col) {
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            __m512 vgamma = _mm512_set1_ps(gamma);
            return _mm512_fmadd_ps(vgamma, vres, v);
        };
        auto biasadd_resext = [bias, res, ldres, gamma](__m512 &v, int row, int col) {
            __m512 vbias = _mm512_loadu_ps(bias + col);
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            __m512 vgamma = _mm512_set1_ps(gamma);
            return _mm512_fmadd_ps(vgamma, vres, _mm512_add_ps(v, vbias));
        };
        auto resmul = [res, ldres](__m512 &v, int row, int col) {
            __m512 vres = _mm512_loadu_ps(res + row * ldres + col);
            return _mm512_mul_ps(v, vres);
        };
        auto silu = [](__m512 &v, int row, int col) {
            const __m512 vone = _mm512_set1_ps(1.0f);
            __m512 vp = BertUtil::vexp(v);
            __m512 vrecip = _mm512_rcp14_ps(vp + vone);
            return vp * vrecip * v;
        };
        auto gelu = [](__m512 &v, int row, int col) {
            const __m512 vone = _mm512_set1_ps(1.0f);
            const __m512 c1 = _mm512_set1_ps(1.702f);
            __m512 vp = BertUtil::vexp(v * c1);
            __m512 vrecip = _mm512_rcp14_ps(vp + vone);
            return vp * vrecip * v;
        };

        switch (kind) {
            case matmul_kinds::Basic: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, no_post_op); break;
            case matmul_kinds::BiasAdd: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd); break;
            case matmul_kinds::BiasAdd_Relu:
                dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd_relu);
                break;
            case matmul_kinds::Silu: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, silu); break;
            case matmul_kinds::Gelu: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, gelu); break;
            case matmul_kinds::Resmul: dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, resmul); break;
            case matmul_kinds::Residential:
                if (bias) {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd_res);
                } else {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, residential);
                }
                break;
            case matmul_kinds::Resext:
                if (bias) {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, biasadd_resext);
                } else {
                    dequant_base(M, N, C_int32, ldc_int32, C, ldc, dequant_op, resext);
                }
                break;
        }
    }
};
