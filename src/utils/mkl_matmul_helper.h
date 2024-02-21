// Copyright (c) 2023-2024 Intel Corporation
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
#include "bfloat16.h"
#include "dtype.h"
#include "float16.h"
#include "my_types.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "timer.h"
#include "matmul_helper.h"
#include "simple_mem_pool.h"
#include "intrinsics_util.h"
#include "copy_util.h"

#include <cstring>
#include <map>
#include <tuple>

#define PINFO true

extern int getPackM();
extern bool enableMKLGemm();
extern bool sharedModel();

class MKLMMHelper {
public:
    MKLMMHelper(MMHelper *helper) {
        baseHelper = helper;
    }

    ~MKLMMHelper() {
        if (baseHelper) delete baseHelper;
    }

    MMHelper *getBase() { return baseHelper; };

    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            int splitOffset, int splitSize, bool verticalSplit, hpj::Matrix<WeiT> &convertedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight,
            bool unused) {
        baseHelper->convertWeight(trans, rows, cols, weight, scales, zeros, splitOffset, splitSize, verticalSplit,
                convertedWeight, scaleWeight, zeroWeight, sumWeight, unused);
    }
    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            int numSplit, int splitIdx, bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight) {
        baseHelper->convertWeight(trans, rows, cols, weight, scales, zeros, numSplit, splitIdx, verticalSplit,
                quantizedWeight, scaleWeight, zeroWeight, sumWeight);
    }
    template <typename OriWeiT, typename WeiT>
    void convertWeight(bool trans, int rows, int cols, const OriWeiT *weight, const float *scales, const float *zeros,
            hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight,
            hpj::Vector<float> &sumWeight) {
        baseHelper->convertWeight(trans, rows, cols, weight, scales, zeros, quantizedWeight, scaleWeight,
                zeroWeight, sumWeight);
    }
    template <typename OriWeiT, typename WeiT>
    void convertWeight(DecoderContext *ctx, bool trans, int rows, int cols, const OriWeiT *weight, const float *scales,
            const float *zeros, bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, hpj::Vector<float> &sumWeight) {
        baseHelper->convertWeight(ctx, trans, rows, cols, weight, scales, zeros, verticalSplit, quantizedWeight, scaleWeight,
                zeroWeight, sumWeight);
    }
    template <typename InT, typename WeiT, typename OutT>
    void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            const float *bias) {
	Timer tmc(PINFO, "compute biasadd relu");
        baseHelper->compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc, bias);
    }

    template <typename WeiT>
    void packWeight(bool trans, hpj::Matrix<WeiT> &src, hpj::Matrix<WeiT> &weight, bool mkl = true) {
        if (!mkl || !enableMKLGemm()) {
            baseHelper->packWeight(trans, src, weight);
            return;
        }

        if (trans) {
            printf("Unsupported trans for MKL MatMul Pack\n");
            exit(1);
        }
        int K = src.Rows();
        int N = src.Cols();

        // BF16
        if (std::is_same_v<WeiT, bfloat16_t>) {
            // packM should be equal to M for the best
            int packM = getPackM();
            if (packM != 0) {
                size_t bsize = cblas_gemm_bf16bf16f32_pack_get_size(CblasBMatrix, packM, N, K);
                int ld = (bsize / sizeof(WeiT)  + K - 1) / K;
                weight.Resize(K, N, ld);
                printf("K N ld bsize getPackM()(%d %d %d, %lld %d) \n", K, N, ld, bsize, packM);
                cblas_gemm_bf16bf16f32_pack(
                    CblasRowMajor, CblasBMatrix, CblasNoTrans, packM, N, K,
                        (const MKL_BF16 *)(src.Data()), N, (MKL_BF16 *)weight.Data());
            } else {
                int ld = src.Stride();
                weight.Resize(K, N, ld);
#pragma omp parallel for
                for (uint64_t i = 0; i < K; ++i) {
                    xft::copy((WeiT *)weight.Data() + i * ld, (WeiT *)src.Data() + i * ld, N);
                }
            }
        } else {
            printf("Unsupported data type for MKL MatMul\n");
            exit(1);
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, bool mkl = true) {
	Timer tmc(PINFO, "compute");
        if (!mkl || !enableMKLGemm()) {
            baseHelper->compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc);
            return;
        }

        if (transA) {
            printf("Unsupported trans for MKL MatMul\n");
            exit(1);
        }

        int ldb = ldc;// no effect for packedB
        compute_core(A, packedB, C, M, N, K, lda, ldb, ldc, alpha, beta);
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_bias(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc,
            const float *bias, bool mkl = true) {
	Timer tmc(PINFO, "compute bias");
        if (!mkl || !enableMKLGemm()) {
            baseHelper->compute_bias(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc, bias);
            return;
        }

        if (transA) {
            printf("Unsupported trans for MKL MatMul\n");
            exit(1);
        }

        int ldb = ldc;// no effect for packedB
        compute_core(A, packedB, C, M, N, K, lda, ldb, ldc, alpha, beta, bias);
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_residential(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const float *bias,
            const InT *res, int ldres, bool mkl = true) {
	Timer tmc(PINFO, "compute residential");
        if (!mkl || !enableMKLGemm()) {
            baseHelper->compute_residential(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc, bias,
                    res, ldres);
            return;
        }

        if (transA) {
            printf("Unsupported trans for MKL MatMul\n");
            exit(1);
        }

        int ldb = ldc;// no effect for packedB
        compute_core(A, packedB, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, res, ldres);
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_resext(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const float *bias,
            float gamma, InT *res, int ldres, bool mkl = true) {
	Timer tmc(PINFO, "compute resext");
        if (!mkl || !enableMKLGemm()) {
            baseHelper->compute_resext(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc, bias,
                    gamma, res, ldres);
            return;
        }

        if (transA) {
            printf("Unsupported trans for MKL MatMul\n");
            exit(1);
        }

        int ldb = ldc;// no effect for packedB
        compute_core(A, packedB, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, res, ldres, gamma);
    }

/*
    template <typename InT, typename WeiT, typename OutT>
    void compute_resmul(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, const InT *res,
        int ldres) {
        // unplemented by MKL lib, just use baseImplementation;
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, bool mkl = true) {
        // unplemented by MKL lib, just use baseImplementation;
    }
*/

    template <typename InT, typename WeiT, typename OutT>
    void compute_cat_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, OutT *silu, int ldd,
            bool mkl = true) {
	Timer tmc(PINFO, "compute cat silu");
        if (!mkl || !enableMKLGemm()) {
            baseHelper->compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc);
            siluSum(C, silu, M, N / 2, ldc, ldd);
            return;
        }

        if (transA) {
            printf("Unsupported trans for MKL MatMul\n");
            exit(1);
        }

        // ldb: no effect for packedB
        compute_core(A, packedB, C, M, N, K, lda, ldc, ldc, alpha, beta);
        siluSum(C, silu, M, N / 2, ldc, ldd);
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_cat_gelu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, const float *sumB, float beta, OutT *C, int ldc, OutT *silu, int ldd,
            bool mkl = true) {
	Timer tmc(PINFO, "compute cat silu");
        if (!mkl || !enableMKLGemm()) {
            baseHelper->compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, sumB, beta, C, ldc);
            geluSum(C, silu, M, N / 2, ldc, ldd);
            return;
        }

        if (transA) {
            printf("Unsupported trans for MKL MatMul\n");
            exit(1);
        }

        // ldb: no effect for packedB
        compute_core(A, packedB, C, M, N, K, lda, ldc, ldc, alpha, beta);
        geluSum(C, silu, M, N / 2, ldc, ldd);
    }

    template <typename InT, typename WeiT, typename OutT>
    void compute_core(const InT *A, const WeiT *B, OutT *C, int M, int N, int K, int lda, int ldb, int ldc,
            float alpha, float beta, const float *bias = nullptr, const InT *R = nullptr, int ldr = 0, float gamma = 1) {
        static_assert(std::is_same_v<OutT, InT>, "Error: MKL compute Input and Output are not the same type.");
        // BF16
        if (std::is_same_v<WeiT, bfloat16_t>) {

            // convert A to BF16
            int ldaH = lda;
            if (std::is_same_v<InT, float>) {
                // Timer tmc(PINFO, "convert A matrix into bf16");
                ldaH *= sizeof(float) / sizeof(WeiT);
#pragma omp parallel for
                for (uint64_t i = 0; i < M; ++i) {
                    bfloat16_t::cvt_float_to_bfloat16((const float *)A + i * lda, (bfloat16_t *)A + i * ldaH, K);
                }
            }

            //printf("MNK (%d %d %d) ldabcr(%d %d %d %d) beta %f(%f) bias %lld\n", M, N, K, ldaH, ldb, ldc, ldr, beta, gamma, bias);

	    // R = nullptr;
            if (R != nullptr) {
                if ((InT *)C != (InT *)R) {
                    // Timer tmc(PINFO, "merge Out and Res");
                    if (beta != 0.0) {
                        printf("Unsupported non-zero beta for MKL MatMul now\n");
                        exit(1);
                    }
#pragma omp parallel for
                    for (uint64_t i = 0; i < M; ++i) {
                        xft::copy(C + i * ldc, R + i * ldr, N);
                    }
                }
                beta += gamma;
            }

            // set gemm out buffer for mkl func
            if (std::is_same_v<OutT, float>) {
                gemm_bf16bf16f32_compute((const MKL_BF16 *)A, (const MKL_BF16 *)B, (float *)C, M, N, K, alpha, beta, ldaH, ldb, ldc, getPackM());
                if (bias != nullptr)
                    biasAdd((float *)C, bias, M, N, ldc);

            } else {
                // add residentail later
                float *shardedOut = (float *)SimpleMemPool::instance().getBuffer(
                    "shardedMKLOut", sizeof(float) * M * ldc);
                gemm_bf16bf16f32_compute((const MKL_BF16 *)A, (const MKL_BF16 *)B, shardedOut, M, N, K, alpha, 0.0, ldaH, ldb, ldc, getPackM());
                // Timer tmc1(PINFO, "merge biasBinaryAdd cvt");
		// if (false)
                biasBinaryAdd(C, beta, shardedOut, bias, M, N, ldc);
            }

        } else {
            printf("Unsupported data type for MKL MatMul\n");
            exit(1);
        }
    }

private:
    static void gemm_bf16bf16f32_compute(const MKL_BF16 *A, const MKL_BF16 *B, float *C, int M, int N, int K, float alpha, float beta,
            int lda, int ldb, int ldc, int packM = 0) {
        if (packM != 0) {
            //printf("packM MNK (%d %d %d) ldabc(%d %d %d) alpha beta %lf %lf pack %d\n", M, N, K, lda, ldb, ldc, alpha, beta, packM);
            cblas_gemm_bf16bf16f32_compute(
                CblasRowMajor, CblasNoTrans, CblasPacked, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            //printf("MNK (%d %d %d) ldabc(%d %d %d) alpha beta %lf %lf pack %d\n", M, N, K, lda, ldb, ldc, alpha, beta, packM);
            cblas_gemm_bf16bf16f32(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }

    // Intrinsic binaryAdd and biasAdd
    // C = (cvt)(C * beta + in + bias)
    template <typename T>
    static void biasBinaryAdd(T *out, float beta, const float *in, const float *bias, int M, int N, int ld) {
        __m512 vbeta = _mm512_set1_ps(beta);

#pragma omp parallel for collapse(2)
        for (uint64_t i = 0; i < M; ++i) {
            for (uint64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto idat = xft::load_avx512(in + i * ld + j);
                if (beta != 0.0) {
                    auto odat = xft::load_avx512((const T *)out + i * ld + j);
                    idat = _mm512_add_ps(idat, odat * vbeta);
                }
                if (bias != nullptr) {
                    auto bdat = xft::load_avx512(bias + j);
                    idat = _mm512_add_ps(idat, bdat);
                }
                xft::store_avx512(out + i * ld + j, mask, idat);
            }
        }
    }

    // Intrinsic biasAdd
    static void biasAdd(float *out, const float *bias, int M, int N, int ld) {
#pragma omp parallel for collapse(2)
        for (uint64_t i = 0; i < M; ++i) {
            for (uint64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto dat = xft::load_avx512((const float *)out + i * ld + j);
                auto bdat = xft::load_avx512(bias + j);
                dat = _mm512_add_ps(dat, bdat);
                xft::store_avx512(out + i * ld + j, mask, dat);
            }
        }
    }

    // compute silu on the left half and then add it with the right half
    template <typename T1, typename T2>
    static void siluSum(const T1 *src, T2 *dst, int M, int N, int lds, int ldd) {
        __m512 one = _mm512_set1_ps(1.f);
        __m512 negOne = _mm512_set1_ps(-1.f);

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto left = xft::load_avx512(mask, src + i * lds + j);
                auto right = xft::load_avx512(mask, src + i * lds + j + N);
                auto x0 = BertUtil::vexp(_mm512_mul_ps(left, negOne));
                auto x1 = _mm512_add_ps(one, x0);
                auto x2 = _mm512_div_ps(left, x1);
                auto res = _mm512_mul_ps(right, x2);
                xft::store_avx512(dst + i * ldd + j, mask, res);
            }
        }
    }

    // compute silu on the left half and then add it with the right half
    template <typename T1, typename T2>
    static void geluSum(const T1 *src, T2 *dst, int M, int N, int lds, int ldd) {
        const __m512 c1 = _mm512_set1_ps(0.044715f);
        const __m512 c2 = _mm512_set1_ps(0.7978845608f);
        const __m512 vone = _mm512_set1_ps(1.0f);
        const __m512 vtwo = _mm512_set1_ps(2.0f);
        const __m512 vhalf = _mm512_set1_ps(0.5f);

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                auto vx = xft::load_avx512(mask, src + i * lds + j);
                auto right = xft::load_avx512(mask, src + i * lds + j + N);
                __m512 vt = c2 * (vx + c1 * vx * vx * vx);
                vt = BertUtil::vexp(vt * vtwo);
                vt = vone - vtwo * _mm512_rcp14_ps(vt + vone); // tanh
                __m512 vy = vx * (vone + vt) * vhalf;
                auto res = _mm512_mul_ps(right, vy);
                xft::store_avx512(dst + i * ldd + j, mask, res);
            }
        }
    }

private:
    MMHelper *baseHelper;
};
