// Copyright (c) 2024 Intel Corporation
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
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

bool firstPhaseDone = false;

bool enableCATMLP() {
    // combine gate&up and calculate together, default enabled
    static int catMlp = -1;
    static int catNextMlp = -1;
    if (!firstPhaseDone) {
        if (catMlp == -1)
            catMlp = (getenv("ENABLE_CAT_MLP") ? atoi(getenv("ENABLE_CAT_MLP")) : 1);
        return catMlp == 1;
    } else {
        if (catNextMlp == -1)
            catNextMlp = (getenv("ENABLE_CAT_NEXT_MLP") ? atoi(getenv("ENABLE_CAT_NEXT_MLP")) : 1);
        return catNextMlp == 1;
    }

}

bool enableMKLGemm() {
    static int mklGemm = -1;
    static int mklNextGemm = -1;
    if (!firstPhaseDone) {
        if (mklGemm == -1)
            mklGemm = (getenv("ENABLE_MKL_GEMM") ? atoi(getenv("ENABLE_MKL_GEMM")) : 0);
        return mklGemm == 1;
    } else {
        if (mklNextGemm == -1)
            mklNextGemm = (getenv("ENABLE_MKL_NEXT_GEMM") ? atoi(getenv("ENABLE_MKL_NEXT_GEMM")) : 0);
        return mklNextGemm == 1;
    }
}

bool sharedModel() {
    static int mklGemm = -1;
    static int mklNextGemm = -1;
    static int envPackM = -1;
    static int envPackNextM = -1;
    if (mklGemm == -1)
        mklGemm = (getenv("ENABLE_MKL_GEMM") ? atoi(getenv("ENABLE_MKL_GEMM")) : 0);
    if (mklNextGemm == -1)
        mklNextGemm = (getenv("ENABLE_MKL_NEXT_GEMM") ? atoi(getenv("ENABLE_MKL_NEXT_GEMM")) : 0);
    if (envPackM == -1)
        envPackM = (getenv("ENV_PACK_M") ? atoi(getenv("ENV_PACK_M")) : 0);
    if (envPackNextM == -1)
        envPackNextM = (getenv("ENV_PACK_NEXT_M") ? atoi(getenv("ENV_PACK_NEXT_M")) : 0);
    // no mkl or both unpacked mkl
    return (mklGemm == 0 && mklNextGemm == 0) || (mklGemm == 1 && mklNextGemm == 1 && envPackM == 0 && envPackNextM == 0);
}

int getPackM() {
    static int envPackM = -1;
    static int envPackNextM = -1;
    if (!firstPhaseDone) {
        if (envPackM == -1)
            envPackM = (getenv("ENV_PACK_M") ? atoi(getenv("ENV_PACK_M")) : 0);
        return envPackM;
    } else {
        if (envPackNextM == -1)
            envPackNextM = (getenv("ENV_PACK_NEXT_M") ? atoi(getenv("ENV_PACK_NEXT_M")) : 0);
        return envPackNextM;
    }
}

bool tunedComm() {
    // Tuning between shm and ccl reduceAdd methods to find the faster way, default enabled
    static int tunedComm = -1;
    if (tunedComm == -1) {
        tunedComm = (getenv("ENABLE_TUNED_COMM") ? atoi(getenv("ENABLE_TUNED_COMM")) : 1);
        if (tunedComm == 1) printf("ENABLE_TUNED_COMM is enabled for faster reduceAdd.\n");
    }
    return tunedComm == 1;
}

int getFlashThresh() {
    // > threshold to enable flash attention, default 1024
    static int envFlashThresh = -1;
    if (envFlashThresh == -1)
        envFlashThresh = (getenv("FLASH_ATTN_THRESHOLD") ? atoi(getenv("FLASH_ATTN_THRESHOLD")) : 1024);
    return envFlashThresh;
}

bool enableSkipMsk() {
    // Skip masked attn in flash attention for better perf of long sequence, default disabled
    static int skipMsk = -1;
    if (skipMsk == -1) {
        skipMsk = (getenv("ENABLE_SKIP_MASK") ? atoi(getenv("ENABLE_SKIP_MASK")) : 0);
        if (skipMsk == 1) printf("ENABLE_SKIP_MASK is enabled for ignoring mask Q*K.\n");
    }
    return skipMsk == 1;
}

bool kvTrans() {
    // Transpose KV Tensor to [batchSize, headNum, seqLen, headSize] for better perf of long sequence, default disabled
    // TODO: add support for reorder and expand when beam_search>1
    static int kvTrans = -1;
    if (kvTrans == -1) { kvTrans = (getenv("ENABLE_KV_TRANS") ? atoi(getenv("ENABLE_KV_TRANS")) : 0); }
    return kvTrans == 1;
}
