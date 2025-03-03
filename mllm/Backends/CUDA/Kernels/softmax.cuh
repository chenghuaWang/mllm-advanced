/**
 * @file softmax.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-02
 *
 * @copyright Copyright (c) 2025
 *
 * Some references:
 * 1. https://zhuanlan.zhihu.com/p/341059988 (oneflow's impl)
 * 2. https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
 *
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mllm::cuda {

// see:
// https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
template<int M_TILE, int N_TILE, int VEC_SIZE>
__global__ void online_safe_softmax_bf16(nv_bfloat16* z, const nv_bfloat16* x, int M, int N);

}  // namespace mllm::cuda