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
 * This softmax impl is highly inspired by oneflow's blog.
 *
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mllm::cuda {

// One warp to process one line, packed 4 float
// for cols <= 1024
template<int THREAD_GROUP_NUM, int ROWS_PER_THREAD, int COLS_PER_THREAD>
__global__ void _warp_level_safe_softmax_fp32(float* __restrict__ z, const float* __restrict__ x,
                                              int rows, int cols);

}  // namespace mllm::cuda