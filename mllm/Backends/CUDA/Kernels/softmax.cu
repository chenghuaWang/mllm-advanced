/**
 * @file softmax.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cute/tensor.hpp>
#include "mllm/Backends/CUDA/Kernels/softmax.cuh"

using namespace cute;

namespace mllm::cuda {

template<int M_TILE, int N_TILE, int VEC_SIZE>
__global__ void online_safe_softmax_bf16(nv_bfloat16* z, const nv_bfloat16* x, int M, int N) {}

}  // namespace mllm::cuda