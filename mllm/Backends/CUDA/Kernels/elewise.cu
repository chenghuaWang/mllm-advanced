/**
 * @file elewise.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/CUDA/Kernels/elewise.cuh"
#include <cute/tensor.hpp>

namespace mllm::cuda {

template<int NUM_ELE_PER_THREAD>
__global__ void vector_add_bf16_v0(nv_bfloat16* z, const nv_bfloat16* x, const nv_bfloat16* y,
                                   int num, const nv_bfloat16 a, const nv_bfloat16 b,
                                   const nv_bfloat16 c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / NUM_ELE_PER_THREAD) {
#pragma unroll
    for (int i = 0; i < NUM_ELE_PER_THREAD; ++i) {
      int global_idx = idx * NUM_ELE_PER_THREAD + i;
      if (global_idx < num) { z[i] = x[i] * a + (y[i] * b + c); }
    }
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));

  // register tensor
  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  nv_bfloat162 a2 = {a, a};
  nv_bfloat162 b2 = {b, b};
  nv_bfloat162 c2 = {c, c};

  auto tzR2 = recast<nv_bfloat162>(tzR);
  auto txR2 = recast<nv_bfloat162>(txR);
  auto tyR2 = recast<nv_bfloat162>(tyR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) { tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2); }

  auto tzRx = recast<nv_bfloat16>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}

// instance to make dynamic lib's symbols visible
template __global__ void vector_add_bf16_v0<8>(nv_bfloat16* z, const nv_bfloat16* x,
                                               const nv_bfloat16* y, int num, const nv_bfloat16 a,
                                               const nv_bfloat16 b, const nv_bfloat16 c);

template<int NUM_ELE_PER_THREAD>
__global__ void vector_sub_bf16_v0(nv_bfloat16* z, const nv_bfloat16* x, const nv_bfloat16* y,
                                   int num) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / NUM_ELE_PER_THREAD) {
#pragma unroll
    for (int i = 0; i < NUM_ELE_PER_THREAD; ++i) {
      int global_idx = idx * NUM_ELE_PER_THREAD + i;
      if (global_idx < num) { z[i] = x[i] - y[i]; }
    }
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));

  // register tensor
  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  auto tzR2 = recast<nv_bfloat162>(tzR);
  auto txR2 = recast<nv_bfloat162>(txR);
  auto tyR2 = recast<nv_bfloat162>(tyR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) { tzR2(i) = txR2(i) - tyR2(i); }

  auto tzRx = recast<nv_bfloat16>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}

// instance to make dynamic lib's symbols visible
template __global__ void vector_sub_bf16_v0<8>(nv_bfloat16* z, const nv_bfloat16* x,
                                               const nv_bfloat16* y, int num);

}  // namespace mllm::cuda