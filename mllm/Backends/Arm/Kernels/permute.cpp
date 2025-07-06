/**
 * @file permute.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else
#include <arm_neon.h>
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/Arm/Kernels/permute.hpp"

namespace mllm::arm {

namespace MLLM_NAMESPACE_ANONYMOUS {
void compute_strides(const int* shape, int ndim, int* strides) {
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) { strides[i] = strides[i + 1] * shape[i + 1]; }
}
}  // namespace MLLM_NAMESPACE_ANONYMOUS

void permute_fp32(const float* __restrict__ input, float* __restrict__ output,
                  const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim) {
  int out_shape[ndim];
  for (int i = 0; i < ndim; ++i) { out_shape[i] = in_shape[perm[i]]; }
  int in_strides[ndim], out_strides[ndim];
  compute_strides(in_shape, ndim, in_strides);
  compute_strides(out_shape, ndim, out_strides);
  int total_elements = 1;
  for (int i = 0; i < ndim; ++i) { total_elements *= in_shape[i]; }
  bool inner_dim_contiguous = (perm[ndim - 1] == ndim - 1);
  int inner_dim_size = out_shape[ndim - 1];
  if (inner_dim_contiguous && inner_dim_size >= 4) {
    int outer_elements = total_elements / inner_dim_size;
    int chunk_size = 4;
    for (int outer_idx = 0; outer_idx < outer_elements; ++outer_idx) {
      int coord[ndim - 1];
      int temp = outer_idx;
      for (int i = ndim - 2; i >= 0; --i) {
        coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }
      int in_offset = 0;
      int out_offset = 0;
      for (int i = 0; i < ndim - 1; ++i) {
        int orig_dim = perm[i];
        in_offset += coord[i] * in_strides[orig_dim];
        out_offset += coord[i] * out_strides[i];
      }
      const float* in_ptr = input + in_offset;
      float* out_ptr = output + out_offset;
      int j = 0;
      for (; j <= inner_dim_size - chunk_size; j += chunk_size) {
        float32x4_t vec = vld1q_f32(in_ptr + j);
        vst1q_f32(out_ptr + j, vec);
      }
      for (; j < inner_dim_size; ++j) { out_ptr[j] = in_ptr[j]; }
    }
  } else {
    for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
      int out_coord[ndim];
      int temp = linear_idx;
      for (int i = ndim - 1; i >= 0; --i) {
        out_coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }
      int in_coord[ndim];
      for (int i = 0; i < ndim; ++i) { in_coord[perm[i]] = out_coord[i]; }
      int in_idx = 0;
      for (int i = 0; i < ndim; ++i) { in_idx += in_coord[i] * in_strides[i]; }
      output[linear_idx] = input[in_idx];
    }
  }
}

void permute_fp16(const float16_t* __restrict__ input, float16_t* __restrict__ output,
                  const int* __restrict__ in_shape, const int* __restrict__ perm, int ndim) {
  int out_shape[ndim];
  for (int i = 0; i < ndim; ++i) { out_shape[i] = in_shape[perm[i]]; }

  int in_strides[ndim], out_strides[ndim];
  compute_strides(in_shape, ndim, in_strides);
  compute_strides(out_shape, ndim, out_strides);

  int total_elements = 1;
  for (int i = 0; i < ndim; ++i) { total_elements *= in_shape[i]; }

  bool inner_dim_contiguous = (perm[ndim - 1] == ndim - 1);
  int inner_dim_size = out_shape[ndim - 1];

  if (inner_dim_contiguous && inner_dim_size >= 8) {
    int outer_elements = total_elements / inner_dim_size;
    const int chunk_size = 8;

    for (int outer_idx = 0; outer_idx < outer_elements; ++outer_idx) {
      int coord[ndim - 1];
      int temp = outer_idx;
      for (int i = ndim - 2; i >= 0; --i) {
        coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }

      int in_offset = 0;
      int out_offset = 0;
      for (int i = 0; i < ndim - 1; ++i) {
        int orig_dim = perm[i];
        in_offset += coord[i] * in_strides[orig_dim];
        out_offset += coord[i] * out_strides[i];
      }

      const float16_t* in_ptr = input + in_offset;
      float16_t* out_ptr = output + out_offset;

      int j = 0;
      for (; j <= inner_dim_size - chunk_size; j += chunk_size) {
        float16x8_t vec = vld1q_f16(in_ptr + j);
        vst1q_f16(out_ptr + j, vec);
      }

      if (inner_dim_size - j >= 4) {
        float16x4_t vec4 = vld1_f16(in_ptr + j);
        vst1_f16(out_ptr + j, vec4);
        j += 4;
      }

      for (; j < inner_dim_size; j++) { out_ptr[j] = in_ptr[j]; }
    }
  } else {
    for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
      int out_coord[ndim];
      int temp = linear_idx;
      for (int i = ndim - 1; i >= 0; --i) {
        out_coord[i] = temp % out_shape[i];
        temp /= out_shape[i];
      }

      int in_coord[ndim];
      for (int i = 0; i < ndim; ++i) { in_coord[perm[i]] = out_coord[i]; }

      int in_idx = 0;
      for (int i = 0; i < ndim; ++i) { in_idx += in_coord[i] * in_strides[i]; }

      output[linear_idx] = input[in_idx];
    }
  }
}

}  // namespace mllm::arm

#endif