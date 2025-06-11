/**
 * @file fp32_s8s_pt.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/X86/Kernels/Quantize/fp32_s8s_pt.hpp"
#include <hwy/highway.h>
#include <algorithm>
#include <cmath>

namespace hn = hwy::HWY_NAMESPACE;

namespace mllm::X86 {

void fp32_s8s_pt_2d_offline(int8_t* __restrict__ Z, float* __restrict__ scale,
                            const float* __restrict__ X, int sequence, int dim) {
  if (dim <= 0 || sequence <= 0) {
    *scale = 0.0f;
    return;
  }

  using namespace hn;  // NOLINT
  const HWY_FULL(float) d_f;
  using V = decltype(Zero(d_f));
  const size_t max_lanes = Lanes(d_f);
  float abs_max = 0.0f;

#pragma omp parallel for reduction(max : abs_max)
  for (int s = 0; s < sequence; ++s) {
    const float* row = X + s * dim;
    V v_max = Set(d_f, 0.0f);

    size_t i = 0;
    for (; i + max_lanes <= static_cast<size_t>(dim); i += max_lanes) {
      const V v = Load(d_f, row + i);
      v_max = Max(v_max, Abs(v));
    }

    // Handle tail elements
    if (i < static_cast<size_t>(dim)) {
      const V v = LoadN(d_f, row + i, static_cast<size_t>(dim) - i);
      v_max = Max(v_max, Abs(v));
    }

    // Reduce vector to scalar
    const float row_max = GetLane(MaxOfLanes(d_f, v_max));
    if (row_max > abs_max) abs_max = row_max;
  }

  // Calculate scale factor
  *scale = abs_max / 127.0f;

  // Special case: all zeros
  if (abs_max == 0.0f) {
#pragma omp parallel for
    for (int s = 0; s < sequence; ++s) { std::fill(Z + s * dim, Z + (s + 1) * dim, 0); }
    return;
  }

  const float inv_scale = 127.0f / abs_max;

// Second pass: quantize and clamp
#pragma omp parallel for
  for (int s = 0; s < sequence; ++s) {
    const float* x_row = X + s * dim;
    int8_t* z_row = Z + s * dim;

    const Rebind<int32_t, decltype(d_f)> d_i32;
    const auto v_inv_scale = Set(d_f, inv_scale);

    size_t i = 0;
    for (; i + max_lanes <= static_cast<size_t>(dim); i += max_lanes) {
      V v = Load(d_f, x_row + i);
      v = Mul(v, v_inv_scale);
      v = Round(v);

      auto v_int32 = ConvertTo(d_i32, v);
      v_int32 = Max(v_int32, Set(d_i32, -127));
      v_int32 = Min(v_int32, Set(d_i32, 127));

      int32_t temp[max_lanes];
      Store(v_int32, d_i32, temp);

      for (size_t j = 0; j < max_lanes; ++j) { z_row[i + j] = static_cast<int8_t>(temp[j]); }
    }

    // Handle tail elements
    if (i < static_cast<size_t>(dim)) {
      const size_t remaining = static_cast<size_t>(dim) - i;
      V v = LoadN(d_f, x_row + i, remaining);
      v = Mul(v, v_inv_scale);
      v = Round(v);

      auto v_int32 = ConvertTo(d_i32, v);
      v_int32 = Max(v_int32, Set(d_i32, -127));
      v_int32 = Min(v_int32, Set(d_i32, 127));

      int32_t temp[max_lanes];
      Store(v_int32, d_i32, temp);

      for (size_t j = 0; j < remaining; ++j) { z_row[i + j] = static_cast<int8_t>(temp[j]); }
    }
  }
}

}  // namespace mllm::X86
