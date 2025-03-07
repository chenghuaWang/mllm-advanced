/**
 * @file element_wise.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <hwy/highway.h>

namespace hw = hwy::HWY_NAMESPACE;

namespace mllm::X86 {
void ele_wise_add_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads) {
  const hw::ScalableTag<float> d;
  const size_t N = hw::Lanes(d);
  size_t i = 0;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (; i + N <= nums; i += N) {
    const auto a_vec = hw::LoadU(d, a + i);
    const auto b_vec = hw::LoadU(d, b + i);
    const auto sum = hw::Add(a_vec, b_vec);
    hw::StoreU(sum, d, c + i);
  }

  if (i < nums) {
    const size_t remaining = nums - i;
    const auto mask = hw::FirstN(d, remaining);

    const auto a_vec = hw::MaskedLoad(mask, d, a + i);
    const auto b_vec = hw::MaskedLoad(mask, d, b + i);
    const auto sum = hw::Add(a_vec, b_vec);

    const auto masked_sum = hw::IfThenElseZero(mask, sum);
    hw::StoreU(masked_sum, d, c + i);
  }
}

void ele_wise_sub_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads) {
  const hw::ScalableTag<float> d;
  const size_t N = hw::Lanes(d);
  size_t i = 0;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (; i + N <= nums; i += N) {
    const auto a_vec = hw::LoadU(d, a + i);
    const auto b_vec = hw::LoadU(d, b + i);
    const auto sum = hw::Sub(a_vec, b_vec);
    hw::StoreU(sum, d, c + i);
  }

  if (i < nums) {
    const size_t remaining = nums - i;
    const auto mask = hw::FirstN(d, remaining);

    const auto a_vec = hw::MaskedLoad(mask, d, a + i);
    const auto b_vec = hw::MaskedLoad(mask, d, b + i);
    const auto sum = hw::Sub(a_vec, b_vec);

    const auto masked_sum = hw::IfThenElseZero(mask, sum);
    hw::StoreU(masked_sum, d, c + i);
  }
}

void ele_wise_mul_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads) {
  const hw::ScalableTag<float> d;
  const size_t N = hw::Lanes(d);
  size_t i = 0;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (; i + N <= nums; i += N) {
    const auto a_vec = hw::LoadU(d, a + i);
    const auto b_vec = hw::LoadU(d, b + i);
    const auto sum = hw::Mul(a_vec, b_vec);
    hw::StoreU(sum, d, c + i);
  }

  if (i < nums) {
    const size_t remaining = nums - i;
    const auto mask = hw::FirstN(d, remaining);

    const auto a_vec = hw::MaskedLoad(mask, d, a + i);
    const auto b_vec = hw::MaskedLoad(mask, d, b + i);
    const auto sum = hw::Mul(a_vec, b_vec);

    const auto masked_sum = hw::IfThenElseZero(mask, sum);
    hw::StoreU(masked_sum, d, c + i);
  }
}

void ele_wise_div_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads) {
  const hw::ScalableTag<float> d;
  const size_t N = hw::Lanes(d);
  size_t i = 0;

#pragma omp parallel for num_threads(threads) schedule(auto) if (threads > 0)
  for (; i + N <= nums; i += N) {
    const auto a_vec = hw::LoadU(d, a + i);
    const auto b_vec = hw::LoadU(d, b + i);
    const auto sum = hw::Div(a_vec, b_vec);
    hw::StoreU(sum, d, c + i);
  }

  if (i < nums) {
    const size_t remaining = nums - i;
    const auto mask = hw::FirstN(d, remaining);

    const auto a_vec = hw::MaskedLoad(mask, d, a + i);
    const auto b_vec = hw::MaskedLoad(mask, d, b + i);
    const auto sum = hw::Div(a_vec, b_vec);

    const auto masked_sum = hw::IfThenElseZero(mask, sum);
    hw::StoreU(masked_sum, d, c + i);
  }
}
}  // namespace mllm::X86
