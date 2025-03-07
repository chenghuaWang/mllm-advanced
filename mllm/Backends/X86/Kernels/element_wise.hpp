/**
 * @file element_wise.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <hwy/highway.h>

namespace mllm::X86 {

void ele_wise_add_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads);

void ele_wise_sub_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads);

void ele_wise_mul_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads);

void ele_wise_div_f32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                      float* HWY_RESTRICT c, size_t nums, int threads);

}  // namespace mllm::X86