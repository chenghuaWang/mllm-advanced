/**
 * @file fp32_s8s_pto_key.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-26
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#ifndef __ANDROID__
#error "This file is only for Arm Android platform"
#endif

#include <cstdint>

namespace mllm::arm {

void fp32_s8s_pto_key_bshd(int8_t* __restrict__ Z, float* __restrict__ scale,
                           const float* __restrict__ X, int B, int S, int H, int D,
                           bool clamp = false, float clamp_min = 0.f, float clamp_max = 0.f);

}