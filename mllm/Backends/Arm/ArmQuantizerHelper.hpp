/**
 * @file ArmQuantizerHelper.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// ArmQuantizerHelper.hpp implements parameter packing functions required by kleidiai. These
// functions are designed to preprocess model parameters prior to inference. We recommend performing
// all parameter packing operations offline rather than during runtime for:
// 1. Memory Optimization: Eliminates temporary memory overhead during online processing
// 2. Execution Efficiency: Reduces computational time during inference phase

#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

namespace mllm::arm {}
