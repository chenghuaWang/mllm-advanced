/**
 * @file Matmul.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/Tensor.hpp"

namespace mllm::nn::F {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A = false, bool transpose_B = false);

}
