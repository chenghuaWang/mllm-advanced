/**
 * @file Softmax.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/Layer.hpp"
#include "mllm/Core/AOps/SoftmaxOp.hpp"

namespace mllm::nn {

class Softmax : public Layer {
 public:
  Softmax();
  explicit Softmax(const SoftmaxOpCargo& cargo);
  explicit Softmax(int axis);
};

}  // namespace mllm::nn