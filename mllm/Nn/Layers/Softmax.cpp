/**
 * @file Softmax.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layers/Softmax.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Nn/Layer.hpp"

namespace mllm::nn {

Softmax::Softmax() : Layer(OpType::kSoftmax, SoftmaxOpCargo{}) {}

Softmax::Softmax(const SoftmaxOpCargo& cargo) : Layer(OpType::kSoftmax, cargo) {}

Softmax::Softmax(int axis) : Layer(OpType::kSoftmax, SoftmaxOpCargo{.axis = axis}) {}

}  // namespace mllm::nn
