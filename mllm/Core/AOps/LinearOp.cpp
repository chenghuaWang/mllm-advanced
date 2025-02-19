/**
 * @file LinearOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

LinearOp::LinearOp(const LinearOpCargo& cargo) : BaseOp(OpType::kLinear), cargo_(cargo) {}

void LinearOp::load(std::shared_ptr<ParameterLoader>& ploader) {
  weight_ = Tensor(ploader->operator[](name() + ".weight"));
  if (cargo_.bias) { bias_ = Tensor(ploader->operator[](name() + ".bias")); }
}

void LinearOp::trace(void* trace_context, std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  NYI("LinearOp::trace is not implemented");
}

void LinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LinearOp::forward is not implemented");
}

void LinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], cargo_.in_channels);

  auto o_shape = i_shape;
  o_shape[o_shape.size() - 1] = cargo_.out_channels;

  outputs.emplace_back(Tensor::empty(o_shape, i.dtype(), i.device()));
}

void LinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
