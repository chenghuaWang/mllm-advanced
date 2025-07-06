/**
 * @file PermuteOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "mllm/Core/AOps/PermuteOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

PermuteOp::PermuteOp(const PermuteOpCargo& cargo) : BaseOp(OpType::kPermute), cargo_(cargo) {}

void PermuteOp::load(const std::shared_ptr<ParameterLoader>& ploader) { MLLM_EMPTY_SCOPE; }

void PermuteOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::PermuteOp>(shared_from_this(), i_irs, o_irs);
}

void PermuteOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("PermuteOp::forward is not implemented");
}

void PermuteOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();
  std::vector<int32_t> new_shape(i_shape.size(), 0);
  for (int i = 0; i < i_shape.size(); ++i) { new_shape[i] = i_shape[cargo_.permute_dims[i]]; }
  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void PermuteOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm