/**
 * @file FillOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/FillOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

FillOp::FillOp(const FillOpCargo& cargo) : BaseOp(OpType::kFill), cargo_(cargo) {}

void FillOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void FillOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                   std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::FillOp>(this, i_irs, o_irs);
}

void FillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("FillOp::forward is not implemented");
}

void FillOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  switch (cargo_.type) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
      // replace, using inputs' memory space
      outputs.emplace_back(inputs[0]);
      break;
    case 5:
      outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
      break;
  }
}

void FillOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  switch (cargo_.type) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4: break;
    case 5: outputs[0].alloc(); break;
  }
}

}  // namespace mllm
