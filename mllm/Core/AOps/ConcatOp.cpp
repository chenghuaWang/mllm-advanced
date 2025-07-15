/**
 * @file ConcatOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/ConcatOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

namespace mllm {

ConcatOp::ConcatOp(const ConcatOpCargo& cargo) : BaseOp(OpType::kConcat), cargo_(cargo) {}

void ConcatOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void ConcatOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::ConcatOp>(shared_from_this(), i_irs, o_irs);
}

void ConcatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("ConcatOp::forward is not implemented");
}

void ConcatOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto at_dim = cargo_.dim;
  if (inputs.empty()) {
    MLLM_ERROR_EXIT(kError, "ConcatOp: no inputs");
    return;
  }
  const int n_dims = inputs[0].shape().size();
  if (at_dim < 0 || at_dim >= n_dims) {
    MLLM_ERROR_EXIT(kError, "ConcatOp: dim {} out of range [0, {})", at_dim, n_dims);
    return;
  }

  // Check
  for (int d = 0; d < n_dims; ++d) {
    if (d == at_dim) continue;
    const int ref = inputs[0].shape()[d];
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i].shape()[d] != ref) {
        MLLM_ERROR_EXIT(kError, "ConcatOp: non-concat dim {} mismatch ({} vs {})", d, ref,
                        inputs[i].shape()[d]);
        return;
      }
    }
  }

  std::vector<int> new_shape = inputs[0].shape();
  new_shape[at_dim] = 0;
  for (const auto& t : inputs) { new_shape[at_dim] += t.shape()[at_dim]; }

  outputs.emplace_back(Tensor::empty(new_shape, inputs[0].dtype(), inputs[0].device()));
}

void ConcatOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm