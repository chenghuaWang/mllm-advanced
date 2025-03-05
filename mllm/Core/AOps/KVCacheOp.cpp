/**
 * @file KVCacheOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/KVCacheOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

KVCacheOp::KVCacheOp(const KVCacheOpCargo& cargo) : BaseOp(OpType::kKVCache), cargo_(cargo) {}

void KVCacheOp::load(const std::shared_ptr<ParameterLoader>& ploader) {}

void KVCacheOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::KVCacheOp>(this, i_irs, o_irs);
}

void KVCacheOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("KVCacheOp::forward is not implemented")
}

void KVCacheOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("KVCacheOp::reshape is not implemented")
}

void KVCacheOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("KVCacheOp::setup is not implemented")
}

}  // namespace mllm
