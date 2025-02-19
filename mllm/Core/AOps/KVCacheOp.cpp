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
#include "mllm/Utils/Common.hpp"

namespace mllm {

KVCacheOp::KVCacheOp(const KVCacheOpCargo& cargo) : BaseOp(OpType::kKVCache), cargo_(cargo) {}

void KVCacheOp::load(std::shared_ptr<ParameterLoader>& ploader) {}

void KVCacheOp::trace(void* trace_context, std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  NYI("KVCacheOp::trace is not implemented")
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
