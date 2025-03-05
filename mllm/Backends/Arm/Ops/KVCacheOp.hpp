/**
 * @file KVCacheOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/KVCacheOp.hpp"

namespace mllm::arm {

// input layout: [B, H, S, D]
// output layout: [B, H, S, D]
// cache layout: [B, H, S, D]
class ArmKVCacheOp final : public KVCacheOp {
 public:
  explicit ArmKVCacheOp(const KVCacheOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  size_t cur_kv_cache_limits_ = 0;
  size_t cur_kv_cache_seq_len_ = 0;
};

class ArmKVCacheOpFactory : public TypedOpFactory<OpType::kKVCache, KVCacheOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const KVCacheOpCargo& cargo) override {
    return std::make_shared<ArmKVCacheOp>(cargo);
  }
};

}  // namespace mllm::arm
