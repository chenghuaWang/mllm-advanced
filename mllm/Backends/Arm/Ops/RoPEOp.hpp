/**
 * @file RoPEOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/RoPEOp.hpp"

namespace mllm::arm {

class ArmRoPEOp final : public RoPEOp {
 public:
  explicit ArmRoPEOp(const RoPEOpCargo& cargo);

  void load(std::shared_ptr<ParameterLoader>& ploader) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  Tensor sin_, cos_;
};

class ArmRoPEOpFactory : public TypedOpFactory<OpType::kKVCache, RoPEOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const RoPEOpCargo& cargo) override {
    return std::make_shared<ArmRoPEOp>(cargo);
  }
};

}  // namespace mllm::arm