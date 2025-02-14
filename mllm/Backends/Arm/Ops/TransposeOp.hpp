/**
 * @file TransposeOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/TransposeOp.hpp"

namespace mllm::arm {

class ArmTransposeOp final : public TransposeOp {
 public:
  explicit ArmTransposeOp(const TransposeOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmTransposeOpFactory : public TypedOpFactory<OpType::kKVCache, TransposeOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const TransposeOpCargo& cargo) override {
    return std::make_shared<ArmTransposeOp>(cargo);
  }
};

}  // namespace mllm::arm
