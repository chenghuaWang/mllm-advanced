/**
 * @file Conv3DOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/Conv3DOp.hpp"

namespace mllm::arm {

class ArmConv3DOp final : public Conv3DOp {
 public:
  explicit ArmConv3DOp(const Conv3DOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class ArmConv3DOpFactory : public TypedOpFactory<OpType::kConv3D, Conv3DOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const Conv3DOpCargo& cargo) override {
    return std::make_shared<ArmConv3DOp>(cargo);
  }
};

}  // namespace mllm::arm