/**
 * @file ElewiseOps.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/ElewiseOp.hpp"

namespace mllm::cuda {

class CUDAAddOp final : public AddOp {
 public:
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CUDAAddOpFactory final : public TypedOpFactory<OpType::kAdd, AddOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const AddOpCargo& cargo) override {
    return std::make_shared<CUDAAddOp>();
  }
};

}  // namespace mllm::cuda