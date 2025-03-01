/**
 * @file D2HOp.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/D2HOp.hpp"

namespace mllm::cuda {

class CUDAD2HOp final : public D2HOp {
 public:
  explicit CUDAD2HOp(const D2HOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CUDAD2HOpFactory final : public TypedOpFactory<OpType::kD2H, D2HOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const D2HOpCargo& cargo) override {
    return std::make_shared<CUDAD2HOp>(cargo);
  }
};

}  // namespace mllm::cuda
